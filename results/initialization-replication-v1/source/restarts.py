"""Initialization and training-selected restarts under a common update budget."""

import argparse
import json
import platform
import shutil
import traceback
import tracemalloc
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from .audit import require
from .circuits import ExactCircuit, make_circuit
from .data import bit_table, probabilities_from_samples
from .diagnostics import FullObjective, fit_full, select_order_penalized, without_times
from .experiment import (
    digest,
    file_digest,
    save_source_snapshot,
    source_receipt,
    write_json,
)
from .generalization import (
    candidate_masks,
    evaluate_fit,
    fit_loglinear,
    make_data,
    objective_kernel,
    qiskit_state,
    selection_record,
    target_and_witness,
)
from .generalization_audit import close
from .iqp import characters
from .training import fit_classical, initial_parameters


def classical_masks(selection, config):
    singles = np.eye(6, dtype=np.int32)
    return {
        "selected_maxent": np.vstack([singles, selection["higher"]["masks"]]),
        "order4_maxent": candidate_masks(6, 1, config["maximum_order"]),
        "order_penalized_maxent": np.vstack(
            [singles, selection["order_penalized"]["masks"]]
        ),
    }


def fresh_classical(data, selection, target, condition, seed, config):
    train = bit_table(6)[data["train"]].astype(np.int32)
    heldout = probabilities_from_samples(np.array(data["heldout"]), 6)
    fits = fit_classical(np.array(data["train"]), 6, seed)
    for name, masks in classical_masks(selection, config).items():
        fits.append(fit_loglinear(train, masks, config, name))
    fits.append(
        {
            "model": "oracle_target",
            "probabilities": target.tolist(),
            "n_parameters": 0,
            "training_seconds": 0,
            "fit_info": {"informed": True},
        }
    )
    for fit in fits:
        evaluate_fit(
            fit, target, heldout, condition, config["evaluation_shots"], seed + 3109
        )
    return fits


def audit_classical(fits, data, selection, target, condition, seed, config, replay):
    models = {
        "empirical",
        "independent",
        "pairwise",
        "autoregressive_mlp",
        "selected_maxent",
        "order4_maxent",
        "order_penalized_maxent",
        "oracle_target",
    }
    require(
        len(fits) == len(models) and {f["model"] for f in fits} == models,
        "Classical model set mismatch",
    )
    masks = classical_masks(selection, config)
    heldout = probabilities_from_samples(np.array(data["heldout"]), 6)
    empirical = probabilities_from_samples(np.array(data["train"]), 6)
    means = bit_table(6)[data["train"]].mean(axis=0)
    direct = {
        "empirical": empirical,
        "independent": np.prod(np.where(bit_table(6), means, 1 - means), axis=1),
        "oracle_target": target,
    }
    reruns = (
        {
            f["model"]: f
            for f in fresh_classical(data, selection, target, condition, seed, config)
        }
        if replay
        else {}
    )
    for fit in fits:
        name = fit["model"]
        p = np.asarray(fit["probabilities"])
        require(
            p.shape == (64,)
            and np.all(np.isfinite(p))
            and np.all(p >= 0)
            and abs(p.sum() - 1) < 1e-11,
            "Invalid classical distribution",
        )
        if name in direct:
            close(p, direct[name], "classical distribution")
        if name in masks:
            close(fit["feature_masks"], masks[name], "classical feature masks")
            weights = np.asarray(fit["parameters"])
            require(
                weights.shape == (len(masks[name]),)
                and np.all(np.isfinite(weights))
                and fit["n_parameters"] == len(weights),
                "Classical parameter mismatch",
            )
            logits = characters(bit_table(6), masks[name]) @ weights
            close(p, np.exp(logits - logsumexp(logits)), "classical probabilities")
        regenerated = {"probabilities": p.tolist()}
        evaluate_fit(
            regenerated,
            target,
            heldout,
            condition,
            config["evaluation_shots"],
            seed + 3109,
        )
        require(
            fit["sample_seed"] == seed + 3109
            and fit["sample_counts"] == regenerated["sample_counts"],
            "Classical sampling mismatch",
        )
        for group in ["metrics", "sample_metrics"]:
            for metric, value in regenerated[group].items():
                close(fit[group][metric], value, f"classical {metric}")
        if replay:
            close(p, reruns[name]["probabilities"], "Classical training replay")
            if "parameters" in fit:
                close(
                    fit["parameters"],
                    reruns[name]["parameters"],
                    "Classical parameter replay",
                )
    return len(fits)


def marginal_product_angles(spec, means):
    means = np.asarray(means, dtype=float)
    if (
        means.shape != (spec.n,)
        or not np.all(np.isfinite(means))
        or np.any((means < 0) | (means > 1))
    ):
        raise ValueError("Invalid training marginals")
    theta = np.zeros(spec.n_parameters)
    for gate in spec.gates:
        if len(gate.qubits) == 1:
            theta[gate.parameter] = np.arcsin(np.sqrt(means[gate.qubits[0]]))
    return theta


def initialization_vectors(spec, means, seed, config):
    count = max(len(updates) for updates in config["allocations"].values())
    nominal = marginal_product_angles(spec, means)
    result = {name: [] for name in config["initializations"]}
    for index in range(count):
        uniform = (
            initial_parameters(spec, seed)
            if index == 0
            else np.random.default_rng(
                np.random.SeedSequence([seed, 3107, index])
            ).uniform(-np.pi, np.pi, spec.n_parameters)
        )
        normal = np.random.default_rng(
            np.random.SeedSequence([seed, 7919, index])
        ).standard_normal(spec.n_parameters)
        values = {
            "uniform": uniform,
            "near_identity": config["near_identity_std"] * normal,
            "marginal_product": nominal + config["marginal_product_std"] * normal,
        }
        for family in result:
            result[family].append(values[family].tolist())
    return result


def select_training_index(final_losses):
    values = np.asarray(final_losses, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError(
            "Selection requires finite training losses for every candidate"
        )
    return int(np.argmin(values))


def policy_resources(candidates):
    training = sum(f["resources"]["classical_phase_evaluations"] for f in candidates)
    diagnostics = (
        sum(f["resources"]["exact_objective_diagnostics"] for f in candidates) * 4096
    )
    return {
        "gradient_updates": sum(f["iterations"] for f in candidates),
        "training_phase_evaluations": training,
        "diagnostic_phase_evaluations": diagnostics,
        "total_phase_evaluations": training + diagnostics,
        "extra_selection_objective_evaluations": 0,
        "training_shots": 0,
    }


def train_policy(spec, train, seed, config, priority, initializations, allocation):
    candidates, failures = [], []
    for index, updates in enumerate(config["allocations"][allocation]):
        try:
            fit = fit_full(
                spec,
                train,
                seed,
                {**config, "updates": updates},
                priority,
                initial=np.array(initializations[index]),
            )
            fit["candidate_index"] = index
            candidates.append(fit)
        except Exception:
            failures.append(
                {
                    "candidate_index": index,
                    "initial_parameters": initializations[index],
                    "traceback": traceback.format_exc(),
                }
            )
    selected = (
        None
        if failures
        else select_training_index(
            [f["diagnostics"]["final"]["loss"] for f in candidates]
        )
    )
    return {
        "allocation": allocation,
        "candidates": candidates,
        "failures": failures,
        "selected_candidate": selected,
        "resources": policy_resources(candidates),
    }


def run_record(config, condition, seed, reference=None):
    spec = make_circuit(6, "iqp", "dense")
    target, witness = target_and_witness(condition)
    close(qiskit_state(spec, witness).probabilities(), target, "expressibility")
    data = make_data(condition, seed, config["train_size"], config["test_size"])
    if reference is not None:
        require(data == reference["data"], "Reference data mismatch")
    train = bit_table(6)[data["train"]].astype(np.int32)
    heldout = probabilities_from_samples(np.array(data["heldout"]), 6)
    selection = selection_record(train, config)
    selection["order_penalized"] = select_order_penalized(train, config)
    start = perf_counter()
    means = train.mean(axis=0)
    initials = initialization_vectors(spec, means, seed, config)
    initial_seconds = perf_counter() - start
    record = {
        "condition": condition,
        "seed": seed,
        "data": data,
        "data_sha256": digest(data),
        "target": target.tolist(),
        "witness": witness.tolist(),
        "circuit": spec.to_dict(),
        "selection": selection,
        "initializations": initials,
        "training_marginals": means.tolist(),
        "initialization_timing": {"seconds": initial_seconds},
        "classical": reference["classical"]
        if reference is not None
        else fresh_classical(data, selection, target, condition, seed, config),
        "policies": [],
        "failures": [],
    }
    for loss in config["losses"]:
        priority = None if loss == "multiscale" else np.array(selection[loss]["masks"])
        for family in config["initializations"]:
            for allocation in config["allocations"]:
                policy = train_policy(
                    spec, train, seed, config, priority, initials[family], allocation
                )
                policy.update(loss=loss, initialization=family)
                # Selection is complete before target or held-out metrics enter candidate records.
                for fit in policy["candidates"]:
                    state = qiskit_state(spec, fit["parameters"])
                    fit["probabilities"] = state.probabilities().tolist()
                    evaluate_fit(
                        fit,
                        target,
                        heldout,
                        condition,
                        config["evaluation_shots"],
                        seed + 3109,
                        state,
                    )
                if policy["failures"]:
                    record["failures"].append(
                        {
                            "loss": loss,
                            "initialization": family,
                            "allocation": allocation,
                            "failures": policy["failures"],
                        }
                    )
                record["policies"].append(policy)
    return record


def validate_config(config):
    require(
        len(config["seeds"]) == len(set(config["seeds"])) and config["seeds"],
        "Duplicate or empty seeds",
    )
    if config.get("fresh_classical", False):
        require(
            not set(config["seeds"]) & set(config["excluded_seeds"]),
            "Replication reuses excluded seeds",
        )
        require(
            config["classical_maxiter"] > 0 and config["classical_l2"] > 0,
            "Invalid classical fitting settings",
        )
    require(
        config["n_qubits"] == 6 and config["maximum_order"] == 4,
        "Unsupported dimensions",
    )
    require(
        config["initializations"] == ["uniform", "near_identity", "marginal_product"],
        "Unexpected initialization families",
    )
    require(
        config["learning_rate"] > 0
        and config["near_identity_std"] > 0
        and config["marginal_product_std"] > 0,
        "Invalid initialization or learning rate",
    )
    require(config["total_updates"] > 0, "Invalid update budget")
    for updates in config["allocations"].values():
        require(
            updates and all(isinstance(v, int) and v > 0 for v in updates),
            "Invalid candidate budgets",
        )
        require(
            sum(updates) == config["total_updates"], "Allocation budgets must match"
        )


def preflight(config):
    rng = np.random.default_rng(991)
    spec = make_circuit(6, "iqp", "dense")
    train = rng.integers(0, 2, (config["train_size"], 6), dtype=np.int32)
    objective = FullObjective(spec, train)
    tracemalloc.start()
    start = perf_counter()
    for _ in range(20):
        objective(rng.normal(size=21))
    elapsed = perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    updates = (
        len(config["conditions"])
        * len(config["seeds"])
        * len(config["losses"])
        * len(config["initializations"])
        * len(config["allocations"])
        * config["total_updates"]
    )
    return {
        "gradient_calls": 20,
        "seconds": elapsed,
        "traced_peak_bytes": peak,
        "estimated_gradient_seconds_for_series": elapsed / 20 * updates,
        "memory_scope": "Python/NumPy traced allocations, excluding process RSS",
        "workers": 1,
        "blas_threads": 1,
        "target_fit_outcomes_used": False,
    }


def run(config_path, output, reference):
    config_path, output, reference = Path(config_path), Path(output), Path(reference)
    config = json.loads(config_path.read_text())
    validate_config(config)
    fresh = config.get("fresh_classical", False)
    output.mkdir(parents=True, exist_ok=False)
    source = source_receipt()
    save_source_snapshot(output, source)
    root = Path(__file__).resolve().parents[2]
    frozen = {}
    for name, path in [
        ("config.json", config_path),
        ("uv.lock", root / "uv.lock"),
        (
            "protocol.md",
            root
            / "docs"
            / (
                "initialization-replication-protocol.md"
                if fresh
                else "initialization-restarts-protocol.md"
            ),
        ),
    ]:
        shutil.copyfile(path, output / name)
        frozen[name] = file_digest(output / name)
    if not fresh:
        ref_index = json.loads((reference / "index.json").read_text())
        require(not ref_index["failed"], "Reference contains failures")
        (output / "reference").mkdir()
        shutil.copyfile(reference / "index.json", output / "reference/index.json")
    write_json(
        output / "manifest.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "config": config,
            "config_sha256": digest(config),
            "source": source,
            "frozen_files": frozen,
            **(
                {"reference_index_sha256": file_digest(reference / "index.json")}
                if not fresh
                else {}
            ),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": {
                name: version(name)
                for name in ["numpy", "scipy", "scikit-learn", "qiskit", "qiskit-aer"]
            },
        },
    )
    index = {
        "expected_jobs": len(config["conditions"]) * len(config["seeds"]),
        "completed": {},
        "failed": {},
    }
    with threadpool_limits(limits=1):
        write_json(output / "preflight.json", preflight(config))
        for condition in config["conditions"]:
            for seed in config["seeds"]:
                name = f"{condition}-s{seed}"
                ref_record = None
                if not fresh:
                    ref_path = reference / "runs" / f"{name}.json"
                    require(
                        file_digest(ref_path) == ref_index["completed"][name],
                        "Reference mismatch",
                    )
                    shutil.copyfile(ref_path, output / "reference" / f"{name}.json")
                    ref_record = json.loads(ref_path.read_text())
                try:
                    require(source_receipt() == source, "Source changed during run")
                    record = run_record(config, condition, seed, ref_record)
                    record.update(
                        config_sha256=digest(config),
                        source_sha256=source["sha256"],
                        **(
                            {"reference_sha256": file_digest(ref_path)}
                            if not fresh
                            else {}
                        ),
                    )
                    path = output / "runs" / f"{name}.json"
                    write_json(path, record)
                    index["completed"][name] = file_digest(path)
                    if record["failures"]:
                        index["failed"][name] = record["failures"]
                    count = sum(len(p["candidates"]) for p in record["policies"])
                    print(
                        f"Completed {name}: {count} candidates, {len(record['failures'])} policy failures",
                        flush=True,
                    )
                except Exception:
                    index["failed"][name] = traceback.format_exc()
                write_json(output / "index.json", index)
    require(not index["failed"], "Failures retained in index.json")


def audit(output, replay=False):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    config = manifest["config"]
    validate_config(config)
    fresh = config.get("fresh_classical", False)
    require(digest(config) == manifest["config_sha256"], "Config receipt mismatch")
    require(
        digest(manifest["source"]["files"]) == manifest["source"]["sha256"],
        "Source receipt mismatch",
    )
    for prefix, files in [
        ("source", manifest["source"]["files"]),
        ("", manifest["frozen_files"]),
    ]:
        for name, checksum in files.items():
            require(
                Path(name).name == name
                and file_digest(output / prefix / name) == checksum,
                "Frozen file mismatch",
            )
    require(
        json.loads((output / "config.json").read_text()) == config,
        "Frozen config mismatch",
    )
    if not fresh:
        require(
            file_digest(output / "reference/index.json")
            == manifest["reference_index_sha256"],
            "Reference index mismatch",
        )
        ref_index = json.loads((output / "reference/index.json").read_text())
    index = json.loads((output / "index.json").read_text())
    expected = {f"{c}-s{s}" for c in config["conditions"] for s in config["seeds"]}
    require(
        set(index["completed"]) == expected
        and index["expected_jobs"] == len(expected)
        and not index["failed"],
        "Incomplete run",
    )
    candidates_checked = policies_checked = replays = old_matches = 0
    classical_checked = classical_replays = 0
    with threadpool_limits(limits=1):
        for name in sorted(expected):
            path = output / "runs" / f"{name}.json"
            require(file_digest(path) == index["completed"][name], "Changed record")
            record = json.loads(path.read_text())
            condition, seed = record["condition"], record["seed"]
            require(
                name == f"{condition}-s{seed}"
                and record["config_sha256"] == manifest["config_sha256"]
                and record["source_sha256"] == manifest["source"]["sha256"],
                "Record identity mismatch",
            )
            if not fresh:
                ref_path = output / "reference" / f"{name}.json"
                require(
                    file_digest(ref_path)
                    == ref_index["completed"][name]
                    == record["reference_sha256"],
                    "Reference mismatch",
                )
                reference = json.loads(ref_path.read_text())
                require(
                    record["classical"] == reference["classical"],
                    "Classical reference mismatch",
                )
                require(record["data"] == reference["data"], "Reference data mismatch")
            data = make_data(condition, seed, config["train_size"], config["test_size"])
            require(
                data == record["data"] and digest(data) == record["data_sha256"],
                "Data mismatch",
            )
            train = bit_table(6)[data["train"]].astype(np.int32)
            means = train.mean(axis=0)
            heldout = probabilities_from_samples(np.array(data["heldout"]), 6)
            empirical = probabilities_from_samples(np.array(data["train"]), 6)
            spec = make_circuit(6, "iqp", "dense")
            target, witness = target_and_witness(condition)
            require(
                digest(record["circuit"]) == digest(spec.to_dict()), "Circuit mismatch"
            )
            close(record["target"], target, "target")
            close(record["witness"], witness, "witness angles")
            close(
                qiskit_state(spec, witness).probabilities(),
                target,
                "witness probabilities",
            )
            close(record["training_marginals"], means, "marginals")
            initials = initialization_vectors(spec, means, seed, config)
            require(
                initials.keys() == record["initializations"].keys(),
                "Initialization families mismatch",
            )
            for family, angles in initials.items():
                close(record["initializations"][family], angles, "initializations")
            selection = selection_record(train, config)
            selection["order_penalized"] = select_order_penalized(train, config)
            require(
                without_times(selection) == without_times(record["selection"]),
                "Selector mismatch",
            )
            if fresh:
                count = audit_classical(
                    record["classical"],
                    data,
                    selection,
                    target,
                    condition,
                    seed,
                    config,
                    replay,
                )
                classical_checked += count
                classical_replays += count if replay else 0
            expected_policies = {
                (loss, family, allocation)
                for loss in config["losses"]
                for family in config["initializations"]
                for allocation in config["allocations"]
            }
            require(
                {
                    (p["loss"], p["initialization"], p["allocation"])
                    for p in record["policies"]
                }
                == expected_policies
                and len(record["policies"]) == len(expected_policies)
                and not record["failures"],
                "Policy set mismatch",
            )
            for policy in record["policies"]:
                loss, family, allocation = (
                    policy[key] for key in ["loss", "initialization", "allocation"]
                )
                budgets = config["allocations"][allocation]
                require(
                    not policy["failures"]
                    and len(policy["candidates"]) == len(budgets),
                    "Missing candidates",
                )
                require(
                    [f["candidate_index"] for f in policy["candidates"]]
                    == list(range(len(budgets))),
                    "Candidate ordering mismatch",
                )
                priority = (
                    None if loss == "multiscale" else np.array(selection[loss]["masks"])
                )
                kernel = objective_kernel(6, priority, config["priority_fraction"])
                recomputed_scores = []
                for fit, budget in zip(policy["candidates"], budgets, strict=True):
                    candidate = fit["candidate_index"]
                    close(
                        fit["initial_parameters"],
                        initials[family][candidate],
                        "candidate initialization",
                    )
                    require(
                        fit["iterations"] == budget
                        and fit["optimizer"] == "Adam"
                        and fit["converged"] is None,
                        "Optimizer mismatch",
                    )
                    require(
                        fit["message"]
                        == "Fixed update budget completed; convergence not asserted",
                        "Stopping rule mismatch",
                    )
                    for field in ["training_history", "gradient_norm_history"]:
                        require(
                            len(fit[field]) == budget
                            and np.all(np.isfinite(fit[field])),
                            "Invalid history",
                        )
                    state = qiskit_state(spec, fit["parameters"])
                    close(
                        fit["probabilities"],
                        state.probabilities(),
                        "Qiskit probabilities",
                    )
                    close(
                        fit["probabilities"],
                        ExactCircuit(spec).probabilities(fit["parameters"]),
                        "NumPy probabilities",
                    )
                    for label, angles in [
                        ("initial", fit["initial_parameters"]),
                        ("final", fit["parameters"]),
                    ]:
                        objective, grad = ExactCircuit(spec).loss_and_grad(
                            angles, empirical, kernel
                        )
                        close(
                            fit["diagnostics"][label]["loss"],
                            objective,
                            "training objective",
                        )
                        close(
                            fit["diagnostics"][label]["gradient_norm"],
                            np.linalg.norm(grad),
                            "training gradient",
                        )
                        if label == "final":
                            recomputed_scores.append(objective)
                    expected_resources = {
                        "gradient_evaluations": budget,
                        "objective_evaluations": budget,
                        "classical_fourier_masks": 64 * budget,
                        "enumerated_masks": 64 * budget,
                        "sampled_masks": 0,
                        "classical_uniform_bitstrings": 0,
                        "classical_phase_evaluations": 4096 * budget,
                        "training_shots": 0,
                        "training_statevector_evaluations": 0,
                        "exact_objective_diagnostics": 2,
                    }
                    require(
                        fit["resources"] == expected_resources,
                        "Candidate budget mismatch",
                    )
                    if (
                        not fresh
                        and family == "uniform"
                        and candidate == 0
                        and budget in [400, 1600]
                    ):
                        old = next(
                            f
                            for f in reference["fits"]
                            if f["loss"] == loss
                            and f["estimator"] == "full"
                            and f["iterations"] == budget
                        )
                        for field in [
                            "parameters",
                            "initial_parameters",
                            "training_history",
                            "gradient_norm_history",
                            "probabilities",
                            "sample_counts",
                        ]:
                            require(
                                fit[field] == old[field],
                                f"Published fit mismatch: {field}",
                            )
                        old_matches += 1
                    if replay:
                        refit = fit_full(
                            spec,
                            train,
                            seed,
                            {**config, "updates": budget},
                            priority,
                            initial=np.array(initials[family][candidate]),
                        )
                        for field in [
                            "parameters",
                            "training_history",
                            "gradient_norm_history",
                        ]:
                            close(fit[field], refit[field], f"Training replay {field}")
                        replays += 1
                    regenerated = {"probabilities": fit["probabilities"]}
                    require(fit["sample_seed"] == seed + 3109, "Sampling seed mismatch")
                    evaluate_fit(
                        regenerated,
                        target,
                        heldout,
                        condition,
                        config["evaluation_shots"],
                        seed + 3109,
                        state,
                    )
                    require(
                        regenerated["sample_counts"] == fit["sample_counts"],
                        "Sampling replay mismatch",
                    )
                    for group in ["metrics", "sample_metrics"]:
                        for metric, value in regenerated[group].items():
                            close(fit[group][metric], value, metric)
                    candidates_checked += 1
                recorded_losses = [
                    f["diagnostics"]["final"]["loss"] for f in policy["candidates"]
                ]
                chosen = select_training_index(recorded_losses)
                require(
                    policy["selected_candidate"] == chosen,
                    "Training-only selection mismatch",
                )
                # Independent adjoint arithmetic may perturb exact ties at machine precision.
                require(
                    recomputed_scores[chosen] <= min(recomputed_scores) + 1e-11,
                    "Selection fails independent objective check",
                )
                require(
                    policy["resources"] == policy_resources(policy["candidates"]),
                    "Policy resource mismatch",
                )
                require(
                    policy["resources"]["gradient_updates"] == config["total_updates"],
                    "Policy update budget mismatch",
                )
                policies_checked += 1
    report = {
        "passed": True,
        "jobs": len(expected),
        "policies_checked": policies_checked,
        "candidates_checked": candidates_checked,
        "training_replays": replays,
        "published_fits_reproduced": old_matches,
        "fresh_classical_checked": classical_checked,
        "fresh_classical_replays": classical_replays,
        "index_sha256": file_digest(output / "index.json"),
    }
    write_json(output / "audit.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "audit"])
    parser.add_argument("--config", default="config/initialization-restarts-v1.json")
    parser.add_argument("--reference", default="results/observable-diagnostics-v1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.action == "run":
        run(args.config, args.output, args.reference)
    else:
        print(json.dumps(audit(args.output, args.replay), indent=2))
