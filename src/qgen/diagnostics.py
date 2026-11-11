"""Factorial controls for observable selection, update budget and estimator noise."""

import argparse
import json
import platform
import shutil
import traceback
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from .audit import require
from .circuits import ExactCircuit, make_circuit
from .data import bit_table, probabilities_from_samples
from .experiment import (
    digest,
    file_digest,
    save_source_snapshot,
    source_receipt,
    write_json,
)
from .generalization import (
    evaluate_fit,
    fit_loglinear,
    fit_observables,
    make_data,
    objective_kernel,
    preflight,
    qiskit_state,
    select_higher,
    selection_record,
    target_and_witness,
)
from .generalization_audit import close
from .iqp import IQPMoments, characters, mask_weights
from .training import initial_parameters


def select_order_penalized(train, config):
    start = perf_counter()
    _, details = select_higher(train, config["selected_count"], config["maximum_order"])
    masks = np.array(details["candidates"])
    scores = np.array(details["scores"]) / (masks.sum(axis=1) - 1)
    order = sorted(
        range(len(masks)),
        key=lambda i: (
            -scores[i],
            int(masks[i].sum()),
            tuple(np.flatnonzero(masks[i])),
        ),
    )
    selected, span = [], {0}
    for i in order:
        code = int(masks[i] @ (1 << np.arange(train.shape[1])))
        if code not in span:
            selected.append(i)
            span |= {value ^ code for value in span}
            if len(selected) == config["selected_count"]:
                break
    return {
        **details,
        "chosen_indices": selected,
        "masks": masks[selected].tolist(),
        "penalized_scores": scores.tolist(),
        "seconds": perf_counter() - start,
    }


class FullObjective:
    def __init__(self, spec, train, priority=None, fraction=0.5):
        self.engine = IQPMoments(spec)
        self.table = bit_table(spec.n).astype(np.int32)
        self.weights = mask_weights(self.table, "multiscale")
        if priority is not None:
            self.weights *= 1 - fraction
            codes = np.asarray(priority) @ (1 << np.arange(spec.n))
            np.add.at(self.weights, codes, fraction / len(codes))
        self.data_moments = characters(train, self.table).mean(axis=0)

    def __call__(self, theta):
        moments, jacobian = self.engine.moments_and_jacobian(
            theta, self.table, self.table
        )
        residual = moments - self.data_moments
        return float(self.weights @ residual**2), 2 * (
            self.weights * residual
        ) @ jacobian


def fit_full(spec, train, seed, config, priority=None):
    initial = initial_parameters(spec, seed)
    theta = initial.copy()
    first, second = np.zeros_like(theta), np.zeros_like(theta)
    objective = FullObjective(spec, train, priority, config["priority_fraction"])
    initial_loss, initial_grad = objective(initial)
    history, norms = [], []
    updates = config["updates"]
    start = perf_counter()
    for step in range(1, updates + 1):
        loss, grad = objective(theta)
        if not np.isfinite(loss) or not np.all(np.isfinite(grad)):
            raise FloatingPointError(f"Nonfinite full gradient at {step}")
        first = 0.9 * first + 0.1 * grad
        second = 0.999 * second + 0.001 * grad**2
        fraction = (step - 1) / max(1, updates - 1)
        rate = config["learning_rate"] * (
            0.1 + 0.9 * (1 + np.cos(np.pi * fraction)) / 2
        )
        theta -= (
            rate
            * (first / (1 - 0.9**step))
            / (np.sqrt(second / (1 - 0.999**step)) + 1e-8)
        )
        history.append(loss)
        norms.append(float(np.linalg.norm(grad)))
    seconds = perf_counter() - start
    final_loss, final_grad = objective(theta)
    return {
        "parameters": theta.tolist(),
        "initial_parameters": initial.tolist(),
        "training_history": history,
        "gradient_norm_history": norms,
        "training_seconds": seconds,
        "diagnostics": {
            "initial": {
                "loss": initial_loss,
                "gradient_norm": float(np.linalg.norm(initial_grad)),
            },
            "final": {
                "loss": final_loss,
                "gradient_norm": float(np.linalg.norm(final_grad)),
            },
        },
        "optimizer": "Adam",
        "iterations": updates,
        "converged": None,
        "message": "Fixed update budget completed; convergence not asserted",
        "resources": {
            "gradient_evaluations": updates,
            "objective_evaluations": updates,
            "classical_fourier_masks": 64 * updates,
            "enumerated_masks": 64 * updates,
            "sampled_masks": 0,
            "classical_uniform_bitstrings": 0,
            "classical_phase_evaluations": 4096 * updates,
            "training_shots": 0,
            "training_statevector_evaluations": 0,
            "exact_objective_diagnostics": 2,
        },
    }


def fit_variant(spec, train, seed, config, priority, estimator):
    if estimator == "full":
        return fit_full(spec, train, seed, config, priority)
    return fit_observables(spec, train, seed, config, priority, estimator)


def without_times(value):
    if isinstance(value, dict):
        return {
            k: without_times(v)
            for k, v in value.items()
            if k not in {"seconds", "training_seconds"}
        }
    if isinstance(value, list):
        return [without_times(v) for v in value]
    return value


def record_job(config, condition, seed, reference):
    target, witness = target_and_witness(condition)
    spec = make_circuit(6, "iqp", "dense")
    close(qiskit_state(spec, witness).probabilities(), target, "expressibility")
    data = make_data(condition, seed, config["train_size"], config["test_size"])
    require(data == reference["data"], "Reference data mismatch")
    train = bit_table(6)[data["train"]].astype(np.int32)
    heldout = probabilities_from_samples(np.array(data["heldout"]), 6)
    selectors = selection_record(train, config)
    selectors["order_penalized"] = select_order_penalized(train, config)
    classical = [f for f in reference["fits"] if not f["model"].startswith("iqp_")]
    masks = np.vstack(
        [np.eye(6, dtype=np.int32), selectors["order_penalized"]["masks"]]
    )
    extra = fit_loglinear(train, masks, config, "order_penalized_maxent")
    evaluate_fit(
        extra, target, heldout, condition, config["evaluation_shots"], seed + 3109
    )
    classical.append(extra)
    record = {
        "condition": condition,
        "seed": seed,
        "data": data,
        "data_sha256": digest(data),
        "target": target.tolist(),
        "witness": witness.tolist(),
        "circuit": spec.to_dict(),
        "selection": selectors,
        "classical": classical,
        "fits": [],
        "failures": [],
    }
    for budget in config["update_budgets"]:
        for loss in config["losses"]:
            priority = (
                None if loss == "multiscale" else np.array(selectors[loss]["masks"])
            )
            for estimator in config["estimators"]:
                name = f"iqp_{loss}_{estimator}_u{budget}"
                try:
                    fit = fit_variant(
                        spec,
                        train,
                        seed,
                        {**config, "updates": budget},
                        priority,
                        estimator,
                    )
                    state = qiskit_state(spec, fit["parameters"])
                    fit.update(
                        model=name,
                        loss=loss,
                        estimator=estimator,
                        n_parameters=21,
                        probabilities=state.probabilities().tolist(),
                    )
                    evaluate_fit(
                        fit,
                        target,
                        heldout,
                        condition,
                        config["evaluation_shots"],
                        seed + 3109,
                        state,
                    )
                    record["fits"].append(fit)
                except Exception:
                    record["failures"].append(
                        {"model": name, "traceback": traceback.format_exc()}
                    )
    return record


def run(config_path, output, reference):
    config_path, output, reference = Path(config_path), Path(output), Path(reference)
    config = json.loads(config_path.read_text())
    require(
        config["n_qubits"] == 6 and config["masks_per_step"] == 64,
        "Unsupported experiment dimensions",
    )
    output.mkdir(parents=True, exist_ok=False)
    source = source_receipt()
    save_source_snapshot(output, source)
    reference_index = json.loads((reference / "index.json").read_text())
    require(not reference_index["failed"], "Reference has failures")
    root = Path(__file__).resolve().parents[2]
    frozen = {}
    for name, path in [
        ("config.json", config_path),
        ("uv.lock", root / "uv.lock"),
        ("protocol.md", root / "docs/observable-diagnostics-protocol.md"),
    ]:
        shutil.copyfile(path, output / name)
        frozen[name] = file_digest(output / name)
    write_json(
        output / "manifest.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "config": config,
            "config_sha256": digest(config),
            "source": source,
            "frozen_files": frozen,
            "reference_index_sha256": file_digest(reference / "index.json"),
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
    (output / "reference").mkdir()
    shutil.copyfile(reference / "index.json", output / "reference/index.json")
    with threadpool_limits(limits=1):
        write_json(
            output / "preflight.json",
            preflight({**config, "updates": max(config["update_budgets"])}),
        )
        for condition in config["conditions"]:
            for seed in config["seeds"]:
                name = f"{condition}-s{seed}"
                source_path = reference / "runs" / f"{name}.json"
                require(
                    file_digest(source_path) == reference_index["completed"][name],
                    "Changed reference",
                )
                shutil.copyfile(source_path, output / "reference" / f"{name}.json")
                try:
                    require(source_receipt() == source, "Source changed during run")
                    result = record_job(
                        config, condition, seed, json.loads(source_path.read_text())
                    )
                    result.update(
                        config_sha256=digest(config),
                        source_sha256=source["sha256"],
                        reference_sha256=file_digest(source_path),
                    )
                    path = output / "runs" / f"{name}.json"
                    write_json(path, result)
                    index["completed"][name] = file_digest(path)
                    if result["failures"]:
                        index["failed"][name] = result["failures"]
                    print(
                        f"Completed {name}: {len(result['fits'])} circuit fits",
                        flush=True,
                    )
                except Exception:
                    index["failed"][name] = traceback.format_exc()
                write_json(output / "index.json", index)
    require(not index["failed"], "Failures saved in index.json")


def audit(output, replay=False):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    config = manifest["config"]
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
    require(
        file_digest(output / "reference/index.json")
        == manifest["reference_index_sha256"],
        "Reference index mismatch",
    )
    reference_index = json.loads((output / "reference/index.json").read_text())
    index = json.loads((output / "index.json").read_text())
    expected = {f"{c}-s{s}" for c in config["conditions"] for s in config["seeds"]}
    require(
        set(index["completed"]) == expected and not index["failed"], "Incomplete run"
    )
    require(index["expected_jobs"] == len(expected), "Wrong job count")
    checked = repeated = old_matches = 0
    with threadpool_limits(limits=1):
        for name in sorted(expected):
            path = output / "runs" / f"{name}.json"
            require(file_digest(path) == index["completed"][name], "Changed record")
            record = json.loads(path.read_text())
            condition, seed = record["condition"], record["seed"]
            require(name == f"{condition}-s{seed}", "Job identity mismatch")
            require(
                record["config_sha256"] == manifest["config_sha256"]
                and record["source_sha256"] == manifest["source"]["sha256"],
                "Record receipt mismatch",
            )
            ref_path = output / "reference" / f"{name}.json"
            require(
                file_digest(ref_path)
                == reference_index["completed"][name]
                == record["reference_sha256"],
                "Reference record mismatch",
            )
            reference = json.loads(ref_path.read_text())
            data = make_data(condition, seed, config["train_size"], config["test_size"])
            require(
                data == record["data"] and digest(data) == record["data_sha256"],
                "Data mismatch",
            )
            train = bit_table(6)[data["train"]].astype(np.int32)
            heldout = probabilities_from_samples(np.array(data["heldout"]), 6)
            target, witness = target_and_witness(condition)
            spec = make_circuit(6, "iqp", "dense")
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
            selection = selection_record(train, config)
            selection["order_penalized"] = select_order_penalized(train, config)
            require(
                without_times(selection) == without_times(record["selection"]),
                "Selector mismatch",
            )
            expected_fits = {
                f"iqp_{loss}_{estimator}_u{budget}"
                for budget in config["update_budgets"]
                for loss in config["losses"]
                for estimator in config["estimators"]
            }
            require(
                {f["model"] for f in record["fits"]} == expected_fits
                and len(record["fits"]) == len(expected_fits)
                and not record["failures"],
                "Fit set mismatch",
            )
            old_classical = [
                f for f in reference["fits"] if not f["model"].startswith("iqp_")
            ]
            require(
                record["classical"][:-1] == old_classical
                and len(record["classical"]) == 8,
                "Classical reference mismatch",
            )
            extra = record["classical"][-1]
            masks = np.vstack(
                [np.eye(6, dtype=np.int32), selection["order_penalized"]["masks"]]
            )
            refit = fit_loglinear(train, masks, config, "order_penalized_maxent")
            for key in ["parameters", "probabilities", "feature_masks"]:
                close(extra[key], refit[key], f"classical {key}")
            for fit in record["fits"] + record["classical"]:
                state = None
                if "estimator" in fit:
                    budget, estimator, loss = (
                        fit["iterations"],
                        fit["estimator"],
                        fit["loss"],
                    )
                    require(
                        fit["model"] == f"iqp_{loss}_{estimator}_u{budget}",
                        "Fit identity mismatch",
                    )
                    require(
                        fit["optimizer"] == "Adam" and fit["converged"] is None,
                        "Optimizer mismatch",
                    )
                    close(
                        fit["initial_parameters"],
                        initial_parameters(spec, seed),
                        "initial angles",
                    )
                    close(
                        fit["probabilities"],
                        ExactCircuit(spec).probabilities(fit["parameters"]),
                        "probabilities",
                    )
                    state = qiskit_state(spec, fit["parameters"])
                    close(
                        fit["probabilities"],
                        state.probabilities(),
                        "Qiskit probabilities",
                    )
                    for field in ["training_history", "gradient_norm_history"]:
                        require(
                            len(fit[field]) == budget
                            and np.all(np.isfinite(fit[field])),
                            "Invalid history",
                        )
                    priority = (
                        None
                        if loss == "multiscale"
                        else np.array(selection[loss]["masks"])
                    )
                    empirical = probabilities_from_samples(np.array(data["train"]), 6)
                    kernel = objective_kernel(6, priority, config["priority_fraction"])
                    for label, angles in [
                        ("initial", fit["initial_parameters"]),
                        ("final", fit["parameters"]),
                    ]:
                        objective, grad = ExactCircuit(spec).loss_and_grad(
                            angles, empirical, kernel
                        )
                        close(fit["diagnostics"][label]["loss"], objective, "objective")
                        close(
                            fit["diagnostics"][label]["gradient_norm"],
                            np.linalg.norm(grad),
                            "gradient",
                        )
                    resources = fit["resources"]
                    require(
                        resources["classical_phase_evaluations"]
                        == budget * (8192 if estimator == "mc" else 4096),
                        "Phase budget mismatch",
                    )
                    require(
                        resources["classical_uniform_bitstrings"]
                        == (128 * budget if estimator == "mc" else 0),
                        "String budget mismatch",
                    )
                    require(
                        resources["classical_fourier_masks"] == 64 * budget
                        and resources["gradient_evaluations"] == budget
                        and resources["objective_evaluations"] == budget,
                        "Update budget mismatch",
                    )
                    require(
                        resources["training_shots"] == 0
                        and resources["training_statevector_evaluations"] == 0,
                        "Unexpected training resource",
                    )
                    if estimator == "full":
                        require(
                            resources["sampled_masks"] == 0
                            and resources["enumerated_masks"] == 64 * budget,
                            "Full-mask budget mismatch",
                        )
                    if (
                        budget == 400
                        and estimator != "full"
                        and loss != "order_penalized"
                    ):
                        old = next(
                            f
                            for f in reference["fits"]
                            if f["model"] == f"iqp_{loss}_{estimator}"
                        )
                        require(
                            without_times({**fit, "model": old["model"]})
                            == without_times(old),
                            "Published short-budget replay mismatch",
                        )
                        old_matches += 1
                    if replay:
                        refit = fit_variant(
                            spec,
                            train,
                            seed,
                            {**config, "updates": budget},
                            priority,
                            estimator,
                        )
                        for field in [
                            "parameters",
                            "training_history",
                            "gradient_norm_history",
                        ]:
                            close(fit[field], refit[field], f"Adam replay {field}")
                        repeated += 1
                regenerated = {"probabilities": fit["probabilities"]}
                require(fit["sample_seed"] == seed + 3109, "Sample seed mismatch")
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
                    for metric in regenerated[group]:
                        close(fit[group][metric], regenerated[group][metric], metric)
                checked += 1
    report = {
        "passed": True,
        "jobs": len(expected),
        "fits_checked": checked,
        "adam_replays": repeated,
        "published_short_fits_reproduced": old_matches,
        "index_sha256": file_digest(output / "index.json"),
    }
    write_json(output / "audit.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "audit"])
    parser.add_argument("--config", default="config/observable-diagnostics-v1.json")
    parser.add_argument("--reference", default="results/observable-generalization-v1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.action == "run":
        run(args.config, args.output, args.reference)
    else:
        print(json.dumps(audit(args.output, args.replay), indent=2))
