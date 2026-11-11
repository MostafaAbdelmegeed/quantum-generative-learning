"""Six-bit observable-selection comparison with expressibility witnesses."""

import argparse
import json
import platform
import shutil
import traceback
from datetime import UTC, datetime
from importlib.metadata import version
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from .circuits import ExactCircuit, make_circuit, to_qiskit
from .data import bit_table, probabilities_from_samples
from .experiment import (
    digest,
    file_digest,
    save_source_snapshot,
    source_receipt,
    write_json,
)
from .iqp import IQPMoments, characters, sample_masks, strong_pair_masks
from .metrics import evaluate, kernel_matrix
from .training import fit_classical, initial_parameters


def target_blocks(condition):
    definitions = {
        "adjacent_pairs": ([(0, 1), (2, 3), (4, 5)], 0.0),
        "shuffled_pairs": ([(0, 3), (1, 4), (2, 5)], 0.0),
        "noisy_pairs": ([(0, 3), (1, 4), (2, 5)], 0.1),
        "parity3": ([(0, 2, 4), (1, 3, 5)], 0.0),
        "parity4": ([(0, 2, 3, 5)], 0.0),
    }
    return definitions[condition]


def target_and_witness(condition):
    blocks, epsilon = target_blocks(condition)
    spec = make_circuit(6, "iqp", "dense")
    angles = np.zeros(spec.n_parameters)
    indices = {gate.qubits: gate.parameter for gate in spec.gates}
    bits = bit_table(6)
    target = np.ones(64)
    occupied = set()
    for block in blocks:
        occupied.update(block)
        odd = bits[:, block].sum(axis=1) % 2
        target *= np.where(odd, epsilon, 1 - epsilon) / 2 ** (len(block) - 1)
        angles[indices[(block[0],)]] = np.arcsin(np.sqrt(epsilon))
        for other in block[1:]:
            angles[indices[tuple(sorted((block[0], other)))]] = np.pi / 4
    for q in set(range(6)) - occupied:
        target /= 2
        angles[indices[(q,)]] = np.pi / 4
    return target, angles


def make_data(condition, seed, train_size, test_size):
    blocks, epsilon = target_blocks(condition)
    streams = [
        np.random.default_rng(s) for s in np.random.SeedSequence([seed, 681]).spawn(2)
    ]
    sets = []
    for rng, count in zip(streams, [train_size, test_size], strict=True):
        values = np.zeros((count, 6), dtype=np.int32)
        occupied = set()
        for block in blocks:
            occupied.update(block)
            values[:, block[1:]] = rng.integers(0, 2, (count, len(block) - 1))
            values[:, block[0]] = values[:, block[1:]].sum(axis=1) % 2
            values[:, block[0]] ^= (rng.random(count) < epsilon).astype(np.int32)
        for q in sorted(set(range(6)) - occupied):
            values[:, q] = rng.integers(0, 2, count)
        sets.append((values @ (1 << np.arange(6))).tolist())
    return {"train": sets[0], "heldout": sets[1]}


def candidate_masks(n, minimum_order=2, maximum_order=4):
    supports = [
        support
        for order in range(minimum_order, maximum_order + 1)
        for support in combinations(range(n), order)
    ]
    masks = np.zeros((len(supports), n), dtype=np.int32)
    for row, support in enumerate(supports):
        masks[row, list(support)] = 1
    return masks


def select_higher(train, count=3, maximum_order=4):
    train = np.asarray(train)
    n = train.shape[1]
    if not 1 <= count <= n or not 2 <= maximum_order <= n:
        raise ValueError("Invalid selector size or maximum order")
    masks = candidate_masks(n, maximum_order=maximum_order)
    moments = characters(train, masks).mean(axis=0)
    means = (1 - 2 * train).mean(axis=0)
    independent = np.prod(np.where(masks, means, 1), axis=1)
    scores = np.abs(moments - independent)
    ordering = sorted(
        range(len(masks)),
        key=lambda i: (
            -scores[i],
            int(masks[i].sum()),
            tuple(np.flatnonzero(masks[i])),
        ),
    )
    span, chosen = {0}, []
    for i in ordering:
        code = int(masks[i] @ (1 << np.arange(n)))
        if code in span:
            continue
        chosen.append(i)
        span |= {value ^ code for value in span}
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise ValueError("Candidate masks cannot supply the requested independent set")
    return masks[chosen], {
        "candidates": masks.tolist(),
        "scores": scores.tolist(),
        "chosen_indices": chosen,
        "character_evaluations": len(train) * len(masks),
    }


def selection_record(train, config):
    start = perf_counter()
    pair = strong_pair_masks(train, config["selected_count"])
    pair_seconds = perf_counter() - start
    start = perf_counter()
    higher, details = select_higher(
        train, config["selected_count"], config["maximum_order"]
    )
    return {
        "pair": {
            "masks": pair.tolist(),
            "seconds": pair_seconds,
            "candidate_count": 15,
            "character_evaluations": len(train) * 15,
        },
        "higher": {
            "masks": higher.tolist(),
            "seconds": perf_counter() - start,
            **details,
        },
    }


def objective_kernel(n, priority, fraction):
    kernel = kernel_matrix(n)
    if priority is None:
        return kernel
    basis = characters(bit_table(n), priority)
    return (1 - fraction) * kernel + fraction * (basis @ basis.T) / len(priority)


def fit_observables(spec, train, seed, config, priority=None, estimator="mc"):
    if estimator not in {"mc", "enumerated"}:
        raise ValueError("Unknown estimator")
    updates, count, half = (
        config[k] for k in ["updates", "masks_per_step", "samples_per_half"]
    )
    if min(updates, count, half) < 1 or config["learning_rate"] <= 0:
        raise ValueError("Invalid training budget")
    fraction = config["priority_fraction"] if priority is not None else 0
    priority_count = round(count * fraction)
    if priority is not None and not (
        0 < priority_count < count and priority_count == count * fraction
    ):
        raise ValueError("Mask budget must represent the mixture exactly")
    streams = [
        np.random.default_rng(s) for s in np.random.SeedSequence([seed, 882]).spawn(3)
    ]
    engine = IQPMoments(spec)
    initial = initial_parameters(spec, seed)
    theta, first, second = (
        initial.copy(),
        np.zeros_like(initial),
        np.zeros_like(initial),
    )
    history, norms = [], []
    table = bit_table(spec.n).astype(np.int32)
    start = perf_counter()
    for step in range(1, updates + 1):
        masks = sample_masks(spec.n, count, "multiscale", streams[0])
        if priority_count:
            masks[-priority_count:] = priority[
                streams[1].integers(0, len(priority), priority_count)
            ]
        if estimator == "mc":
            left = streams[2].integers(0, 2, (half, spec.n), dtype=np.int32)
            right = streams[2].integers(0, 2, (half, spec.n), dtype=np.int32)
            loss, gradient = engine.loss_and_grad(theta, masks, train, left, right)
        else:
            moments, jacobian = engine.moments_and_jacobian(theta, masks, table)
            residual = moments - characters(train, masks).mean(axis=0)
            loss = float(np.mean(residual**2))
            gradient = 2 * residual @ jacobian / count
        if not np.isfinite(loss) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError(f"Nonfinite fit at update {step}")
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient**2
        decay = (step - 1) / max(1, updates - 1)
        rate = config["learning_rate"] * (0.1 + 0.9 * (1 + np.cos(np.pi * decay)) / 2)
        theta -= (
            rate
            * (first / (1 - 0.9**step))
            / (np.sqrt(second / (1 - 0.999**step)) + 1e-8)
        )
        history.append(loss)
        norms.append(float(np.linalg.norm(gradient)))
    seconds = perf_counter() - start
    exact = ExactCircuit(spec)
    empirical = probabilities_from_samples(train @ (1 << np.arange(spec.n)), spec.n)
    kernel = objective_kernel(spec.n, priority, fraction)
    diagnostics = {}
    for label, values in [("initial", initial), ("final", theta)]:
        loss, grad = exact.loss_and_grad(values, empirical, kernel)
        diagnostics[label] = {
            "loss": loss,
            "gradient_norm": float(np.linalg.norm(grad)),
        }
    return {
        "parameters": theta.tolist(),
        "initial_parameters": initial.tolist(),
        "training_history": history,
        "gradient_norm_history": norms,
        "training_seconds": seconds,
        "diagnostics": diagnostics,
        "optimizer": "Adam",
        "converged": None,
        "iterations": updates,
        "message": "Fixed update budget completed; convergence not asserted",
        "resources": {
            "gradient_evaluations": updates,
            "objective_evaluations": updates,
            "classical_fourier_masks": updates * count,
            "classical_uniform_bitstrings": 2 * updates * half
            if estimator == "mc"
            else 0,
            "classical_phase_evaluations": updates
            * count
            * (2 * half if estimator == "mc" else 2**spec.n),
            "training_shots": 0,
            "training_statevector_evaluations": 0,
            "exact_objective_diagnostics": 2,
        },
    }


def fit_loglinear(train, masks, config, name):
    start = perf_counter()
    basis = characters(bit_table(train.shape[1]), masks)
    data_moments = characters(train, masks).mean(axis=0)
    ridge = config["classical_l2"]

    def objective(weights):
        logits = basis @ weights
        normalizer = logsumexp(logits)
        p = np.exp(logits - normalizer)
        return float(
            normalizer - data_moments @ weights + ridge * (weights @ weights) / 2
        ), (p @ basis - data_moments + ridge * weights)

    result = minimize(
        objective,
        np.zeros(len(masks)),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": config["classical_maxiter"], "ftol": 1e-12, "gtol": 1e-8},
    )
    logits = basis @ result.x
    return {
        "model": name,
        "probabilities": np.exp(logits - logsumexp(logits)).tolist(),
        "training_seconds": perf_counter() - start,
        "n_parameters": len(masks),
        "parameters": result.x.tolist(),
        "feature_masks": masks.tolist(),
        "fit_info": {
            "converged": bool(result.success),
            "message": str(result.message),
            "iterations": int(result.nit),
            "nfev": int(result.nfev),
        },
    }


def score_distribution(p, target, heldout, condition):
    p = np.asarray(p)
    blocks, _ = target_blocks(condition)
    bits = bit_table(6)
    block_spins = np.column_stack(
        [(-1.0) ** bits[:, block].sum(axis=1) for block in blocks]
    )
    all_even = np.all(block_spins == 1, axis=1)
    return {
        **evaluate(p, target, heldout, 6, condition),
        "block_moments": (p @ block_spins).tolist(),
        "block_moment_mae": float(np.mean(np.abs((p - target) @ block_spins))),
        "all_even_probability": float(p @ all_even),
        "all_even_probability_error": float(abs((p - target) @ all_even)),
        "marginal_mae": float(np.mean(np.abs((p - target) @ bits))),
    }


def qiskit_state(spec, theta):
    circuit, params = to_qiskit(spec)
    return Statevector(circuit.assign_parameters(dict(zip(params, theta, strict=True))))


def evaluate_fit(fit, target, heldout, condition, shots, seed, state=None):
    p = np.asarray(fit["probabilities"])
    fit["metrics"] = score_distribution(p, target, heldout, condition)
    if state is None:
        draws = np.random.default_rng(seed).choice(64, shots, p=p / p.sum())
        counts = np.bincount(draws, minlength=64)
    else:
        state.seed(seed)
        sampled = state.sample_counts(shots)
        counts = np.zeros(64, dtype=int)
        for key, value in sampled.items():
            counts[int(key, 2)] = value
    fit["sample_counts"] = counts.tolist()
    fit["sample_seed"] = seed
    fit["sample_metrics"] = score_distribution(
        counts / shots, target, heldout, condition
    )


def run_record(config, condition, seed):
    spec = make_circuit(6, "iqp", "dense")
    target, witness = target_and_witness(condition)
    witness_error = float(
        np.max(np.abs(qiskit_state(spec, witness).probabilities() - target))
    )
    if witness_error > 1e-12:
        raise ValueError("Target expressibility witness failed")
    data = make_data(condition, seed, config["train_size"], config["test_size"])
    train = bit_table(6)[data["train"]].astype(np.int32)
    heldout = probabilities_from_samples(np.array(data["heldout"]), 6)
    selection = selection_record(train, config)
    record = {
        "condition": condition,
        "seed": seed,
        "data": data,
        "data_sha256": digest(data),
        "target": target.tolist(),
        "circuit": spec.to_dict(),
        "selection": selection,
        "witness": {
            "parameters": witness.tolist(),
            "max_probability_error": witness_error,
        },
        "fits": [],
        "failures": [],
    }
    fits = fit_classical(np.array(data["train"]), 6, seed)
    for masks, name in [
        (
            np.vstack([np.eye(6, dtype=np.int32), selection["higher"]["masks"]]),
            "selected_maxent",
        ),
        (candidate_masks(6, 1, config["maximum_order"]), "order4_maxent"),
    ]:
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
        record["fits"].append(fit)
    for loss in config["losses"]:
        priority = None if loss == "multiscale" else np.array(selection[loss]["masks"])
        for estimator in config["estimators"]:
            name = f"iqp_{loss}_{estimator}"
            try:
                fit = fit_observables(spec, train, seed, config, priority, estimator)
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


def preflight(config):
    import tracemalloc

    spec = make_circuit(6, "iqp", "dense")
    rng = np.random.default_rng(991)
    train = rng.integers(0, 2, (config["train_size"], 6), dtype=np.int32)
    tracemalloc.start()
    start = perf_counter()
    for _ in range(20):
        IQPMoments(spec).loss_and_grad(
            rng.normal(size=21),
            sample_masks(6, config["masks_per_step"], "multiscale", rng),
            train,
            rng.integers(0, 2, (config["samples_per_half"], 6)),
            rng.integers(0, 2, (config["samples_per_half"], 6)),
        )
    seconds = perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "gradient_calls": 20,
        "seconds": seconds,
        "traced_peak_bytes": peak,
        "estimated_mc_fit_seconds": seconds / 20 * config["updates"],
        "workers": 1,
        "blas_threads": 1,
        "memory_scope": "Python/NumPy traced allocations, not process RSS",
        "target_fit_outcomes_used": False,
    }


def run(config_path, output):
    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text())
    if config["n_qubits"] != 6 or config["maximum_order"] != 4:
        raise ValueError("This protocol uses six bits and maximum order four")
    output.mkdir(parents=True, exist_ok=False)
    source = source_receipt()
    save_source_snapshot(output, source)
    root = Path(__file__).resolve().parents[2]
    frozen = {}
    for name, path in [
        ("config.json", config_path),
        ("uv.lock", root / "uv.lock"),
        ("protocol.md", root / "docs/observable-generalization-protocol.md"),
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
                try:
                    if source_receipt() != source:
                        raise RuntimeError("Source changed during the run")
                    record = run_record(config, condition, seed)
                    record.update(
                        config_sha256=digest(config), source_sha256=source["sha256"]
                    )
                    path = output / "runs" / f"{name}.json"
                    write_json(path, record)
                    index["completed"][name] = file_digest(path)
                    if record["failures"]:
                        index["failed"][name] = record["failures"]
                    print(
                        f"Completed {name}; failed fits: {len(record['failures'])}",
                        flush=True,
                    )
                except Exception:
                    index["failed"][name] = traceback.format_exc()
                write_json(output / "index.json", index)
    if index["failed"]:
        raise RuntimeError("Failed fits retained in index.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/observable-generalization-v1.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.config, args.output)
