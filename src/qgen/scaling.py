"""A larger, deliberately classically tractable IQP training demonstration."""

import json
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
from qiskit import qasm3, transpile
from qiskit_aer import AerSimulator
from threadpoolctl import threadpool_limits

from .circuits import CircuitSpec, make_circuit, to_qiskit
from .experiment import (
    digest,
    file_digest,
    save_source_snapshot,
    source_receipt,
    write_json,
)
from .iqp import bandwidths, binary_matrix, fit_iqp_mc, strong_pair_masks
from .training import initial_parameters


def paired_bits(n, count, rng):
    if n < 2 or n % 2 or count < 2:
        raise ValueError("Paired data requires an even width and at least two samples")
    return np.repeat(rng.integers(0, 2, (count, n // 2), dtype=np.int32), 2, axis=1)


def encode_bits(bits):
    # Stored strings follow Qiskit display order: q[n-1] ... q[0].
    return ["".join(map(str, row[::-1])) for row in bits]


def decode_bits(strings, n):
    if not strings or any(len(s) != n or set(s) - {"0", "1"} for s in strings):
        raise ValueError("Invalid saved bitstrings")
    return np.array([[int(c) for c in s[::-1]] for s in strings], dtype=np.int32)


def pair_diagnostics(bits):
    bits = np.asarray(bits)
    paired = bits[:, 0::2] == bits[:, 1::2]
    return {
        "valid_fraction": float(paired.all(axis=1).mean()),
        "pair_agreement": float(paired.mean()),
        "mean_absolute_marginal_bias": float(np.abs(bits.mean(axis=0) - 0.5).mean()),
    }


def sample_mmd(left, right, loss="multiscale", block_size=128):
    """Unbiased MMD from two independent sample sets, with bounded work arrays."""
    left = binary_matrix(left, np.shape(left)[1], "left")
    right = binary_matrix(right, left.shape[1], "right")
    if min(len(left), len(right)) < 2:
        raise ValueError("MMD needs at least two samples per set")
    bands = bandwidths(left.shape[1], loss)

    def kernel_sum(a, b):
        total = 0.0
        bsum = b.sum(axis=1)
        for start in range(0, len(a), block_size):
            batch = a[start : start + block_size]
            distance = batch.sum(axis=1)[:, None] + bsum - 2 * (batch @ b.T)
            for band in bands:
                total += np.exp(-distance / (2 * band)).sum() / len(bands)
        return total

    aa = (kernel_sum(left, left) - len(left)) / (len(left) * (len(left) - 1))
    bb = (kernel_sum(right, right) - len(right)) / (len(right) * (len(right) - 1))
    return float(aa + bb - 2 * kernel_sum(left, right) / (len(left) * len(right)))


def sample_qiskit_mps(spec, theta, shots, seed):
    """Sample a noiseless Qiskit circuit with a matrix product state simulator."""
    circuit, parameters = to_qiskit(spec)
    bound = circuit.assign_parameters(dict(zip(parameters, theta, strict=True)))
    bound.measure_all()
    simulator = AerSimulator(
        method="matrix_product_state",
        max_parallel_threads=1,
        matrix_product_state_truncation_threshold=0.0,
        mps_sample_measure_algorithm="mps_apply_measure",
    )
    compiled = transpile(bound, simulator, optimization_level=1, seed_transpiler=17)
    start = perf_counter()
    outcome = simulator.run(
        compiled, shots=shots, seed_simulator=seed, memory=True
    ).result()
    if not outcome.success:
        raise RuntimeError("Qiskit MPS sampling failed")
    strings = outcome.get_memory()
    bits = decode_bits(strings, spec.n)
    return bits, {
        "method": "matrix_product_state",
        "noise": "none",
        "shots": shots,
        "seed": seed,
        "seconds": perf_counter() - start,
        "truncation_threshold": 0,
        "depth": compiled.depth(),
        "gate_counts": dict(compiled.count_ops()),
        "qasm": qasm3.dumps(compiled),
    }


def _scaling_job(config, n, seed, source):
    if source_receipt()["sha256"] != source["sha256"]:
        raise RuntimeError("Source changed during the scaling run")
    with threadpool_limits(limits=1):
        streams = [
            np.random.default_rng(s)
            for s in np.random.SeedSequence([seed, n, 103]).spawn(5)
        ]
        train = paired_bits(n, config["train_size"], streams[0])
        heldout = paired_bits(n, config["test_size"], streams[1])
        spec = make_circuit(n, "iqp", "line", seed=seed)
        selection_start = perf_counter()
        priority_fraction = config.get("priority_fraction", 0)
        priority = strong_pair_masks(train, n // 2) if priority_fraction else None
        selection_seconds = perf_counter() - selection_start
        fit = fit_iqp_mc(
            spec,
            train,
            "multiscale",
            seed,
            config["updates"],
            config["masks_per_step"],
            config["samples_per_half"],
            config["learning_rate"],
            priority_masks=priority,
            priority_fraction=priority_fraction,
        )
        model, sampling = sample_qiskit_mps(
            spec, fit["parameters"], config["evaluation_shots"], seed + 3109
        )
        initial, initial_sampling = sample_qiskit_mps(
            spec, fit["initial_parameters"], config["evaluation_shots"], seed + 3109
        )
        # An independent-bit baseline learns only single-bit marginals.
        independent = (streams[2].random(model.shape) < train.mean(axis=0)).astype(
            np.int32
        )
        # The empirical baseline resamples the training set.
        empirical = train[streams[3].integers(0, len(train), len(model))]
        # A simple classical model given the same known adjacent-pair structure.
        pair_means = train[:, 0::2].mean(axis=0)
        pair_model = np.repeat(
            (streams[4].random((len(model), n // 2)) < pair_means).astype(np.int32),
            2,
            axis=1,
        )
        sets = {
            "iqp": model,
            "initial_iqp": initial,
            "independent": independent,
            "empirical": empirical,
            "paired_classical": pair_model,
        }
        metrics = {
            name: {**pair_diagnostics(bits), "heldout_mmd_u": sample_mmd(bits, heldout)}
            for name, bits in sets.items()
        }
        data = {
            "train": encode_bits(train),
            "heldout": encode_bits(heldout),
            "samples": {name: encode_bits(bits) for name, bits in sets.items()},
        }
        return {
            "n": n,
            "seed": seed,
            "config_sha256": digest(config),
            "source_sha256": source["sha256"],
            "circuit": spec.to_dict(),
            "fit": fit,
            "data": data,
            "data_sha256": digest(data),
            "metrics": metrics,
            "sampling": sampling,
            "initial_sampling": initial_sampling,
            "feature_selection": {
                "masks": priority.tolist() if priority is not None else [],
                "fraction": priority_fraction,
                "seconds": selection_seconds,
            },
        }


def run_scaling(config_path, output, workers=1):
    config, output = json.loads(Path(config_path).read_text()), Path(output)
    source = source_receipt()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest["config_sha256"] != digest(config)
            or manifest["source"]["sha256"] != source["sha256"]
        ):
            raise ValueError("Use a new output directory for changed config or source")
    else:
        root = Path(__file__).resolve().parents[2]
        write_json(
            manifest_path,
            {
                "created_utc": datetime.now(UTC).isoformat(),
                "config": config,
                "config_sha256": digest(config),
                "source": source,
                "python": platform.python_version(),
                "dependencies": {
                    p: version(p) for p in ["numpy", "scipy", "qiskit", "qiskit-aer"]
                },
                "lockfile_sha256": file_digest(root / "uv.lock")
                if (root / "uv.lock").exists()
                else None,
            },
        )
    save_source_snapshot(output, source)
    jobs = [
        (n, seed, f"n{n}-s{seed}") for n in config["widths"] for seed in config["seeds"]
    ]
    if len({j[2] for j in jobs}) != len(jobs) or workers < 1:
        raise ValueError("Invalid jobs or worker count")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {}
        for n, seed, name in jobs:
            path = output / "runs" / f"{name}.json"
            if path.exists():
                record = json.loads(path.read_text())
                if (
                    record["config_sha256"] != digest(config)
                    or record["source_sha256"] != source["sha256"]
                    or (record["n"], record["seed"]) != (n, seed)
                ):
                    raise ValueError(f"Mismatched result: {name}")
            else:
                pending[pool.submit(_scaling_job, config, n, seed, source)] = name
        for future in as_completed(pending):
            name = pending[future]
            result = future.result()
            write_json(output / "runs" / f"{name}.json", result)
            print(
                f"Completed {name}: valid fraction = {result['metrics']['iqp']['valid_fraction']:.3f}",
                flush=True,
            )
    complete = {
        name: file_digest(output / "runs" / f"{name}.json") for _, _, name in jobs
    }
    write_json(
        output / "index.json",
        {"expected_jobs": len(jobs), "completed": complete, "failed": 0},
    )


def audit_scaling(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    config = manifest["config"]
    index = json.loads((output / "index.json").read_text())
    if manifest["config_sha256"] != digest(config) or manifest["source"][
        "sha256"
    ] != digest(manifest["source"]["files"]):
        raise ValueError("Manifest receipt mismatch")
    for name, checksum in manifest["source"]["files"].items():
        if Path(name).name != name or file_digest(output / "source" / name) != checksum:
            raise ValueError("Source snapshot mismatch")
    expected = {f"n{n}-s{seed}" for n in config["widths"] for seed in config["seeds"]}
    if (
        set(index["completed"]) != expected
        or index["expected_jobs"] != len(expected)
        or index["failed"]
    ):
        raise ValueError("Incomplete scaling run")
    checks = 0
    for name in sorted(expected):
        path = output / "runs" / f"{name}.json"
        if file_digest(path) != index["completed"][name]:
            raise ValueError(f"Changed result: {path}")
        record = json.loads(path.read_text())
        n, seed = record["n"], record["seed"]
        if (
            name != f"n{n}-s{seed}"
            or record["config_sha256"] != manifest["config_sha256"]
            or record["source_sha256"] != manifest["source"]["sha256"]
            or digest(record["data"]) != record["data_sha256"]
        ):
            raise ValueError("Result receipt mismatch")
        data = record["data"]
        streams = [
            np.random.default_rng(s)
            for s in np.random.SeedSequence([seed, n, 103]).spawn(5)
        ]
        train = paired_bits(n, config["train_size"], streams[0])
        heldout = paired_bits(n, config["test_size"], streams[1])
        if (
            encode_bits(train) != data["train"]
            or encode_bits(heldout) != data["heldout"]
        ):
            raise ValueError("Data regeneration mismatch")
        fraction = config.get("priority_fraction", 0)
        selected = strong_pair_masks(train, n // 2).tolist() if fraction else []
        selection = record.get("feature_selection", {"masks": [], "fraction": 0})
        if selection["masks"] != selected or selection["fraction"] != fraction:
            raise ValueError("Training-only feature selection mismatch")
        size = (config["evaluation_shots"], n)
        baseline_bits = {
            "independent": (streams[2].random(size) < train.mean(axis=0)).astype(
                np.int32
            ),
            "empirical": train[streams[3].integers(0, len(train), size[0])],
            "paired_classical": np.repeat(
                (
                    streams[4].random((size[0], n // 2)) < train[:, 0::2].mean(axis=0)
                ).astype(np.int32),
                2,
                axis=1,
            ),
        }
        for model_name, values in baseline_bits.items():
            if encode_bits(values) != data["samples"][model_name]:
                raise ValueError("Classical sampling replay mismatch")
        resources = record["fit"]["resources"]
        if resources["training_shots"] or resources["training_statevector_evaluations"]:
            raise ValueError("Unexpected quantum training resource")
        if (
            resources["classical_phase_evaluations"]
            != 2
            * config["updates"]
            * config["masks_per_step"]
            * config["samples_per_half"]
            or len(record["fit"]["training_history"]) != config["updates"]
        ):
            raise ValueError("Training budget mismatch")
        for model, strings in data["samples"].items():
            bits = decode_bits(strings, n)
            if len(bits) != config["evaluation_shots"]:
                raise ValueError("Sampling budget mismatch")
            computed = {
                **pair_diagnostics(bits),
                "heldout_mmd_u": sample_mmd(bits, heldout),
            }
            if any(
                not np.isclose(v, record["metrics"][model][k], atol=1e-12, rtol=0)
                for k, v in computed.items()
            ):
                raise ValueError("Metric mismatch")
            checks += 1
        spec = CircuitSpec.from_dict(record["circuit"])
        if spec != make_circuit(n, "iqp", "line", seed=seed):
            raise ValueError("Scaling circuit specification mismatch")
        if not np.allclose(
            initial_parameters(spec, seed),
            record["fit"]["initial_parameters"],
            atol=1e-12,
            rtol=0,
        ):
            raise ValueError("Scaling initialization mismatch")
        replay, _ = sample_qiskit_mps(
            spec, record["fit"]["parameters"], config["evaluation_shots"], seed + 3109
        )
        if encode_bits(replay) != data["samples"]["iqp"]:
            raise ValueError("Qiskit sampling replay mismatch")
        initial_replay, _ = sample_qiskit_mps(
            spec,
            record["fit"]["initial_parameters"],
            config["evaluation_shots"],
            seed + 3109,
        )
        if encode_bits(initial_replay) != data["samples"]["initial_iqp"]:
            raise ValueError("Initial Qiskit sampling replay mismatch")
    report = {
        "passed": True,
        "jobs": len(expected),
        "sample_sets_checked": checks,
        "qiskit_sampling_replays": 2 * len(expected),
        "index_sha256": file_digest(output / "index.json"),
    }
    write_json(output / "audit.json", report)
    return report
