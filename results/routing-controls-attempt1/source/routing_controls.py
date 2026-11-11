"""Equal-parameter, expressible sparse IQP families on simulated coupling graphs."""

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
from qiskit import qasm3, transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from threadpoolctl import threadpool_limits

from .audit import require
from .circuits import CircuitSpec, ExactCircuit, Gate, make_circuit, to_qiskit
from .data import bit_table, probabilities_from_samples
from .deployment import logical_probabilities, noise_model
from .diagnostics import fit_full
from .experiment import (
    digest,
    file_digest,
    save_source_snapshot,
    source_receipt,
    write_json,
)
from .generalization import (
    make_data,
    qiskit_state,
    score_distribution,
    target_and_witness,
    target_blocks,
)
from .generalization_audit import close
from .training import initial_parameters


def sparse_spec(condition, topology):
    blocks, _ = target_blocks(condition)
    occupied = set(q for block in blocks for q in block)
    components = list(blocks) + [(q,) for q in sorted(set(range(6)) - occupied)]
    edges = {(block[0], q) for block in blocks for q in block[1:]}
    if topology == "path_bridges":
        edges.update(
            (left[-1], right[0])
            for left, right in zip(components[:-1], components[1:], strict=True)
        )
    elif topology == "hub_bridges":
        edges.update((components[0][0], block[0]) for block in components[1:])
    else:
        raise ValueError("Unknown sparse topology")
    edges = tuple(sorted(tuple(sorted(edge)) for edge in edges))
    require(len(edges) == len(set(edges)) == 5, "Expected a five-edge tree")
    supports = [(q,) for q in range(6)] + list(edges)
    spec = CircuitSpec(
        6,
        "iqp",
        topology,
        edges,
        tuple(Gate("px", support, i) for i, support in enumerate(supports)),
        11,
    )
    _, dense_witness = target_and_witness(condition)
    dense = make_circuit(6, "iqp", "dense")
    angle_map = {gate.qubits: dense_witness[gate.parameter] for gate in dense.gates}
    return spec, np.array([angle_map[gate.qubits] for gate in spec.gates])


def compile_layout(spec, theta, device, config):
    if device == "line":
        edges = [(q, q + 1) for q in range(5)]
    elif device == "ring":
        edges = [(q, (q + 1) % 6) for q in range(6)]
    elif device == "star":
        edges = [(0, q) for q in range(1, 6)]
    else:
        raise ValueError("Unknown device")
    coupling = CouplingMap(edges + [(b, a) for a, b in edges])
    circuit, parameters = to_qiskit(spec)
    circuit = circuit.assign_parameters(dict(zip(parameters, theta, strict=True)))
    circuit.measure_all()
    return transpile(
        circuit,
        basis_gates=["rz", "sx", "x", "cx"],
        coupling_map=coupling,
        initial_layout=list(range(6)),
        optimization_level=config["optimization_level"],
        seed_transpiler=config["transpiler_seed"],
    )


def deploy(spec, theta, device, config, seed):
    compiled = compile_layout(spec, theta, device, config)
    mapping = []
    for instruction in compiled.data:
        if instruction.operation.name == "measure":
            mapping.append(
                [
                    compiled.find_bit(instruction.qubits[0]).index,
                    compiled.find_bit(instruction.clbits[0]).index,
                ]
            )
    metadata = {
        "device": device,
        "depth": compiled.depth(),
        "cx_count": compiled.count_ops().get("cx", 0),
        "gate_counts": dict(compiled.count_ops()),
        "measurement_mapping": mapping,
        "initial_layout": list(range(6)),
        "qasm": qasm3.dumps(compiled),
    }
    outcomes = []
    start = perf_counter()
    ideal = ExactCircuit(spec).probabilities(theta)
    for error in config["errors"]:
        simulator = AerSimulator(
            method="density_matrix",
            noise_model=noise_model(error),
            max_parallel_threads=1,
        )
        probe = compiled.remove_final_measurements(inplace=False)
        probe.save_probabilities()
        result = simulator.run(probe, shots=1, seed_simulator=seed).result()
        require(result.success, "Density-matrix simulation failed")
        p = logical_probabilities(np.asarray(result.data(0)["probabilities"]), compiled)
        if error == 0:
            close(p, ideal, "Noiseless routed distribution")
        sampled = simulator.run(
            compiled, shots=config["evaluation_shots"], seed_simulator=seed
        ).result()
        require(sampled.success, "Sampling failed")
        counts = np.zeros(64, dtype=int)
        for key, count in sampled.get_counts().items():
            counts[int(key.replace(" ", ""), 2)] = count
        outcomes.append(
            {
                "error": error,
                "probabilities": p.tolist(),
                "sample_counts": counts.tolist(),
                "sample_seed": seed,
                "tvd_from_ideal_output": float(np.abs(p - ideal).sum() / 2),
            }
        )
    return {
        "compilation": metadata,
        "outcomes": outcomes,
        "seconds": perf_counter() - start,
    }


def run_record(config, condition, seed, reference):
    data = make_data(condition, seed, config["train_size"], config["test_size"])
    require(data == reference["data"], "Data differs from classical references")
    train = bit_table(6)[data["train"]].astype(np.int32)
    heldout = probabilities_from_samples(np.array(data["heldout"]), 6)
    target, _ = target_and_witness(condition)
    record = {
        "condition": condition,
        "seed": seed,
        "data": data,
        "data_sha256": digest(data),
        "target": target.tolist(),
        "fits": [],
        "failures": [],
        "classical": [
            f for f in reference["fits"] if not f["model"].startswith("iqp_")
        ],
    }
    for topology in config["topologies"]:
        try:
            spec, witness = sparse_spec(condition, topology)
            close(
                qiskit_state(spec, witness).probabilities(),
                target,
                "Sparse expressibility",
            )
            fit = fit_full(spec, train, seed, config)
            ideal = qiskit_state(spec, fit["parameters"]).probabilities()
            fit.update(
                topology=topology,
                circuit=spec.to_dict(),
                witness=witness.tolist(),
                probabilities=ideal.tolist(),
                metrics=score_distribution(ideal, target, heldout, condition),
                deployments=[],
            )
            for device in config["devices"]:
                deployed = deploy(spec, fit["parameters"], device, config, seed + 3109)
                for outcome in deployed["outcomes"]:
                    outcome["metrics"] = score_distribution(
                        np.array(outcome["probabilities"]), target, heldout, condition
                    )
                    outcome["sample_metrics"] = score_distribution(
                        np.array(outcome["sample_counts"]) / config["evaluation_shots"],
                        target,
                        heldout,
                        condition,
                    )
                    outcome["target_tvd_change"] = (
                        outcome["metrics"]["tvd"] - fit["metrics"]["tvd"]
                    )
                fit["deployments"].append(deployed)
            record["fits"].append(fit)
        except Exception:
            record["failures"].append(
                {"topology": topology, "traceback": traceback.format_exc()}
            )
    return record


def run(config_path, output, reference):
    config_path, output, reference = Path(config_path), Path(output), Path(reference)
    config = json.loads(config_path.read_text())
    require(config["n_qubits"] == 6, "This protocol requires six qubits")
    output.mkdir(parents=True, exist_ok=False)
    source = source_receipt()
    save_source_snapshot(output, source)
    root = Path(__file__).resolve().parents[2]
    frozen = {}
    for name, path in [
        ("config.json", config_path),
        ("uv.lock", root / "uv.lock"),
        ("protocol.md", root / "docs/routing-controls-protocol.md"),
    ]:
        shutil.copyfile(path, output / name)
        frozen[name] = file_digest(output / name)
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
    with threadpool_limits(limits=1):
        spec, _ = sparse_spec("adjacent_pairs", "path_bridges")
        start = perf_counter()
        deploy(
            spec,
            np.random.default_rng(991).normal(size=11),
            "line",
            {**config, "errors": [0.005], "evaluation_shots": 64},
            991,
        )
        write_json(
            output / "preflight.json",
            {
                "single_compilation_and_noise_evaluation_seconds": perf_counter()
                - start,
                "workers": 1,
                "simulator_threads": 1,
                "blas_threads": 1,
                "target_fit_outcomes_used": False,
            },
        )
        for condition in config["conditions"]:
            for seed in config["seeds"]:
                name = f"{condition}-s{seed}"
                ref_path = reference / "runs" / f"{name}.json"
                require(
                    file_digest(ref_path) == ref_index["completed"][name],
                    "Reference mismatch",
                )
                shutil.copyfile(ref_path, output / "reference" / f"{name}.json")
                try:
                    require(source_receipt() == source, "Source changed during run")
                    record = run_record(
                        config, condition, seed, json.loads(ref_path.read_text())
                    )
                    record.update(
                        config_sha256=digest(config),
                        source_sha256=source["sha256"],
                        reference_sha256=file_digest(ref_path),
                    )
                    path = output / "runs" / f"{name}.json"
                    write_json(path, record)
                    index["completed"][name] = file_digest(path)
                    if record["failures"]:
                        index["failed"][name] = record["failures"]
                    print(
                        f"Completed {name}: {len(record['fits'])} circuits", flush=True
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
    ref_index = json.loads((output / "reference/index.json").read_text())
    index = json.loads((output / "index.json").read_text())
    expected = {f"{c}-s{s}" for c in config["conditions"] for s in config["seeds"]}
    require(
        set(index["completed"]) == expected
        and index["expected_jobs"] == len(expected)
        and not index["failed"],
        "Incomplete run",
    )
    checked = replays = 0
    with threadpool_limits(limits=1):
        for name in sorted(expected):
            path = output / "runs" / f"{name}.json"
            require(file_digest(path) == index["completed"][name], "Changed result")
            record = json.loads(path.read_text())
            c, seed = record["condition"], record["seed"]
            require(
                name == f"{c}-s{seed}"
                and record["config_sha256"] == manifest["config_sha256"]
                and record["source_sha256"] == manifest["source"]["sha256"],
                "Record identity mismatch",
            )
            ref_path = output / "reference" / f"{name}.json"
            require(
                file_digest(ref_path)
                == ref_index["completed"][name]
                == record["reference_sha256"],
                "Reference mismatch",
            )
            reference = json.loads(ref_path.read_text())
            require(
                record["classical"]
                == [f for f in reference["fits"] if not f["model"].startswith("iqp_")],
                "Classical reference mismatch",
            )
            data = make_data(c, seed, config["train_size"], config["test_size"])
            require(
                data == record["data"] and digest(data) == record["data_sha256"],
                "Data mismatch",
            )
            target, _ = target_and_witness(c)
            close(record["target"], target, "target")
            heldout = probabilities_from_samples(np.array(data["heldout"]), 6)
            train = bit_table(6)[data["train"]].astype(np.int32)
            require(
                {f["topology"] for f in record["fits"]} == set(config["topologies"])
                and len(record["fits"]) == len(config["topologies"])
                and not record["failures"],
                "Fit set mismatch",
            )
            for fit in record["fits"]:
                spec, witness = sparse_spec(c, fit["topology"])
                require(
                    digest(spec.to_dict()) == digest(fit["circuit"]), "Circuit mismatch"
                )
                close(fit["witness"], witness, "witness angles")
                close(
                    qiskit_state(spec, witness).probabilities(),
                    target,
                    "witness probabilities",
                )
                ideal = ExactCircuit(spec).probabilities(fit["parameters"])
                close(fit["probabilities"], ideal, "ideal probabilities")
                close(
                    ideal,
                    qiskit_state(spec, fit["parameters"]).probabilities(),
                    "Qiskit probabilities",
                )
                close(
                    fit["initial_parameters"],
                    initial_parameters(spec, seed),
                    "initial parameters",
                )
                for key, value in score_distribution(ideal, target, heldout, c).items():
                    close(fit["metrics"][key], value, key)
                require(
                    fit["iterations"] == config["updates"]
                    and fit["optimizer"] == "Adam"
                    and fit["converged"] is None,
                    "Optimizer mismatch",
                )
                for field in ["training_history", "gradient_norm_history"]:
                    require(
                        len(fit[field]) == config["updates"]
                        and np.all(np.isfinite(fit[field])),
                        "Invalid history",
                    )
                # A deterministic rerun also validates initialization, diagnostics and resource accounting.
                if replay:
                    refit = fit_full(spec, train, seed, config)
                    for key in [
                        "parameters",
                        "training_history",
                        "gradient_norm_history",
                    ]:
                        close(fit[key], refit[key], f"Training replay {key}")
                    require(fit["resources"] == refit["resources"], "Resource mismatch")
                require(
                    fit["resources"]["classical_phase_evaluations"]
                    == 4096 * config["updates"]
                    and fit["resources"]["sampled_masks"] == 0
                    and fit["resources"]["training_shots"] == 0,
                    "Budget mismatch",
                )
                require(
                    [d["compilation"]["device"] for d in fit["deployments"]]
                    == config["devices"],
                    "Device set mismatch",
                )
                for deployed in fit["deployments"]:
                    device = deployed["compilation"]["device"]
                    require(
                        [o["error"] for o in deployed["outcomes"]] == config["errors"],
                        "Error set mismatch",
                    )
                    if replay:
                        rerun = deploy(
                            spec, fit["parameters"], device, config, seed + 3109
                        )
                        require(
                            rerun["compilation"] == deployed["compilation"],
                            "Compilation replay mismatch",
                        )
                    for i, outcome in enumerate(deployed["outcomes"]):
                        p = np.array(outcome["probabilities"])
                        counts = np.array(outcome["sample_counts"])
                        require(
                            counts.shape == (64,)
                            and np.all(counts >= 0)
                            and np.all(counts == counts.astype(int))
                            and counts.sum() == config["evaluation_shots"],
                            "Sample budget mismatch",
                        )
                        require(outcome["sample_seed"] == seed + 3109, "Seed mismatch")
                        for values, field in [
                            (p, "metrics"),
                            (counts / config["evaluation_shots"], "sample_metrics"),
                        ]:
                            for metric, value in score_distribution(
                                values, target, heldout, c
                            ).items():
                                close(outcome[field][metric], value, metric)
                        close(
                            outcome["target_tvd_change"],
                            outcome["metrics"]["tvd"] - fit["metrics"]["tvd"],
                            "TVD change",
                        )
                        close(
                            outcome["tvd_from_ideal_output"],
                            np.abs(p - ideal).sum() / 2,
                            "Implementation TVD",
                        )
                        if outcome["error"] == 0:
                            close(p, ideal, "Noiseless routing")
                        if replay:
                            close(
                                p, rerun["outcomes"][i]["probabilities"], "Noise replay"
                            )
                            require(
                                outcome["sample_counts"]
                                == rerun["outcomes"][i]["sample_counts"],
                                "Sample replay",
                            )
                            replays += 1
                        checked += 1
    report = {
        "passed": True,
        "jobs": len(expected),
        "noise_evaluations_checked": checked,
        "noise_evaluations_replayed": replays,
        "training_fits_replayed": len(expected) * len(config["topologies"])
        if replay
        else 0,
        "index_sha256": file_digest(output / "index.json"),
    }
    write_json(output / "audit.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "audit"])
    parser.add_argument("--config", default="config/routing-controls-v1.json")
    parser.add_argument("--reference", default="results/observable-generalization-v1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.action == "run":
        run(args.config, args.output, args.reference)
    else:
        print(json.dumps(audit(args.output, args.replay), indent=2))
