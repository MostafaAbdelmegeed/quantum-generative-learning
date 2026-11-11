"""Recompute result metrics from saved arrays without using training metrics."""

import json
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector

from .circuits import CircuitSpec, make_circuit, to_qiskit
from .data import bit_table, make_dataset
from .experiment import digest, file_digest, jobs_for, source_receipt, write_json
from .training import initial_parameters


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_experiment(output, deep=False):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    index = json.loads((output / "index.json").read_text())
    config = manifest["config"]
    require(
        manifest["config_sha256"] == digest(config), "Configuration receipt mismatch"
    )
    require(
        manifest["source"]["sha256"] == digest(manifest["source"]["files"]),
        "Source receipt mismatch",
    )
    for name, expected in manifest["source"]["files"].items():
        if Path(name).name != name:
            raise ValueError("Invalid source snapshot path")
        if file_digest(output / "source" / name) != expected:
            raise ValueError(f"Source snapshot mismatch: {name}")
    jobs = jobs_for(config)
    require(
        index["expected_jobs"] == len(jobs) == len(index["completed"]), "Incomplete run"
    )
    require(not index["failed"], "Failed jobs remain")
    require(
        set(index["completed"]) == {job["id"] for job in jobs},
        "Unexpected indexed jobs",
    )
    n, checked, noise_replays = (config["n_qubits"], 0, 0)
    bits = bit_table(n)
    hamming = np.sum(bits[:, None] != bits[None, :], axis=-1)
    kernel = (
        sum(np.exp(-hamming / (2 * n * b)) for b in [0.01, 0.1, 0.25, 0.5, 1, 10]) / 6
    )

    def verify(p, metrics, target, test, name):
        nonlocal checked
        p = np.asarray(p, float)
        require(
            p.shape == target.shape and np.all(np.isfinite(p)),
            "Invalid distribution shape or nonfinite probabilities",
        )
        require(
            p.min() >= -1e-12 and np.isclose(p.sum(), 1, atol=1e-09, rtol=0),
            "Negative or unnormalized probabilities",
        )
        p = np.maximum(p, 0)
        require(
            np.isclose(np.sum(abs(p - target)) / 2, metrics["tvd"], atol=1e-11, rtol=0),
            "Mismatch: tvd",
        )
        middle = (p + target) / 2
        js = 0.0
        for dist in [p, target]:
            mask = dist > 0
            js += np.sum(dist[mask] * np.log(dist[mask] / middle[mask])) / 2
        require(
            np.isclose(js, metrics["js_nats"], atol=1e-11, rtol=0), "Mismatch: js_nats"
        )
        diff = p - test
        require(
            np.isclose(
                diff @ kernel @ diff, metrics["heldout_mmd"], atol=1e-11, rtol=0
            ),
            "Mismatch: heldout_mmd",
        )
        parity = (-1.0) ** bits.sum(axis=1)
        require(
            np.isclose(
                abs((p - target) @ parity), metrics["parity_error"], atol=1e-11, rtol=0
            ),
            "Mismatch: parity_error",
        )
        if name == "bars_stripes":
            valid = target > 0
            require(
                np.isclose(
                    p[~valid].sum(), metrics["invalid_mass"], atol=1e-11, rtol=0
                ),
                "Mismatch: invalid_mass",
            )
            require(
                np.isclose(
                    np.mean(1 - (1 - p[valid]) ** 1024),
                    metrics["expected_coverage_1024"],
                    rtol=0,
                ),
                "Mismatch: expected_coverage_1024",
            )
        checked += 1

    for job in jobs:
        path = output / "runs" / f"{job['id']}.json"
        require(
            file_digest(path) == index["completed"][job["id"]],
            f"Changed result: {path}",
        )
        result = json.loads(path.read_text())
        require(result["job"] == job, "Mismatch: job")
        require(
            result["source_sha256"] == manifest["source"]["sha256"],
            "Mismatch: source_sha256",
        )
        require(
            result["config_sha256"] == manifest["config_sha256"],
            "Mismatch: config_sha256",
        )
        data = result["dataset"]
        require(digest(data) == result["dataset_sha256"], "Mismatch: dataset_sha256")
        regenerated = make_dataset(
            job["dataset"], n, job["seed"], config["train_size"], config["test_size"]
        )
        require(
            data["train_counts"]
            == np.bincount(regenerated.train_samples, minlength=1 << n).tolist(),
            "Mismatch: train_counts",
        )
        require(
            data["test_counts"]
            == np.bincount(regenerated.test_samples, minlength=1 << n).tolist(),
            "Mismatch: test_counts",
        )
        target, test = (regenerated.target, regenerated.test)
        require(np.allclose(target, data["target"], rtol=0), "Mismatch: target")
        fits = result.get("fits", [result.get("fit")])
        for fit in fits:
            verify(fit["probabilities"], fit["metrics"], target, test, job["dataset"])
        if job["kind"] == "quantum":
            fit = result["fit"]
            verify(
                fit["initial_probabilities"],
                fit["initial_metrics"],
                target,
                test,
                job["dataset"],
            )
            spec = CircuitSpec.from_dict(result["circuit"])
            require(
                spec
                == make_circuit(
                    n, config["family"], job["topology"], config["depth"], job["seed"]
                ),
                "Circuit specification mismatch",
            )
            require(
                np.allclose(
                    initial_parameters(spec, job["seed"]),
                    fit["initial_parameters"],
                    atol=1e-12,
                    rtol=0,
                ),
                "Initialization mismatch",
            )
            circuit, parameters = to_qiskit(spec)
            qiskit_p = Statevector(
                circuit.assign_parameters(
                    dict(zip(parameters, fit["parameters"], strict=True))
                )
            ).probabilities()
            require(
                np.allclose(qiskit_p, fit["probabilities"], atol=1e-10, rtol=0),
                "Mismatch: probabilities",
            )
            initial_p = Statevector(
                circuit.assign_parameters(
                    dict(zip(parameters, fit["initial_parameters"], strict=True))
                )
            ).probabilities()
            require(
                np.allclose(
                    initial_p, fit["initial_probabilities"], atol=1e-10, rtol=0
                ),
                "Mismatch: initial_probabilities",
            )
            history_length = (
                config["updates"]
                if job["shots"]
                else fit["resources"]["objective_evaluations"]
            )
            require(
                len(fit["training_history"]) == history_length,
                "Mismatch: training_history",
            )
            require(
                np.all(np.isfinite(fit["training_history"])),
                "Mismatch: training_history",
            )
            if config["mode"] == "iqp_mc":
                resource = fit["resources"]
                require(
                    resource["training_shots"]
                    == resource["training_statevector_evaluations"]
                    == 0,
                    "Mismatch: training_shots",
                )
                require(
                    resource["objective_evaluations"] == config["updates"],
                    "Mismatch: objective_evaluations",
                )
                require(
                    resource["classical_uniform_bitstrings"]
                    == 2 * config["updates"] * config["samples_per_half"],
                    "Mismatch: classical_uniform_bitstrings",
                )
                require(
                    resource["classical_fourier_masks"]
                    == config["updates"] * config["masks_per_step"],
                    "Mismatch: classical_fourier_masks",
                )
                require(
                    resource["classical_phase_evaluations"]
                    == 2
                    * config["updates"]
                    * config["samples_per_half"]
                    * config["masks_per_step"],
                    "Mismatch: classical_phase_evaluations",
                )
            if job["shots"]:
                require(
                    fit["resources"]["objective_evaluations"] == 2 * config["updates"],
                    "Mismatch: objective_evaluations",
                )
                require(
                    fit["resources"]["training_shots"]
                    == 2 * config["updates"] * job["shots"],
                    "Mismatch: training_shots",
                )
            for deployment in result.get("deployments", []):
                verify(
                    deployment["probabilities"],
                    deployment["metrics"],
                    target,
                    test,
                    job["dataset"],
                )
                verify(
                    deployment["sampled_probabilities"],
                    deployment["sampled_metrics"],
                    target,
                    test,
                    job["dataset"],
                )
                counts = (
                    np.asarray(deployment["sampled_probabilities"])
                    * deployment["evaluation_shots"]
                )
                require(
                    np.allclose(counts, np.round(counts), rtol=0),
                    "Mismatch: np.allclose(counts",
                )
                if deployment["cx_depolarizing"] == 0:
                    require(
                        np.allclose(
                            deployment["probabilities"], qiskit_p, atol=1e-09, rtol=0
                        ),
                        "Mismatch: probabilities",
                    )
            if deep and "deployments" in result:
                from .deployment import evaluate_deployment

                compilation, replay = evaluate_deployment(
                    spec,
                    fit["parameters"],
                    config["noise_levels"],
                    config["evaluation_shots"],
                    job["seed"],
                )
                require(
                    compilation["qasm"] == result["compilation"]["qasm"],
                    "Compiled QASM replay mismatch",
                )
                for expected, actual in zip(result["deployments"], replay, strict=True):
                    require(
                        np.allclose(
                            actual["probabilities"],
                            expected["probabilities"],
                            atol=1e-09,
                            rtol=0,
                        ),
                        "Noisy density matrix replay mismatch",
                    )
                    require(
                        np.allclose(
                            actual["sampled_probabilities"],
                            expected["sampled_probabilities"],
                            atol=1e-10,
                            rtol=0,
                        ),
                        "Seeded Aer sampling replay mismatch",
                    )
                    noise_replays += 1
    report = {
        "passed": True,
        "jobs": len(jobs),
        "distributions_checked": checked,
        "noise_replays": noise_replays,
        "deep": deep,
        "index_sha256": file_digest(output / "index.json"),
        "source_sha256": manifest["source"]["sha256"],
        "auditor_source_sha256": source_receipt()["sha256"],
    }
    write_json(output / ("deep-audit.json" if deep else "audit.json"), report)
    return report
