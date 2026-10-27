"""Recompute result metrics from saved arrays without using training metrics."""

import json
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector

from .circuits import CircuitSpec, to_qiskit
from .data import bit_table, make_dataset
from .experiment import digest, file_digest, jobs_for, write_json


def audit_experiment(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    index = json.loads((output / "index.json").read_text())
    config = manifest["config"]
    assert manifest["config_sha256"] == digest(config), "Configuration receipt mismatch"
    assert manifest["source"]["sha256"] == digest(manifest["source"]["files"]), "Source receipt mismatch"
    for name, expected in manifest["source"]["files"].items():
        if Path(name).name != name:
            raise ValueError("Invalid source snapshot path")
        if file_digest(output / "source" / name) != expected:
            raise ValueError(f"Source snapshot mismatch: {name}")
    jobs = jobs_for(config)
    assert index["expected_jobs"] == len(jobs) == len(index["completed"]), "Incomplete run"
    assert not index["failed"], "Failed jobs remain"
    n, checked = config["n_qubits"], 0
    bits = bit_table(n)
    hamming = np.sum(bits[:, None] != bits[None, :], axis=-1)
    kernel = sum(np.exp(-hamming / (2 * n * b)) for b in [.01, .1, .25, .5, 1, 10]) / 6

    def verify(p, metrics, target, test, name):
        nonlocal checked
        p = np.asarray(p, float)
        assert p.shape == target.shape and np.all(np.isfinite(p))
        assert p.min() >= -1e-12 and np.isclose(p.sum(), 1, atol=1e-9)
        p = np.maximum(p, 0)
        assert np.isclose(np.sum(abs(p - target)) / 2, metrics["tvd"], atol=1e-11)
        middle = (p + target) / 2
        js = 0.0
        for dist in [p, target]:
            mask = dist > 0
            js += np.sum(dist[mask] * np.log(dist[mask] / middle[mask])) / 2
        assert np.isclose(js, metrics["js_nats"], atol=1e-11)
        diff = p - test
        assert np.isclose(diff @ kernel @ diff, metrics["heldout_mmd"], atol=1e-11)
        parity = (-1.) ** bits.sum(axis=1)
        assert np.isclose(abs((p - target) @ parity), metrics["parity_error"], atol=1e-11)
        if name == "bars_stripes":
            valid = target > 0
            assert np.isclose(p[~valid].sum(), metrics["invalid_mass"], atol=1e-11)
            assert np.isclose(np.mean(1 - (1 - p[valid]) ** 1024), metrics["expected_coverage_1024"])
        checked += 1

    for job in jobs:
        path = output / "runs" / f"{job['id']}.json"
        assert file_digest(path) == index["completed"][job["id"]], f"Changed result: {path}"
        result = json.loads(path.read_text())
        assert result["job"] == job
        assert result["source_sha256"] == manifest["source"]["sha256"]
        assert result["config_sha256"] == manifest["config_sha256"]
        data = result["dataset"]
        assert digest(data) == result["dataset_sha256"]
        regenerated = make_dataset(job["dataset"], n, job["seed"], config["train_size"], config["test_size"])
        assert data["train_counts"] == np.bincount(regenerated.train_samples, minlength=1 << n).tolist()
        assert data["test_counts"] == np.bincount(regenerated.test_samples, minlength=1 << n).tolist()
        target, test = regenerated.target, regenerated.test
        assert np.allclose(target, data["target"])
        fits = result.get("fits", [result.get("fit")])
        for fit in fits:
            verify(fit["probabilities"], fit["metrics"], target, test, job["dataset"])
        if job["kind"] == "quantum":
            fit = result["fit"]
            verify(fit["initial_probabilities"], fit["initial_metrics"], target, test, job["dataset"])
            spec = CircuitSpec.from_dict(result["circuit"])
            circuit, parameters = to_qiskit(spec)
            qiskit_p = Statevector(circuit.assign_parameters(dict(zip(parameters, fit["parameters"])))).probabilities()
            assert np.allclose(qiskit_p, fit["probabilities"], atol=1e-10)
            initial_p = Statevector(circuit.assign_parameters(dict(zip(parameters, fit["initial_parameters"])))).probabilities()
            assert np.allclose(initial_p, fit["initial_probabilities"], atol=1e-10, rtol=0)
            history_length = config["updates"] if job["shots"] else fit["resources"]["objective_evaluations"]
            assert len(fit["training_history"]) == history_length
            assert np.all(np.isfinite(fit["training_history"]))
            if config["mode"] == "iqp_mc":
                resource = fit["resources"]
                assert resource["training_shots"] == resource["training_statevector_evaluations"] == 0
                assert resource["objective_evaluations"] == config["updates"]
                assert resource["classical_uniform_bitstrings"] == 2 * config["updates"] * config["samples_per_half"]
                assert resource["classical_fourier_masks"] == config["updates"] * config["masks_per_step"]
                assert resource["classical_phase_evaluations"] == 2 * config["updates"] * config["samples_per_half"] * config["masks_per_step"]
            if job["shots"]:
                assert fit["resources"]["objective_evaluations"] == 2 * config["updates"]
                assert fit["resources"]["training_shots"] == 2 * config["updates"] * job["shots"]
            for deployment in result.get("deployments", []):
                verify(deployment["probabilities"], deployment["metrics"], target, test, job["dataset"])
                verify(deployment["sampled_probabilities"], deployment["sampled_metrics"], target, test, job["dataset"])
                counts = np.asarray(deployment["sampled_probabilities"]) * deployment["evaluation_shots"]
                assert np.allclose(counts, np.round(counts))
                if deployment["cx_depolarizing"] == 0:
                    assert np.allclose(deployment["probabilities"], qiskit_p, atol=1e-9)
    report = {"passed": True, "jobs": len(jobs), "distributions_checked": checked,
              "index_sha256": file_digest(output / "index.json"), "source_sha256": manifest["source"]["sha256"]}
    write_json(output / "audit.json", report)
    return report

