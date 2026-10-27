"""Versioned runs with resumable records and source receipts."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import json
import os
import platform
import traceback

import numpy as np
from qiskit.quantum_info import Statevector
from threadpoolctl import threadpool_limits

from .circuits import ExactCircuit, make_circuit, to_qiskit
from .data import make_dataset
from .deployment import evaluate_deployment
from .metrics import evaluate
from .training import fit_classical, fit_exact, fit_shots


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_receipt():
    package = Path(__file__).parent
    files = {p.name: file_digest(p) for p in sorted(package.glob("*.py"))}
    return {"files": files, "sha256": digest(files)}


def jobs_for(config):
    jobs = []
    for dataset in config["datasets"]:
        for seed in config["seeds"]:
            common = {"dataset": dataset, "seed": seed}
            if config.get("classical_baselines"):
                jobs.append({**common, "kind": "classical", "id": f"{dataset}-s{seed}-classical"})
            for topology in config["topologies"]:
                for loss in config["losses"]:
                    shots_list = config["shots"] if config["mode"] == "shots" else [0]
                    for shots in shots_list:
                        name = f"{dataset}-s{seed}-{config['family']}-{topology}-{loss}-{shots}shots"
                        jobs.append({**common, "kind": "quantum", "topology": topology,
                                     "loss": loss, "shots": shots, "id": name})
    if len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("Duplicate experiment jobs")
    return jobs


def _run_job(config, job, source):
    if source_receipt()["sha256"] != source["sha256"]:
        raise RuntimeError("Source changed after the run manifest was created")
    data = make_dataset(job["dataset"], config["n_qubits"], job["seed"], config["train_size"], config["test_size"])
    dataset = {"name": data.name, "n": data.n, "train_size": config["train_size"],
               "test_size": config["test_size"], "target": data.target.tolist(),
               "train_counts": np.bincount(data.train_samples, minlength=1 << data.n).tolist(),
               "test_counts": np.bincount(data.test_samples, minlength=1 << data.n).tolist()}
    result = {"job": job, "config_sha256": digest(config), "source_sha256": source["sha256"],
              "dataset": dataset, "dataset_sha256": digest(dataset)}

    def score(probabilities):
        return evaluate(np.asarray(probabilities), data.target, data.test, data.n, data.name)

    if job["kind"] == "classical":
        result["fits"] = fit_classical(data.train_samples, data.n, job["seed"])
        for fit in result["fits"]:
            fit["metrics"] = score(fit["probabilities"])
        return result
    spec = make_circuit(data.n, config["family"], job["topology"], config["depth"], job["seed"])
    if job["shots"]:
        fit = fit_shots(spec, data.train, config["train_size"], job["loss"], job["seed"], job["shots"], config["updates"])
    else:
        fit = fit_exact(spec, data.train, job["loss"], job["seed"], config["maxiter"])
    fit["metrics"] = score(fit["probabilities"])
    fit["initial_metrics"] = score(fit["initial_probabilities"])
    circuit, parameters = to_qiskit(spec)
    qiskit_p = Statevector(circuit.assign_parameters(dict(zip(parameters, fit["parameters"])))).probabilities()
    error = float(np.max(abs(qiskit_p - np.asarray(fit["probabilities"]))))
    if error > 1e-10:
        raise RuntimeError(f"Qiskit probability mismatch: {error}")
    result.update({"circuit": spec.to_dict(), "fit": fit, "qiskit_probability_max_error": error})
    if "noise_levels" in config:
        compilation, deployments = evaluate_deployment(spec, fit["parameters"], config["noise_levels"],
                                                       config["evaluation_shots"], job["seed"])
        for deployment in deployments:
            deployment["metrics"] = score(deployment["probabilities"])
            deployment["sampled_metrics"] = score(deployment["sampled_probabilities"])
            if deployment["cx_depolarizing"] == 0:
                compiled_error = float(np.max(abs(np.asarray(deployment["probabilities"]) - qiskit_p)))
                if compiled_error > 1e-9:
                    raise RuntimeError(f"Routing changed logical outputs: {compiled_error}")
                result["compiled_probability_max_error"] = compiled_error
        result.update({"compilation": compilation, "deployments": deployments})
    return result


def run_job(config, job, source):
    with threadpool_limits(limits=1):
        return _run_job(config, job, source)


def run_experiment(config_path, output, workers=1):
    if workers < 1:
        raise ValueError("workers must be positive")
    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text())
    source = source_receipt()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["config_sha256"] != digest(config) or manifest["source"]["sha256"] != source["sha256"]:
            raise ValueError("Existing run has different config or source; choose a new output directory")
    else:
        root = Path(__file__).resolve().parents[2]
        manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "config": config,
                    "config_sha256": digest(config), "source": source, "python": platform.python_version(),
                    "dependencies": {p: version(p) for p in ["numpy", "scipy", "scikit-learn", "qiskit", "qiskit-aer"]},
                    "lockfile_sha256": file_digest(root / "uv.lock") if (root / "uv.lock").exists() else None}
        write_json(manifest_path, manifest)
    jobs = jobs_for(config)
    pending = []
    for job in jobs:
        path = output / "runs" / f"{job['id']}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record["config_sha256"] != digest(config) or record["source_sha256"] != source["sha256"] or record["job"] != job:
                raise ValueError(f"Mismatched existing result: {path}")
        else:
            pending.append(job)
    print(f"{config['experiment']}: {len(jobs) - len(pending)}/{len(jobs)} already complete", flush=True)
    failures = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_job, config, job, source): job for job in pending}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                write_json(output / "runs" / f"{job['id']}.json", result)
                score = result.get("fit", {}).get("metrics", {}).get("tvd")
                print(f"Completed {job['id']}" + (f" TVD={score:.4f}" if score is not None else ""), flush=True)
            except Exception:
                failure = {"job": job, "traceback": traceback.format_exc()}
                failures.append(failure)
                write_json(output / "failures" / f"{job['id']}.json", failure)
                print(f"FAILED {job['id']}: {failure['traceback']}", flush=True)
    complete = {job["id"]: file_digest(output / "runs" / f"{job['id']}.json") for job in jobs
                if (output / "runs" / f"{job['id']}.json").exists()}
    write_json(output / "index.json", {"expected_jobs": len(jobs), "completed": complete, "failed": len(failures)})
    if failures:
        raise RuntimeError(f"{len(failures)} jobs failed; inspect {output / 'failures'}")

