"""Validate frozen observable-comparison records and optionally replay Adam."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits

from .audit import require
from .circuits import CircuitSpec, ExactCircuit, make_circuit
from .data import bit_table, probabilities_from_samples
from .experiment import digest, file_digest, write_json
from .generalization import (
    evaluate_fit,
    fit_observables,
    make_data,
    objective_kernel,
    qiskit_state,
    score_distribution,
    selection_record,
    target_and_witness,
)
from .iqp import characters
from .training import initial_parameters


def close(actual, expected, label):
    require(np.shape(actual) == np.shape(expected), f"Shape mismatch: {label}")
    require(np.allclose(actual, expected, atol=1e-11, rtol=0), f"Mismatch: {label}")


def audit(output, replay=False):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    config = manifest["config"]
    require(
        manifest["config_sha256"] == digest(config), "Configuration receipt mismatch"
    )
    require(
        manifest["source"]["sha256"] == digest(manifest["source"]["files"]),
        "Source receipt mismatch",
    )
    for folder, files in [
        ("source", manifest["source"]["files"]),
        ("", manifest["frozen_files"]),
    ]:
        for name, expected in files.items():
            require(Path(name).name == name, "Unsafe snapshot path")
            require(
                file_digest(output / folder / name) == expected, "Frozen file mismatch"
            )
    require(
        json.loads((output / "config.json").read_text()) == config,
        "Frozen configuration mismatch",
    )
    index = json.loads((output / "index.json").read_text())
    expected_jobs = {
        f"{condition}-s{seed}"
        for condition in config["conditions"]
        for seed in config["seeds"]
    }
    require(index["expected_jobs"] == len(expected_jobs), "Wrong job count")
    require(
        set(index["completed"]) == expected_jobs and not index["failed"],
        "Incomplete run",
    )
    checked = replays = 0
    with threadpool_limits(limits=1):
        for condition in config["conditions"]:
            for seed in config["seeds"]:
                name = f"{condition}-s{seed}"
                path = output / "runs" / f"{name}.json"
                require(file_digest(path) == index["completed"][name], "Changed result")
                record = json.loads(path.read_text())
                require(
                    (record["condition"], record["seed"]) == (condition, seed),
                    "Wrong job identity",
                )
                require(
                    record["source_sha256"] == manifest["source"]["sha256"],
                    "Wrong source",
                )
                require(
                    record["config_sha256"] == manifest["config_sha256"], "Wrong config"
                )
                require(not record["failures"], "Failed fits remain")
                data = make_data(
                    condition, seed, config["train_size"], config["test_size"]
                )
                require(
                    record["data"] == data and record["data_sha256"] == digest(data),
                    "Data mismatch",
                )
                target, witness = target_and_witness(condition)
                close(record["target"], target, "target")
                spec = CircuitSpec.from_dict(record["circuit"])
                require(spec == make_circuit(6, "iqp", "dense"), "Circuit mismatch")
                close(record["witness"]["parameters"], witness, "witness angles")
                close(
                    ExactCircuit(spec).probabilities(witness),
                    target,
                    "witness probabilities",
                )
                close(
                    qiskit_state(spec, witness).probabilities(),
                    target,
                    "Qiskit witness",
                )
                train = bit_table(6)[data["train"]].astype(np.int32)
                heldout = probabilities_from_samples(np.array(data["heldout"]), 6)
                selection = selection_record(train, config)
                for selector in selection:
                    for key in selection[selector]:
                        if key != "seconds":
                            close(
                                record["selection"][selector][key],
                                selection[selector][key],
                                key,
                            )
                expected_models = {
                    "empirical",
                    "independent",
                    "pairwise",
                    "autoregressive_mlp",
                    "selected_maxent",
                    "order4_maxent",
                    "oracle_target",
                }
                expected_models |= {
                    f"iqp_{loss}_{estimator}"
                    for loss in config["losses"]
                    for estimator in config["estimators"]
                }
                require(len(record["fits"]) == len(expected_models), "Wrong fit count")
                require(
                    {f["model"] for f in record["fits"]} == expected_models,
                    "Wrong model set",
                )
                for fit in record["fits"]:
                    p = np.array(fit["probabilities"])
                    computed = score_distribution(p, target, heldout, condition)
                    for metric, value in computed.items():
                        close(fit["metrics"][metric], value, metric)
                    state = None
                    if fit["model"].startswith("iqp_"):
                        close(
                            p,
                            ExactCircuit(spec).probabilities(fit["parameters"]),
                            "NumPy circuit probabilities",
                        )
                        state = qiskit_state(spec, fit["parameters"])
                        close(p, state.probabilities(), "Qiskit circuit probabilities")
                        close(
                            fit["initial_parameters"],
                            initial_parameters(spec, seed),
                            "initial angles",
                        )
                        require(
                            fit["model"] == f"iqp_{fit['loss']}_{fit['estimator']}",
                            "Wrong fit identity",
                        )
                        require(
                            fit["iterations"] == config["updates"],
                            "Update count mismatch",
                        )
                        require(
                            fit["optimizer"] == "Adam" and fit["converged"] is None,
                            "Optimizer mismatch",
                        )
                        for key in ["training_history", "gradient_norm_history"]:
                            require(
                                len(fit[key]) == config["updates"]
                                and np.all(np.isfinite(fit[key])),
                                "Invalid history",
                            )
                        mc = fit["estimator"] == "mc"
                        expected_resources = {
                            "objective_evaluations": config["updates"],
                            "gradient_evaluations": config["updates"],
                            "classical_fourier_masks": config["updates"]
                            * config["masks_per_step"],
                            "classical_uniform_bitstrings": 2
                            * config["updates"]
                            * config["samples_per_half"]
                            if mc
                            else 0,
                            "classical_phase_evaluations": config["updates"]
                            * config["masks_per_step"]
                            * (2 * config["samples_per_half"] if mc else 64),
                            "training_shots": 0,
                            "training_statevector_evaluations": 0,
                            "exact_objective_diagnostics": 2,
                        }
                        require(
                            fit["resources"] == expected_resources, "Budget mismatch"
                        )
                        priority = (
                            None
                            if fit["loss"] == "multiscale"
                            else np.array(selection[fit["loss"]]["masks"])
                        )
                        kernel = objective_kernel(
                            6, priority, config["priority_fraction"]
                        )
                        empirical = probabilities_from_samples(
                            np.array(data["train"]), 6
                        )
                        for label, values in [
                            ("initial", fit["initial_parameters"]),
                            ("final", fit["parameters"]),
                        ]:
                            loss, gradient = ExactCircuit(spec).loss_and_grad(
                                values, empirical, kernel
                            )
                            close(
                                fit["diagnostics"][label]["loss"],
                                loss,
                                "objective diagnostic",
                            )
                            close(
                                fit["diagnostics"][label]["gradient_norm"],
                                np.linalg.norm(gradient),
                                "gradient diagnostic",
                            )
                        if replay:
                            repeated = fit_observables(
                                spec, train, seed, config, priority, fit["estimator"]
                            )
                            for key in [
                                "parameters",
                                "training_history",
                                "gradient_norm_history",
                            ]:
                                close(fit[key], repeated[key], f"Adam replay {key}")
                            replays += 1
                    elif "feature_masks" in fit:
                        close(
                            p,
                            softmax(
                                characters(bit_table(6), np.array(fit["feature_masks"]))
                                @ fit["parameters"]
                            ),
                            "classical parameters",
                        )
                    elif fit["model"] == "oracle_target":
                        close(p, target, "oracle")
                    require(fit["sample_seed"] == seed + 3109, "Sampling seed mismatch")
                    regenerated = {"probabilities": p.tolist()}
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
                        fit["sample_counts"] == regenerated["sample_counts"],
                        "Sample replay mismatch",
                    )
                    for metric, value in regenerated["sample_metrics"].items():
                        close(fit["sample_metrics"][metric], value, f"sample {metric}")
                    checked += 1
    report = {
        "passed": True,
        "jobs": len(expected_jobs),
        "fits_checked": checked,
        "adam_replays": replays,
        "index_sha256": file_digest(output / "index.json"),
    }
    write_json(output / "audit.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args.output, args.replay), indent=2))
