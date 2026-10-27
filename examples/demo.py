"""Train a six-qubit generator and draw unfiltered samples through Qiskit."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import qasm3
from qiskit.primitives import StatevectorSampler
from threadpoolctl import threadpool_limits

from qgen.circuits import ExactCircuit, make_circuit, to_qiskit
from qgen.data import bit_table, make_dataset
from qgen.experiment import source_receipt, write_json
from qgen.iqp import fit_iqp_mc
from qgen.metrics import evaluate
from qgen.training import fit_exact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["qcbm", "iqp"], default="qcbm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts/demo")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    data = make_dataset("bars_stripes", 6, args.seed, 512, 1024)
    spec = make_circuit(
        6,
        args.method,
        "dense" if args.method == "iqp" else "line",
        depth=3,
        seed=args.seed,
    )
    with threadpool_limits(limits=1):
        if args.method == "qcbm":
            fit = fit_exact(spec, data.train, "fixed", args.seed, maxiter=2000)
        else:
            fit = fit_iqp_mc(
                spec,
                bit_table(6)[data.train_samples],
                "multiscale",
                args.seed,
                updates=1500,
                masks_per_step=64,
                samples_per_half=256,
                learning_rate=0.03,
            )
    probabilities = ExactCircuit(spec).probabilities(fit["parameters"])
    metrics = evaluate(probabilities, data.target, data.test, 6, data.name)
    circuit, parameters = to_qiskit(spec)
    bound = circuit.assign_parameters(
        dict(zip(parameters, fit["parameters"], strict=True))
    )
    bound.measure_all()
    sampled = (
        StatevectorSampler(seed=2026)
        .run([bound], shots=40)
        .result()[0]
        .data.meas.get_bitstrings()
    )
    outcomes = np.array([int(s, 2) for s in sampled])
    invalid = int(np.count_nonzero(data.target[outcomes] == 0))
    write_json(
        output / "demo.json",
        {
            "method": args.method,
            "seed": args.seed,
            "sampling_seed": 2026,
            "source": source_receipt(),
            "circuit": spec.to_dict(),
            "fit": fit,
            "metrics": metrics,
            "unfiltered_samples": sampled,
            "invalid_samples": invalid,
        },
    )
    (output / "circuit.qasm").write_text(qasm3.dumps(bound), encoding="utf-8")
    fig, axes = plt.subplots(4, 10, figsize=(11, 5))
    for ax, value in zip(axes.flat, outcomes, strict=True):
        ax.imshow(bit_table(6)[value].reshape(2, 3), cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#b34436" if data.target[value] == 0 else "#d4d9dd")
            spine.set_linewidth(2)
    fig.suptitle(
        f"{args.method.upper()} | 40 unfiltered Qiskit simulator samples\n{invalid} invalid patterns outlined in red"
    )
    fig.tight_layout()
    fig.savefig(output / "samples.png", dpi=150)
    plt.close(fig)
    print(
        f"TVD: {metrics['tvd']:.4f}; invalid probability: {metrics['invalid_mass']:.1%}"
    )
    print(
        f"Training: {fit['training_seconds']:.2f}s; quantum training shots: {fit['resources']['training_shots']}"
    )
    print(f"Saved circuit, parameters, metrics and sample image to {output.resolve()}")


if __name__ == "__main__":
    main()
