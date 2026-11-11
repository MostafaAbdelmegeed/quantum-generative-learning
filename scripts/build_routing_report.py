"""Report ideal fits separately from routing cost and implementation noise."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qgen.experiment import file_digest, write_json

LABELS = {
    "adjacent_pairs": "Adjacent pairs",
    "shuffled_pairs": "Shuffled pairs",
    "noisy_pairs": "Noisy pairs",
    "parity3": "Three-bit parity",
    "parity4": "Four-bit parity",
}


def build(series, destination):
    series, destination = Path(series), Path(destination)
    config = json.loads((series / "manifest.json").read_text())["config"]
    index = json.loads((series / "index.json").read_text())
    audit = json.loads((series / "audit.json").read_text())
    if not audit["passed"] or audit["index_sha256"] != file_digest(
        series / "index.json"
    ):
        raise ValueError("Matching audit required")
    rows, ideals, classical = [], [], []
    for name, checksum in sorted(index["completed"].items()):
        path = series / "runs" / f"{name}.json"
        if file_digest(path) != checksum:
            raise ValueError("Changed result")
        record = json.loads(path.read_text())
        for fit in record["classical"]:
            classical.append(
                {
                    "condition": record["condition"],
                    "seed": record["seed"],
                    "model": fit["model"],
                    **fit["metrics"],
                }
            )
        for fit in record["fits"]:
            base = {
                "condition": record["condition"],
                "seed": record["seed"],
                "topology": fit["topology"],
            }
            ideals.append(
                {
                    **base,
                    **fit["metrics"],
                    "final_gradient_norm": fit["diagnostics"]["final"]["gradient_norm"],
                }
            )
            for deployed in fit["deployments"]:
                compilation = deployed["compilation"]
                for outcome in deployed["outcomes"]:
                    rows.append(
                        {
                            **base,
                            "device": compilation["device"],
                            "cx_count": compilation["cx_count"],
                            "depth": compilation["depth"],
                            "error": outcome["error"],
                            **outcome["metrics"],
                            "ideal_target_tvd": fit["metrics"]["tvd"],
                            "target_tvd_change": outcome["target_tvd_change"],
                            "tvd_from_ideal_output": outcome["tvd_from_ideal_output"],
                            "noiseless_max_probability_error": outcome[
                                "noiseless_max_probability_error"
                            ],
                        }
                    )
    write_json(
        destination / "routing-controls-summary.json",
        {"audit": audit, "ideal": ideals, "classical": classical, "deployments": rows},
    )
    groups = defaultdict(list)
    for row in rows:
        groups[row["condition"], row["topology"], row["device"], row["error"]].append(
            row
        )

    def mean(c, topology, device, error, metric):
        return float(np.mean([r[metric] for r in groups[c, topology, device, error]]))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.15,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.3))
    colors = ["#245b85", "#bd592d"]
    for i, (topology, color) in enumerate(
        zip(config["topologies"], colors, strict=True)
    ):
        x = np.arange(5) + (i - 0.5) * 0.2
        values = [
            np.mean(
                [
                    r["tvd"]
                    for r in ideals
                    if r["condition"] == c and r["topology"] == topology
                ]
            )
            for c in config["conditions"]
        ]
        axes[0].scatter(
            x, values, marker="D", color=color, label=topology.replace("_", " "), s=42
        )
        for j, c in enumerate(config["conditions"]):
            subset = [
                r for r in ideals if r["condition"] == c and r["topology"] == topology
            ]
            axes[0].scatter(
                x[j] + np.linspace(-0.03, 0.03, 3),
                [r["tvd"] for r in subset],
                color=color,
                s=14,
                alpha=0.5,
            )
        for device, marker in zip(config["devices"], ["o", "^", "s"], strict=True):
            subset = [
                r
                for r in rows
                if r["topology"] == topology
                and r["device"] == device
                and r["error"] == 0.02
            ]
            axes[1].scatter(
                [r["cx_count"] for r in subset],
                [r["tvd_from_ideal_output"] for r in subset],
                color=color,
                marker=marker,
                s=13,
                alpha=0.15,
            )
            axes[1].scatter(
                np.mean([r["cx_count"] for r in subset]),
                np.mean([r["tvd_from_ideal_output"] for r in subset]),
                color=color,
                marker=marker,
                s=75,
                label=f"{topology.split('_')[0]} / {device}",
            )
    classical_means = [
        np.mean(
            [
                r["tvd"]
                for r in classical
                if r["condition"] == c and r["model"] == "selected_maxent"
            ]
        )
        for c in config["conditions"]
    ]
    axes[0].scatter(
        np.arange(5) + 0.3,
        classical_means,
        color="#222222",
        marker="x",
        s=40,
        label="Classical selected features",
    )
    axes[0].set_xticks(
        range(5), ["Adjacent", "Shuffled", "Noisy", "Parity 3", "Parity 4"]
    )
    axes[0].set_ylabel("TVD to target before deployment")
    axes[0].set_ylim(-0.02, 0.8)
    axes[0].set_title(
        "Ideal fits: two expressible 11-parameter families", loc="left", fontsize=11
    )
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].set_xlabel("Routed CX count")
    axes[1].set_ylabel("TVD from the ideal circuit output")
    axes[1].set_title("Implementation change at CX error 0.02", loc="left", fontsize=11)
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    fig.text(
        0.06,
        0.015,
        "Left: three seeds per condition. Right: large markers average 15 condition/seed fits; faint points show every fit. Simulated noise only.",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    (destination / "figures").mkdir(exist_ok=True)
    fig.savefig(destination / "figures/routing-controls.png", dpi=180)
    plt.close(fig)
    lines = [
        "# Routing controls: complete tables",
        "",
        "Every configuration from the [frozen protocol](routing-controls-protocol.md). Values are three-seed means; the [summary JSON](routing-controls-summary.json) includes every seed and secondary metric.",
        "",
        "![Ideal target fit and routed CX cost versus the noise-induced output change.](figures/routing-controls.png)",
        "",
        "| Condition | Circuit | Device | CX gates | Ideal target TVD | Target TVD at 0.005 | Target TVD at 0.02 | Output change at 0.02 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for c in config["conditions"]:
        for topology in config["topologies"]:
            for device in config["devices"]:
                values = [
                    mean(c, topology, device, 0, "cx_count"),
                    mean(c, topology, device, 0, "ideal_target_tvd"),
                    mean(c, topology, device, 0.005, "tvd"),
                    mean(c, topology, device, 0.02, "tvd"),
                    mean(c, topology, device, 0.02, "tvd_from_ideal_output"),
                ]
                lines.append(
                    f"| {LABELS[c]} | {topology} | {device} | {values[0]:.1f} | "
                    + " | ".join(f"{v:.4f}" for v in values[1:])
                    + " |"
                )
    lines += [
        "",
        "Output change is TVD between noisy and ideal model distributions. It differs from the change in TVD to the target. Noise can reduce target error for a poorly fitted circuit without making its implementation more faithful.",
        "",
        f"The audit replayed {audit['training_fits_replayed']} training fits and {audit['noise_evaluations_replayed']} compiled noise evaluations, including sample counts. Maximum noiseless probability discrepancy was {max(r['noiseless_max_probability_error'] or 0 for r in rows):.3g}; the final protocol's numerical bound is 1e-8.",
        "",
    ]
    (destination / "routing-controls-tables.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", default="results/routing-controls-v1")
    parser.add_argument("--destination", default="docs")
    args = parser.parse_args()
    build(args.series, args.destination)
