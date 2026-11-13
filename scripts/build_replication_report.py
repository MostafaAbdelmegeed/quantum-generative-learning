"""Report the frozen primary comparisons on new dataset seeds."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from build_restarts_report import CONDITIONS, build

from qgen.experiment import write_json
from qgen.replication_analysis import paired_summary


def summarize(series, destination):
    series, destination = Path(series), Path(destination)
    config = json.loads((series / "manifest.json").read_text())["config"]
    if (
        config["version"] != "initialization-replication-v1"
        or len(config["seeds"]) != 10
    ):
        raise ValueError("This report requires the frozen ten-seed replication")
    build(series, destination)
    summary = json.loads(
        (destination / "initialization-replication-summary.json").read_text()
    )
    selected = {
        (r["condition"], r["seed"], r["initialization"]): r
        for r in summary["selected_policy_rows"]
        if r["loss"] == "order_penalized" and r["allocation"] == "single"
    }
    classical = {
        (r["condition"], r["seed"]): r
        for r in summary["classical_rows"]
        if r["model"] == "order_penalized_maxent"
    }
    primary, comparisons = [], []
    lines = [
        "# Replication: prespecified comparisons",
        "",
        "The primary comparison is near-identity minus uniform TVD under order-penalized weighting with one 1,600-update run. The frozen descriptive criterion requires a mean difference below -0.01 and improvements on at least eight of ten seeds. Negative differences favor near-identity starts.",
        "",
        "| Condition | Uniform mean | Near-identity mean | Mean difference | Median difference | Difference range | Improved / tied / worse | Criterion met |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for condition in config["conditions"]:
        near = np.array(
            [
                selected[condition, seed, "near_identity"]["tvd"]
                for seed in config["seeds"]
            ]
        )
        uniform = np.array(
            [selected[condition, seed, "uniform"]["tvd"] for seed in config["seeds"]]
        )
        control = np.array(
            [classical[condition, seed]["tvd"] for seed in config["seeds"]]
        )
        differences = near - uniform
        stats = paired_summary(differences)
        primary.append(
            {
                "condition": condition,
                "seeds": config["seeds"],
                "near_identity_tvd": near.tolist(),
                "uniform_tvd": uniform.tolist(),
                "differences": differences.tolist(),
                **stats,
            }
        )
        comparisons.append(
            {
                "condition": condition,
                "seeds": config["seeds"],
                "near_identity_tvd": near.tolist(),
                "classical_tvd": control.tolist(),
                "differences": (near - control).tolist(),
                "mean_difference": float(np.mean(near - control)),
                "minimum_difference": float(np.min(near - control)),
                "maximum_difference": float(np.max(near - control)),
                "improvements": int(np.count_nonzero(near - control < -1e-8)),
            }
        )
        lines.append(
            f"| {CONDITIONS[condition]} | {uniform.mean():.4f} | {near.mean():.4f} | {stats['mean']:+.4f} | {stats['median']:+.4f} | {stats['minimum']:+.4f} to {stats['maximum']:+.4f} | {stats['improvements']} / {stats['ties']} / {stats['regressions']} | {'Yes' if stats['benefit_replicated'] else 'No'} |"
        )
    lines += [
        "",
        "## Comparison with the selected classical model",
        "",
        "Both methods use the same order-penalized selected features and training data. Capacities and fitting costs are not matched. The two parity comparisons were declared separately in the protocol; all conditions are retained below.",
        "",
        "| Condition | Circuit mean | Classical mean | Mean circuit minus classical | Paired range | Circuit lower TVD |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for row in comparisons:
        lines.append(
            f"| {CONDITIONS[row['condition']]} | {np.mean(row['near_identity_tvd']):.4f} | {np.mean(row['classical_tvd']):.4f} | {row['mean_difference']:+.4f} | {row['minimum_difference']:+.4f} to {row['maximum_difference']:+.4f} | {row['improvements']}/10 |"
        )
    lines += [
        "",
        "## Every primary paired difference",
        "",
        "| Seed | Adjacent pairs | Shuffled pairs | Noisy pairs | Three-bit parity | Four-bit parity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for i, seed in enumerate(config["seeds"]):
        lines.append(
            f"| {seed} | "
            + " | ".join(f"{row['differences'][i]:+.4f}" for row in primary)
            + " |"
        )
    lines += [
        "",
        "The [complete policy tables](initialization-replication-tables.md) and [summary](initialization-replication-summary.json) retain every candidate and all secondary comparisons. The old three-seed discovery results are not pooled into these estimates.",
        "",
    ]
    (destination / "initialization-replication-primary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    write_json(
        destination / "initialization-replication-primary.json",
        {
            "primary_comparisons": primary,
            "selected_classical_comparisons": comparisons,
            "criterion": {
                "mean_difference_below": -0.01,
                "at_least_improvements": 8,
                "seeds": 10,
                "tie_threshold": 1e-8,
            },
        },
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    labels = ["Adjacent", "Shuffled", "Noisy", "Parity 3", "Parity 4"]
    for i, row in enumerate(primary):
        axes[0].scatter(
            i + np.linspace(-0.12, 0.12, 10),
            row["differences"],
            color="#245b85",
            alpha=0.6,
            s=22,
        )
        axes[0].scatter(i, row["mean"], color="#bd592d", marker="D", s=45, zorder=4)
    axes[0].axhline(0, color="#555555", linewidth=0.8)
    axes[0].set_ylabel("Near-identity minus uniform TVD")
    axes[0].set_title("Primary initialization comparison", loc="left")
    for j, field, label, color in [
        (0, "near_identity_tvd", "Circuit", "#245b85"),
        (1, "classical_tvd", "Selected classical", "#bd592d"),
    ]:
        for i, row in enumerate(comparisons):
            x = i + (j - 0.5) * 0.22
            axes[1].scatter(
                x + np.linspace(-0.04, 0.04, 10),
                row[field],
                color=color,
                alpha=0.6,
                s=18,
            )
            axes[1].scatter(
                x,
                np.mean(row[field]),
                color=color,
                marker="D",
                s=40,
                label=label if i == 0 else None,
                zorder=4,
            )
    axes[1].set_ylabel("Total variation distance")
    axes[1].set_ylim(bottom=0)
    axes[1].set_title("Near-identity circuit and classical control", loc="left")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xticks(range(5), labels)
    fig.text(
        0.06,
        0.015,
        "Ten new seeds; order-penalized weighting; single 1,600-update runs. Points: every seed. Diamonds: means.",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(destination / "figures/initialization-replication-primary.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", default="results/initialization-replication-v1")
    parser.add_argument("--destination", default="docs")
    args = parser.parse_args()
    summarize(args.series, args.destination)
