"""Summarize the frozen selector, budget and estimator factorial comparison."""

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
    manifest = json.loads((series / "manifest.json").read_text())
    config = manifest["config"]
    audit = json.loads((series / "audit.json").read_text())
    index = json.loads((series / "index.json").read_text())
    if not audit["passed"] or audit["index_sha256"] != file_digest(
        series / "index.json"
    ):
        raise ValueError("Matching audit required")
    rows, classical, selections = [], [], []
    groups = defaultdict(list)
    for name, checksum in sorted(index["completed"].items()):
        path = series / "runs" / f"{name}.json"
        if file_digest(path) != checksum:
            raise ValueError("Changed record")
        record = json.loads(path.read_text())
        selections.append(
            {
                "condition": record["condition"],
                "seed": record["seed"],
                "selection": record["selection"],
            }
        )
        for fit in record["fits"]:
            row = {
                "condition": record["condition"],
                "seed": record["seed"],
                "loss": fit["loss"],
                "estimator": fit["estimator"],
                "updates": fit["iterations"],
                **fit["metrics"],
                "final_gradient_norm": fit["diagnostics"]["final"]["gradient_norm"],
                "final_training_loss": fit["diagnostics"]["final"]["loss"],
                "training_seconds": fit["training_seconds"],
            }
            rows.append(row)
            groups[
                row["condition"], row["loss"], row["estimator"], row["updates"]
            ].append(row)
        for fit in record["classical"]:
            classical.append(
                {
                    "condition": record["condition"],
                    "seed": record["seed"],
                    "model": fit["model"],
                    **fit["metrics"],
                    "fit_info": fit.get("fit_info", {}),
                }
            )
    lookup = {
        (r["condition"], r["seed"], r["loss"], r["estimator"], r["updates"]): r
        for r in rows
    }
    differences = []
    for row in rows:
        c, s, loss, est, budget = (
            row[k] for k in ["condition", "seed", "loss", "estimator", "updates"]
        )
        for effect, reference in [
            (
                "order_penalty_minus_higher",
                (c, s, "higher", est, budget) if loss == "order_penalized" else None,
            ),
            ("long_minus_short", (c, s, loss, est, 400) if budget == 1600 else None),
            (
                "full_minus_enumerated",
                (c, s, loss, "enumerated", budget) if est == "full" else None,
            ),
        ]:
            if reference is not None:
                differences.append(
                    {
                        "effect": effect,
                        "condition": c,
                        "seed": s,
                        "loss": loss,
                        "estimator": est,
                        "updates": budget,
                        "tvd_difference": row["tvd"] - lookup[reference]["tvd"],
                    }
                )
    means = {
        key: float(np.mean([r["tvd"] for r in values]))
        for key, values in groups.items()
    }
    destination.mkdir(parents=True, exist_ok=True)
    write_json(
        destination / "observable-diagnostics-summary.json",
        {
            "audit": audit,
            "config_sha256": manifest["config_sha256"],
            "rows": rows,
            "classical": classical,
            "paired_differences": differences,
            "selection": selections,
            "means": [
                {
                    "condition": k[0],
                    "loss": k[1],
                    "estimator": k[2],
                    "updates": k[3],
                    "tvd": v,
                }
                for k, v in means.items()
            ],
        },
    )
    colors = ["#245b85", "#bd592d", "#647d33", "#76528b"]
    labels = ["Ordinary MMD", "Pair weighting", "Higher order", "Order penalty"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.15,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True, sharex=True)
    for i, budget in enumerate(config["update_budgets"]):
        for j, estimator in enumerate(config["estimators"]):
            ax = axes[i, j]
            for k, (loss, label, color) in enumerate(
                zip(config["losses"], labels, colors, strict=True)
            ):
                x = np.arange(5) + (k - 1.5) * 0.17
                ax.scatter(
                    x,
                    [means[c, loss, estimator, budget] for c in config["conditions"]],
                    color=color,
                    s=30,
                    marker="D",
                    label=label,
                    zorder=4,
                )
                for position, c in enumerate(config["conditions"]):
                    ax.scatter(
                        x[position] + np.linspace(-0.025, 0.025, 3),
                        [
                            r["tvd"]
                            for r in sorted(
                                groups[c, loss, estimator, budget],
                                key=lambda r: r["seed"],
                            )
                        ],
                        color=color,
                        alpha=0.45,
                        s=10,
                        zorder=3,
                    )
            ax.set_title(f"{estimator}; {budget:,} updates", loc="left")
            ax.set_ylim(-0.02, 0.8)
            ax.set_xticks(
                range(5), ["Adjacent", "Shuffled", "Noisy", "Parity 3", "Parity 4"]
            )
    for ax in axes[:, 0]:
        ax.set_ylabel("Total variation distance")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.text(
        0.05,
        0.015,
        "Diamonds: three-seed means; small points: every seed. Lower is better. Exact target distributions; six-qubit CPU simulation.",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    (destination / "figures").mkdir(exist_ok=True)
    fig.savefig(destination / "figures/observable-diagnostics.png", dpi=180)
    plt.close(fig)
    lines = [
        "# Observable diagnostics: complete tables",
        "",
        "This supplement reports every configuration from the [frozen protocol](observable-diagnostics-protocol.md). Values are three-seed mean TVD. The [summary JSON](observable-diagnostics-summary.json) retains every seed, secondary metric, final gradient norm and paired difference.",
        "",
        "![All four losses at two budgets under three estimators, showing every seed.](figures/observable-diagnostics.png)",
        "",
    ]
    for estimator in config["estimators"]:
        for budget in config["update_budgets"]:
            lines += [
                f"## {estimator}, {budget:,} updates",
                "",
                "| Condition | Ordinary MMD | Pair | Higher order | Order penalty |",
                "|---|---:|---:|---:|---:|",
            ]
            for c in config["conditions"]:
                lines.append(
                    f"| {LABELS[c]} | "
                    + " | ".join(
                        f"{means[c, loss, estimator, budget]:.4f}"
                        for loss in config["losses"]
                    )
                    + " |"
                )
            lines.append("")
    lines += [
        "## Learned classical references",
        "",
        "The seven learned references use the same data. Optimizer costs and capacities are not matched to IQP. The oracle, omitted here, has TVD zero.",
        "",
        "| Condition | Empirical | Independent | Pairwise | Autoregressive | Higher-order selected | Order-4 | Order-penalized selected |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in config["conditions"]:
        models = [
            "empirical",
            "independent",
            "pairwise",
            "autoregressive_mlp",
            "selected_maxent",
            "order4_maxent",
            "order_penalized_maxent",
        ]
        lines.append(
            f"| {LABELS[c]} | "
            + " | ".join(
                f"{np.mean([r['tvd'] for r in classical if r['condition'] == c and r['model'] == m]):.4f}"
                for m in models
            )
            + " |"
        )
    lines += [
        "",
        "## Verification",
        "",
        f"The audit checked {audit['fits_checked']} circuit/classical/reference distributions, reproduced {audit['published_short_fits_reproduced']} published short-budget fits exactly apart from elapsed times, and replayed {audit['adam_replays']} new-series Adam fits. No fit was selected by evaluation performance.",
        "",
    ]
    (destination / "observable-diagnostics-tables.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", default="results/observable-diagnostics-v1")
    parser.add_argument("--destination", default="docs")
    args = parser.parse_args()
    build(args.series, args.destination)
