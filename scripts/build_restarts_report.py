"""Summarize every initialization policy and paired seed difference."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qgen.experiment import file_digest, write_json

CONDITIONS = {
    "adjacent_pairs": "Adjacent pairs",
    "shuffled_pairs": "Shuffled pairs",
    "noisy_pairs": "Noisy pairs",
    "parity3": "Three-bit parity",
    "parity4": "Four-bit parity",
}
LOSSES = {
    "multiscale": "Ordinary MMD",
    "pair": "Pair",
    "higher": "Higher order",
    "order_penalized": "Order penalty",
}
FAMILIES = {
    "uniform": "Uniform",
    "near_identity": "Near identity",
    "marginal_product": "Marginal product",
}


def build(series, destination):
    series, destination = Path(series), Path(destination)
    config = json.loads((series / "manifest.json").read_text())["config"]
    prefix = config["version"].removesuffix("-v1")
    seed_count = len(config["seeds"])
    index = json.loads((series / "index.json").read_text())
    audit = json.loads((series / "audit.json").read_text())
    if not audit["passed"] or audit["index_sha256"] != file_digest(
        series / "index.json"
    ):
        raise ValueError("Matching audit required")
    rows, candidates, classical = [], [], []
    groups = defaultdict(list)
    for name, checksum in sorted(index["completed"].items()):
        path = series / "runs" / f"{name}.json"
        if file_digest(path) != checksum:
            raise ValueError("Changed record")
        record = json.loads(path.read_text())
        identity = {key: record[key] for key in ["condition", "seed"]}
        for policy in record["policies"]:
            setting = {
                key: policy[key] for key in ["loss", "initialization", "allocation"]
            }
            for fit in policy["candidates"]:
                candidate = {
                    **identity,
                    **setting,
                    "candidate_index": fit["candidate_index"],
                    "selected": fit["candidate_index"] == policy["selected_candidate"],
                    **fit["metrics"],
                    "sample_metrics": fit["sample_metrics"],
                    "final_training_loss": fit["diagnostics"]["final"]["loss"],
                    "final_gradient_norm": fit["diagnostics"]["final"]["gradient_norm"],
                    "training_seconds": fit["training_seconds"],
                }
                candidates.append(candidate)
                if candidate["selected"]:
                    row = {
                        **candidate,
                        "resources": policy["resources"],
                        "policy_training_seconds": sum(
                            f["training_seconds"] for f in policy["candidates"]
                        ),
                    }
                    rows.append(row)
                    groups[
                        row["condition"],
                        row["loss"],
                        row["initialization"],
                        row["allocation"],
                    ].append(row)
        classical.extend(
            {
                **identity,
                "model": fit["model"],
                **fit["metrics"],
                "n_parameters": fit["n_parameters"],
                "training_seconds": fit["training_seconds"],
                "fit_info": fit.get("fit_info", {}),
            }
            for fit in record["classical"]
        )
    keys = ["condition", "seed", "loss", "initialization", "allocation"]
    lookup = {tuple(row[k] for k in keys): row for row in rows}
    differences = []
    for row in rows:
        c, seed, loss, family, allocation = (row[k] for k in keys)
        references = []
        if allocation == "restarts":
            references.append(
                ("restarts_minus_single", (c, seed, loss, family, "single"))
            )
        if family != "uniform":
            references.append(
                ("initialization_minus_uniform", (c, seed, loss, "uniform", allocation))
            )
        for effect, ref in references:
            other = lookup[ref]
            differences.append(
                {
                    "effect": effect,
                    **{key: row[key] for key in keys},
                    "tvd_difference": row["tvd"] - other["tvd"],
                    "training_loss_difference": row["final_training_loss"]
                    - other["final_training_loss"],
                }
            )
    means = {
        key: float(np.mean([r["tvd"] for r in group])) for key, group in groups.items()
    }
    destination.mkdir(parents=True, exist_ok=True)
    write_json(
        destination / f"{prefix}-summary.json",
        {
            "series": config["version"],
            "audit": audit,
            "selected_policy_rows": rows,
            "all_candidate_rows": candidates,
            "classical_rows": classical,
            "paired_differences": differences,
            "mean_tvd": [
                {
                    **dict(
                        zip(
                            ["condition", "loss", "initialization", "allocation"],
                            key,
                            strict=True,
                        )
                    ),
                    "tvd": value,
                }
                for key, value in means.items()
            ],
        },
    )
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
    fig, axes = plt.subplots(4, 3, figsize=(15, 13), sharex=True, sharey=True)
    for i, loss in enumerate(config["losses"]):
        for j, family in enumerate(config["initializations"]):
            ax = axes[i, j]
            for allocation, offset, color, label in [
                ("single", -0.12, "#245b85", "One 1,600-update run"),
                ("restarts", 0.12, "#bd592d", "Four 400-update runs"),
            ]:
                x = np.arange(5) + offset
                ax.scatter(
                    x,
                    [means[c, loss, family, allocation] for c in config["conditions"]],
                    color=color,
                    s=35,
                    marker="D",
                    label=label,
                    zorder=4,
                )
                for position, condition in enumerate(config["conditions"]):
                    points = sorted(
                        groups[condition, loss, family, allocation],
                        key=lambda row: row["seed"],
                    )
                    ax.scatter(
                        x[position] + np.linspace(-0.025, 0.025, len(points)),
                        [r["tvd"] for r in points],
                        color=color,
                        alpha=0.5,
                        s=13,
                        zorder=3,
                    )
            ax.set_title(f"{LOSSES[loss]} / {FAMILIES[family]}", loc="left")
            ax.set_ylim(-0.02, 0.8)
            ax.set_xticks(
                range(5), ["Adjacent", "Shuffled", "Noisy", "Parity 3", "Parity 4"]
            )
    for ax in axes[:, 0]:
        ax.set_ylabel("Total variation distance")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.text(
        0.05,
        0.015,
        f"Diamonds: {seed_count}-seed means. Small points: every seed. Restarts selected by training loss. Lower TVD is better.",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    (destination / "figures").mkdir(exist_ok=True)
    fig.savefig(destination / f"figures/{prefix}.png", dpi=180)
    plt.close(fig)
    lines = [
        "# Initialization and restarts: complete tables",
        "",
        f"Values are mean exact TVD over {seed_count} seeds: {', '.join(map(str, config['seeds']))}. Every policy uses 1,600 gradient updates. The [protocol]({prefix}-protocol.md) fixes the starts and training-only selection rule. The [summary JSON]({prefix}-summary.json) contains all {len(rows):,} selected policies, all {len(candidates):,} candidates, secondary metrics and every paired difference.",
        "",
        f"![All losses and initializations under single and restart allocations, including every seed.](figures/{prefix}.png)",
        "",
    ]
    for family in config["initializations"]:
        for allocation in config["allocations"]:
            lines += [
                f"## {FAMILIES[family]}, {allocation}",
                "",
                "| Condition | Ordinary MMD | Pair | Higher order | Order penalty |",
                "|---|---:|---:|---:|---:|",
            ]
            for condition in config["conditions"]:
                lines.append(
                    f"| {CONDITIONS[condition]} | "
                    + " | ".join(
                        f"{means[condition, loss, family, allocation]:.4f}"
                        for loss in config["losses"]
                    )
                    + " |"
                )
            lines.append("")
    lines += [
        "## Classical references",
        "",
        "These learned models use the same data; their capacities and optimizer costs are not matched to the circuit fits. The oracle has TVD zero and is not a learned model.",
        "",
        "| Condition | Empirical | Independent | Pairwise | Autoregressive | Higher-order selected | Order-4 | Order-penalized selected |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in config["conditions"]:
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
            f"| {CONDITIONS[condition]} | "
            + " | ".join(
                f"{np.mean([r['tvd'] for r in classical if r['condition'] == condition and r['model'] == model]):.4f}"
                for model in models
            )
            + " |"
        )
    for effect, title in [
        ("restarts_minus_single", "Restart minus single differences"),
        ("initialization_minus_uniform", "Initialization minus uniform differences"),
    ]:
        lines += [
            "",
            f"## {title}",
            "",
            "Negative values favor the first policy. Differences are paired on dataset seed, loss and the settings not being varied.",
            "",
            "| Condition | Loss | Initialization | Allocation | "
            + " | ".join(f"Seed {seed}" for seed in config["seeds"])
            + " |",
            "|---|---|---|---|" + "---:|" * seed_count,
        ]
        paired = defaultdict(dict)
        for row in differences:
            if row["effect"] == effect:
                paired[
                    row["condition"],
                    row["loss"],
                    row["initialization"],
                    row["allocation"],
                ][row["seed"]] = row["tvd_difference"]
        for (condition, loss, family, allocation), values in sorted(paired.items()):
            lines.append(
                f"| {CONDITIONS[condition]} | {LOSSES[loss]} | {FAMILIES[family]} | {allocation} | "
                + " | ".join(f"{values[seed]:+.4f}" for seed in config["seeds"])
                + " |"
            )
    verification = f"The audit checked {audit['policies_checked']} policies and {audit['candidates_checked']} candidates and replayed {audit['training_replays']} circuit fits. "
    if config.get("fresh_classical", False):
        verification += f"It checked {audit['fresh_classical_checked']} fresh classical/reference distributions and replayed {audit['fresh_classical_replays']} of them, including the known-target oracle references. "
    else:
        verification += (
            f"It reproduced {audit['published_fits_reproduced']} published fits. "
        )
    lines += [
        "",
        "## Verification",
        "",
        verification
        + "The single and restart policies use 6,561,792 and 6,586,368 phase evaluations including objective diagnostics. All candidates receive 2,048 evaluation samples; this evaluation work is separate from training.",
        "",
    ]
    (destination / f"{prefix}-tables.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", default="results/initialization-restarts-v1")
    parser.add_argument("--destination", default="docs")
    args = parser.parse_args()
    build(args.series, args.destination)
