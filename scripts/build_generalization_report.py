"""Build the observable-comparison report from its verified archive."""

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qgen.experiment import file_digest, write_json

ROOT = Path(__file__).resolve().parents[1]
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
    index = json.loads((series / "index.json").read_text())
    audit = json.loads((series / "audit.json").read_text())
    if not audit["passed"] or audit["index_sha256"] != file_digest(
        series / "index.json"
    ):
        raise ValueError("A matching audit receipt is required")
    rows, failures, selections = [], [], []
    grouped = defaultdict(list)
    for name, checksum in sorted(index["completed"].items()):
        path = series / "runs" / f"{name}.json"
        if file_digest(path) != checksum:
            raise ValueError(f"Changed result: {name}")
        record = json.loads(path.read_text())
        train_bits = (np.asarray(record["data"]["train"])[:, None] >> np.arange(6)) & 1
        spins = 1 - 2 * train_bits
        covariance = spins.T @ spins / len(spins) - np.outer(
            spins.mean(axis=0), spins.mean(axis=0)
        )
        pairs = list(combinations(range(6), 2))
        record["selection"]["pair"].update(
            candidate_supports=[list(pair) for pair in pairs],
            scores=[float(abs(covariance[pair])) for pair in pairs],
            score_provenance="Recomputed from saved training samples for this report",
        )
        selections.append(
            {
                "condition": record["condition"],
                "seed": record["seed"],
                "selectors": record["selection"],
            }
        )
        for fit in record["fits"]:
            row = {
                "condition": record["condition"],
                "seed": record["seed"],
                "model": fit["model"],
                **fit["metrics"],
                "training_seconds": fit["training_seconds"],
            }
            rows.append(row)
            grouped[record["condition"], fit["model"]].append(row)
            info = fit.get("fit_info", {})
            if info.get("converged") is False or info.get("warnings"):
                failures.append(
                    {
                        "condition": record["condition"],
                        "seed": record["seed"],
                        "model": fit["model"],
                        **info,
                    }
                )
    summary = []
    for (condition, model), values in grouped.items():
        summary.append(
            {
                "condition": condition,
                "model": model,
                **{
                    key: float(np.mean([r[key] for r in values]))
                    for key in [
                        "tvd",
                        "js_nats",
                        "heldout_mmd",
                        "block_moment_mae",
                        "all_even_probability_error",
                        "training_seconds",
                    ]
                },
            }
        )
    paired = []
    lookup = {(r["condition"], r["model"], r["seed"]): r for r in rows}
    for condition in config["conditions"]:
        for estimator in config["estimators"]:
            for loss in ["pair", "higher"]:
                for seed in config["seeds"]:
                    paired.append(
                        {
                            "condition": condition,
                            "estimator": estimator,
                            "loss": loss,
                            "seed": seed,
                            "tvd_difference": lookup[
                                condition, f"iqp_{loss}_{estimator}", seed
                            ]["tvd"]
                            - lookup[condition, f"iqp_multiscale_{estimator}", seed][
                                "tvd"
                            ],
                        }
                    )
    destination.mkdir(parents=True, exist_ok=True)
    write_json(
        destination / "observable-generalization-summary.json",
        {
            "audit": audit,
            "config_sha256": manifest["config_sha256"],
            "rows": rows,
            "means": summary,
            "paired_tvd_differences": paired,
            "optimizer_failures": failures,
            "selections": selections,
        },
    )
    means = {(r["condition"], r["model"]): r for r in summary}
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.16,
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(11.6, 8), sharex=True, sharey=True)
    colors = ["#245b85", "#bd592d", "#647d33", "#222222"]
    labels = [
        "Ordinary MMD",
        "Pair weighting",
        "Higher-order weighting",
        "Classical selected features",
    ]
    for ax, estimator in zip(axes, config["estimators"], strict=True):
        models = [f"iqp_{loss}_{estimator}" for loss in config["losses"]] + [
            "selected_maxent"
        ]
        for i, (model, label, color) in enumerate(
            zip(models, labels, colors, strict=True)
        ):
            x = np.arange(len(config["conditions"])) + (i - 1.5) * 0.17
            y = [means[c, model]["tvd"] for c in config["conditions"]]
            ax.scatter(x, y, marker="D", color=color, s=42, label=label, zorder=4)
            for j, condition in enumerate(config["conditions"]):
                values = sorted(grouped[condition, model], key=lambda r: r["seed"])
                ax.scatter(
                    x[j] + np.linspace(-0.035, 0.035, len(values)),
                    [r["tvd"] for r in values],
                    color=color,
                    alpha=0.55,
                    s=15,
                    zorder=3,
                )
        ax.set_title(
            "Monte Carlo moments"
            if estimator == "mc"
            else "Enumerated moments; same sampled masks",
            loc="left",
        )
        ax.set_ylabel("Total variation distance")
        ax.set_ylim(-0.015, 0.8)
    axes[0].legend(ncol=2, loc="upper right", frameon=False)
    axes[1].set_xticks(
        np.arange(len(config["conditions"])), [LABELS[c] for c in config["conditions"]]
    )
    fig.suptitle(
        "Observable weighting did not reliably improve this bounded IQP comparison",
        x=0.07,
        ha="left",
        fontsize=14,
    )
    fig.text(
        0.07,
        0.015,
        "Diamonds: three-seed means. Small points: every seed. Lower is better. Six qubits, 400 Adam updates; no hardware runs.",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    (destination / "figures").mkdir(exist_ok=True)
    fig.savefig(destination / "figures/observable-generalization.png", dpi=180)
    plt.close(fig)

    lines = [
        "# Observable weighting: bounded generalization test",
        "",
        "11 November 2026. Five synthetic six-bit conditions, three seeds, 90 circuit fits and 90 learned classical fits, with 15 oracle references. All execution used local classical hardware and Qiskit simulation.",
        "",
        "Learned weighting did not reliably improve distributional fidelity in this comparison. Higher-order weighting worsened mean Monte Carlo TVD on both clean-pair conditions, noisy pairs and three-bit parity. On four-bit parity, its mean was close to ordinary MMD. The classical maximum-entropy model using the same selected features had lower mean TVD than every IQP variant in all five conditions.",
        "",
        "These results qualify the earlier 32-qubit adjacent-pair result. They do not refute that result: this experiment uses a dense six-qubit circuit, a shorter optimizer schedule, and a different higher-order selector. It establishes neither quantum advantage nor a general failure of observable weighting.",
        "",
        "![Mean and individual-seed TVD for ordinary, pair-weighted and higher-order IQP training, beside the classical model using the selected features.](figures/observable-generalization.png)",
        "",
        "## Exact distributional fidelity",
        "",
        "TVD ranges from zero for an exact match to one for disjoint support. Tables show three-seed means. All seeds, secondary metrics and signed paired differences are in the [machine-readable summary](observable-generalization-summary.json).",
        "",
        "| Condition | MC: ordinary | MC: pair | MC: higher order | Classical selected features |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in config["conditions"]:
        models = [
            "iqp_multiscale_mc",
            "iqp_pair_mc",
            "iqp_higher_mc",
            "selected_maxent",
        ]
        lines.append(
            f"| {LABELS[c]} | "
            + " | ".join(f"{means[c, m]['tvd']:.4f}" for m in models)
            + " |"
        )
    lines += [
        "",
        "## Moment-estimation control",
        "",
        "Both versions use identical initial angles, data, sampled masks and Adam schedules. Enumerating all 64 uniform strings removes moment-estimation noise; mask-sampling noise remains. The control costs half as many phase evaluations here. It is not a fully deterministic-gradient comparison.",
        "",
        "| Condition | Enumerated: ordinary | Enumerated: pair | Enumerated: higher order |",
        "|---|---:|---:|---:|",
    ]
    for c in config["conditions"]:
        lines.append(
            f"| {LABELS[c]} | "
            + " | ".join(
                f"{means[c, f'iqp_{loss}_enumerated']['tvd']:.4f}"
                for loss in config["losses"]
            )
            + " |"
        )
    lines += [
        "",
        "Removing moment noise substantially improves ordinary MMD on shuffled pairs, but worsens it on noisy pairs. Higher-order weighting still has poor clean-pair solutions with enumerated moments. Estimator noise therefore does not explain every poor fit, and this comparison does not establish a single cause of the optimizer failures.",
        "",
        "## Classical controls",
        "",
        "These models use the same training examples. Their capacities and optimization costs are not matched to IQP.",
        "",
        "| Condition | Empirical | Independent | Pairwise max-ent | Autoregressive MLP | Selected max-ent | Order-4 max-ent |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in config["conditions"]:
        models = [
            "empirical",
            "independent",
            "pairwise",
            "autoregressive_mlp",
            "selected_maxent",
            "order4_maxent",
        ]
        lines.append(
            f"| {LABELS[c]} | "
            + " | ".join(f"{means[c, m]['tvd']:.4f}" for m in models)
            + " |"
        )
    lines += [
        "",
        "The selected classical model has nine parameters: six single-bit moments and three selected moments. The full degree-four model has 56 parameters. The oracle uses the known target directly and has TVD zero; it is not a learned comparator.",
        "",
        "## What failed, and what the selector learned",
        "",
        "All 90 circuit fits completed their fixed 400-update budget. None claims optimizer convergence. All three adjacent-pair runs with higher-order weighting ended near TVD 0.5, under both estimators. Other losses also produced poor seed-specific fits, despite the existence of exact target parameters.",
        "",
        "The higher-order score is departure from independent marginals, not a cumulant. On clean pairs it often ranks four-bit products of pair constraints ahead of individual pairs. The GF(2) filter prevents algebraic redundancy but does not force the lowest-order representation. This changes the weighted objective and can change optimization behavior; these results do not isolate that mechanism causally.",
        "",
        "On noisy pairs, both selectors choose the same three pairs and yield identical circuit fits. On three- and four-bit parity, the higher-order selector recovers the true block constraints in all seeds. It fills the remaining feature slots with empirical dependencies, so recovering the true constraints alone does not guarantee a better fitted distribution.",
        "",
        f"The archive retains {len(failures)} classical optimizer-limit failures: every noisy-pair fit of the 56-parameter maximum-entropy model reached 300 iterations. The smaller selected model and the remaining classical optimizers reported no convergence warnings. There were no execution exceptions.",
        "",
        "Noisy pairs require special care: their correct all-even probability is 0.729. The report evaluates TVD against the corrupted target and saves absolute all-even probability error and each block moment. A clean-pair validity score would prefer the wrong target.",
        "",
        "## Paired seed differences",
        "",
        "Each entry is weighted-loss TVD minus ordinary-MMD TVD for the same condition, seed and estimator. Negative values favor weighting. Seed order is 42, 137, 2026.",
        "",
        "| Condition | Estimator | Pair weighting differences | Higher-order differences |",
        "|---|---|---|---|",
    ]
    for c in config["conditions"]:
        for estimator in config["estimators"]:
            values = []
            for loss in ["pair", "higher"]:
                values.append(
                    ", ".join(
                        f"{next(r['tvd_difference'] for r in paired if r['condition'] == c and r['estimator'] == estimator and r['loss'] == loss and r['seed'] == seed):+.4f}"
                        for seed in config["seeds"]
                    )
                )
            lines.append(f"| {LABELS[c]} | {estimator} | {values[0]} | {values[1]} |")
    lines += [
        "",
        "## Budget, checks and reproduction",
        "",
        "The [frozen protocol](observable-generalization-protocol.md) fixes 512 training and 1,024 held-out examples, 21 circuit parameters, 64 masks per update and 400 Adam updates. Each Monte Carlo fit uses two independent 64-string halves: 3,276,800 phase evaluations. Each enumerated-moment control uses 1,638,400. Feature selection adds 7,680 pair character evaluations or 25,600 higher-order character evaluations per dataset. Runtime and all candidate scores are saved separately.",
        "",
        "The expressibility construction sets star-edge angles within each block to pi/4, with a single-X rotation for the corrupted parity sector. Both NumPy and Qiskit match each known target to numerical precision. Dense connectivity supports every column permutation. Adjacent and shuffled pairs share latent samples, but initial angles are not permuted between conditions; cross-condition differences therefore cannot be assigned solely to column order.",
        "",
        f"The audit checked {audit['fits_checked']} fitted/reference distributions and sampling records and replayed {audit['adam_replays']} Adam fits. Numerical tests cover target construction, data permutation, selector independence, gradient agreement, shared stochastic masks, classical parity expressibility and rejection of a corrupted metric even after its checksum is updated.",
        "",
        "A separate replay loaded the archived package source and repeated every circuit fit, classical fit and sampling call. All 15 complete records matched exactly except elapsed-time fields; the [comparison receipt](verification/observable-generalization-v1-replay.json) records the result. Exact equality was checked in the recorded Windows environment, not across platforms.",
        "",
        "The new archive is [observable-generalization-v1](../results/observable-generalization-v1/). The seven published baseline snapshots are unchanged. The original explorer and release report remain scoped to those seven series; this follow-up has its own report and figure.",
        "",
        "```bash",
        "uv run --locked --extra dev python -m pytest",
        "uv run --locked python -m qgen.generalization --config config/observable-generalization-v1.json --output artifacts/observable-generalization-new",
        "uv run --locked python -m qgen.generalization_audit --output artifacts/observable-generalization-new --replay",
        "uv run --locked python scripts/replay_generalization.py --snapshot results/observable-generalization-v1 --output artifacts/observable-generalization-replay-new",
        "uv run --locked python scripts/build_generalization_report.py --series results/observable-generalization-v1 --destination docs",
        "```",
        "",
        "The runner refuses an existing output directory. It saves the frozen protocol, lockfile, configuration, source snapshot, environment, optimizer histories, sampled counts and checksums. The sampled circuit outputs come from Qiskit Statevector sampling; no physical device was used.",
        "",
        "A useful next experiment would separate selector design from optimization sensitivity, using a new frozen protocol. The present results do not justify scaling this higher-order selector to larger widths.",
        "",
    ]
    (destination / "observable-generalization-results.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", default="results/observable-generalization-v1")
    parser.add_argument("--destination", default="docs")
    args = parser.parse_args()
    build(args.series, args.destination)
