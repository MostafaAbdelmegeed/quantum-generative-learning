"""Create the release figures and machine-readable summary from audited runs."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qgen.audit import audit_experiment
from qgen.experiment import write_json
from qgen.scaling import audit_scaling

ROOT = Path(__file__).resolve().parents[1]
BASELINE_SERIES = {
    "reproduction-v1",
    "reproduction-v2",
    "shots-v1",
    "connectivity-v1",
    "iqp-mc-v1",
    "scaling-v1",
    "scaling-pairs-v1",
}
LABELS = {
    "bars_stripes": "Bars and stripes",
    "gaussian_mixture": "Gaussian mixture",
    "even_parity": "Even parity",
}


def main():
    records, audits, rows = {}, {}, []
    for folder in sorted((ROOT / "results").iterdir()):
        if folder.name not in BASELINE_SERIES:
            continue
        if not (folder / "manifest.json").exists():
            continue
        manifest = json.loads((folder / "manifest.json").read_text())
        mode = manifest["config"]["mode"]
        audits[folder.name] = (
            audit_scaling(folder) if mode == "scaling" else audit_experiment(folder)
        )
        series = [
            json.loads(f.read_text()) for f in sorted((folder / "runs").glob("*.json"))
        ]
        records[folder.name] = series
        if mode == "scaling":
            continue
        for record in series:
            base = {
                "series": folder.name,
                "dataset": record["job"]["dataset"],
                "seed": record["job"]["seed"],
            }
            if record["job"]["kind"] == "classical":
                rows.extend(
                    {
                        **base,
                        "model": fit["model"],
                        "mode": "classical",
                        **fit["metrics"],
                    }
                    for fit in record["fits"]
                )
            else:
                rows.append(
                    {
                        **base,
                        "model": record["circuit"]["family"],
                        "mode": mode,
                        "topology": record["job"]["topology"],
                        "loss": record["job"]["loss"],
                        "shots_per_call": record["job"]["shots"],
                        **record["fit"]["metrics"],
                    }
                )
    group = defaultdict(list)
    for row in rows:
        group[
            (
                row["series"],
                row["dataset"],
                row["model"],
                row.get("topology"),
                row.get("loss"),
                row.get("shots_per_call"),
            )
        ].append(row["tvd"])
    summary = [
        {
            "series": k[0],
            "dataset": k[1],
            "model": k[2],
            "topology": k[3],
            "loss": k[4],
            "shots_per_call": k[5],
            "n": len(v),
            "tvd_mean": float(np.mean(v)),
            "tvd_min": min(v),
            "tvd_max": max(v),
        }
        for k, v in group.items()
    ]
    write_json(
        ROOT / "docs" / "release-summary.json",
        {"audits": audits, "metrics": rows, "groups": summary},
    )
    figures = ROOT / "docs" / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "figure.dpi": 140,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, dataset in zip(axes, LABELS, strict=True):
        for loss, color in [("fixed", "#da7d35"), ("multiscale", "#277b83")]:
            means = []
            for budget, series in [(250, "reproduction-v1"), (2000, "reproduction-v2")]:
                values = [
                    r["tvd"]
                    for r in rows
                    if r["series"] == series
                    and r["dataset"] == dataset
                    and r.get("loss") == loss
                ]
                means.append(np.mean(values))
                ax.scatter([budget] * len(values), values, s=20, color=color, alpha=0.5)
            ax.plot([250, 2000], means, marker="o", color=color, label=loss)
        ax.set_title(LABELS[dataset])
        ax.set_xticks([250, 2000])
        ax.set_xlabel("L-BFGS-B iteration cap")
        ax.set_ylim(0, 0.55)
    axes[0].set_ylabel("TVD (lower is better)")
    axes[-1].legend()
    fig.suptitle("Optimization budget changes the bars-and-stripes conclusion")
    fig.tight_layout()
    fig.savefig(figures / "budget-followup.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, dataset in zip(axes, LABELS, strict=True):
        for offset, series, color, label in [
            (-0.17, "connectivity-v1", "#da7d35", "Exact L-BFGS-B"),
            (0.17, "iqp-mc-v1", "#277b83", "Classical MC + Adam"),
        ]:
            for x, topology in enumerate(["dense", "line", "tree"]):
                values = [
                    r["tvd"]
                    for r in rows
                    if r["series"] == series
                    and r["dataset"] == dataset
                    and r.get("topology") == topology
                ]
                ax.bar(
                    x + offset,
                    np.mean(values),
                    0.32,
                    color=color,
                    alpha=0.8,
                    label=label if x == 0 else None,
                )
                ax.scatter(
                    x + offset + np.linspace(-0.06, 0.06, len(values)),
                    values,
                    s=18,
                    color="#20303b",
                    zorder=3,
                )
        ax.set_title(LABELS[dataset])
        ax.set_xticks(range(3), ["dense", "line", "tree"])
        ax.set_ylim(0, 0.9)
    axes[0].set_ylabel("Final exact TVD (diagnostic)")
    axes[-1].legend(fontsize=8)
    fig.suptitle(
        "IQP training: same data, circuits and initial angles; different optimizers"
    )
    fig.tight_layout()
    fig.savefig(figures / "iqp-estimator-comparison.png", bbox_inches="tight")
    plt.close(fig)

    scale = records.get("scaling-v1", [])
    if scale:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        for model, label, color in [
            ("initial_iqp", "Initial circuit", "#a8afb4"),
            ("iqp", "Trained IQP", "#277b83"),
            ("independent", "Independent bits", "#9a5f83"),
            ("paired_classical", "Classical pair model", "#da7d35"),
        ]:
            widths = sorted({r["n"] for r in scale})
            for ax, metric in zip(
                axes[:2], ["valid_fraction", "pair_agreement"], strict=True
            ):
                means = [
                    np.mean([r["metrics"][model][metric] for r in scale if r["n"] == n])
                    for n in widths
                ]
                ax.plot(widths, means, marker="o", color=color, label=label)
                for n in widths:
                    values = [r["metrics"][model][metric] for r in scale if r["n"] == n]
                    ax.scatter([n] * len(values), values, color=color, alpha=0.4, s=16)
                ax.set_xticks(widths)
                ax.set_xlabel("Qubits")
                ax.set_ylim(-0.025, 1.04)
        for metric, color, label in [
            ("training_seconds", "#277b83", "Classical training"),
            ("seconds", "#da7d35", "Qiskit sampling"),
        ]:
            ys = [
                np.mean(
                    [
                        r["fit" if metric == "training_seconds" else "sampling"][metric]
                        for r in scale
                        if r["n"] == n
                    ]
                )
                for n in widths
            ]
            axes[2].plot(widths, ys, marker="o", label=label, color=color)
        axes[0].set_ylabel("Fraction of entirely valid samples")
        axes[1].set_ylabel("Mean adjacent-pair agreement")
        axes[0].legend(fontsize=8, loc="center right")
        axes[2].set_xticks(widths)
        axes[2].set_xlabel("Qubits")
        axes[2].set_ylabel("Elapsed seconds (mean of three runs)")
        axes[2].legend(fontsize=8)
        fig.suptitle(
            "8 to 32 qubits: an easy classical target checks the complete workflow"
        )
        fig.tight_layout()
        fig.savefig(figures / "scaling-workflow.png", bbox_inches="tight")
        plt.close(fig)
    priority = records.get("scaling-pairs-v1", [])
    if scale and priority:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for series, label, color in [
            (scale, "Ordinary multiscale MMD", "#da7d35"),
            (priority, "50% learned pair observables", "#277b83"),
        ]:
            for ax, metric in zip(
                axes, ["valid_fraction", "heldout_mmd_u"], strict=True
            ):
                means = [
                    np.mean(
                        [r["metrics"]["iqp"][metric] for r in series if r["n"] == n]
                    )
                    for n in widths
                ]
                ax.plot(widths, means, marker="o", label=label, color=color)
                for n in widths:
                    values = [
                        r["metrics"]["iqp"][metric] for r in series if r["n"] == n
                    ]
                    ax.scatter([n] * len(values), values, color=color, alpha=0.5, s=20)
                ax.set_xticks(widths)
                ax.set_xlabel("Qubits")
        axes[0].axhline(1, color="#6a7479", ls="--", lw=1, label="Classical pair model")
        axes[0].set_ylim(0.75, 1.015)
        axes[0].set_ylabel("Entirely valid samples (higher is better)")
        axes[0].legend(fontsize=8, loc="lower left")
        axes[1].set_ylabel("Held-out multiscale MMD (same evaluation kernel)")
        axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        fig.suptitle(
            "Same training budget: stronger sample validity with learned pair weighting"
        )
        fig.tight_layout()
        fig.savefig(figures / "pair-priority-followup.png", bbox_inches="tight")
        plt.close(fig)
    print(
        f"Audited {sum(a['jobs'] for a in audits.values())} records across {len(audits)} experiment series"
    )


if __name__ == "__main__":
    main()
