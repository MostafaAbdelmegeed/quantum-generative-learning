"""Figures and tables from saved, audited experiment records."""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .audit import audit_experiment
from .data import bit_table
from .experiment import write_json

LABELS = {
    "bars_stripes": "Bars and stripes",
    "gaussian_mixture": "Gaussian mixture",
    "even_parity": "Even parity",
}


def build_report(paths, output):
    expected = {"reproduction-v1", "shots-v1", "connectivity-v1"}
    experiments = {
        json.loads((Path(p) / "manifest.json").read_text())["config"]["experiment"]
        for p in paths
    }
    if len(paths) != 3 or experiments != expected:
        raise ValueError(
            "This plot set requires the three v1 experiments. Use scripts/build_release_report.py for all series."
        )
    output = Path(output)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    records, audits = [], {}
    for path in map(Path, paths):
        audits[path.name] = audit_experiment(path)
        records.extend(
            json.loads(file.read_text())
            for file in sorted((path / "runs").glob("*.json"))
        )
    rows = []
    for record in records:
        job = record["job"]
        base = {"dataset": job["dataset"], "seed": job["seed"]}
        if job["kind"] == "classical":
            for fit in record["fits"]:
                rows.append(
                    {
                        **base,
                        "model": fit["model"],
                        "mode": "classical",
                        "shots": 0,
                        **fit["metrics"],
                    }
                )
        else:
            spec = record["circuit"]
            model = f"{spec['family']}/{spec['topology']}/{job['loss']}"
            rows.append(
                {
                    **base,
                    "model": model,
                    "mode": "shots" if job["shots"] else "exact",
                    "shots": job["shots"],
                    **record["fit"]["metrics"],
                }
            )
    write_json(output / "aggregate.json", {"audits": audits, "metrics": rows})
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "figure.dpi": 120,
        }
    )
    datasets = list(LABELS)
    classical = ["empirical", "independent", "pairwise", "autoregressive_mlp"]
    model_order = classical + ["qcbm/line/fixed", "qcbm/line/multiscale"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    for ax, dataset in zip(axes, datasets, strict=True):
        available = []
        for model in model_order:
            scores = [
                r["tvd"]
                for r in rows
                if r["dataset"] == dataset
                and r["model"] == model
                and r["mode"] != "shots"
            ]
            if not scores:
                continue
            x = len(available)
            available.append(
                model.replace("qcbm/line/", "QCBM ").replace(
                    "autoregressive_mlp", "AR MLP"
                )
            )
            ax.bar(
                x,
                np.mean(scores),
                color="#2c7580" if model in classical else "#e88a42",
                alpha=0.8,
            )
            ax.scatter(
                x + np.linspace(-0.12, 0.12, len(scores)),
                scores,
                color="#172b3a",
                s=18,
                zorder=4,
            )
        ax.set_xticks(range(len(available)), available, rotation=35, ha="right")
        ax.set_title(LABELS[dataset])
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Total variation distance (lower is better)")
    fig.suptitle("Six-qubit exact training: each dot is one seed")
    fig.tight_layout()
    fig.savefig(figures / "exact-comparison.png", bbox_inches="tight")
    plt.close(fig)

    selected = [
        r
        for r in records
        if r["job"]["kind"] == "quantum"
        and not r["job"]["shots"]
        and r["circuit"]["family"] == "qcbm"
        and r["job"]["loss"] == "multiscale"
        and r["job"]["seed"] == 42
    ]
    if selected:
        fig, axes = plt.subplots(len(selected), 2, figsize=(12, 3 * len(selected)))
        axes = np.atleast_2d(axes)
        for (left, right), record in zip(axes, selected, strict=True):
            p = np.array(record["fit"]["probabilities"])
            target = record["dataset"]["target"]
            left.plot(target, color="#172b3a", label="Target", lw=2)
            left.plot(p, color="#df7b32", label="Learned circuit", lw=1.4)
            left.set_title(LABELS[record["job"]["dataset"]])
            left.set_xlabel("Bitstring as an integer")
            left.set_ylabel("Probability")
            left.legend()
            history = record["fit"]["training_history"]
            right.plot(np.maximum(history, 1e-12), color="#2c7580")
            right.set_yscale("log")
            right.set_xlabel("Objective evaluation")
            right.set_ylabel("Training MMD squared")
        fig.suptitle("Representative seed 42, selected in advance")
        fig.tight_layout()
        fig.savefig(figures / "learned-distributions.png", bbox_inches="tight")
        plt.close(fig)

    shot_rows = [r for r in rows if r["mode"] == "shots"]
    if shot_rows:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for ax, dataset in zip(axes, datasets, strict=True):
            for loss, color in [("fixed", "#e88a42"), ("multiscale", "#2c7580")]:
                budgets = sorted({r["shots"] for r in shot_rows})
                means = []
                for shots in budgets:
                    values = [
                        r["tvd"]
                        for r in shot_rows
                        if r["dataset"] == dataset
                        and r["model"].endswith(loss)
                        and r["shots"] == shots
                    ]
                    means.append(np.mean(values))
                    ax.scatter(
                        [shots * 200] * len(values),
                        values,
                        color=color,
                        s=20,
                        alpha=0.5,
                    )
                ax.plot(
                    np.array(budgets) * 200, means, marker="o", color=color, label=loss
                )
            ax.set_xscale("log")
            ax.set_xlabel("Training shots (100 SPSA updates)")
            ax.set_title(LABELS[dataset])
            ax.set_ylim(0, 1)
        axes[0].set_ylabel("Final exact TVD (diagnostic)")
        axes[-1].legend()
        fig.suptitle("Finite-shot training with a fixed optimizer schedule")
        fig.tight_layout()
        fig.savefig(figures / "shot-training.png", bbox_inches="tight")
        plt.close(fig)

    deployed = [r for r in records if "deployments" in r]
    if deployed:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for ax, dataset in zip(axes, datasets, strict=True):
            for topology, color in [
                ("dense", "#e88a42"),
                ("line", "#2c7580"),
                ("tree", "#6b5ca5"),
            ]:
                groups = defaultdict(list)
                for record in deployed:
                    if (
                        record["job"]["dataset"] == dataset
                        and record["job"]["topology"] == topology
                    ):
                        for d in record["deployments"]:
                            groups[d["cx_depolarizing"]].append(d["metrics"]["tvd"])
                xs = sorted(groups)
                if xs:
                    mean = np.array([np.mean(groups[x]) for x in xs])
                    low = [min(groups[x]) for x in xs]
                    high = [max(groups[x]) for x in xs]
                    ax.plot(xs, mean, marker=".", label=topology, color=color)
                    ax.fill_between(xs, low, high, color=color, alpha=0.13)
            ax.set_title(LABELS[dataset])
            ax.set_xlabel("CX depolarizing parameter")
        axes[0].set_ylabel("TVD after routing and noise")
        axes[-1].legend()
        fig.suptitle("IQP connectivity study: mean and range across three seeds")
        fig.tight_layout()
        fig.savefig(figures / "connectivity-noise.png", bbox_inches="tight")
        plt.close(fig)

    bas = next((r for r in selected if r["job"]["dataset"] == "bars_stripes"), None)
    if bas:
        rng = np.random.default_rng(2026)
        p = np.array(bas["fit"]["probabilities"])
        outcomes = rng.choice(len(p), 40, p=p)
        target = np.array(bas["dataset"]["target"])
        fig, axes = plt.subplots(4, 10, figsize=(11, 5.2))
        for ax, outcome in zip(axes.flat, outcomes, strict=True):
            ax.imshow(
                bit_table(6)[outcome].reshape(2, 3),
                cmap="Blues",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#bd4b35" if target[outcome] == 0 else "#bbbbbb")
                spine.set_linewidth(2)
        invalid = sum(target[outcome] == 0 for outcome in outcomes)
        fig.suptitle(
            f"40 unfiltered samples from the learned circuit | {invalid} invalid patterns outlined in red\nModel seed 42; sampling seed 2026"
        )
        fig.tight_layout()
        fig.savefig(figures / "generated-patterns.png", bbox_inches="tight")
        plt.close(fig)
    print(f"Wrote audited aggregate and figures to {output}")
