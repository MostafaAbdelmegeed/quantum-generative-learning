"""Draw the IQP training/sampling diagram with archived example outputs."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
INK = "#213d4a"
TEAL = "#197c80"
MUTED = "#52666e"


def main():
    record = json.loads(
        (ROOT / "results/scaling-pairs-v1/runs/n8-s42.json").read_text()
    )
    samples = record["data"]["samples"]["iqp"][:4]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.color": INK,
            "svg.hashsalt": "qgen-method-overview",
        }
    )
    fig, ax = plt.subplots(figsize=(13, 4.15))
    fig.patch.set_facecolor("#f5f8f8")
    ax.set(xlim=(0, 13), ylim=(0, 4.15))
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    ax.text(0.4, 3.68, "Train on a CPU", fontsize=17, weight="bold")
    ax.text(0.4, 3.28, "Estimate output correlations", color=MUTED)
    ax.text(
        0.4,
        2.57,
        r"$m_a(\theta)=\mathbb{E}_{x\sim p_\theta}[(-1)^{a\cdot x}]$",
        fontsize=15,
    )
    ax.text(
        0.4,
        1.98,
        "Fourier masks + random bitstrings\nIndependent-pair loss estimate\nAdam parameter updates",
        linespacing=1.65,
        va="top",
    )
    ax.text(0.4, 0.6, "No quantum training shots", color=TEAL, weight="bold")

    ax.text(4.5, 3.68, "Export the circuit", fontsize=17, weight="bold")
    ax.text(4.5, 3.28, "Equivalent IQP circuit (schematic)", color=MUTED)
    for y in [2.62, 2.11, 1.60, 1.09]:
        ax.plot([4.72, 8.32], [y, y], color=MUTED, lw=1.1, zorder=1)
        ax.text(4.46, y, r"$|0\rangle$", va="center", ha="center", fontsize=12)
        for x in [5.02, 7.65]:
            ax.add_patch(
                Rectangle((x, y - 0.17), 0.37, 0.34, fc="white", ec=TEAL, lw=1.4)
            )
            ax.text(x + 0.185, y, "H", ha="center", va="center", fontsize=11)
    ax.add_patch(Rectangle((5.77, 0.82), 1.47, 2.07, fc="#e0eeee", ec=TEAL, lw=1.5))
    ax.text(6.505, 1.98, r"$D(\theta)$", fontsize=23, ha="center")
    ax.text(6.505, 1.50, "Commuting\nZ rotations", ha="center", linespacing=1.4)
    ax.text(4.5, 0.6, "Qiskit circuit + saved OpenQASM", color=TEAL, weight="bold")

    ax.text(9.2, 3.68, "Inspect samples", fontsize=17, weight="bold")
    ax.text(9.2, 3.28, "Recorded 8-qubit outputs", color=MUTED)
    for i, bits in enumerate(samples):
        grouped = "  ".join(bits[j : j + 2] for j in range(0, len(bits), 2))
        ax.text(9.35, 2.55 - i * 0.49, grouped, family="DejaVu Sans Mono", fontsize=16)
    ax.text(9.2, 0.6, "Validity, bias and held-out MMD", color=TEAL, weight="bold")
    for x1, x2 in [(3.68, 4.18), (8.53, 9.02)]:
        ax.add_patch(
            FancyArrowPatch(
                (x1, 2.05),
                (x2, 2.05),
                arrowstyle="-|>",
                mutation_scale=16,
                lw=1.6,
                color=TEAL,
            )
        )
    ax.text(
        0.4,
        0.14,
        "Training uses classical estimates. This archive evaluates circuits with local Qiskit simulators.",
        fontsize=10,
        color=MUTED,
    )
    output = ROOT / "docs/figures"
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "method-overview.svg", metadata={"Date": None})
    fig.savefig(output / "method-overview.png", dpi=180)
    plt.close(fig)
    print("Wrote docs/figures/method-overview.svg and .png")


if __name__ == "__main__":
    main()
