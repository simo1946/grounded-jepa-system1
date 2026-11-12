"""Figures 2 and 3 of the paper, drawn from results/paper/gpu_cube.json (paper_values_gpu.py).

Bars are the mean over seeds; dots are the individual seeds. Colours follow a fixed categorical order
(blue, orange, aqua) with a neutral grey for baselines; text stays in ink colours.

    python paper_figures.py ../../results/paper/gpu_cube.json ../../paper/figures
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA, GREY, DARK = "#2a78d6", "#eb6834", "#1baf7a", "#a7b0ba", "#52514e"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
LABELS = {"C1": "C1 LeWM", "C2": "C2 +proprio", "C3": "C3 +object", "C5": "C5 hybrid\n+object"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "svg.hashsalt": "paper",
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "xtick.labelcolor": INK, "ytick.labelcolor": INK})


def _axes(ax, ylabel: bool) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel("success (%)")


def _bars(ax, success: dict, configs: list[str], series: list[tuple[str, str, str]]) -> None:
    """series: (evaluation key, legend label, colour); bars side by side per configuration, 2-pt gaps."""
    n, width = len(series), 0.8 / len(series)
    for j, (ev, label, colour) in enumerate(series):
        xs, ys = [], []
        for i, c in enumerate(configs):
            d = success.get(c, {}).get(ev)
            if d is None:
                continue
            x = i + (j - (n - 1) / 2) * width
            xs.append(x)
            ys.append(d["mean"])
            seeds = list(d["seeds"].values())
            ax.scatter([x] * len(seeds), seeds, s=10, color=INK, zorder=3, linewidths=0)
        ax.bar(xs, ys, width=width, color=colour, label=label, edgecolor="white", linewidth=1.0, zorder=2)
    ax.set_xticks(range(len(configs)), [LABELS[c] for c in configs])
    ax.tick_params(axis="x", length=0)
    ax.legend(frameon=False, loc="upper left", fontsize=8, labelcolor=INK)


def fig2(success: dict, out: Path) -> None:
    configs = ["C1", "C2", "C3", "C5"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    _axes(a, True)
    _axes(b, False)
    _bars(a, success, configs, [("s1_nt", "System 1 (mean action)", BLUE),
                                ("sel16", "best-of-16 by world model", ORANGE),
                                ("cem", "CEM planning", AQUA)])
    _bars(b, success, configs, [("s1_50", "flat System 1", GREY), ("hier_50", "two-level (subgoal network)", BLUE)])
    a.set_title("(a) goal 25 steps ahead", color=INK, fontsize=10)
    b.set_title("(b) goal 50 steps ahead", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200, metadata={"Software": None})
    plt.close(fig)


def fig3(success: dict, out: Path) -> None:
    configs = ["C3", "C5"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    _axes(ax, True)
    _bars(ax, success, configs, [("rec1", "camera always on", DARK),
                                 ("cam0.5-hold", "50% lost: hold last frame", ORANGE),
                                 ("cam0.5-predict", "50% lost: world-model prediction", BLUE)])
    ax.set_xticks(range(len(configs)), ["C3 +object", "C5 hybrid +object"])
    fig.tight_layout()
    fig.savefig(out, dpi=200, metadata={"Software": None})
    plt.close(fig)


if __name__ == "__main__":
    values, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    success = json.loads(values.read_text())["success"]
    out_dir.mkdir(parents=True, exist_ok=True)
    fig2(success, out_dir / "fig2_success.png")
    fig3(success, out_dir / "fig3_memory.png")
    print(json.dumps({"figures": [str(out_dir / "fig2_success.png"), str(out_dir / "fig3_memory.png")]}))
