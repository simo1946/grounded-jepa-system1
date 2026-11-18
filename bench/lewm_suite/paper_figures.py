"""The paper's figures.

    python paper_figures.py ../../results/paper/gpu_cube.json ../../paper/figures

Figure 1 (the system) is a diagram; its two camera frames are those of Figure 2. Figure 2 (the task) shows camera
frames from paper/figures/data: simulator states saved by figure_examples.py and figure_rollout.py, rendered at
384 pixels by figure_render.py (the model sees 64-pixel frames). Figures 3 to 5 are drawn from gpu_cube.json
(paper_values_gpu.py): bars are the mean over seeds, dots the individual seeds, error bars the 95% pair-bootstrap
interval.

Style: drawn at the size printed in the paper (6.4 or 4.0 inches wide) in a Times-compatible serif to match the
text; categorical colours in a fixed order (blue, orange, aqua; checked for colour-blind separation) with grey for
baselines; text in ink colours; 300 dpi.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

BLUE, ORANGE, AQUA, GREY, DARK = "#2a78d6", "#eb6834", "#1baf7a", "#a7b0ba", "#52514e"
INK, MUTED, GRID, FILL, HEAD = "#0b0b0b", "#52514e", "#e6e5e1", "#eef4fc", "#fdf3ec"
LABELS = {"C1": "C1 LeWM", "C2": "C2 +proprio", "C3": "C3 +object", "C5": "C5 hybrid\n+object"}
SHORT = {"C1": "C1\nLeWM", "C2": "C2\n+proprio", "C3": "C3\n+object", "C5": "C5 hybrid\n+object"}
DPI = 300
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "TeX Gyre Termes", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.linewidth": 0.6, "xtick.color": MUTED,
    "ytick.color": MUTED, "xtick.labelcolor": INK, "ytick.labelcolor": INK, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "svg.hashsalt": "paper", "savefig.dpi": DPI})


def _save(fig, out: Path) -> None:
    """Opaque RGB PNG: Word and LibreOffice tint transparent PNGs grey when they make the PDF."""
    from PIL import Image
    fig.savefig(out, dpi=DPI, metadata={"Software": None}, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    with Image.open(out) as im:
        rgb = im.convert("RGB")
    rgb.save(out, dpi=(DPI, DPI))


def _axes(ax, ylabel: str | None, ymax: float = 100, room: float = 0) -> None:
    """room: extra space above ymax for an in-panel legend (the axis and its ticks stop at ymax)."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, ymax + room)
    if room:
        ax.spines["left"].set_bounds(0, ymax)
        ax.set_yticks(np.linspace(0, ymax, 6))
    ax.tick_params(axis="x", length=0)
    if ylabel:
        ax.set_ylabel(ylabel)


def _bars(ax, success: dict, configs: list[str], series: list[tuple[str, str, str]], missing: str = "not run") -> None:
    """series: (evaluation key, legend label, colour). Bars side by side per configuration, white gaps between
    them; 95% interval as a thin ink line; seeds as dots spread across the bar so they do not hide each other."""
    n = len(series)
    width = 0.8 / n
    for j, (ev, label, colour) in enumerate(series):
        xs, ys, lo, hi = [], [], [], []
        for i, c in enumerate(configs):
            x = i + (j - (n - 1) / 2) * width
            d = success.get(c, {}).get(ev)
            if d is None:
                ax.text(x, 1.5, missing, rotation=90, ha="center", va="bottom", fontsize=6.5, color=MUTED)
                continue
            xs.append(x)
            ys.append(d["mean"])
            lo.append(d["mean"] - d["ci_lo"])
            hi.append(d["ci_hi"] - d["mean"])
            seeds = list(d["seeds"].values())
            dx = np.linspace(-0.22, 0.22, len(seeds)) * width if len(seeds) > 1 else [0.0]
            ax.scatter(x + np.asarray(dx), seeds, s=7, color="white", edgecolors=INK, linewidths=0.6, zorder=4)
        ax.bar(xs, ys, width=width, color=colour, label=label, edgecolor="white", linewidth=0.8, zorder=2)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.6, capthick=0.7,
                    zorder=3)
    ax.set_xticks(range(len(configs)), [LABELS.get(c, c) for c in configs])


def fig_success(success: dict, out: Path) -> None:
    configs = ["C1", "C2", "C3", "C5"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.75), sharey=True, gridspec_kw={"wspace": 0.08})
    _axes(a, "success (%)", room=22)
    _axes(b, None, room=22)
    _bars(a, success, configs, [("s1_nt", "System 1 (mean action)", BLUE),
                                ("sel16", "best-of-16 by world model", ORANGE),
                                ("cem", "CEM planning", AQUA)])
    _bars(b, success, configs, [("s1_50", "flat System 1", GREY), ("hier_50", "two-level System 1", BLUE)])
    a.set_title("(a) goal 25 steps ahead", color=INK, loc="left", pad=4)
    b.set_title("(b) goal 50 steps ahead", color=INK, loc="left", pad=4)
    for ax in (a, b):
        ax.legend(frameon=False, loc="upper left", ncols=2, handlelength=1.0, handletextpad=0.4,
                  columnspacing=0.9, borderaxespad=0.2, labelcolor=INK)
    _save(fig, out)


def fig_memory(success: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 2.55))
    _axes(ax, "success (%)")
    _bars(ax, success, ["C3", "C5"], [("rec1", "camera always on", DARK),
                                      ("cam0.5-hold", "50% lost: hold last frame", ORANGE),
                                      ("cam0.5-predict", "50% lost: world-model prediction", BLUE)])
    ax.set_xticks([0, 1], ["C3 +object", "C5 hybrid +object"])
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02), ncols=2, handlelength=1.0,
              columnspacing=0.8, handletextpad=0.4, borderaxespad=0, labelcolor=INK)
    _save(fig, out)


def fig_latent(diag: dict, out: Path, success_cm: float = 4.0) -> None:
    configs = ["C1", "C2", "C3", "C5"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.45), gridspec_kw={"wspace": 0.32})
    x, w = np.arange(len(configs)), 0.38
    real = [diag[c]["cube_probe_cm"] for c in configs]
    pred = [diag[c]["cube_probe_pred_cm"] for c in configs]
    _axes(a, "cube position error (cm)", ymax=11)
    a.bar(x - w / 2, real, w, color=BLUE, edgecolor="white", linewidth=0.8, label="from the real frame", zorder=2)
    a.bar(x + w / 2, pred, w, color=ORANGE, edgecolor="white", linewidth=0.8,
          label="from a latent predicted 5 blocks ahead", zorder=2)
    a.axhline(success_cm, color=INK, linewidth=0.8, linestyle=(0, (4, 2)), zorder=3)
    a.text(len(configs) - 0.45, success_cm + 0.15, "success radius, 4 cm", ha="right", va="bottom", fontsize=7,
           color=INK)
    a.set_xticks(x, [SHORT[c] for c in configs])
    a.set_title("(a) reading the cube's position (linear probe)", color=INK, loc="left", pad=4)
    a.legend(frameon=False, loc="upper left", ncols=1, handlelength=1.0, handletextpad=0.4, borderaxespad=0.2,
             labelcolor=INK)
    arm = [diag[c]["r2_arm"] for c in configs]
    cube = [diag[c]["r2_cube"] for c in configs]
    _axes(b, "R² of the latent goal distance", ymax=0.75)
    b.bar(x - w / 2, arm, w, color=DARK, edgecolor="white", linewidth=0.8, label="arm displacement alone", zorder=2)
    b.bar(x + w / 2, cube, w, color=AQUA, edgecolor="white", linewidth=0.8, label="cube displacement alone", zorder=2)
    for xi, v in zip(x + w / 2, cube, strict=True):
        b.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    b.set_xticks(x, [SHORT[c] for c in configs])
    b.set_title("(b) what the latent goal distance tracks", color=INK, loc="left", pad=4)
    b.legend(frameon=False, loc="upper left", ncols=1, handlelength=1.0, handletextpad=0.4, borderaxespad=0.2,
             labelcolor=INK)
    _save(fig, out)


def _frame(ax, img, title: str | None = None, edge: str = MUTED) -> None:
    ax.imshow(img, interpolation="antialiased")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(edge)
        s.set_linewidth(0.8)
    if title:
        ax.set_title(title, fontsize=7.5, color=INK, pad=2.5)


def fig_task(data: Path, out: Path, n_roll: int = 6) -> None:
    """(a) an already-solved pair, (b) a non-trivial pair, (c) System 1 solving pair (b). rollout_render.npz holds
    the frame after every step, so frames[i] is the frame after step i + 1; step 0 is the start frame."""
    triv, roll = np.load(data / "already_solved_render.npz"), np.load(data / "rollout_render.npz")
    tinfo = json.loads((data / "already_solved.json").read_text())
    rinfo = json.loads((data / "rollout.json").read_text())
    frames = [roll["start"], *roll["frames"]]
    steps = np.unique(np.linspace(0, len(frames) - 1, n_roll).round().astype(int))
    fig = plt.figure(figsize=(6.4, 3.55))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.42, 1.0], hspace=0.42)
    gt = outer[0].subgridspec(1, 5, width_ratios=[1, 1, 0.22, 1, 1], wspace=0.06)
    gs = outer[1].subgridspec(1, n_roll, wspace=0.07)
    top = [fig.add_subplot(gt[0, i]) for i in (0, 1, 3, 4)]
    _frame(top[0], triv["start"], "start")
    _frame(top[1], triv["goal"], f"goal ({tinfo['goal_offset']} steps later)")
    _frame(top[2], roll["start"], "start")
    _frame(top[3], roll["goal"], f"goal ({rinfo['goal_offset']} steps later)", edge=BLUE)
    for ax, title, note in ((top[0], "(a) already solved at the start", f"cube moved {100 * tinfo['cube_moved_m']:.0f} cm"),
                            (top[2], "(b) a pair that needs work", f"cube moved {100 * rinfo['cube_moved_m']:.0f} cm")):
        pos = ax.get_position()
        fig.text(pos.x0, pos.y1 + 0.06, title, fontsize=8.5, color=INK, ha="left")
        fig.text(pos.x0, pos.y0 - 0.012, note, fontsize=7, color=MUTED, ha="left", va="top")
    row = [fig.add_subplot(gs[0, i]) for i in range(len(steps))]
    for ax, t in zip(row, steps, strict=True):
        last = t == len(frames) - 1
        _frame(ax, frames[t], "start" if t == 0 else (f"step {t}: success" if last else f"step {t}"),
               edge=BLUE if last else MUTED)
    pos = row[0].get_position()
    fig.text(pos.x0, pos.y1 + 0.06, "(c) System 1 solving pair (b)", fontsize=8.5, color=INK, ha="left")
    _save(fig, out)


def _box(ax, xy, wh, text, *, face=FILL, edge=BLUE, dashed=False, fs=7.5):
    x, y = xy
    w, h = wh
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.06", facecolor=face,
                                edgecolor=edge, linewidth=0.9, linestyle=(0, (3, 2)) if dashed else "solid"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.25)
    return (x, y, w, h)


def _arrow(ax, p, q, *, color=INK, dashed=False, style="-|>", conn="arc3"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=7, color=color, linewidth=0.8,
                                 linestyle=(0, (3, 2)) if dashed else "solid", connectionstyle=conn,
                                 shrinkA=0, shrinkB=0))


def fig_system(data: Path, out: Path) -> None:
    roll = np.load(data / "rollout_render.npz")
    fig = plt.figure(figsize=(6.4, 2.9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 29)
    ax.axis("off")
    H = 4.6
    # frames
    for img, (x, y), label in ((roll["start"], (0.6, 15.8), "camera frame"), (roll["goal"], (0.6, 3.0), "goal frame")):
        ia = fig.add_axes([x / 64, y / 29, 6.0 / 64, 6.0 / 29 * (64 / 29) * (2.9 / 6.4)])
        _frame(ia, img)
        ax.text(x + 3.0, y - 0.9, label, ha="center", va="top", fontsize=7, color=MUTED)
    # band labels
    ax.text(0.6, 27.6, "World model", fontsize=8, color=INK, fontweight="bold", ha="left")
    ax.text(0.6, 11.4, "Control (test time)", fontsize=8, color=INK, fontweight="bold", ha="left")
    ax.plot([0.6, 63.4], [12.9, 12.9], color=GRID, linewidth=0.8)
    # top row
    yT = 16.4
    enc = _box(ax, (9.0, yT), (10.0, H), "ViT-tiny encoder\n(patch 8, 64 px)")
    zt = _box(ax, (22.5, yT), (7.0, H), "latent $z_t$\n192-d")
    pred = _box(ax, (33.0, yT), (14.0, H), "predictor\ntransformer or\nMamba-3/attention")
    znx = _box(ax, (50.5, yT), (7.5, H), "predicted\n$\\hat z_{t+1}$")
    heads = _box(ax, (9.0, 22.6), (20.5, 3.6), "training-only heads: inverse dynamics;\nstate alignment (robot, +cube)",
                 face=HEAD, edge=ORANGE, dashed=True, fs=7)
    ax.text(40.0, 23.4, "action block $a_t$ (5 steps)", ha="center", va="bottom", fontsize=7.5, color=INK)
    _arrow(ax, (40.0, 23.3), (40.0, yT + H))
    _arrow(ax, (6.8, yT + H / 2), (enc[0], yT + H / 2))
    _arrow(ax, (enc[0] + enc[2], yT + H / 2), (zt[0], yT + H / 2))
    _arrow(ax, (zt[0] + zt[2], yT + H / 2), (pred[0], yT + H / 2))
    _arrow(ax, (pred[0] + pred[2], yT + H / 2), (znx[0], yT + H / 2))
    _arrow(ax, (zt[0] + zt[2] / 2, yT + H), (zt[0] + zt[2] / 2, heads[1]), color=ORANGE, dashed=True)
    # bottom row
    yB = 3.6
    enc2 = _box(ax, (9.0, yB), (10.0, H), "same encoder\n$\\rightarrow z_{\\mathrm{goal}}$")
    sub = _box(ax, (22.5, yB), (11.0, H), "subgoal network\n(far goals only,\ngated)")
    s1 = _box(ax, (37.0, yB), (12.5, H), "System 1 policy\n$(z_t, z_{\\mathrm{goal}})\\rightarrow$\n25 actions")
    act = _box(ax, (52.5, yB), (10.9, H), "execute 5 blocks\n× 5 steps,\nthen look again", face="white",
               edge=MUTED)
    _arrow(ax, (6.8, yB + H / 2), (enc2[0], yB + H / 2))
    _arrow(ax, (enc2[0] + enc2[2], yB + H / 2), (sub[0], yB + H / 2))
    _arrow(ax, (sub[0] + sub[2], yB + H / 2), (s1[0], yB + H / 2))
    _arrow(ax, (s1[0] + s1[2], yB + H / 2), (act[0], yB + H / 2))
    # z_t feeds the subgoal network and System 1 (a bus below the world-model row)
    xz, ybus = zt[0] + zt[2] / 2, 10.4
    ax.plot([xz, xz], [yT, ybus], color=BLUE, linewidth=0.8)
    ax.plot([xz, s1[0] + s1[2] / 2], [ybus, ybus], color=BLUE, linewidth=0.8)
    _arrow(ax, (sub[0] + sub[2] / 2 + 2.0, ybus), (sub[0] + sub[2] / 2 + 2.0, yB + H), color=BLUE)
    _arrow(ax, (s1[0] + s1[2] / 2, ybus), (s1[0] + s1[2] / 2, yB + H), color=BLUE)
    ax.text(xz + 0.5, ybus + 0.4, "$z_t$", fontsize=7.5, color=BLUE, ha="left", va="bottom")
    _save(fig, out)


if __name__ == "__main__":
    values, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    gpu = json.loads(values.read_text())
    data = out_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    made = {"fig3_success.png": lambda p: fig_success(gpu["success"], p),
            "fig4_latent.png": lambda p: fig_latent(gpu["diagnostics"], p),
            "fig5_memory.png": lambda p: fig_memory(gpu["success"], p)}
    if (data / "rollout_render.npz").exists() and (data / "already_solved_render.npz").exists():
        made |= {"fig1_system.png": lambda p: fig_system(data, p), "fig2_task.png": lambda p: fig_task(data, p)}
    for name, draw in made.items():
        draw(out_dir / name)
    print(json.dumps({"figures": [str(out_dir / n) for n in made]}))
