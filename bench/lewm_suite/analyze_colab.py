"""Aggregate the GPU study (colab_plan.py): per-config success over seeds,
paired contrasts with exact McNemar tests (pooled over seeds, Holm-corrected),
diagnostics, and figures.

    python analyze_colab.py --profile standard        # writes results/colab_summary.md, CSVs, figures
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from cluster_bootstrap import cluster_bootstrap

HERE = Path(__file__).resolve().parent
N_BOOT = 10_000

NAMES = {
    "C1": "LeWM", "C2": "+ proprio grounding", "C3": "+ object grounding", "C4": "hybrid + proprio",
    "C5": "hybrid + object", "C6": "+ object + ARL", "C7": "hybrid + object + ARL", "C8": "IDM only",
    "C9": "+ object, MobileNetV4-S",
}
EVAL_ORDER = ["s1_all", "cem_all", "s1_nt", "s1_pk", "s1f_nt", "s1f_pk", "s1_50", "hier_nt", "hier_pk", "hier_50", "sel16", "sel16n", "gate", "cem", "rec1",
              "cam0.5-blank", "cam0.5-hold", "cam0.5-predict", "cam0.8-hold", "cam0.8-predict"]
EVAL_TITLES = {
    "s1_all": "System 1, all pairs (authors' protocol)", "cem_all": "CEM planner, all pairs (authors' protocol)",
    "s1_nt": "System 1, non-trivial +25", "s1_pk": "System 1, grasp windows",
    "s1f_nt": "flow System 1, non-trivial +25", "s1f_pk": "flow System 1, grasp windows", "s1_50": "System 1, non-trivial +50",
    "hier_nt": "two-level, non-trivial +25", "hier_pk": "two-level, grasp windows", "hier_50": "two-level, +50",
    "sel16": "best-of-16 by world model", "sel16n": "best-of-16, candidates near the mean (std x 0.25)", "gate": "verify-then-act (CEM fallback)", "cem": "CEM planner",
    "rec1": "System 1, replan every block", "cam0.5-blank": "camera 50% off, blank",
    "cam0.5-hold": "camera 50% off, hold last", "cam0.5-predict": "camera 50% off, world-model prediction",
    "cam0.8-hold": "camera 80% off, hold last", "cam0.8-predict": "camera 80% off, world-model prediction",
}


def mcnemar_exact(a, b) -> dict:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    n = n10 + n01
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p": 1.0}
    from scipy.stats import binomtest                     # exact; stable for large discordant counts
    p = float(binomtest(min(n10, n01), n, 0.5).pvalue)
    return {"a_only": n10, "b_only": n01, "p": p}


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def label_of(r: dict) -> str | None:
    c, pf = r.get("controller", "cem"), r.get("policy_file") or "policy.pt"
    off, pairs = r["goal_offset"], r.get("pairs", "all")
    suffix = {("nontrivial", 25): "nt", ("pickup", 20): "pk", ("nontrivial", 50): "50", ("all", 25): "all"}.get((pairs, off))
    if suffix is None or (r.get("cost") or "mse") != "mse":
        return None
    if c == "cem":
        return {"nt": "cem", "all": "cem_all"}.get(suffix)
    if c == "s1s2":
        return "gate" if suffix == "nt" and r.get("k") == 16 else None
    if r.get("k") == 16:
        if suffix != "nt" or r.get("select", "cost") != "cost":
            return None
        std = 1.0 if r.get("std_scale") is None else r["std_scale"]
        return {1.0: "sel16", 0.25: "sel16n"}.get(std)
    if r.get("receding", 5) == 1:
        if suffix != "nt" or pf != "policy_s1.pt":
            return None
        return "rec1" if not r.get("camera_drop") else f"cam{r['camera_drop']:g}-{r.get('fill')}"
    base = {"policy_s1.pt": "s1", "policy_hier_gated.pt": "hier", "policy_flow.pt": "s1f"}.get(pf)
    return f"{base}_{suffix}" if base else None


def load(home: Path, num_eval: int, configs: dict, task: str = "cube", results_dir: Path | None = None) -> dict:
    """{(cfg_id, seed, label): successes array} for one environment (result files start with the task name).
    Result files are read from ``results_dir`` (default: home/results)."""
    by_spec = {v: k for k, v in configs.items()}
    out = {}
    for f in sorted((results_dir or home / "results").glob(f"{task}__*.json")):
        r = json.loads(f.read_text())
        if r.get("num_eval") != num_eval or r["spec"] not in by_spec:
            continue
        lab = label_of(r)
        if lab:
            key = (by_spec[r["spec"]], r["seed"], lab)
            if key in out:
                raise ValueError(f"two result files for {key}: {f.name} and {META[key]['file']}")
            out[key] = np.asarray(r["successes"], bool)
            META[key] = {"file": f.name, "eval_seed": r.get("eval_seed"), "goal_offset": r.get("goal_offset"),
                         "budget": r.get("eval_budget"), "pairs": r.get("pairs"),
                         "ids": (tuple(r["episodes_idx"]), tuple(r["start_steps"])) if r.get("episodes_idx") else None}
    return out


META: dict = {}     # (cfg, seed, label) -> provenance of the loaded result file


def _check_same_pairs(ka, kb) -> None:
    """Paired tests need identical start/goal pairs in the same order: check the evaluation seed and, when
    the result files record them, the exact pair identities."""
    a, b = META.get(ka), META.get(kb)
    if a is None or b is None:
        return
    if a["eval_seed"] != b["eval_seed"]:
        raise ValueError(f"different evaluation seeds: {a['file']} vs {b['file']}")
    if a["ids"] is not None and b["ids"] is not None and a["ids"] != b["ids"]:
        raise ValueError(f"different start/goal pairs: {a['file']} vs {b['file']}")


def table(res: dict, cfgs: list[str]) -> tuple[list[dict], str]:
    evs = [e for e in EVAL_ORDER if any(k[2] == e for k in res)]          # only columns with results
    rows, md = [], ["| config | " + " | ".join(EVAL_TITLES[e] for e in evs) + " |",
                    "|---" * (len(evs) + 1) + "|"]
    for c in cfgs:
        cells = []
        for e in evs:
            vals = [100 * v.mean() for (cc, s, ee), v in sorted(res.items()) if cc == c and ee == e]
            if not vals:
                cells.append("")
                continue
            m = float(np.mean(vals))
            se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else float("nan")
            rows.append({"config": c, "name": NAMES[c], "eval": e, "mean": m, "se": se, "seeds": len(vals)})
            cells.append(f"{m:.0f}" + (f" ± {se:.0f}" if len(vals) > 1 else "") + f" ({len(vals)})")
        md.append(f"| {c} {NAMES[c]} | " + " | ".join(cells) + " |")
    return rows, "\n".join(md)


PRIMARY = {  # pre-registered family (notebook: H1, H4, H5, H6 and the ARL effect); Holm-corrected together
    ("object grounding", "s1_nt"), ("object grounding", "s1_pk"),
    ("ARL (transformer)", "s1_nt"), ("ARL (hybrid)", "s1_nt"),
    ("hybrid predictor (object)", "s1_nt"),
    ("camera 0.5 + prediction: hybrid vs transformer (object)", "cam0.5"),
    ("all best parts vs LeWM", "s1_nt"), ("all best parts vs LeWM", "s1_pk"),
    ("all configs: two-level vs flat, +50", "hier_50"), ("all configs: two-level vs flat, grasp", "hier_pk"),
}


def contrasts_list() -> list[tuple[str, str, tuple, tuple]]:
    """(description, eval, (config(s), eval A), (config(s), eval B)); a config entry may be '*' = every
    config that has both evaluations (pairs pooled over configs and seeds)."""
    L = []
    for e in ("s1_nt", "s1_pk"):
        L += [("proprio grounding", e, ("C2", e), ("C1", e)),
              ("object grounding", e, ("C3", e), ("C2", e)),
              ("hybrid predictor (proprio)", e, ("C4", e), ("C2", e)),
              ("hybrid predictor (object)", e, ("C5", e), ("C3", e)),
              ("ARL (transformer)", e, ("C6", e), ("C3", e)),
              ("ARL (hybrid)", e, ("C7", e), ("C5", e)),
              ("all best parts vs LeWM", e, ("C7", e), ("C1", e)),
              ("IDM only vs IDM + proprio", e, ("C8", e), ("C2", e)),
              ("MobileNetV4-S vs ViT-tiny encoder", e, ("C9", e), ("C3", e))]
    L += [("all configs: best-of-16 near the mean vs mean (exploratory)", "sel16n", ("*", "sel16n"), ("*", "s1_nt"))]
    L += [("all configs: flow vs Gaussian System 1, +25", "s1f_nt", ("*", "s1f_nt"), ("*", "s1_nt")),
          ("all configs: flow vs Gaussian System 1, grasp", "s1f_pk", ("*", "s1f_pk"), ("*", "s1_pk"))]
    L += [("all configs: two-level vs flat, +50", "hier_50", ("*", "hier_50"), ("*", "s1_50")),
          ("all configs: two-level vs flat, grasp", "hier_pk", ("*", "hier_pk"), ("*", "s1_pk")),
          ("all configs: two-level vs flat, +25", "hier_nt", ("*", "hier_nt"), ("*", "s1_nt"))]
    for c in ("C1", "C2", "C3", "C4", "C5", "C6", "C7"):
        L += [(f"{c} best-of-16 vs mean", "sel16", (c, "sel16"), (c, "s1_nt")),
              (f"{c} verify-then-act vs mean", "gate", (c, "gate"), (c, "s1_nt")),
              (f"{c} two-level vs flat, +50", "hier_50", (c, "hier_50"), (c, "s1_50")),
              (f"{c} two-level vs flat, grasp", "hier_pk", (c, "hier_pk"), (c, "s1_pk"))]
    for c in ("C1", "C3", "C6", "C7"):
        L.append((f"{c} System 1 vs CEM", "cem", (c, "s1_nt"), (c, "cem")))
        L.append((f"{c} System 1 vs CEM, all pairs", "cem_all", (c, "s1_all"), (c, "cem_all")))
    L += [(f"{c} {NAMES[c]} vs LeWM, all pairs", "s1_all", (c, "s1_all"), ("C1", "s1_all")) for c in ("C2", "C3", "C5")]
    for c in ("C3", "C5", "C6", "C7"):
        for p in ("0.5", "0.8"):
            L.append((f"{c} camera {p}: prediction vs hold", f"cam{p}", (c, f"cam{p}-predict"), (c, f"cam{p}-hold")))
    for p in ("0.5", "0.8"):
        L += [(f"camera {p} + prediction: hybrid vs transformer (object)", f"cam{p}",
               ("C5", f"cam{p}-predict"), ("C3", f"cam{p}-predict")),
              (f"camera {p} + prediction: hybrid vs transformer (object + ARL)", f"cam{p}",
               ("C7", f"cam{p}-predict"), ("C6", f"cam{p}-predict"))]
    return L


def _paired(res: dict, A: tuple, B: tuple):
    """Matched (A, B) success arrays and per-unit differences, units = (config, seed)."""
    (ca, ea), (cb, eb) = A, B
    if ca == "*":
        units = sorted({(c, s) for (c, s, e) in res if e == ea and (c, s, eb) in res})
        keys = [((c, s, ea), (c, s, eb)) for c, s in units]
    else:
        seeds = sorted({s for (c, s, e) in res if c == ca and e == ea} & {s for (c, s, e) in res if c == cb and e == eb})
        keys = [((ca, s, ea), (cb, s, eb)) for s in seeds]
    if not keys:
        return None
    for ka, kb in keys:
        if len(res[ka]) != len(res[kb]):
            raise ValueError(f"unequal numbers of pairs: {ka} vs {kb}")
        _check_same_pairs(ka, kb)
    a = np.concatenate([res[ka] for ka, _ in keys])
    b = np.concatenate([res[kb] for _, kb in keys])
    d = [100 * (res[ka].mean() - res[kb].mean()) for ka, kb in keys]
    stacked = (np.stack([res[ka] for ka, _ in keys]), np.stack([res[kb] for _, kb in keys]))
    return a, b, d, stacked


def contrasts(res: dict) -> tuple[list[dict], str]:
    out = []
    for desc, ev, A, B in contrasts_list():
        m = _paired(res, A, B)
        if m is None:
            continue
        a, b, d, (sa, sb) = m
        t = mcnemar_exact(a, b)
        # the same pairs are shared by every unit: resample pairs (all units together), see cluster_bootstrap.py
        bs = cluster_bootstrap(sa, sb, None, N_BOOT)
        out.append({"contrast": desc, "eval": ev, "a": f"{A[0]}:{A[1]}", "b": f"{B[0]}:{B[1]}", "units": len(d),
                    "primary": (desc, ev) in PRIMARY, "diff_pts": float(np.mean(d)),
                    "diff_se": float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan"),
                    "a_pct": 100 * a.mean(), "b_pct": 100 * b.mean(), **t,
                    "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"], "p_boot": bs["p_boot"], "clusters": bs["clusters"]})
    did = verification_did_rows(res)
    out += did
    prim = [r for r in out if r["primary"]]
    for r, ph in zip(prim, holm([r["p"] for r in prim])):
        r["p_holm"] = ph
    for r, ph in zip(prim, holm([r.get("p_boot", r["p"]) for r in prim])):
        r["p_boot_holm"] = ph
    md = ["| contrast | eval | A | B | units | A − B (pts) | 95% CI (pair bootstrap) | A-only | B-only | p | "
          "p (Holm, primary) | p boot (Holm, primary) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(out, key=lambda r: not r["primary"]):
        se = f" ± {r['diff_se']:.1f}" if r["units"] > 1 else ""
        star = "**" if r["primary"] else ""
        ph = f"{r['p_holm']:.3g}" if "p_holm" in r else "—"
        a_pct = f"{r['a_pct']:.0f}%" if r.get("a_pct") is not None else ""
        b_pct = f"{r['b_pct']:.0f}%" if r.get("b_pct") is not None else ""
        ci = f"[{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]" if "ci_lo" in r else "—"
        pbh = f"{r['p_boot_holm']:.3g}" if "p_boot_holm" in r and "p_boot" in r else "—"
        md.append(f"| {star}{r['contrast']}{star} | {r['eval']} | {a_pct} | {b_pct} | {r['units']} | "
                  f"{r['diff_pts']:+.1f}{se} | {ci} | {r.get('a_only', '')} | {r.get('b_only', '')} | {r['p']:.3g} | "
                  f"{ph} | {pbh} |")
    md.append("")
    md.append("Bold rows are the pre-registered family (Holm-corrected together); the rest are exploratory. "
              "Units = (config, seed) groups of paired start/goal pairs. McNemar p pools units as if independent "
              "(descriptive); the bootstrap resamples the shared pairs with all units together (10,000 draws).")
    return out, "\n".join(md)


def verification_did_rows(res: dict) -> list[dict]:
    """H3: does ARL raise (best-of-16 - mean)? Per start/goal pair (same pairs across configs), the
    selection gain is in {-1, 0, 1}; the ARL effect is gain(ARL) - gain(base), tested with a
    Wilcoxon signed-rank test over all pairs and seeds."""
    from scipy.stats import wilcoxon
    rows = []
    for base, arl in (("C3", "C6"), ("C5", "C7")):
        seeds = [s for s in sorted({s for (c, s, e) in res}) if all((c, s, e) in res for c in (base, arl)
                                                                   for e in ("sel16", "s1_nt"))]
        if not seeds:
            continue
        g = lambda c, s: res[(c, s, "sel16")].astype(int) - res[(c, s, "s1_nt")].astype(int)  # noqa: E731
        diff = np.concatenate([g(arl, s) - g(base, s) for s in seeds])
        per_seed = [100 * (g(arl, s).mean() - g(base, s).mean()) for s in seeds]
        p = float(wilcoxon(diff, zero_method="wilcox").pvalue) if np.any(diff != 0) else 1.0
        rows.append({"contrast": f"ARL makes selection useful ({base} → {arl})", "eval": "sel16 − s1_nt",
                     "a": f"{arl}:gain", "b": f"{base}:gain", "units": len(seeds), "primary": True,
                     "diff_pts": float(np.mean(per_seed)),
                     "diff_se": float(np.std(per_seed, ddof=1) / np.sqrt(len(per_seed))) if len(seeds) > 1 else float("nan"),
                     "a_pct": None, "b_pct": None, "p": p})
    return rows


def diagnostics(home: Path, configs: dict, results_dir: Path | None = None) -> tuple[list[dict], str]:
    from models import Spec
    rows = []
    rdir = results_dir or home / "results"
    for c, spec in configs.items():
        name = Spec.parse(spec).name()
        rp = rdir / f"rollout_diag__{name}__seed0.json"
        pp = rdir / f"probe_diag__{name}__seed0.json"
        if not (rp.exists() and pp.exists()):
            continue
        r, p = json.loads(rp.read_text()), json.loads(pp.read_text())
        fine = r.get("perturbed_identifiability", {})
        rows.append({"config": c, "coarse_top1": r["true_chunk_top1_at_k5"],
                     "fine_top1_s0.1": fine.get("sigma=0.1", {}).get("true_top1"),
                     "fine_top1_s0.3": fine.get("sigma=0.3", {}).get("true_top1"),
                     "rollout_err_over_copy_k5": r["pred_err_over_copy_by_k"][-1],
                     "cube_probe_cm": 100 * p["cube"]["probe_err_real_frames_m"],
                     "cube_probe_pred_cm": 100 * p["cube"]["probe_err_predicted_k5_m"],
                     "r2_arm": p["latent_dist_vs_displacements"]["r2_arm_only"],
                     "r2_cube": p["latent_dist_vs_displacements"]["r2_cube_only"]})
    md = ["| config | true chunk vs 15 real (top-1) | vs perturbed σ=0.1 | σ=0.3 | rollout error / copy | "
          "cube probe (cm) | cube probe, predicted (cm) | R² arm | R² cube |", "|---" * 9 + "|"]
    for d in rows:
        f = lambda x, s="{:.2f}": "" if x is None else s.format(x)  # noqa: E731
        md.append(f"| {d['config']} {NAMES[d['config']]} | {f(d['coarse_top1'])} | {f(d['fine_top1_s0.1'])} | "
                  f"{f(d['fine_top1_s0.3'])} | {f(d['rollout_err_over_copy_k5'], '{:.3f}')} | "
                  f"{f(d['cube_probe_cm'], '{:.1f}')} | {f(d['cube_probe_pred_cm'], '{:.1f}')} | "
                  f"{f(d['r2_arm'])} | {f(d['r2_cube'], '{:.3f}')} |")
    return rows, "\n".join(md)


# ---- figures (matplotlib; one colour per series, recessive grid, no dual axes)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # reference categorical slots 1-4, fixed order
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_success(rows: list[dict], cfgs: list[str], path: Path) -> None:
    """Dot + whisker (mean ± SE over seeds) per config: System 1 on two protocols, two panels."""
    import matplotlib.pyplot as plt
    evs = [("s1_nt", "Non-trivial pairs, goal +25"), ("s1_pk", "Grasp-centred windows (H-JEPA protocol)")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.42 * len(cfgs) + 1.4), sharey=True)
    for ax, (e, title) in zip(axes, evs):
        _style(ax)
        for i, c in enumerate(cfgs):
            r = next((x for x in rows if x["config"] == c and x["eval"] == e), None)
            if r is None:
                continue
            se = 0 if np.isnan(r["se"]) else r["se"]
            ax.errorbar(r["mean"], i, xerr=se, fmt="o", color=SERIES[0], ms=6, elinewidth=2, capsize=0)
            ax.text(r["mean"] + se + 3, i, f"{r['mean']:.0f}%", va="center", fontsize=8, color=INK)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlim(0, 105)
        ax.set_xlabel("success (%), mean ± SE over seeds", fontsize=9, color=MUTED)
    axes[0].set_yticks(range(len(cfgs)), [f"{c}  {NAMES[c]}" for c in cfgs])
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_verification(res: dict, cfgs: list[str], path: Path) -> None:
    """Signed change vs System 1 mean, bars about zero: selection (slot 1) and verify-then-act (slot 2)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(cfgs) + 1.4))
    _style(ax)
    h = 0.36
    for j, (e, lab) in enumerate((("sel16", "best-of-16 by world model"), ("gate", "verify-then-act"))):
        for i, c in enumerate(cfgs):
            seeds = [s for (cc, s, ee) in res if cc == c and ee == e and (c, s, "s1_nt") in res]
            if not seeds:
                continue
            d = [100 * (res[(c, s, e)].mean() - res[(c, s, "s1_nt")].mean()) for s in seeds]
            se = float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0
            ax.barh(i + (j - 0.5) * h, np.mean(d), height=h * 0.9, color=SERIES[j], xerr=se,
                    error_kw={"ecolor": MUTED, "elinewidth": 1}, label=lab if i == 0 else None)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(range(len(cfgs)), [f"{c}  {NAMES[c]}" for c in cfgs])
    ax.invert_yaxis()
    ax.set_xlabel("change in success vs System 1 mean (points)", fontsize=9, color=MUTED)
    ax.set_title("Does checking actions with the world model help?", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_memory(rows: list[dict], path: Path) -> None:
    """Success with the camera missing, per fill rule (columns), for the memory configs (rows)."""
    import matplotlib.pyplot as plt
    cfgs = [c for c in ("C3", "C5", "C6", "C7") if any(r["config"] == c and r["eval"] == "rec1" for r in rows)]
    if not cfgs:
        return
    evs = ["rec1", "cam0.5-blank", "cam0.5-hold", "cam0.5-predict", "cam0.8-hold", "cam0.8-predict"]
    labels = ["camera on", "50% off: blank", "50%: hold", "50%: predict", "80%: hold", "80%: predict"]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(cfgs) * 1.6 + 1.6))
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    w = 0.8 / len(cfgs)
    for j, c in enumerate(cfgs):
        xs, ys, es = [], [], []
        for i, e in enumerate(evs):
            r = next((x for x in rows if x["config"] == c and x["eval"] == e), None)
            if r:
                xs.append(i + (j - (len(cfgs) - 1) / 2) * w)
                ys.append(r["mean"])
                es.append(0 if np.isnan(r["se"]) else r["se"])
        color = SERIES[j]
        ax.errorbar(xs, ys, yerr=es, fmt="o", color=color, ms=6, elinewidth=1.5, label=f"{c} {NAMES[c]}")
    ax.set_xticks(range(len(evs)), labels, fontsize=8)
    ax.set_ylabel("success (%), decision every block", fontsize=9, color=MUTED)
    ax.set_ylim(0, 100)
    ax.set_title("Memory under camera dropout", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(profile: str, home: Path, task: str = "cube") -> dict:
    os.environ["STABLEWM_HOME"] = str(home)
    from colab_plan import CONFIGS, PROFILES
    res = load(home, PROFILES[profile]["num_eval"], CONFIGS, task)
    cfgs = [c for c in CONFIGS if any(k[0] == c for k in res)]
    rows, t_md = table(res, cfgs)
    con, c_md = contrasts(res)
    # diagnostics files carry no task name and are only produced for Cube
    diag, d_md = diagnostics(home, {c: CONFIGS[c] for c in cfgs}) if task == "cube" else ([], "(Cube only)")
    out_dir = home / "results" / ("colab" if task == "cube" else f"colab_{task}")
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = []
    if rows:
        fig_success(rows, cfgs, out_dir / "fig_success.png")
        fig_verification(res, cfgs, out_dir / "fig_verification.png")
        fig_memory(rows, out_dir / "fig_memory.png")
        figs = sorted(str(p) for p in out_dir.glob("fig_*.png"))
    md = [f"# GPU study: {task} ({profile} profile)", "",
          f"{len({(c, s) for (c, s, e) in res})} trained models, {len(res)} evaluations; success in % (seeds).", "",
          "## Success by configuration", "", t_md, "",
          "## Paired contrasts (exact McNemar on pairs pooled over seeds; ARL rows: Wilcoxon signed-rank)", "",
          c_md, "",
          "## Diagnostics (seed 0)", "", d_md, ""]
    (out_dir / "summary.md").write_text("\n".join(md))
    import csv
    for name, data in (("success.csv", rows), ("contrasts.csv", con), ("diagnostics.csv", diag)):
        if data:
            with open(out_dir / name, "w", newline="") as fh:
                fields = list(dict.fromkeys(k for row in data for k in row))     # union, first-seen order
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerows(data)
    print("\n".join(md))
    return {"summary": str(out_dir / "summary.md"), "figures": figs}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="standard")
    ap.add_argument("--home", default=os.environ.get("STABLEWM_HOME", str(HERE / "swm_home")))
    ap.add_argument("--task", default="cube", choices=["cube", "tworoom"])
    a = ap.parse_args()
    main(a.profile, Path(a.home), a.task)
