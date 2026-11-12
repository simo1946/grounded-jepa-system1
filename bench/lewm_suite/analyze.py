"""Collect every Cube result into one component-importance table.

For each protocol (goal offset, budget, pair set, eval seed) it lists success
on all pairs and on the non-trivial subset (pairs whose start does not already
satisfy the goal), and runs exact McNemar tests for paired contrasts that
isolate ONE component (grounding, SA head, predictor, encoder, controller,
cost, seed). Results are paired only when they use the identical start/goal
pairs (same protocol key), which holds because pairs are drawn from
eval_seed with a deterministic picker.

    python analyze.py                # prints markdown, writes results/summary.{md,json}
"""
from __future__ import annotations

import json
from math import comb
from pathlib import Path

import numpy as np

HOME = Path(__file__).resolve().parent / "swm_home"
RES = HOME / "results"


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sided exact McNemar (binomial on discordant pairs)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    n = n10 + n01
    if n == 0:
        return {"a_only": 0, "b_only": 0, "p": 1.0}
    k = min(n10, n01)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"a_only": n10, "b_only": n01, "p": round(p, 4)}


def ctrl_label(r: dict) -> str:
    c = r.get("controller") or "cem"
    # early result files did not record k/rho/cost; they used the defaults (16, 0.6, mse)
    k, rho, cost = r.get("k") or 16, r.get("rho") or 0.6, r.get("cost") or "mse"
    if r["spec"] == "random":
        return "random"
    pf = r.get("policy_file")
    if c != "cem" and pf and pf != "policy.pt":
        r = {**r, "policy_file": None}
        return f"{ctrl_label(r)} <{pf.removesuffix('.pt')}>"
    tags = ""
    if r.get("receding", 5) != 5:
        tags += f" rec={r['receding']}"
    if r.get("camera_drop"):
        tags += f" cam-drop={r['camera_drop']:g}/{r.get('fill')}"
    if tags:
        return ctrl_label({**r, "receding": 5, "camera_drop": 0}) + tags
    if c == "cem":
        lab = "CEM"
    elif c == "s1" and r.get("select") == "random":
        return f"S1 random sample of {k}"
    elif c == "s1":
        lab = "S1 mean" if k == 1 else f"S1 best-of-{k}"
    else:
        lab = f"S1+S2 gated (K={k}, rho={rho:g})"
    return f"{lab} [{cost}]" if c != "s1" or k != 1 else lab


def load() -> list[dict]:
    rows = []
    for f in sorted(RES.glob("cube__*.json")):
        r = json.loads(f.read_text())
        succ = np.asarray(r["successes"], int)
        off = r["goal_offset"]
        if "trivial" in r:
            triv = np.asarray(r["trivial"], bool)
        else:                                   # older results: mask saved separately (all-pairs protocol)
            m = RES / f"trivial_mask_off{off}.npy"
            triv = np.load(m).astype(bool) if m.exists() and r.get("pairs", "all") == "all" else None
        r.update(file=f.name, succ=succ, triv=triv, ctrl=ctrl_label(r),
                 proto=(off, r["eval_budget"], r.get("pairs", "all"), r["eval_seed"], r["num_eval"]))
        rows.append(r)
    return rows


def key(r: dict) -> tuple:
    return (r["spec"], r.get("seed") if r["spec"] != "random" else None, r["ctrl"])


# contrasts: (description, component, a(spec, seed, ctrl), b(...)) evaluated per protocol and seed
def contrasts(seeds):
    out = []
    for s in seeds:
        g = "lewm+idm+sa"
        out += [
            ("grounding (IDM+SA) under CEM", "grounding", ("lewm+idm+sa", s, "CEM [mse]"), ("lewm", s, "CEM [mse]")),
            ("grounding (IDM+SA) under S1", "grounding", ("lewm+idm+sa", s, "S1 mean"), ("lewm", s, "S1 mean")),
            ("SA head (IDM+SA vs IDM) under S1", "grounding", (g, s, "S1 mean"), ("lewm+idm", s, "S1 mean")),
            ("SA head (IDM+SA vs IDM) under CEM", "grounding", (g, s, "CEM [mse]"), ("lewm+idm", s, "CEM [mse]")),
            ("hybrid Mamba-3/attn predictor vs LeWM predictor, S1", "predictor",
             ("hybrid+idm+sa", s, "S1 mean"), (g, s, "S1 mean")),
            ("hybrid Mamba-3/attn predictor vs LeWM predictor, CEM", "predictor",
             ("hybrid+idm+sa", s, "CEM [mse]"), (g, s, "CEM [mse]")),
            ("MobileNetV4-S encoder vs ViT-tiny, S1", "encoder",
             (g + "+enc=mnv4s", s, "S1 mean"), (g, s, "S1 mean")),
            ("MobileNetV4-S encoder vs ViT-tiny, CEM", "encoder",
             (g + "+enc=mnv4s", s, "CEM [mse]"), (g, s, "CEM [mse]")),
        ]
        for spec in (g, "lewm", "hybrid+idm+sa"):
            out += [
                (f"S1 mean vs CEM ({spec})", "controller", (spec, s, "S1 mean"), (spec, s, "CEM [mse]")),
                (f"S1 mean vs S1 best-of-16 by latent MSE ({spec})", "selection",
                 (spec, s, "S1 mean"), (spec, s, "S1 best-of-16 [mse]")),
                (f"S1 best-of-16 by latent MSE vs random sample ({spec})", "selection",
                 (spec, s, "S1 best-of-16 [mse]"), (spec, s, "S1 random sample of 16")),
                (f"S1 best-of-16 by reach critic vs random sample ({spec})", "selection",
                 (spec, s, "S1 best-of-16 [reach]"), (spec, s, "S1 random sample of 16")),
                (f"S1 mean vs random sample ({spec})", "selection",
                 (spec, s, "S1 mean"), (spec, s, "S1 random sample of 16")),
                (f"S1 best-of-16 by reach critic vs S1 mean ({spec})", "selection",
                 (spec, s, "S1 best-of-16 [reach]"), (spec, s, "S1 mean")),
                (f"gated S1+S2 (latent MSE) vs S1 mean ({spec})", "gating",
                 (spec, s, "S1+S2 gated (K=16, rho=0.6) [mse]"), (spec, s, "S1 mean")),
                (f"gated S1+S2 (reach) vs S1 mean ({spec})", "gating",
                 (spec, s, "S1+S2 gated (K=16, rho=0.6) [reach]"), (spec, s, "S1 mean")),
                (f"CEM reach cost vs CEM latent MSE ({spec})", "cost",
                 (spec, s, "CEM [reach]"), (spec, s, "CEM [mse]")),
            ]
    O, G = "lewm+idm+sa+obj", "lewm+idm+sa"
    out += [
        ("object-aware grounding vs proprio grounding, S1", "grounding", (O, 0, "S1 mean"), (G, 0, "S1 mean")),
        ("object-aware grounding vs plain LeWM, S1", "grounding", (O, 0, "S1 mean"), ("lewm", 0, "S1 mean")),
        ("object-aware grounding vs proprio grounding, CEM", "grounding", (O, 0, "CEM [mse]"), (G, 0, "CEM [mse]")),
    ]
    for spec in (O, G):
        out += [
            (f"CEM object-aware cost vs latent MSE ({spec})", "cost", (spec, 0, "CEM [probe]"), (spec, 0, "CEM [mse]")),
            (f"S1 best-of-16 by object-aware cost vs S1 mean ({spec})", "selection",
             (spec, 0, "S1 best-of-16 [probe]"), (spec, 0, "S1 mean")),
            (f"gated S1+S2 (object-aware) vs S1 mean ({spec})", "gating",
             (spec, 0, "S1+S2 gated (K=16, rho=0.6) [probe]"), (spec, 0, "S1 mean")),
        ]
    cached = (G, 0, "S1 mean <policy_cached>")
    for f, desc in (("policy_s-proprio", "+ proprioception"), ("policy_s-proprio-obj", "+ proprio + object sensor (1 cm)"),
                    ("policy_s-proprio-obj-noisy", "+ proprio + object sensor (2 cm, 20% dropout)"),
                    ("policy_s-goalcoords", "+ sensors + goal as coordinates (different task spec)"),
                    ("policy_hier", "two-level S1 (subgoal net)")):
        out.append((f"S1 {desc} vs S1 latents only", "sensors" if "s-" in f else "hierarchy",
                    (G, 0, f"S1 mean <{f}>"), cached))
    out.append(("cached-latent S1 vs original S1 (training-pipeline check)", "check", cached, (G, 0, "S1 mean")))
    for r in (1, 2, 3):
        out.append((f"continual learning round {r} vs equal-steps control", "continual",
                    (G, 0, f"S1 mean <policy_loop_r{r}>"), (G, 0, f"S1 mean <policy_ctrl_r{r}>")))
        out.append((f"continual learning round {r} vs start", "continual",
                    (G, 0, f"S1 mean <policy_loop_r{r}>"), cached))
    out += [("success-filtered own experience vs matched expert-only fine-tuning (1800 steps)", "continual",
             (G, 0, "S1 mean <policy_loop_filtered>"), (G, 0, "S1 mean <policy_ctrl_1800>")),
            ("expert-only fine-tuning +1800 steps vs start (is S1 under-trained?)", "check",
             (G, 0, "S1 mean <policy_ctrl_1800>"), cached),
            ("two-level S1, distance-gated (post-hoc) vs S1 latents only", "hierarchy",
             (G, 0, "S1 mean <policy_hier_gated>"), cached)]
    for spec in (G, "hybrid+idm+sa"):
        base = (spec, 0, "S1 mean rec=1")
        for fill in ("blank", "hold", "predict"):
            out.append((f"camera 50% dropped, fill={fill} vs full camera ({spec})", "memory",
                        (spec, 0, f"S1 mean rec=1 cam-drop=0.5/{fill}"), base))
        out.append((f"camera dropped: world-model prediction vs hold last ({spec})", "memory",
                    (spec, 0, "S1 mean rec=1 cam-drop=0.5/predict"), (spec, 0, "S1 mean rec=1 cam-drop=0.5/hold")))
    out.append(("camera dropped + prediction: hybrid vs LeWM predictor", "predictor",
                ("hybrid+idm+sa", 0, "S1 mean rec=1 cam-drop=0.5/predict"), (G, 0, "S1 mean rec=1 cam-drop=0.5/predict")))
    out += [("CEM grounded vs random", "baseline", ("lewm+idm+sa", 0, "CEM [mse]"), ("random", None, "random")),
            ("S1 mean grounded vs random", "baseline", ("lewm+idm+sa", 0, "S1 mean"), ("random", None, "random"))]
    return out


def main() -> None:
    rows = load()
    seeds = sorted({r["seed"] for r in rows if r["spec"] != "random"})
    protos = sorted({r["proto"] for r in rows})
    md, js = ["# Cube results (reduced CPU setup: 400 episodes, 64 px, 2000 steps)", ""], {"protocols": []}
    for p in protos:
        off, bud, pairs, es, n = p
        sub = {key(r): r for r in rows if r["proto"] == p}
        title = f"goal +{off} steps, budget {bud}, pairs={pairs}, eval_seed={es}, n={n}"
        md += [f"## {title}", "", "| model | seed | controller | success (all) | success (non-trivial) | s/decision |",
               "|---|---|---|---|---|---|"]
        prow = {"protocol": title, "results": [], "contrasts": []}
        for k_, r in sorted(sub.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][2])):
            succ, triv = r["succ"], r["triv"]
            nt = f"{int(succ[~triv].sum())}/{int((~triv).sum())}" if triv is not None and (~triv).any() else "n/a"
            st = r.get("system12_stats") or {}
            spd = ""
            if st.get("env_decisions"):
                spd = f"{(st['s1_seconds'] + st['s2_seconds']) / st['env_decisions']:.2f}"
            md.append(f"| {r['spec']} | {'' if k_[1] is None else k_[1]} | {r['ctrl']} | "
                      f"{int(succ.sum())}/{len(succ)} | {nt} | {spd} |")
            prow["results"].append({"spec": r["spec"], "seed": k_[1], "controller": r["ctrl"],
                                    "success": int(succ.sum()), "n": len(succ), "nontrivial": nt, "file": r["file"]})
        md += ["", "| contrast (A vs B) | component | A | B | A-only | B-only | McNemar p |", "|---|---|---|---|---|---|---|"]
        for desc, comp, a, b in contrasts(seeds):
            if a not in sub or b not in sub:
                continue
            ra, rb = sub[a], sub[b]
            mask = np.ones(len(ra["succ"]), bool)
            if pairs == "all" and ra["triv"] is not None:
                mask = ~ra["triv"]                    # trivial pairs carry no information on control
            t = mcnemar_exact(ra["succ"][mask], rb["succ"][mask])
            sa, sb = int(ra["succ"][mask].sum()), int(rb["succ"][mask].sum())
            md.append(f"| {desc} (seed {a[1]}) | {comp} | {sa}/{mask.sum()} | {sb}/{mask.sum()} | "
                      f"{t['a_only']} | {t['b_only']} | {t['p']} |")
            prow["contrasts"].append({"contrast": desc, "seed": a[1], "component": comp,
                                      "a": sa, "b": sb, "n": int(mask.sum()), **t})
        md.append("")
        js["protocols"].append(prow)
    md.append("Contrasts on 'pairs=all' protocols use only the non-trivial subset. p = two-sided exact McNemar.")
    text = "\n".join(md)
    (RES / "summary.md").write_text(text)
    (RES / "summary.json").write_text(json.dumps(js, indent=2))
    print(text)


if __name__ == "__main__":
    main()
