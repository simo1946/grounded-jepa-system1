"""Tables and paired tests for prereg_overnight.md (E3 expressive System 1, E5 clean labels / DAgger).

python analyze_overnight.py  ->  markdown on stdout, swm_home/results/overnight_summary.json
"""
from __future__ import annotations

import json

from scipy.stats import binomtest

from models import HOME

R = HOME / "results"
TAIL = "__nontrivial__off25_bud50_es42.json"
G, O = "lewm+idm+sa", "lewm+idm+sa+obj"


def load(spec, ctrl):
    p = R / f"cube__{spec}__seed0__{ctrl}{TAIL}"
    return json.loads(p.read_text())["successes"] if p.exists() else None


def mcnemar(a, b):
    wa = sum(x and not y for x, y in zip(a, b))
    wb = sum(y and not x for x, y in zip(a, b))
    return {"a": sum(a), "b": sum(b), "a_only": wa, "b_only": wb,
            "p": binomtest(wa, wa + wb, 0.5).pvalue if wa + wb else 1.0}


def row(label, spec, ca, cb, out):
    a, b = load(spec, ca), load(spec, cb)
    if a is None or b is None:
        print(f"| {label} | {spec} | pending | | |")
        return
    m = mcnemar(a, b)
    out[f"{label} [{spec}]"] = m
    print(f"| {label} | {spec} | {m['a']} vs {m['b']} | {m['a_only']} / {m['b_only']} | {m['p']:.3f} |")


def main():
    out = {}
    print("| contrast | model | successes /50 | discordant | p (exact McNemar) |")
    print("|---|---|---|---|---|")
    for spec in (G, O):
        row("H4 flow vs Gaussian (deterministic, 6k steps)", spec, "s1_k1__pol-policy_flow6k", "s1_k1__pol-policy_gauss6k", out)
        row("H5 flow: best-of-16 by world model vs deterministic", spec, "s1__pol-policy_flow6k", "s1_k1__pol-policy_flow6k", out)
        row("H5b flow: best-of-16 vs random sample", spec, "s1__pol-policy_flow6k", "s1_pickrandom__pol-policy_flow6k", out)
        row("Gaussian 6k: best-of-16 vs mean", spec, "s1__pol-policy_gauss6k", "s1_k1__pol-policy_gauss6k", out)
        row("6k Gaussian vs original policy (budget)", spec, "s1_k1__pol-policy_gauss6k", "s1_k1", out)
    for spec in (G, O):
        row("clean simulated oracle chunks vs noisy expert labels (Gaussian)", spec, "s1_k1__pol-policy_osim6k",
            "s1_k1__pol-policy_gauss6k", out)
        row("clean simulated oracle chunks vs noisy expert labels (flow)", spec, "s1_k1__pol-policy_osimflow6k",
            "s1_k1__pol-policy_flow6k", out)
        row("clean simulated chunks: flow vs Gaussian", spec, "s1_k1__pol-policy_osimflow6k", "s1_k1__pol-policy_osim6k", out)
    row("H7 DAgger round 2 vs equal-steps control", G, "s1_k1__pol-policy_sdagger_r2", "s1_k1__pol-policy_sctrl_r2", out)
    row("DAgger round 1 vs equal-steps control", G, "s1_k1__pol-policy_sdagger_r1", "s1_k1__pol-policy_sctrl_r1", out)
    row("DAgger round 2 vs its start point", G, "s1_k1__pol-policy_sdagger_r2", "s1_k1__pol-policy_osim6k", out)
    print("\nFailed first attempt (block-level oracle labels, invalid for chunk policies; prereg addendum 2):")
    row("block-level labels vs noisy expert labels", G, "s1_k1__pol-policy_oracle6k", "s1_k1__pol-policy_gauss6k", out)
    row("block-level labels vs noisy expert labels", O, "s1_k1__pol-policy_oracle6k", "s1_k1__pol-policy_gauss6k", out)
    for spec in (G,):
        d = HOME / "checkpoints" / f"cube/{spec}/seed0" / "dagger"
        for f in sorted(d.glob("sround*.pt")) if d.exists() else []:
            import torch
            r = torch.load(f, map_location="cpu", weights_only=False)
            out[f"collection {f.stem}"] = {k: r[k] for k in ("success_rate", "blocks", "minutes", "policy_file")}
            print(f"\n{f.stem}: learner success on training pairs {r['success_rate']:.2f}, "
                  f"{r['blocks']} labelled block states, policy {r['policy_file']}")
    (R / "overnight_summary.json").write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    main()
