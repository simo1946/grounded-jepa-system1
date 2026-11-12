"""Tables and paired tests for the verification-regime study (prereg_verification.md).

python analyze_verify.py   ->  prints markdown, writes swm_home/results/verify_summary.json
"""
from __future__ import annotations

import json

from scipy.stats import binomtest

from models import HOME

R = HOME / "results"
TAIL = "__nontrivial__off25_bud50_es42.json"
MODELS = {"grounded": "lewm+idm+sa", "object": "lewm+idm+sa+obj"}
SPREADS = [0.25, 0.5, 1.0, 2.0, 4.0]


def load(spec: str, ctrl: str):
    p = R / f"cube__{spec}__seed0__{ctrl}{TAIL}"
    return json.loads(p.read_text())["successes"] if p.exists() else None


def mcnemar(a, b):
    """Exact two-sided McNemar on paired 0/1 outcomes; returns (wins_a, wins_b, p)."""
    wa = sum(x and not y for x, y in zip(a, b))
    wb = sum(y and not x for x, y in zip(a, b))
    return wa, wb, (binomtest(wa, wa + wb, 0.5).pvalue if wa + wb else 1.0)


def tag(s: float) -> str:
    return "" if s == 1.0 else f"__std{s:g}"


def main() -> dict:
    out = {"E1": {}, "E2": {}}
    print("## E1: spread sweep (successes / 50)\n")
    print("| model | spread | mean (k=1) | WM best-of-16 | random sample | WM - random | WM vs mean (p) |")
    print("|---|---|---|---|---|---|---|")
    for name, spec in MODELS.items():
        mean = load(spec, "s1_k1")
        for s in SPREADS:
            cost = load(spec, "s1" + tag(s))
            rnd = load(spec, "s1_pickrandom" + tag(s))
            row = {"mean": sum(mean), "cost": sum(cost) if cost else None, "random": sum(rnd) if rnd else None}
            if cost:
                row["cost_vs_mean"] = mcnemar(cost, mean)
            if cost and rnd:
                row["cost_vs_random"] = mcnemar(cost, rnd)
            out["E1"][f"{name}@{s:g}"] = row
            gap = f"{row['cost'] - row['random']:+d}" if cost and rnd else "—"
            p = f"{row['cost_vs_mean'][2]:.3f}" if cost else "—"
            print(f"| {name} | {s:g} | {row['mean']} | {row['cost'] if cost else '—'} | "
                  f"{row['random'] if rnd else '—'} | {gap} | {p} |")

    spec = MODELS["grounded"]
    pool = ["policy_cached", "policy_p1", "policy_p2", "policy_p3", "policy_p4"]
    members = {m: load(spec, f"s1_k1__pol-{m}") for m in pool}
    sel = load(spec, "s1_k5__pol-policy_cached__pool5-select")
    avg = load(spec, "s1_k1__pol-policy_cached__pool5-mean")
    print("\n## E2: pool of 5 independently trained System 1 policies (grounded model)\n")
    if all(v is not None for v in members.values()):
        ms = {m: sum(v) for m, v in members.items()}
        out["E2"]["members"] = ms
        out["E2"]["member_mean"] = sum(ms.values()) / len(ms)
        print("members (k=1):", ", ".join(f"{m.removeprefix('policy_')} {v}" for m, v in ms.items()),
              f"| average {out['E2']['member_mean']:.1f}")
        if sel:
            out["E2"]["select"] = sum(sel)
            out["E2"]["select_vs_member0"] = mcnemar(sel, members["policy_cached"])
            out["E2"]["members_beaten"] = sum(sum(sel) > v for v in ms.values())
            print(f"WM selects among the 5 means: {sum(sel)}; vs member 0 (primary): "
                  f"{out['E2']['select_vs_member0']}; beats {out['E2']['members_beaten']}/5 members")
            if avg:
                out["E2"]["mean"] = sum(avg)
                out["E2"]["select_vs_average"] = mcnemar(sel, avg)
                print(f"average of the 5 means: {sum(avg)}; selection vs averaging: {out['E2']['select_vs_average']}")
    else:
        print("pending")
    (R / "verify_summary.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
