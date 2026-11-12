"""Analysis of the feedback-frequency study, as pre-registered in prereg_feedback.md (plan and 2026-11-09 addendum).

Decision rule (addendum): cluster bootstrap over the 40 evaluation episodes behind the 60 pairs (each drawn
episode brings all of its pairs, for all four models), 10,000 resamples; 95% interval of the mean success
difference; two-sided bootstrap p, Holm-corrected over H1-H11. An effect is claimed only if the Holm p < 0.05
and the sign is the same in all four models. Also reported: per-model wins/losses with an interval over that
model's pairs, the same bootstrap clustered by pair (to show the effect of clustering by episode), and the
pooled exact McNemar test of the original plan (descriptive).

    python analyze_feedback.py            # writes results/feedback_summary.json and prints a table
    python analyze_feedback.py --runs ../../results/feedback_study/runs --json   # JSON only, on stdout
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from cluster_bootstrap import cluster_bootstrap, holm
from evaluate import result_path
from models import HOME
from scipy.stats import binomtest

MODELS = [("lewm", 0), ("lewm", 1), ("lewm+idm+sa+obj", 0), ("lewm+idm+sa+obj", 1)]
NAMES = {"lewm": "C1", "lewm+idm+sa+obj": "C3"}
CONDITIONS = {   # name -> (controller, k, policy file, receding blocks, executed steps)
    "S1-25": ("s1", 1, "policy_s1.pt", 5, 25),
    "S1-5": ("s1", 1, "policy_s1.pt", 1, 5),
    "S1-1": ("s1", 1, "policy_s1.pt", 1, 1),
    "IDM-1": ("gcidm", 16, "gcidm.pt", 1, 1),
    "IDM-1-noH": ("gcidm", 16, "gcidm_noh.pt", 1, 1),
    "OIDM-1": ("gcidm_official", 16, "official_gcidm.pt", 1, 1),
    "OIDM-1-H1": ("gcidm_official", 16, "official_gcidm_h1.pt", 1, 1),
    "S1c5": ("s1short", 16, "policy_c5.pt", 1, 5),
    "S1c1": ("s1short", 16, "policy_c1.pt", 1, 1),
}
CONTRASTS = [("H1a", "S1-5", "S1-25"), ("H1b", "S1-1", "S1-25"), ("H2", "IDM-1", "S1-1"),
             ("H3", "IDM-1", "S1-25"), ("H4", "IDM-1", "IDM-1-noH"),
             ("H5", "OIDM-1", "IDM-1"), ("H6", "OIDM-1", "S1-25"), ("H7", "OIDM-1", "OIDM-1-H1"),
             ("H8", "S1c5", "S1-5"), ("H9", "S1c1", "S1-1"), ("H10", "S1c5", "S1-25"), ("H11", "S1c1", "S1-25")]
N_BOOT = 10_000


def load(spec: str, seed: int, cond: str, runs: Path | None = None) -> dict:
    ctrl, k, pol, rec, ex = CONDITIONS[cond]
    p = result_path("cube", spec, seed, ctrl, k, 0.6, "mse", "nontrivial", 25, 50, 42, "cost", pol, rec,
                    num_eval=60, exec_steps=ex)
    return json.loads((runs / p.name if runs else p).read_text())


def mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """a, b: 0/1 outcomes on the same pairs -> (a only, b only, exact two-sided p)."""
    n10, n01 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    return n10, n01, float(p)


def main(runs: Path | None = None, json_only: bool = False) -> dict:
    res = {(s, sd, c): load(s, sd, c, runs) for s, sd in MODELS for c in CONDITIONS}
    ref = res[MODELS[0] + ("S1-25",)]
    for r in res.values():                                   # every condition ran on the same pairs
        if r["episodes_idx"] != ref["episodes_idx"] or r["start_steps"] != ref["start_steps"] \
                or r["eval_seed"] != ref["eval_seed"] or len(r["successes"]) != 60:
            raise RuntimeError(f"pairs differ: {r['run']} {r['controller']} {r['policy_file']}")
    episodes = np.asarray(ref["episodes_idx"])
    succ = {key: np.asarray(r["successes"], float) for key, r in res.items()}
    table = {f"{NAMES[s]} seed {sd}": {c: round(100 * succ[(s, sd, c)].mean(), 1) for c in CONDITIONS}
             for s, sd in MODELS}
    pooled_pct = {c: round(100 * np.mean([succ[(s, sd, c)].mean() for s, sd in MODELS]), 1) for c in CONDITIONS}
    timing = {c: {"median_decision_s": round(float(np.median([res[(s, sd, c)]["decision_seconds"]["median"]
                                                               for s, sd in MODELS])), 4),
                  "decision_calls_per_batch_episode": float(np.median([res[(s, sd, c)]["decision_seconds"]["calls"]
                                                                        for s, sd in MODELS])) / 12,
                  "eval_minutes_median": float(np.median([res[(s, sd, c)]["eval_minutes"] for s, sd in MODELS]))}
              for c in CONDITIONS}
    contrasts, p_boot = {}, []
    for name, a, b in CONTRASTS:
        A = np.stack([succ[(s, sd, a)] for s, sd in MODELS])
        B = np.stack([succ[(s, sd, b)] for s, sd in MODELS])
        per = {}
        for i, (s, sd) in enumerate(MODELS):
            n10, n01, p = mcnemar(A[i], B[i])
            bs = cluster_bootstrap(A[i], B[i], episodes, N_BOOT)
            per[f"{NAMES[s]} seed {sd}"] = {"a": round(100 * A[i].mean(), 1), "b": round(100 * B[i].mean(), 1),
                                            "a_only": n10, "b_only": n01, "p_mcnemar": round(p, 4),
                                            "ci": [round(bs["ci_lo"], 1), round(bs["ci_hi"], 1)]}
        by_ep = cluster_bootstrap(A, B, episodes, N_BOOT)
        by_pair = cluster_bootstrap(A, B, None, N_BOOT)
        n10, n01, p = mcnemar(A.ravel(), B.ravel())
        signs = {int(np.sign(v["a"] - v["b"])) for v in per.values()}
        contrasts[name] = {"a": a, "b": b, "per_model": per,
                           "pooled": {"a": round(100 * A.mean(), 1), "b": round(100 * B.mean(), 1),
                                      "a_only": n10, "b_only": n01, "p_mcnemar": p},
                           "bootstrap_episode": by_ep, "bootstrap_pair": by_pair,
                           "same_sign_all_models": len(signs) == 1 and 0 not in signs}
        p_boot.append(by_ep["p_boot"])
    for (name, _, _), ph in zip(CONTRASTS, holm(p_boot)):
        c = contrasts[name]
        c["p_boot_holm"] = ph
        c["claimed"] = bool(c["same_sign_all_models"] and ph < 0.05)
    out = {"success_pct": table, "pooled_pct": pooled_pct, "contrasts": contrasts, "timing": timing,
           "pairs": {"n": 60, "episodes": int(len(set(episodes.tolist()))), "eval_seed": ref["eval_seed"],
                     "goal_offset": 25, "budget": 50, "episodes_idx": ref["episodes_idx"],
                     "start_steps": ref["start_steps"]},
           "bootstrap": {"n_boot": N_BOOT, "seed": 0, "cluster": "evaluation episode", "level": 0.95}}
    if json_only:
        print(json.dumps(out, indent=1))
        return out
    path = HOME / "results" / "feedback_summary.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(table, indent=1), json.dumps(pooled_pct))
    for name, c in contrasts.items():
        e, q = c["bootstrap_episode"], c["bootstrap_pair"]
        print(f"{name} {c['a']} vs {c['b']}: {e['diff_pts']:+.1f} pts, 95% CI [{e['ci_lo']:+.1f}, {e['ci_hi']:+.1f}] "
              f"(by pair [{q['ci_lo']:+.1f}, {q['ci_hi']:+.1f}]), p_boot {e['p_boot']:.4f}, Holm {c['p_boot_holm']:.4f}, "
              f"claimed {c['claimed']}; per model", {m: (v["a"], v["b"], v["a_only"], v["b_only"], v["ci"])
                                                       for m, v in c["per_model"].items()})
    print(json.dumps(timing, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=None, help="folder of the result files (default: HOME/results)")
    ap.add_argument("--json", action="store_true", help="print only the summary as JSON")
    a = ap.parse_args()
    main(a.runs, a.json)
