"""Numbers of the re-run CPU-study experiments (run_rerun.sh), as one JSON object on stdout.

Reads the result files that evaluate.py, critic.py and contrastive_verifier.py wrote for the re-run (copied to
results/cpu_rerun/), finds each one by its recorded settings (not by file name), and checks that every paired
comparison uses the same start/goal pairs. Counts are integers; p-values are exact two-sided McNemar tests.

    python analyze_rerun.py --runs ../../results/cpu_rerun
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

G, O = "lewm+idm+sa", "lewm+idm+sa+obj"
K = 16                                                       # candidates per decision (evaluate.py --k 16)


class Runs:
    def __init__(self, folder: Path):
        self.files = {f.name: json.loads(f.read_text()) for f in sorted(folder.glob("*.json"))}

    def one(self, **want) -> dict:
        """The single result file whose recorded fields equal ``want`` (std_scale 1.0 is stored as None)."""
        def ok(r):
            for k, v in want.items():
                got = r.get(k)
                if k == "std_scale" and got is None:
                    got = 1.0
                if k == "select" and got is None:
                    got = "cost"
                if got != v:
                    return False
            return True
        hits = [(n, r) for n, r in self.files.items() if "successes" in r and ok(r)]
        if len(hits) != 1:
            raise LookupError(f"{len(hits)} result files match {want}: {[n for n, _ in hits]}")
        return hits[0][1]

    def named(self, name: str) -> dict:
        return self.files[name]


def solved(r: dict) -> int:
    return int(sum(r["successes"]))


def mcnemar(a: dict, b: dict) -> dict:
    if (a.get("episodes_idx"), a.get("start_steps")) != (b.get("episodes_idx"), b.get("start_steps")):
        raise ValueError(f"different pairs: {a['run']} vs {b['run']}")
    x, y = np.asarray(a["successes"], bool), np.asarray(b["successes"], bool)
    n10, n01 = int((x & ~y).sum()), int((~x & y).sum())
    p = float(binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else 1.0
    return {"a_only": n10, "b_only": n01, "p_mcnemar": p}


def nontrivial(**kw) -> dict:
    return {"task": "cube", "pairs": "nontrivial", "goal_offset": 25, "eval_budget": 50, "eval_seed": 42, **kw}


def main(folder: Path) -> dict:
    R = Runs(folder)
    out: dict = {}

    r = R.one(task="cube", spec="random", pairs="all", num_eval=50)
    out["random_standard"] = {"solved": solved(r), "pairs": len(r["successes"]),
                              "solved_at_start": int(sum(r["trivial"]))}

    rep = {}
    for spec, tag in ((O, "C3"), (G, "C2")):
        for seed in (0, 1):
            rep[(tag, seed)] = R.one(**nontrivial(spec=spec, seed=seed, controller="s1", k=1, num_eval=50,
                                                  policy_file="policy_s1.pt"))
    a = {"successes": rep[("C3", 0)]["successes"] + rep[("C3", 1)]["successes"],
         "episodes_idx": rep[("C3", 0)]["episodes_idx"] * 2, "start_steps": rep[("C3", 0)]["start_steps"] * 2, "run": "C3"}
    b = {"successes": rep[("C2", 0)]["successes"] + rep[("C2", 1)]["successes"],
         "episodes_idx": rep[("C2", 1)]["episodes_idx"] * 2, "start_steps": rep[("C2", 1)]["start_steps"] * 2, "run": "C2"}
    out["replication"] = {"C3_solved": solved(a), "C2_solved": solved(b), "pairs": len(a["successes"]),
                          **mcnemar(a, b),
                          "per_seed": {f"s{s}": {"C3_solved": solved(rep[("C3", s)]), "C2_solved": solved(rep[("C2", s)]),
                                                 **mcnemar(rep[("C3", s)], rep[("C2", s)])} for s in (0, 1)}}

    s1 = R.one(**nontrivial(spec="lewm", seed=0, controller="s1", k=1, num_eval=3, env_chunk=1))
    cem = R.one(**nontrivial(spec="lewm", seed=0, controller="cem", num_eval=3, env_chunk=1))
    t1, tc = s1["decision_seconds"], cem["decision_seconds"]
    ratio = tc["median"] / t1["median"]
    out["timing"] = {"s1": {"median_s": t1["median"], "decisions": t1["calls"]},
                     "cem": {"median_s": tc["median"], "decisions": tc["calls"]},
                     "ratio": ratio, "ratio_rounded": int(round(ratio, -1)), "robots": 1}

    comps = []
    for spec, tag in ((G, "C2"), (O, "C3")):
        for s in (0.25, 0.5, 1.0, 2.0):
            wm = R.one(**nontrivial(spec=spec, seed=0, controller="s1", k=K, num_eval=50, std_scale=s, select="cost",
                                    policy_file="policy_s1.pt"))
            rnd = R.one(**nontrivial(spec=spec, seed=0, controller="s1", k=K, num_eval=50, std_scale=s, select="random",
                                     policy_file="policy_s1.pt"))
            comps.append({"model": tag, "std_scale": s, "world_model": solved(wm), "random": solved(rnd),
                          **mcnemar(wm, rnd)})
    margins = [c["world_model"] - c["random"] for c in comps]
    by = {(c["model"], c["std_scale"]): c for c in comps}
    out["selection"] = {"comparisons": len(comps), "wins": sum(m > 0 for m in margins), "ties": sum(m == 0 for m in margins),
                        "losses": sum(m < 0 for m in margins), "max_margin": max(margins), "min_margin": min(margins),
                        "max_loss": max(0, -min(margins)), "significant": sum(c["p_mcnemar"] < 0.05 for c in comps),
                        "pairs": 50, "widest": {t: by[(t, 2.0)] for t in ("C2", "C3")},
                        "near_mean": {t: by[(t, 0.25)] for t in ("C2", "C3")}, "rows": comps}

    mean = R.one(**nontrivial(spec=O, seed=0, controller="s1", k=1, num_eval=50, policy_file="policy_gauss6k.pt"))
    sel = {m: R.one(**nontrivial(spec=O, seed=0, controller="s1", k=K, num_eval=50, policy_file="policy_gauss6k.pt",
                                 select=m)) for m in ("oracle", "critic", "random", "cost")}
    out["ceiling"] = {"pairs": len(mean["successes"]), "mean_solved": solved(mean), "oracle_solved": solved(sel["oracle"]),
                      "oracle_gain": solved(sel["oracle"]) - solved(mean),
                      "oracle_failed": len(mean["successes"]) - solved(sel["oracle"]),
                      **mcnemar(sel["oracle"], mean),
                      "critic_solved": solved(sel["critic"]), "critic_p": mcnemar(sel["critic"], mean)["p_mcnemar"],
                      "random_solved": solved(sel["random"]),
                      "world_model_solved": solved(sel["cost"])}

    crit = R.named(f"critic__{O}__seed0.json")
    out["critic"] = {"train_states": crit["train_states"], "train_outcomes": crit["train_states"] * K,
                     "spearman": crit["heldout_eval_set"]["spearman_mean"],
                     "heldout_states": crit["heldout_eval_set"]["states"]}

    vin = R.named(f"verifier__{O}__seed0__verifier_in.json")["offline"]
    vsel = R.one(**nontrivial(spec=O, seed=0, controller="s1", k=K, num_eval=50, policy_file="policy_gauss6k.pt",
                              select="verifier", selector_file="verifier_in.pt"))
    vhard = R.one(**nontrivial(spec=O, seed=0, controller="s1", k=K, num_eval=50, policy_file="policy_gauss6k.pt",
                               select="verifier", selector_file="verifier_hard.pt"))
    vret = R.one(**nontrivial(spec=O, seed=0, controller="s1", k=1, num_eval=50, policy_file="verifier_in_retrieval.pt"))
    out["verifier"] = {"top1": vin["top1_true_chunk_of_1024"], "spearman": vin["spearman_mean"],
                       "select_solved": solved(vsel), "select_p": mcnemar(vsel, mean)["p_mcnemar"],
                       "hard_solved": solved(vhard), "retrieval_solved": solved(vret),
                       "retrieval_p": mcnemar(vret, mean)["p_mcnemar"]}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    a = ap.parse_args()
    print(json.dumps(main(a.runs), indent=1))
