"""Every number the paper cites from the GPU study, as one JSON object on stdout.

Reads the per-pair result files of the GPU study (results/gpu_study/runs) and computes, with the functions of
analyze_colab.py: success per configuration and evaluation (mean and standard error over seeds, each seed, and a
95% pair-bootstrap interval),
every paired contrast (pooled and per seed: success of A and B, pairs solved only by A or only by B, exact
McNemar p, pair-bootstrap interval and p, Holm correction over the pre-registered family), the read-out and
ranking diagnostics, and evaluation times. Success and differences are in percent (points); counts are integers.

    osaudit run --input ../../results/gpu_study/runs -o ../../results/paper/gpu_cube.json -- \
        python paper_values_gpu.py --runs ../../results/gpu_study/runs --task cube
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from analyze_colab import NAMES, PRIMARY, _paired, contrasts_list, diagnostics, holm, load, mcnemar_exact, verification_did_rows
from cluster_bootstrap import cluster_bootstrap
from study_configs import CONFIGS, PROFILES

N_BOOT = 10_000


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def success(res: dict) -> dict:
    out: dict = {}
    for (c, s, e), v in sorted(res.items()):
        out.setdefault(c, {}).setdefault(e, {"seeds": {}, "solved": {}, "pairs": int(len(v))})
        out[c][e]["seeds"][f"s{s}"] = float(100 * v.mean())
        out[c][e]["solved"][f"s{s}"] = int(v.sum())
    for c in out:
        for e, d in out[c].items():
            x = np.array(list(d["seeds"].values()))
            d["mean"] = float(x.mean())
            d["n_seeds"] = len(x)
            if len(x) > 1:
                d["se"] = float(x.std(ddof=1) / np.sqrt(len(x)))
            # 95% interval of the success over seeds: pairs resampled, each bringing its outcomes for all seeds
            units = np.stack([res[(c, s, e)] for (c2, s, e2) in sorted(res) if c2 == c and e2 == e]).astype(float)
            bs = cluster_bootstrap(units, np.zeros_like(units), None, N_BOOT)
            d["ci_lo"], d["ci_hi"] = bs["ci_lo"], bs["ci_hi"]
    return out


def contrasts(res: dict) -> dict:
    out, primary = {}, []
    for desc, ev, A, B in contrasts_list():
        m = _paired(res, A, B)
        if m is None:
            continue
        a, b, d, (sa, sb) = m
        t = mcnemar_exact(a, b)
        bs = cluster_bootstrap(sa, sb, None, N_BOOT)
        row = {"a_pct": float(100 * a.mean()), "b_pct": float(100 * b.mean()), "diff_pts": float(np.mean(d)),
               "a_only": t["a_only"], "b_only": t["b_only"], "p_mcnemar": t["p"], "units": len(d),
               "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"], "p_boot": bs["p_boot"],
               "primary": (desc, ev) in PRIMARY}
        row["outcomes"] = int(a.size)
        if A[0] == "*":                                    # pooled over models: each model-seed run, and which pairs
            won = sa.astype(bool) & ~sb.astype(bool)       # A solved and B did not, per run and pair
            lost = sb.astype(bool) & ~sa.astype(bool)
            row["per_run_a_only"] = [int(x) for x in won.sum(1)]
            row["per_run_b_only"] = [int(x) for x in lost.sum(1)]
            row["distinct_pairs_a_only"] = int(won.any(0).sum())
            row["pairs_a_only_in_every_run"] = int(won.all(0).sum())
        if A[0] != "*":                                    # one configuration: also each seed separately
            seeds = sorted({s for (c, s, e) in res if c == A[0] and e == A[1]}
                           & {s for (c, s, e) in res if c == B[0] and e == B[1]})
            row["per_seed"] = {}
            for s in seeds:
                ra, rb = res[(A[0], s, A[1])], res[(B[0], s, B[1])]
                ts = mcnemar_exact(ra, rb)
                row["per_seed"][f"s{s}"] = {"a_pct": float(100 * ra.mean()), "b_pct": float(100 * rb.mean()),
                                           "a_only": ts["a_only"], "b_only": ts["b_only"], "p_mcnemar": ts["p"]}
            ps = [v["p_mcnemar"] for v in row["per_seed"].values()]
            row["per_seed_p_min"], row["per_seed_p_max"] = min(ps), max(ps)
        key = slug(f"{desc}__{ev}") if not desc.endswith(ev) else slug(desc)
        if key in out:
            raise ValueError(f"duplicate contrast key {key}")
        out[key] = row
        if row["primary"]:
            primary.append(key)
    for r in verification_did_rows(res):                   # H3: Wilcoxon on the change in selection gain
        key = slug(r["contrast"])
        out[key] = {"diff_pts": r["diff_pts"], "p_wilcoxon": r["p"], "units": r["units"], "primary": True}
        primary.append(key)
    ps = [out[k].get("p_boot", out[k].get("p_wilcoxon")) for k in primary]
    for k, ph in zip(primary, holm(ps), strict=True):
        out[k]["p_holm"] = ph
    return out


def timing(runs: Path, task: str, num_eval: int) -> dict:
    """Evaluation wall-clock minutes per configuration, seed and evaluation (as recorded by evaluate.py)."""
    from analyze_colab import label_of
    by_spec = {v: k for k, v in CONFIGS.items()}
    out: dict = {}
    for f in sorted(runs.glob(f"{task}__*.json")):
        r = json.loads(f.read_text())
        lab = label_of(r) if r.get("num_eval") == num_eval and r.get("spec") in by_spec else None
        if lab and r.get("eval_minutes") is not None:
            out.setdefault(by_spec[r["spec"]], {}).setdefault(lab, {})[f"s{r['seed']}"] = float(r["eval_minutes"])
    return out


def standard_protocol(runs: Path, task: str, num_eval: int) -> dict:
    """The authors' protocol (pairs drawn from all held-out pairs): how many of its pairs are already solved at the
    start (every result file records this as `trivial`; all files must agree), and what CEM on C1 solves among
    the pairs that are not."""
    trivial, cem = None, {}
    for f in sorted(runs.glob(f"{task}__*.json")):
        r = json.loads(f.read_text())
        if r.get("pairs") != "all" or r.get("num_eval") != num_eval or "trivial" not in r:
            continue
        if trivial is None:
            trivial = r["trivial"]
        elif r["trivial"] != trivial:
            raise ValueError(f"{f.name}: different standard-protocol pairs")
        if r["spec"] == "lewm" and r.get("controller", "cem") == "cem":
            t, s = np.asarray(r["trivial"], bool), np.asarray(r["successes"], bool)
            cem[f"s{r['seed']}"] = {"solved": int(s.sum()), "pairs": int(len(s)), "solved_not_trivial": int(s[~t].sum()),
                                    "not_trivial_pairs": int((~t).sum()), "solved_trivial": int(s[t].sum())}
    if trivial is None:
        return {}
    return {"pairs": len(trivial), "solved_at_start": int(sum(trivial)),
            "share_solved_at_start": float(np.mean(trivial)), "cem_c1": cem}


def main(runs: Path, task: str, profile: str) -> dict:
    res = load(runs.parent, PROFILES[profile]["num_eval"], CONFIGS, task, results_dir=runs)
    out = {"task": task, "profile": profile, "names": {c: NAMES[c] for c in CONFIGS},
           "models": len({(c, s) for (c, s, e) in res}), "evaluations": len(res),
           "success": success(res), "contrasts": contrasts(res),
           "eval_minutes": timing(runs, task, PROFILES[profile]["num_eval"]),
           "standard_protocol": standard_protocol(runs, task, PROFILES[profile]["num_eval"])}
    if task == "cube":
        diag, _ = diagnostics(runs.parent, CONFIGS, results_dir=runs)
        out["diagnostics"] = {d.pop("config"): {k.replace(".", "_"): v for k, v in d.items()} for d in diag}
        r2 = [(d["r2_arm"], d["r2_cube"]) for d in out["diagnostics"].values()]
        out["diagnostics_range"] = {"models": len(r2), "r2_arm_min": min(a for a, _ in r2),
                                    "r2_arm_max": max(a for a, _ in r2), "r2_cube_max": max(c for _, c in r2)}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--task", default="cube", choices=["cube", "tworoom"])
    ap.add_argument("--profile", default="free")
    a = ap.parse_args()
    print(json.dumps(main(a.runs, a.task, a.profile), indent=1))
