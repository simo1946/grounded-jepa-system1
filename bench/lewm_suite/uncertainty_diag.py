"""E4 (prereg_overnight.md): does System 1's uncertainty at the start of an evaluation pair
predict that the episode will fail?

Scores (higher = less confident), from cached latents of the pair's start and goal frames:
  goal_dist    latent squared distance start -> goal (difficulty baseline, no uncertainty)
  gauss_std    the Gaussian policy's own predicted std (mean over the chunk)
  pool_disagr  mean pairwise rms between 5 independently trained Gaussian policies' proposals
  flow_spread  rms spread of 16 flow-policy samples around their mean
Outcome: per-pair success of the run that uses that policy (k = 1), same 50 non-trivial pairs.
Reports AUROC for predicting failure (bootstrap 95% CI) and success on the most-confident half.

python uncertainty_diag.py
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import torch

from evaluate import pick_pairs
from hybrid_wm.control.latent_policy import policy_from_checkpoint
from latent_cache import cache_path
from models import HOME, Spec, TrainConfig
from tasks import get_task, trivial_mask

R = HOME / "results"
TAIL = "__nontrivial__off25_bud50_es42.json"


def eval_pairs(n: int = 50, seed: int = 42, offset: int = 25):
    import stable_worldmodel as swm
    task = get_task("cube", TrainConfig().img_size)
    ds = swm.data.HDF5Dataset("cube_eval", cache_dir=str(HOME), keys_to_cache=list(task.process_cols))
    excl = lambda e, s: trivial_mask(task, ds, e, s, offset)  # noqa: E731
    eps, starts = pick_pairs(np.asarray(ds.lengths), offset, n, seed, excl)
    return np.asarray(ds.offsets)[eps] + np.asarray(starts)


def auroc(score: np.ndarray, fail: np.ndarray) -> float:
    pos, neg = score[fail], score[~fail]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean()
    return float(gt)


def boot_ci(score, fail, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(len(score), size=len(score))
        a = auroc(score[i], fail[i])
        if np.isfinite(a):
            vals.append(a)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


@torch.no_grad()
def scores(spec: str, rows: np.ndarray) -> dict:
    sp = Spec.parse(spec)
    ck = HOME / "checkpoints" / f"cube/{sp.name()}/seed0"
    c = torch.load(cache_path(sp.name(), 0, "cube_eval"), map_location="cpu")
    z0, zg = c["z"][rows], c["z"][rows + 25]
    out = {"goal_dist": (zg - z0).square().sum(-1).numpy()}
    load = lambda f: policy_from_checkpoint(torch.load(ck / f, map_location="cpu")).eval()  # noqa: E731
    if (ck / "policy.pt").exists():
        out["gauss_std"] = load("policy.pt")(z0, zg)[1].mean((1, 2)).numpy()
    pool = ["policy_cached.pt"] + [f"policy_p{i}.pt" for i in range(1, 5)]
    if all((ck / f).exists() for f in pool):
        means = [load(f).sample(z0, zg, 1)[:, 0] for f in pool]
        d = [(a - b).square().mean((1, 2)).sqrt() for a, b in itertools.combinations(means, 2)]
        out["pool_disagr"] = torch.stack(d).mean(0).numpy()
    if (ck / "policy_flow6k.pt").exists():
        s = load("policy_flow6k.pt").sample(z0, zg, 16, generator=torch.Generator().manual_seed(0))
        out["flow_spread"] = (s - s.mean(1, keepdim=True)).square().mean((1, 2, 3)).sqrt().numpy()
    return out


def success(spec: str, ctrl: str):
    p = R / f"cube__{spec}__seed0__{ctrl}{TAIL}"
    return np.asarray(json.loads(p.read_text())["successes"], dtype=bool) if p.exists() else None


def main() -> dict:
    rows = eval_pairs()
    # (score, run whose failures it should predict)
    plan = {"lewm+idm+sa": {"gauss_std": "s1_k1", "pool_disagr": "s1_k1__pol-policy_cached",
                            "flow_spread": "s1_k1__pol-policy_flow6k"},
            "lewm+idm+sa+obj": {"gauss_std": "s1_k1", "flow_spread": "s1_k1__pol-policy_flow6k"}}
    res = {}
    print("| model | uncertainty score | run | failures | AUROC uncertainty [95% CI] | AUROC goal distance | "
          "success, most-confident half (uncertainty / goal distance) | all |")
    print("|---|---|---|---|---|---|---|---|")
    for spec, items in plan.items():
        sc = scores(spec, rows)
        for name, ctrl in items.items():
            ok = success(spec, ctrl)
            if ok is None or name not in sc:
                continue
            fail = ~ok
            u, gd = sc[name], sc["goal_dist"]
            half = len(u) // 2
            conf_u, conf_d = np.argsort(u)[:half], np.argsort(gd)[:half]
            r = {"auroc": auroc(u, fail), "ci": boot_ci(u, fail), "auroc_goal_dist": auroc(gd, fail),
                 "ci_goal_dist": boot_ci(gd, fail), "spearman_with_goal_dist": float(
                     np.corrcoef(np.argsort(np.argsort(u)), np.argsort(np.argsort(gd)))[0, 1]),
                 "success_conf_half": int(ok[conf_u].sum()), "success_conf_half_goal_dist": int(ok[conf_d].sum()),
                 "success_all": int(ok.sum()), "n": len(ok)}
            res[f"{spec}:{name}"] = r
            print(f"| {spec} | {name} | {ctrl} | {int(fail.sum())}/50 | {r['auroc']:.2f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}] | "
                  f"{r['auroc_goal_dist']:.2f} | {r['success_conf_half']}/{half} / {r['success_conf_half_goal_dist']}/{half} | "
                  f"{r['success_all']}/50 |")
    (R / "uncertainty_diag.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    main()
