"""How well does each goal-distance signal rank true temporal distance?

On held-out clips (eval dataset) of 11 blocks, for every in-clip pair (0, k),
k = 0..10, compare
  * latent MSE  ||z_0 - z_k||^2   (the authors' GoalMSE planning cost)
  * reachability critic E[steps]  (train_reach.py)
with the true k: Spearman rank correlation, and per-k means. Cross-clip pairs
(different episodes) are reported separately: a good cost should rank them
as "far".

    python progress_diag.py --spec lewm+idm+sa --seed 0
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

torch.set_flush_denormal(True)

from models import HOME, Spec, TrainConfig, build_model  # noqa: E402
from tasks import get_task  # noqa: E402


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr  # tie-aware (true k has many ties when pooled)
    return float(spearmanr(x, y).statistic)


@torch.inference_mode()
def main(spec_str: str, seed: int, n_clips: int, data: str) -> dict:
    import stable_worldmodel as swm
    from stable_pretraining import data as dt

    from hybrid_wm.control.reachability import ReachabilityCritic

    torch.set_num_threads(1)
    cfg = TrainConfig()
    task = get_task("cube", cfg.img_size)
    spec = Spec.parse(spec_str)
    ckpt = HOME / "checkpoints" / f"cube/{spec.name()}/seed{seed}"
    info = json.loads((ckpt / "train_info.json").read_text())
    wm = build_model(spec, task.action_dim, info["state_dim"], cfg)
    wm.load_state_dict(torch.load(ckpt / "weights.pt", map_location="cpu"))
    lewm = wm.lewm.eval()
    rk = torch.load(ckpt / "reach.pt", map_location="cpu")
    crit = ReachabilityCritic(rk["latent_dim"], max_steps=rk["max_steps"])
    crit.load_state_dict(rk["state_dict"])
    crit.eval()

    ds = swm.data.HDF5Dataset(data, cache_dir=str(HOME), num_steps=11, frameskip=cfg.frameskip,
                              keys_to_load=["pixels"])
    ds.transform = dt.transforms.Compose(
        dt.transforms.ToImage(**dt.dataset_stats.ImageNet, source="pixels", target="pixels"),
        dt.transforms.Resize(cfg.img_size, source="pixels", target="pixels"))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), size=min(n_clips, len(ds)), replace=False)
    embs = []
    for lo in range(0, len(idx), 16):
        px = torch.stack([ds[int(i)]["pixels"] for i in idx[lo:lo + 16]])
        embs.append(lewm.encode({"pixels": px})["emb"])
    emb = torch.cat(embs)                                            # [N, 11, D]
    ks = np.arange(11)
    mse = torch.stack([(emb[:, 0] - emb[:, k]).square().sum(-1) for k in ks], 1).numpy()   # [N, 11]
    rch = torch.stack([crit(emb[:, 0], emb[:, k]) for k in ks], 1).numpy()
    true = np.broadcast_to(ks, mse.shape)
    perm = np.roll(np.arange(len(emb)), 1)                            # cross-clip pairs
    mse_x = (emb[:, 0] - emb[perm, -1]).square().sum(-1).numpy()
    rch_x = crit(emb[:, 0], emb[perm, -1]).numpy()

    # within-clip ranking (pooled) and per-clip ranking (does cost increase along one trajectory?)
    per_clip = lambda c: float(np.nanmean([spearman(c[i], ks) for i in range(len(c))]))  # noqa: E731
    out = {"spec": spec_str, "seed": seed, "clips": int(len(emb)),
           "spearman_pooled": {"latent_mse": spearman(mse.ravel(), true.ravel()),
                               "reach": spearman(rch.ravel(), true.ravel())},
           "spearman_per_clip_mean": {"latent_mse": per_clip(mse), "reach": per_clip(rch)},
           "mean_by_k": {"latent_mse": mse.mean(0).round(2).tolist(), "reach": rch.mean(0).round(2).tolist()},
           "cross_clip_mean": {"latent_mse": float(mse_x.mean()), "reach": float(rch_x.mean())},
           # fraction of cross-clip pairs ranked farther than the in-clip k=10 pair of the same start
           "cross_farther_than_k10": {"latent_mse": float((mse_x > mse[:, 10]).mean()),
                                      "reach": float((rch_x > rch[:, 10]).mean())}}
    print(json.dumps(out, indent=2))
    (HOME / "results" / f"progress_diag__{spec.name()}__seed{seed}.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="lewm+idm+sa")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clips", type=int, default=256)
    ap.add_argument("--data", default="cube_eval")
    a = ap.parse_args()
    main(a.spec, a.seed, a.clips, a.data)
