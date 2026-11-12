"""Is the world model's action-conditioned rollout good enough to plan with?

On held-out clips (eval dataset) of 1 context frame + 5 action blocks
(planning setup: history_len 1, horizon 5, frameskip 5):

  * prediction error of the rollout under the TRUE actions, per horizon k,
    against the "nothing moves" baseline ||z_0 - z_k||^2;
  * action identifiability: roll out the true chunk and 15 chunks taken from
    other clips, score each by GoalMSE against the real z_5 (exactly the CEM
    objective with the real outcome as goal). Report how often the true chunk
    ranks first (chance 1/16) and its mean rank.

If the true actions are not identifiable from the model's predictions, no
optimiser over that model's rollouts (CEM, gradient, gating) can plan well.

    python rollout_diag.py --spec lewm+idm+sa --seed 0
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

torch.set_flush_denormal(True)

from models import HOME, Spec, TrainConfig, build_model  # noqa: E402
from tasks import get_task  # noqa: E402

H = 5          # planning horizon in blocks
N_ALT = 15     # alternative chunks per clip


@torch.inference_mode()
def main(spec_str: str, seed: int, n_clips: int, data: str) -> dict:
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from stable_pretraining import data as dt
    from stable_worldmodel.data import column_normalizer

    torch.set_num_threads(1)
    cfg = TrainConfig()
    task = get_task("cube", cfg.img_size)
    spec = Spec.parse(spec_str)
    ckpt = HOME / "checkpoints" / f"cube/{spec.name()}/seed{seed}"
    info = json.loads((ckpt / "train_info.json").read_text())
    wm = build_model(spec, task.action_dim, info["state_dim"], cfg)
    wm.load_state_dict(torch.load(ckpt / "weights.pt", map_location="cpu"))
    lewm = wm.lewm.eval()

    # action statistics from the TRAINING set (as in training and planning)
    train_ds = swm.data.HDF5Dataset("cube", cache_dir=str(HOME), num_steps=H + 1, frameskip=cfg.frameskip,
                                    keys_to_load=["pixels", "action"], keys_to_cache=["action"])
    ds = swm.data.HDF5Dataset(data, cache_dir=str(HOME), num_steps=H + 1, frameskip=cfg.frameskip,
                              keys_to_load=["pixels", "action"], keys_to_cache=["action"])
    img = dt.transforms.Compose(dt.transforms.ToImage(**dt.dataset_stats.ImageNet, source="pixels", target="pixels"),
                                dt.transforms.Resize(cfg.img_size, source="pixels", target="pixels"))
    ds.transform = spt.data.transforms.Compose(img, column_normalizer(train_ds, "action", "action"))

    rng = np.random.default_rng(0)
    order = rng.permutation(len(ds))
    px, act = [], []
    for i in order:
        s = ds[int(i)]
        a = s["action"][:H].float()
        if not torch.isfinite(a).all():
            continue
        px.append(s["pixels"])
        act.append(a)
        if len(px) == n_clips:
            break
    px, act = torch.stack(px), torch.stack(act)                      # [N, 6, C, h, w], [N, 5, 25]
    N = len(px)
    z = torch.cat([lewm.encode({"pixels": px[i:i + 32]})["emb"] for i in range(0, N, 32)])   # [N, 6, D]

    # candidates: true chunk first, then chunks from N_ALT other clips
    alt = torch.stack([torch.roll(torch.arange(N), j) for j in range(1, N_ALT + 1)], 1)       # [N, 15]
    cands = torch.cat([act[:, None], act[alt]], 1)                                            # [N, 16, 5, 25]
    preds = []
    for i in range(0, N, 8):
        info_d = {"pixels": px[i:i + 8, None, :1]}                  # [b, 1, 1, C, h, w]; encoded once
        out = lewm.rollout(info_d, cands[i:i + 8])
        preds.append(out["predicted_emb"])                          # [b, 16, 1 + 5, D]
    pred = torch.cat(preds)

    k = torch.arange(1, H + 1)
    err_true = (pred[:, 0, k] - z[:, k]).square().sum(-1)            # [N, 5]
    err_copy = (z[:, :1] - z[:, k]).square().sum(-1)                 # [N, 5]
    cost = (pred[:, :, -1] - z[:, None, -1]).square().sum(-1)        # [N, 16] GoalMSE vs real z_5
    rank = (cost < cost[:, :1]).sum(1)                               # 0 = true chunk best
    # same test one block ahead (k = 1): is the immediate effect of the action identifiable?
    cost1 = (pred[:, :, 1] - z[:, None, 1]).square().sum(-1)
    rank1 = (cost1 < cost1[:, :1]).sum(1)
    # fine-grained identifiability: true chunk vs 15 perturbed copies (noise sigma, normalised action units).
    # This is the regime of selecting among System 1 samples or CEM's final iterations.
    fine = {}
    gen = torch.Generator().manual_seed(1)
    for sigma in (0.1, 0.3, 1.0):
        noisy = act[:, None] + sigma * torch.randn((N, N_ALT, *act.shape[1:]), generator=gen)
        cf = torch.cat([act[:, None], noisy], 1)
        pf = torch.cat([lewm.rollout({"pixels": px[i:i + 8, None, :1]}, cf[i:i + 8])["predicted_emb"]
                        for i in range(0, N, 8)])
        c = (pf[:, :, -1] - z[:, None, -1]).square().sum(-1)
        r_ = (c < c[:, :1]).sum(1)
        fine[f"sigma={sigma:g}"] = {"true_top1": float((r_ == 0).float().mean()), "mean_rank": float(r_.float().mean())}
    res = {"spec": spec_str, "seed": seed, "clips": N, "candidates": N_ALT + 1, "perturbed_identifiability": fine,
           "pred_err_true_actions_by_k": err_true.mean(0).round(decimals=2).tolist(),
           "copy_baseline_err_by_k": err_copy.mean(0).round(decimals=2).tolist(),
           "pred_err_over_copy_by_k": (err_true.mean(0) / err_copy.mean(0)).round(decimals=3).tolist(),
           "true_chunk_top1_at_k5": float((rank == 0).float().mean()),
           "true_chunk_mean_rank_at_k5": float(rank.float().mean()),
           "true_chunk_top1_at_k1": float((rank1 == 0).float().mean()),
           "true_chunk_mean_rank_at_k1": float(rank1.float().mean()),
           "chance_top1": 1 / (N_ALT + 1), "chance_mean_rank": N_ALT / 2}
    print(json.dumps(res, indent=2))
    (HOME / "results" / f"rollout_diag__{spec.name()}__seed{seed}.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="lewm+idm+sa")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clips", type=int, default=256)
    ap.add_argument("--data", default="cube_eval")
    a = ap.parse_args()
    main(a.spec, a.seed, a.clips, a.data)
