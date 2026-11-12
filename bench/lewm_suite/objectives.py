"""Task-aware planning cost: weigh what the task is judged on.

The authors' GoalMSE compares whole latents; on Cube the latent goal distance
is explained by the arm (R^2 ~ 0.5) and hardly by the cube (R^2 ~ 0.03), so
the planner matches the arm pose and ignores the object (probe_diag.py).
``ProbeGoalCost`` adds an object term read out by a linear probe:

    cost = ||P(z_pred) - P(z_goal)||^2 / tol^2  +  ||z_pred - z_goal||^2 / scale

P is a ridge probe latent -> object position fitted on TRAINING frames (object
labels at training time only, like the 'obj' grounding; planning sees pixels
only). tol = the task's success tolerance (0.04 m); scale = median squared
latent displacement over the planning horizon on training data, so both terms
are O(1) for a typical move. Both constants are fixed a priori, not tuned on
evaluation episodes.
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn

H = 5


class ProbeGoalCost(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, mean: torch.Tensor, tol: float, scale: float):
        super().__init__()
        self.register_buffer("w", weight)
        self.register_buffer("b", bias)
        self.register_buffer("m", mean)
        self.tol, self.scale = tol, scale

    def readout(self, z: torch.Tensor) -> torch.Tensor:
        return (z.to(self.w.dtype) - self.m) @ self.w + self.b

    def distance(self, z: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        obj = (self.readout(z) - self.readout(z_goal)).square().sum(-1) / self.tol ** 2
        lat = (z - z_goal).square().sum(-1).to(obj.dtype) / self.scale
        return obj + lat

    def forward(self, info_dict: dict) -> torch.Tensor:           # stable-worldmodel Objective: -> [B, S]
        pred = info_dict["predicted_emb"][:, :, -1]
        goal = info_dict["goal_emb"][:, -1].unsqueeze(1)
        return self.distance(pred, goal.expand_as(pred)).to(pred.dtype)


@torch.inference_mode()
def fit_probe(lewm, task, cfg, home, n_frames: int = 4000, seed: int = 0, lam: float = 1.0) -> dict:
    """Ridge probe latent -> object position, and the latent scale, on training frames."""
    import stable_worldmodel as swm
    from stable_pretraining import data as dt

    col = task.object_cols[0]
    ds = swm.data.HDF5Dataset("cube", cache_dir=str(home), num_steps=H + 1, frameskip=cfg.frameskip,
                              keys_to_load=["pixels", col], keys_to_cache=["action"])
    ds.transform = dt.transforms.Compose(
        dt.transforms.ToImage(**dt.dataset_stats.ImageNet, source="pixels", target="pixels"),
        dt.transforms.Resize(cfg.img_size, source="pixels", target="pixels"))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n_frames // 2, len(ds)), replace=False)
    zs, ys, disp = [], [], []
    for lo in range(0, len(idx), 64):
        items = [ds[int(i)] for i in idx[lo:lo + 64]]
        px = torch.stack([it["pixels"][[0, H]] for it in items])               # first and last frame
        z = lewm.encode({"pixels": px.to(next(lewm.parameters()).device)})["emb"].double().cpu()   # [b, 2, D]
        y = torch.stack([it[col][[0, H]] for it in items]).double()
        zs.append(z.reshape(-1, z.shape[-1]))
        ys.append(y.reshape(-1, y.shape[-1]))
        disp.append((z[:, 1] - z[:, 0]).square().sum(-1))
    z, y = torch.cat(zs), torch.cat(ys)
    mz, my = z.mean(0), y.mean(0)
    zc = z - mz
    w = torch.linalg.solve(zc.T @ zc + lam * torch.eye(z.shape[1], dtype=z.dtype), zc.T @ (y - my))
    resid = ((zc @ w + my) - y).norm(dim=-1)
    return {"weight": w.float(), "bias": my.float(), "mean": mz.float(),
            "scale": float(torch.cat(disp).median()), "train_err_m": float(resid.mean()), "n": int(len(z))}


def load_or_fit(lewm, task, cfg, home, ckpt_dir, tol: float = 0.04) -> ProbeGoalCost:
    path = ckpt_dir / "probe_cost.pt"
    if path.exists():
        p = torch.load(path, map_location="cpu")
    else:
        p = fit_probe(lewm, task, cfg, home)
        torch.save(p, path)
        print(json.dumps({"probe_cost_fitted": str(path), "train_err_m": p["train_err_m"], "scale": p["scale"]}),
              flush=True)
    return ProbeGoalCost(p["weight"], p["bias"], p["mean"], tol, p["scale"])
