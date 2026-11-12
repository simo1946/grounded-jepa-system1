"""Outcome critic for System 1 candidates (prereg_critic.md).

f(z_t, z_goal, chunk) -> (predicted log final cube-goal distance, success logit), trained only on real
(simulated) outcomes of executed candidates (gen_critic_data.py). At run time it sees latents and the
candidate chunk, never the simulator state.

python critic.py --spec lewm+idm+sa+obj             # train on *_cube_p*.pt, evaluate offline on *_cube_eval_p*.pt
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from latent_cache import cache_path
from models import HOME, Spec

SUCCESS_M = 0.04          # the task's success tolerance (cube within 4 cm of the goal)
EPS_M = 0.01              # log-distance offset (1 cm)


class OutcomeCritic(nn.Module):
    def __init__(self, latent_dim: int, chunk_dim: int, hidden: int = 512):
        super().__init__()
        self.chunk_dim = chunk_dim
        self.net = nn.Sequential(
            nn.Linear(3 * latent_dim + chunk_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, 2))

    def forward(self, z: torch.Tensor, zg: torch.Tensor, cand: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """z, zg [B, D]; cand [B, K, ...] -> (log distance [B, K], success logit [B, K])."""
        B, K = cand.shape[:2]
        c = cand.reshape(B, K, -1)
        ctx = torch.cat([z, zg, zg - z], -1).unsqueeze(1).expand(B, K, -1)
        out = self.net(torch.cat([ctx, c], -1))
        return out[..., 0], out[..., 1]

    @torch.no_grad()
    def score(self, z, zg, cand) -> torch.Tensor:
        """Lower is better: predicted log distance to the goal."""
        return self(z, zg, cand)[0]


def load_parts(spec_name: str, data: str) -> dict:
    tag = spec_name.replace("+", "_")
    parts = sorted((HOME / "datasets").glob(f"critic_{tag}__{data}_p*.pt"))
    if not parts:
        raise FileNotFoundError(f"no critic data for {spec_name} / {data}")
    d = [torch.load(p, map_location="cpu", weights_only=False) for p in parts]
    return {k: torch.cat([x[k] for x in d]) for k in ("rows", "goal_rows", "cand", "dist")}


def spearman_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ra, rb = a.argsort(1).argsort(1).astype(float), b.argsort(1).argsort(1).astype(float)
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean(1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    return np.where(den > 0, (ra * rb).sum(1) / np.where(den > 0, den, 1), np.nan)


def offline_eval(critic: OutcomeCritic, z: torch.Tensor, d: dict) -> dict:
    with torch.no_grad():
        pred = critic.score(z[d["rows"]], z[d["goal_rows"]], d["cand"]).numpy()
    real = d["dist"].numpy()
    # states where the choice matters: real outcomes differ in success across candidates
    ok = real <= SUCCESS_M
    informative = ok.any(1) & ~ok.all(1)
    pick = pred.argmin(1)
    n = len(real)
    sp = spearman_rows(pred, real)
    res = {"states": n, "informative_states": int(informative.sum()),
           "spearman_mean": float(np.nanmean(sp)),
           "spearman_informative": float(np.nanmean(sp[informative])) if informative.any() else float("nan")}
    for name, idx in (("mean_candidate", np.zeros(n, int)), ("critic_pick", pick), ("oracle_pick", real.argmin(1))):
        res[f"success_{name}"] = float(ok[np.arange(n), idx].mean())
        res[f"success_{name}_informative"] = float(ok[np.arange(n), idx][informative].mean()) if informative.any() else float("nan")
    res["success_random_candidate"] = float(ok[:, 1:].mean())
    return res


def train(spec: str, seed: int = 0, steps: int = 4000, batch: int = 1024, lr: float = 1e-3, out: str = "critic.pt") -> dict:
    sp = Spec.parse(spec)
    tr = load_parts(sp.name(), "cube")
    ztr = torch.load(cache_path(sp.name(), seed, "cube"), map_location="cpu")["z"]
    n_states, K = tr["dist"].shape
    chunk_dim = tr["cand"][0, 0].numel()
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_states, generator=g)
    n_val = max(1, n_states // 10)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    torch.manual_seed(seed)
    critic = OutcomeCritic(ztr.shape[1], chunk_dim)
    opt = torch.optim.AdamW(critic.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    logd = torch.log(tr["dist"] + EPS_M)
    succ = (tr["dist"] <= SUCCESS_M).float()
    t0, log = time.time(), []
    for step in range(1, steps + 1):
        s = tr_idx[torch.randint(len(tr_idx), (batch,), generator=g)]
        j = torch.randint(K, (batch,), generator=g)
        pl, ps = critic(ztr[tr["rows"][s]], ztr[tr["goal_rows"][s]], tr["cand"][s, j].unsqueeze(1))
        loss = F.mse_loss(pl[:, 0], logd[s, j]) + F.binary_cross_entropy_with_logits(ps[:, 0], succ[s, j])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps:
            critic.eval()
            v = offline_eval(critic, ztr, {k: tr[k][val_idx] for k in tr})
            critic.train()
            log.append({"step": step, "loss": round(loss.item(), 4), "val_spearman": round(v["spearman_mean"], 3),
                        "val_success_critic": round(v["success_critic_pick"], 3), "min": round((time.time() - t0) / 60, 2)})
            print(json.dumps(log[-1]), flush=True)
    critic.eval()
    res = {"train_states": len(tr_idx), "log": log}
    try:
        ev = load_parts(sp.name(), "cube_eval")
        zev = torch.load(cache_path(sp.name(), seed, "cube_eval"), map_location="cpu")["z"]
        res["heldout_eval_set"] = offline_eval(critic, zev, ev)
    except FileNotFoundError:
        pass
    ck = HOME / "checkpoints" / f"cube/{sp.name()}/seed{seed}"
    torch.save({"state_dict": critic.state_dict(), "latent_dim": ztr.shape[1], "chunk_dim": chunk_dim, "meta": res}, ck / out)
    (HOME / "results" / f"critic__{sp.name()}__seed{seed}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "log"}), flush=True)
    return res


def load_critic(path) -> OutcomeCritic:
    pk = torch.load(path, map_location="cpu", weights_only=False)
    c = OutcomeCritic(pk["latent_dim"], pk["chunk_dim"])
    c.load_state_dict(pk["state_dict"])
    return c.eval()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4000)
    a = ap.parse_args()
    train(a.spec, a.seed, a.steps)
