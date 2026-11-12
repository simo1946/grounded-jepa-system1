"""A GC-IDM-style step-wise controller, and execution of any controller for a fixed number of steps.

GC-IDM (Latent Geometry Beyond Search, arXiv 2605.08732) maps (z_now, z_goal, remaining horizon) to one
action and is called again after every environment step. Its code is not released; this is a
reimplementation from the paper's description (Sec. 4.2 and App. F): a 3-layer MLP of width 512 with
LayerNorm, GELU and dropout 0.1 on [z_now, z_goal]; the remaining horizon, divided by H_max = 50, is
encoded with 64 sinusoidal features and a 2-layer MLP and modulates the backbone output through
zero-initialised AdaLN-Zero; trained with MSE on (z_t, z_{t+h}, a_t), h uniform in 1..H_max.

    python gcidm.py --spec lewm --seed 0                 # trains gcidm.pt in the checkpoint dir
    python gcidm.py --spec lewm --seed 0 --no-horizon    # trains gcidm_noh.pt
"""
from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any

import numpy as np
import torch
from torch import nn

torch.set_flush_denormal(True)

H_MAX = 50


def sinusoidal(x: torch.Tensor, dim: int = 64) -> torch.Tensor:
    """x [B] in [0, 1] -> [B, dim] sinusoidal features."""
    half = dim // 2
    freqs = torch.exp(-math.log(10_000.0) * torch.arange(half, device=x.device, dtype=x.dtype) / half)
    ang = x[:, None] * 1_000.0 * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class GCIDM(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden: int = 512, horizon: bool = True,
                 dropout: float = 0.1, emb_dim: int = 64):
        super().__init__()
        self.use_horizon, self.emb_dim = horizon, emb_dim

        def block(i: int) -> list:
            return [nn.Linear(i, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
        self.backbone = nn.Sequential(*block(2 * latent_dim), *block(hidden), *block(hidden))
        if horizon:
            self.h_mlp = nn.Sequential(nn.Linear(emb_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
            self.ada = nn.Linear(hidden, 2 * hidden)
            nn.init.zeros_(self.ada.weight)                # AdaLN-Zero: the horizon starts with no effect
            nn.init.zeros_(self.ada.bias)
        self.head = nn.Linear(hidden, action_dim)

    def forward(self, z_t: torch.Tensor, z_goal: torch.Tensor, h: torch.Tensor | None = None) -> torch.Tensor:
        """z_t, z_goal [B, D]; h [B] remaining steps (1..H_MAX) -> action [B, A] (z-scored)."""
        x = self.backbone(torch.cat([z_t, z_goal], dim=-1))
        if self.use_horizon:
            if h is None:
                raise ValueError("this controller needs the remaining horizon")
            c = self.h_mlp(sinusoidal(h.to(x.dtype) / H_MAX, self.emb_dim))
            gamma, beta = self.ada(c).chunk(2, dim=-1)
            x = x * (1 + gamma) + beta
        return self.head(x)


def training_pairs(cache: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows t with at least one later frame in the same episode, and the largest horizon allowed per row."""
    rows, hmax = [], []
    for off, length in zip(cache["offsets"].tolist(), cache["lengths"].tolist()):
        t = torch.arange(length - 1)
        rows.append(off + t)
        hmax.append((length - 1 - t).clamp(max=H_MAX))
    return torch.cat(rows), torch.cat(hmax)


def train(spec_str: str, seed: int, horizon: bool, epochs: int = 50, batch: int = 1024, lr: float = 1e-3,
          task: str = "cube") -> dict:
    from latent_cache import cache_path
    from models import HOME, Spec

    spec = Spec.parse(spec_str)
    ckpt = HOME / "checkpoints" / f"{task}/{spec.name()}/seed{seed}"
    cache = torch.load(cache_path(spec.name(), seed, task), map_location="cpu")
    z, action = cache["z"].float(), cache["action"].float()
    rows, hmax = training_pairs(cache)
    ok = torch.isfinite(action[rows]).all(-1)
    rows, hmax = rows[ok], hmax[ok]
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = GCIDM(z.shape[1], action.shape[1], horizon=horizon)
    steps_per_epoch = math.ceil(len(rows) / batch)
    steps = epochs * steps_per_epoch
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr / 100)
    log, t0, step = [], time.time(), 0
    net.train()
    for ep in range(epochs):
        perm = rows[torch.randperm(len(rows), generator=g)]
        hm = hmax[torch.searchsorted(rows, perm)]
        for i in range(0, len(perm), batch):
            r, m = perm[i:i + batch], hm[i:i + batch]
            h = (torch.rand(len(r), generator=g) * m).floor().long() + 1      # uniform in 1..m
            pred = net(z[r], z[r + h], h.float())
            loss = (pred - action[r]).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
        log.append({"epoch": ep + 1, "mse": round(loss.item(), 4), "min": round((time.time() - t0) / 60, 2)})
        if (ep + 1) % 10 == 0:
            print(json.dumps(log[-1]), flush=True)
    out = "gcidm.pt" if horizon else "gcidm_noh.pt"
    meta = {"policy_type": "gcidm", "state_dict": net.state_dict(), "latent_dim": z.shape[1],
            "action_dim": action.shape[1], "horizon": horizon, "h_max": H_MAX,
            "params": sum(p.numel() for p in net.parameters()),
            "trained": {"epochs": epochs, "batch": batch, "lr": lr, "steps": steps, "pairs": len(rows)}, "log": log}
    tmp = ckpt / (out + ".tmp")
    torch.save(meta, tmp)
    tmp.replace(ckpt / out)
    print(json.dumps({"saved": str(ckpt / out), "params": meta["params"], "steps": steps,
                      "minutes": round((time.time() - t0) / 60, 2)}), flush=True)
    return meta


def gcidm_from_checkpoint(pk: dict) -> GCIDM:
    net = GCIDM(pk["latent_dim"], pk["action_dim"], horizon=pk["horizon"])
    net.load_state_dict(pk["state_dict"])
    return net.eval()


class LatentStepSolver:
    """Solver-protocol wrapper for controllers that read only the encoder: it encodes the current and goal
    frames and asks ``act(z_now, z_goal, steps_done)`` for a short chunk [n, c, A] of standardised actions.
    The chunk fills the first c steps of the returned plan (the rest repeats its last action); StepPolicy
    then executes exactly c steps before asking again."""

    def __init__(self, cem, lewm, act):
        self.cem, self.lewm, self.act = cem, lewm, act
        self.step = 0                                       # steps already executed in the current episodes

    def configure(self, *, action_space, n_envs: int, config: Any) -> None:
        self.cem.configure(action_space=action_space, n_envs=n_envs, config=config)
        self.block = config.action_block

    @property
    def action_dim(self) -> int:
        return self.cem.action_dim

    @property
    def n_envs(self) -> int:
        return self.cem.n_envs

    @property
    def horizon(self) -> int:
        return self.cem.horizon

    def __call__(self, *args, **kwargs) -> dict:
        return self.solve(*args, **kwargs)

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        p0 = next(self.lewm.parameters())
        z_now = self.lewm.encode({"pixels": info_dict["pixels"].to(p0.device, p0.dtype)})["emb"][:, -1]
        z_goal = self.lewm.encode({"pixels": info_dict["goal"].to(p0.device, p0.dtype)})["emb"][:, -1]
        chunk = self.act(z_now.float(), z_goal.float(), self.step)            # [n, c, A]
        n, c, a_dim = chunk.shape
        total = self.horizon * self.block
        if c > total:
            raise ValueError("chunk longer than the plan")
        steps = torch.cat([chunk, chunk[:, -1:].expand(n, total - c, a_dim)], dim=1)
        return {"actions": steps.reshape(n, self.horizon, self.block * a_dim).detach().cpu().clone()}


def gcidm_act(net: GCIDM, budget: int):
    """Our GC-IDM-style controller: one action; remaining steps h = budget - t + 1 (t 1-based), clamped."""
    def act(z_now: torch.Tensor, z_goal: torch.Tensor, steps_done: int) -> torch.Tensor:
        t = steps_done + 1
        h = torch.full((len(z_now),), float(min(max(budget - t + 1, 1), H_MAX)), device=z_now.device)
        return net(z_now, z_goal, h)[:, None]
    return act


def step_policy(base_cls):
    """Subclass of WorldModelPolicy that executes only the first ``exec_steps`` steps of every plan and then
    decides again from a new camera frame. It also tells a LatentStepSolver how many steps have been executed."""

    class StepPolicy(base_cls):
        def __init__(self, *a, exec_steps: int, **k):
            super().__init__(*a, **k)
            if exec_steps < 1:
                raise ValueError("exec_steps must be >= 1")
            self.exec_steps = exec_steps

        def set_env(self, env: Any) -> None:
            super().set_env(env)
            n = getattr(env, "num_envs", 1)
            self._since = np.zeros(n, dtype=np.int64)
            self._t = 0
            if hasattr(self.solver, "step"):
                self.solver.step = 0

        def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
            if self.flatten_receding_horizon < self.exec_steps:
                raise ValueError("the plan is shorter than exec_steps")
            if info_dict.get("_needs_flush") is not None:
                raise RuntimeError("auto-reset is not expected in dataset evaluation")
            for i, buf in enumerate(self._action_buffer):
                if self._since[i] >= self.exec_steps:
                    buf.clear()                              # force a new decision at this step
            if hasattr(self.solver, "step"):
                self.solver.step = self._t
            before = [len(b) for b in self._action_buffer]
            out = super().get_action(info_dict, **kwargs)
            for i, buf in enumerate(self._action_buffer):
                if before[i] == 0 and len(buf) == self.flatten_receding_horizon - 1:
                    self._since[i] = 1                       # replanned at this step and executed its first action
                else:
                    self._since[i] += 1
            self._t += 1
            return out

    return StepPolicy


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", default="cube")
    ap.add_argument("--no-horizon", action="store_true")
    ap.add_argument("--epochs", type=int, default=50)
    a = ap.parse_args()
    train(a.spec, a.seed, not a.no_horizon, epochs=a.epochs, task=a.task)
