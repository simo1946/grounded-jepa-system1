"""System 1 trained to output a short chunk (5 or 1 steps) instead of 25.

Same network family (ChunkPolicy: MLP on [z_now, z_goal, z_goal - z_now], Gaussian output), objective
(Gaussian NLL), goal distances (hindsight goal k blocks ahead, k uniform in 1..5, i.e. 5 to 25 steps), start
rows, optimiser and schedule as the 25-step System 1 (cached_policy.py, GPU-study recipe). Only the output
length changes: the target is the first ``chunk_steps`` actions after the current frame.

    python short_chunk.py --spec lewm --seed 0 --chunk-steps 5     # -> policy_c5.pt
    python short_chunk.py --spec lewm --seed 0 --chunk-steps 1     # -> policy_c1.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

torch.set_flush_denormal(True)

from cached_policy import H, OfflineWindows  # noqa: E402
from latent_cache import cache_path  # noqa: E402
from models import HOME, Spec  # noqa: E402


def short_batch(cache: dict, windows: OfflineWindows, n: int, chunk_steps: int, g: torch.Generator) -> dict:
    """Start rows as in the 25-step policy; goal k blocks ahead (k in 1..H); target = first chunk_steps actions."""
    fs = cache["frameskip"]
    r = windows.starts[torch.randint(len(windows.starts), (n,), generator=g)]
    k = torch.randint(1, H + 1, (n,), generator=g)
    rows = r[:, None] + torch.arange(chunk_steps)[None]                    # [n, c]
    act = cache["action"][rows].reshape(n, 1, chunk_steps * cache["action"].shape[1])
    return {"z_t": cache["z"][r], "z_g": cache["z"][r + k * fs], "action": act,
            "mask": torch.ones(n, 1, dtype=torch.bool), "k": k}


def train(spec_str: str, seed: int, chunk_steps: int, steps: int = 3000, batch: int = 256, lr: float = 1e-3,
          task: str = "cube") -> dict:
    from hybrid_wm.control.latent_policy import ChunkPolicy
    if chunk_steps < 1:
        raise ValueError("chunk_steps must be >= 1")
    spec = Spec.parse(spec_str)
    ckpt = HOME / "checkpoints" / f"{task}/{spec.name()}/seed{seed}"
    cache = torch.load(cache_path(spec.name(), seed, task), map_location="cpu")
    if chunk_steps > cache["frameskip"]:
        raise ValueError("the shortest goal is one block ahead; chunks longer than a block are the 25-step policy")
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    a_dim = cache["action"].shape[1]
    pol = ChunkPolicy(cache["z"].shape[1], 1, chunk_steps * a_dim)
    win = OfflineWindows(cache)
    opt = torch.optim.AdamW(pol.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    log, t0 = [], time.time()
    for step in range(1, steps + 1):
        b = short_batch(cache, win, batch, chunk_steps, g)
        loss = pol.loss(b["z_t"], b["z_g"], b["action"].float(), b["mask"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps:
            log.append({"step": step, "nll": round(loss.item(), 4), "min": round((time.time() - t0) / 60, 2)})
            print(json.dumps(log[-1]), flush=True)
    out = ckpt / f"policy_c{chunk_steps}.pt"
    meta = {"policy_type": "short_chunk", "state_dict": pol.state_dict(), "latent_dim": cache["z"].shape[1],
            "chunk_steps": chunk_steps, "step_action_dim": a_dim, "params": sum(p.numel() for p in pol.parameters()),
            "trained": {"steps": steps, "batch": batch, "lr": lr, "goal_blocks": [1, H]}, "log": log}
    tmp = out.with_suffix(".pt.tmp")
    torch.save(meta, tmp)
    tmp.replace(out)
    print(json.dumps({"saved": str(out), "params": meta["params"], "minutes": round((time.time() - t0) / 60, 2)}),
          flush=True)
    return meta


def short_chunk_act(path: Path, device: str | torch.device):
    """Controller for LatentStepSolver: the mean chunk [n, c, A] of a short-chunk System 1."""
    from hybrid_wm.control.latent_policy import ChunkPolicy
    pk = torch.load(path, map_location="cpu")
    c, a_dim = pk["chunk_steps"], pk["step_action_dim"]
    pol = ChunkPolicy(pk["latent_dim"], 1, c * a_dim)
    pol.load_state_dict(pk["state_dict"])
    pol = pol.to(device).eval()

    def act(z_now: torch.Tensor, z_goal: torch.Tensor, steps_done: int) -> torch.Tensor:
        mean, _ = pol(z_now, z_goal)
        return mean.reshape(len(z_now), c, a_dim)
    return act


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", default="cube")
    ap.add_argument("--chunk-steps", type=int, required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    train(a.spec, a.seed, a.chunk_steps, a.steps, a.batch, task=a.task)
