"""Improve System 1 with real-world outcomes (prereg_critic.md, addendum E7).

Reward-weighted fine-tuning (advantage-weighted regression): from the executed-candidate data of
gen_critic_data.py (each System 1 candidate's real final cube distance to the goal), fine-tune the
policy toward candidates that did better than the average candidate in the same situation:
    weight = exp(-(d - mean_state(d)) / beta), clipped.
Half of each batch is the original expert data, so the policy stays anchored. The control arm uses the
same data and steps with uniform weights, so any gain comes from the real-outcome signal.

python awr_policy.py --spec lewm+idm+sa+obj --init policy_gauss6k.pt --out policy_awr.pt
python awr_policy.py --spec lewm+idm+sa+obj --init policy_gauss6k.pt --out policy_awrctrl.pt --uniform
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from cached_policy import H, OfflineWindows
from critic import load_parts
from hybrid_wm.control.latent_policy import policy_from_checkpoint
from latent_cache import cache_path
from models import HOME, Spec

torch.set_flush_denormal(True)


def train(spec: str, init: str, out: str, steps: int = 2000, batch: int = 256, beta: float = 0.02,
          max_w: float = 20.0, uniform: bool = False, seed: int = 0, lr: float = 3e-4) -> dict:
    sp = Spec.parse(spec)
    ck = HOME / "checkpoints" / f"cube/{sp.name()}/seed{seed}"
    pk = torch.load(ck / init, map_location="cpu")
    pol = policy_from_checkpoint(pk)
    cache = torch.load(cache_path(sp.name(), seed, "cube"), map_location="cpu")
    z = cache["z"]
    d = load_parts(sp.name(), "cube")
    dist = d["dist"]                                                           # [N, K] metres
    w = torch.exp(-(dist - dist.mean(1, keepdim=True)) / beta).clamp(max=max_w)
    if uniform:
        w = torch.ones_like(w)
    w = w / w.mean()
    N, K = dist.shape
    win = OfflineWindows(cache)
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(pol.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    half = batch // 2
    log, t0 = [], time.time()
    for step in range(1, steps + 1):
        s = torch.randint(N, (half,), generator=g)                             # executed candidates
        j = torch.randint(K, (half,), generator=g)
        l_out = pol.loss(z[d["rows"][s]], z[d["goal_rows"][s]], d["cand"][s, j],
                         torch.ones(half, H, dtype=torch.bool), weight=w[s, j])
        b = win.sample(batch - half, g)                                        # expert anchor (hindsight BC)
        k = torch.randint(1, H + 1, (batch - half,), generator=g)
        rows = torch.arange(batch - half)
        mask = torch.arange(H)[None] < k[:, None]
        l_exp = pol.loss(b["z"][:, 0], b["z"][rows, k], b["action"].float(), mask)
        loss = l_out + l_exp
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps:
            log.append({"step": step, "loss_outcome": round(l_out.item(), 4), "loss_expert": round(l_exp.item(), 4),
                        "min": round((time.time() - t0) / 60, 2)})
            print(json.dumps(log[-1]), flush=True)
    meta = dict(pk)
    meta["state_dict"] = {k_: v.cpu() for k_, v in pol.state_dict().items()}
    meta["trained_awr"] = {"init": init, "steps": steps, "batch": batch, "beta": beta, "max_w": max_w,
                           "uniform": uniform, "candidates": N * K, "lr": lr, "log": log}
    torch.save(meta, ck / out)
    print(json.dumps({"saved": str(ck / out), "minutes": round((time.time() - t0) / 60, 2),
                      "weight_eff_frac": float((w.sum() ** 2 / (w ** 2).sum()) / w.numel())}), flush=True)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--uniform", action="store_true")
    a = ap.parse_args()
    train(a.spec, a.init, a.out, a.steps, beta=a.beta, uniform=a.uniform)
