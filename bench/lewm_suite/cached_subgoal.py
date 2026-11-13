"""Two-level System 1: train the slow subgoal network on cached latents and
bundle it with a fast (low-level) policy file.

The low-level policy is unchanged; at decision time the solver feeds it
z_sub = SubgoalNet(z_now, z_goal) instead of z_goal (system12.GatedSolver,
``subgoal``). Validation on the held-out cache reports the subgoal error
against two trivial predictors (stay at z_t; jump to z_goal).

    python cached_subgoal.py --spec lewm+idm+sa --seed 0 --base policy_cached.pt --out policy_hier.pt
    python cached_subgoal.py --spec lewm+idm+sa --seed 0 --base policy_s1.pt --out policy_hier_gated.pt --gate auto

``--gate auto`` uses the subgoal only for far goals: ||z_goal - z_now||^2 above the
median squared latent displacement over the policy horizon (H blocks), measured
on training data.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

torch.set_flush_denormal(True)

from latent_cache import cache_path  # noqa: E402
from models import HOME, Spec  # noqa: E402

H = 5


def windows(cache: dict, n_blocks: int) -> torch.Tensor:
    fs, out = cache["frameskip"], []
    for off, length in zip(cache["offsets"].tolist(), cache["lengths"].tolist(), strict=True):
        last = length - 1 - n_blocks * fs
        if last >= 0:
            out.append(torch.arange(off, off + last + 1))
    return torch.cat(out)


def batch(cache, starts, n, g):
    fs = cache["frameskip"]
    r = starts[torch.randint(len(starts), (n,), generator=g)]
    return cache["z"][r[:, None] + torch.arange(2 * H + 1)[None] * fs]          # [n, 2H+1, D]


@torch.no_grad()
def horizon_displacement_median(cache: dict) -> float:
    k, d = H * cache["frameskip"], []
    for off, length in zip(cache["offsets"].tolist(), cache["lengths"].tolist(), strict=True):
        z = cache["z"][off:off + length]
        if len(z) > k:
            d.append((z[k:] - z[:-k]).square().sum(-1))
    return float(torch.cat(d).median())


@torch.no_grad()
def validate(net, cache, g, n=4096, device="cpu") -> dict:
    from hybrid_wm.control.latent_policy import subgoal_batch
    emb = batch(cache, windows(cache, 2 * H), n, g)
    z_t, z_g, z_s = subgoal_batch(emb, H, g)
    z_t, z_g, z_s = z_t.to(device), z_g.to(device), z_s.to(device)
    err = lambda p: float((p - z_s).square().sum(-1).mean())  # noqa: E731
    far = (z_g - z_s).abs().sum(-1) > 0                                         # goals beyond the horizon
    errf = lambda p: float((p - z_s)[far].square().sum(-1).mean())  # noqa: E731
    pred = net(z_t, z_g)
    return {"all": {"subgoal_net": err(pred), "stay_z_t": err(z_t), "goal_z_g": err(z_g)},
            "far_goals": {"subgoal_net": errf(pred), "stay_z_t": errf(z_t), "goal_z_g": errf(z_g)}}


def main(spec_str: str, seed: int, base: str, out: str, steps: int, gate: str = "none",
         device: str = "cpu", task: str = "cube") -> None:
    from hybrid_wm.control.latent_policy import SubgoalNet, subgoal_batch

    spec = Spec.parse(spec_str)
    ckpt = HOME / "checkpoints" / f"{task}/{spec.name()}/seed{seed}"
    tr = torch.load(cache_path(spec.name(), seed, task), map_location="cpu")
    va_path = cache_path(spec.name(), seed, f"{task}_eval")
    va = torch.load(va_path, map_location="cpu") if va_path.exists() else None
    g = torch.Generator().manual_seed(seed)
    net = SubgoalNet(tr["z"].shape[1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    starts, log, t0 = windows(tr, 2 * H), [], time.time()
    for step in range(1, steps + 1):
        loss = net.loss(*(x.to(device) for x in subgoal_batch(batch(tr, starts, 128, g), H, g)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps:
            log.append({"step": step, "mse": round(loss.item(), 3), "min": round((time.time() - t0) / 60, 2)})
            print(json.dumps(log[-1]), flush=True)
    val = validate(net.eval(), va, torch.Generator().manual_seed(1), device=device) if va is not None else None
    meta = torch.load(ckpt / base, map_location="cpu")
    meta.update({"subgoal_state_dict": {k: v.cpu() for k, v in net.state_dict().items()}, "subgoal_horizon": H,
                 "subgoal_val": val, "subgoal_base_policy": base, "subgoal_log": log})
    if gate == "auto":
        meta["subgoal_gate"] = horizon_displacement_median(tr)
        meta["subgoal_gate_note"] = "median squared latent displacement over H blocks (training cache)"
    elif gate != "none":
        meta["subgoal_gate"] = float(gate)
    tmp = ckpt / (out + ".tmp")                    # atomic
    torch.save(meta, tmp)
    tmp.replace(ckpt / out)
    print(json.dumps({"saved": str(ckpt / out), "val": val}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", default="policy_cached.pt")
    ap.add_argument("--out", default="policy_hier.pt")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--gate", default="none", help="'none', 'auto', or a number")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--task", default="cube", choices=["cube", "tworoom"])
    a = ap.parse_args()
    main(a.spec, a.seed, a.base, a.out, a.steps, a.gate, a.device, a.task)
