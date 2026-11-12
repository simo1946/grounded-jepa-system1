"""Train a System 1 policy from the latent cache (fast path of train_policy.py).

Same objective as train_policy.py (hindsight relabelling, goal = frame k in
1..H blocks ahead, Gaussian NLL on the executed action blocks, later blocks
masked), but on precomputed latents, with optional
  * simulated sensors (sensors.SensorSpec) as extra policy inputs, and
  * on-policy experience buffers (learn_loop.py): the robot's own episodes,
    relabelled with the goals they actually reached; episodes that reached
    the commanded goal also contribute (start, commanded goal) examples
    weighted by ``success_weight``.

    python cached_policy.py --spec lewm+idm+sa --seed 0 --sensors proprio+obj --out policy_s-proprio_obj.pt
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

torch.set_flush_denormal(True)

from latent_cache import cache_path  # noqa: E402
from models import HOME, Spec  # noqa: E402
from sensors import SensorEncoder, SensorSpec  # noqa: E402

H = 5


def block_displacement_median(cache: dict) -> float:
    """Median squared latent displacement over one block (frameskip steps) within
    episodes: the verify-then-act gate's absolute threshold (tau_abs)."""
    fs, d = cache["frameskip"], []
    for off, length in zip(cache["offsets"].tolist(), cache["lengths"].tolist()):
        z = cache["z"][off:off + length]
        if len(z) > fs:
            d.append((z[fs:] - z[:-fs]).square().sum(-1))
    return float(torch.cat(d).median())


class OfflineWindows:
    """Windows of H+1 frames (frameskip apart) inside one episode of the cache."""

    def __init__(self, cache: dict):
        self.c, fs = cache, cache["frameskip"]
        self.fs = fs
        starts = []
        for off, length in zip(cache["offsets"].tolist(), cache["lengths"].tolist()):
            last = length - 1 - H * fs                       # frame rows up to start + H*fs must exist
            if last >= 0:
                starts.append(torch.arange(off, off + last + 1))
        self.starts = torch.cat(starts)
        self.frame = torch.arange(H + 1) * fs                # frame offsets
        self.act = (torch.arange(H)[:, None] * fs + torch.arange(fs)[None]).reshape(-1)   # action row offsets

    def sample(self, n: int, g: torch.Generator) -> dict:
        r = self.starts[torch.randint(len(self.starts), (n,), generator=g)]
        fr = r[:, None] + self.frame[None]                   # [n, H+1]
        ar = r[:, None] + self.act[None]                     # [n, H*fs]
        A = self.c["action"].shape[1]
        return {"z": self.c["z"][fr], "action": self.c["action"][ar].reshape(n, H, self.fs * A),
                "proprio": self.c["proprio"][fr], "object": self.c["object"][fr]}


class Experience:
    """On-policy episodes (learn_loop.finish_episodes): z/proprio/object over F
    decision frames, A action blocks (A = F if the commanded goal was reached
    by the last block, else F - 1), z_goal, goal_obj, success."""

    def __init__(self, episodes: list[dict]):
        self.eps = [e for e in episodes if e["action"].shape[0] >= 1 and (e["success"] or e["z"].shape[0] >= 2)]

    def __len__(self):
        return len(self.eps)

    def sample(self, n: int, g: torch.Generator, success_weight: float) -> dict:
        D = self.eps[0]["z"].shape[1]
        P, A = self.eps[0]["proprio"].shape[1], self.eps[0]["action"].shape[1]
        z_t, z_g = torch.zeros(n, D), torch.zeros(n, D)
        act = torch.full((n, H, A), float("nan"))
        mask = torch.zeros(n, H, dtype=torch.bool)
        p_t, o_t, o_g = torch.zeros(n, P), torch.zeros(n, 3), torch.zeros(n, 3)
        w = torch.ones(n)
        idx = torch.randint(len(self.eps), (n,), generator=g)
        for i, e in enumerate(idx.tolist()):
            ep = self.eps[e]
            F, A = ep["z"].shape[0], ep["action"].shape[0]      # frames, actions (A = F on success)
            use_true = bool(ep["success"]) and float(torch.rand(1, generator=g)) < 0.5
            if use_true or F < 2:     # (start, commanded goal), reached after A blocks (F < 2 only for successes)
                t = int(torch.randint(A, (1,), generator=g))
                k, goal_z, goal_o = A - t, ep["z_goal"], ep["goal_obj"]
                w[i] = success_weight
            else:                                             # hindsight: a frame actually reached later
                t = int(torch.randint(F - 1, (1,), generator=g))
                k = int(torch.randint(1, min(H, F - 1 - t) + 1, (1,), generator=g))
                goal_z, goal_o = ep["z"][t + k], ep["object"][t + k]
            z_t[i], z_g[i], p_t[i], o_t[i], o_g[i] = ep["z"][t], goal_z, ep["proprio"][t], ep["object"][t], goal_o
            n_blk = min(H, A - t)
            act[i, :n_blk] = ep["action"][t:t + n_blk]
            mask[i, :min(k, n_blk)] = True
        return {"z_t": z_t, "z_g": z_g, "action": act, "mask": mask, "proprio": p_t, "obj": o_t,
                "goal_obj": o_g, "weight": w}


def train(spec_str: str, seed: int, sensors: str, out: str, steps: int, init: str | None = None,
          experience: list[str] | None = None, mix: float = 0.5, success_weight: float = 2.0,
          lr: float = 1e-3, batch: int = 64, success_only: bool = False, device: str = "cpu",
          policy_seed: int | None = None, policy_type: str = "gauss", task: str = "cube") -> dict:
    from hybrid_wm.control.latent_policy import ChunkPolicy, FlowChunkPolicy

    spec = Spec.parse(spec_str)
    ckpt = HOME / "checkpoints" / f"{task}/{spec.name()}/seed{seed}"
    cache = torch.load(cache_path(spec.name(), seed, task), map_location="cpu")
    sspec = SensorSpec.parse(sensors)
    stats = SensorEncoder.fit_stats(cache["proprio"], cache["object"])
    pseed = seed if policy_seed is None else policy_seed     # policy init/sampling seed (default: model seed)
    g = torch.Generator().manual_seed(pseed)
    enc = SensorEncoder(sspec, stats, generator=g)
    act_dim = cache["frameskip"] * cache["action"].shape[1]
    extra_dim = sspec.dim(cache["proprio"].shape[1], cache["object"].shape[1]) if sspec.groups() else 0
    if policy_seed is not None:                              # explicit seed: reproducible, distinct init
        torch.manual_seed(policy_seed)
    if policy_type == "flow":
        pol = FlowChunkPolicy(cache["z"].shape[1], H, act_dim, extra_dim=extra_dim)
    elif policy_type == "gauss":
        pol = ChunkPolicy(cache["z"].shape[1], H, act_dim, extra_dim=extra_dim)
    else:
        raise ValueError(f"unknown policy type {policy_type!r}")
    # gate threshold: the original policy's value when it exists (CPU study), else measured on the cache
    tau_abs = (torch.load(ckpt / "policy.pt", map_location="cpu")["tau_abs"] if (ckpt / "policy.pt").exists()
               else block_displacement_median(cache))
    if init:
        pol.load_state_dict(torch.load(ckpt / init, map_location="cpu")["state_dict"])
    pol = pol.to(device)
    off = OfflineWindows(cache)
    exp = Experience([e for f in (experience or []) for e in torch.load(f, map_location="cpu")["episodes"]
                      if e["success"] or not success_only])
    n_on = int(round(batch * mix)) if len(exp) else 0
    opt = torch.optim.AdamW(pol.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    torch.manual_seed(seed if policy_seed is None else policy_seed)
    log, t0 = [], time.time()
    for step in range(1, steps + 1):
        b = off.sample(batch - n_on, g)
        k = torch.randint(1, H + 1, (batch - n_on,), generator=g)
        rows = torch.arange(batch - n_on)
        z_t, z_g = b["z"][:, 0], b["z"][rows, k]
        mask = torch.arange(H)[None] < k[:, None]
        act, w = b["action"], torch.ones(batch - n_on)
        p_t, o_t, o_g = b["proprio"][:, 0], b["object"][:, 0], b["object"][rows, k]
        if n_on:
            e = exp.sample(n_on, g, success_weight)
            z_t, z_g = torch.cat([z_t, e["z_t"]]), torch.cat([z_g, e["z_g"]])
            act, mask, w = torch.cat([act, e["action"]]), torch.cat([mask, e["mask"]]), torch.cat([w, e["weight"]])
            p_t, o_t, o_g = torch.cat([p_t, e["proprio"]]), torch.cat([o_t, e["obj"]]), torch.cat([o_g, e["goal_obj"]])
        extra = enc(p_t, o_t, o_g)
        to = lambda x: None if x is None else x.to(device, non_blocking=True)  # noqa: E731
        loss = pol.loss(to(z_t), to(z_g), to(act.float()), to(mask), extra=to(extra), weight=to(w))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 250 == 0 or step == steps:
            log.append({"step": step, "nll": round(loss.item(), 4), "min": round((time.time() - t0) / 60, 2)})
            print(json.dumps(log[-1]), flush=True)
    meta = {"state_dict": {k: v.cpu() for k, v in pol.state_dict().items()}, "latent_dim": cache["z"].shape[1],
            "policy_type": policy_type, "flow_steps": getattr(pol, "n_steps", None),
            "horizon": H, "action_dim": act_dim, "extra_dim": extra_dim, "tau_abs": tau_abs, "sensors": sspec.to_dict(), "sensor_stats": stats,
            "state_cols": cache["state_cols"], "object_cols": cache["object_cols"],
            "trained": {"steps": steps, "init": init, "experience": experience or [], "mix": mix,
                        "success_weight": success_weight, "lr": lr, "on_policy_episodes": len(exp),
                        "success_only": success_only, "batch": batch, "device": device, "policy_seed": pseed},
            "log": log}
    tmp = ckpt / (out + ".tmp")                    # atomic
    torch.save(meta, tmp)
    tmp.replace(ckpt / out)
    print(json.dumps({"saved": str(ckpt / out), "minutes": round((time.time() - t0) / 60, 2)}), flush=True)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--task", default="cube", choices=["cube", "tworoom"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sensors", default="none")
    ap.add_argument("--out", required=True, help="file name inside the checkpoint dir")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--init", default=None, help="policy file (in the checkpoint dir) to fine-tune")
    ap.add_argument("--experience", nargs="*", default=None, help="on-policy episode files (learn_loop.py)")
    ap.add_argument("--mix", type=float, default=0.5)
    ap.add_argument("--success-weight", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--success-only", action="store_true", help="keep only episodes that reached their goal")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--policy-seed", type=int, default=None, help="policy init/sampling seed (default: --seed)")
    ap.add_argument("--policy-type", choices=["gauss", "flow"], default="gauss",
                    help="System 1 head: Gaussian chunk policy or flow-matching chunk policy")
    a = ap.parse_args()
    train(a.spec, a.seed, a.sensors, a.out, a.steps, a.init, a.experience, a.mix, a.success_weight, a.lr,
          batch=a.batch, success_only=a.success_only, device=a.device, policy_seed=a.policy_seed,
          policy_type=a.policy_type, task=a.task)
