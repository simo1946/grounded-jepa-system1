"""Residual practice with a success measure (prereg_critic.md, addendum E8).

System 1 stays frozen; a residual network learns a bounded per-block correction to its deterministic
chunk. Each practice attempt starts from a training-set state (true simulator state at frame r), executes
the corrected 25-step chunk in a scratch simulator, and is scored either by the real outcome
(-10 x final cube-goal distance in metres + 1 within 4 cm) or, as a control, by the world model's
imagined outcome (negative predicted latent distance to the goal). Updates: group-relative policy
gradient with PPO clipping (4 corrections per situation, advantages normalised within the group).

python residual_rl.py --spec lewm+idm+sa+obj --base policy_gauss6k.pt --reward real   --out policy_res_real.pt
python residual_rl.py --spec lewm+idm+sa+obj --base policy_gauss6k.pt --reward latent --out policy_res_latent.pt
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.distributions import Normal  # noqa: E402

from cached_policy import H, OfflineWindows  # noqa: E402
from hybrid_wm.control.latent_policy import ResidualChunkPolicy, policy_from_checkpoint  # noqa: E402
from latent_cache import cache_path  # noqa: E402
from models import HOME, Spec, TrainConfig, build_model  # noqa: E402
from oracle_policy import action_scaler  # noqa: E402

torch.set_flush_denormal(True)
SUCCESS_M = 0.04
_W = {}


def _init_worker(data: str):
    from oracle_sim import OracleSim
    from tasks import get_task
    task = get_task("cube", TrainConfig().img_size)
    with h5py.File(HOME / "datasets" / f"{data}.h5", "r") as f:
        _W.update(qpos=f["qpos"][:], qvel=f["qvel"][:], ctrl=f["control"][:], bp=f["privileged_block_0_pos"][:])
    _W["sim"] = OracleSim(task.env_id, **task.eval_kwargs)


def _run(job):
    rows, goals, raw = job
    out = np.zeros(len(rows), dtype=np.float32)
    for i, (r, g) in enumerate(zip(rows, goals, strict=True)):
        pos, _ = _W["sim"].execute(_W["qpos"][r], _W["qvel"][r], _W["ctrl"][r], raw[i])
        out[i] = np.linalg.norm(pos - _W["bp"][g])
    return out


def real_distances(pool, rows: np.ndarray, goals: np.ndarray, raw: np.ndarray, n_workers: int) -> np.ndarray:
    parts = np.array_split(np.arange(len(rows)), n_workers * 2)
    res = pool.map(_run, [(rows[p], goals[p], raw[p]) for p in parts])
    return np.concatenate(res)


def load_world_model(spec: Spec, seed: int):
    task_dim = 5
    ck = HOME / "checkpoints" / f"cube/{spec.name()}/seed{seed}"
    info = json.loads((ck / "train_info.json").read_text())
    wm = build_model(spec, task_dim, info["state_dim"], TrainConfig())
    wm.load_state_dict(torch.load(ck / "weights.pt", map_location="cpu"))
    lewm = wm.lewm.eval().requires_grad_(False)
    return lewm


@torch.no_grad()
def imagined_reward(lewm, z_t: torch.Tensor, z_g: torch.Tensor, chunks: torch.Tensor) -> torch.Tensor:
    """Negative squared latent distance between the world model's predicted final latent and the goal."""
    info = {"pixels": torch.zeros(len(z_t), 1, 1, 1), "emb": z_t[:, None, None, :]}
    pred = lewm.rollout(info, chunks[:, None])["predicted_emb"][:, 0, -1]
    return -(pred - z_g).square().sum(-1)


def main(spec_str: str, base_file: str, reward: str, out: str, iters: int = 120, ctx: int = 128, group: int = 4,
         epochs: int = 4, lr: float = 3e-4, l2: float = 0.01, clip: float = 0.2, seed: int = 0, workers: int = 2,
         eval_every: int = 20, n_val: int = 256) -> dict:
    spec = Spec.parse(spec_str)
    ck = HOME / "checkpoints" / f"cube/{spec.name()}/seed{seed}"
    pk = torch.load(ck / base_file, map_location="cpu")
    base = policy_from_checkpoint(pk).eval()
    cache = torch.load(cache_path(spec.name(), seed, "cube"), map_location="cpu")
    z, fs, A = cache["z"], cache["frameskip"], cache["action"].shape[1]
    mean, std = action_scaler(cache, "cube")
    res = ResidualChunkPolicy(base, z.shape[1], n_blocks=H, block_dim=fs * A, step_dim=A)
    opt = torch.optim.Adam([p for n, p in res.named_parameters() if not n.startswith("base.")], lr=lr)
    lewm = load_world_model(spec, seed) if reward == "latent" else None
    win = OfflineWindows(cache)
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)

    with h5py.File(HOME / "datasets" / "cube.h5", "r") as f:
        bp = torch.from_numpy(f["privileged_block_0_pos"][:])

    def contexts(n, gen):
        """Non-trivial situations only (the cube must move > 4 cm to reach the goal), as in evaluation."""
        rs, gs = [], []
        while sum(len(x) for x in rs) < n:
            r = win.starts[torch.randint(len(win.starts), (4 * n,), generator=gen)]
            gl = r + torch.randint(1, H + 1, (4 * n,), generator=gen) * fs
            keep = (bp[r] - bp[gl]).norm(dim=-1) > SUCCESS_M
            rs.append(r[keep])
            gs.append(gl[keep])
        return torch.cat(rs)[:n], torch.cat(gs)[:n]

    def to_raw(chunks):                                                          # [N, H, fs*A] -> [N, H*fs, A]
        return chunks.reshape(len(chunks), H * fs, A).double().numpy() * std + mean

    vr, vg = contexts(n_val, torch.Generator().manual_seed(999))                 # fixed validation situations
    pool = mp.get_context("fork").Pool(workers, initializer=_init_worker, initargs=("cube",))
    t0, log = time.time(), []
    with torch.no_grad():
        vbase = res.base_chunk(z[vr], z[vg])
    d_base = real_distances(pool, vr.numpy(), vg.numpy(), to_raw(vbase), workers)
    print(json.dumps({"val_base_success": float((d_base <= SUCCESS_M).mean()),
                      "val_base_dist": float(d_base.mean())}), flush=True)
    for it in range(1, iters + 1):
        r, gl = contexts(ctx, g)
        zt, zg = z[r], z[gl]
        with torch.no_grad():
            b = res.base_chunk(zt, zg)
            mu = res.correction(zt, zg)
            sd = res.log_std.exp()
            corr = mu[:, None] + sd * torch.randn(ctx, group, mu.shape[1], generator=g)
            old_logp = Normal(mu[:, None], sd).log_prob(corr).sum(-1)                     # [ctx, group]
            chunks = torch.stack([res.apply(b, corr[:, j]) for j in range(group)], 1)     # [ctx, group, H, fs*A]
        flat = chunks.reshape(ctx * group, H, fs * A)
        rr, gg = r.repeat_interleave(group).numpy(), gl.repeat_interleave(group).numpy()
        d = real_distances(pool, rr, gg, to_raw(flat), workers).reshape(ctx, group)
        if reward == "real":
            R = torch.from_numpy(-10.0 * d + (d <= SUCCESS_M)).float()
        else:
            R = imagined_reward(lewm, zt.repeat_interleave(group, 0), zg.repeat_interleave(group, 0), flat).view(ctx, group)
        adv = (R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True) + 1e-6)
        for _ in range(epochs):
            mu_n = res.correction(zt, zg)
            logp = Normal(mu_n[:, None], res.log_std.exp()).log_prob(corr).sum(-1)
            ratio = (logp - old_logp).exp()
            pg = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
            loss = pg + l2 * mu_n.square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(res.parameters(), 1.0)
            opt.step()
        row = {"it": it, "practice_success": float((d <= SUCCESS_M).mean()), "practice_dist": float(d.mean()),
               "corr_abs": float(mu.abs().mean()), "std": float(res.log_std.exp().mean()),
               "min": round((time.time() - t0) / 60, 1)}
        if it % eval_every == 0 or it == iters:
            with torch.no_grad():
                vc = res.apply(vbase, res.correction(z[vr], z[vg]))
            dv = real_distances(pool, vr.numpy(), vg.numpy(), to_raw(vc), workers)
            row.update(val_success=float((dv <= SUCCESS_M).mean()), val_dist=float(dv.mean()),
                       val_base_success=float((d_base <= SUCCESS_M).mean()))
        log.append(row)
        if it % 5 == 0 or "val_success" in row:
            print(json.dumps(row), flush=True)
    pool.close()
    meta = {"policy_type": "residual", "base": pk, "state_dict": {k: v.cpu() for k, v in res.state_dict().items()},
            "residual": {"latent_dim": z.shape[1], "n_blocks": H, "block_dim": fs * A, "step_dim": A, "hidden": 256,
                         "max_delta": 1.0},
            "latent_dim": pk["latent_dim"], "horizon": pk["horizon"], "action_dim": pk["action_dim"],
            "extra_dim": pk.get("extra_dim", 0), "tau_abs": pk["tau_abs"],
            "practice": {"reward": reward, "iters": iters, "ctx": ctx, "group": group, "epochs": epochs, "lr": lr,
                         "l2": l2, "clip": clip, "attempts": iters * ctx * group, "log": log}}
    torch.save(meta, ck / out)
    print(json.dumps({"saved": str(ck / out), "minutes": round((time.time() - t0) / 60, 1)}), flush=True)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--reward", choices=["real", "latent"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    main(a.spec, a.base, a.reward, a.out, iters=a.iters, workers=a.workers)
