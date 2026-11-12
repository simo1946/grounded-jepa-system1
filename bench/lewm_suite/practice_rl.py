"""Full-episode residual practice with a real-world success measure (prereg_critic.md, addendum E9).

Episodes match evaluation: a non-trivial training-set start, goal = the frame 25 steps later, 50-step
budget = two decisions of 25 steps, stop at success (cube within 4 cm of the goal). System 1 stays frozen;
a residual network corrects each decision's chunk. Decision 1 uses the start frame's cached latent; the
state after decision 1 is rendered and encoded for decision 2 (renderer verified against the dataset).

Two reward schemes:
  sparse  the episode's success (0/1), credited to both decisions (as in ResFiT);
  shaped  per decision: 10 x (distance before - distance after, metres) + 0.3 the first time the gripper
          grasps the cube + 1 at success; each decision is credited with its reward-to-go.
Updates: group-relative policy gradient (4 corrected episodes per start), PPO clipping, L2 on the correction.

python practice_rl.py --reward shaped --out policy_prac_shaped.pt
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
from models import HOME, Spec  # noqa: E402
from oracle_policy import action_scaler  # noqa: E402
from residual_rl import load_world_model  # noqa: E402

torch.set_flush_denormal(True)
SUCCESS_M, GRASP_BONUS, SUCCESS_BONUS, PROGRESS_SCALE = 0.04, 0.3, 1.0, 10.0
_W = {}


def _init_worker(data: str):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    from tasks import apply_render_profile, get_task
    task = get_task("cube", 64)
    apply_render_profile(task)
    env = gym.make(task.env_id, **task.eval_kwargs)
    env.reset(seed=0)
    _W["u"] = env.unwrapped
    with h5py.File(HOME / "datasets" / f"{data}.h5", "r") as f:
        _W.update(qpos=f["qpos"][:], qvel=f["qvel"][:], ctrl=f["control"][:], bp=f["privileged_block_0_pos"][:])


def _execute(goal_pos, raw):
    """Run raw actions until success; return (final distance, success, grasped, steps used)."""
    u, grasped = _W["u"], False
    d = np.inf
    for t, a in enumerate(raw):
        u.step(np.clip(a, -1, 1))
        i = u.compute_ob_info()
        cube, eff = i["privileged/block_0_pos"], i["proprio/effector_pos"]
        d = float(np.linalg.norm(cube - goal_pos))
        grasped |= bool(i["proprio/gripper_contact"][0] > 0.5 and np.linalg.norm(cube - eff) <= 0.02)
        if d <= SUCCESS_M:
            return d, True, grasped, t + 1
    return d, False, grasped, len(raw)


def _phase1(job):
    rows, goals, raw = job
    u, out = _W["u"], []
    for r, g, a in zip(rows, goals, raw):
        u.set_state(_W["qpos"][r], _W["qvel"][r])
        u._data.ctrl[:] = _W["ctrl"][r]
        d, ok, gr, _ = _execute(_W["bp"][g], a)
        img = None if ok else u.render().copy()
        out.append((d, ok, gr, img, u._data.qpos.copy(), u._data.qvel.copy(), u._data.ctrl.copy()))
    return out


def _phase2(job):
    states, goals, raw = job
    u, out = _W["u"], []
    for (q, v, c), g, a in zip(states, goals, raw):
        u.set_state(q, v)
        u._data.ctrl[:] = c
        d, ok, gr, _ = _execute(_W["bp"][g], a)
        out.append((d, ok, gr))
    return out


def _split(n, k):
    return [p for p in np.array_split(np.arange(n), k) if len(p)]


class Practice:
    def __init__(self, spec_str: str, base_file: str, seed: int = 0, workers: int = 2):
        self.spec = Spec.parse(spec_str)
        ck = HOME / "checkpoints" / f"cube/{self.spec.name()}/seed{seed}"
        self.ck, self.pk = ck, torch.load(ck / base_file, map_location="cpu")
        base = policy_from_checkpoint(self.pk).eval()
        cache = torch.load(cache_path(self.spec.name(), seed, "cube"), map_location="cpu")
        self.z, self.fs, self.A = cache["z"], cache["frameskip"], cache["action"].shape[1]
        self.mean, self.std = action_scaler(cache, "cube")
        self.res = ResidualChunkPolicy(base, self.z.shape[1], n_blocks=H, block_dim=self.fs * self.A, step_dim=self.A)
        self.lewm = load_world_model(self.spec, seed)
        import stable_pretraining as spt
        from torchvision.transforms import v2 as T
        self.img_t = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True),
                                T.Normalize(**spt.data.dataset_stats.ImageNet), T.Resize(size=64)])
        with h5py.File(HOME / "datasets" / "cube.h5", "r") as f:
            self.bp = torch.from_numpy(f["privileged_block_0_pos"][:])
        self.win = OfflineWindows(cache)
        self.workers = workers
        self.pool = mp.get_context("fork").Pool(workers, initializer=_init_worker, initargs=("cube",))

    def starts(self, n, gen):
        """Evaluation-like starts: non-trivial, goal 25 steps ahead."""
        rs = []
        while sum(len(x) for x in rs) < n:
            r = self.win.starts[torch.randint(len(self.win.starts), (4 * n,), generator=gen)]
            rs.append(r[(self.bp[r] - self.bp[r + H * self.fs]).norm(dim=-1) > SUCCESS_M])
        r = torch.cat(rs)[:n]
        return r, r + H * self.fs

    def to_raw(self, chunks):
        return chunks.reshape(len(chunks), H * self.fs, self.A).double().numpy() * self.std + self.mean

    @torch.no_grad()
    def encode(self, imgs):
        x = torch.stack([self.img_t(im) for im in imgs])[:, None]
        return self.lewm.encode({"pixels": x})["emb"][:, -1]

    def episodes(self, r, g, sample: bool, gen=None):
        """Run full episodes. Returns per-episode dict of tensors; corrections and their old log-probs."""
        n = len(r)
        z0, zg = self.z[r], self.z[g]
        with torch.no_grad():
            b1, mu1 = self.res.base_chunk(z0, zg), self.res.correction(z0, zg)
            sd = self.res.log_std.exp()
            c1 = mu1 + sd * torch.randn(mu1.shape, generator=gen) if sample else mu1
            lp1 = Normal(mu1, sd).log_prob(c1).sum(-1)
            ch1 = self.res.apply(b1, c1)
        raw1 = self.to_raw(ch1)
        parts = _split(n, 2 * self.workers)
        res1 = sum(self.pool.map(_phase1, [(r[p].numpy(), g[p].numpy(), raw1[p]) for p in parts]), [])
        d1 = torch.tensor([x[0] for x in res1]); ok1 = torch.tensor([x[1] for x in res1]); gr1 = torch.tensor([x[2] for x in res1])
        live = torch.nonzero(~ok1).squeeze(-1)
        d2, ok2, gr2 = d1.clone(), torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool)
        c2 = torch.zeros_like(c1); lp2 = torch.zeros(n); z1 = torch.zeros_like(z0)
        if len(live):
            z1[live] = self.encode([res1[i][3] for i in live.tolist()])
            with torch.no_grad():
                b2, mu2 = self.res.base_chunk(z1[live], zg[live]), self.res.correction(z1[live], zg[live])
                cc = mu2 + sd * torch.randn(mu2.shape, generator=gen) if sample else mu2
                lp2[live] = Normal(mu2, sd).log_prob(cc).sum(-1)
                c2[live] = cc
                raw2 = self.to_raw(self.res.apply(b2, cc))
            states = [(res1[i][4], res1[i][5], res1[i][6]) for i in live.tolist()]
            parts = _split(len(live), 2 * self.workers)
            res2 = sum(self.pool.map(_phase2, [([states[j] for j in p], g[live][p].numpy(), raw2[p]) for p in parts]), [])
            d2[live] = torch.tensor([x[0] for x in res2])
            ok2[live] = torch.tensor([x[1] for x in res2])
            gr2[live] = torch.tensor([x[2] for x in res2])
        d0 = (self.bp[r] - self.bp[g]).norm(dim=-1).float()
        return {"d0": d0, "d1": d1, "d2": d2, "ok1": ok1, "ok2": ok2, "gr1": gr1, "gr2": gr2, "success": ok1 | ok2,
                "z0": z0, "z1": z1, "zg": zg, "c1": c1, "c2": c2, "lp1": lp1, "lp2": lp2, "live": ~ok1}

    def close(self):
        self.pool.close()


def returns(ep: dict, reward: str):
    if reward == "sparse":
        s = ep["success"].float()
        return s, s
    r1 = PROGRESS_SCALE * (ep["d0"] - ep["d1"]) + GRASP_BONUS * ep["gr1"].float() + SUCCESS_BONUS * ep["ok1"].float()
    r2 = (PROGRESS_SCALE * (ep["d1"] - ep["d2"]) + GRASP_BONUS * (ep["gr2"] & ~ep["gr1"]).float()
          + SUCCESS_BONUS * ep["ok2"].float()) * ep["live"].float()
    return r1 + r2, r2


def group_adv(G: torch.Tensor, valid: torch.Tensor, group: int) -> torch.Tensor:
    Gv, m = G.view(-1, group), valid.view(-1, group).float()
    cnt = m.sum(1, keepdim=True).clamp(min=1)
    mean = (Gv * m).sum(1, keepdim=True) / cnt
    var = ((Gv - mean) ** 2 * m).sum(1, keepdim=True) / cnt
    adv = (Gv - mean) / (var.sqrt() + 1e-6) * m * (cnt >= 2).float()
    return adv.view(-1)


def main(reward: str, out: str, spec: str = "lewm+idm+sa+obj", base: str = "policy_gauss6k.pt", iters: int = 100,
         ctx: int = 96, group: int = 4, epochs: int = 4, lr: float = 3e-4, l2: float = 0.01, clip: float = 0.2,
         seed: int = 0, workers: int = 2, n_val: int = 192, eval_every: int = 20) -> dict:
    P = Practice(spec, base, seed, workers)
    res = P.res
    opt = torch.optim.Adam([p for n, p in res.named_parameters() if not n.startswith("base.")], lr=lr)
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    vr, vg = P.starts(n_val, torch.Generator().manual_seed(999))
    t0, log = time.time(), []
    base_val = float(P.episodes(vr, vg, sample=False)["success"].float().mean())
    print(json.dumps({"val_base_success": base_val}), flush=True)
    for it in range(1, iters + 1):
        r, gl = P.starts(ctx, g)
        r, gl = r.repeat_interleave(group), gl.repeat_interleave(group)
        ep = P.episodes(r, gl, sample=True, gen=g)
        G1, G2 = returns(ep, reward)
        a1 = group_adv(G1, torch.ones_like(G1, dtype=torch.bool), group)
        a2 = group_adv(G2, ep["live"], group)
        zs = torch.cat([ep["z0"], ep["z1"]]); zgs = torch.cat([ep["zg"], ep["zg"]])
        cs = torch.cat([ep["c1"], ep["c2"]]); olp = torch.cat([ep["lp1"], ep["lp2"]])
        adv = torch.cat([a1, a2]); keep = torch.cat([torch.ones_like(ep["live"]), ep["live"]])
        zs, zgs, cs, olp, adv = zs[keep], zgs[keep], cs[keep], olp[keep], adv[keep]
        for _ in range(epochs):
            mu = res.correction(zs, zgs)
            logp = Normal(mu, res.log_std.exp()).log_prob(cs).sum(-1)
            ratio = (logp - olp).exp()
            pg = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
            loss = pg + l2 * mu.square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(res.parameters(), 1.0)
            opt.step()
        row = {"it": it, "practice_success": float(ep["success"].float().mean()), "mean_return": float(G1.mean()),
               "min": round((time.time() - t0) / 60, 1)}
        if it % eval_every == 0 or it == iters:
            row["val_success"] = float(P.episodes(vr, vg, sample=False)["success"].float().mean())
            row["val_base_success"] = base_val
        log.append(row)
        if it % 5 == 0 or "val_success" in row:
            print(json.dumps(row), flush=True)
    P.close()
    meta = {"policy_type": "residual", "base": P.pk, "state_dict": {k: v.cpu() for k, v in res.state_dict().items()},
            "residual": {"latent_dim": P.z.shape[1], "n_blocks": H, "block_dim": P.fs * P.A, "step_dim": P.A,
                         "hidden": 256, "max_delta": 1.0},
            "latent_dim": P.pk["latent_dim"], "horizon": P.pk["horizon"], "action_dim": P.pk["action_dim"],
            "extra_dim": P.pk.get("extra_dim", 0), "tau_abs": P.pk["tau_abs"],
            "practice": {"kind": "full-episode", "reward": reward, "iters": iters, "ctx": ctx, "group": group,
                         "episodes": iters * ctx * group, "log": log}}
    torch.save(meta, P.ck / out)
    print(json.dumps({"saved": str(P.ck / out), "minutes": round((time.time() - t0) / 60, 1)}), flush=True)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", choices=["sparse", "shaped"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--n-val", type=int, default=192)
    a = ap.parse_args()
    main(a.reward, a.out, iters=a.iters, n_val=a.n_val)
