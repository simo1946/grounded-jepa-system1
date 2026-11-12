"""DAgger data collection: run the current System 1 on training start/goal pairs and record the
states it visits, for clean oracle labels (oracle_policy.DaggerBuffer).

The learner acts as deployed: it proposes a 5-block chunk and commits to it (receding 5). The
solver is called every block only to record the state at each block start; it replans when the
committed chunk is used up. Recorded per block: latent, oracle inputs (arm and cube state).
The goal pose comes from the dataset row of each pair.

python dagger.py --spec lewm+idm+sa --policy-file policy_oracle6k.pt --out round1.pt --episodes 100 --round 1
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from models import HOME, Spec, TrainConfig, build_model  # noqa: E402
from oracle import ORACLE_COLS  # noqa: E402
from sensors import info_key, readings_from_info  # noqa: E402
from tasks import apply_render_profile, get_task, trivial_mask  # noqa: E402

H = 5
SIM_COLS = ("qpos", "qvel", "control")


def _last(v) -> np.ndarray:
    v = v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
    v = v[:, -1] if v.ndim >= 3 else v
    return v.reshape(v.shape[0], -1).astype(np.float64)


class DaggerRecorder:
    def __init__(self, lewm, policy, state_cols, object_cols):
        self.lewm, self.policy, self.state_cols, self.object_cols = lewm, policy, state_cols, object_cols
        self.reset()

    def configure(self, *, action_space, n_envs: int, config) -> None:
        self._n, self._a = n_envs, int(np.prod(action_space.shape[1:])) * config.action_block

    @property
    def n_envs(self):
        return self._n

    @property
    def action_dim(self):
        return self._a

    @property
    def horizon(self):
        return H

    def __call__(self, *a, **k):
        return self.solve(*a, **k)

    def reset(self):
        self.goals, self.envs, self.plans = None, [], []

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action=None) -> dict:
        z = self.lewm.encode({"pixels": info_dict["pixels"].float()})["emb"][:, -1]
        zg = self.lewm.encode({"pixels": info_dict["goal"].float()})["emb"][:, -1]
        _, _, goal_obj = readings_from_info(info_dict, self.state_cols, self.object_cols)
        oin = {c: _last(info_dict[info_key(c)]) for c in ORACLE_COLS}
        for c in SIM_COLS:                                   # full simulator state, for simulated labels
            oin[c] = _last(info_dict[c])
        if self.goals is None:                               # first call of a chunk of envs: all live, in order
            self.goals = goal_obj.clone()
            self.envs = [{"z": [], "oracle_in": {c: [] for c in (*ORACLE_COLS, *SIM_COLS)}, "z_goal": zg[i].clone().float(),
                          "goal_obj": goal_obj[i].clone()} for i in range(len(z))]
            self.plans = [None] * len(z)
        env_ids = torch.cdist(goal_obj.double(), self.goals.double()).argmin(1).tolist()
        need = [row for row, e in enumerate(env_ids) if self.plans[e] is None or len(self.plans[e]) == 0]
        if need:
            idx = torch.tensor(need)
            chunk = self.policy.sample(z[idx], zg[idx], 1)[:, 0].float().cpu()          # deterministic [m, H, A]
            for j, row in enumerate(need):
                self.plans[env_ids[row]] = list(chunk[j])
        out = torch.zeros(len(z), H, self._a)
        for row, e in enumerate(env_ids):
            rec = self.envs[e]
            rec["z"].append(z[row].clone().float().cpu())
            for c in (*ORACLE_COLS, *SIM_COLS):
                rec["oracle_in"][c].append(oin[c][row])
            plan = self.plans[e]
            out[row, :len(plan)] = torch.stack(plan)
            plan.pop(0)                                      # this block is executed now (receding 1)
        return {"actions": out}


def collect(spec_str: str, seed: int, policy_file: str, n_episodes: int, round_seed: int) -> dict:
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from sklearn import preprocessing
    from torchvision.transforms import v2 as T

    from hybrid_wm.control.latent_policy import policy_from_checkpoint

    cfg = TrainConfig()
    task = get_task("cube", cfg.img_size)
    apply_render_profile(task)
    spec = Spec.parse(spec_str)
    ckpt = HOME / "checkpoints" / f"cube/{spec.name()}/seed{seed}"
    info = json.loads((ckpt / "train_info.json").read_text())
    wm = build_model(spec, task.action_dim, info["state_dim"], cfg)
    wm.load_state_dict(torch.load(ckpt / "weights.pt", map_location="cpu"))
    lewm = wm.lewm.eval().requires_grad_(False)
    lewm.interpolate_pos_encoding = True
    pol = policy_from_checkpoint(torch.load(ckpt / policy_file, map_location="cpu")).eval()

    ds = swm.data.HDF5Dataset("cube", cache_dir=str(HOME), keys_to_cache=list(task.process_cols))
    d = np.asarray(ds.get_col_data("action"))
    process = {"action": preprocessing.StandardScaler().fit(d[~np.isnan(d).any(axis=1)])}
    img_t = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True),
                       T.Normalize(**spt.data.dataset_stats.ImageNet), T.Resize(size=cfg.img_size)])
    rng = np.random.default_rng(20_000 + round_seed)
    lengths, offsets = np.asarray(ds.lengths), np.asarray(ds.offsets)
    pos = np.asarray(ds.get_col_data("privileged_block_0_pos"))
    yaw = np.asarray(ds.get_col_data("privileged_block_0_yaw"))
    eps, starts = [], []
    while len(eps) < n_episodes:                             # training pairs, non-trivial
        e = int(rng.integers(len(lengths)))
        s = int(rng.integers(lengths[e] - task.goal_offset))
        if not trivial_mask(task, ds, [e], [s], task.goal_offset)[0]:
            eps.append(e)
            starts.append(s)
    rec = DaggerRecorder(lewm, pol, task.state_cols, task.object_cols)
    policy = swm.policy.WorldModelPolicy(
        solver=rec, config=swm.PlanConfig(horizon=H, receding_horizon=1, action_block=cfg.frameskip),
        process=process, transform={"pixels": img_t, "goal": img_t})
    episodes, succ, chunk, t0 = [], [], 5, time.time()
    for lo in range(0, n_episodes, chunk):
        rec.reset()
        ep_c, st_c = eps[lo:lo + chunk], starts[lo:lo + chunk]
        world = swm.World(task.env_id, num_envs=len(ep_c), image_shape=(cfg.img_size, cfg.img_size),
                          max_episode_steps=2 * task.eval_budget, **task.eval_kwargs)
        world.set_policy(policy)
        m = world.evaluate(dataset=ds, episodes_idx=ep_c, start_steps=st_c, goal_offset=task.goal_offset,
                           eval_budget=task.eval_budget, callables=task.eval_callables)
        world.close()
        for i, (e, s) in enumerate(zip(ep_c, st_c)):
            r = rec.envs[i]
            grow = offsets[e] + s + task.goal_offset
            if not np.allclose(r["goal_obj"].numpy(), pos[grow], atol=1e-4):
                raise RuntimeError("env order does not match the pair order")
            episodes.append({"z": torch.stack(r["z"]), "z_goal": r["z_goal"],
                             "oracle_in": {c: np.stack(v) for c, v in r["oracle_in"].items()},
                             "goal_pos": pos[grow].astype(np.float64), "goal_yaw": yaw[grow].astype(np.float64),
                             "success": bool(m["episode_successes"][i])})
        succ += list(np.asarray(m["episode_successes"]).astype(bool))
    # clean labels: the oracle simulated closed-loop for H*fs steps from every visited block state
    from oracle_sim import OracleSim
    sim = OracleSim(task.env_id, **task.eval_kwargs)
    for e in episodes:
        S = e["oracle_in"]
        e["chunk_raw"] = np.stack([sim.chunk(S["qpos"][t], S["qvel"][t], S["control"][t], e["goal_pos"], e["goal_yaw"],
                                             H * cfg.frameskip) for t in range(len(S["qpos"]))]).astype(np.float32)
    return {"episodes": episodes, "pairs": list(zip(eps, starts)), "success_rate": float(np.mean(succ)),
            "blocks": int(sum(e["z"].shape[0] for e in episodes)), "minutes": round((time.time() - t0) / 60, 1),
            "policy_file": policy_file}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy-file", required=True)
    ap.add_argument("--out", required=True, help="file name in <checkpoint>/dagger/")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--round", type=int, required=True, help="round number (seeds the training pairs)")
    a = ap.parse_args()
    res = collect(a.spec, a.seed, a.policy_file, a.episodes, a.round)
    d = HOME / "checkpoints" / f"cube/{Spec.parse(a.spec).name()}/seed{a.seed}" / "dagger"
    d.mkdir(exist_ok=True)
    torch.save(res, d / a.out)
    print(json.dumps({k: v for k, v in res.items() if k not in ("episodes", "pairs")}), flush=True)
