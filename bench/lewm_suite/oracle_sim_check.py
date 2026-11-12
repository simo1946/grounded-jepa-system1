"""Ceiling check for clean oracle labels: run the oracle itself on the 50 non-trivial evaluation
pairs, toward each pair's goal cube pose, recomputing its action every `hold` steps (1 = closed
loop per step, as it generated the data; 5 = one action held per 5-step block, which is what
block-level labels teach the policy).

python oracle_sim_check.py --hold 1 5
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

from models import HOME, TrainConfig  # noqa: E402
from oracle import ORACLE_COLS, cube_oracle  # noqa: E402
from sensors import info_key  # noqa: E402
from tasks import apply_render_profile, get_task  # noqa: E402


def _last(v):
    v = np.asarray(v)
    v = v[:, -1] if v.ndim >= 3 else v
    return v.reshape(v.shape[0], -1).astype(np.float64)


def make_chunk_policy(task, chunk: int = 25):
    """Executes clean oracle chunks open-loop: every `chunk` steps, simulate the oracle closed-loop
    from the current state in a scratch env and replay those actions (what a perfect chunk policy does)."""
    import stable_worldmodel as swm

    from oracle_sim import OracleSim

    class ChunkOracle(swm.policy.BasePolicy):
        def __init__(self):
            super().__init__()
            self.type, self.sim, self.plan = "chunk_oracle", OracleSim(task.env_id, **task.eval_kwargs), None

        def get_action(self, info_dict, **kwargs):
            step = _last(info_dict["step_idx"]).reshape(-1).astype(int)
            qpos, qvel, ctrl = _last(info_dict["qpos"]), _last(info_dict["qvel"]), _last(info_dict["control"])
            tp, ty = _last(info_dict["goal_privileged_block_0_pos"]), _last(info_dict["goal_privileged_block_0_yaw"])
            n = len(step)
            if self.plan is None or len(self.plan) != n:
                self.plan = [None] * n
            act = np.zeros((n, 5), dtype=np.float32)
            for i in range(n):
                if step[i] % chunk == 0 or self.plan[i] is None or len(self.plan[i]) == 0:
                    self.plan[i] = list(self.sim.chunk(qpos[i], qvel[i], ctrl[i], tp[i], ty[i], chunk))
                act[i] = self.plan[i].pop(0)
            return act

    return ChunkOracle()


def make_policy(hold: int):
    import stable_worldmodel as swm

    class HeldOracle(swm.policy.BasePolicy):
        def __init__(self):
            super().__init__()
            self.type, self.prev = "held_oracle", None

        def get_action(self, info_dict, **kwargs):
            S = {c: _last(info_dict[info_key(c)]) for c in ORACLE_COLS}
            tp, ty = _last(info_dict["goal_privileged_block_0_pos"]), _last(info_dict["goal_privileged_block_0_yaw"])
            step = _last(info_dict["step_idx"]).reshape(-1).astype(int)
            act = cube_oracle(S["proprio_effector_pos"], S["proprio_effector_yaw"], S["proprio_gripper_contact"],
                              S["privileged_block_0_pos"], S["privileged_block_0_yaw"], tp, ty)
            if self.prev is not None and self.prev.shape == act.shape:
                keep = (step % hold) != 0                    # inside a block: hold the block's first action
                act[keep] = self.prev[keep]
            self.prev = act.copy()
            return act.astype(np.float32)

    return HeldOracle()


def run(hold: int, n: int = 50, seed: int = 42) -> dict:
    """hold > 0: oracle action recomputed every `hold` steps; hold < 0: open-loop oracle chunks of -hold steps."""
    import stable_worldmodel as swm

    from uncertainty_diag import eval_pairs
    cfg = TrainConfig()
    task = get_task("cube", cfg.img_size)
    apply_render_profile(task)
    ds = swm.data.HDF5Dataset("cube_eval", cache_dir=str(HOME), keys_to_cache=list(task.process_cols))
    rows = eval_pairs(n, seed)
    offs = np.asarray(ds.offsets)
    eps = [int(np.searchsorted(offs, r, side="right") - 1) for r in rows]
    starts = [int(r - offs[e]) for r, e in zip(rows, eps)]
    succ = []
    for lo in range(0, n, 5):
        world = swm.World(task.env_id, num_envs=len(eps[lo:lo + 5]), image_shape=(cfg.img_size, cfg.img_size),
                          max_episode_steps=2 * task.eval_budget, **task.eval_kwargs)
        world.set_policy(make_policy(hold) if hold > 0 else make_chunk_policy(task, -hold))
        m = world.evaluate(dataset=ds, episodes_idx=eps[lo:lo + 5], start_steps=starts[lo:lo + 5],
                           goal_offset=task.goal_offset, eval_budget=task.eval_budget, callables=task.eval_callables)
        world.close()
        succ += np.asarray(m["episode_successes"]).astype(int).tolist()
    return {"hold": hold, "successes": succ, "success": int(sum(succ)), "n": n}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=int, nargs="+", default=[1, 5],
                    help="> 0: hold each oracle action this many steps; < 0: open-loop oracle chunks of -hold steps")
    a = ap.parse_args()
    path = HOME / "results" / "oracle_sim_check.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    for h in a.hold:
        r = run(h)
        out[f"hold{h}"] = r
        print(json.dumps({k: v for k, v in r.items() if k != "successes"}), flush=True)
        path.write_text(json.dumps(out, indent=1))
