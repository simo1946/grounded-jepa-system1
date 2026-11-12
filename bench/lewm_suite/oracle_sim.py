"""Clean oracle action chunks by simulation.

From a simulator state (qpos, qvel, ctrl), roll the oracle closed-loop for `steps` env steps in a
scratch (non-rendering) Cube env toward a target cube pose, and return its actions. Replayed
open-loop from the same state, the chunk reproduces the oracle's trajectory up to contact
non-determinism (MuJoCo's solver warm-start is not part of qpos/qvel). These are self-consistent
labels for a chunk policy; holding one oracle action per block is not (oracle_sim_check.py:
48/50 closed loop vs 8/50 held per 5-step block).
"""
from __future__ import annotations

import numpy as np

from oracle import cube_oracle


class OracleSim:
    def __init__(self, env_id: str = "swm/OGBCube-v0", **env_kwargs):
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401  (registers the envs)
        kw = {k: v for k, v in env_kwargs.items() if k not in ("width", "height")}
        self.env = gym.make(env_id, **kw)
        self.env.reset(seed=0)
        self.u = self.env.unwrapped

    def _info(self):
        i = self.u.compute_ob_info()
        return (i["proprio/effector_pos"][None], i["proprio/effector_yaw"][None], i["proprio/gripper_contact"][None],
                i["privileged/block_0_pos"][None], i["privileged/block_0_yaw"][None])

    def chunk(self, qpos, qvel, ctrl, target_pos, target_yaw, steps: int = 25) -> np.ndarray:
        """Raw oracle actions [steps, 5] from this state toward the target cube pose."""
        self.u.set_state(np.asarray(qpos, dtype=np.float64), np.asarray(qvel, dtype=np.float64))
        if ctrl is not None:
            self.u._data.ctrl[:] = np.asarray(ctrl, dtype=np.float64)
        tp, ty = np.asarray(target_pos, dtype=np.float64)[None], np.asarray(target_yaw, dtype=np.float64).reshape(1, 1)
        out = np.zeros((steps, 5))
        for j in range(steps):
            a = cube_oracle(*self._info(), tp, ty)[0]
            out[j] = a
            self.u.step(a)
        return out

    def execute(self, qpos, qvel, ctrl, actions) -> tuple[np.ndarray, np.ndarray]:
        """Execute raw actions [T, 5] open-loop from this state; return the final cube (position, yaw)."""
        self.u.set_state(np.asarray(qpos, dtype=np.float64), np.asarray(qvel, dtype=np.float64))
        if ctrl is not None:
            self.u._data.ctrl[:] = np.asarray(ctrl, dtype=np.float64)
        for a in np.asarray(actions, dtype=np.float64):
            self.u.step(np.clip(a, -1, 1))
        i = self.u.compute_ob_info()
        return i["privileged/block_0_pos"].copy(), i["privileged/block_0_yaw"].copy()
