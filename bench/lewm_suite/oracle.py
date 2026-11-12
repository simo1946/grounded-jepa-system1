"""Vectorised OGBench Cube Markov oracle (the expert behind the authors' data, without its noise).

Reimplements ogbench.manipspace.oracles.markov.cube_markov.CubeMarkovOracle.select_action for N
states at once, so any state (an expert frame, or a state the learner visited) can be labelled
with the clean expert action toward any target cube pose. After the cube is placed, the original
moves the arm to a random final pose (irrelevant to the task's success test); here the arm holds
its pose instead (``final_pos`` / ``final_yaw`` default to the current effector pose).
Verified against the original in test_bench_units.py.
"""
from __future__ import annotations

import numpy as np

ORACLE_COLS = ("proprio_effector_pos", "proprio_effector_yaw", "proprio_gripper_contact",
               "privileged_block_0_pos", "privileged_block_0_yaw")


def _shape(diff: np.ndarray, min_norm: float) -> np.ndarray:
    n = np.linalg.norm(diff, axis=-1, keepdims=True)
    return np.where(n >= min_norm, diff, diff / (n + 1e-6) * min_norm)


def _shortest_yaw(eff_yaw: np.ndarray, obj_yaw: np.ndarray, n: int = 4) -> np.ndarray:
    sym = obj_yaw[:, None] + (np.arange(-n, n + 1) * 2 * np.pi / n)[None]      # [N, 2n+1]
    return sym[np.arange(len(sym)), np.abs(eff_yaw[:, None] - sym).argmin(1)]


def cube_oracle(eff_pos, eff_yaw, contact, block_pos, block_yaw, target_pos, target_yaw,
                final_pos=None, final_yaw=None, min_norm: float = 0.4) -> np.ndarray:
    """All inputs [N, ...] float arrays (yaws and contact [N] or [N, 1]). Returns raw actions [N, 5]."""
    f = lambda x: np.asarray(x, dtype=np.float64)  # noqa: E731
    eff_pos, block_pos, target_pos = f(eff_pos), f(block_pos), f(target_pos)
    eff_yaw, contact = f(eff_yaw).reshape(-1), f(contact).reshape(-1)
    block_yaw = _shortest_yaw(eff_yaw, f(block_yaw).reshape(-1))
    target_yaw = _shortest_yaw(eff_yaw, f(target_yaw).reshape(-1))
    final_pos = eff_pos if final_pos is None else f(final_pos)
    final_yaw = eff_yaw if final_yaw is None else f(final_yaw).reshape(-1)
    N = len(eff_pos)
    above_off = np.array([0.0, 0.0, 0.18])
    above_thr = 0.16
    closed, opened = contact > 0.5, contact < 0.1
    above = eff_pos[:, 2] > above_thr
    xy_al = np.linalg.norm(block_pos[:, :2] - eff_pos[:, :2], axis=-1) <= 0.04
    pos_al = np.linalg.norm(block_pos - eff_pos, axis=-1) <= 0.02
    t_xy_al = np.linalg.norm(target_pos[:, :2] - block_pos[:, :2], axis=-1) <= 0.04
    t_pos_al = np.linalg.norm(target_pos - block_pos, axis=-1) <= 0.02
    lift = np.stack([block_pos[:, 0], block_pos[:, 1], np.full(N, above_off[2] * 2)], -1)
    lift_rel = np.stack([block_pos[:, 0], block_pos[:, 1], np.full(N, above_thr * 2)], -1)
    # (destination, yaw error, gripper) per phase, in the original's order
    phases = [
        (block_pos + above_off, block_yaw - eff_yaw, -1.0),           # 1 move above the block
        (block_pos, block_yaw - eff_yaw, -1.0),                       # 2 move to the block
        (block_pos, block_yaw - eff_yaw, 1.0),                        # 3 grasp
        (lift, target_yaw - block_yaw, 1.0),                          # 4 move in the air
        (target_pos + above_off, target_yaw - block_yaw, 1.0),        # 5 move above the target
        (target_pos, target_yaw - block_yaw, 1.0),                    # 6 move to the target
        (target_pos, target_yaw - block_yaw, -1.0),                   # 7 release
        (lift_rel, final_yaw - eff_yaw, -1.0),                        # 8 move in the air
        (final_pos, final_yaw - eff_yaw, -1.0),                       # 9 move to the final pose
    ]
    nt = ~t_pos_al
    cond = [nt & ~xy_al,
            nt & xy_al & ~pos_al,
            nt & pos_al & ~closed,
            nt & pos_al & closed & ~above & ~t_xy_al,
            nt & pos_al & closed & above & ~t_xy_al,
            nt,                                                        # 6: the remaining not-placed cases
            t_pos_al & ~opened,
            t_pos_al & opened & ~above,
            t_pos_al]
    phase = np.select(cond, np.arange(9), default=8)
    act = np.zeros((N, 5))
    for i, (dest, dyaw, grip) in enumerate(phases):
        m = phase == i
        if not m.any():
            continue
        act[m, :3] = _shape(dest[m] - eff_pos[m], min_norm) * 5.0
        act[m, 3] = dyaw[m] * 3.0
        act[m, 4] = grip
    return np.clip(act, -1, 1)
