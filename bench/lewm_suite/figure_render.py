"""Render the camera frames of Figures 1 and 2 at high resolution (illustration only).

The model sees 64-pixel frames. For print, the same simulator states (saved by figure_examples.py and
figure_rollout.py) are rendered again by the same Cube environment and camera at --size pixels, with the
simulator's default (higher-quality) shadows. Writes already_solved_render.npz and rollout_render.npz next to the
inputs.

    MUJOCO_GL=egl python figure_render.py --data ../../paper/figures/data --size 384
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def renderer(size: int):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401  (registers the swm/ environments)

    env = gym.make("swm/OGBCube-v0", env_type="single", multiview=False, width=size, height=size,
                   visualize_info=False, ob_type="states", terminate_at_goal=True)
    env.reset(seed=0)
    u = env.unwrapped

    def render(qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        u.set_state(np.asarray(qpos, np.float64), np.asarray(qvel, np.float64))
        return np.asarray(u.render(), np.uint8).copy()
    return env, render


def main(data: Path, size: int) -> None:
    env, render = renderer(size)
    try:
        a = np.load(data / "already_solved.npz")
        np.savez_compressed(data / "already_solved_render.npz",
                            start=render(a["start_qpos"], a["start_qvel"]), goal=render(a["goal_qpos"], a["goal_qvel"]))
        r = np.load(data / "rollout.npz")
        frames = np.stack([render(q, v) for q, v in zip(r["qpos"], r["qvel"], strict=True)])
        np.savez_compressed(data / "rollout_render.npz", frames=frames,
                            start=render(r["start_qpos"], r["start_qvel"]), goal=render(r["goal_qpos"], r["goal_qvel"]))
    finally:
        env.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--size", type=int, default=384)
    a = ap.parse_args()
    main(a.data, a.size)
