"""Collect a task dataset with the authors' expert, then flatten nested HDF5
columns (``privileged/block_0_pos`` -> ``privileged_block_0_pos``), which is
the naming the authors' reader and eval callables expect.

    python collect.py --task cube --name cube --episodes 400 --seed 3072
    python collect.py --task cube --name cube_eval --episodes 80 --seed 777
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

from models import HOME  # noqa: E402  (sets STABLEWM_HOME)
from tasks import apply_render_profile, get_task  # noqa: E402


def flatten_h5(path, column_2d: bool = False) -> list:
    """Move nested datasets ('privileged/x') to flat names ('privileged_x'). column_2d: also store per-row scalar
    columns as '<name>_2d' [N, 1] (the authors' dataset concatenates state columns along axis 1, which needs 2-D;
    the env's setters need the scalar form)."""
    import h5py

    moved = []
    with h5py.File(path, "r+") as f:
        nested = []
        f.visit(lambda name: nested.append(name) if "/" in name and isinstance(f[name], h5py.Dataset) else None)
        for name in nested:
            flat = name.replace("/", "_")
            if flat in f:
                raise RuntimeError(f"cannot flatten {name!r}: {flat!r} already exists")
            data = f[name][...]
            f.create_dataset(flat, data=data)
            if column_2d and data.ndim == 1:                   # keep the 1-D column for the env's setters
                f.create_dataset(flat + "_2d", data=data.reshape(-1, 1))
            moved.append(flat)
        for top in {n.split("/")[0] for n in nested}:
            del f[top]
    return moved


def main(task_name: str, name: str, episodes: int, seed: int, img: int, num_envs: int) -> None:
    import numpy as np
    import stable_worldmodel as swm

    task = get_task(task_name, img)
    apply_render_profile(task)
    path = HOME / "datasets" / f"{name}.h5"
    if path.exists():
        raise FileExistsError(f"{path} exists; delete it to recollect")
    # write under a temporary name and rename at the end, so an interrupted run never
    # leaves a file that looks complete
    final, path = path, HOME / "datasets" / f"{name}__partial.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    world = swm.World(task.env_id, num_envs=num_envs, image_shape=(img, img), max_episode_steps=task.episode_steps,
                      **task.collect_kwargs)
    world.set_policy(task.expert())
    rng = np.random.default_rng(seed)
    t0 = time.time()
    world.collect(path, episodes=episodes, seed=int(rng.integers(0, 1_000_000)), format="hdf5", progress=False)
    moved = flatten_h5(path)
    path.rename(final)
    path = final
    print(f"collected {episodes} {task_name} episodes -> {path} in {time.time() - t0:.0f}s; "
          f"flattened {len(moved)} columns", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["tworoom", "cube"])
    ap.add_argument("--name", required=True)
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--img", type=int, default=64)
    ap.add_argument("--num-envs", type=int, default=4)
    a = ap.parse_args()
    main(a.task, a.name, a.episodes, a.seed, a.img, a.num_envs)
