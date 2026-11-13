"""Check whether two evaluation pairs share an identical goal image.

The camera-dropout memory in system12.py recognised an episode by its goal latent. Before the fix (reset per batch + a
duplicate check), two pairs with identical goal images could have shared memory. This rebuilds the evaluation pairs
exactly as evaluate.py does and reports any identical or near-identical goal frames. No duplicates means the
camera-dropout results were not affected.

python check_goal_collisions.py --task cube --num-eval 60          # GPU study pairs
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np

from evaluate import pick_pairs
from models import HOME, TrainConfig
from tasks import get_task, trivial_mask


def main(task_name: str, num_eval: int, goal_offset: int, eval_seed: int) -> dict:
    import stable_worldmodel as swm
    task = get_task(task_name, TrainConfig().img_size)
    ds = swm.data.HDF5Dataset(f"{task_name}_eval", cache_dir=str(HOME), keys_to_cache=list(task.process_cols))
    exclude = lambda e, s: trivial_mask(task, ds, e, s, goal_offset)  # noqa: E731
    eps, starts = pick_pairs(np.asarray(ds.lengths), goal_offset, num_eval, eval_seed, exclude)
    f = h5py.File(Path(HOME) / "datasets" / f"{task_name}_eval.h5", "r")
    off = f["ep_offset"][:]
    rows = [int(off[e] + s + goal_offset) for e, s in zip(eps, starts, strict=True)]
    goals = np.stack([f["pixels"][r] for r in rows]).astype(np.float32)
    n = len(goals)
    diff = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            diff[i, j] = np.abs(goals[i] - goals[j]).max()
    i, j = np.unravel_index(np.argmin(diff), diff.shape)
    same_episode = sum(len(v) - 1 for v in [[k for k, e in enumerate(eps) if e == u] for u in set(eps)] if len(v) > 1)
    res = {"task": task_name, "pairs": n, "identical_goal_images": int((diff == 0).sum()),
           "closest_pair_max_pixel_difference": float(diff[i, j]), "closest_pair": [int(i), int(j)],
           "extra_pairs_from_an_already_used_episode": int(same_episode),
           "episodes_idx": [int(e) for e in eps], "start_steps": [int(t) for t in starts]}   # the pair identities
    print(json.dumps(res))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="cube")
    ap.add_argument("--num-eval", type=int, default=60)
    ap.add_argument("--goal-offset", type=int, default=25)
    ap.add_argument("--eval-seed", type=int, default=42)
    ap.add_argument("--home", default=os.environ.get("STABLEWM_HOME"))
    a = ap.parse_args()
    main(a.task, a.num_eval, a.goal_offset, a.eval_seed)
