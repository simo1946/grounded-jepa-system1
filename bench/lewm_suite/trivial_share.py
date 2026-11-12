"""How many Cube start/goal pairs are already solved at the start.

A pair is a start row and the row ``goal_offset`` steps later in the same episode, as in the LeWM evaluation
(le-wm ``eval.py``, commit 8edfeb3: start rows with step_idx <= episode length - goal_offset - 1). A pair is
already solved when the cube at the start is within ``threshold`` metres of the cube at the goal, the success
test of the OGBench Cube environment (``cube_env._compute_successes``: distance <= 0.04 m).

Reports, for one HDF5 dataset in the stable-worldmodel layout (one dataset per column):
  * all_pairs: every valid pair;
  * lewm_eval_draw: the ``num_eval`` start rows that le-wm ``eval.py`` draws with ``seed`` (same calls:
    ``default_rng(seed).choice(len(valid) - 1, size=num_eval, replace=False)``, then sorted).

Reads only the episode, step and cube-position columns, so it runs on the official 46 GB file without
loading frames. Prints one JSON object on stdout (no clock, no randomness beyond the seeded draw).

    python trivial_share.py /path/to/cube_single_expert.h5
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

POS_COLUMN = "privileged_block_0_pos"


def _col(f, *names: str) -> np.ndarray:
    for n in names:
        if n in f:
            return np.asarray(f[n][:])
    raise KeyError(f"none of {names} in the file (columns: {sorted(f.keys())[:40]})")


def valid_starts(ep: np.ndarray, step: np.ndarray, goal_offset: int) -> np.ndarray:
    """Row indices that le-wm eval.py accepts as start rows (step_idx <= length - goal_offset - 1)."""
    ids, inv = np.unique(ep, return_inverse=True)
    length = np.zeros(len(ids), dtype=np.int64)
    np.maximum.at(length, inv, step.astype(np.int64) + 1)
    return np.nonzero(step <= (length - goal_offset - 1)[inv])[0]


def lewm_eval_draw(valid: np.ndarray, num_eval: int, seed: int) -> np.ndarray:
    """The start rows le-wm eval.py draws (same generator calls, same off-by-one in the range)."""
    g = np.random.default_rng(seed)
    pick = g.choice(len(valid) - 1, size=num_eval, replace=False)
    return np.sort(valid[pick])


def solved_at_start(pos: np.ndarray, ep: np.ndarray, step: np.ndarray, rows: np.ndarray, goal_offset: int,
                    threshold: float) -> np.ndarray:
    goal = rows + goal_offset
    if not (np.array_equal(ep[goal], ep[rows]) and np.array_equal(step[goal], step[rows] + goal_offset)):
        raise RuntimeError("rows are not stored episode by episode in step order; the goal row cannot be found")
    return np.linalg.norm(pos[rows] - pos[goal], axis=-1) <= threshold


def summary(path: Path, goal_offset: int = 25, threshold: float = 0.04, num_eval: int = 50, seed: int = 42) -> dict:
    import h5py
    try:
        import hdf5plugin  # noqa: F401  (compressed columns)
    except ImportError:
        pass
    with h5py.File(path, "r") as f:
        if "ep_idx" in f or "episode_idx" in f:
            ep = _col(f, "ep_idx", "episode_idx").astype(np.int64)
        else:                                       # our collector: episodes given by ep_len / ep_offset
            lengths, offsets = np.asarray(f["ep_len"][:]), np.asarray(f["ep_offset"][:])
            ep = np.empty(int(lengths.sum()), dtype=np.int64)
            for i, (o, n) in enumerate(zip(offsets, lengths, strict=True)):
                ep[o:o + n] = i
        step = _col(f, "step_idx").astype(np.int64)
        pos = _col(f, POS_COLUMN).astype(np.float64).reshape(len(ep), -1)
    valid = valid_starts(ep, step, goal_offset)
    triv = solved_at_start(pos, ep, step, valid, goal_offset, threshold)
    drawn = lewm_eval_draw(valid, num_eval, seed)
    triv_drawn = solved_at_start(pos, ep, step, drawn, goal_offset, threshold)
    lengths = np.bincount(np.unique(ep, return_inverse=True)[1])
    return {
        "dataset": path.name,
        "column_sha256": hashlib.sha256(np.ascontiguousarray(pos).tobytes()).hexdigest(),
        "episodes": int(len(lengths)), "rows": int(len(ep)),
        "episode_length_min": int(lengths.min()), "episode_length_max": int(lengths.max()),
        "goal_offset": goal_offset, "threshold_m": threshold,
        "all_pairs": {"pairs": int(len(valid)), "solved_at_start": int(triv.sum()),
                      "share": float(triv.mean())},
        "lewm_eval_draw": {"num_eval": num_eval, "seed": seed, "solved_at_start": int(triv_drawn.sum()),
                           "share": float(triv_drawn.mean()), "rows": drawn.tolist()},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path, nargs="+")
    ap.add_argument("--goal-offset", type=int, default=25)
    ap.add_argument("--threshold", type=float, default=0.04)
    ap.add_argument("--num-eval", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = {p.stem: summary(p, a.goal_offset, a.threshold, a.num_eval, a.seed) for p in a.dataset}
    print(json.dumps(out, indent=1))
