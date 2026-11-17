"""An already-solved evaluation pair, for the task figure of the paper (illustration only).

Among the 50 pairs of the authors' protocol drawn in the CPU re-run (the random-policy result file records them and
which ones are already solved), takes the already-solved pair whose arm moves farthest between the start and the
goal frame: the frames differ, yet the goal condition (cube within 4 cm) already holds at the start. Writes
already_solved.npz (the two 64-pixel frames and their simulator states) and already_solved.json (which pair, cube
and arm displacement).

    python figure_examples.py --random-result ../../results/cpu_rerun/cube__random__off25_bud50_es42.json \
        --data external/cube_data/cube_eval.h5 --out ../../paper/figures/data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main(random_result: Path, data: Path, out: Path) -> dict:
    r = json.loads(random_result.read_text())
    eps, starts, triv, offset = r["episodes_idx"], r["start_steps"], np.asarray(r["trivial"], bool), r["goal_offset"]
    with h5py.File(data, "r") as f:
        off = f["ep_offset"][:]
        cube, arm = f["privileged_block_0_pos"][:], f["proprio_effector_pos"][:]
        rows = [(int(off[e] + s), int(off[e] + s + offset)) for e, s in zip(eps, starts, strict=True)]
        arm_moved = np.array([np.linalg.norm(arm[g] - arm[s]) for s, g in rows])
        cube_moved = np.array([np.linalg.norm(cube[g] - cube[s]) for s, g in rows])
        pick = int(np.argmax(np.where(triv, arm_moved, -1.0)))
        s_row, g_row = rows[pick]
        start, goal = f["pixels"][s_row], f["pixels"][g_row]
        st = {f"{w}_{k}": f[k][r] for k in ("qpos", "qvel") for w, r in (("start", s_row), ("goal", g_row))}
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "already_solved.npz", start=start, goal=goal, **st)
    info = {"source": random_result.name, "pair_index": pick, "episode": int(eps[pick]),
            "start_step": int(starts[pick]), "goal_offset": int(offset), "already_solved": bool(triv[pick]),
            "arm_moved_m": float(arm_moved[pick]), "cube_moved_m": float(cube_moved[pick]),
            "already_solved_pairs": int(triv.sum()), "pairs": len(triv)}
    (out / "already_solved.json").write_text(json.dumps(info, indent=1))
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--random-result", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    print(json.dumps(main(a.random_result, a.data, a.out)))
