"""Clean oracle chunk labels for hindsight windows of the training data (oracle_sim.py).

Window: start frame row r (H*fs steps inside its episode), hindsight goal k in 1..H blocks ahead
(row g = r + k*fs). Label: the oracle simulated closed-loop for H*fs steps from the state at r toward
the cube pose at g (raw actions). Rows refer to the dataset / latent cache.

python gen_oracle_chunks.py --n 10000 --part 0 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from models import HOME, TrainConfig  # noqa: E402
from oracle_sim import OracleSim  # noqa: E402
from tasks import get_task  # noqa: E402

H, FS = 5, 5


def main(n: int, part: int, seed: int, data: str = "cube") -> None:
    task = get_task("cube", TrainConfig().img_size)
    with h5py.File(HOME / "datasets" / f"{data}.h5", "r") as f:
        off, L = f["ep_offset"][:], f["ep_len"][:]
        qpos, qvel, ctrl = f["qpos"][:], f["qvel"][:], f["control"][:]
        bp, by = f["privileged_block_0_pos"][:], f["privileged_block_0_yaw"][:]
    starts = np.concatenate([o + np.arange(l - H * FS) for o, l in zip(off, L, strict=True) if l > H * FS])
    rng = np.random.default_rng(1000 + seed)
    r = np.sort(rng.choice(starts, n, replace=False))
    k = rng.integers(1, H + 1, n)
    g = r + k * FS
    sim = OracleSim(task.env_id, **task.eval_kwargs)
    lab = np.zeros((n, H * FS, 5), dtype=np.float32)
    t0 = time.time()
    for i in range(n):
        lab[i] = sim.chunk(qpos[r[i]], qvel[r[i]], ctrl[r[i]], bp[g[i]], by[g[i]], H * FS)
        if (i + 1) % 2000 == 0:
            print(json.dumps({"done": i + 1, "of": n, "min": round((time.time() - t0) / 60, 1)}), flush=True)
    out = HOME / "datasets" / f"{data}_oracle_chunks_p{part}.pt"
    torch.save({"rows": torch.from_numpy(r), "goal_rows": torch.from_numpy(g), "actions_raw": torch.from_numpy(lab),
                "data": data, "steps": H * FS}, out)
    print(json.dumps({"saved": str(out), "minutes": round((time.time() - t0) / 60, 1)}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.n, a.part, a.seed)
