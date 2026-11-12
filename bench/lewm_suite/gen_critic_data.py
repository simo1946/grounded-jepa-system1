"""Real-outcome data for a candidate critic (prereg_critic.md).

From dataset states (window start r, hindsight goal g = r + k*fs, k in 1..H), System 1 proposes 16
candidate chunks (candidate 0 = its mean). Each is executed open-loop in a scratch simulator from the
true state at r; the outcome is the final cube distance to the goal cube position (metres).

python gen_critic_data.py --spec lewm+idm+sa+obj --policy-file policy_gauss6k.pt --data cube --n 1500 --part 0
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

from cached_policy import H, OfflineWindows  # noqa: E402
from hybrid_wm.control.latent_policy import policy_from_checkpoint  # noqa: E402
from latent_cache import cache_path  # noqa: E402
from models import HOME, Spec, TrainConfig  # noqa: E402
from oracle_policy import action_scaler  # noqa: E402
from oracle_sim import OracleSim  # noqa: E402
from tasks import get_task  # noqa: E402

K = 16


def main(spec: str, seed: int, policy_file: str, data: str, n: int, part: int) -> None:
    sp = Spec.parse(spec)
    task = get_task("cube", TrainConfig().img_size)
    cache = torch.load(cache_path(sp.name(), seed, data), map_location="cpu")
    pol = policy_from_checkpoint(torch.load(HOME / "checkpoints" / f"cube/{sp.name()}/seed{seed}" / policy_file,
                                            map_location="cpu")).eval()
    mean, std = action_scaler(cache, data)
    fs, A = cache["frameskip"], cache["action"].shape[1]
    with h5py.File(HOME / "datasets" / f"{data}.h5", "r") as f:
        qpos, qvel, ctrl, bp = f["qpos"][:], f["qvel"][:], f["control"][:], f["privileged_block_0_pos"][:]
    win = OfflineWindows(cache)
    g_ = torch.Generator().manual_seed(5000 + 97 * part + (0 if data == "cube" else 1))
    r = win.starts[torch.randint(len(win.starts), (n,), generator=g_)]
    k = torch.randint(1, H + 1, (n,), generator=g_)
    g = r + k * fs
    with torch.no_grad():
        cand = pol.sample(cache["z"][r], cache["z"][g], K, generator=g_)                     # [n, K, H, fs*A]
    raw = cand.view(n, K, H * fs, A).double().numpy() * std + mean                          # env actions
    sim = OracleSim(task.env_id, **task.eval_kwargs)
    dist = np.zeros((n, K), dtype=np.float32)
    t0 = time.time()
    for i in range(n):
        ri, gi = int(r[i]), int(g[i])
        for j in range(K):
            pos, _ = sim.execute(qpos[ri], qvel[ri], ctrl[ri], raw[i, j])
            dist[i, j] = np.linalg.norm(pos - bp[gi])
        if (i + 1) % 250 == 0:
            print(json.dumps({"done": i + 1, "of": n, "min": round((time.time() - t0) / 60, 1)}), flush=True)
    out = HOME / "datasets" / f"critic_{sp.name().replace('+', '_')}__{data}_p{part}.pt"
    torch.save({"rows": r, "goal_rows": g, "cand": cand.float(), "dist": torch.from_numpy(dist), "policy_file": policy_file,
                "spec": spec, "data": data}, out)
    best = dist.min(1)
    print(json.dumps({"saved": str(out), "minutes": round((time.time() - t0) / 60, 1),
                      "success_mean_candidate": float((dist[:, 0] <= 0.04).mean()),
                      "success_any_candidate": float((best <= 0.04).mean()),
                      "success_random_candidate": float((dist[:, 1:] <= 0.04).mean())}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy-file", required=True)
    ap.add_argument("--data", default="cube")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--part", type=int, default=0)
    a = ap.parse_args()
    main(a.spec, a.seed, a.policy_file, a.data, a.n, a.part)
