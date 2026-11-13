"""Diagnostic: single-chunk success of the practised correction on HELD-OUT episodes (cube_eval states),
non-trivial situations, true simulator outcomes. Separates 'does not generalise to new episodes' from
'does not survive closed loop'."""
import json
import multiprocessing as mp

import h5py
import torch

from cached_policy import H, OfflineWindows
from hybrid_wm.control.latent_policy import policy_from_checkpoint
from latent_cache import cache_path
from models import HOME
from oracle_policy import action_scaler
import residual_rl as rr

S = "lewm+idm+sa+obj"
ck = HOME / "checkpoints" / f"cube/{S}/seed0"
cache = torch.load(cache_path(S, 0, "cube_eval"), map_location="cpu")
train_cache = torch.load(cache_path(S, 0, "cube"), map_location="cpu")
z, fs, A = cache["z"], cache["frameskip"], cache["action"].shape[1]
mean, std = action_scaler(train_cache, "cube")
with h5py.File(HOME / "datasets" / "cube_eval.h5", "r") as f:
    bp = torch.from_numpy(f["privileged_block_0_pos"][:])
win = OfflineWindows(cache)
g = torch.Generator().manual_seed(7)
r = win.starts[torch.randint(len(win.starts), (4000,), generator=g)]
gl = r + torch.randint(1, H + 1, (4000,), generator=g) * fs
keep = (bp[r] - bp[gl]).norm(dim=-1) > 0.04
r, gl = r[keep][:400], gl[keep][:400]
pool = mp.get_context("fork").Pool(2, initializer=rr._init_worker, initargs=("cube_eval",))
out = {"states": int(len(r))}
for name in ("policy_gauss6k", "policy_res_real", "policy_res_latent"):
    pol = policy_from_checkpoint(torch.load(ck / f"{name}.pt", map_location="cpu")).eval()
    with torch.no_grad():
        ch = pol.sample(z[r], z[gl], 1)[:, 0]
    raw = ch.reshape(len(ch), H * fs, A).double().numpy() * std + mean
    d = rr.real_distances(pool, r.numpy(), gl.numpy(), raw, 2)
    kk = ((gl - r) // fs).numpy()
    out[name] = {"success": float((d <= 0.04).mean()), "dist_cm": float(100 * d.mean()),
                 "success_by_blocks_ahead": {int(k): float((d[kk == k] <= 0.04).mean()) for k in range(1, H + 1)},
                 "n_by_blocks_ahead": {int(k): int((kk == k).sum()) for k in range(1, H + 1)}}
pool.close()
print(json.dumps(out))
(HOME / "results" / "residual_heldout.json").write_text(json.dumps(out, indent=1))
