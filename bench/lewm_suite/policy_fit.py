"""Offline fit of a System 1 policy on held-out latents (no environment).

For hindsight windows from the held-out cache: distance (rms per action dimension, normalised
units, valid blocks only) between the executed chunk and
  - the deterministic proposal (Gaussian mean / flow zero-noise sample),
  - the closest of k samples (how well the sampled set covers what the expert did),
  - an average sample.

python policy_fit.py --spec lewm+idm+sa --policies policy_gauss6k.pt policy_flow6k.pt
"""
from __future__ import annotations

import argparse
import json

import torch

from cached_policy import H, OfflineWindows
from hybrid_wm.control.latent_policy import policy_from_checkpoint
from latent_cache import cache_path
from models import HOME, Spec


@torch.no_grad()
def fit(spec: str, seed: int, files: list[str], n: int = 2048, k: int = 16) -> dict:
    sp = Spec.parse(spec)
    ckpt = HOME / "checkpoints" / f"cube/{sp.name()}/seed{seed}"
    cache = torch.load(cache_path(sp.name(), seed, "cube_eval"), map_location="cpu")
    g = torch.Generator().manual_seed(0)
    b = OfflineWindows(cache).sample(n, g)
    kk = torch.randint(1, H + 1, (n,), generator=g)
    z_t, z_g = b["z"][:, 0], b["z"][torch.arange(n), kk]
    act = b["action"].float()
    valid = (torch.arange(H)[None] < kk[:, None]) & torch.isfinite(act).all(-1)      # [n, H]
    act = torch.nan_to_num(act)

    def rms(x):                                                                       # x [n, ..., H, A]
        e = (x - act.view(n, *([1] * (x.dim() - 3)), H, -1)).square().mean(-1)         # [n, ..., H]
        v = valid.view(n, *([1] * (x.dim() - 3)), H).to(e.dtype)
        return (e * v).sum(-1) / v.sum(-1).clamp_min(1)                               # [n, ...]

    out = {"spec": spec, "n_windows": n, "k": k, "data_rms": float(act[valid].square().mean().sqrt())}
    for f in files:
        pol = policy_from_checkpoint(torch.load(ckpt / f, map_location="cpu")).eval()
        s = pol.sample(z_t, z_g, k, generator=torch.Generator().manual_seed(1))       # [n, k, H, A]
        e = rms(s)                                                                    # [n, k] mean squared
        out[f] = {"deterministic": float(e[:, 0].mean().sqrt()),
                  "best_of_k": float(e.min(1).values.mean().sqrt()),
                  "average_sample": float(e[:, 1:].mean().sqrt())}
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=2048)
    a = ap.parse_args()
    res = fit(a.spec, a.seed, a.policies, a.n)
    sp = Spec.parse(a.spec)
    (HOME / "results" / f"policy_fit__{sp.name()}__seed{a.seed}.json").write_text(json.dumps(res, indent=1))
