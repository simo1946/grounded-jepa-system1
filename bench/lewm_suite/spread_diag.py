"""Candidate spread in normalised action units, to compare with the world model's
perturbation identifiability (rollout_diag.py: true chunk vs copies + sigma noise).

python spread_diag.py --spec lewm+idm+sa [--pool policy_cached.pt policy_p1.pt ...]
"""
from __future__ import annotations

import argparse
import itertools
import json

import torch

from cached_policy import H, OfflineWindows
from hybrid_wm.control.latent_policy import ChunkPolicy
from latent_cache import cache_path
from models import HOME, Spec


def load_policy(ckpt, f):
    pk = torch.load(ckpt / f, map_location="cpu")
    pol = ChunkPolicy(pk["latent_dim"], pk["horizon"], pk["action_dim"], extra_dim=pk.get("extra_dim", 0))
    pol.load_state_dict(pk["state_dict"])
    return pol.eval()


@torch.no_grad()
def main(spec: str, seed: int, n: int, pool: list[str] | None) -> dict:
    sp = Spec.parse(spec)
    ckpt = HOME / "checkpoints" / f"cube/{sp.name()}/seed{seed}"
    cache = torch.load(cache_path(sp.name(), seed, "cube_eval"), map_location="cpu")
    g = torch.Generator().manual_seed(0)
    b = OfflineWindows(cache).sample(n, g)
    k = torch.randint(1, H + 1, (n,), generator=g)
    z_t, z_g = b["z"][:, 0], b["z"][torch.arange(n), k]
    out = {"spec": spec, "n_states": n}
    _, std = load_policy(ckpt, "policy.pt")(z_t, z_g)
    out["policy_std_rms"] = float(std.square().mean().sqrt())          # per action dim, normalised units
    if pool:
        means = [load_policy(ckpt, f)(z_t, z_g)[0] for f in pool]
        d = [float((a - c).square().mean().sqrt()) for a, c in itertools.combinations(means, 2)]
        out["pool_pairwise_rms"] = sum(d) / len(d)                       # per dim, between member means
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--pool", nargs="*", default=None)
    a = ap.parse_args()
    main(a.spec, a.seed, a.n, a.pool)
