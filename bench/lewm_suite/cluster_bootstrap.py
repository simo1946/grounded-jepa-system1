"""Paired success differences with a cluster bootstrap.

Every model (or seed) is evaluated on the same start/goal pairs, and several pairs can come from the same
evaluation episode, so model-pair outcomes are not independent. The bootstrap resamples clusters (episodes, or
pairs when the episode is not recorded) with replacement; each drawn cluster brings all of its pairs, for all
models together. Statistic: mean over models and pairs of (success A - success B), in percentage points.
"""
from __future__ import annotations

import numpy as np


def cluster_bootstrap(a: np.ndarray, b: np.ndarray, clusters: np.ndarray | None = None, n_boot: int = 10_000,
                      seed: int = 0, level: float = 0.95) -> dict:
    """a, b: 0/1 outcomes [units, pairs] on the same pairs (a unit is one trained model); clusters: [pairs] ids
    (default: each pair is its own cluster). Returns the observed difference (points), a percentile interval
    and a two-sided bootstrap p (twice the smaller tail; resolution 2 / n_boot)."""
    a, b = np.atleast_2d(np.asarray(a, float)), np.atleast_2d(np.asarray(b, float))
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    n_pairs = a.shape[1]
    clusters = np.arange(n_pairs) if clusters is None else np.asarray(clusters)
    if len(clusters) != n_pairs:
        raise ValueError("one cluster id per pair is needed")
    ids, inv = np.unique(clusters, return_inverse=True)
    d = (a - b).sum(0)                                      # per pair, summed over units
    s = np.bincount(inv, weights=d, minlength=len(ids))     # per cluster: sum of differences
    n = np.bincount(inv, minlength=len(ids)) * a.shape[0]   # per cluster: number of unit-pair outcomes
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    counts = np.zeros((n_boot, len(ids)))
    np.add.at(counts, (np.repeat(np.arange(n_boot), len(ids)), draws.ravel()), 1)
    stat = 100 * (counts @ s) / (counts @ n)
    obs = 100 * s.sum() / n.sum()
    lo, hi = np.quantile(stat, [(1 - level) / 2, 1 - (1 - level) / 2])
    p = min(1.0, 2 * min((stat <= 0).mean(), (stat >= 0).mean()))
    p = max(p, 2 / n_boot)                                 # bootstrap resolution
    return {"diff_pts": float(obs), "ci_lo": float(lo), "ci_hi": float(hi), "p_boot": float(p),
            "clusters": int(len(ids)), "units": int(a.shape[0]), "n_boot": n_boot}


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    out, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = running
    return out
