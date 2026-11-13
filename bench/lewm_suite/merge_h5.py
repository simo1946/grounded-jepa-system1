"""Merge datasets collected in parallel (collect.py parts) into one.

Every per-step column is concatenated; the per-episode columns ``ep_len`` and
``ep_offset`` are rebuilt. Parts must have identical columns, dtypes and
per-row shapes. Copies in chunks, so memory stays bounded.

    python merge_h5.py --out cube --parts cube_p0 cube_p1 cube_p2 cube_p3 [--delete-parts]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from models import HOME

EPISODE_COLS = ("ep_len", "ep_offset")
CHUNK = 4096


def merge(out: str, parts: list[str], delete_parts: bool = False) -> dict:
    import h5py

    paths = [HOME / "datasets" / f"{p}.h5" for p in parts]
    dst_path = HOME / "datasets" / f"{out}.h5"
    if dst_path.exists():
        raise FileExistsError(f"{dst_path} exists")
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(missing)
    srcs = [h5py.File(p, "r") for p in paths]
    try:
        cols = sorted(k for k in srcs[0].keys() if k not in EPISODE_COLS)
        for f, p in zip(srcs[1:], paths[1:], strict=True):
            other = sorted(k for k in f.keys() if k not in EPISODE_COLS)
            if other != cols:
                raise ValueError(f"{p.name}: columns differ from {paths[0].name}")
            for c in cols:
                if f[c].dtype != srcs[0][c].dtype or f[c].shape[1:] != srcs[0][c].shape[1:]:
                    raise ValueError(f"{p.name}: column {c!r} has a different dtype or row shape")
        rows = [int(f["ep_len"][:].sum()) for f in srcs]
        for f, r, p in zip(srcs, rows, paths, strict=True):
            for c in cols:
                if f[c].shape[0] != r:
                    raise ValueError(f"{p.name}: column {c!r} has {f[c].shape[0]} rows, episodes sum to {r}")
        lens = np.concatenate([f["ep_len"][:] for f in srcs])
        offsets = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(srcs[0]["ep_offset"].dtype)
        total = int(lens.sum())
        tmp = dst_path.with_suffix(".tmp.h5")
        with h5py.File(tmp, "w") as dst:
            dst.create_dataset("ep_len", data=lens.astype(srcs[0]["ep_len"].dtype))
            dst.create_dataset("ep_offset", data=offsets)
            for c in cols:
                ref = srcs[0][c]
                d = dst.create_dataset(c, shape=(total, *ref.shape[1:]), dtype=ref.dtype, chunks=ref.chunks)
                pos = 0
                for f in srcs:
                    n = f[c].shape[0]
                    for lo in range(0, n, CHUNK):
                        blk = f[c][lo:lo + CHUNK]
                        d[pos + lo: pos + lo + len(blk)] = blk
                    pos += n
        tmp.rename(dst_path)
    finally:
        for f in srcs:
            f.close()
    if delete_parts:
        for p in paths:
            p.unlink()
    info = {"merged": str(dst_path), "parts": parts, "episodes": int(len(lens)), "rows": total}
    print(json.dumps(info), flush=True)
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--delete-parts", action="store_true")
    a = ap.parse_args()
    merge(a.out, a.parts, a.delete_parts)
