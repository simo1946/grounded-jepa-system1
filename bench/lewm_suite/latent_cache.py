"""Encode every frame of a dataset once with a trained (frozen) world model.

System 1 policies are trained on frozen latents, so encoding each clip on
every training step (train_policy.py: ~48 min on 2 CPU cores) is wasted
work. The cache stores, per dataset row (env step):
  z        [N, D]  latent of the frame
  action   [N, A]  action, z-scored with the same scaler as training/planning
  proprio  [N, P]  robot proprioception (task.state_cols), raw
  object   [N, 3]  object position (task.object_cols), raw (privileged; used
                   only to simulate an object-position sensor)
  offsets, lengths per episode
Blocks of `frameskip` steps are assembled by the consumers exactly as
stable-worldmodel's HDF5Dataset does (checked in ``verify``).

    python latent_cache.py --spec lewm+idm+sa --seed 0 --data cube
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

torch.set_flush_denormal(True)

from models import HOME, Spec, TrainConfig, build_model  # noqa: E402
from tasks import get_task, task_of  # noqa: E402


def cache_path(spec_name: str, seed: int, data: str):
    return HOME / "cache" / f"{spec_name.replace('+', '_').replace('=', '-')}__seed{seed}__{data}.pt"


def block_actions(action: np.ndarray, rows: np.ndarray, frameskip: int) -> np.ndarray:
    """Action block leaving row r: concat of actions r .. r+frameskip-1 -> [len(rows), frameskip*A]."""
    idx = rows[:, None] + np.arange(frameskip)[None]
    return action[idx].reshape(len(rows), -1)


@torch.inference_mode()
def main(spec_str: str, seed: int, data: str, verify: bool = True, device: str = "cpu", workers: int = 0) -> None:
    import stable_worldmodel as swm
    from stable_pretraining import data as dt
    from stable_worldmodel.data.utils import get_scaler

    cfg = TrainConfig()
    task = get_task(task_of(data), cfg.img_size)
    spec = Spec.parse(spec_str)
    ckpt = HOME / "checkpoints" / f"{task.name}/{spec.name()}/seed{seed}"
    info = json.loads((ckpt / "train_info.json").read_text())
    wm = build_model(spec, task.action_dim, info["state_dim"], cfg)
    wm.load_state_dict(torch.load(ckpt / "weights.pt", map_location="cpu"))
    lewm = wm.lewm.eval().to(device)

    ds = swm.data.HDF5Dataset(data, cache_dir=str(HOME), num_steps=1, keys_to_load=["pixels"],
                              keys_to_cache=["action"])
    ds.transform = dt.transforms.Compose(
        dt.transforms.ToImage(**dt.dataset_stats.ImageNet, source="pixels", target="pixels"),
        dt.transforms.Resize(cfg.img_size, source="pixels", target="pixels"))
    # action scaler: identical to training (column_normalizer fitted on the TRAINING dataset)
    train_ds = ds if data == task.name else swm.data.HDF5Dataset(task.name, cache_dir=str(HOME), keys_to_cache=["action"])
    scaler = get_scaler("zscore")
    scaler.fit(np.array(train_ds.get_col_data("action")))
    action = scaler.transform(np.array(ds.get_col_data("action"))).astype(np.float32)

    offsets, lengths = np.asarray(ds.offsets), np.asarray(ds.lengths)
    n = int(offsets[-1] + lengths[-1])
    if len(ds) != n:
        raise RuntimeError(f"expected one clip per row ({n}), got {len(ds)}")
    t0, zs = time.time(), []
    # clips (num_steps=1) are indexed in row order, so a non-shuffled loader yields rows 0..n-1
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, num_workers=workers)
    for b, item in enumerate(loader):
        px = item["pixels"].to(device)                                    # [b, 1, C, H, W]
        zs.append(lewm.encode({"pixels": px})["emb"][:, 0].float().cpu())
        if b % 40 == 0:
            print(json.dumps({"rows": b * 256, "of": n, "min": round((time.time() - t0) / 60, 1)}), flush=True)
    z = torch.cat(zs)

    def col(name):
        a = np.asarray(ds.get_col_data(name), dtype=np.float32)
        return a.reshape(a.shape[0], -1)

    proprio = np.concatenate([col(c) for c in task.state_cols], 1)
    obj = (np.concatenate([col(c) for c in task.object_cols], 1) if task.object_cols
           else np.zeros((len(proprio), 0), dtype=np.float32))
    out = {"spec": spec.name(), "seed": seed, "data": data, "frameskip": cfg.frameskip,
           "z": z, "action": torch.from_numpy(action), "proprio": torch.from_numpy(proprio),
           "object": torch.from_numpy(obj), "offsets": torch.from_numpy(offsets.astype(np.int64)),
           "lengths": torch.from_numpy(lengths.astype(np.int64)),
           "state_cols": list(task.state_cols), "object_cols": list(task.object_cols)}

    if verify:   # the cache must reproduce the training loader's latents and action blocks exactly
        import stable_pretraining as spt
        from stable_pretraining import data as dt2
        from stable_worldmodel.data import column_normalizer
        chk = swm.data.HDF5Dataset(data, cache_dir=str(HOME), num_steps=2, frameskip=cfg.frameskip,
                                   keys_to_load=["pixels", "action"], keys_to_cache=["action"])
        img = dt2.transforms.Compose(
            dt2.transforms.ToImage(**dt2.dataset_stats.ImageNet, source="pixels", target="pixels"),
            dt2.transforms.Resize(cfg.img_size, source="pixels", target="pixels"))
        chk.transform = spt.data.transforms.Compose(img, column_normalizer(train_ds, "action", "action"))
        rng = np.random.default_rng(0)
        worst_a, worst_z = 0.0, 0.0
        for i in rng.choice(len(chk), 20, replace=False):
            s = chk[int(i)]
            ep, st = chk.clip_indices[int(i)]
            row = int(offsets[ep] + st)
            zc = lewm.encode({"pixels": s["pixels"][None, 1:2].to(device)})["emb"][0, 0].float().cpu()
            # relative error: a wrong row gives ~1; batched vs single-frame numerics (TF32 on GPU) stay < 1e-2
            worst_z = max(worst_z, float((z[row + cfg.frameskip] - zc).norm() / zc.norm().clamp_min(1e-8)))
            a_ref = s["action"][0].numpy()
            a_mine = block_actions(action, np.array([row]), cfg.frameskip)[0]
            if np.isfinite(a_ref).all():
                worst_a = max(worst_a, float(np.abs(a_ref - a_mine).max()))
        out["verify"] = {"max_rel_latent_diff": worst_z, "max_abs_action_block_diff": worst_a}
        print(json.dumps(out["verify"]), flush=True)
        if worst_a > 1e-4 or worst_z > 2e-2:
            raise RuntimeError(f"cache does not match the training loader: {out['verify']}")

    path = cache_path(spec.name(), seed, data)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")   # atomic: a cut-off run never leaves a partial cache
    torch.save(out, tmp)
    tmp.replace(path)
    print(json.dumps({"saved": str(path), "rows": n, "minutes": round((time.time() - t0) / 60, 1)}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="cube")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=0, help="DataLoader workers for reading frames")
    a = ap.parse_args()
    main(a.spec, a.seed, a.data, not a.no_verify, a.device, a.workers)
