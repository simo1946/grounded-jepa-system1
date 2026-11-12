"""Train the System 1 latent policy on top of a trained (frozen) world model.

Hindsight relabelling on the same offline dataset: clips of H+1 frames
(frameskip 5), goal = frame k in 1..H, target = executed action blocks
(z-scored like the planner's actions), blocks after the goal masked.
Also measures tau_abs = median squared latent displacement over one block,
used by the verify-then-act gate.

    python train_policy.py --task cube --spec lewm+idm+sa --seed 0
"""
from __future__ import annotations

import argparse
import json
import time

import torch

torch.set_flush_denormal(True)

from models import HOME, Spec, TrainConfig, build_model  # noqa: E402
from tasks import get_task  # noqa: E402

HORIZON = 5  # planner horizon in blocks (le-wm eval config)


def make_clips(task, cfg: TrainConfig, name: str):
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from stable_pretraining import data as dt
    from stable_worldmodel.data import column_normalizer

    ds = swm.data.HDF5Dataset(name, cache_dir=str(HOME), num_steps=HORIZON + 1, frameskip=cfg.frameskip,
                              keys_to_load=["pixels", "action"], keys_to_cache=["action"])
    img = dt.transforms.Compose(dt.transforms.ToImage(**dt.dataset_stats.ImageNet, source="pixels", target="pixels"),
                                dt.transforms.Resize(cfg.img_size, source="pixels", target="pixels"))
    ds.transform = spt.data.transforms.Compose(img, column_normalizer(ds, "action", "action"))
    return ds


def main(task_name: str, spec_str: str, seed: int, steps: int, data: str) -> None:
    import numpy as np

    from hybrid_wm.control.latent_policy import ChunkPolicy, hindsight_batch

    cfg = TrainConfig()
    task = get_task(task_name, cfg.img_size)
    spec = Spec.parse(spec_str)
    run = f"{task_name}/{spec.name()}/seed{seed}"
    ckpt = HOME / "checkpoints" / run
    info = json.loads((ckpt / "train_info.json").read_text())
    wm = build_model(spec, task.action_dim, info["state_dim"], cfg)
    wm.load_state_dict(torch.load(ckpt / "weights.pt", map_location="cpu"))
    lewm = wm.lewm.eval().requires_grad_(False)

    torch.manual_seed(seed)
    ds = make_clips(task, cfg, data)
    g = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True, drop_last=True, num_workers=2,
                                         generator=g, persistent_workers=True)
    act_block = cfg.frameskip * task.action_dim
    pol = ChunkPolicy(cfg.embed_dim, HORIZON, act_block)
    opt = torch.optim.AdamW(pol.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    disp, log, t0, step, it = [], [], time.time(), 0, iter(loader)
    while step < steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        with torch.no_grad():
            emb = lewm.encode({"pixels": batch["pixels"]})["emb"]               # [B, H+1, D]
            disp.append((emb[:, 1:] - emb[:, :-1]).square().sum(-1).flatten())
        z_t, z_g, chunk, mask = hindsight_batch(emb, batch["action"].float(), HORIZON, g)
        loss = pol.loss(z_t, z_g, chunk, mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 100 == 0 or step == steps:
            log.append({"step": step, "nll": round(loss.item(), 4), "min": round((time.time() - t0) / 60, 1)})
            print(json.dumps(log[-1]), flush=True)
    d = torch.cat(disp)
    tau_abs = float(d[torch.isfinite(d)].median())
    torch.save({"state_dict": pol.state_dict(), "latent_dim": cfg.embed_dim, "horizon": HORIZON,
                "action_dim": act_block, "tau_abs": tau_abs, "log": log}, ckpt / "policy.pt")
    print(json.dumps({"run": run, "policy_saved": True, "tau_abs": tau_abs,
                      "minutes": round((time.time() - t0) / 60, 1)}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["tworoom", "cube"])
    ap.add_argument("--spec", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--data", default=None)
    a = ap.parse_args()
    main(a.task, a.spec, a.seed, a.steps, a.data or a.task)
