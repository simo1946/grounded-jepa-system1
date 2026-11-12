"""Train the reachability critic on a trained (frozen) world model's latents.

Clips of 11 consecutive blocks (frameskip 5); positives are (i <= j) pairs
labelled j - i, negatives pair frames from different clips.

    python train_reach.py --task cube --spec lewm+idm+sa --seed 0
"""
from __future__ import annotations

import argparse
import json
import time

import torch

torch.set_flush_denormal(True)

from models import HOME, Spec, TrainConfig, build_model  # noqa: E402
from tasks import get_task  # noqa: E402

CLIP_BLOCKS = 11


def main(task_name: str, spec_str: str, seed: int, steps: int, data: str) -> None:
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from stable_pretraining import data as dt

    from hybrid_wm.control.reachability import ReachabilityCritic

    cfg = TrainConfig()
    task = get_task(task_name, cfg.img_size)
    spec = Spec.parse(spec_str)
    run = f"{task_name}/{spec.name()}/seed{seed}"
    ckpt = HOME / "checkpoints" / run
    info = json.loads((ckpt / "train_info.json").read_text())
    wm = build_model(spec, task.action_dim, info["state_dim"], cfg)
    wm.load_state_dict(torch.load(ckpt / "weights.pt", map_location="cpu"))
    lewm = wm.lewm.eval().requires_grad_(False)

    ds = swm.data.HDF5Dataset(data, cache_dir=str(HOME), num_steps=CLIP_BLOCKS, frameskip=cfg.frameskip,
                              keys_to_load=["pixels"])
    ds.transform = dt.transforms.Compose(
        dt.transforms.ToImage(**dt.dataset_stats.ImageNet, source="pixels", target="pixels"),
        dt.transforms.Resize(cfg.img_size, source="pixels", target="pixels"))
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, drop_last=True, num_workers=2,
                                         generator=g, persistent_workers=True)
    crit = ReachabilityCritic(cfg.embed_dim, max_steps=CLIP_BLOCKS - 1)
    opt = torch.optim.AdamW(crit.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    log, t0, step, it = [], time.time(), 0, iter(loader)
    while step < steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        with torch.no_grad():
            emb = lewm.encode({"pixels": batch["pixels"]})["emb"]        # [B, 11, D]
        loss = crit.loss(emb, generator=g)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        step += 1
        if step % 100 == 0 or step == steps:
            with torch.no_grad():
                d = [crit(emb[:, 0], emb[:, k]).mean().item() for k in (0, 2, 5, 10)]
            log.append({"step": step, "ce": round(loss.item(), 4), "pred_steps@0/2/5/10": [round(x, 2) for x in d],
                        "min": round((time.time() - t0) / 60, 1)})
            print(json.dumps(log[-1]), flush=True)
    torch.save({"state_dict": crit.state_dict(), "latent_dim": cfg.embed_dim, "max_steps": CLIP_BLOCKS - 1,
                "log": log}, ckpt / "reach.pt")
    print(json.dumps({"run": run, "reach_saved": True, "minutes": round((time.time() - t0) / 60, 1)}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["tworoom", "cube"])
    ap.add_argument("--spec", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--data", default=None)
    a = ap.parse_args()
    main(a.task, a.spec, a.seed, a.steps, a.data or a.task)
