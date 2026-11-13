"""Train a LeWM-family world model on a suite task.

Loss = authors' LeWM loss (port of ``lejepa_forward``):
    MSE(predict(emb[:, :H], act[:, :H]), emb[:, 1:]) + 0.09 * SIGReg(emb^T)
  + idm_weight * L_inverse_dynamics   (if spec has ``idm``)
  + sa_weight  * L_state_alignment    (if spec has ``sa``; robot proprioception only)
  + arl_weight * L_action_resolution  (if spec has ``arl``; hybrid_wm.grounding)
  + MoE balance term                  (hybrid predictor only)

    python train.py --task cube --spec lewm --seed 0 --steps 2000
    python train.py --task cube --spec lewm+idm+sa --seed 0 --steps 2000
    python train.py --task cube --spec hybrid+idm+sa+obj+arl --seed 0 --steps 10000 --batch-size 128
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time

import torch

torch.set_flush_denormal(True)

from models import HOME, Spec, TrainConfig, build_model, cfg_dict, count, state_columns  # noqa: E402
from tasks import get_task  # noqa: E402


def make_dataset(task, cfg: TrainConfig, name: str, state_cols: tuple):
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from stable_pretraining import data as dt
    from stable_worldmodel.data import column_normalizer

    ds = swm.data.HDF5Dataset(name, cache_dir=str(HOME), num_steps=cfg.num_preds + cfg.history_size,
                              frameskip=cfg.frameskip, keys_to_load=["pixels", "action"],
                              keys_to_cache=["action", *state_cols],
                              keys_to_merge={"state": list(state_cols)})
    img = dt.transforms.Compose(dt.transforms.ToImage(**dt.dataset_stats.ImageNet, source="pixels", target="pixels"),
                                dt.transforms.Resize(cfg.img_size, source="pixels", target="pixels"))
    ds.transform = spt.data.transforms.Compose(img, column_normalizer(ds, "action", "action"),
                                               column_normalizer(ds, "state", "state"))
    return ds


def losses(model, sigreg, batch, cfg: TrainConfig, device: str):
    batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
    raw_action = batch["action"]
    action_valid = torch.isfinite(raw_action).all(-1)          # boundary NaNs (masked for the IDM)
    batch["action"] = torch.nan_to_num(raw_action, 0.0)        # authors' handling for the predictor
    lewm = model.lewm
    out = lewm.encode({"pixels": batch["pixels"], "action": batch["action"]})
    emb, act_emb = out["emb"], out["act_emb"]
    Hs, target = cfg.history_size, emb[:, cfg.num_preds:]
    if getattr(model, "arl", False):
        from hybrid_wm.grounding import action_resolution_loss, perturb_actions
        # ONE predictor pass over the executed and the perturbed action sequences, so every candidate shares
        # the same BatchNorm statistics; the prediction loss uses the executed part (index 0)
        cand = perturb_actions(batch["action"][:, :Hs], cfg.arl_candidates, cfg.arl_sigma_min, cfg.arl_sigma_max)
        B, K = cand.shape[:2]
        c0 = emb[:, :Hs]
        ctx = c0.unsqueeze(1).expand(B, K, *c0.shape[1:]).reshape(B * K, *c0.shape[1:])
        pred_all = lewm.predict(ctx, lewm.action_encoder(cand.reshape(B * K, *cand.shape[2:]))).reshape(B, K, Hs, -1)
        pred = pred_all[:, 0]
    else:
        pred = lewm.predict(emb[:, :Hs], act_emb[:, :Hs])
    aux = getattr(lewm.predictor, "aux_loss", None)
    parts = {"pred": (pred - target).pow(2).mean(), "sigreg": sigreg(emb.transpose(0, 1))}
    total = parts["pred"] + cfg.sigreg_weight * parts["sigreg"]
    if getattr(model, "arl", False):
        parts["arl"], parts["arl_acc"] = action_resolution_loss(pred_all, target, cfg.arl_gamma)
        total = total + cfg.arl_weight * parts["arl"]
    if model.idm_head is not None:
        parts["idm"] = model.idm_head.loss(emb, raw_action, action_valid)
        total = total + cfg.idm_weight * parts["idm"]
    if model.sa_head is not None:
        parts["sa"] = model.sa_head.loss(emb, batch["state"].float())
        total = total + cfg.sa_weight * parts["sa"]
    if aux is not None:
        total = total + aux
    return total, parts


def main(task_name: str, spec_str: str, seed: int, steps: int, device: str, data: str,
         batch_size: int | None = None, lr: float | None = None, num_workers: int = 2) -> None:
    from stable_worldmodel.wm.loss import SIGReg

    cfg = TrainConfig()
    over = {k: v for k, v in (("batch_size", batch_size), ("lr", lr)) if v is not None}
    cfg = dataclasses.replace(cfg, **over) if over else cfg
    task = get_task(task_name, cfg.img_size)
    spec = Spec.parse(spec_str)
    run = f"{task_name}/{spec.name()}/seed{seed}"
    torch.manual_seed(seed)
    ds = make_dataset(task, cfg, data, state_columns(task, spec))
    n_val = int(len(ds) * (1 - cfg.train_split))
    split_g = torch.Generator().manual_seed(3072)               # same split for every run
    train_set, val_set = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val], generator=split_g)
    g = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                                         num_workers=num_workers, generator=g,
                                         persistent_workers=num_workers > 0, pin_memory=device.startswith("cuda"))
    state_dim = int(ds.get_dim("state"))
    model = build_model(spec, task.action_dim, state_dim, cfg).to(device).train()
    sigreg = SIGReg(knots=17, num_proj=cfg.sigreg_proj).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    info = {"run": run, "task": task_name, "spec": spec.name(), "seed": seed, "steps": steps, "dataset": data,
            "params_total": count(model), "params_lewm": count(model.lewm),
            "params_encoder": count(model.lewm.encoder), "state_dim": state_dim,
            "train_clips": len(train_set), "val_clips": n_val, "config": cfg_dict(cfg), "task_notes": task.notes}
    print(json.dumps(info), flush=True)

    out_dir = HOME / "checkpoints" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    # resume state on LOCAL disk: it is rewritten every 250 steps, and on a mounted Google Drive every
    # rewrite leaves the old copy in the Drive bin (counted against the quota). Only weights.pt goes to Drive.
    resume = HOME / "resume" / run / "resume.pt"
    resume.parent.mkdir(parents=True, exist_ok=True)
    legacy = out_dir / "resume.pt"                                # older versions kept it next to the weights
    step, log, t0 = 0, [], time.time()
    src = resume if resume.exists() else (legacy if legacy.exists() else None)
    if src is not None:
        st = torch.load(src, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step, log = st["step"], st["log"]
        torch.set_rng_state(st["rng"])
        info["resumed_at_steps"] = st.get("resumed_at_steps", []) + [step]   # provenance: a resumed run is
        print(json.dumps({"resumed_at_step": step}), flush=True)             # valid but not bit-identical
    t_last, it = time.time(), iter(loader)
    while step < steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        total, parts = losses(model, sigreg, batch, cfg, device)
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        if hasattr(model.lewm.predictor, "update_routing_biases"):
            model.lewm.predictor.update_routing_biases()
        step += 1
        if step % 50 == 0 or step == steps:
            row = {"step": step, "loss": round(total.item(), 4), **{k: round(v.item(), 4) for k, v in parts.items()},
                   "min": round((time.time() - t0) / 60, 1), "s_per_step": round((time.time() - t_last) / 50, 2)}
            t_last = time.time()
            log.append(row)
            print(json.dumps(row), flush=True)
        if step % 250 == 0 and step < steps:
            tmp_r = resume.with_name("resume.pt.tmp")             # atomic, like the final weights
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step, "log": log, "rng": torch.get_rng_state(),
                        "resumed_at_steps": info.get("resumed_at_steps", [])}, tmp_r)
            tmp_r.replace(resume)

    model.eval()
    sums, n = {}, 0
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(val_set, batch_size=64, num_workers=num_workers):
            _, parts = losses(model, sigreg, batch, cfg, device)
            b = len(batch["action"])
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v.item() * b
            n += b
    info.update({"val": {k: v / n for k, v in sums.items()}, "train_minutes": round((time.time() - t0) / 60, 1),
                 "log": log})
    tmp = out_dir / "weights.pt.tmp"                             # atomic: a partial file never looks finished
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, tmp)
    tmp.replace(out_dir / "weights.pt")
    (out_dir / "train_info.json").write_text(json.dumps(info, indent=2))
    resume.unlink(missing_ok=True)
    legacy.unlink(missing_ok=True)
    print(json.dumps({"run": run, "val": info["val"], "train_minutes": info["train_minutes"]}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["tworoom", "cube"])
    ap.add_argument("--spec", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--data", default=None, help="dataset name (default: task name)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=None, help="override TrainConfig.batch_size (32)")
    ap.add_argument("--lr", type=float, default=None, help="override TrainConfig.lr (3e-4)")
    ap.add_argument("--num-workers", type=int, default=2)
    a = ap.parse_args()
    main(a.task, a.spec, a.seed, a.steps, a.device, a.data or a.task, a.batch_size, a.lr, a.num_workers)
