"""Does the latent encode what the Cube task is judged on (the cube position)?

Success on Cube is ||cube_pos - goal_cube_pos|| <= 0.04 m; the arm pose does
not count. A latent-goal planner can only move the cube to the goal if the
latent (and the world model's predictions of it) carry the cube position
well below that tolerance.

Linear (ridge) probes, fitted on TRAIN frames and tested on held-out EVAL
clips, read out
  * cube position (privileged_block_0_pos) and
  * end-effector position (proprio_effector_pos)
from (a) encoded real frames and (b) the world model's 5-block rollout under
the true actions. Errors are in metres; a constant-mean probe is the floor.
Also reports the share of the latent goal distance explained by the arm vs
the cube (regressing ||z_0 - z_5||^2 on squared arm and cube displacement).

    python probe_diag.py --spec lewm+idm+sa --seed 0
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

torch.set_flush_denormal(True)

from models import HOME, Spec, TrainConfig, build_model  # noqa: E402
from tasks import get_task  # noqa: E402

H = 5
CUBE, EFF = "privileged_block_0_pos", "proprio_effector_pos"


def _load(ds, n, rng, need_actions):
    order = rng.permutation(len(ds))
    out = {"pixels": [], "action": [], CUBE: [], EFF: []}
    for i in order:
        s = ds[int(i)]
        if need_actions and not torch.isfinite(s["action"][:H]).all():
            continue
        for key in out:
            if key in s:
                out[key].append(s[key] if key != "action" else s[key][:H].float())
        if len(out["pixels"]) == n:
            break
    return {k: torch.stack(v) for k, v in out.items() if v}


def ridge(x: torch.Tensor, y: torch.Tensor, lam: float = 1.0):
    mx, my = x.mean(0), y.mean(0)
    xc = x - mx
    w = torch.linalg.solve(xc.T @ xc + lam * torch.eye(x.shape[1], dtype=x.dtype), xc.T @ (y - my))
    return lambda q: (q - mx) @ w + my


@torch.inference_mode()
def main(spec_str: str, seed: int, n_fit: int, n_test: int) -> dict:
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from stable_pretraining import data as dt
    from stable_worldmodel.data import column_normalizer

    torch.set_num_threads(1)
    cfg = TrainConfig()
    task = get_task("cube", cfg.img_size)
    spec = Spec.parse(spec_str)
    ckpt = HOME / "checkpoints" / f"cube/{spec.name()}/seed{seed}"
    info = json.loads((ckpt / "train_info.json").read_text())
    wm = build_model(spec, task.action_dim, info["state_dim"], cfg)
    wm.load_state_dict(torch.load(ckpt / "weights.pt", map_location="cpu"))
    lewm = wm.lewm.eval()
    img = dt.transforms.Compose(dt.transforms.ToImage(**dt.dataset_stats.ImageNet, source="pixels", target="pixels"),
                                dt.transforms.Resize(cfg.img_size, source="pixels", target="pixels"))
    keys = ["pixels", "action", CUBE, EFF]
    fit_ds = swm.data.HDF5Dataset("cube", cache_dir=str(HOME), num_steps=1, keys_to_load=keys,
                                  keys_to_cache=["action"])
    fit_ds.transform = img
    test_ds = swm.data.HDF5Dataset("cube_eval", cache_dir=str(HOME), num_steps=H + 1, frameskip=cfg.frameskip,
                                   keys_to_load=keys, keys_to_cache=["action"])
    test_ds.transform = spt.data.transforms.Compose(img, column_normalizer(fit_ds, "action", "action"))
    rng = np.random.default_rng(0)
    fit = _load(fit_ds, n_fit, rng, need_actions=False)
    test = _load(test_ds, n_test, rng, need_actions=True)

    enc = lambda p: torch.cat([lewm.encode({"pixels": p[i:i + 64]})["emb"] for i in range(0, len(p), 64)])  # noqa: E731
    zf = enc(fit["pixels"])[:, 0].double()
    zt = enc(test["pixels"]).double()                                  # [N, 6, D]
    preds = [lewm.rollout({"pixels": test["pixels"][i:i + 16, None, :1]}, test["action"][i:i + 16, None])
             ["predicted_emb"][:, 0] for i in range(0, len(zt), 16)]
    zp = torch.cat(preds).double()                                     # [N, 6, D] (index 0 = context)

    res = {"spec": spec_str, "seed": seed, "fit_frames": len(zf), "test_clips": len(zt)}
    for name, col in (("cube", CUBE), ("effector", EFF)):
        y_fit = fit[col][:, 0].double()
        y = test[col].double()                                         # [N, 6, 3]
        probe = ridge(zf, y_fit)
        err = lambda a, b: float((a - b).norm(dim=-1).mean())          # noqa: E731
        res[name] = {
            "probe_err_real_frames_m": err(probe(zt[:, 0]), y[:, 0]),
            "probe_err_predicted_k5_m": err(probe(zp[:, H]), y[:, H]),
            "mean_baseline_err_m": err(y_fit.mean(0).expand_as(y[:, 0]), y[:, 0]),
            "true_displacement_0_to_5_m": err(y[:, 0], y[:, H]),
            "share_clips_moving_gt_4cm": float(((y[:, 0] - y[:, H]).norm(dim=-1) > 0.04).double().mean()),
        }
    # what dominates the latent goal distance?
    dz = (zt[:, 0] - zt[:, H]).square().sum(-1)
    da = (test[EFF][:, 0] - test[EFF][:, H]).double().square().sum(-1)
    dc = (test[CUBE][:, 0] - test[CUBE][:, H]).double().square().sum(-1)
    X = torch.stack([da, dc, torch.ones_like(da)], 1)
    coef = torch.linalg.lstsq(X, dz[:, None]).solution[:, 0]
    r2 = lambda yhat: float(1 - ((dz - yhat).square().sum() / (dz - dz.mean()).square().sum()))  # noqa: E731
    res["latent_dist_vs_displacements"] = {
        "r2_arm_only": r2(torch.linalg.lstsq(X[:, [0, 2]], dz[:, None]).solution[:, 0] @ X[:, [0, 2]].T),
        "r2_cube_only": r2(torch.linalg.lstsq(X[:, [1, 2]], dz[:, None]).solution[:, 0] @ X[:, [1, 2]].T),
        "r2_both": r2(X @ coef), "coef_arm_cube_const": coef.tolist()}
    print(json.dumps(res, indent=2))
    (HOME / "results" / f"probe_diag__{spec.name()}__seed{seed}.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="lewm+idm+sa")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fit", type=int, default=2000)
    ap.add_argument("--test", type=int, default=256)
    a = ap.parse_args()
    main(a.spec, a.seed, a.fit, a.test)
