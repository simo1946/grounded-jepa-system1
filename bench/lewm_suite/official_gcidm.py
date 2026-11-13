"""Run the GC-IDM authors' own code on this study's latents.

Their repository (github.com/hdnndh/Latent-Geometry-Beyond-Search-Amortizing-Planning-in-World-Models) is
licensed CC BY-NC 4.0, so it is not copied here. Clone it separately and check out the pinned commit:

    git clone https://github.com/hdnndh/Latent-Geometry-Beyond-Search-Amortizing-Planning-in-World-Models gcidm_repo
    git -C gcidm_repo checkout 48c45b1cb2b34dd2c1c61d222c8309de567fde55

Then, per world model:

    python official_gcidm.py export --spec lewm --seed 0                       # latents -> their .npz format
    python official_gcidm.py train  --spec lewm --seed 0 --repo gcidm_repo     # their train_idm.py, unmodified
    python official_gcidm.py train  --spec lewm --seed 0 --repo gcidm_repo --max-goal-horizon 1   # their ablation

The export writes our frozen latents of the training episodes (the same rows as the latent cache) with the
raw, unstandardised actions and episode ids, which is what their extract step produces from a LeWM encoder.
Training uses their Cube settings from their README (50 epochs, lr 1e-3, batch 1,024, action dim 5,
frameskip 1). Evaluation (evaluate.py --controller gcidm_official) loads their model with their load_idm,
encodes the current and goal frames with our encoder, passes min(budget - steps done, H_max) as in their
GoalConditionedPolicy, and standardises their raw action for the evaluation pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

PINNED_COMMIT = "48c45b1cb2b34dd2c1c61d222c8309de567fde55"


def npz_path(spec_name: str, seed: int, task: str = "cube") -> Path:
    from models import HOME
    return HOME / "cache" / "gcidm_official" / f"{spec_name.replace('+', '_')}__seed{seed}__{task}.npz"


def ckpt_name(max_goal_horizon: int) -> str:
    return "official_gcidm.pt" if max_goal_horizon == 50 else f"official_gcidm_h{max_goal_horizon}.pt"


def export(spec_str: str, seed: int, task: str = "cube") -> Path:
    import stable_worldmodel as swm
    from sklearn import preprocessing

    from latent_cache import cache_path
    from models import HOME, Spec
    spec = Spec.parse(spec_str)
    cache = torch.load(cache_path(spec.name(), seed, task), map_location="cpu")
    ds = swm.data.HDF5Dataset(task, cache_dir=str(HOME), keys_to_cache=["action"])
    raw = np.asarray(ds.get_col_data("action"), dtype=np.float32)
    if len(raw) != len(cache["z"]):
        raise RuntimeError("the latent cache and the dataset have different numbers of rows")
    scaler = preprocessing.StandardScaler().fit(raw)               # as latent_cache.py (NaN rows ignored)
    std = scaler.transform(raw).astype(np.float32)
    cached = cache["action"].numpy()
    ok = np.isfinite(std).all(1) & np.isfinite(cached).all(1)
    if not np.allclose(std[ok], cached[ok], atol=1e-4):             # same rows, same order as the cache
        raise RuntimeError("raw actions do not match the cached (standardised) actions")
    ep = np.zeros(len(raw), dtype=np.int64)
    for i, (off, length) in enumerate(zip(cache["offsets"].tolist(), cache["lengths"].tolist(), strict=True)):
        ep[off:off + length] = i
    out = npz_path(spec.name(), seed, task)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, embeddings=cache["z"].float().numpy(), actions=raw, episode_ids=ep,
                        embed_dim=cache["z"].shape[1])
    print(json.dumps({"saved": str(out), "rows": len(raw), "episodes": int(ep.max() + 1),
                      "finite_action_rows": int(np.isfinite(raw).all(1).sum())}), flush=True)
    return out


def check_repo(repo: Path) -> None:
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    if head != PINNED_COMMIT:
        raise RuntimeError(f"{repo} is at {head}, expected the pinned commit {PINNED_COMMIT}")
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"{repo} has local modifications; the authors' code must be unmodified")


def train(spec_str: str, seed: int, repo: Path, max_goal_horizon: int = 50, task: str = "cube",
          device: str = "cpu") -> Path:
    from models import HOME, Spec
    repo = repo.resolve()
    check_repo(repo)
    spec = Spec.parse(spec_str)
    out = HOME / "checkpoints" / f"{task}/{spec.name()}/seed{seed}" / ckpt_name(max_goal_horizon)
    cmd = [sys.executable, "train_idm.py", "train", "--embeddings", str(npz_path(spec.name(), seed, task)),
           "--output", str(out), "--action-dim", "5", "--frameskip", "1", "--epochs", "50", "--lr", "1e-3",
           "--batch-size", "1024", "--max-goal-horizon", str(max_goal_horizon), "--device", device]
    print(json.dumps({"running": " ".join(cmd[1:]), "cwd": str(repo)}), flush=True)
    subprocess.run(cmd, cwd=repo, check=True, env={**os.environ, "PYTHONPATH": str(repo)})
    print(json.dumps({"saved": str(out)}), flush=True)
    return out


def official_act(path: Path, budget: int, action_scaler, device: str | torch.device, repo: Path | None = None):
    """Controller for gcidm.LatentStepSolver backed by the authors' model (loaded with their load_idm)."""
    repo = Path(repo or os.environ.get("GCIDM_REPO", "gcidm_repo")).resolve()
    check_repo(repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from train_idm import load_idm  # the authors' loader, from their repository
    idm = load_idm(str(path), device=str(device))
    mean = torch.as_tensor(action_scaler.mean_, dtype=torch.float32, device=device)
    scale = torch.as_tensor(action_scaler.scale_, dtype=torch.float32, device=device)

    def act(z_now: torch.Tensor, z_goal: torch.Tensor, steps_done: int) -> torch.Tensor:
        remaining = min(budget - steps_done, idm.max_horizon)            # as in their GoalConditionedPolicy
        steps = torch.full((len(z_now),), remaining, device=z_now.device, dtype=torch.long)
        raw = idm(z_now, z_goal, steps)                                   # [n, 5] raw action
        return ((raw - mean) / scale)[:, None]                            # the pipeline un-standardises it
    return act


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["export", "train"])
    ap.add_argument("--spec", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", default="cube")
    ap.add_argument("--repo", type=Path, default=Path(os.environ.get("GCIDM_REPO", "gcidm_repo")))
    ap.add_argument("--max-goal-horizon", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if a.command == "export":
        export(a.spec, a.seed, a.task)
    else:
        train(a.spec, a.seed, a.repo, a.max_goal_horizon, a.task, a.device)
