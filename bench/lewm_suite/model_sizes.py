"""Parameter counts of every model the paper describes, from the model code itself (one JSON object on stdout).

World models are built with models.build_model for each configuration (training-only grounding heads
included and also counted separately); policies with their constructors at the sizes the study used.
Counts are exact integers; the paper prints them in millions.

    python model_sizes.py
"""
from __future__ import annotations

import os

import torch

os.environ.setdefault("HWM_IMG_SIZE", "64")

from models import Spec, TrainConfig, build_model, state_columns  # noqa: E402
from study_configs import CONFIGS  # noqa: E402
from tasks import get_task  # noqa: E402

STATE_DIMS = {"proprio_effector_pos": 3, "proprio_effector_yaw": 1, "proprio_gripper_opening": 1,
              "proprio_joint_pos": 6, "privileged_block_0_pos": 3}   # column widths in the Cube data
ACTION_DIM = 5


def count(module: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def world_models() -> dict:
    cfg, task = TrainConfig(), get_task("cube", TrainConfig().img_size)
    out = {}
    for c in ("C1", "C2", "C3", "C5", "C6", "C7"):
        spec = Spec.parse(CONFIGS[c])
        state_dim = sum(STATE_DIMS[k] for k in state_columns(task, spec))
        torch.manual_seed(0)
        try:
            m = build_model(spec, ACTION_DIM, state_dim, cfg)
        except ImportError as e:                          # the hybrid needs mamba-ssm or the CPU shim
            out[c] = {"error": str(e)}
            continue
        heads = sum(count(getattr(m, h)) for h in ("idm_head", "sa_head") if getattr(m, h, None) is not None)
        out[c] = {"spec": CONFIGS[c], "params": count(m), "grounding_head_params": heads}
    return out


def policies() -> dict:
    from gcidm import GCIDM

    from hybrid_wm.control.latent_policy import ChunkPolicy
    d = TrainConfig().embed_dim
    s1 = ChunkPolicy(d, 5, 5 * ACTION_DIM)
    c5, c1 = ChunkPolicy(d, 1, 5 * ACTION_DIM), ChunkPolicy(d, 1, ACTION_DIM)
    return {"system1": count(s1), "short_chunk_5": count(c5), "short_chunk_1": count(c1),
            "gcidm_ours": count(GCIDM(d, ACTION_DIM)), "gcidm_ours_no_horizon": count(GCIDM(d, ACTION_DIM, horizon=False))}


def official_gcidm(repo: str | None) -> dict | None:
    """The GC-IDM authors' model at the settings used (official_gcidm.py), built from their code if cloned."""
    if not repo or not os.path.isdir(repo):
        return None
    import sys
    sys.path.insert(0, repo)
    from idm.model import GoalConditionedIDM, IDMConfig
    cfg = IDMConfig(embed_dim=TrainConfig().embed_dim, action_dim=ACTION_DIM, frameskip=1, max_horizon=50)
    return {"params": count(GoalConditionedIDM(cfg)), "commit": __import__("official_gcidm").PINNED_COMMIT}


if __name__ == "__main__":
    from instrument_io import emit, quiet_stdout
    with quiet_stdout():                                   # stable-pretraining logs to stdout
        out = {"world_models": world_models(), "policies": policies()}
        off = official_gcidm(os.environ.get("GCIDM_REPO"))
        if off:
            out["official_gcidm"] = off
    emit(out)
