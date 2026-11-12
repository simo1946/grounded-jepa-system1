"""Simulated sensors for System 1 (robot proprioception, object-position detector).

A ``SensorSpec`` fixes which channels the policy receives besides the image
latents, and how they are corrupted:

  proprio    robot proprioception (task.state_cols): effector pose, gripper,
             joint angles; Gaussian noise ``proprio_noise`` (in z-score units)
  obj        current object position from a simulated detector: true position
             + N(0, obj_noise^2) metres per axis
  goal_obj   the goal's object position (goal given as coordinates rather than
             only as an image; a different, easier task specification, always
             reported separately)
  dropout    per sample and per channel group, the reading is missing with
             probability p (replaced by 0 and flagged by a mask bit)

Channels are z-scored with training-set statistics. The same spec drives
training (from the latent cache) and evaluation (from the env's info dict),
so the policy sees identical corruption at train and test time.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SensorSpec:
    proprio: bool = False
    obj: bool = False
    goal_obj: bool = False
    proprio_noise: float = 0.0      # z-score units
    obj_noise: float = 0.0          # metres
    dropout: float = 0.0

    @staticmethod
    def parse(s: str) -> "SensorSpec":
        """e.g. 'none', 'proprio', 'proprio+obj', 'proprio+obj+goalobj',
        options as key=value: 'proprio+obj+objnoise=0.01+drop=0.2'."""
        kw: dict = {}
        if s in ("", "none"):
            return SensorSpec()
        for t in s.split("+"):
            if t == "proprio":
                kw["proprio"] = True
            elif t == "obj":
                kw["obj"] = True
            elif t == "goalobj":
                kw["goal_obj"] = True
            elif t.startswith("objnoise="):
                kw["obj_noise"] = float(t.split("=", 1)[1])
            elif t.startswith("pnoise="):
                kw["proprio_noise"] = float(t.split("=", 1)[1])
            elif t.startswith("drop="):
                kw["dropout"] = float(t.split("=", 1)[1])
            else:
                raise ValueError(f"unknown sensor token {t!r}")
        spec = SensorSpec(**kw)
        if spec.goal_obj and not spec.obj:
            raise ValueError("'goalobj' needs 'obj'")
        return spec

    def name(self) -> str:
        if not (self.proprio or self.obj):
            return "none"
        parts = (["proprio"] if self.proprio else []) + (["obj"] if self.obj else []) + \
                (["goalobj"] if self.goal_obj else [])
        if self.proprio_noise:
            parts.append(f"pnoise={self.proprio_noise:g}")
        if self.obj_noise:
            parts.append(f"objnoise={self.obj_noise:g}")
        if self.dropout:
            parts.append(f"drop={self.dropout:g}")
        return "+".join(parts)

    def groups(self) -> list[str]:
        return (["proprio"] if self.proprio else []) + (["obj"] if self.obj else []) + \
               (["goal_obj"] if self.goal_obj else [])

    def dim(self, proprio_dim: int, obj_dim: int = 3) -> int:
        sizes = {"proprio": proprio_dim, "obj": obj_dim, "goal_obj": obj_dim}
        g = self.groups()
        return sum(sizes[k] for k in g) + len(g)          # + one missing-flag per group

    def to_dict(self) -> dict:
        return asdict(self)


class SensorEncoder:
    """Builds the policy's ``extra`` vector from raw readings."""

    def __init__(self, spec: SensorSpec, stats: dict, generator: torch.Generator | None = None):
        self.spec, self.stats, self.gen = spec, stats, generator

    @staticmethod
    def fit_stats(proprio: torch.Tensor, obj: torch.Tensor) -> dict:
        f = lambda x: (x.mean(0), x.std(0).clamp_min(1e-6))  # noqa: E731
        (pm, ps), (om, os_) = f(proprio.float()), f(obj.float())
        return {"proprio_mean": pm, "proprio_std": ps, "obj_mean": om, "obj_std": os_}

    def __call__(self, proprio: torch.Tensor | None, obj: torch.Tensor | None,
                 goal_obj: torch.Tensor | None) -> torch.Tensor | None:
        """Raw readings [B, P], [B, 3], [B, 3] -> extra [B, E] (None if no channels)."""
        s, st, g = self.spec, self.stats, self.gen
        groups = s.groups()
        if not groups:
            return None
        ref = proprio if proprio is not None else obj
        B = ref.shape[0]
        out = []
        for name in groups:
            if name == "proprio":
                x = (proprio.float() - st["proprio_mean"]) / st["proprio_std"]
                if s.proprio_noise:
                    x = x + s.proprio_noise * torch.randn(x.shape, generator=g)
            else:
                raw = (obj if name == "obj" else goal_obj).float()
                if name == "obj" and s.obj_noise:
                    raw = raw + s.obj_noise * torch.randn(raw.shape, generator=g)
                x = (raw - st["obj_mean"]) / st["obj_std"]
            missing = torch.zeros(B, 1)
            if s.dropout and name != "goal_obj":         # the goal specification is never dropped
                missing = (torch.rand(B, 1, generator=g) < s.dropout).float()
                x = x * (1 - missing)
            out += [x, missing]
        return torch.cat(out, -1)


def info_key(col: str) -> str:
    """Dataset column -> env info key ('proprio_effector_pos' -> 'proprio/effector_pos')."""
    head, _, tail = col.partition("_")
    return f"{head}/{tail}"


def readings_from_info(info_dict: dict, state_cols, object_cols) -> tuple:
    """(proprio [n, P], obj [n, 3], goal_obj [n, 3]) from the planner's info dict."""
    def last(v):
        v = torch.as_tensor(np.asarray(v) if not torch.is_tensor(v) else v)
        v = v[:, -1] if v.ndim >= 3 else v
        return v.reshape(v.shape[0], -1).float()
    proprio = torch.cat([last(info_dict[info_key(c)]) for c in state_cols], -1)
    obj = torch.cat([last(info_dict[info_key(c)]) for c in object_cols], -1)
    goal = torch.cat([last(info_dict[f"goal_{c}"]) for c in object_cols], -1)
    return proprio, obj, goal
