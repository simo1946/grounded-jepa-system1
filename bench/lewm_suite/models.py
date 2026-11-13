"""Model construction: the authors' LeWM wrapper with swappable predictor,
encoder and grounding heads.

Spec string, e.g. ``lewm``, ``lewm+idm``, ``lewm+idm+sa``,
``lewm+idm+sa+enc=mnv4s``, ``hybrid+idm``:
  * first token: predictor (``lewm`` transformer | ``hybrid`` ours)
  * ``idm``: inverse-dynamics loss, ``sa``: proprioceptive state alignment
  * ``obj`` (with ``sa``): the alignment target also includes the task's
    object state (training-time supervision only; planning sees pixels only)
  * ``arl``: action-resolution loss (train.py): the predictor must rank the
    executed action above nearby perturbed actions by how well each predicts
    the observed next latent
Image size: ``HWM_IMG_SIZE`` (default 64; patch = size / 8) applies to every
script, so data, training and evaluation stay consistent.
  * ``enc=<name>``: encoder (default ``vit_tiny`` = LeWM's from-scratch ViT-tiny)
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
os.environ.setdefault("STABLEWM_HOME", str(HERE / "swm_home"))
HOME = Path(os.environ["STABLEWM_HOME"])

ENCODERS = {
    "vit_tiny": None,                                   # LeWM default (stable-pretraining vit_hf tiny)
    "mnv4s": "mobilenetv4_conv_small",                  # 8.4 ms @224 on 2 CPU cores (bench/results)
    "mobileone_s0": "mobileone_s0",
    "convnext_atto": "convnext_atto",
}


@dataclass(frozen=True)
class TrainConfig:
    img_size: int = int(os.environ.get("HWM_IMG_SIZE", 64))
    patch_size: int = int(os.environ.get("HWM_IMG_SIZE", 64)) // 8
    embed_dim: int = 192
    history_size: int = 3
    num_preds: int = 1
    frameskip: int = 5
    sigreg_weight: float = 0.09
    sigreg_proj: int = 1024
    idm_weight: float = 1.0          # SMWM's value for OGBench-Cube
    sa_weight: float = 1.0
    arl_weight: float = 1.0          # action-resolution loss (spec token 'arl'); fixed a priori
    arl_candidates: int = 4          # perturbed action sequences per clip
    arl_sigma_min: float = 0.05      # perturbation scale, log-uniform, normalised action units
    arl_sigma_max: float = 0.5
    arl_gamma: float = 0.1           # required margin: perturbed predictions >= (1 + gamma) x farther (squared)
    lr: float = 3e-4
    weight_decay: float = 1e-3
    batch_size: int = 32
    grad_clip: float = 1.0
    train_split: float = 0.9


@dataclass(frozen=True)
class Spec:
    predictor: str = "lewm"
    encoder: str = "vit_tiny"
    idm: bool = False
    sa: bool = False
    obj: bool = False
    arl: bool = False

    @staticmethod
    def parse(s: str) -> "Spec":
        toks = s.split("+")
        if toks[0] not in ("lewm", "hybrid"):
            raise ValueError(f"spec must start with 'lewm' or 'hybrid': {s!r}")
        kw = {"predictor": toks[0]}
        for t in toks[1:]:
            if t == "idm":
                kw["idm"] = True
            elif t == "sa":
                kw["sa"] = True
            elif t == "obj":
                kw["obj"] = True
            elif t == "arl":
                kw["arl"] = True
            elif t.startswith("enc="):
                enc = t[4:]
                if enc not in ENCODERS:
                    raise ValueError(f"unknown encoder {enc!r}; choose from {sorted(ENCODERS)}")
                kw["encoder"] = enc
            else:
                raise ValueError(f"unknown spec token {t!r}")
        if kw.get("obj") and not kw.get("sa"):
            raise ValueError("'obj' extends the state-alignment target and needs 'sa'")
        return Spec(**kw)

    def name(self) -> str:
        parts = [self.predictor] + (["idm"] if self.idm else []) + (["sa"] if self.sa else []) + (["obj"] if self.obj else []) + \
                (["arl"] if self.arl else [])
        if self.encoder != "vit_tiny":
            parts.append(f"enc={self.encoder}")
        return "+".join(parts)


def state_columns(task, spec: "Spec") -> tuple:
    """Columns merged into the 'state' alignment target for this spec."""
    return tuple(task.state_cols) + (tuple(task.object_cols) if spec.obj else ())


class TimmFrameEncoder(nn.Module):
    """Adapts a timm CNN to the ViT interface LeWM.encode expects
    (``encoder(pixels, interpolate_pos_encoding=True).last_hidden_state[:, 0]``)."""

    def __init__(self, timm_id: str, out_dim: int):
        super().__init__()
        import timm
        self.body = timm.create_model(timm_id, pretrained=False, num_classes=0)
        was_training = self.body.training
        self.body.eval()
        with torch.no_grad():  # attributes like num_features can differ from the pooled output width
            width = self.body(torch.zeros(1, 3, 64, 64)).shape[-1]
        self.body.train(was_training)
        self.proj = nn.Linear(width, out_dim)

    def forward(self, pixels: torch.Tensor, **_):
        feat = self.proj(self.body(pixels))
        return SimpleNamespace(last_hidden_state=feat.unsqueeze(1))


class GroundedLeWM(nn.Module):
    """The authors' LeWM plus optional grounding heads. The heads are used only
    in training; planning uses ``.lewm`` unchanged."""

    def __init__(self, lewm: nn.Module, idm_head: Optional[nn.Module], sa_head: Optional[nn.Module],
                 arl: bool = False):
        super().__init__()
        self.lewm = lewm
        self.idm_head = idm_head
        self.sa_head = sa_head
        self.arl = arl                    # training-time flag only (no parameters)


def build_model(spec: Spec, action_dim: int, state_dim: int, cfg: TrainConfig) -> GroundedLeWM:
    from stable_pretraining.backbone.utils import vit_hf
    from stable_worldmodel.wm.lewm import LeWM
    from stable_worldmodel.wm.lewm.module import MLP, Embedder, Predictor

    from hybrid_wm.grounding import InverseDynamicsHead, StateAlignmentHead

    d = cfg.embed_dim
    if spec.encoder == "vit_tiny":
        encoder = vit_hf("tiny", patch_size=cfg.patch_size, image_size=cfg.img_size, pretrained=False,
                         use_mask_token=False)
    else:
        encoder = TimmFrameEncoder(ENCODERS[spec.encoder], d)
    if spec.predictor == "lewm":
        predictor = Predictor(num_frames=cfg.history_size, input_dim=d, hidden_dim=d, output_dim=d, depth=6,
                              heads=16, mlp_dim=2048, dim_head=64, dropout=0.1, emb_dropout=0.0)
    else:
        from hybrid_wm.integrations.lewm import HybridPredictor
        predictor = HybridPredictor(num_frames=cfg.history_size, input_dim=d, hidden_dim=d, output_dim=d)
    mlp = lambda: MLP(input_dim=d, output_dim=d, hidden_dim=2048, norm_fn=partial(nn.BatchNorm1d))  # noqa: E731
    act_block = cfg.frameskip * action_dim
    lewm = LeWM(encoder=encoder, predictor=predictor, action_encoder=Embedder(input_dim=act_block, emb_dim=d),
                projector=mlp(), pred_proj=mlp())
    return GroundedLeWM(lewm,
                        InverseDynamicsHead(d, act_block) if spec.idm else None,
                        StateAlignmentHead(d, state_dim) if spec.sa else None, arl=spec.arl)


def count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def cfg_dict(cfg) -> dict:
    return asdict(cfg)
