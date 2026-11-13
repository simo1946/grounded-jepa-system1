"""Hybrid predictor backbone: Mamba-3 + sliding-window attention, MoE FFNs,
dual memory, AdaLN-zero action conditioning, optional two-timescale stream.

Block (pre-norm, AdaLN-zero as in DiT / LeWM's predictor)::

    h += Engram(h, keys)                                   (selected layers)
    h += g1 * Mixer(norm(h) * (1 + s1) + b1)                (Mamba-3 or attention)
    h += Episodic(norm(h))                                  (selected layers)
    h += g2 * FFN(norm(h) * (1 + s2) + b2)                  (MoE or dense SwiGLU)
    (b1, s1, g1, b2, s2, g2) = Linear(cond), zero-initialised -> identity at init

Every component is causal, and ``forward`` with state == step-by-step
streaming (tested), so the same weights train in parallel and run online.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .config import BackboneConfig, Mamba3Config, MoEConfig
from .layers.attention import GatedWindowAttention
from .layers.mamba3 import Mamba3Mixer
from .layers.memory import EpisodicMemory, NGramMemory
from .layers.moe import MoEFFN, SwiGLU
from .state import State


class _DenseFFN(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.ffn = SwiGLU(d, hidden)

    def forward(self, x):
        return self.ffn(x), x.new_zeros((), dtype=torch.float32)


class Block(nn.Module):
    def __init__(self, d: int, kind: str, use_moe: bool, cond_dim: int, mamba: Mamba3Config,
                 attention, moe: MoEConfig, dense_hidden: int, engram=None, episodic=None, layer_id: int = 0):
        super().__init__()
        self.kind = kind
        self.mixer = Mamba3Mixer(d, mamba) if kind == "M" else GatedWindowAttention(d, attention)
        self.ffn = MoEFFN(d, moe) if use_moe else _DenseFFN(d, dense_hidden)
        self.norm1 = nn.RMSNorm(d, elementwise_affine=False)
        self.norm2 = nn.RMSNorm(d, elementwise_affine=False)
        self.ada = nn.Linear(cond_dim, 6 * d)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)
        self.engram = NGramMemory(d, engram, layer_id) if engram is not None else None
        self.episodic = EpisodicMemory(d, episodic) if episodic is not None else None
        self.mem_norm = nn.RMSNorm(d) if episodic is not None else None

    def init_state(self, batch: int, device=None) -> State:
        st: State = {"mixer": self.mixer.init_state(batch, device)}
        if self.engram is not None:
            st["engram"] = self.engram.init_state(batch, device)
        if self.episodic is not None:
            st["episodic"] = self.episodic.init_state(batch, device)
        return st

    def forward(self, h, cond, t, keys, state: Optional[State]):
        st_in = state or {}
        st: State = {}
        if self.engram is not None:
            if keys is None:
                raise ValueError("this block has an Engram memory: pass integer `keys` [B, L]")
            e, st["engram"] = self.engram(h, keys, st_in.get("engram"))
            h = h + e
        b1, s1, g1, b2, s2, g2 = self.ada(cond).chunk(6, dim=-1)
        x = self.norm1(h) * (1 + s1) + b1
        if self.kind == "M":
            if state is not None and h.shape[1] == 1:
                y, st["mixer"] = self.mixer.step(x[:, 0], st_in["mixer"])
                y = y.unsqueeze(1)
            else:
                y, st["mixer"] = self.mixer(x, st_in.get("mixer"))
        else:
            y, st["mixer"] = self.mixer(x, t, st_in.get("mixer"))
        h = h + g1 * y
        if self.episodic is not None:
            r, st["episodic"] = self.episodic(self.mem_norm(h), st_in.get("episodic"))
            h = h + r
        f, aux = self.ffn(self.norm2(h) * (1 + s2) + b2)
        return h + g2 * f, st, aux


class HybridBackbone(nn.Module):
    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        def make(i: int, kind: str, use_moe: bool, with_mem: bool = True) -> Block:
            eng = cfg.engram if (with_mem and cfg.engram is not None and i in cfg.engram.layers) else None
            epi = cfg.episodic if (with_mem and cfg.episodic is not None and i in cfg.episodic.layers) else None
            return Block(d, kind, use_moe, cfg.cond_dim, cfg.mamba, cfg.attention, cfg.moe, cfg.dense_hidden,
                         eng, epi, layer_id=i)

        self.blocks = nn.ModuleList(
            make(i, cfg.layer_kind(i), cfg.ffn == "moe" and i >= cfg.first_dense_layers)
            for i in range(cfg.n_layers))
        self.hier = cfg.hierarchy
        if self.hier is not None:
            self.high_in = nn.Linear(d, d)
            self.high_blocks = nn.ModuleList(make(1000 + j, "M", False, with_mem=False)
                                             for j in range(self.hier.n_layers))
            self.high_cond = nn.Parameter(torch.zeros(cfg.cond_dim))
            self.high_out = nn.Linear(d, cfg.cond_dim)
            nn.init.zeros_(self.high_out.weight)
            nn.init.zeros_(self.high_out.bias)
            self.high_null = nn.Parameter(torch.zeros(cfg.cond_dim))

    # --------------------------------------------------------------- state
    def init_state(self, batch: int, device=None) -> State:
        st: State = {"pos": 0, "blocks": [b.init_state(batch, device) for b in self.blocks]}
        if self.hier is not None:
            st["high"] = {"blocks": [b.init_state(batch, device) for b in self.high_blocks],
                          "last": self.high_null.detach().expand(batch, -1).clone()}
        return st

    def moe_layers(self) -> List[MoEFFN]:
        return [m for m in self.modules() if isinstance(m, MoEFFN)]

    # ------------------------------------------------------------- forward
    def _high(self, h_low: torch.Tensor, state: Optional[State], pos0: int):
        """Run the slow stream on chunk-end tokens and return the per-token
        conditioning for the upper layers. Token i only receives a high-level
        output produced at a position strictly before i (causal)."""
        B, L, _ = h_low.shape
        hst = state["high"] if state is not None else None
        last = hst["last"] if hst is not None else self.high_null.expand(B, -1)
        bstates = list(hst["blocks"]) if hst is not None else [None] * len(self.high_blocks)
        gpos = torch.arange(L, device=h_low.device) + pos0
        ends = torch.nonzero((gpos + 1) % self.hier.stride == 0).squeeze(-1)
        if ends.numel():
            x = self.high_in(h_low[:, ends])
            c = self.high_cond.expand(B, x.shape[1], -1)
            tz = torch.zeros(B, x.shape[1], device=x.device)
            for j, blk in enumerate(self.high_blocks):
                x, bstates[j], _ = blk(x, c, tz, None, bstates[j])
            outs = self.high_out(x)                                          # [B, n_ends, cond]
            table, new_last = torch.cat([last.unsqueeze(1), outs], dim=1), outs[:, -1]
        else:
            table, new_last = last.unsqueeze(1), last
        idx = (ends.unsqueeze(0) < torch.arange(L, device=h_low.device).unsqueeze(1)).sum(1)
        bstates = [s if s is not None else blk.init_state(B, h_low.device)
                   for s, blk in zip(bstates, self.high_blocks, strict=True)]
        return table[:, idx], {"blocks": bstates, "last": new_last}

    def forward(self, h: torch.Tensor, cond: torch.Tensor, t: torch.Tensor,
                keys: Optional[torch.Tensor] = None, state: Optional[State] = None
                ) -> Tuple[torch.Tensor, State, torch.Tensor]:
        """h [B, L, D], cond [B, L, cond_dim], t [B, L] relative time, keys [B, L] int."""
        pos0 = state["pos"] if state is not None else 0
        aux = h.new_zeros((), dtype=torch.float32)
        new_blocks = []
        split = self.hier.split if self.hier is not None else len(self.blocks)
        new_state: State = {"pos": pos0 + h.shape[1]}
        for i, blk in enumerate(self.blocks):
            if self.hier is not None and i == split:
                inj, new_state["high"] = self._high(h, state, pos0)
                cond = cond + inj
            bst = state["blocks"][i] if state is not None else None
            h, s, a = blk(h, cond, t, keys, bst)
            new_blocks.append(s)
            aux = aux + a
        new_state["blocks"] = new_blocks
        return h, new_state, aux
