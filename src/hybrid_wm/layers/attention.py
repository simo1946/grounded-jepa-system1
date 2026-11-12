"""Causal sliding-window attention for the hybrid backbone.

Why each piece is here:
* Sliding window + causal mask: the attention layers recover exact, recent
  recall (the known weakness of fixed-size SSM state) at bounded cost, so the
  whole stack stays O(L * window) and streams with a bounded KV cache.
* Continuous-time RoPE: rotations use the actual timestamp of each step, so
  irregularly sampled sensors/time-lapse frames get correct relative timing,
  and outputs are invariant to a global time shift.
* QK-norm (RMSNorm on per-head q and k) for stability.
* Element-wise, head-specific sigmoid gate on the SDPA output (the best
  placement reported in "Gated Attention for LLMs", NeurIPS 2025, used in the
  Qwen3-Next/3.5 line).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AttentionConfig
from ..state import State


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class ContinuousRoPE(nn.Module):
    """RoPE driven by real time. Periods are geometric between the configured
    min and max (in model time units); angle = 2*pi * t / period."""

    def __init__(self, head_dim: int, min_period: float, max_period: float):
        super().__init__()
        n = head_dim // 2
        periods = torch.exp(torch.linspace(math.log(min_period), math.log(max_period), n))
        self.register_buffer("omega", 2 * math.pi / periods, persistent=False)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x: [B, L, H, hd], t: [B, L] (relative time, float32 is safe)."""
        ang = t.to(torch.float32).unsqueeze(-1) * self.omega            # [B, L, hd/2]
        cos = torch.cat([ang.cos(), ang.cos()], -1).unsqueeze(2).to(x.dtype)
        sin = torch.cat([ang.sin(), ang.sin()], -1).unsqueeze(2).to(x.dtype)
        return x * cos + _rotate_half(x) * sin


class GatedWindowAttention(nn.Module):
    def __init__(self, d_model: int, cfg: AttentionConfig):
        super().__init__()
        self.cfg = cfg
        self.h, self.hkv = cfg.n_heads, cfg.n_kv_heads
        self.hd = d_model // cfg.n_heads
        self.q_proj = nn.Linear(d_model, self.h * self.hd, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * self.hkv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.h * self.hd, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, self.h * self.hd) if cfg.output_gate else None
        self.q_norm = nn.RMSNorm(self.hd) if cfg.qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.hd) if cfg.qk_norm else nn.Identity()
        self.rope = ContinuousRoPE(self.hd, cfg.rope_min_period, cfg.rope_max_period)

    def init_state(self, batch: int, device=None, dtype=None) -> State:
        device = device or self.q_proj.weight.device
        dtype = dtype or self.q_proj.weight.dtype
        empty = torch.zeros(batch, self.hkv, 0, self.hd, device=device, dtype=dtype)
        return {"k": empty, "v": empty.clone()}

    def forward(self, x: torch.Tensor, t: torch.Tensor, state: Optional[State] = None
                ) -> Tuple[torch.Tensor, State]:
        """x: [B, L, D]; t: [B, L] relative time. With ``state`` the new tokens
        also attend to the cached (already rotated) keys of the window."""
        B, L, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(B, L, self.h, self.hd))
        k, v = self.kv_proj(x).view(B, L, 2, self.hkv, self.hd).unbind(2)
        k = self.k_norm(k)
        q, k = self.rope(q, t), self.rope(k, t)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)    # [B, H, L, hd]

        if state is None:
            state = self.init_state(B, x.device, k.dtype)
        k_all = torch.cat([state["k"], k], dim=2)
        v_all = torch.cat([state["v"], v], dim=2)
        C = state["k"].shape[2]

        qi = torch.arange(L, device=x.device).unsqueeze(1) + C                 # absolute query index
        kj = torch.arange(C + L, device=x.device).unsqueeze(0)
        mask = (kj <= qi) & (qi - kj < self.cfg.window)                         # [L, C+L]

        rep = self.h // self.hkv
        ka = k_all.repeat_interleave(rep, dim=1) if rep > 1 else k_all
        va = v_all.repeat_interleave(rep, dim=1) if rep > 1 else v_all
        y = F.scaled_dot_product_attention(q, ka, va, attn_mask=mask)          # [B, H, L, hd]
        y = y.transpose(1, 2).reshape(B, L, self.h * self.hd)
        if self.gate_proj is not None:
            y = y * torch.sigmoid(self.gate_proj(x))
        keep = self.cfg.window - 1                                              # keys the next token may see
        new_state = {"k": k_all[:, :, -keep:] if keep > 0 else k_all[:, :, :0],
                     "v": v_all[:, :, -keep:] if keep > 0 else v_all[:, :, :0]}
        return self.o_proj(y), new_state

    def step(self, x: torch.Tensor, t: torch.Tensor, state: State) -> Tuple[torch.Tensor, State]:
        """x: [B, D], t: [B]."""
        y, state = self.forward(x.unsqueeze(1), t.unsqueeze(1), state)
        return y[:, 0], state
