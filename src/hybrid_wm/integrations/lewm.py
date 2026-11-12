"""Drop-in hybrid predictor for LeWorldModel (stable-worldmodel / le-wm).

LeWM's predictor contract: ``predictor(x, c) -> y`` with x, c, y of shape
[B, T, D] (frame embeddings, action embeddings) and a ``num_frames``
attribute (context window used by ``LeWM.rollout``). ``HybridPredictor``
satisfies it, so the authors' encoder, projectors, loss, rollout and CEM
planner stay untouched and *only the predictor differs* in comparisons.

Inside: the hybrid_wm backbone (Mamba-3 / causal window attention, AdaLN-zero
action conditioning, DeepSeek-style MoE FFNs), sized to match LeWM's
10.79M-parameter transformer predictor by default.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..backbone import HybridBackbone
from ..config import AttentionConfig, BackboneConfig, Mamba3Config, MoEConfig


class HybridPredictor(nn.Module):
    def __init__(self, *, num_frames: int, input_dim: int, hidden_dim: int, output_dim: int | None = None,
                 depth: int = 6, layout: str = "MMMA", attn_heads: int = 4, attn_kv_heads: int = 2,
                 d_state: int = 64, headdim: int = 32, n_routed: int = 8, top_k: int = 2, n_shared: int = 1,
                 expert_hidden: int = 256, dense_hidden: int = 2304, first_dense_layers: int = 1,
                 ffn: str = "moe", backend: str = "auto"):
        super().__init__()
        self.num_frames = num_frames
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.output_dim = output_dim or input_dim
        cfg = BackboneConfig(
            d_model=hidden_dim, n_layers=depth, layout=layout, ffn=ffn,  # type: ignore[arg-type]
            first_dense_layers=first_dense_layers, dense_hidden=dense_hidden, cond_dim=hidden_dim,
            mamba=Mamba3Config(d_state=d_state, headdim=headdim, backend=backend),  # type: ignore[arg-type]
            attention=AttentionConfig(n_heads=attn_heads, n_kv_heads=attn_kv_heads, window=max(num_frames, 1),
                                      rope_min_period=1.0, rope_max_period=64.0),
            moe=MoEConfig(n_routed=n_routed, top_k=top_k, n_shared=n_shared, expert_hidden=expert_hidden))
        self.in_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.cond_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.backbone = HybridBackbone(cfg)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, self.output_dim) if hidden_dim != self.output_dim else nn.Identity()
        self.aux_loss = torch.zeros(())

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] frame embeddings, c: [B, T, D] action embeddings -> [B, T, D]."""
        B, T, _ = x.shape
        t = torch.arange(T, device=x.device, dtype=torch.float32).expand(B, T)  # frame index
        h, _, aux = self.backbone(self.in_proj(x), self.cond_proj(c), t)
        self.aux_loss = aux                         # MoE balance term, read by the training loop
        return self.out_proj(self.norm(h))

    def update_routing_biases(self) -> None:
        for moe in self.backbone.moe_layers():
            moe.update_routing_bias()

    def active_params_per_token(self) -> int:
        """Parameters touched per token (routed experts counted top_k times)."""
        total = 0
        for name, p in self.named_parameters():
            if ".experts." in name:
                continue
            total += p.numel()
        for moe in self.backbone.moe_layers():
            total += moe.cfg.top_k * sum(p.numel() for p in moe.experts[0].parameters())
        return total
