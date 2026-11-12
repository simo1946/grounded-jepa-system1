"""DeepSeek-V3-style mixture-of-experts feed-forward layer.

The MoE sits on the *feed-forward* (channel-mixing) sublayer, never on the
recurrent sequence mixer: FFN experts treat tokens independently, so tokens
can be gathered per expert and only the selected experts run (true sparsity).
MoE-Mamba also places its experts as FFNs between Mamba layers.

Routing (DeepSeek-V3 technical report, eqs. 12-20):
* affinity s_i = sigmoid(u . e_i)
* top-k selection uses s_i + b_i, where b_i is a non-learned bias;
  the gate value uses s_i only, normalised over the selected experts;
* after each optimiser step, b_i -= gamma for overloaded experts and
  b_i += gamma for underloaded ones (``update_routing_bias``);
* a very small complementary sequence-wise balance loss.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ..config import MoEConfig


class SwiGLU(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.w_gate = nn.Linear(d, hidden, bias=False)
        self.w_up = nn.Linear(d, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MoEFFN(nn.Module):
    def __init__(self, d_model: int, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(d_model, cfg.n_routed, bias=False)
        self.experts = nn.ModuleList(SwiGLU(d_model, cfg.expert_hidden) for _ in range(cfg.n_routed))
        self.shared = SwiGLU(d_model, cfg.expert_hidden * cfg.n_shared) if cfg.n_shared > 0 else None
        self.register_buffer("route_bias", torch.zeros(cfg.n_routed))
        self.register_buffer("load_accum", torch.zeros(cfg.n_routed))
        nn.init.normal_(self.router.weight, std=0.02)

    def route(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: [N, D] -> (selected expert ids [N, k], gate weights [N, k], affinities [N, E])."""
        s = torch.sigmoid(self.router(x).float())
        sel = torch.topk(s + self.route_bias, self.cfg.top_k, dim=-1).indices
        g = s.gather(1, sel)
        g = g / g.sum(-1, keepdim=True).clamp_min(1e-9) * self.cfg.route_scale
        return sel, g, s

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: [B, L, D] -> (y, sequence-wise balance loss)."""
        B, L, D = x.shape
        xf = x.reshape(B * L, D)
        sel, g, s = self.route(xf)
        k, E = self.cfg.top_k, self.cfg.n_routed

        flat_e = sel.reshape(-1)
        order = torch.argsort(flat_e, stable=True)
        tok = torch.arange(B * L, device=x.device).repeat_interleave(k)[order]
        gate = g.reshape(-1)[order]
        counts = torch.bincount(flat_e, minlength=E)

        out = torch.zeros_like(xf)
        start = 0
        for e, c in enumerate(counts.tolist()):  # only selected tokens reach each expert
            if c:
                idx = tok[start:start + c]
                out.index_add_(0, idx, self.experts[e](xf[idx]) * gate[start:start + c, None].to(xf.dtype))
                start += c
        if self.shared is not None:
            out = out + self.shared(xf)

        if self.training:
            with torch.no_grad():
                self.load_accum += counts.to(self.load_accum.dtype)

        aux = x.new_zeros((), dtype=torch.float32)
        if self.training and self.cfg.seq_aux_weight > 0:
            # f_i = E/(k*T) * #tokens of the sequence routed to i ; P_i = mean_t s_i/sum_j s_j
            onehot = torch.zeros(B * L, E, device=x.device).scatter_(1, sel, 1.0).view(B, L, E)
            f = onehot.sum(1) * (E / (k * L))
            P = (s / s.sum(-1, keepdim=True)).view(B, L, E).mean(1)
            aux = self.cfg.seq_aux_weight * (f * P).sum(-1).mean()
        return out.view(B, L, D), aux

    @torch.no_grad()
    def update_routing_bias(self) -> None:
        """Call once per optimiser step. Sync load across ranks if distributed."""
        load = self.load_accum.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(load)
        if load.sum() > 0:
            self.route_bias += self.cfg.bias_update_speed * torch.sign(load.mean() - load)
        self.load_accum.zero_()
