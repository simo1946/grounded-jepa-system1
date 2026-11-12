"""Mamba-3 (SISO) sequence mixer with two interchangeable backends.

* ``official``: ``mamba_ssm.Mamba3`` (CUDA/Triton kernels) for GPU training.
* ``reference``: a pure-PyTorch port of the authors' reference math from
  ``tests/ops/triton/test_mamba3_siso.py`` in state-spaces/mamba (Apache-2.0):
  ``mamba3_siso_fwd_ref`` (parallel) and ``mamba3_siso_step_ref`` (recurrent).
  It runs anywhere (CPU, phones, Jetson) and supports streaming with O(1)
  work per step, which the official step kernel only guarantees on H100.

Both backends have the *same parameter names and shapes* (``in_proj``,
``dt_bias``, ``B_bias``, ``C_bias``, ``B_norm.weight``, ``C_norm.weight``,
``D``, ``out_proj``), so a state dict trained with the official kernels loads
into the reference module for edge deployment, and vice versa.

Recurrence (per head, from the reference step function)::

    angle_t = (angle_{t-1} + tanh(theta_t)*pi * dt_t) mod 2*pi
    q_t, k_t = RoPE(C_t + C_bias, angle_t), RoPE(B_t + B_bias, angle_t)
    alpha_t = exp(A_t*dt_t), lam_t = sigmoid(trap_t)
    S_t = alpha_t S_{t-1} + (1-lam_t) dt_t alpha_t k_{t-1} v_{t-1}^T + lam_t dt_t k_t v_t^T
    y_t = (S_t q_t + D v_t) * silu(z_t)

i.e. exponential-trapezoidal discretisation with a complex (rotary) state.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Mamba3Config
from ..state import State

TWO_PI = 2 * math.pi


class _RMSNorm(nn.Module):
    """Plain RMSNorm with the same parameter layout as mamba_ssm's gated
    RMSNorm when called without a gate (``weight`` only, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ct = torch.promote_types(x.dtype, torch.float32)
        xf = x.to(ct)
        out = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps) * self.weight.to(ct)
        return out.to(x.dtype)


def _heavy_tail(x: torch.Tensor) -> torch.Tensor:
    # f(x) = 1 + x (x >= 0); 1 / (1 - x) (x < 0)  -- as in mamba_ssm
    return x.clamp_min(0) + torch.reciprocal(1 - x.clamp_max(0))


def _rotary(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate interleaved pairs of the last dim; dims beyond the angle count
    are left unrotated (matches the reference padding with cos=1, sin=0)."""
    pairs = t.view(*t.shape[:-1], -1, 2)
    t0, t1 = pairs[..., 0], pairs[..., 1]
    n = cos.shape[-1]
    if n < t0.shape[-1]:
        pad = t0.shape[-1] - n
        cos = F.pad(cos, (0, pad), value=1.0)
        sin = F.pad(sin, (0, pad), value=0.0)
    r0 = t0 * cos - t1 * sin
    r1 = t0 * sin + t1 * cos
    return torch.stack([r0, r1], dim=-1).view_as(t)


def _segsum(x: torch.Tensor) -> torch.Tensor:
    """Stable segment sum: out[..., i, j] = sum_{k=j+1..i} x[..., k] (j <= i), -inf above."""
    T = x.size(-1)
    x = x.unsqueeze(-1).expand(*x.shape, T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    seg = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
    return seg.masked_fill(~mask, -torch.inf)


class Mamba3Reference(nn.Module):
    """Pure-PyTorch Mamba-3 SISO with the official parameterisation."""

    def __init__(self, d_model: int, cfg: Mamba3Config):
        super().__init__()
        self.d_model = d_model
        self.d_state = cfg.d_state
        self.headdim = cfg.headdim
        self.A_floor = cfg.A_floor
        self.chunk_size = cfg.chunk_size
        self.d_inner = cfg.expand * d_model
        if self.d_inner % cfg.headdim:
            raise ValueError("expand * d_model must divide by headdim")
        self.nheads = self.d_inner // cfg.headdim
        split = int(cfg.d_state * cfg.rope_fraction)
        split -= split % 2
        self.num_rope_angles = split // 2
        if self.num_rope_angles <= 0:
            raise ValueError("d_state too small for RoPE angles")

        # Order: [z, x, B, C, dd_dt, dd_A, trap, angle] (ngroups=1, SISO)
        d_in = 2 * self.d_inner + 2 * self.d_state + 3 * self.nheads + self.num_rope_angles
        self.in_proj = nn.Linear(d_model, d_in, bias=False)
        dt = torch.exp(torch.rand(self.nheads) * (math.log(cfg.dt_max) - math.log(cfg.dt_min)) + math.log(cfg.dt_min))
        dt = dt.clamp(min=cfg.dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        self.dt_bias._no_weight_decay = True  # type: ignore[attr-defined]
        self.B_bias = nn.Parameter(torch.ones(self.nheads, 1, self.d_state))
        self.C_bias = nn.Parameter(torch.ones(self.nheads, 1, self.d_state))
        self.B_norm = _RMSNorm(self.d_state)
        self.C_norm = _RMSNorm(self.d_state)
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.D._no_weight_decay = True  # type: ignore[attr-defined]
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    # ------------------------------------------------------------------ utils
    def _project(self, u: torch.Tensor):
        """u: [B, L, D] -> per-head tensors shaped like the reference inputs."""
        Bsz, L, _ = u.shape
        z, x, Bm, Cm, dd_dt, dd_A, trap, angles = torch.split(
            self.in_proj(u),
            [self.d_inner, self.d_inner, self.d_state, self.d_state,
             self.nheads, self.nheads, self.nheads, self.num_rope_angles],
            dim=-1,
        )
        ct = torch.promote_types(u.dtype, torch.float32)  # compute dtype (fp32, or fp64 in tests)
        z = z.view(Bsz, L, self.nheads, self.headdim)
        v = x.view(Bsz, L, self.nheads, self.headdim)
        k = self.B_norm(Bm).unsqueeze(2).expand(Bsz, L, self.nheads, self.d_state)
        q = self.C_norm(Cm).unsqueeze(2).expand(Bsz, L, self.nheads, self.d_state)
        A = torch.clamp(-_heavy_tail(dd_A.to(ct)), max=-self.A_floor)
        DT = F.softplus(dd_dt.to(ct) + self.dt_bias.to(ct))  # [B, L, H]
        ADT = A * DT
        ang = (torch.tanh(angles.to(ct)) * math.pi).unsqueeze(2).expand(Bsz, L, self.nheads, self.num_rope_angles)
        return (z.to(ct), v.to(ct), k.to(ct), q.to(ct), DT, ADT, torch.sigmoid(trap.to(ct)), ang)

    def init_state(self, batch: int, device=None, dtype=None) -> State:
        """Zero state. dtype defaults to promote(param dtype, float32)."""
        device = device or self.in_proj.weight.device
        dtype = dtype or torch.promote_types(self.in_proj.weight.dtype, torch.float32)
        kw = dict(device=device, dtype=dtype)
        return {
            "angle": torch.zeros(batch, self.nheads, self.num_rope_angles, **kw),
            "ssm": torch.zeros(batch, self.nheads, self.headdim, self.d_state, **kw),
            "k": torch.zeros(batch, self.nheads, self.d_state, **kw),
            "v": torch.zeros(batch, self.nheads, self.headdim, **kw),
        }

    # ------------------------------------------------------------- parallel
    def _chunk(self, z, v, k, q, DT, ADT, lam, ang, st: State):
        """Parallel (quadratic) form over one chunk with an initial state.
        Port of ``mamba3_siso_fwd_ref`` vectorised over the batch."""
        qb = q + self.C_bias.squeeze(1).to(q.dtype)
        kb = k + self.B_bias.squeeze(1).to(k.dtype)
        ang_cum = torch.cumsum(ang * DT.unsqueeze(-1), dim=1) + st["angle"].unsqueeze(1)
        ang_cum = ang_cum - TWO_PI * torch.floor(ang_cum / TWO_PI)
        cos, sin = torch.cos(ang_cum), torch.sin(ang_cum)

        DT_s = F.pad(DT[:, 1:], (0, 0, 0, 1))      # dt_{t+1}, 0 at the end
        lam_s = F.pad(lam[:, 1:], (0, 0, 0, 1))
        shifted_gamma = DT_s * (1 - lam_s)          # [B, L, H]
        scale = DT * lam + shifted_gamma
        qk_dot = (kb * qb).sum(-1) * shifted_gamma  # skip-term correction

        qr, kr = _rotary(qb, cos, sin), _rotary(kb, cos, sin)
        k_scaled = kr * scale.unsqueeze(-1)

        # initial-state contribution (previous k/v interact through the trapezoid)
        acc = st["ssm"] + (st["v"].unsqueeze(-1) * st["k"].unsqueeze(-2)) * (DT[:, 0] * (1 - lam[:, 0]))[..., None, None]

        adt_h = ADT.transpose(1, 2)                                  # [B, H, L]
        decay = torch.exp(_segsum(adt_h))                            # [B, H, L, L]
        QK = torch.einsum("bthd,bshd->bhts", qr, k_scaled)
        y = torch.einsum("bhts,bshd->bthd", torch.tril(QK) * decay, v)
        da_cs = torch.exp(torch.cumsum(adt_h, dim=-1))               # [B, H, L]
        y = y + torch.einsum("bhpn,bthn,bht->bthp", acc, qr, da_cs)
        y = y + self.D.to(v.dtype)[None, None, :, None] * v
        y = y - v * qk_dot.unsqueeze(-1)
        y = y * F.silu(z)

        tot = adt_h.sum(-1)                                          # [B, H]
        rev = torch.exp(tot.unsqueeze(-1) - torch.cumsum(adt_h, dim=-1))
        v_scaled = v * rev.transpose(1, 2).unsqueeze(-1)
        new_ssm = acc * torch.exp(tot)[..., None, None] + torch.einsum("bthn,bthp->bhpn", k_scaled, v_scaled)
        new_state = {"angle": ang_cum[:, -1], "ssm": new_ssm, "k": kr[:, -1], "v": v[:, -1]}
        return y, new_state

    def forward(self, u: torch.Tensor, state: Optional[State] = None) -> Tuple[torch.Tensor, State]:
        Bsz, L, _ = u.shape
        st = state if state is not None else self.init_state(Bsz, u.device)
        z, v, k, q, DT, ADT, lam, ang = self._project(u)
        ys = []
        for s in range(0, L, self.chunk_size):
            e = min(L, s + self.chunk_size)
            y, st = self._chunk(z[:, s:e], v[:, s:e], k[:, s:e], q[:, s:e], DT[:, s:e], ADT[:, s:e],
                                lam[:, s:e], ang[:, s:e], st)
            ys.append(y)
        y = torch.cat(ys, dim=1).reshape(Bsz, L, self.d_inner)
        return self.out_proj(y.to(u.dtype)), st

    # ------------------------------------------------------------- streaming
    def step(self, u: torch.Tensor, state: State) -> Tuple[torch.Tensor, State]:
        """One token: u [B, D] -> y [B, D]. Port of ``mamba3_siso_step_ref``."""
        z, v, k, q, DT, ADT, lam, ang = self._project(u.unsqueeze(1))
        z, v, k, q = z[:, 0], v[:, 0], k[:, 0], q[:, 0]
        DT, ADT, lam, ang = DT[:, 0], ADT[:, 0], lam[:, 0], ang[:, 0]
        q = q + self.C_bias.squeeze(1).to(q.dtype)
        k = k + self.B_bias.squeeze(1).to(k.dtype)
        angle = state["angle"] + ang * DT.unsqueeze(-1)
        angle = angle - TWO_PI * torch.floor(angle / TWO_PI)
        cos, sin = torch.cos(angle), torch.sin(angle)
        qr, kr = _rotary(q, cos, sin), _rotary(k, cos, sin)
        alpha = torch.exp(ADT)
        beta = (1 - lam) * DT * alpha
        gamma = lam * DT
        ssm = (alpha[..., None, None] * state["ssm"]
               + beta[..., None, None] * (state["v"].unsqueeze(-1) * state["k"].unsqueeze(-2))
               + gamma[..., None, None] * (v.unsqueeze(-1) * kr.unsqueeze(-2)))
        y = torch.einsum("bhpn,bhn->bhp", ssm, qr) + self.D.to(v.dtype)[None, :, None] * v
        y = (y * F.silu(z)).reshape(u.shape[0], self.d_inner)
        return self.out_proj(y.to(u.dtype)), {"angle": angle, "ssm": ssm, "k": kr, "v": v}


def official_mamba3_available() -> bool:
    try:
        from mamba_ssm import Mamba3  # noqa: F401
    except Exception:
        return False
    return torch.cuda.is_available()


class Mamba3Mixer(nn.Module):
    """Backend-selecting wrapper. Parameters live under ``core`` with the
    official names, so checkpoints are interchangeable between backends."""

    def __init__(self, d_model: int, cfg: Mamba3Config):
        super().__init__()
        self.cfg = cfg
        backend = cfg.backend
        if backend == "auto":
            backend = "official" if official_mamba3_available() else "reference"
        self.backend = backend
        if backend == "official":
            from mamba_ssm import Mamba3

            self.core = Mamba3(d_model=d_model, d_state=cfg.d_state, expand=cfg.expand, headdim=cfg.headdim,
                               rope_fraction=cfg.rope_fraction, dt_min=cfg.dt_min, dt_max=cfg.dt_max,
                               dt_init_floor=cfg.dt_init_floor, A_floor=cfg.A_floor, chunk_size=cfg.chunk_size,
                               is_mimo=False, is_outproj_norm=False)
        elif backend == "reference":
            self.core = Mamba3Reference(d_model, cfg)
        else:
            raise ValueError(f"unknown backend {backend!r}")
        self.d_model = d_model

    def to_reference_backend(self) -> "Mamba3Mixer":
        """Swap the official kernels for the reference implementation in place,
        keeping the trained weights (e.g. before exporting to an edge device)."""
        if self.backend != "reference":
            ref = Mamba3Reference(self.d_model, self.cfg).to(self.core.in_proj.weight.device)
            ref.load_state_dict(self.core.state_dict())
            self.core, self.backend = ref, "reference"
        return self

    def _ref_core(self) -> Mamba3Reference:
        if not isinstance(self.core, Mamba3Reference):
            raise RuntimeError(
                "Stateful/streaming calls need the reference backend (the official Mamba-3 step kernel is "
                "documented as H100-only). Call model.to_reference_backend() once before streaming.")
        return self.core

    def init_state(self, batch: int, device=None) -> State:
        if isinstance(self.core, Mamba3Reference):
            return self.core.init_state(batch, device)
        return Mamba3Reference.init_state(self.core, batch, device)  # same shapes/attrs

    def forward(self, x: torch.Tensor, state: Optional[State] = None) -> Tuple[torch.Tensor, Optional[State]]:
        if self.backend == "official" and state is None:
            return self.core(x), None
        return self._ref_core()(x, state)

    def step(self, x: torch.Tensor, state: State) -> Tuple[torch.Tensor, State]:
        return self._ref_core().step(x, state)
