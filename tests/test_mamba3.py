import math

import pytest
import torch
import torch.nn.functional as F

from hybrid_wm.config import Mamba3Config
from hybrid_wm.layers.mamba3 import Mamba3Mixer, Mamba3Reference

from .third_party.mamba3_siso_ref import mamba3_siso_fwd_ref, mamba3_siso_step_ref

torch.set_default_dtype(torch.float32)
CFG = Mamba3Config(d_state=16, expand=2, headdim=8, chunk_size=7, backend="reference")


def _model(d=16, seed=0, cfg=CFG):
    torch.manual_seed(seed)
    m = Mamba3Reference(d, cfg)
    with torch.no_grad():  # randomise the non-trivially initialised params too
        for p in (m.B_bias, m.C_bias, m.D, m.B_norm.weight, m.C_norm.weight):
            p.add_(0.3 * torch.randn_like(p))
    return m.double()


def _oracle_inputs(m, u):
    """Raw (pre-activation) inputs in the layout the authors' reference expects."""
    B, L, _ = u.shape
    z, x, Bm, Cm, dd_dt, dd_A, trap, ang = torch.split(
        m.in_proj(u), [m.d_inner, m.d_inner, m.d_state, m.d_state, m.nheads, m.nheads, m.nheads,
                       m.num_rope_angles], dim=-1)
    A = torch.clamp(-(dd_A.clamp_min(0) + 1 / (1 - dd_A.clamp_max(0))), max=-m.A_floor)
    DT = F.softplus(dd_dt + m.dt_bias)
    return dict(
        Q=m.C_norm(Cm).unsqueeze(2).expand(B, L, m.nheads, m.d_state),
        K=m.B_norm(Bm).unsqueeze(2).expand(B, L, m.nheads, m.d_state),
        V=x.view(B, L, m.nheads, m.headdim),
        ADT=(A * DT).transpose(1, 2), DT=DT.transpose(1, 2), Trap=trap.transpose(1, 2),
        Q_bias=m.C_bias.squeeze(1), K_bias=m.B_bias.squeeze(1),
        Angles=ang.unsqueeze(2).expand(B, L, m.nheads, m.num_rope_angles),
        D=m.D, Z=z.view(B, L, m.nheads, m.headdim))


@torch.no_grad()
def test_matches_authors_parallel_reference():
    m = _model()
    u = torch.randn(3, 20, 16, dtype=torch.float64)
    ours, _ = m(u)
    y, _ = mamba3_siso_fwd_ref(**_oracle_inputs(m, u), dtype=torch.float64)
    oracle = m.out_proj(y.reshape(3, 20, -1))
    # The authors' parallel reference casts the rotation angles with `.float()` even in
    # float64, so it is only float32-accurate. Exactness is established transitively:
    # step == authors' step (1e-9) and our parallel == our step (1e-9), tests below.
    torch.testing.assert_close(ours, oracle, rtol=1e-5, atol=1e-6)


@torch.no_grad()
def test_matches_authors_step_reference():
    m = _model()
    u = torch.randn(2, 12, 16, dtype=torch.float64)
    st = m.init_state(2, dtype=torch.float64)
    st = {k: v.double() for k, v in st.items()}
    ours = []
    for t in range(12):
        y, st = m.step(u[:, t], st)
        ours.append(y)
    ours = torch.stack(ours, 1)
    y, _ = mamba3_siso_step_ref(**_oracle_inputs(m, u))
    torch.testing.assert_close(ours, m.out_proj(y.reshape(2, 12, -1)), rtol=1e-9, atol=1e-9)


@torch.no_grad()
def test_parallel_equals_streaming_and_chunking():
    m = _model()
    u = torch.randn(2, 30, 16, dtype=torch.float64)
    full, st_full = m(u)                       # chunk_size=7 -> 5 chunks with carried state
    a, st = m(u[:, :11])
    b, st2 = m(u[:, 11:], st)                  # resume from state
    torch.testing.assert_close(torch.cat([a, b], 1), full, rtol=1e-9, atol=1e-9)
    st = {k: v.double() for k, v in m.init_state(2).items()}
    steps = []
    for t in range(30):
        y, st = m.step(u[:, t], st)
        steps.append(y)
    torch.testing.assert_close(torch.stack(steps, 1), full, rtol=1e-9, atol=1e-9)
    for k in st_full:
        torch.testing.assert_close(st[k], st_full[k], rtol=1e-9, atol=1e-9)


@torch.no_grad()
def test_causal():
    m = _model()
    u = torch.randn(1, 16, 16, dtype=torch.float64)
    base, _ = m(u)
    u2 = u.clone()
    u2[:, 10] += 3.0
    out, _ = m(u2)
    # Exact in math; the parallel trapezoid form adds and subtracts a dt_{t+1} term
    # (as in the authors' reference), so allow float64 rounding only.
    torch.testing.assert_close(base[:, :10], out[:, :10], rtol=0, atol=1e-12)
    assert not torch.allclose(base[:, 10:], out[:, 10:])


def test_gradients_flow_to_all_parameters():
    m = Mamba3Reference(16, CFG)
    y, _ = m(torch.randn(2, 9, 16))
    y.square().mean().backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, name


def test_state_dict_compatible_with_official_mamba3():
    """Checkpoints must move between the official GPU module and the reference."""
    try:
        from mamba_ssm.modules.mamba3 import Mamba3  # needs triton importable (kernels not run)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"official Mamba3 not importable here: {e}")
    off = Mamba3(d_model=16, d_state=16, expand=2, headdim=8, is_mimo=False, is_outproj_norm=False)
    ref = Mamba3Reference(16, CFG)
    so, sr = off.state_dict(), ref.state_dict()
    assert {k: tuple(v.shape) for k, v in so.items()} == {k: tuple(v.shape) for k, v in sr.items()}
    ref.load_state_dict(so)  # strict


def test_mixer_reference_backend_roundtrip():
    mix = Mamba3Mixer(16, CFG)
    assert mix.backend == "reference"
    x = torch.randn(2, 5, 16)
    y, st = mix(x)
    assert y.shape == x.shape and st is not None
