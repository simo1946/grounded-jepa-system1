
import pytest
import torch

from hybrid_wm.config import AttentionConfig, EngramConfig, EpisodicConfig, MoEConfig
from hybrid_wm.layers.attention import GatedWindowAttention
from hybrid_wm.layers.memory import EpisodicMemory, NGramMemory
from hybrid_wm.layers.moe import MoEFFN

torch.manual_seed(0)


# ----------------------------------------------------------------- attention
def _attn(window=4):
    torch.manual_seed(0)
    return GatedWindowAttention(32, AttentionConfig(n_heads=4, n_kv_heads=2, window=window)).eval()


@torch.no_grad()
def test_attention_causal_and_windowed():
    a = _attn(window=4)
    x, t = torch.randn(1, 12, 32), torch.arange(12.0)[None]
    base, _ = a(x, t)
    x2 = x.clone()
    x2[:, 5] += 3
    out, _ = a(x2, t)
    torch.testing.assert_close(out[:, :5], base[:, :5])            # past unaffected
    assert not torch.allclose(out[:, 5:9], base[:, 5:9])           # inside window: affected
    torch.testing.assert_close(out[:, 9:], base[:, 9:])            # beyond window: unaffected


@torch.no_grad()
def test_attention_streaming_equals_parallel_with_irregular_time():
    a = _attn(window=5)
    x = torch.randn(2, 15, 32)
    t = torch.cumsum(torch.rand(2, 15) + 0.05, 1)
    full, _ = a(x, t)
    st, outs = a.init_state(2), []
    for i in range(15):
        y, st = a.step(x[:, i], t[:, i], st)
        outs.append(y)
    torch.testing.assert_close(torch.stack(outs, 1), full, rtol=1e-5, atol=1e-5)
    assert st["k"].shape[2] == 4                                    # bounded cache = window - 1


@torch.no_grad()
def test_attention_invariant_to_global_time_shift():
    a = _attn(window=8)
    x, t = torch.randn(1, 10, 32), torch.cumsum(torch.rand(1, 10), 1)
    y1, _ = a(x, t)
    y2, _ = a(x, t + 37.0)
    torch.testing.assert_close(y1, y2, rtol=1e-4, atol=1e-4)


# ------------------------------------------------------------------------ MoE
def _moe(**kw):
    torch.manual_seed(0)
    return MoEFFN(16, MoEConfig(n_routed=6, top_k=2, n_shared=1, expert_hidden=24, **kw))


def test_moe_is_truly_sparse_and_matches_dense_reference():
    m = _moe().eval()
    x = torch.randn(3, 7, 16)
    seen = {"n": 0}
    for e in m.experts:
        e.register_forward_hook(lambda mod, i, o: seen.__setitem__("n", seen["n"] + i[0].shape[0]))
    y, _ = m(x)
    assert seen["n"] == 3 * 7 * 2                                   # exactly top-k tokens per expert call
    xf = x.reshape(-1, 16)
    sel, g, _ = m.route(xf)
    ref = m.shared(xf).clone()
    for n in range(xf.shape[0]):
        for j in range(2):
            ref[n] += g[n, j] * m.experts[int(sel[n, j])](xf[n:n + 1])[0]
    torch.testing.assert_close(y.reshape(-1, 16), ref, rtol=1e-5, atol=1e-5)


def test_moe_bias_routes_but_does_not_weight():
    m = _moe()
    m.route_bias[0] = 100.0
    xf = torch.randn(10, 16)
    sel, g, s = m.route(xf)
    assert (sel == 0).any(-1).all()                                 # bias forces selection
    expected = s.gather(1, sel) / s.gather(1, sel).sum(-1, keepdim=True)
    torch.testing.assert_close(g, expected)                         # gate uses affinities only


def test_moe_bias_update_direction():
    m = _moe(bias_update_speed=0.01)
    m.load_accum.copy_(torch.tensor([100.0, 0, 0, 0, 0, 0]))
    m.update_routing_bias()
    assert m.route_bias[0] < 0 and (m.route_bias[1:] > 0).all()
    assert m.load_accum.sum() == 0


def test_moe_gradients_and_aux():
    m = _moe().train()
    y, aux = m(torch.randn(2, 9, 16))
    (y.square().mean() + aux).backward()
    assert aux > 0
    assert m.router.weight.grad.abs().sum() > 0
    assert m.load_accum.sum() == 2 * 9 * 2


# --------------------------------------------------------------------- Engram
def _engram():
    torch.manual_seed(0)
    return NGramMemory(16, EngramConfig(ngram_orders=(1, 2, 3), n_hash_heads=2, table_size_per_head=97,
                                        embed_dim_per_head=4, conv_kernel=3), layer_id=0).eval()


@torch.no_grad()
def test_engram_causal_and_streaming():
    e = _engram()
    h, keys = torch.randn(2, 10, 16), torch.randint(0, 1000, (2, 10))
    base, _ = e(h, keys)
    k2 = keys.clone()
    k2[:, 6] += 1
    out, _ = e(h, k2)
    torch.testing.assert_close(out[:, :6], base[:, :6])
    assert not torch.allclose(out[:, 6:], base[:, 6:])
    st, steps = e.init_state(2), []
    for t in range(10):
        y, st = e(h[:, t:t + 1], keys[:, t:t + 1], st)
        steps.append(y)
    torch.testing.assert_close(torch.cat(steps, 1), base, rtol=1e-5, atol=1e-6)


def test_engram_rejects_bad_keys_loudly():
    e = _engram()
    h = torch.randn(1, 3, 16)
    with pytest.raises(TypeError):
        e(h, torch.zeros(1, 3))                                     # float keys: no silent 0.0 fallback
    with pytest.raises(TypeError):
        e(h, torch.zeros(1, 3, dtype=torch.int16))
    with pytest.raises(ValueError):
        e(h, torch.full((1, 3), -1))


def test_engram_hashing_deterministic_and_tables_distinct():
    a, b = _engram(), _engram()
    keys = torch.randint(0, 10_000, (2, 8))
    hist = torch.zeros(2, 2, dtype=torch.long)
    assert torch.equal(a.hash_ids(keys + 1, hist), b.hash_ids(keys + 1, hist))
    assert len(set(a.primes.tolist())) == a.primes.numel()


# ------------------------------------------------------------------- episodic
def _episodic(**kw):
    torch.manual_seed(0)
    return EpisodicMemory(16, EpisodicConfig(n_slots=3, key_dim=8, value_dim=8, write_every=4, **kw))


@torch.no_grad()
def test_episodic_streaming_equals_parallel():
    m = _episodic()
    h = torch.randn(2, 18, 16)
    full, st_full = m(h)
    st, outs = m.init_state(2), []
    for t in range(18):
        y, st = m(h[:, t:t + 1], st)
        outs.append(y)
    torch.testing.assert_close(torch.cat(outs, 1), full, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(st["k"], st_full["k"])


@torch.no_grad()
def test_episodic_reads_only_completed_chunks():
    m = _episodic()
    h = torch.randn(1, 16, 16)
    base, _ = m(h)
    h2 = h.clone()
    h2[:, 5] += 4.0                                                 # chunk 1 = positions 4..7
    out, _ = m(h2)
    others = [i for i in range(8) if i != 5]
    torch.testing.assert_close(out[:, others], base[:, others])     # same/earlier chunks unaffected
    assert not torch.allclose(out[:, 8:], base[:, 8:])              # later chunks see the new write


def test_episodic_empty_is_finite_and_write_gate_learns():
    m = _episodic()
    y, _ = m(torch.randn(1, 2, 16))                                 # nothing written yet
    assert torch.isfinite(y).all()
    y, _ = m(torch.randn(1, 12, 16))
    y[:, 4:].square().mean().backward()
    assert m.write_gate[0].weight.grad.abs().sum() > 0              # writes are differentiable


@torch.no_grad()
def test_episodic_overwrites_weakest_slot():
    m = _episodic(strength_decay=1.0)
    st = m.init_state(1)
    st["strength"] = torch.tensor([[0.9, 0.1, 0.5]])
    st["k"] = torch.ones(1, 3, 8)
    new = m._write(torch.randn(1, 4, 16), st)
    changed = (new["k"] != st["k"]).any(-1)[0]
    assert changed.tolist() == [False, True, False]


# ----------------------------------------------------------------------- time
