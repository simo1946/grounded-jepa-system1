"""Dual memory: static conditional lookup (Engram-style) + writable episodic store.

NGramMemory (static knowledge)
    Follows DeepSeek's Engram demo (deepseek-ai/Engram, Apache-2.0):
    multi-head hashed n-grams (XOR of key*odd multiplier, distinct prime table
    size per head), a context-aware gate
        g = sigmoid(signed_sqrt(<RMSNorm(W_k e), RMSNorm(h)> / sqrt(d)))
    and a causal depthwise short convolution (dilation = max n-gram order):
        out = g*W_v e + SiLU(Conv(RMSNorm(g*W_v e))).
    Keys are discrete IDs from the caller. Rows are learned during training and
    read-only at inference; deterministic addressing lets very large tables be
    prefetched from host memory. Unlike the LLM demo, unigrams are allowed,
    because here there is no token-embedding table carrying them.

EpisodicMemory (writable scratchpad)
    Slots written *at inference time*. Every ``write_every`` steps the last
    chunk is summarised by learned salience pooling and written with a learned
    strength into the slot with the lowest (decayed) strength. Reads use the
    memory as it stood at the start of the chunk -> strictly causal, and
    parallel training equals step-by-step streaming exactly.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import EngramConfig, EpisodicConfig
from ..state import State

_MAX_KEY = 2 ** 31 - 2


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    r = int(math.isqrt(n))
    return all(n % f for f in range(3, r + 1, 2))


def _distinct_primes(start: int, count: int, used: set) -> List[int]:
    out, n = [], start
    while len(out) < count:
        n += 1
        if n not in used and _is_prime(n):
            out.append(n)
            used.add(n)
    return out


def _signed_sqrt(x: torch.Tensor) -> torch.Tensor:
    return x.abs().clamp_min(1e-6).sqrt() * x.sign()


class NGramMemory(nn.Module):
    def __init__(self, d_model: int, cfg: EngramConfig, layer_id: int = 0):
        super().__init__()
        self.cfg, self.d = cfg, d_model
        self.max_n = max(cfg.ngram_orders)
        g = torch.Generator().manual_seed(cfg.seed + 10007 * layer_id)
        # odd multipliers < 2^31 so key*mult stays < 2^62 (no int64 overflow)
        self.register_buffer("mult", torch.randint(0, 2 ** 30, (self.max_n,), generator=g) * 2 + 1,
                             persistent=False)
        used: set = set()
        primes = [p for _ in cfg.ngram_orders
                  for p in _distinct_primes(cfg.table_size_per_head - 1, cfg.n_hash_heads, used)]
        self.register_buffer("primes", torch.tensor(primes, dtype=torch.long), persistent=False)
        self.register_buffer("offsets", torch.tensor([0] + primes[:-1], dtype=torch.long).cumsum(0),
                             persistent=False)
        self.table = nn.Embedding(sum(primes), cfg.embed_dim_per_head)
        nn.init.normal_(self.table.weight, std=0.02)
        f = len(primes) * cfg.embed_dim_per_head
        self.key_proj = nn.Linear(f, d_model)
        self.value_proj = nn.Linear(f, d_model)
        self.norm_k = nn.RMSNorm(d_model)
        self.norm_q = nn.RMSNorm(d_model)
        self.conv_norm = nn.RMSNorm(d_model)
        self.dil = self.max_n
        self.conv = nn.Conv1d(d_model, d_model, cfg.conv_kernel, groups=d_model, bias=False, dilation=self.dil)
        self.conv_ctx = (cfg.conv_kernel - 1) * self.dil

    @staticmethod
    def validate_keys(keys: torch.Tensor) -> None:
        if not isinstance(keys, torch.Tensor) or keys.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"memory keys must be an int32/int64 tensor, got "
                            f"{getattr(keys, 'dtype', type(keys))} (no silent fallback)")
        if keys.numel() and (keys.min() < 0 or keys.max() > _MAX_KEY):
            raise ValueError(f"memory keys must be in [0, {_MAX_KEY}]")

    def init_state(self, batch: int, device=None, dtype=None) -> State:
        device = device or self.table.weight.device
        dtype = dtype or self.table.weight.dtype
        return {"keys": torch.zeros(batch, self.max_n - 1, dtype=torch.long, device=device),  # 0 = pad
                "conv": torch.zeros(batch, self.d, self.conv_ctx, device=device, dtype=dtype)}

    def hash_ids(self, keys: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        """keys [B, L] (already +1 shifted), hist [B, max_n-1] -> table rows [B, L, n_tables]."""
        L = keys.shape[1]
        full = torch.cat([hist, keys], dim=1)                              # [B, max_n-1+L]
        cols = []
        for n in self.cfg.ngram_orders:
            mix = torch.zeros_like(keys)
            for j in range(n):                                             # key at t-j
                s = self.max_n - 1 - j
                mix = mix ^ (full[:, s:s + L] * self.mult[j])
            cols.append(mix)
        mixes = torch.stack(cols, dim=-1).repeat_interleave(self.cfg.n_hash_heads, dim=-1)
        return mixes % self.primes + self.offsets

    def forward(self, h: torch.Tensor, keys: torch.Tensor, state: Optional[State] = None
                ) -> Tuple[torch.Tensor, State]:
        self.validate_keys(keys)
        B, L, _ = h.shape
        if keys.shape != (B, L):
            raise ValueError(f"keys must have shape {(B, L)}, got {tuple(keys.shape)}")
        st = state if state is not None else self.init_state(B, h.device, h.dtype)
        k1 = keys.long() + 1                                               # reserve 0 for padding
        e = self.table(self.hash_ids(k1, st["keys"])).flatten(-2)          # [B, L, F]
        gate = (self.norm_k(self.key_proj(e)) * self.norm_q(h)).sum(-1, keepdim=True) / math.sqrt(self.d)
        value = torch.sigmoid(_signed_sqrt(gate)) * self.value_proj(e)
        x = torch.cat([st["conv"], self.conv_norm(value).transpose(1, 2)], dim=2)
        conv = F.silu(self.conv(x))[..., -L:].transpose(1, 2)              # causal: only past context
        new_hist = torch.cat([st["keys"], k1], dim=1)[:, -(self.max_n - 1):] if self.max_n > 1 else st["keys"]
        new_state = {"keys": new_hist, "conv": x[..., -self.conv_ctx:] if self.conv_ctx else st["conv"]}
        return value + conv, new_state


class EpisodicMemory(nn.Module):
    def __init__(self, d_model: int, cfg: EpisodicConfig):
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(d_model, cfg.key_dim, bias=False)
        self.k_proj = nn.Linear(d_model, cfg.key_dim, bias=False)
        self.v_proj = nn.Linear(d_model, cfg.value_dim, bias=False)
        self.out_proj = nn.Linear(cfg.value_dim, d_model, bias=False)
        self.salience = nn.Linear(d_model, 1)
        self.write_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 1))
        self.null_k = nn.Parameter(torch.zeros(cfg.key_dim))
        self.null_v = nn.Parameter(torch.zeros(cfg.value_dim))
        self.scale = cfg.key_dim ** -0.5

    def init_state(self, batch: int, device=None, dtype=None) -> State:
        c = self.cfg
        device = device or self.q_proj.weight.device
        dtype = dtype or self.q_proj.weight.dtype
        z = lambda *s: torch.zeros(*s, device=device, dtype=dtype)  # noqa: E731
        return {"k": z(batch, c.n_slots, c.key_dim), "v": z(batch, c.n_slots, c.value_dim),
                "strength": z(batch, c.n_slots), "age": z(batch, c.n_slots),
                "buf": z(batch, 0, self.q_proj.in_features), "pos": 0}

    def _read(self, h: torch.Tensor, st: State) -> torch.Tensor:
        B = h.shape[0]
        keys = torch.cat([self.null_k.expand(B, 1, -1), st["k"]], dim=1)
        vals = torch.cat([self.null_v.expand(B, 1, -1), st["v"]], dim=1)
        eff = st["strength"] * self.cfg.strength_decay ** st["age"]                 # decayed strength
        bias = torch.cat([torch.zeros_like(eff[:, :1]), torch.log(eff.clamp_min(1e-30))], dim=1)
        bias = bias.masked_fill(torch.cat([torch.zeros_like(eff[:, :1], dtype=torch.bool), eff <= 0], 1),
                                float("-inf"))                                      # empty slots unreadable
        scores = torch.einsum("bld,bsd->bls", self.q_proj(h), keys) * self.scale + bias.unsqueeze(1)
        return self.out_proj(torch.einsum("bls,bsd->bld", scores.softmax(-1), vals))

    def _write(self, buf: torch.Tensor, st: State) -> State:
        w = self.salience(buf).squeeze(-1).softmax(-1)                               # [B, C]
        summary = torch.einsum("bc,bcd->bd", w, buf)
        g = torch.sigmoid(self.write_gate(summary)).squeeze(-1)                     # write strength
        eff = st["strength"] * self.cfg.strength_decay ** st["age"]
        slot = torch.argmin(eff - 1e-6 * st["age"], dim=1)                          # weakest (oldest on ties)
        oh = F.one_hot(slot, self.cfg.n_slots).to(buf.dtype)                        # [B, S]
        keep = 1 - oh
        return {"k": st["k"] * keep.unsqueeze(-1) + oh.unsqueeze(-1) * self.k_proj(summary).unsqueeze(1),
                "v": st["v"] * keep.unsqueeze(-1) + oh.unsqueeze(-1) * self.v_proj(summary).unsqueeze(1),
                "strength": st["strength"] * keep + oh * g.unsqueeze(-1),
                "age": (st["age"] + 1) * keep}

    def forward(self, h: torch.Tensor, state: Optional[State] = None) -> Tuple[torch.Tensor, State]:
        B, L, _ = h.shape
        st = dict(state) if state is not None else self.init_state(B, h.device, h.dtype)
        C, outs, i = self.cfg.write_every, [], 0
        while i < L:
            n = min(L - i, C - st["pos"] % C)                       # tokens until the next write
            seg = h[:, i:i + n]
            outs.append(self._read(seg, st))                         # memory as of chunk start
            st["buf"] = torch.cat([st["buf"], seg], dim=1)
            st["pos"] += n
            if st["pos"] % C == 0:
                st.update(self._write(st["buf"], st))
                st["buf"] = st["buf"][:, :0]
            i += n
        return torch.cat(outs, dim=1), st
