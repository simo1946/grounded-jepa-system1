"""Typed, validated configuration.

Every config validates itself in ``__post_init__`` so that an inconsistent
setting fails loudly at construction time instead of producing a model that
silently does something else.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Tuple

MixerBackend = Literal["auto", "official", "reference"]


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


@dataclass(frozen=True)
class Mamba3Config:
    """Mamba-3 (SISO) mixer. Field names mirror ``mamba_ssm.Mamba3``.

    backend:
      * ``official``  -> ``mamba_ssm.Mamba3`` CUDA/Triton kernels (training on GPU).
      * ``reference`` -> pure-PyTorch port of the authors' reference math
        (CPU, edge devices, streaming, tests). Same parameters and state dict.
      * ``auto``      -> official when importable and on CUDA, else reference.
    """

    d_state: int = 64
    expand: int = 2
    headdim: int = 32
    rope_fraction: float = 0.5
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    A_floor: float = 1e-4
    chunk_size: int = 64
    backend: MixerBackend = "auto"

    def __post_init__(self) -> None:
        _require(self.rope_fraction in (0.5, 1.0), "rope_fraction must be 0.5 or 1.0 (as in mamba_ssm)")
        _require(self.d_state >= 4, "d_state must be >= 4")
        _require(self.headdim > 0 and self.expand > 0, "headdim and expand must be positive")
        _require(0 < self.dt_min < self.dt_max, "need 0 < dt_min < dt_max")
        _require(self.chunk_size > 0, "chunk_size must be positive")


@dataclass(frozen=True)
class AttentionConfig:
    """Causal sliding-window attention with continuous-time RoPE and an
    element-wise, head-specific sigmoid gate on the SDPA output."""

    n_heads: int = 4
    n_kv_heads: int = 4
    window: int = 64
    rope_min_period: float = 0.05  # seconds (in units of `time_scale`)
    rope_max_period: float = 600.0
    qk_norm: bool = True
    output_gate: bool = True

    def __post_init__(self) -> None:
        _require(self.n_heads > 0 and self.n_kv_heads > 0, "head counts must be positive")
        _require(self.n_heads % self.n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads")
        _require(self.window >= 1, "window must be >= 1")
        _require(0 < self.rope_min_period < self.rope_max_period, "need 0 < rope_min_period < rope_max_period")


@dataclass(frozen=True)
class MoEConfig:
    """DeepSeek-V3-style MoE feed-forward: sigmoid affinities, bias-based
    (auxiliary-loss-free) balancing used only for expert selection,
    normalised gates over the selected experts, shared experts, and a small
    complementary sequence-wise balance loss."""

    n_routed: int = 8
    n_shared: int = 1
    top_k: int = 2
    expert_hidden: int = 128
    bias_update_speed: float = 1e-3
    seq_aux_weight: float = 1e-4
    route_scale: float = 1.0

    def __post_init__(self) -> None:
        _require(self.n_routed >= 1, "n_routed must be >= 1")
        _require(1 <= self.top_k <= self.n_routed, "need 1 <= top_k <= n_routed")
        _require(self.n_shared >= 0, "n_shared must be >= 0")
        _require(self.expert_hidden > 0, "expert_hidden must be positive")
        _require(self.bias_update_speed >= 0 and self.seq_aux_weight >= 0, "weights must be >= 0")


@dataclass(frozen=True)
class EngramConfig:
    """Static conditional memory (Engram-style hashed n-gram lookup).

    Keys are discrete IDs supplied by the caller (for example route-segment or
    command IDs). N-grams only
    look at the current and past keys, so the module is causal.
    """

    layers: Tuple[int, ...] = (1,)
    ngram_orders: Tuple[int, ...] = (1, 2, 3)
    n_hash_heads: int = 4
    table_size_per_head: int = 50_021  # rounded up to a distinct prime per head
    embed_dim_per_head: int = 32
    conv_kernel: int = 4
    pad_key: int = 0
    seed: int = 0

    def __post_init__(self) -> None:
        _require(len(self.ngram_orders) > 0 and min(self.ngram_orders) >= 1, "ngram orders must be >= 1")
        _require(len(set(self.ngram_orders)) == len(self.ngram_orders), "ngram orders must be unique")
        _require(self.n_hash_heads >= 1 and self.table_size_per_head >= 2, "invalid hash table size")
        _require(self.embed_dim_per_head > 0 and self.conv_kernel >= 1, "invalid embedding/conv size")


@dataclass(frozen=True)
class EpisodicConfig:
    """Writable episodic memory (the 'scratchpad').

    Every ``write_every`` steps the model summarises the last chunk with a
    learned salience-weighted pool and writes it with a learned strength into
    the weakest slot. Tokens read the memory as it stood at the start of their
    chunk, so reads are strictly causal and identical in streaming mode.
    """

    layers: Tuple[int, ...] = (2,)
    n_slots: int = 32
    key_dim: int = 64
    value_dim: int = 64
    write_every: int = 8
    strength_decay: float = 0.98  # per write cycle; old memories fade unless strong

    def __post_init__(self) -> None:
        _require(self.n_slots >= 1, "n_slots must be >= 1")
        _require(self.key_dim > 0 and self.value_dim > 0, "memory dims must be positive")
        _require(self.write_every >= 1, "write_every must be >= 1")
        _require(0 < self.strength_decay <= 1, "strength_decay must be in (0, 1]")


@dataclass(frozen=True)
class HierarchyConfig:
    """Two-timescale (H-JEPA-style) stream. Layers ``< split`` run at the base
    rate; their output at the end of each ``stride``-step chunk feeds a slower
    stream whose result conditions layers ``>= split`` during the *next*
    chunk (strictly causal)."""

    split: int = 2
    stride: int = 4
    n_layers: int = 2

    def __post_init__(self) -> None:
        _require(self.split >= 1, "split must be >= 1")
        _require(self.stride >= 2, "stride must be >= 2")
        _require(self.n_layers >= 1, "n_layers must be >= 1")


@dataclass(frozen=True)
class BackboneConfig:
    """Hybrid predictor backbone.

    ``layout`` is repeated to ``n_layers``: ``M`` = Mamba-3 layer,
    ``A`` = sliding-window attention layer (Qwen3.5 uses 3:1, i.e. ``MMMA``).
    """

    d_model: int = 128
    n_layers: int = 4
    layout: str = "MMMA"
    ffn: Literal["moe", "dense"] = "moe"
    first_dense_layers: int = 1
    dense_hidden: int = 256
    cond_dim: int = 128
    mamba: Mamba3Config = field(default_factory=Mamba3Config)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    engram: EngramConfig | None = None
    episodic: EpisodicConfig | None = None
    hierarchy: HierarchyConfig | None = None

    def __post_init__(self) -> None:
        _require(self.d_model > 0 and self.n_layers > 0, "d_model and n_layers must be positive")
        _require(len(self.layout) > 0 and set(self.layout) <= {"M", "A"}, "layout may only contain 'M' and 'A'")
        _require(0 <= self.first_dense_layers <= self.n_layers, "first_dense_layers out of range")
        _require(self.d_model % self.attention.n_heads == 0, "d_model must divide by attention.n_heads")
        _require((self.d_model // self.attention.n_heads) % 2 == 0, "attention head dim must be even (RoPE)")
        _require((self.mamba.expand * self.d_model) % self.mamba.headdim == 0,
                 "mamba.expand * d_model must divide by mamba.headdim")
        for name, sub in (("engram", self.engram), ("episodic", self.episodic)):
            if sub is not None:
                _require(all(0 <= i < self.n_layers for i in sub.layers), f"{name}.layers out of range")
        if self.hierarchy is not None:
            _require(self.hierarchy.split < self.n_layers, "hierarchy.split must be < n_layers")

    def layer_kind(self, i: int) -> str:
        return self.layout[i % len(self.layout)]


@dataclass(frozen=True)
class WorldModelConfig:
    """JEPA world model (LeWM-style recipe: next-embedding prediction with
    teacher forcing + SIGReg on the encoder embeddings, no EMA by default)."""

    latent_dim: int = 128
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    sigreg_weight: float = 0.1
    sigreg_slices: int = 1024
    sigreg_knots: int = 17
    rollout_weight: float = 0.0
    rollout_steps: int = 0
    stop_grad_target: bool = False
    projector: Literal["bn", "ln"] = "bn"
    time_scale: float = 1.0  # seconds per model time unit

    def __post_init__(self) -> None:
        _require(self.latent_dim > 0, "latent_dim must be positive")
        _require(self.sigreg_weight >= 0 and self.rollout_weight >= 0, "loss weights must be >= 0")
        _require(self.sigreg_slices >= 1 and self.sigreg_knots >= 3, "invalid SIGReg settings")
        _require(self.rollout_steps >= 0, "rollout_steps must be >= 0")
        _require(self.rollout_weight == 0 or self.rollout_steps >= 2, "rollout loss needs rollout_steps >= 2")
        _require(self.time_scale > 0, "time_scale must be positive")
