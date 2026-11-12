"""System 1: a goal-conditioned latent policy that proposes action chunks.

pi(z_t, z_goal) -> Gaussian over the next H action blocks (normalised action
space, the same space the planner samples in). Trained offline by hindsight
relabelling on reward-free trajectories: the goal is the embedding of a frame
k blocks ahead (k uniform in 1..H) and the target is the action chunk actually
executed; blocks after the goal frame are masked, since they do not
contribute to reaching it.

Optional side information (``extra``, e.g. encoded robot or object sensors,
see ``extra_dim``) is concatenated to the latent inputs; with
``extra_dim = 0`` the module is exactly the latent-only policy.

The policy is only a *proposer*. Whether its proposal is trusted is decided
by the world model (verify-then-act, see ``system12.py``).
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChunkPolicy(nn.Module):
    def __init__(self, latent_dim: int, horizon: int, action_dim: int, hidden: int = 512,
                 min_log_std: float = -5.0, max_log_std: float = 1.0, extra_dim: int = 0):
        super().__init__()
        self.horizon, self.action_dim, self.extra_dim = horizon, action_dim, extra_dim
        self.min_log_std, self.max_log_std = min_log_std, max_log_std
        self.net = nn.Sequential(
            nn.Linear(3 * latent_dim + extra_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, 2 * horizon * action_dim))

    def _inputs(self, z_t: torch.Tensor, z_goal: torch.Tensor, extra: torch.Tensor | None) -> torch.Tensor:
        parts = [z_t, z_goal, z_goal - z_t]
        if self.extra_dim:
            if extra is None or extra.shape[-1] != self.extra_dim:
                raise ValueError(f"policy expects extra inputs of size {self.extra_dim}")
            parts.append(extra.to(device=z_t.device, dtype=z_t.dtype))
        elif extra is not None:
            raise ValueError("policy was built without extra inputs")
        return torch.cat(parts, dim=-1)

    def forward(self, z_t: torch.Tensor, z_goal: torch.Tensor,
                extra: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """z_t, z_goal [B, D] (+ extra [B, E]) -> mean, std [B, H, A]."""
        out = self.net(self._inputs(z_t, z_goal, extra))
        mean, log_std = out.view(-1, self.horizon, 2 * self.action_dim).chunk(2, dim=-1)
        log_std = log_std.clamp(self.min_log_std, self.max_log_std)
        return mean, log_std.exp()

    def loss(self, z_t: torch.Tensor, z_goal: torch.Tensor, actions: torch.Tensor,
             block_mask: torch.Tensor, extra: torch.Tensor | None = None,
             weight: torch.Tensor | None = None) -> torch.Tensor:
        """actions [B, H, A] (normalised); block_mask [B, H] bool (blocks that
        lead to the goal and have finite targets); optional per-sample weight
        [B]. Gaussian NLL, masked."""
        mean, std = self(z_t, z_goal, extra)
        valid = block_mask & torch.isfinite(actions).all(-1)
        nll = F.gaussian_nll_loss(mean, torch.nan_to_num(actions), std.square(), reduction="none", full=False)
        nll = nll.mean(-1)                                     # over action dims -> [B, H]
        w = valid.to(nll.dtype)
        if weight is not None:
            w = w * weight.to(nll.dtype).unsqueeze(1)
        return (nll * w).sum() / w.sum().clamp_min(1.0)

    @torch.no_grad()
    def sample(self, z_t: torch.Tensor, z_goal: torch.Tensor, k: int,
               generator: torch.Generator | None = None, extra: torch.Tensor | None = None,
               std_scale: float = 1.0) -> torch.Tensor:
        """k candidate chunks per row: [B, k, H, A]; candidate 0 is the mean."""
        mean, std = self(z_t, z_goal, extra)
        std = std * std_scale
        # draw on the generator's device (a CPU generator cannot fill a CUDA tensor), then move
        gen_dev = generator.device if generator is not None else mean.device
        eps = torch.randn((mean.shape[0], k, *mean.shape[1:]), generator=generator, device=gen_dev,
                          dtype=mean.dtype).to(mean.device)
        eps[:, 0] = 0
        return mean.unsqueeze(1) + std.unsqueeze(1) * eps


class FlowChunkPolicy(nn.Module):
    """System 1 as a conditional flow-matching policy over action chunks.

    A Gaussian head cannot represent several distinct ways of reaching a goal
    (it averages them); a flow policy transports noise to the action chunk
    distribution and can. Training is rectified-flow matching: x_t = (1-t) x0
    + t x1 with x0 ~ N(0, I), target velocity x1 - x0, masked per block like
    ``ChunkPolicy.loss``. Sampling integrates the velocity field with Euler
    steps from x0; candidate 0 starts from x0 = 0 (deterministic output),
    ``std_scale`` scales the starting noise of the others.
    Same interface as ``ChunkPolicy``: ``loss``, ``sample``.
    """

    def __init__(self, latent_dim: int, horizon: int, action_dim: int, hidden: int = 512,
                 extra_dim: int = 0, n_steps: int = 10, time_dim: int = 64):
        super().__init__()
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        self.horizon, self.action_dim, self.extra_dim, self.n_steps = horizon, action_dim, extra_dim, n_steps
        self.time_dim = time_dim
        d_in = 3 * latent_dim + extra_dim + horizon * action_dim + time_dim
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, horizon * action_dim))

    def _cond(self, z_t: torch.Tensor, z_goal: torch.Tensor, extra: torch.Tensor | None) -> torch.Tensor:
        parts = [z_t, z_goal, z_goal - z_t]
        if self.extra_dim:
            if extra is None or extra.shape[-1] != self.extra_dim:
                raise ValueError(f"policy expects extra inputs of size {self.extra_dim}")
            parts.append(extra.to(device=z_t.device, dtype=z_t.dtype))
        elif extra is not None:
            raise ValueError("policy was built without extra inputs")
        return torch.cat(parts, dim=-1)

    def _temb(self, t: torch.Tensor) -> torch.Tensor:
        half = self.time_dim // 2
        freqs = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype) * (-math.log(1000.0) / max(half - 1, 1)))
        ang = 1000.0 * t[:, None] * freqs[None]
        return torch.cat([ang.sin(), ang.cos()], dim=-1)

    def velocity(self, cond: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """cond [B, C], x [B, H, A], t [B] -> velocity [B, H, A]."""
        h = torch.cat([cond, x.flatten(1), self._temb(t)], dim=-1)
        return self.net(h).view(-1, self.horizon, self.action_dim)

    def loss(self, z_t: torch.Tensor, z_goal: torch.Tensor, actions: torch.Tensor,
             block_mask: torch.Tensor, extra: torch.Tensor | None = None,
             weight: torch.Tensor | None = None) -> torch.Tensor:
        cond = self._cond(z_t, z_goal, extra)
        x1 = torch.nan_to_num(actions)
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
        xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1
        err = (self.velocity(cond, xt, t) - (x1 - x0)).square().mean(-1)        # [B, H]
        w = (block_mask & torch.isfinite(actions).all(-1)).to(err.dtype)
        if weight is not None:
            w = w * weight.to(err.dtype).unsqueeze(1)
        return (err * w).sum() / w.sum().clamp_min(1.0)

    @torch.no_grad()
    def sample(self, z_t: torch.Tensor, z_goal: torch.Tensor, k: int,
               generator: torch.Generator | None = None, extra: torch.Tensor | None = None,
               std_scale: float = 1.0) -> torch.Tensor:
        """k candidate chunks per row: [B, k, H, A]; candidate 0 starts from zero noise."""
        cond = self._cond(z_t, z_goal, extra)
        B = cond.shape[0]
        gen_dev = generator.device if generator is not None else cond.device
        x = torch.randn((B, k, self.horizon, self.action_dim), generator=generator, device=gen_dev,
                        dtype=cond.dtype).to(cond.device) * std_scale
        x[:, 0] = 0
        x = x.reshape(B * k, self.horizon, self.action_dim)
        cond = cond.repeat_interleave(k, dim=0)
        dt = 1.0 / self.n_steps
        for i in range(self.n_steps):
            t = torch.full((B * k,), i * dt, device=cond.device, dtype=cond.dtype)
            x = x + dt * self.velocity(cond, x, t)
        return x.view(B, k, self.horizon, self.action_dim)


class ResidualChunkPolicy(nn.Module):
    """A frozen System 1 plus a learned, bounded per-block correction (residual practice, e.g. RL with a
    real-world success signal). The correction is shared across the fs steps of a block:
    chunk[b] += max_delta * tanh(net(z_t, z_goal))[b]. The last layer starts at zero, so an untrained
    residual reproduces the base policy exactly."""

    def __init__(self, base: nn.Module, latent_dim: int, n_blocks: int, block_dim: int, step_dim: int,
                 hidden: int = 256, max_delta: float = 1.0, init_log_std: float = -1.2):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.horizon, self.action_dim, self.extra_dim = base.horizon, base.action_dim, getattr(base, "extra_dim", 0)
        self.n_blocks, self.block_dim, self.step_dim, self.max_delta = n_blocks, block_dim, step_dim, max_delta
        self.net = nn.Sequential(nn.Linear(3 * latent_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                 nn.Linear(hidden, n_blocks * step_dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.log_std = nn.Parameter(torch.full((n_blocks * step_dim,), init_log_std))

    def correction(self, z_t: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        """Mean correction [B, n_blocks * step_dim] (normalised action units)."""
        return self.max_delta * torch.tanh(self.net(torch.cat([z_t, z_goal, z_goal - z_t], -1)))

    def apply(self, chunk: torch.Tensor, corr: torch.Tensor) -> torch.Tensor:
        """chunk [B, H, A] (A = steps_per_block * step_dim) + per-block correction [B, n_blocks * step_dim]."""
        B, H, A = chunk.shape
        c = corr.view(B, self.n_blocks, 1, self.step_dim).expand(B, self.n_blocks, A // self.step_dim, self.step_dim)
        out = chunk.clone()
        out[:, : self.n_blocks] = out[:, : self.n_blocks] + c.reshape(B, self.n_blocks, A)
        return out

    @torch.no_grad()
    def base_chunk(self, z_t, z_goal, extra=None) -> torch.Tensor:
        return self.base.sample(z_t, z_goal, 1, extra=extra)[:, 0]

    @torch.no_grad()
    def sample(self, z_t: torch.Tensor, z_goal: torch.Tensor, k: int, generator: torch.Generator | None = None,
               extra: torch.Tensor | None = None, std_scale: float = 1.0) -> torch.Tensor:
        """k candidates [B, k, H, A]; candidate 0 = base chunk + mean correction (deterministic)."""
        base = self.base_chunk(z_t, z_goal, extra)
        mu = self.correction(z_t, z_goal)
        gen_dev = generator.device if generator is not None else mu.device
        eps = torch.randn((mu.shape[0], k, mu.shape[1]), generator=generator, device=gen_dev, dtype=mu.dtype).to(mu.device)
        eps[:, 0] = 0
        corr = mu.unsqueeze(1) + std_scale * self.log_std.exp() * eps
        return torch.stack([self.apply(base, corr[:, j]) for j in range(k)], 1)


def policy_from_checkpoint(pk: dict) -> nn.Module:
    """Rebuild a System 1 policy (Gaussian, flow, or residual-corrected) from a saved checkpoint dict."""
    kind = pk.get("policy_type", "gauss")
    if kind == "residual":
        base = policy_from_checkpoint(pk["base"])
        r = pk["residual"]
        pol = ResidualChunkPolicy(base, r["latent_dim"], r["n_blocks"], r["block_dim"], r["step_dim"],
                                  hidden=r.get("hidden", 256), max_delta=r.get("max_delta", 1.0))
        pol.load_state_dict(pk["state_dict"])
        return pol
    if kind == "flow":
        pol = FlowChunkPolicy(pk["latent_dim"], pk["horizon"], pk["action_dim"], extra_dim=pk.get("extra_dim", 0),
                              n_steps=pk.get("flow_steps", 10))
    elif kind == "gauss":
        pol = ChunkPolicy(pk["latent_dim"], pk["horizon"], pk["action_dim"], extra_dim=pk.get("extra_dim", 0))
    else:
        raise ValueError(f"unknown policy_type {kind!r}")
    pol.load_state_dict(pk["state_dict"])
    return pol


class PolicyPool(nn.Module):
    """Several independently trained System 1 policies used as one proposer.

    ``mode='select'``: ``sample`` returns the members' mean chunks as the
    candidates (k must equal the number of members; candidate 0 is member 0),
    so a verifier chooses between plausible, diverse proposals rather than
    between noise perturbations of one proposal.
    ``mode='mean'``: one candidate, the average of the members' means (the
    verifier-free way to combine them).
    """

    def __init__(self, members: list[nn.Module], mode: str = "select"):
        super().__init__()
        if mode not in ("select", "mean"):
            raise ValueError("mode must be 'select' or 'mean'")
        if len(members) < 2:
            raise ValueError("a pool needs at least two policies")
        shapes = {(m.horizon, m.action_dim, m.extra_dim) for m in members}
        if len(shapes) != 1:
            raise ValueError(f"pool members disagree on (horizon, action_dim, extra_dim): {shapes}")
        self.members, self.mode = nn.ModuleList(members), mode
        self.horizon, self.action_dim, self.extra_dim = shapes.pop()

    @property
    def k(self) -> int:
        return len(self.members) if self.mode == "select" else 1

    @torch.no_grad()
    def sample(self, z_t: torch.Tensor, z_goal: torch.Tensor, k: int,
               generator: torch.Generator | None = None, extra: torch.Tensor | None = None,
               std_scale: float = 1.0) -> torch.Tensor:
        if k != self.k:
            raise ValueError(f"pool in mode {self.mode!r} proposes {self.k} candidates, asked for {k}")
        # each member's deterministic proposal (Gaussian: its mean; flow: its zero-noise sample)
        means = torch.stack([m.sample(z_t, z_goal, 1, extra=extra)[:, 0] for m in self.members], 1)  # [B, M, H, A]
        return means if self.mode == "select" else means.mean(1, keepdim=True)


def hindsight_batch(emb: torch.Tensor, actions: torch.Tensor, horizon: int,
                    generator: torch.Generator | None = None):
    """From clips emb [B, T, D] (T >= horizon + 1) and action blocks [B, T, A],
    draw a goal k in 1..horizon per clip. Returns z_t, z_goal, chunk [B, H, A],
    block_mask [B, H] (blocks j < k)."""
    B, T, _ = emb.shape
    if T < horizon + 1:
        raise ValueError(f"need clips of >= {horizon + 1} frames, got {T}")
    k = torch.randint(1, horizon + 1, (B,), generator=generator, device=emb.device)
    z_goal = emb[torch.arange(B, device=emb.device), k]
    mask = torch.arange(horizon, device=emb.device).unsqueeze(0) < k.unsqueeze(1)
    return emb[:, 0], z_goal, actions[:, :horizon], mask


class SubgoalNet(nn.Module):
    """Slow timescale of a two-level System 1: from (z_t, z_goal) predict the
    latent ``horizon`` blocks ahead on the way to the goal (the goal itself if
    it is closer). The fast policy then acts toward this subgoal, so goals
    beyond the policy's horizon are reached by chaining subgoals (cf. HIQL,
    Park et al. 2023; H-JEPA's top-down subgoals). Trained by regression on
    hindsight windows (``subgoal_batch``); a residual on z_t keeps outputs near
    the data manifold."""

    def __init__(self, latent_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * latent_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, latent_dim))

    def forward(self, z_t: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        return z_t + self.net(torch.cat([z_t, z_goal, z_goal - z_t], dim=-1))

    def loss(self, z_t: torch.Tensor, z_goal: torch.Tensor, z_sub: torch.Tensor) -> torch.Tensor:
        return (self(z_t, z_goal) - z_sub).square().sum(-1).mean()


def subgoal_batch(emb: torch.Tensor, horizon: int, generator: torch.Generator | None = None):
    """From clips emb [B, T, D] with T >= 2 * horizon + 1: goal k in 1..2H,
    subgoal target = frame min(k, H). Returns z_t, z_goal, z_sub."""
    B, T, _ = emb.shape
    if T < 2 * horizon + 1:
        raise ValueError(f"need clips of >= {2 * horizon + 1} frames, got {T}")
    k = torch.randint(1, 2 * horizon + 1, (B,), generator=generator, device=emb.device)
    rows = torch.arange(B, device=emb.device)
    return emb[:, 0], emb[rows, k], emb[rows, k.clamp(max=horizon)]
