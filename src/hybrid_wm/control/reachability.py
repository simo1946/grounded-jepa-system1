"""Learned reachability (temporal distance) critic for latent planning.

Finding that motivates it (bench/lewm_suite, OGBench-Cube): choosing among
System 1 proposals by latent squared distance to the goal *lowered* success
(90% -> 78%), and the CEM planner optimising that same cost reached only
64-66%. Euclidean distance in a JEPA latent is not "how far is the goal".

The critic estimates the number of action blocks needed to go from state z_a
to goal z_b, learned from offline trajectories:
  * positive pairs: frames i <= j of the same clip, target bin = j - i
    (0 .. max_steps);
  * negative pairs: frames from different clips -> the "far" bin;
  * classification over bins (robust to the multimodal, bounded target);
    expected steps = sum_k p_k * value_k, with the far bin valued at
    ``far_value``.
Directional by construction (no reversed pairs: going back may be
impossible, e.g. once a cube has been dropped).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReachabilityCritic(nn.Module):
    def __init__(self, latent_dim: int, max_steps: int = 10, hidden: int = 512, far_value: float | None = None):
        super().__init__()
        self.max_steps = max_steps
        self.n_bins = max_steps + 2                      # 0..max_steps, far
        self.net = nn.Sequential(
            nn.Linear(3 * latent_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, self.n_bins))
        values = torch.arange(self.n_bins, dtype=torch.float32)
        values[-1] = float(far_value if far_value is not None else 2 * max_steps)
        self.register_buffer("values", values)

    def logits(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_a, z_b, z_b - z_a], dim=-1))

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """Expected action blocks from z_a to z_b; any leading shape [..., D] -> [...]."""
        p = self.logits(z_a, z_b).float().softmax(-1)
        return (p * self.values).sum(-1)

    def loss(self, emb: torch.Tensor, n_pos: int = 4, neg_weight: float = 0.25,
             generator: torch.Generator | None = None) -> torch.Tensor:
        """emb [B, T, D] latents of consecutive blocks from B clips (B >= 2)."""
        B, T, _ = emb.shape
        if B < 2 or T < 2:
            raise ValueError("need >= 2 clips of >= 2 frames")
        dev = emb.device
        i = torch.randint(0, T, (B, n_pos), generator=generator, device=dev)
        j = torch.randint(0, T, (B, n_pos), generator=generator, device=dev)
        i, j = torch.minimum(i, j), torch.maximum(i, j)
        rows = torch.arange(B, device=dev)[:, None].expand(B, n_pos)
        za, zb = emb[rows, i], emb[rows, j]
        tgt = (j - i).clamp_max(self.max_steps)
        pos = F.cross_entropy(self.logits(za, zb).reshape(-1, self.n_bins), tgt.reshape(-1))
        # negatives: goal from another clip (shift by a random non-zero offset)
        shift = int(torch.randint(1, B, (1,), generator=generator).item())
        ia = torch.randint(0, T, (B,), generator=generator, device=dev)
        ib = torch.randint(0, T, (B,), generator=generator, device=dev)
        za_n = emb[torch.arange(B, device=dev), ia]
        zb_n = emb[(torch.arange(B, device=dev) + shift) % B, ib]
        far = torch.full((B,), self.n_bins - 1, device=dev, dtype=torch.long)
        neg = F.cross_entropy(self.logits(za_n, zb_n), far)
        return pos + neg_weight * neg


class ReachObjective(nn.Module):
    """stable-worldmodel Objective: expected blocks from the last predicted
    latent to the goal latent. Drop-in replacement for GoalMSE."""

    def __init__(self, critic: ReachabilityCritic, pred_key: str = "predicted_emb", goal_key: str = "goal_emb"):
        super().__init__()
        self.critic, self.pred_key, self.goal_key = critic, pred_key, goal_key

    def forward(self, info_dict: dict) -> torch.Tensor:
        pred = info_dict[self.pred_key][:, :, -1]                   # [B, S, D]
        goal = info_dict[self.goal_key][:, -1]                      # [B, D]
        return self.critic(pred, goal[:, None].expand_as(pred))     # [B, S]
