"""Action- and state-grounding losses for JEPA world models.

LeWM-style training (next-embedding MSE + SIGReg) does not force the latent
to keep what control needs. Two auxiliary heads, trained jointly with the
encoder, push it to:

InverseDynamicsHead (as in Sensorimotor World Models, arXiv:2606.20104):
    a_hat_t = H([z_t, z_{t+1}]),  L_inv = MSE(a_hat_t, a_t)
    H is a 2-layer MLP. a_t is the (normalised) action block executed
    between consecutive model frames. SMWM uses it as the only anti-collapse
    term; here it is added on top of SIGReg.

StateAlignmentHead (our formulation; the idea follows "Physically Grounded
JEPA World Models", arXiv:2609.03565, whose exact loss we could not access):
    [s_hat_{t+1}, ds_hat_t] = G([z_t, z_{t+1}])
    L_sa = MSE(s_hat_{t+1}, s_{t+1}) + MSE(ds_hat_t, s_{t+1} - s_t)
    s is *robot proprioception only* (joints, end effector, gripper), which
    a real robot measures. Object poses are deliberately excluded: they are
    privileged simulator state.

(The Cube study's ``obj`` variant adds the object position to s as a
training-only target; see bench/lewm_suite/models.py.)

action_resolution_loss (ours): the predictor must place the observed next
latents closer to its prediction for the executed actions than to its
predictions for M nearby perturbed action sequences (log-ratio hinge, see the
function). It targets the failure measured on Cube: rollouts accurate on data,
but unable to separate action chunks closer than the policy's own noise, so
selection and planning exploit model error.

Both grounding losses mask transitions whose action or state contains NaN (sequence
boundaries in stable-worldmodel datasets) instead of training on zero-filled
targets.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def _mlp(d_in: int, d_out: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, d_out))


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """pred/target [..., D], valid [...] bool -> mean squared error over valid rows (0 if none)."""
    err = (pred - torch.nan_to_num(target)).pow(2).mean(-1)
    w = valid.to(err.dtype)
    return (err * w).sum() / w.sum().clamp_min(1.0)


class InverseDynamicsHead(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden: int = 512):
        super().__init__()
        self.net = _mlp(2 * latent_dim, action_dim, hidden)

    def forward(self, z_t: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_t, z_next], dim=-1))

    def loss(self, emb: torch.Tensor, actions: torch.Tensor,
             action_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """emb [B, T, D]; actions [B, T, A] (actions[:, t] leaves frame t);
        action_valid [B, T] bool. Uses transitions t = 0..T-2."""
        if emb.shape[:2] != actions.shape[:2]:
            raise ValueError(f"emb {tuple(emb.shape)} and actions {tuple(actions.shape)} disagree on [B, T]")
        pred = self(emb[:, :-1], emb[:, 1:])
        tgt = actions[:, :-1]
        valid = torch.isfinite(tgt).all(-1)
        if action_valid is not None:
            valid = valid & action_valid[:, :-1]
        return _masked_mse(pred, tgt, valid)


class StateAlignmentHead(nn.Module):
    def __init__(self, latent_dim: int, state_dim: int, hidden: int = 512):
        super().__init__()
        self.state_dim = state_dim
        self.net = _mlp(2 * latent_dim, 2 * state_dim, hidden)

    def forward(self, z_t: torch.Tensor, z_next: torch.Tensor):
        out = self.net(torch.cat([z_t, z_next], dim=-1))
        return out[..., :self.state_dim], out[..., self.state_dim:]

    def loss(self, emb: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """emb [B, T, D]; state [B, T, S] (normalised proprioception per frame)."""
        if emb.shape[:2] != state.shape[:2]:
            raise ValueError(f"emb {tuple(emb.shape)} and state {tuple(state.shape)} disagree on [B, T]")
        s_next_hat, ds_hat = self(emb[:, :-1], emb[:, 1:])
        s_t, s_next = state[:, :-1], state[:, 1:]
        valid = torch.isfinite(s_t).all(-1) & torch.isfinite(s_next).all(-1)
        return _masked_mse(s_next_hat, s_next, valid) + _masked_mse(ds_hat, s_next - s_t, valid)


def perturb_actions(actions: torch.Tensor, n_candidates: int = 4, sigma_min: float = 0.05,
                    sigma_max: float = 0.5, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """[B, T, A] -> [B, 1 + M, T, A]: the executed actions (index 0) and M copies
    with Gaussian noise, scale log-uniform in [sigma_min, sigma_max]."""
    import math

    if n_candidates < 1 or not 0 < sigma_min <= sigma_max:
        raise ValueError("need n_candidates >= 1 and 0 < sigma_min <= sigma_max")
    B, M = actions.shape[0], n_candidates
    dev, dt = actions.device, actions.dtype
    gdev = "cpu" if generator is not None else dev
    u = torch.rand((B, M, 1, 1), generator=generator, device=gdev).to(dev, dt)
    sigma = torch.exp(math.log(sigma_min) + u * (math.log(sigma_max) - math.log(sigma_min)))
    eps = torch.randn((B, M, *actions.shape[1:]), generator=generator, device=gdev).to(dev, dt)
    return torch.cat([actions.unsqueeze(1), actions.unsqueeze(1) + sigma * eps], 1)


def action_resolution_loss(pred_all: torch.Tensor, target: torch.Tensor, gamma: float = 0.1,
                           eps: float = 1e-6):
    """Hinge on log distance ratios: for every perturbed candidate m,
        max(0, log d_true - log d_m + log(1 + gamma)),
    where d = mean squared error of a candidate's prediction to the observed next
    latents. pred_all [B, 1 + M, T, D] comes from ONE predictor call over the true
    (index 0) and perturbed action sequences, so all candidates share the same
    normalisation (with BatchNorm, separate calls let a contrastive loss separate
    candidates by batch statistics). The loss is scale-free and bounded: once a
    perturbed prediction is (1 + gamma) times farther than the true one, it gets
    no gradient, so predictions are not pushed apart without limit (an unbounded
    contrastive objective inflated the predictions in our first version).
    Returns (loss, accuracy): accuracy = share of clips whose true actions are
    strictly closest."""
    import math

    d = (pred_all - target.detach().unsqueeze(1)).pow(2).sum(-1).mean(-1)               # [B, 1+M]
    ld = torch.log(d + eps)
    loss = torch.relu(ld[:, :1] - ld[:, 1:] + math.log1p(gamma)).mean()
    acc = (d[:, 0] < d[:, 1:].min(1).values).float().mean()
    return loss, acc.detach()
