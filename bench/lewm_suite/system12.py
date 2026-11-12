"""System 1 / System 2 control with verify-then-act, as a drop-in solver for
stable-worldmodel's WorldModelPolicy.

Per replanning call and per environment:
  1. System 1: the latent policy proposes K action chunks for (z_now, z_goal)
     (candidate 0 = policy mean).
  2. Verify: the world model imagines each chunk once (K rollouts, versus
     samples x iterations = 9,000 for the CEM baseline) and scores it with
     the same GoalMSE cost the planner uses.
  3. Act if the best imagined chunk makes enough progress:
         cost_S1 <= rho * cost_now   or   cost_S1 <= tau_abs
     (tau_abs = typical one-block latent displacement, measured on training
     data; rho fixed a priori, not tuned on evaluation episodes).
  4. Otherwise System 2: the authors' CEM solver, warm-started from the best
     System 1 chunk.

``mode='s1'`` always acts on System 1 (fast path only, for ablation).
``subgoal`` (two-level System 1): the policy acts toward
SubgoalNet(z_now, z_goal) instead of z_goal.
``camera_drop`` (partial observability): from an episode's second decision
on, the camera frame is missing with this probability. The missing latent is
filled by ``fill``: 'blank' (encode a black frame: what a memoryless system
sees), 'hold' (last latent), or 'predict' (world-model rollout of the last
latent estimate under the action blocks executed since; the predictor acts
as a state estimator). Envs are tracked across calls by their goal latent.
``select='random'`` (ablation control) picks a uniformly random sampled
candidate (never the mean) instead of the lowest-cost one: it separates the
effect of sampling noise from the effect of world-model selection.
Statistics (acceptance, timing) are collected in ``self.stats``.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch


def _expand(info: dict, idx: torch.Tensor, k: int, device=None, dtype=None) -> dict:
    """Select envs ``idx`` and add a candidate axis of size k; tensors go to the
    model's device and float dtype (as CEMSolver does)."""
    out = {}
    for key, v in info.items():
        if torch.is_tensor(v):
            v = v[idx]
            if device is not None:
                v = v.to(device=device, dtype=dtype if v.is_floating_point() else None)
            out[key] = v.unsqueeze(1).expand(v.shape[0], k, *v.shape[1:])
        elif isinstance(v, np.ndarray):
            out[key] = np.repeat(v[idx.numpy()][:, None, ...], k, axis=1)
        else:
            out[key] = v
    return out


def _select(info: dict, idx: torch.Tensor) -> dict:
    out = {}
    for key, v in info.items():
        if torch.is_tensor(v):
            out[key] = v[idx]
        elif isinstance(v, np.ndarray):
            out[key] = v[idx.numpy()]
        else:
            out[key] = v
    return out


class GatedSolver:
    def __init__(self, cem, lewm, policy, *, k: int = 16, rho: float = 0.6, tau_abs: float = 0.0,
                 mode: str = "s1s2", seed: int = 0, distance=None, select: str = "cost", extra_fn=None,
                 subgoal=None, camera_drop: float = 0.0, fill: str = "blank", receding: int = 5,
                 subgoal_gate: float | None = None, std_scale: float = 1.0, scorer=None):
        if mode not in ("s1s2", "s1"):
            raise ValueError("mode must be 's1s2' or 's1'")
        if select not in ("cost", "random", "scorer"):
            raise ValueError("select must be 'cost', 'random' or 'scorer'")
        if select == "scorer" and (scorer is None or mode != "s1" or k < 2):
            raise ValueError("select='scorer' needs a scorer, mode 's1' and k >= 2")
        self.scorer = scorer              # (info_dict, z_now, z_target, cand [n, K, H, A]) -> [n, K], lower = better
        if select == "random" and (mode != "s1" or k < 2):
            raise ValueError("select='random' is an S1-only control and needs k >= 2")
        self.select = select
        if fill not in ("blank", "hold", "predict"):
            raise ValueError("fill must be 'blank', 'hold' or 'predict'")
        if camera_drop and mode != "s1":
            raise ValueError("camera dropout is implemented for System 1 only")
        self.subgoal, self.camera_drop, self.fill, self.receding = subgoal, camera_drop, fill, receding
        self.subgoal_gate = subgoal_gate  # use the subgoal only if ||z_goal - z_now||^2 exceeds this
        if std_scale < 0:
            raise ValueError("std_scale must be >= 0")
        self.std_scale = std_scale        # spread of System 1 candidates around the mean
        self.drop_gen = torch.Generator().manual_seed(seed + 7919)
        self.memory: list[dict] = []      # per tracked env: goal latent, last latent estimate, executed blocks
        self.extra_fn = extra_fn          # info_dict -> extra policy inputs (simulated sensors) or None
        self.cem, self.cost, self.lewm, self.policy = cem, cem.cost, lewm, policy
        self.k, self.rho, self.tau_abs, self.mode = k, rho, tau_abs, mode
        # distance(z_now, z_goal) -> [n], in the same units as the planner's cost
        self.distance = distance or (lambda a, b: (a - b).square().sum(-1))
        self.gen = torch.Generator().manual_seed(seed)
        self.stats = {"calls": 0, "env_decisions": 0, "s1_accepted": 0, "s1_seconds": 0.0, "s2_seconds": 0.0,
                      "camera_dropped": 0}

    # --- Solver protocol (delegated) ------------------------------------------------
    def configure(self, *, action_space, n_envs: int, config: Any) -> None:
        self.cem.configure(action_space=action_space, n_envs=n_envs, config=config)

    @property
    def action_dim(self) -> int:
        return self.cem.action_dim

    @property
    def n_envs(self) -> int:
        return self.cem.n_envs

    @property
    def horizon(self) -> int:
        return self.cem.horizon

    def __call__(self, *args, **kwargs) -> dict:
        return self.solve(*args, **kwargs)

    # --- System 1 / 2 ---------------------------------------------------------------
    @torch.inference_mode()
    def reset_episodes(self) -> None:
        """Forget per-episode memory. Called by evaluate.py before each new batch of episodes, so memory is
        never carried from one batch to the next."""
        self.memory = []

    def _sync(self) -> None:
        if torch.cuda.is_available():                     # CUDA runs asynchronously: wait before reading the clock
            torch.cuda.synchronize()

    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        t0 = time.perf_counter()
        n = len(next(iter(v for v in info_dict.values() if torch.is_tensor(v) or isinstance(v, np.ndarray))))
        p0 = next(self.lewm.parameters())
        dtype, dev = p0.dtype, p0.device
        z_now = self.lewm.encode({"pixels": info_dict["pixels"].to(dev, dtype)})["emb"][:, -1]
        z_goal = self.lewm.encode({"pixels": info_dict["goal"].to(dev, dtype)})["emb"][:, -1]
        slots = None
        if self.camera_drop:
            z_now, slots = self._perceive(info_dict, z_now, z_goal, dtype)
        cost_now = self.distance(z_now, z_goal)

        extra = self.extra_fn(info_dict) if self.extra_fn is not None else None
        z_target = z_goal
        if self.subgoal is not None:
            z_sub = self.subgoal(z_now, z_goal)
            if self.subgoal_gate is None:
                z_target = z_sub
            else:                                         # far goals only
                far = ((z_goal - z_now).square().sum(-1) > self.subgoal_gate).unsqueeze(-1)
                z_target = torch.where(far, z_sub, z_goal)
        kw = {"std_scale": self.std_scale} if self.std_scale != 1.0 else {}
        cand = self.policy.sample(z_now, z_target, self.k, generator=self.gen, extra=extra, **kw)   # [n, K, H, A]
        cand = cand.to(dtype)
        if cand.shape[2] < self.horizon:
            raise ValueError(f"policy horizon {cand.shape[2]} < planner horizon {self.horizon}")
        cand = cand[:, :, : self.horizon]
        if self.select in ("random", "scorer"):           # no world-model rollout
            if self.select == "random":                   # control: random sampled candidate
                best = torch.randint(1, self.k, (n,), generator=self.gen).to(cand.device)
            else:                                         # external scorer (outcome critic or true-outcome ceiling)
                best = torch.as_tensor(self.scorer(info_dict, z_now, z_target, cand)).argmin(1).to(cand.device)
            s1_plan = cand[torch.arange(n, device=cand.device), best]
            s1_cost = torch.zeros(n, dtype=cand.dtype)
            accept = torch.ones(n, dtype=torch.bool)
            self._sync()
            t1 = time.perf_counter()
            self.stats["calls"] += 1
            self.stats["env_decisions"] += n
            self.stats["s1_accepted"] += n
            self.stats["s1_seconds"] += t1 - t0
            return {"actions": s1_plan.detach().cpu().clone(), "costs": s1_cost.tolist(),
                    "s1_accepted": accept.tolist()}
        if self.k == 1:                                   # no selection needed: skip the rollout
            costs = torch.zeros(n, 1, dtype=cand.dtype, device=cand.device)
            if self.mode != "s1":
                costs = self.cost.get_cost(_expand(info_dict, torch.arange(n), 1, dev, dtype), cand)
        else:
            costs = self.cost.get_cost(_expand(info_dict, torch.arange(n), self.k, dev, dtype), cand)  # [n, K]
        best = costs.argmin(1)
        s1_cost = costs.gather(1, best[:, None]).squeeze(1)
        s1_plan = cand[torch.arange(n, device=cand.device), best]                    # [n, H, A]
        cost_now = cost_now.to(s1_cost.device)
        accept = (s1_cost <= self.rho * cost_now) | (s1_cost <= self.tau_abs)
        if self.mode == "s1":
            accept = torch.ones_like(accept)
        self._sync()
        t1 = time.perf_counter()

        actions = s1_plan.detach().cpu().clone()
        rej = torch.nonzero(~accept).squeeze(-1).cpu()
        if rej.numel():
            out = self.cem.solve(_select(info_dict, rej), init_action=s1_plan[rej.to(s1_plan.device)].detach())
            actions[rej] = out["actions"].to(actions.dtype)
        self._sync()
        t2 = time.perf_counter()

        if slots is not None:                             # remember what was executed until the next decision
            for row, m in enumerate(slots):
                m["z"], m["blocks"] = z_now[row].clone(), actions[row, : self.receding].clone()
        self.stats["calls"] += 1
        self.stats["env_decisions"] += n
        self.stats["s1_accepted"] += int(accept.sum())
        self.stats["s1_seconds"] += t1 - t0
        self.stats["s2_seconds"] += t2 - t1
        return {"actions": actions, "costs": s1_cost.cpu().tolist(), "s1_accepted": accept.cpu().tolist()}

    # --- partial observability ------------------------------------------------------
    def _perceive(self, info_dict: dict, z_real: torch.Tensor, z_goal: torch.Tensor, dtype) -> tuple:
        slots, z_used = [], z_real.clone()
        dropped, fill_rows = [], []
        for row in range(len(z_real)):
            # batch composition changes between calls, so match goals up to float noise
            m = next((m for m in self.memory
                      if torch.allclose(m["goal"], z_goal[row], rtol=1e-4, atol=1e-4)), None)
            if any(m is s for s in slots):                # two episodes of one batch share a goal latent
                raise RuntimeError("two episodes in one batch have the same goal latent; per-episode memory "
                                   "would mix them")
            if m is None:                                 # first decision of an episode: camera available
                m = {"goal": z_goal[row].clone(), "z": None, "blocks": None}
                self.memory.append(m)
            elif float(torch.rand(1, generator=self.drop_gen)) < self.camera_drop:
                dropped.append(row)
                fill_rows.append(m)
            slots.append(m)
        if dropped:
            dev = z_real.device
            idx = torch.tensor(dropped, device=dev)
            if self.fill == "blank":
                px = info_dict["pixels"][idx.cpu()].to(dev, dtype)
                black = self._black(px)
                z_used[idx] = self.lewm.encode({"pixels": black})["emb"][:, -1]
            elif self.fill == "hold":
                z_used[idx] = torch.stack([m["z"] for m in fill_rows]).to(z_used.dtype)
            else:
                z0 = torch.stack([m["z"] for m in fill_rows]).to(dtype)
                blocks = torch.stack([m["blocks"] for m in fill_rows]).to(dev, dtype)   # [d, R, A]
                info = {"pixels": torch.zeros(len(idx), 1, 1, 1, dtype=dtype, device=dev), "emb": z0[:, None, None, :]}
                pred = self.lewm.rollout(info, blocks[:, None])["predicted_emb"][:, 0, -1]
                z_used[idx] = pred.to(z_used.dtype)
            self.stats["camera_dropped"] += len(dropped)
        return z_used, slots

    @staticmethod
    def _black(px: torch.Tensor) -> torch.Tensor:
        """A black frame in the planner's normalised pixel space (ImageNet stats)."""
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=px.dtype, device=px.device)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=px.dtype, device=px.device)
        shape = [1] * px.ndim
        shape[-3] = 3
        return ((0 - mean) / std).view(shape).expand_as(px).clone()
