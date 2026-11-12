"""Continual learning from the robot's own successes and errors.

Each round:
  1. Act: System 1 (with mild exploration) runs on start/goal pairs from the
     TRAINING episodes (never the evaluation pairs), replanning every block.
     Recorded per decision: latent, executed action block, proprioception,
     object position, whether the env reports the goal reached.
  2. Reinject: every episode becomes training data.
       - failures: relabelled with the goals they actually reached
         (hindsight), so a wrong decision for goal g is a correct decision
         for the state it produced;
       - successes: also kept with the commanded goal, up-weighted (reward).
     Surprise (world-model one-block prediction error on these on-policy
     transitions vs expert transitions) is logged as the signal that the
     robot is off the training distribution.
  3. Learn: fine-tune the policy on expert data + all on-policy episodes so
     far. Control arm: identical fine-tuning steps on expert data only, so
     any gain is attributable to the reinjected experience, not to extra
     training.
  4. Evaluate both arms on the same 50 non-trivial evaluation pairs.

    python learn_loop.py --spec lewm+idm+sa --seed 0 --rounds 3 --episodes 100
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_flush_denormal(True)

from models import HOME, Spec, TrainConfig, build_model  # noqa: E402
from sensors import readings_from_info  # noqa: E402
from tasks import apply_render_profile, get_task, trivial_mask  # noqa: E402

H = 5


class Recorder:
    """Solver that acts with System 1 (mean + std_scale * noise) and records
    what it saw and did. Replanning every block, so one call = one block."""

    def __init__(self, lewm, policy, state_cols, object_cols, std_scale: float, seed: int):
        self.lewm, self.policy = lewm, policy
        self.state_cols, self.object_cols = state_cols, object_cols
        self.std_scale, self.gen = std_scale, torch.Generator().manual_seed(seed)
        self.reset()

    def configure(self, *, action_space, n_envs: int, config) -> None:
        self._n, self._a = n_envs, int(np.prod(action_space.shape[1:])) * config.action_block

    @property
    def n_envs(self):
        return self._n

    @property
    def action_dim(self):
        return self._a

    @property
    def horizon(self):
        return H

    def __call__(self, *a, **k):
        return self.solve(*a, **k)

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action=None) -> dict:
        # The policy calls the solver only for live envs; rows are matched to
        # envs by their (unique) goal object position.
        z = self.lewm.encode({"pixels": info_dict["pixels"].float()})["emb"][:, -1]
        zg = self.lewm.encode({"pixels": info_dict["goal"].float()})["emb"][:, -1]
        chunk = self.policy.sample(z, zg, 2, generator=self.gen, std_scale=self.std_scale)[:, 1]   # [n, H, A]
        proprio, obj, goal_obj = readings_from_info(info_dict, self.state_cols, self.object_cols)
        if self.goals is None:                               # first call of a chunk: every env is live
            self.goals = goal_obj.clone()
            self.envs = [{"z": [], "action": [], "proprio": [], "object": [], "z_goal": zg[i].clone(),
                          "goal_obj": goal_obj[i].clone()} for i in range(len(z))]
        env_ids = torch.cdist(goal_obj.double(), self.goals.double()).argmin(1).tolist()
        for row, e in enumerate(env_ids):
            rec = self.envs[e]
            for key, v in (("z", z), ("action", chunk[:, 0]), ("proprio", proprio), ("object", obj)):
                rec[key].append(v[row].clone().float())
        return {"actions": chunk.cpu()}

    def reset(self):
        self.goals, self.envs = None, []


def finish_episodes(envs: list[dict], successes) -> list[dict]:
    """Frames F (recorded decision points) and actions A per env. A success
    ends with an action whose outcome (the goal) was not rendered to the
    policy, so A = F; otherwise the last action's outcome is unobserved and
    is dropped (A = F - 1)."""
    out = []
    for rec, ok in zip(envs, successes):
        F = len(rec["z"])
        A = F if ok else F - 1
        if F < 1 or A < 1:
            continue
        st = lambda key, n: torch.stack(rec[key][:n])  # noqa: E731
        out.append({"z": st("z", F), "action": st("action", A), "proprio": st("proprio", F),
                    "object": st("object", F), "z_goal": rec["z_goal"].float(), "goal_obj": rec["goal_obj"].float(),
                    "success": bool(ok), "reached_block": A if ok else -1})
    return out


@torch.inference_mode()
def surprise(lewm, episodes: list[dict]) -> float:
    """Mean squared one-block prediction error of the world model on (z_t, a_t) -> z_{t+1}."""
    errs = []
    for e in episodes:
        T = min(e["action"].shape[0], e["z"].shape[0] - 1)    # transitions with an observed outcome
        if T < 1:
            continue
        info = {"pixels": torch.zeros(T, 1, 1, 1), "emb": e["z"][:T, None, None, :]}
        pred = lewm.rollout(info, e["action"][:T, None, None, :])["predicted_emb"][:, 0, 1]
        errs.append((pred - e["z"][1:T + 1]).square().sum(-1))
    return float(torch.cat(errs).mean()) if errs else float("nan")


def collect(lewm, policy_meta, task, cfg, n_episodes: int, seed: int, std_scale: float) -> tuple:
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from sklearn import preprocessing
    from torchvision.transforms import v2 as T

    from hybrid_wm.control.latent_policy import ChunkPolicy

    pol = ChunkPolicy(policy_meta["latent_dim"], policy_meta["horizon"], policy_meta["action_dim"])
    pol.load_state_dict(policy_meta["state_dict"])
    pol.eval()
    ds = swm.data.HDF5Dataset("cube", cache_dir=str(HOME), keys_to_cache=list(task.process_cols))
    d = np.asarray(ds.get_col_data("action"))
    process = {"action": preprocessing.StandardScaler().fit(d[~np.isnan(d).any(axis=1)])}
    img_t = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True),
                       T.Normalize(**spt.data.dataset_stats.ImageNet), T.Resize(size=cfg.img_size)])
    # training-episode start/goal pairs, non-trivial, drawn per round
    rng = np.random.default_rng(10_000 + seed)
    lengths = np.asarray(ds.lengths)
    eps, starts = [], []
    while len(eps) < n_episodes:
        e = int(rng.integers(len(lengths)))
        s = int(rng.integers(lengths[e] - task.goal_offset))
        if not trivial_mask(task, ds, [e], [s], task.goal_offset)[0]:
            eps.append(e)
            starts.append(s)
    rec = Recorder(lewm, pol, task.state_cols, task.object_cols, std_scale, seed)
    policy = swm.policy.WorldModelPolicy(
        solver=rec, config=swm.PlanConfig(horizon=H, receding_horizon=1, action_block=cfg.frameskip),
        process=process, transform={"pixels": img_t, "goal": img_t})
    episodes, chunk = [], 5
    for lo in range(0, n_episodes, chunk):
        rec.reset()
        world = swm.World(task.env_id, num_envs=chunk, image_shape=(cfg.img_size, cfg.img_size),
                          max_episode_steps=2 * task.eval_budget, **task.eval_kwargs)
        world.set_policy(policy)
        m = world.evaluate(dataset=ds, episodes_idx=eps[lo:lo + chunk], start_steps=starts[lo:lo + chunk],
                           goal_offset=task.goal_offset, eval_budget=task.eval_budget, callables=task.eval_callables)
        world.close()
        episodes += finish_episodes(rec.envs, np.asarray(m["episode_successes"]).astype(bool).tolist())
    return episodes, {"pairs": list(zip(eps, starts))}


def main(spec_str: str, seed: int, rounds: int, n_episodes: int, steps: int, std_scale: float) -> None:
    import cached_policy
    import evaluate

    cfg = TrainConfig()
    task = get_task("cube", cfg.img_size)
    apply_render_profile(task)
    spec = Spec.parse(spec_str)
    ckpt = HOME / "checkpoints" / f"cube/{spec.name()}/seed{seed}"
    info = json.loads((ckpt / "train_info.json").read_text())
    wm = build_model(spec, task.action_dim, info["state_dim"], cfg)
    wm.load_state_dict(torch.load(ckpt / "weights.pt", map_location="cpu"))
    lewm = wm.lewm.eval().requires_grad_(False)
    lewm.interpolate_pos_encoding = True
    exp_dir = ckpt / "experience"
    exp_dir.mkdir(exist_ok=True)
    log_path = HOME / "results" / f"learn_loop__{spec.name()}__seed{seed}.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {"rounds": []}

    ev = lambda pf: evaluate.main("cube", spec_str, seed, 50, 42, "cube", "cube_eval", None, None, "cpu", 10,  # noqa: E731
                                  "s1", 0.6, 1, 5, "mse", "nontrivial", True, "cost", pf)
    if not (ckpt / "policy_cached.pt").exists():
        cached_policy.train(spec_str, seed, "none", "policy_cached.pt", 1500)
    ev("policy_cached.pt")
    loop_prev, ctrl_prev, files = "policy_cached.pt", "policy_cached.pt", []
    for r in range(1, rounds + 1):
        t0 = time.time()
        f = exp_dir / f"round{r}.pt"
        if not f.exists():
            meta = torch.load(ckpt / loop_prev, map_location="cpu")
            eps, pairs = collect(lewm, meta, task, cfg, n_episodes, seed * 100 + r, std_scale)
            torch.save({"episodes": eps, "round": r, "policy": loop_prev, **pairs}, f)
        eps = torch.load(f, map_location="cpu")["episodes"]
        files.append(str(f))
        stats = {"round": r, "episodes": len(eps), "train_successes": int(sum(e["success"] for e in eps)),
                 "surprise_on_policy": surprise(lewm, eps), "collect_min": round((time.time() - t0) / 60, 1)}
        loop_f, ctrl_f = f"policy_loop_r{r}.pt", f"policy_ctrl_r{r}.pt"
        if not (ckpt / loop_f).exists():
            cached_policy.train(spec_str, seed, "none", loop_f, steps, init=loop_prev, experience=files, lr=3e-4)
        if not (ckpt / ctrl_f).exists():
            cached_policy.train(spec_str, seed, "none", ctrl_f, steps, init=ctrl_prev, lr=3e-4)
        for name, pf in (("loop", loop_f), ("control", ctrl_f)):
            res = ev(pf) or json.loads(evaluate.result_path(
                "cube", spec_str, seed, "s1", 1, 0.6, "mse", "nontrivial", 25, 50, 42, "cost", pf).read_text())
            stats[f"eval_{name}"] = res["success_rate"]
        print(json.dumps(stats), flush=True)
        log["rounds"] = [x for x in log["rounds"] if x["round"] != r] + [stats]
        log_path.write_text(json.dumps(log, indent=2))
        loop_prev, ctrl_prev = loop_f, ctrl_f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="lewm+idm+sa")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--std-scale", type=float, default=0.5)
    a = ap.parse_args()
    main(a.spec, a.seed, a.rounds, a.episodes, a.steps, a.std_scale)
