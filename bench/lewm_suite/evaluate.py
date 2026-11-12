"""Planning evaluation with the authors' protocol (le-wm config/eval/<task>.yaml).

World.evaluate in dataset mode on held-out start/goal pairs (goal = state
`goal_offset` steps later, `budget` env steps), WorldModelPolicy + CEMSolver
(300 samples x 30 iterations, top-30), horizon 5, receding 5, frameskip 5,
GoalMSE cost. Only ``model.lewm`` plans; grounding heads are not used.

    python evaluate.py --task cube --spec lewm --seed 0
    python evaluate.py --task cube --spec random
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_flush_denormal(True)

from models import HOME, Spec, TrainConfig, build_model, state_columns  # noqa: E402
from tasks import apply_render_profile, get_task, trivial_mask  # noqa: E402


def pick_pairs(lengths, goal_offset: int, n: int, seed: int, exclude=None):
    """exclude(episodes, starts) -> bool mask of pairs to drop (vectorised)."""
    pairs = [(ep, s) for ep, L in enumerate(lengths) for s in range(L) if s + goal_offset < L]
    if exclude is not None and pairs:
        drop = exclude([p[0] for p in pairs], [p[1] for p in pairs])
        pairs = [p for p, d in zip(pairs, drop) if not d]
    if len(pairs) < n:
        raise ValueError(f"only {len(pairs)} valid start states for {n} evaluations")
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(len(pairs), size=n, replace=False).tolist())
    return [pairs[i][0] for i in pick], [pairs[i][1] for i in pick]


def _time_decisions(policy, out: list) -> None:
    """Record the wall-clock time of every decision: one call of the solver's solve() (encoding the frames,
    choosing the actions, copying them to the CPU), with CUDA synchronised. The policy calls the solver only
    when it replans; the steps in between replay the stored plan and are not counted."""
    solver = policy.solver
    if getattr(solver, "_timed", False):
        return
    orig = solver.solve

    def timed(*a, **k):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter()
        r = orig(*a, **k)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        out.append(time.perf_counter() - t)
        return r
    solver.solve = timed            # both solvers' __call__ forward to self.solve
    solver._timed = True


def _refresh_state_infos(world) -> None:
    """Before the first decision, the evaluator's infos still hold the default-reset robot/object state (only
    the start frame and dataset-named columns are injected). Sensor policies read that state, so overwrite the
    env-named state keys with the restored start state (compute_ob_info) when the run begins."""
    orig = world._run

    def run(max_steps, mode, on_step):
        for i, env in enumerate(world.envs.envs):
            fn = getattr(env.unwrapped, "compute_ob_info", None)
            if fn is None:
                continue
            for k, v in fn().items():
                if k in world.infos and not k.startswith("goal"):
                    arr = np.array(world.infos[k], copy=True)
                    arr[i] = np.broadcast_to(np.asarray(v, dtype=arr.dtype), arr[i].shape)
                    world.infos[k] = arr
        return orig(max_steps=max_steps, mode=mode, on_step=on_step)
    world._run = run





def result_path(task_name: str, spec_str: str, seed: int, controller: str, k: int, rho: float, cost: str,
                pairs: str, goal_offset: int, budget: int, eval_seed: int, select: str = "cost",
                policy_file: str = "policy.pt", receding: int = 5, camera_drop: float = 0.0,
                fill: str = "blank", num_eval: int = 50, std_scale: float = 1.0, pool: list | None = None,
                pool_mode: str = "select", exec_steps: int = 0):
    run = f"{task_name}/random" if spec_str == "random" else f"{task_name}/{Spec.parse(spec_str).name()}/seed{seed}"
    ctrl = "" if controller == "cem" else f"__{controller}"
    if controller != "cem" and k != 16:
        ctrl += f"_k{k}"
    if controller == "s1s2" and rho != 0.6:
        ctrl += f"_rho{rho:g}"
    if select != "cost":
        ctrl += f"_pick{select}"
    if controller != "cem" and policy_file != "policy.pt":
        ctrl += f"__pol-{policy_file.removesuffix('.pt')}"
    if controller != "cem" and pool:
        ctrl += f"__pool{len(pool) + 1}-{pool_mode}"
    if controller != "cem" and std_scale != 1.0:
        ctrl += f"__std{std_scale:g}"
    if receding != 5:
        ctrl += f"__rec{receding}"
    if exec_steps:
        ctrl += f"__exec{exec_steps}"
    if camera_drop:
        ctrl += f"__cam{camera_drop:g}-{fill}"
    if cost != "mse":
        ctrl += f"__cost{cost}"
    if pairs != "all":
        ctrl += f"__{pairs}"
    if num_eval != 50:
        ctrl += f"_n{num_eval}"
    return HOME / "results" / f"{run.replace('/', '__')}{ctrl}__off{goal_offset}_bud{budget}_es{eval_seed}.json"


def true_outcome_scorer(task, action_scaler):
    """Selection ceiling: execute each candidate open-loop in a scratch simulator from the env's true
    state and score it by the final cube distance to the goal cube (metres). Privileged; not deployable."""
    from oracle_sim import OracleSim
    sim = OracleSim(task.env_id, **task.eval_kwargs)
    mean, scale = np.asarray(action_scaler.mean_), np.asarray(action_scaler.scale_)
    A = len(mean)

    def last(v):
        v = v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
        v = v[:, -1] if v.ndim >= 3 else v
        return v.reshape(v.shape[0], -1).astype(np.float64)

    def score(info, z, zg, cand):
        q, v, c = last(info["qpos"]), last(info["qvel"]), last(info["control"])
        gp = last(info["goal_privileged_block_0_pos"])
        n, K = cand.shape[:2]
        raw = cand.detach().cpu().double().numpy().reshape(n, K, -1, A) * scale + mean
        out = np.zeros((n, K))
        for i in range(n):
            for j in range(K):
                pos, _ = sim.execute(q[i], v[i], c[i], raw[i, j])
                out[i, j] = np.linalg.norm(pos - gp[i])
        return out

    return score


def main(task_name: str, spec_str: str, seed: int, num_eval: int, eval_seed: int, data: str, eval_data: str,
         goal_offset: int | None, budget: int | None, device: str, cem_batch: int, controller: str = "cem",
         rho: float = 0.6, k: int = 16, env_chunk: int = 0, cost: str = "mse", pairs: str = "all",
         skip_existing: bool = False, select: str = "cost", policy_file: str = "policy.pt", receding: int = 5,
         camera_drop: float = 0.0, fill: str = "blank", std_scale: float = 1.0, pool: list | None = None,
         pool_mode: str = "select", critic_file: str = "critic.pt", exec_steps: int = 0) -> dict | None:
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from sklearn import preprocessing
    from stable_worldmodel.planning import GoalMSE, ShootingCostEvaluator
    from stable_worldmodel.planning.solver.cem import CEMSolver
    from torchvision.transforms import v2 as T

    cfg = TrainConfig()
    fresh_state = False                                     # sensor policies read the true env state
    task = get_task(task_name, cfg.img_size)
    apply_render_profile(task)
    goal_offset = task.goal_offset if goal_offset is None else goal_offset
    budget = task.eval_budget if budget is None else budget
    sel_tag = select if select != "verifier" else "verifier-" + Path(critic_file).stem.removeprefix("verifier_")
    out_path = result_path(task_name, spec_str, seed, controller, k, rho, cost, pairs, goal_offset, budget, eval_seed,
                           sel_tag, policy_file, receding, camera_drop, fill, num_eval, std_scale, pool, pool_mode,
                           exec_steps)
    if controller in ("gcidm", "gcidm_official") and exec_steps != 1:
        raise ValueError("GC-IDM controllers predict one action: use --exec-steps 1")
    if controller == "s1short":                               # execute exactly the chunk the policy was trained for
        c = torch.load(HOME / "checkpoints" / f"{task_name}/{Spec.parse(spec_str).name()}/seed{seed}" / policy_file,
                       map_location="cpu")["chunk_steps"]
        if exec_steps != c:
            raise ValueError(f"this policy outputs {c}-step chunks: use --exec-steps {c}")
    if skip_existing and out_path.exists():
        print(json.dumps({"skipped_existing": str(out_path.name)}), flush=True)
        return None
    chunk = env_chunk if env_chunk > 0 else num_eval
    eval_ds = swm.data.HDF5Dataset(eval_data, cache_dir=str(HOME), keys_to_cache=list(task.process_cols))
    train_ds = swm.data.HDF5Dataset(data, cache_dir=str(HOME), keys_to_cache=list(task.process_cols))

    if spec_str == "random":
        policy, run = swm.policy.RandomPolicy(), f"{task_name}/random"
    else:
        spec = Spec.parse(spec_str)
        run = f"{task_name}/{spec.name()}/seed{seed}"
        process = {}
        for col in task.process_cols:                            # same statistics as training
            d = np.asarray(train_ds.get_col_data(col))
            sc = preprocessing.StandardScaler().fit(d[~np.isnan(d).any(axis=1)])
            process[col] = sc
            if col != "action":
                process[f"goal_{col}"] = sc
        img_t = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True),
                           T.Normalize(**spt.data.dataset_stats.ImageNet), T.Resize(size=cfg.img_size)])
        state_dim = 0
        for c in state_columns(task, spec):                     # must match training's merged 'state'
            col = np.asarray(train_ds.get_col_data(c))
            state_dim += col.reshape(col.shape[0], -1).shape[1]
        model = build_model(spec, task.action_dim, state_dim, cfg)
        model.load_state_dict(torch.load(HOME / "checkpoints" / run / "weights.pt", map_location="cpu"))
        lewm = model.lewm.to(device).eval().requires_grad_(False)
        lewm.interpolate_pos_encoding = True
        distance, tau = None, None
        if cost == "reach":
            from hybrid_wm.control.reachability import ReachabilityCritic, ReachObjective
            rk = torch.load(HOME / "checkpoints" / run / "reach.pt", map_location="cpu")
            critic = ReachabilityCritic(rk["latent_dim"], max_steps=rk["max_steps"])
            critic.load_state_dict(rk["state_dict"])
            critic = critic.to(device).eval().requires_grad_(False)
            objective, distance, tau = ReachObjective(critic), critic, 1.0   # tau: within one block
        elif cost == "probe":                                 # task-aware: object readout + latent term
            from objectives import load_or_fit
            objective = load_or_fit(lewm, task, cfg, HOME, HOME / "checkpoints" / run).to(device)
            distance, tau = objective.distance, 1.0                        # tau: within the success tolerance
        else:
            objective = GoalMSE()
        solver = CEMSolver(ShootingCostEvaluator(lewm, objective), batch_size=cem_batch, num_samples=300,
                           var_scale=1.0, n_steps=30, topk=30, device=device, seed=eval_seed)
        if controller == "gcidm":                             # step-wise inverse dynamics (gcidm.py)
            from gcidm import LatentStepSolver, gcidm_act, gcidm_from_checkpoint
            pk = torch.load(HOME / "checkpoints" / run / policy_file, map_location="cpu")
            solver = LatentStepSolver(solver, lewm, gcidm_act(gcidm_from_checkpoint(pk).to(device), budget))
        elif controller == "gcidm_official":                  # the authors' GC-IDM code (official_gcidm.py)
            from gcidm import LatentStepSolver
            from official_gcidm import official_act
            solver = LatentStepSolver(solver, lewm, official_act(HOME / "checkpoints" / run / policy_file,
                                                                 budget, process["action"], device))
        elif controller == "s1short":                         # System 1 trained for short chunks (short_chunk.py)
            from gcidm import LatentStepSolver
            from short_chunk import short_chunk_act
            solver = LatentStepSolver(solver, lewm, short_chunk_act(HOME / "checkpoints" / run / policy_file,
                                                                    device))
        elif controller != "cem":
            from system12 import GatedSolver

            from hybrid_wm.control.latent_policy import policy_from_checkpoint
            pk = torch.load(HOME / "checkpoints" / run / policy_file, map_location="cpu")
            if pk.get("policy_type") == "retrieval":           # contrastive retrieval over an expert menu
                from contrastive_verifier import retrieval_from_checkpoint
                pol = retrieval_from_checkpoint(pk).to(device).eval()
            else:
                pol = policy_from_checkpoint(pk).to(device).eval()
            if pool:                                          # extra independently trained policies
                from hybrid_wm.control.latent_policy import PolicyPool
                members = [pol]
                for f in pool:
                    pm = torch.load(HOME / "checkpoints" / run / f, map_location="cpu")
                    members.append(policy_from_checkpoint(pm).to(device).eval())
                pol = PolicyPool(members, pool_mode)
                if k != pol.k:
                    raise ValueError(f"--k must be {pol.k} for a {len(members)}-policy pool in mode {pool_mode!r}")
            extra_fn, subgoal = None, None
            if "subgoal_state_dict" in pk:                    # two-level System 1
                from hybrid_wm.control.latent_policy import SubgoalNet
                subgoal = SubgoalNet(pk["latent_dim"])
                subgoal.load_state_dict(pk["subgoal_state_dict"])
                subgoal = subgoal.to(device).eval()
            if pk.get("extra_dim", 0):   # simulated sensors, as in training
                from sensors import SensorEncoder, SensorSpec, readings_from_info
                senc = SensorEncoder(SensorSpec(**pk["sensors"]), pk["sensor_stats"],
                                     generator=torch.Generator().manual_seed(eval_seed))
                extra_fn = lambda info: senc(*readings_from_info(info, pk["state_cols"], pk["object_cols"]))  # noqa: E731
                fresh_state = True
            scorer = None
            if select == "critic":                            # outcome critic trained on real outcomes
                from critic import load_critic
                crit = load_critic(HOME / "checkpoints" / run / critic_file)
                scorer = lambda info, z, zg, cand: crit.score(z.float().cpu(), zg.float().cpu(), cand.float().cpu())  # noqa: E731
            elif select == "verifier":                        # contrastive verifier (CLM / CoVer-VLA style)
                from contrastive_verifier import load_verifier
                ver = load_verifier(HOME / "checkpoints" / run / critic_file)
                scorer = lambda info, z, zg, cand: ver.score(z.float().cpu(), zg.float().cpu(), cand.float().cpu())  # noqa: E731
            elif select == "oracle":                          # ceiling: true simulated outcome (privileged)
                scorer = true_outcome_scorer(task, process["action"])
            solver = GatedSolver(solver, lewm, pol, k=k, rho=rho, tau_abs=pk["tau_abs"] if tau is None else tau,
                               mode=controller, seed=eval_seed, distance=distance,
                               select="scorer" if scorer is not None else select, scorer=scorer,
                               extra_fn=extra_fn, subgoal=subgoal, camera_drop=camera_drop, fill=fill,
                               receding=receding, subgoal_gate=pk.get("subgoal_gate"), std_scale=std_scale)
        policy_cls, extra_kw = swm.policy.WorldModelPolicy, {}
        if exec_steps:                                       # act for exec_steps steps, then look again
            from gcidm import step_policy
            policy_cls, extra_kw = step_policy(swm.policy.WorldModelPolicy), {"exec_steps": exec_steps}
        policy = policy_cls(
            solver=solver, config=swm.PlanConfig(horizon=5, receding_horizon=receding, action_block=cfg.frameskip),
            process=process, transform={"pixels": img_t, "goal": img_t}, **extra_kw)

    if pairs == "pickup":                                  # H-JEPA's grasp-centred Cube windows
        from tasks import pickup_candidates
        ce, cs = pickup_candidates(eval_ds, goal_offset)
        keep = ~trivial_mask(task, eval_ds, ce, cs, goal_offset)
        ce, cs = ce[keep], cs[keep]
        if len(ce) < num_eval:
            raise ValueError(f"only {len(ce)} pickup windows for {num_eval} evaluations")
        order = np.sort(np.random.default_rng(eval_seed).permutation(len(ce))[:num_eval])
        episodes, starts = ce[order].tolist(), cs[order].tolist()
    else:
        exclude = None
        if pairs == "nontrivial":
            exclude = lambda e, st: trivial_mask(task, eval_ds, e, st, goal_offset)  # noqa: E731
        episodes, starts = pick_pairs(np.asarray(eval_ds.lengths), goal_offset, num_eval, eval_seed, exclude)
    triv = trivial_mask(task, eval_ds, episodes, starts, goal_offset)
    # Episodes are independent, so evaluating the same pairs in sequential env
    # batches changes only peak memory (Cube: ~0.45 GB per MuJoCo env).
    t0 = time.time()
    successes, decision_s = [], []
    for lo in range(0, num_eval, chunk):
        ep_c, st_c = episodes[lo:lo + chunk], starts[lo:lo + chunk]
        world = swm.World(task.env_id, num_envs=len(ep_c), image_shape=(cfg.img_size, cfg.img_size),
                          max_episode_steps=2 * budget, **task.eval_kwargs)
        world.set_policy(policy)
        solver_ = getattr(policy, "solver", None)
        if hasattr(solver_, "reset_episodes"):                # per-episode memory never crosses batches
            solver_.reset_episodes()
        _time_decisions(policy, decision_s)
        if fresh_state:
            _refresh_state_infos(world)
        m = world.evaluate(dataset=eval_ds, episodes_idx=ep_c, start_steps=st_c, goal_offset=goal_offset,
                           eval_budget=budget, callables=task.eval_callables)
        successes.extend(np.asarray(m["episode_successes"]).astype(int).tolist())
        world.close()
        del world
    metrics = {"success_rate": 100.0 * sum(successes) / len(successes), "episode_successes": successes}
    stats = getattr(getattr(policy, "solver", None), "stats", None)
    result = {"run": run, "task": task_name, "spec": spec_str, "seed": seed, "eval_seed": eval_seed,
              "controller": controller, "rho": rho if controller != "cem" else None,
              "k": k if controller != "cem" else None, "cost": cost,
              "select": select if controller != "cem" else None,
              "policy_file": policy_file if controller != "cem" else None,
              "receding": receding, "exec_steps": exec_steps or None, "camera_drop": camera_drop, "std_scale": std_scale,
              "pool": [policy_file, *pool] if pool else None, "pool_mode": pool_mode if pool else None, "fill": fill if camera_drop else None,
              "system12_stats": stats,
              "selector_file": critic_file if select in ("critic", "verifier") and controller != "cem" else None,
              "num_eval": num_eval, "env_chunk": chunk, "goal_offset": goal_offset, "eval_budget": budget,
              "success_rate": float(metrics["success_rate"]), "pairs": pairs,
              "trivial": triv.astype(int).tolist(),
              "nontrivial_success_rate": (100.0 * float(np.asarray(metrics["episode_successes"])[~triv].mean())
                                          if (~triv).any() else None),
              "successes": np.asarray(metrics["episode_successes"]).astype(int).tolist(),
              "episodes_idx": [int(e) for e in episodes], "start_steps": [int(s_) for s_ in starts],
              "decision_seconds": ({"median": float(np.median(decision_s)), "p10": float(np.percentile(decision_s, 10)),
                                    "p90": float(np.percentile(decision_s, 90)), "calls": len(decision_s),
                                    "envs_per_call": chunk, "device": device} if decision_s else None),
              "eval_minutes": round((time.time() - t0) / 60, 1)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")   # atomic: --skip-existing never sees a partial result
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(out_path)
    print(json.dumps({key: v for key, v in result.items() if key not in ("successes", "trivial")}), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["tworoom", "cube"])
    ap.add_argument("--spec", required=True, help="model spec (see models.py) or 'random'")
    ap.add_argument("--seed", type=int, default=0, help="training seed of the checkpoint")
    ap.add_argument("--num-eval", type=int, default=50)
    ap.add_argument("--eval-seed", type=int, default=42)
    ap.add_argument("--data", default=None)
    ap.add_argument("--eval-data", default=None)
    ap.add_argument("--goal-offset", type=int, default=None)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--cem-batch", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--controller", choices=["cem", "s1s2", "s1", "gcidm", "gcidm_official", "s1short"],
                    default="cem",
                    help="cem = authors' planner; s1s2 = verify-then-act System 1/2; s1 = System 1 only; "
                         "gcidm = step-wise GC-IDM-style controller (gcidm.py); gcidm_official = the authors' GC-IDM "
                         "(official_gcidm.py); s1short = System 1 trained for short chunks (short_chunk.py)")
    ap.add_argument("--rho", type=float, default=0.6)
    ap.add_argument("--k", type=int, default=16, help="System 1 candidates verified per decision")
    ap.add_argument("--cost", choices=["mse", "reach", "probe"], default="mse",
                    help="mse = authors' GoalMSE; reach = learned reachability critic (train_reach.py); "
                         "probe = object-aware cost (objectives.py)")
    ap.add_argument("--pairs", choices=["all", "nontrivial", "pickup"], default="all",
                    help="all = authors' protocol; nontrivial = only pairs not already solved at the start; "
                         "pickup = H-JEPA's grasp-centred windows (use --goal-offset 20 --budget 30)")
    ap.add_argument("--select", choices=["cost", "random", "critic", "oracle", "verifier"], default="cost",
                    help="System 1 candidate choice: lowest world-model cost, a random sample (control), the outcome "
                         "critic (critic.py), or the true simulated outcome (privileged ceiling)")
    ap.add_argument("--critic-file", default="critic.pt",
                    help="outcome critic (--select critic) or contrastive verifier (--select verifier) file")
    ap.add_argument("--receding", type=int, default=5, help="blocks executed per decision (authors: 5)")
    ap.add_argument("--camera-drop", type=float, default=0.0, help="System 1: P(camera frame missing)")
    ap.add_argument("--fill", choices=["blank", "hold", "predict"], default="blank")
    ap.add_argument("--policy-file", default="policy.pt", help="System 1 policy file in the checkpoint dir")
    ap.add_argument("--skip-existing", action="store_true", help="do nothing if this result already exists")
    ap.add_argument("--env-chunk", type=int, default=-1,
                    help="envs simulated at once (default: 5 for cube, all for others)")
    ap.add_argument("--std-scale", type=float, default=1.0, help="System 1: spread of candidates around the mean")
    ap.add_argument("--pool", nargs="*", default=None,
                    help="System 1: extra policy files; with --policy-file they form a pool of proposers")
    ap.add_argument("--pool-mode", choices=["select", "mean"], default="select",
                    help="pool: world model selects among members' means, or average the means")
    ap.add_argument("--exec-steps", type=int, default=0,
                    help="steps executed from each plan before deciding again from a new frame (0 = whole receding plan)")
    a = ap.parse_args()
    if a.env_chunk < 0:
        a.env_chunk = 5 if a.task == "cube" else 0
    main(a.task, a.spec, a.seed, a.num_eval, a.eval_seed, a.data or a.task, a.eval_data or f"{a.task}_eval",
         a.goal_offset, a.budget, a.device, a.cem_batch, a.controller, a.rho, a.k, a.env_chunk, a.cost, a.pairs,
         a.skip_existing, a.select, a.policy_file, a.receding, a.camera_drop, a.fill, a.std_scale, a.pool,
         a.pool_mode, a.critic_file, a.exec_steps)
