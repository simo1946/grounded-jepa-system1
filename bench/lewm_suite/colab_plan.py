"""GPU study plan (Colab or any CUDA machine) and a resumable parallel job runner.

The plan covers every comparison still open after the CPU study
(docs/assessment.md) plus the untested pairings of its best parts:

  configs (world model spec)                                   seeds (tier)
  C1 lewm                          LeWM baseline                0 (1) | 1, 2 (2)
  C2 lewm+idm+sa                   proprio grounding            0 (1) | 1, 2 (2)
  C3 lewm+idm+sa+obj               + object-aware grounding     0 (1) | 1, 2 (2)
  C4 hybrid+idm+sa                 our hybrid predictor         0 (1) | 1, 2 (2)
  C5 hybrid+idm+sa+obj             hybrid x object grounding    0 (1) | 1, 2 (2)   (new pair)
  C6 lewm+idm+sa+obj+arl           + action-resolution loss     0 (1) | 1, 2 (2)   (new)
  C7 hybrid+idm+sa+obj+arl         all best parts               0 (1) | 1, 2 (2)   (new)
  C8 lewm+idm                      IDM-only ablation            0 (3)
  C9 lewm+idm+sa+obj+enc=mnv4s     fast encoder                 0 (3)

  per model: latent caches -> System 1 policy (cached latents) -> two-level
  System 1 (distance-gated subgoal network) -> evaluations:
    s1_nt   System 1, 100 non-trivial pairs, goal +25          (all)
    s1f_*   System 1 as a flow-matching policy (same data and steps), non-trivial and grasp windows
    s1_pk   System 1, H-JEPA grasp-centred windows              (all)
    s1_50   System 1, non-trivial, goal +50, budget 100          (all)
    hier_*  two-level System 1 on the same three protocols       (all)
    sel16   best-of-16 System 1 samples by world-model cost      (all; tests ARL)
    sel16n  best-of-16 with candidates kept near the mean (std x 0.25; exploratory, see prereg_verification.md)
    gate    verify-then-act, CEM fallback                        (all)
    cem     CEM planner                                          (C1, C3, C6, C7)
    cam*    camera dropout 50/80% with fill hold/predict/blank   (C3, C5, C6, C7)
    diag    latent probes + action identifiability                (seed 0)

Jobs skip when their outputs exist, so the notebook can be re-run after a
disconnect. Training uses the GPU slots; everything else runs in parallel
worker slots (MuJoCo rendering is CPU-bound).

    python colab_plan.py --profile standard --tiers 1 --workers 4 --run
    python colab_plan.py --profile smoke --tiers 1 --configs C1 C7 --workers 1 --run
    python colab_plan.py --profile standard --tiers 1 2 3 --list
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from study_configs import CEM_SET, CONFIGS, MEMORY_SET, PROFILES, TIERS  # noqa: F401  (re-exported)

HERE = Path(__file__).resolve().parent

STALE_TOLERANCE_S = 3600


@dataclass
class Job:
    name: str
    cmd: list[str]
    outputs: list[Path]
    deps: list[str] = field(default_factory=list)
    gpu: bool = False
    backup: Path | None = None      # persistent copy of outputs[0] (restored instead of re-running)
    refs: list[Path] = field(default_factory=list)   # inputs the outputs were built from (model weights, policy)

    def stale(self) -> bool:
        """An output is stale when an input it was built from is missing, or was rewritten well after it
        (a model retrained after its weights were lost: its old policies and results no longer match).
        The tolerance absorbs cloud-storage timestamp jitter; a real retrain is hours or a session later."""
        if not self.refs:
            return False
        if not all(r.exists() for r in self.refs):
            return True
        newest_ref = max(r.stat().st_mtime for r in self.refs)
        return any(o.exists() and o.stat().st_mtime < newest_ref - STALE_TOLERANCE_S for o in self.outputs)

    def done(self) -> bool:
        return bool(self.outputs) and all(p.exists() for p in self.outputs) and not self.stale()

    def set_aside(self) -> list[Path]:
        """Move existing (stale) outputs into a 'stale' folder beside them, so a job that skips existing
        outputs re-runs; nothing is deleted."""
        moved = []
        for o in self.outputs:
            if o.exists():
                dst = o.parent / "stale" / f"{time.strftime('%Y%m%d-%H%M%S')}__{o.name}"
                dst.parent.mkdir(parents=True, exist_ok=True)
                o.replace(dst)
                moved.append(dst)
        return moved


def build(profile: str, tiers: list[int], only: list[str] | None, home: Path, workers: int,
          env_chunk: int | None = None, backup_dir: Path | None = None, task: str = "cube",
          extra: tuple = ()) -> list[Job]:
    os.environ["STABLEWM_HOME"] = str(home)
    from evaluate import result_path
    from latent_cache import cache_path
    from models import Spec
    from tasks import get_task

    T = get_task(task, int(os.environ.get("HWM_IMG_SIZE", "64")))
    ev_name = f"{task}_eval"

    P = dict(PROFILES[profile])
    if env_chunk:
        P["env_chunk"] = env_chunk
    py = sys.executable
    jobs: list[Job] = []
    ds = lambda name: home / "datasets" / f"{name}.h5"  # noqa: E731

    # ---- data: our expert collector in parallel parts + merge (seeds disjoint between train and eval)
    dep_of = {}
    for name, n, seed0 in (((task, P["n_train"], 3072), (ev_name, P["n_eval"], 777_000))
                           if T.data_source == "collect" else ()):
        # small parts, so a disconnect loses at most one part per worker (with --backup-dir)
        parts = max(1, min(workers, n), math.ceil(n / P.get("part_size", 25)))
        per = [n // parts + (1 if i < n % parts else 0) for i in range(parts)]
        pnames = [f"{name}_p{i}" for i in range(parts)]
        for i, (pn, k) in enumerate(zip(pnames, per, strict=True)):
            jobs.append(Job(f"collect:{pn}", [py, "collect.py", "--task", task, "--name", pn, "--episodes", str(k),
                                              "--seed", str(seed0 + i), "--num-envs", "1",
                                              "--img", os.environ.get("HWM_IMG_SIZE", "64")],
                            [ds(pn)], backup=(backup_dir / f"{pn}.h5") if backup_dir else None))
        jobs.append(Job(f"merge:{name}", [py, "merge_h5.py", "--out", name, "--parts", *pnames, "--delete-parts"],
                        [ds(name)], deps=[f"collect:{pn}" for pn in pnames]))
        dep_of[name] = f"merge:{name}"
    # merged parts are deleted, so a collect job counts as done once its merge is
    for j in jobs:
        if j.name.startswith("collect:"):
            base = j.name.split(":")[1].rsplit("_p", 1)[0]
            j.outputs = [ds(base)] if ds(base).exists() else j.outputs

    tiers_def = P.get("tiers", TIERS)
    cem_set, memory_set = P.get("cem_set", CEM_SET), P.get("memory_set", MEMORY_SET)
    wanted = P.get("evals")                                  # None = every evaluation
    pairs = [(c, s) for t in tiers for (c, s) in tiers_def.get(t, []) if not only or c in only]
    if not T.object_cols:                                    # no object state: object-grounding configs do not apply
        pairs = [(c, s) for c, s in pairs if "obj" not in CONFIGS[c]]
    for cfg_id, seed in pairs:
        spec_str = CONFIGS[cfg_id]
        spec = Spec.parse(spec_str)
        tag = f"{cfg_id}s{seed}"
        ck = home / "checkpoints" / task / spec.name() / f"seed{seed}"
        jobs.append(Job(f"train:{tag}", [py, "train.py", "--task", task, "--spec", spec_str, "--seed", str(seed),
                                         "--steps", str(P["train_steps"]), "--batch-size", str(P["batch"]),
                                         "--lr", str(P["lr"]), "--num-workers", "4"],
                        [ck / "weights.pt"], deps=[dep_of[task]], gpu=True))
        for data in (task, ev_name):
            jobs.append(Job(f"cache:{tag}:{data}", [py, "latent_cache.py", "--spec", spec_str, "--seed", str(seed),
                                                    "--data", data, "--workers", "2"],
                            [cache_path(spec.name(), seed, data)],
                            deps=[f"train:{tag}", dep_of[data]], refs=[ck / "weights.pt"]))
        jobs.append(Job(f"policy:{tag}", [py, "cached_policy.py", "--task", task, "--spec", spec_str, "--seed", str(seed),
                                          "--out", "policy_s1.pt", "--steps", str(P["policy_steps"]),
                                          "--batch", "256"],
                        [ck / "policy_s1.pt"], deps=[f"cache:{tag}:{task}"], refs=[ck / "weights.pt"]))
        jobs.append(Job(f"policyf:{tag}", [py, "cached_policy.py", "--task", task, "--spec", spec_str, "--seed", str(seed),
                                           "--out", "policy_flow.pt", "--steps", str(P["policy_steps"]),
                                           "--batch", "256", "--policy-type", "flow"],
                        [ck / "policy_flow.pt"], deps=[f"cache:{tag}:{task}"], refs=[ck / "weights.pt"]))
        jobs.append(Job(f"hier:{tag}", [py, "cached_subgoal.py", "--task", task, "--spec", spec_str, "--seed", str(seed),
                                        "--base", "policy_s1.pt", "--out", "policy_hier_gated.pt", "--gate", "auto",
                                        "--steps", str(P["subgoal_steps"])],
                        [ck / "policy_hier_gated.pt"], deps=[f"policy:{tag}", f"cache:{tag}:{ev_name}"],
                        refs=[ck / "weights.pt", ck / "policy_s1.pt"]))

        n = P["num_eval"]

        def ev(label, deps, controller="s1", k=1, cost="mse", pairs_="nontrivial", off=25, bud=50,
               policy="policy_s1.pt", receding=5, cam=0.0, fill="blank", std=1.0):
            if wanted is not None and label not in wanted:
                return
            if task != "cube" and (pairs_ == "pickup" or cam):   # grasp windows and camera tests: Cube only
                return
            cmd = [py, "evaluate.py", "--task", task, "--spec", spec_str, "--seed", str(seed),
                   "--controller", controller, "--cost", cost, "--pairs", pairs_, "--goal-offset", str(off),
                   "--budget", str(bud), "--num-eval", str(n), "--env-chunk", str(P["env_chunk"]),
                   "--skip-existing"]
            if controller != "cem":
                cmd += ["--k", str(k), "--policy-file", policy, "--receding", str(receding)]
                if cam:
                    cmd += ["--camera-drop", str(cam), "--fill", fill]
                if std != 1.0:
                    cmd += ["--std-scale", str(std)]
            out = result_path(task, spec_str, seed, controller, k, 0.6, cost, pairs_, off, bud, 42, "cost",
                              policy if controller != "cem" else "policy.pt", receding if controller != "cem" else 5,
                              cam, fill, n, std)
            refs = [ck / "weights.pt"] + ([ck / policy] if controller != "cem" else [])
            jobs.append(Job(f"eval:{tag}:{label}", cmd, [out], deps=deps + [f"train:{tag}", dep_of[ev_name]],
                            refs=refs))

        s1, hier = [f"policy:{tag}"], [f"hier:{tag}"]
        ev("s1_nt", s1)
        ev("s1_pk", s1, pairs_="pickup", off=20, bud=30)
        ev("s1_50", s1, off=50, bud=100)
        ev("hier_nt", hier, policy="policy_hier_gated.pt")
        ev("hier_pk", hier, policy="policy_hier_gated.pt", pairs_="pickup", off=20, bud=30)
        ev("hier_50", hier, policy="policy_hier_gated.pt", off=50, bud=100)
        ev("s1f_nt", [f"policyf:{tag}"], policy="policy_flow.pt")
        ev("s1f_pk", [f"policyf:{tag}"], policy="policy_flow.pt", pairs_="pickup", off=20, bud=30)
        ev("sel16", s1, k=16)
        ev("sel16n", s1, k=16, std=0.25)
        ev("gate", s1, controller="s1s2", k=16)
        if cfg_id in cem_set or "cem_all" in extra:          # cem_all: follow-up, CEM on every configuration
            ev("cem", [f"train:{tag}"], controller="cem")
        if task != "cube" or "appendix" in extra:           # authors' protocol (every pair): comparable to published numbers
            ev("s1_all", s1, pairs_="all")
            if cfg_id in cem_set or "cem_all" in extra:
                ev("cem_all", [f"train:{tag}"], controller="cem", pairs_="all")
        if cfg_id in memory_set and task == "cube":
            ev("rec1", s1, receding=1)
            for cam, fill in ((0.5, "blank"), (0.5, "hold"), (0.5, "predict"), (0.8, "hold"), (0.8, "predict")):
                ev(f"cam{cam:g}-{fill}", s1, receding=1, cam=cam, fill=fill)
        if seed == 0 and task == "cube":                     # diagnostics read Cube-specific columns
            res = home / "results"
            jobs.append(Job(f"diag:{tag}:probe", [py, "probe_diag.py", "--spec", spec_str, "--seed", "0",
                                                  "--test", str(P["diag_clips"])],
                            [res / f"probe_diag__{spec.name()}__seed0.json"], deps=[f"train:{tag}", "merge:cube_eval"],
                            refs=[ck / "weights.pt"]))
            jobs.append(Job(f"diag:{tag}:rollout", [py, "rollout_diag.py", "--spec", spec_str, "--seed", "0",
                                                    "--clips", str(P["diag_clips"])],
                            [res / f"rollout_diag__{spec.name()}__seed0.json"], deps=[f"train:{tag}", "merge:cube_eval"],
                            refs=[ck / "weights.pt"]))
    names = {j.name for j in jobs}
    for j in jobs:
        missing = [d for d in j.deps if d not in names]
        if missing:
            raise RuntimeError(f"{j.name}: unknown dependencies {missing}")
    return jobs


def _backup(src: Path, dst: Path) -> None:
    """Copy src to dst atomically (a partial copy never looks complete)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)


def check_space(home: Path, stop_gb: float = 1.0, warn_gb: float = 1.5) -> bool:
    """Free space where models and results are written (on Colab: Google Drive via symlinks). Informational
    only: the run is never blocked (a lost file is caught afterwards: every finished training job is checked
    to be readable, and anything built from a lost or retrained model is redone)."""
    ok = True
    seen = set()
    for d in (home, home / "checkpoints", home / "results"):
        if not d.exists():
            continue
        real = d.resolve()
        usage = shutil.disk_usage(real)
        key = (usage.total, usage.free)
        if key in seen:
            continue
        seen.add(key)
        free_gb = usage.free / 1e9
        if free_gb < stop_gb:
            print(f"WARNING: only {free_gb:.1f} GB free at {real}. The run continues, but a full Google Drive "
                  f"can lose files; freeing space (empty the Drive bin) is safer.", flush=True)
            ok = False
        elif free_gb < warn_gb:
            print(f"WARNING: {free_gb:.1f} GB free at {real}; a tier adds about 1 GB of models and results.",
                  flush=True)
    return ok


def verify_output(job: Job) -> bool:
    """A finished training job must leave a readable weights file; otherwise it is marked failed (and its
    dependents wait) instead of being trusted. Other jobs: existence is enough."""
    if not job.name.startswith("train:"):
        return True
    try:
        import torch
        torch.load(job.outputs[0], map_location="cpu", weights_only=True)
        return True
    except Exception as e:                                         # noqa: BLE001
        print(f"[{time.strftime('%H:%M:%S')}] {job.name}: weights file unreadable ({type(e).__name__})", flush=True)
        return False


def log_path(log_dir: Path, name: str) -> Path:
    return log_dir / (name.replace(":", "_") + ".log")


def run(jobs: list[Job], workers: int, gpu_slots: int, log_dir: Path, poll: float = 5.0) -> dict:
    log_dir.mkdir(parents=True, exist_ok=True)
    by = {j.name: j for j in jobs}
    state = {j.name: ("done" if j.done() else "pending") for j in jobs}
    running: dict[str, tuple[subprocess.Popen, float, object]] = {}
    t_start, last_print = time.time(), 0.0

    def ready(j):
        return state[j.name] == "pending" and all(state[d] == "done" for d in j.deps)

    def blocked(j):
        return any(state[d] in ("failed", "blocked") for d in j.deps)

    try:
        while True:
            for j in jobs:                                       # propagate failures
                if state[j.name] == "pending" and blocked(j):
                    state[j.name] = "blocked"
            for name, (proc, t0, fh) in list(running.items()):
                rc = proc.poll()
                if rc is None:
                    continue
                fh.close()
                ok = rc == 0 and by[name].done() and verify_output(by[name])
                if ok and by[name].backup is not None:            # persist before anything can delete it
                    _backup(by[name].outputs[0], by[name].backup)
                state[name] = "done" if ok else "failed"
                msg = f"[{time.strftime('%H:%M:%S')}] {'done  ' if ok else 'FAILED'} {name} ({(time.time() - t0) / 60:.1f} min)"
                if not ok:
                    msg += f" rc={rc}, log: {log_path(log_dir, name)}"
                print(msg, flush=True)
                del running[name]
            gpu_busy = sum(by[n].gpu for n in running)
            for j in jobs:
                if len(running) >= workers:
                    break
                if not ready(j) or j.name in running:
                    continue
                if j.gpu and gpu_busy >= gpu_slots:
                    continue
                if j.backup is not None and j.backup.exists():   # finished in an earlier session
                    j.outputs[0].parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(j.backup, j.outputs[0])
                    state[j.name] = "done"
                    print(f"[{time.strftime('%H:%M:%S')}] restored {j.name} from {j.backup}", flush=True)
                    continue
                for moved in j.set_aside():                     # stale outputs: kept, but out of the way
                    print(f"[{time.strftime('%H:%M:%S')}] stale  {j.name}: moved old output to {moved}", flush=True)
                fh = open(log_path(log_dir, j.name), "w")
                proc = subprocess.Popen(j.cmd, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT)
                running[j.name] = (proc, time.time(), fh)
                state[j.name] = "running"
                gpu_busy += j.gpu
                print(f"[{time.strftime('%H:%M:%S')}] start  {j.name}", flush=True)
            if not running and not any(ready(j) for j in jobs):
                break
            if time.time() - last_print > 300:
                counts = {s: list(state.values()).count(s) for s in ("done", "running", "pending", "failed", "blocked")}
                print(f"[{time.strftime('%H:%M:%S')}] progress {counts}, elapsed {(time.time() - t_start) / 60:.0f} min",
                      flush=True)
                last_print = time.time()
            time.sleep(poll)
    except KeyboardInterrupt:
        for proc, _, fh in running.values():
            proc.terminate()
            fh.close()
        print("interrupted: running jobs terminated; re-run to resume", flush=True)
        raise
    counts = {s: list(state.values()).count(s) for s in ("done", "pending", "failed", "blocked")}
    print(json.dumps({"finished": counts, "minutes": round((time.time() - t_start) / 60, 1),
                      "failed": [n for n, s in state.items() if s == "failed"]}), flush=True)
    return state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="standard", choices=sorted(PROFILES))
    ap.add_argument("--task", default="cube", choices=["cube", "tworoom"],
                    help="environment: cube (main study) or tworoom (navigation)")
    ap.add_argument("--tiers", type=int, nargs="+", default=[1])
    ap.add_argument("--configs", nargs="*", default=None, help="subset, e.g. C1 C7")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--env-chunk", type=int, default=None, help="envs simulated at once per evaluation "
                    "(~0.45 GB each; default from the profile)")
    ap.add_argument("--data-only", action="store_true", help="only collect and merge the datasets")
    ap.add_argument("--gpu-slots", type=int, default=1, help="concurrent training jobs on the GPU")
    ap.add_argument("--home", default=os.environ.get("STABLEWM_HOME", str(HERE / "swm_home")))
    ap.add_argument("--backup-dir", default=None, help="persistent folder (e.g. on Drive) for finished collection "
                    "parts; a re-run restores them instead of collecting again")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--extra", nargs="*", default=[], choices=["appendix", "cem_all"],
                    help="appendix = Cube on the authors' protocol (every pair: System 1 for all models, CEM for the "
                         "CEM configs); cem_all = follow-up: CEM on every configuration, not only the profile's CEM set "
                         "(with appendix also on every pair)")
    ap.add_argument("--only-extra", action="store_true", help="list/run only the extra study and what it needs")
    a = ap.parse_args()
    home = Path(a.home)
    jobs = build(a.profile, a.tiers, a.configs, home, a.workers, a.env_chunk,
                 Path(a.backup_dir) if a.backup_dir else None, task=a.task, extra=tuple(a.extra))
    if a.only_extra:                     # the extra jobs plus what they still need (finished jobs end the search,
        by = {j.name: j for j in jobs}   # so data that was deleted after training is not collected again)
        ends = (":s1_all", ":cem_all") if "appendix" in a.extra else ()
        ends += (":cem",) if "cem_all" in a.extra else ()
        keep, todo = set(), [j.name for j in jobs if ends and j.name.endswith(ends)]
        while todo:
            nme = todo.pop()
            if nme not in keep:
                keep.add(nme)
                if not by[nme].done():
                    todo += by[nme].deps
        jobs = [j for j in jobs if j.name in keep]
    if a.data_only:
        jobs = [j for j in jobs if j.name.startswith(("collect:", "merge:"))]
    if a.list or not a.run:
        for j in jobs:
            status = "done   " if j.done() else ("stale  " if j.outputs and all(o.exists() for o in j.outputs) else "pending")
            print(f"{status} {'GPU ' if j.gpu else '    '}{j.name}")
        print(f"{sum(j.done() for j in jobs)}/{len(jobs)} done")
    if a.run:
        check_space(home)                                   # warns only; never stops the run
        run(jobs, a.workers, a.gpu_slots, home / "logs" / "plan")


if __name__ == "__main__":
    main()
