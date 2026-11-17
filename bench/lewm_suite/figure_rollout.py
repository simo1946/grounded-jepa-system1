"""Camera frames of one System 1 rollout, for the task figure of the paper (illustration only).

Re-runs the CPU re-run's evaluation of System 1 (mean action, k = 1) on the object-grounded model C3, seed 0
(policy_s1.pt), on the same 50 non-trivial pairs, and records the 64-pixel frame the model sees after every step.
Nothing in the evaluation changes: the run only adds a recorder to the world's per-step callback and writes its
result file to the output folder instead of $STABLEWM_HOME/results.

Example pair: among the pairs System 1 solves, the one whose cube moves farthest between the start and the goal
frame. Writes rollout.npz (frames and simulator states of that pair until success, its start and goal frames and
states) and
rollout.json (which pair, outcome, and whether the 50 outcomes equal those of the re-run's result file).

    python figure_rollout.py --out ../../paper/figures/data --check ../../results/cpu_rerun/<C3 s1 k1 file>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import evaluate

SPEC, SEED, POLICY = "lewm+idm+sa+obj", 0, "policy_s1.pt"


def record(world_cls, store: list) -> None:
    """Wrap World._run so that every step appends (frames, terminated) for this world to store."""
    orig = world_cls._run

    def run(self, *a, on_step=None, **k):
        rec = {"frames": [], "done": [], "qpos": [], "qvel": []}
        store.append(rec)

        def step(world, mask):
            if on_step is not None:
                on_step(world, mask)
            px = np.asarray(world.infos["pixels"])
            rec["frames"].append((px[:, -1] if px.ndim == 5 else px).copy())
            for k in ("qpos", "qvel"):                         # simulator state, to render the figure at high resolution
                v = np.asarray(world.infos[k])
                rec[k].append((v[:, -1] if v.ndim == 3 else v).copy())
            rec["done"].append(np.asarray(world.terminateds, bool).copy())
        return orig(self, *a, on_step=step, **k)
    world_cls._run = run


def main(out: Path, check: Path | None) -> dict:
    import h5py
    import stable_worldmodel as swm

    from models import HOME
    out.mkdir(parents=True, exist_ok=True)
    store: list = []
    record(swm.World, store)
    evaluate.result_path = lambda *a, **k: out / "rollout_eval.json"   # keep $STABLEWM_HOME/results untouched
    res = evaluate.main("cube", SPEC, SEED, 50, 42, "cube", "cube_eval", None, None, "cpu", 10, controller="s1", k=1,
                        env_chunk=5, pairs="nontrivial", policy_file=POLICY)
    # one recording per batch of environments, in pair order
    frames, done, qpos, qvel = [], [], [], []
    for rec in store:
        frames.extend(np.stack(rec["frames"], 1))                             # [envs, steps, ...]
        done.extend(np.stack(rec["done"], 1))
        qpos.extend(np.stack(rec["qpos"], 1))
        qvel.extend(np.stack(rec["qvel"], 1))
    succ = np.asarray(res["successes"], bool)
    eps, starts, offset = res["episodes_idx"], res["start_steps"], res["goal_offset"]

    data = Path(HOME) / "datasets" / "cube_eval.h5"
    with h5py.File(data, "r") as f:
        off = f["ep_offset"][:]
        cube = f["privileged_block_0_pos"][:]
        effector = f["proprio_effector_pos"][:]
        pixels = f["pixels"]
        rows = [(int(off[e] + s), int(off[e] + s + offset)) for e, s in zip(eps, starts, strict=True)]
        moved = np.array([np.linalg.norm(cube[g] - cube[s]) for s, g in rows])
        pick = int(np.argmax(np.where(succ, moved, -1.0)))
        s_row, g_row = rows[pick]
        goal_px, start_px = pixels[g_row], pixels[s_row]
        state_rows = {k: f[k][[s_row, g_row]] for k in ("qpos", "qvel")}
        start_state = {"cube": cube[s_row].tolist(), "effector": effector[s_row].tolist()}
        goal_state = {"cube": cube[g_row].tolist(), "effector": effector[g_row].tolist()}
    first = int(np.argmax(done[pick])) if done[pick].any() else len(done[pick]) - 1
    roll = frames[pick][: first + 1]
    np.savez_compressed(out / "rollout.npz", frames=roll, goal=goal_px, start=start_px,
                        qpos=qpos[pick][: first + 1], qvel=qvel[pick][: first + 1],
                        start_qpos=state_rows["qpos"][0], start_qvel=state_rows["qvel"][0],
                        goal_qpos=state_rows["qpos"][1], goal_qvel=state_rows["qvel"][1])
    info = {"spec": SPEC, "config": "C3", "seed": SEED, "policy_file": POLICY, "controller": "s1", "k": 1,
            "pair_index": pick, "episode": int(eps[pick]), "start_step": int(starts[pick]),
            "goal_offset": int(offset), "success": bool(succ[pick]), "steps_to_success": first + 1,
            "cube_moved_m": float(moved[pick]), "solved": int(succ.sum()), "pairs": len(succ),
            "start_state": start_state, "goal_state": goal_state}
    if check is not None:
        ref = json.loads(check.read_text())
        info["same_outcomes_as"] = check.name
        info["same_outcomes"] = ref["successes"] == res["successes"] and ref["episodes_idx"] == res["episodes_idx"]
    (out / "rollout.json").write_text(json.dumps(info, indent=1))
    return info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", type=Path, default=None, help="re-run result file whose outcomes should match")
    a = ap.parse_args()
    print(json.dumps(main(a.out, a.check)))
