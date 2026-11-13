"""Task registry for the LeWM benchmark suite (authors' environments/configs).

Each task mirrors stable-worldmodel's collector script and le-wm's eval
config. Deviations (image size, episode counts, Cube render quality) are
explicit fields so every run records them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Task:
    name: str
    env_id: str
    action_dim: int
    episode_steps: int
    collect_kwargs: Dict
    eval_kwargs: Dict
    expert: Callable[[], object]
    process_cols: Tuple[str, ...]          # columns z-scored for the planner (authors: keys_to_cache)
    state_cols: Tuple[str, ...]            # robot proprioception for state alignment (no object state)
    eval_callables: List[Dict]
    goal_offset: int = 25                  # le-wm eval config
    eval_budget: int = 50                  # le-wm eval config
    render_profile: Optional[Tuple[int, int]] = None   # (shadowsize, offsamples); None = authors' default
    # A start/goal pair is "trivial" when the start state already satisfies the
    # env's success test for the goal (no action needed). (column, threshold)
    trivial_check: Optional[Tuple[str, float]] = None
    # privileged object state, used ONLY as a training-time alignment target
    # by specs with the 'obj' token (never an input at planning time)
    object_cols: Tuple[str, ...] = ()
    # optional custom trivial test: fn(start_rows_data, goal_rows_data) -> bool array, on the
    # trivial_fn_col column (overrides trivial_check)
    trivial_fn: Optional[Callable] = None
    trivial_fn_col: object = ""            # a column name, or a tuple of names (fn then gets lists)
    # where the training/evaluation data come from: "collect" (our expert collector)
    data_source: str = "collect"
    notes: str = ""


def _tworoom_expert():
    from stable_worldmodel.envs.two_room import ExpertPolicy
    return ExpertPolicy(action_noise=2.0, action_repeat_prob=0.05)


def _cube_expert():
    from stable_worldmodel.envs.ogbench import ExpertPolicy
    return ExpertPolicy()


def get_task(name: str, img: int) -> Task:
    if name == "tworoom":
        return Task(
            name="tworoom", env_id="swm/TwoRoom-v1", action_dim=2, episode_steps=100,
            collect_kwargs=dict(render_mode="rgb_array"),
            eval_kwargs=dict(render_mode="rgb_array"),
            expert=_tworoom_expert, process_cols=("action", "proprio"), state_cols=("proprio",),
            eval_callables=[{"method": "_set_state", "args": {"state": {"value": "proprio"}}},
                            {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_proprio"}}}],
            trivial_check=("proprio", 16.0),                    # two_room env: dist < 16
            notes="collector: scripts/data/collect_tworooms.py; eval: le-wm config/eval/tworoom.yaml")
    if name == "cube":
        common = dict(env_type="single", multiview=False, width=img, height=img, visualize_info=False)
        return Task(
            name="cube", env_id="swm/OGBCube-v0", action_dim=5, episode_steps=200,
            collect_kwargs=dict(**common, terminate_at_goal=False, mode="data_collection"),
            eval_kwargs=dict(**common, ob_type="states", terminate_at_goal=True),
            expert=_cube_expert, process_cols=("action",),
            state_cols=("proprio_effector_pos", "proprio_effector_yaw", "proprio_gripper_opening",
                        "proprio_joint_pos"),
            eval_callables=[
                {"method": "set_state", "args": {"qpos": {"value": "qpos"}, "qvel": {"value": "qvel"}}},
                {"method": "set_target_pos", "args": {
                    "cube_id": {"value": 0, "in_dataset": False},
                    "target_pos": {"value": "goal_privileged_block_0_pos"},
                    "target_quat": {"value": "goal_privileged_block_0_quat"}}}],
            render_profile=(512, 0),
            trivial_check=("privileged_block_0_pos", 0.04),     # cube_env._compute_successes: <= 0.04 m
            object_cols=("privileged_block_0_pos",),
            notes=("collector: scripts/data/collect_cube.py (single view); eval: le-wm config/eval/cube.yaml. "
                   "Render profile shadowsize=512, offsamples=0 (authors: 4096, 4) for 2.5x faster CPU "
                   "rendering; identical for all variants, training and evaluation."))
    raise ValueError(f"unknown task {name!r}")


def task_of(data: str) -> str:
    """Task name of a dataset name: 'cube', 'cube_eval', 'cube_p3', 'tworoom_eval' -> 'cube' / 'tworoom'."""
    for t in ("cube", "tworoom"):
        if data == t or data.startswith(t + "_"):
            return t
    raise ValueError(f"cannot tell the task of dataset {data!r}")


def apply_render_profile(task: Task) -> None:
    """Patch MuJoCo renderer creation (OGBench envs) to the task's render
    profile. Must be called before any env renders; idempotent."""
    if task.render_profile is None:
        return
    from ogbench.manipspace.envs import env as og_env

    cls = og_env.CustomMuJoCoEnv
    if getattr(cls, "_hybrid_wm_render_patched", None) == task.render_profile:
        return
    original = getattr(cls, "_hybrid_wm_original_init_renderer", cls._initialize_renderer)
    shadow, samples = task.render_profile

    def _init(self):
        self._model.vis.quality.shadowsize = shadow
        self._model.vis.quality.offsamples = samples
        return original(self)

    cls._hybrid_wm_original_init_renderer = original
    cls._initialize_renderer = _init
    cls._hybrid_wm_render_patched = task.render_profile


def pickup_candidates(dataset, goal_offset: int, height_delta: float = 0.02, contact_threshold: float = 0.5):
    """H-JEPA's Cube evaluation windows ('cube_pickup_centered', kevinghst/H-JEPA,
    h_jepa/scripts/generate_dataset_eval_trajs.py, MIT): per episode, find the
    first step the cube is lifted by >= height_delta, walk back over the
    contiguous gripper-contact run to the grasp step, and centre a
    goal_offset-step window on it. Returns (episodes, starts), one per episode."""
    import numpy as np
    pos = np.asarray(dataset.get_col_data("privileged_block_0_pos"))
    contact = np.asarray(dataset.get_col_data("proprio_gripper_contact"))
    contact = contact[:, 0] if contact.ndim == 2 else contact
    eps, starts, mid = [], [], goal_offset // 2
    for ep, (off, length) in enumerate(zip(np.asarray(dataset.offsets), np.asarray(dataset.lengths), strict=True)):
        if length < goal_offset:
            continue
        z = pos[off:off + length, 2]
        lifted = np.flatnonzero(z >= z[0] + height_delta)
        if lifted.size == 0:
            continue
        lift = int(lifted[0])
        c = contact[off:off + length] >= contact_threshold
        if c[lift]:
            grasp = lift
        else:
            prior = np.flatnonzero(c[:lift + 1])
            if prior.size == 0:
                continue
            grasp = int(prior[-1])
        while grasp - 1 >= 0 and c[grasp - 1]:
            grasp -= 1
        start = grasp - mid
        if start < 0 or start + goal_offset > length:
            continue
        eps.append(ep)
        starts.append(start)
    return np.asarray(eps, dtype=np.int64), np.asarray(starts, dtype=np.int64)


def trivial_mask(task: Task, dataset, episodes, starts, goal_offset: int):
    """Bool array: pair already satisfies the success test at its start."""
    import numpy as np
    offs = np.asarray(dataset.offsets)
    idx = offs[np.asarray(episodes)] + np.asarray(starts)
    if task.trivial_fn is not None:
        if isinstance(task.trivial_fn_col, tuple):
            cols = [np.asarray(dataset.get_col_data(c)) for c in task.trivial_fn_col]
            return np.asarray(task.trivial_fn([c[idx] for c in cols], [c[idx + goal_offset] for c in cols]), dtype=bool)
        data = np.asarray(dataset.get_col_data(task.trivial_fn_col))
        return np.asarray(task.trivial_fn(data[idx], data[idx + goal_offset]), dtype=bool)
    if task.trivial_check is None:
        return np.zeros(len(episodes), dtype=bool)
    col, thr = task.trivial_check
    data = np.asarray(dataset.get_col_data(col))
    a, b = data[idx], data[idx + goal_offset]
    return np.linalg.norm(a.reshape(len(idx), -1) - b.reshape(len(idx), -1), axis=-1) <= thr
