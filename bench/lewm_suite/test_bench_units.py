"""Unit tests for the bench-side pieces that do not need an environment:
sensor encoding, on-policy experience sampling, episode assembly, policy
extra inputs, the latent-cache action blocking. Run: pytest -q test_bench_units.py"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from hybrid_wm.control.latent_policy import ChunkPolicy


def test_policy_extra_and_weight():
    pol = ChunkPolicy(8, 5, 4, extra_dim=3)
    z, zg, ex = torch.randn(6, 8), torch.randn(6, 8), torch.randn(6, 3)
    mean, std = pol(z, zg, ex)
    assert mean.shape == (6, 5, 4) and (std > 0).all()
    with pytest.raises(ValueError):
        pol(z, zg)
    act, mask = torch.randn(6, 5, 4), torch.ones(6, 5, dtype=torch.bool)
    l0 = pol.loss(z, zg, act, mask, extra=ex, weight=torch.zeros(6))
    assert float(l0) == 0.0                                          # zero weight -> no signal
    base = ChunkPolicy(8, 5, 4)
    with pytest.raises(ValueError):
        base(z, zg, ex)
    s = pol.sample(z, zg, 3, extra=ex, std_scale=0.0)
    assert torch.allclose(s[:, 1], s[:, 0])                          # std_scale 0 -> all candidates = mean


def test_sensor_spec_and_encoder():
    from sensors import SensorEncoder, SensorSpec, info_key
    assert SensorSpec.parse("none").groups() == []
    sp = SensorSpec.parse("proprio+obj+goalobj+objnoise=0.01+drop=0.5")
    assert sp.name() == "proprio+obj+goalobj+objnoise=0.01+drop=0.5"
    assert sp.dim(11) == 11 + 3 + 3 + 3
    with pytest.raises(ValueError):
        SensorSpec.parse("goalobj")
    stats = SensorEncoder.fit_stats(torch.randn(100, 11), torch.randn(100, 3))
    enc = SensorEncoder(sp, stats, generator=torch.Generator().manual_seed(0))
    x = enc(torch.randn(1000, 11), torch.randn(1000, 3), torch.randn(1000, 3))
    assert x.shape == (1000, 20)
    miss_p, miss_o, miss_g = x[:, 11], x[:, 15], x[:, 19]
    assert 0.4 < float(miss_p.mean()) < 0.6 and 0.4 < float(miss_o.mean()) < 0.6
    assert float(miss_g.sum()) == 0.0                                # goal spec never dropped
    assert torch.all(x[miss_p.bool(), :11] == 0)
    assert info_key("proprio_effector_pos") == "proprio/effector_pos"
    assert info_key("privileged_block_0_pos") == "privileged/block_0_pos"


def _episode(F, success, D=4):
    A = F if success else F - 1
    return {"z": torch.arange(F).float()[:, None].repeat(1, D), "action": torch.arange(A).float()[:, None].repeat(1, 2),
            "proprio": torch.zeros(F, 3), "object": torch.arange(F).float()[:, None].repeat(1, 3),
            "z_goal": torch.full((D,), 99.0), "goal_obj": torch.full((3,), 99.0), "success": success,
            "reached_block": A if success else -1}


def test_experience_semantics():
    from cached_policy import Experience, H
    g = torch.Generator().manual_seed(0)
    exp = Experience([_episode(4, False), _episode(1, False), _episode(3, True)])
    assert len(exp) == 2                                             # 1-frame failure has no transition
    b = exp.sample(400, g, success_weight=2.0)
    true_goal = b["z_g"][:, 0] == 99
    assert true_goal.any() and (~true_goal).any()
    assert torch.all(b["weight"][true_goal] == 2.0) and torch.all(b["weight"][~true_goal] == 1.0)
    # hindsight: goal frame index t+k, masked blocks = k, actions are the executed ones
    hs = ~true_goal
    t, k = b["z_t"][hs, 0], b["z_g"][hs, 0] - b["z_t"][hs, 0]
    assert torch.all(k >= 1) and torch.all(k <= H)
    assert torch.equal(b["mask"][hs].sum(1).float(), k)
    assert torch.equal(b["action"][hs, 0, 0], t)                     # first block = action taken at t
    # true goal: blocks from t to the end of the episode
    tg = true_goal
    assert torch.all(b["mask"][tg].sum(1) == (3 - b["z_t"][tg, 0]).clamp(max=H))


def test_finish_episodes():
    from learn_loop import finish_episodes
    rec = lambda F: {"z": [torch.zeros(4)] * F, "action": [torch.zeros(2)] * F, "proprio": [torch.zeros(3)] * F,  # noqa: E731
                     "object": [torch.zeros(3)] * F, "z_goal": torch.zeros(4), "goal_obj": torch.zeros(3)}
    eps = finish_episodes([rec(3), rec(3), rec(1)], [True, False, False])
    assert [(e["z"].shape[0], e["action"].shape[0], e["success"]) for e in eps] == [(3, 3, True), (3, 2, False)]


def test_block_actions_row_major():
    from latent_cache import block_actions
    a = np.arange(20 * 2).reshape(20, 2).astype(np.float32)
    blk = block_actions(a, np.array([0, 5]), 5)
    assert blk.shape == (2, 10)
    assert np.array_equal(blk[1], a[5:10].reshape(-1))


def test_camera_dropout_fill():
    import types

    from system12 import GatedSolver

    class L(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def encode(self, d):
            p = d["pixels"]
            return {"emb": p.flatten(2).mean(-1, keepdim=True).expand(*p.shape[:2], 4).clone()}

        def rollout(self, info, a):
            z = info["emb"][:, 0, 0]
            return {"predicted_emb": (z + a.sum((1, 2, 3))[:, None])[:, None, None, :]}

    class P:
        def sample(self, z, g, k, generator=None, extra=None, std_scale=1.0):
            return torch.ones(z.shape[0], k, 5, 2)

    GatedSolver.horizon = property(lambda s: 5)
    goal = torch.arange(3.).view(3, 1, 1, 1, 1).expand(3, 1, 3, 2, 2).clone()
    expect = {"predict": 14.0, "hold": 10.0}
    for fill, val in expect.items():
        sol = GatedSolver(types.SimpleNamespace(cost=None, horizon=5), L(), P(), k=1, mode="s1",
                          camera_drop=1.0, fill=fill, receding=1)
        for step in range(3):
            sol.solve({"pixels": torch.full((3, 1, 3, 2, 2), 10.0 * (step + 1)), "goal": goal})
        assert len(sol.memory) == 3 and sol.stats["camera_dropped"] == 6
        assert all(abs(float(m["z"][0]) - val) < 1e-5 for m in sol.memory)
    with pytest.raises(ValueError):
        GatedSolver(types.SimpleNamespace(cost=None, horizon=5), L(), P(), mode="s1s2", camera_drop=0.5)
    # memory is per batch: reset clears it, and two episodes of one batch with the same goal are refused
    sol.reset_episodes()
    assert sol.memory == []
    same = torch.zeros(2, 1, 3, 2, 2)
    with pytest.raises(RuntimeError):
        sol.solve({"pixels": torch.ones(2, 1, 3, 2, 2), "goal": same})


def test_surprise_handles_success_and_failure_episodes():
    from learn_loop import surprise

    class L:
        def rollout(self, info, a):
            z = info["emb"][:, 0, 0]
            return {"predicted_emb": torch.stack([z, z + a[:, 0, 0, :1].expand_as(z)], 1)[:, None]}

    ok = {"z": torch.zeros(3, 4), "action": torch.zeros(3, 2)}       # success: A = F
    fail = {"z": torch.zeros(3, 4), "action": torch.zeros(2, 2)}     # failure: A = F - 1
    assert surprise(L(), [ok, fail]) == 0.0


def test_merge_h5_concatenates_and_rebuilds_offsets(tmp_path, monkeypatch):
    import h5py

    import merge_h5

    (tmp_path / "datasets").mkdir()
    monkeypatch.setattr(merge_h5, "HOME", tmp_path)

    def part(name, lens, base):
        n = sum(lens)
        with h5py.File(tmp_path / "datasets" / f"{name}.h5", "w") as f:
            f.create_dataset("ep_len", data=np.array(lens, dtype=np.int32))
            f.create_dataset("ep_offset", data=np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64))
            f.create_dataset("pixels", data=np.full((n, 2, 2, 3), base, dtype=np.uint8), chunks=(1, 2, 2, 3))
            f.create_dataset("action", data=np.arange(n * 5, dtype=np.float32).reshape(n, 5) + base)

    part("p0", [3, 2], 0)
    part("p1", [4], 100)
    info = merge_h5.merge("all", ["p0", "p1"])
    assert info["episodes"] == 3 and info["rows"] == 9
    with h5py.File(tmp_path / "datasets" / "all.h5") as f:
        assert f["ep_len"][:].tolist() == [3, 2, 4] and f["ep_offset"][:].tolist() == [0, 3, 5]
        assert f["pixels"][5, 0, 0, 0] == 100 and f["pixels"][4, 0, 0, 0] == 0
        assert np.allclose(f["action"][5], np.arange(5) + 100)
    part("bad", [2], 0)
    with h5py.File(tmp_path / "datasets" / "bad.h5", "a") as f:
        del f["action"]
    with pytest.raises(ValueError):
        merge_h5.merge("all2", ["p0", "bad"])


def test_training_losses_every_spec_path():
    """train.losses on a real batch for each predictor/grounding/ARL path (skipped without the dataset)."""
    from models import HOME, Spec, TrainConfig, build_model, state_columns
    if not (HOME / "datasets" / "cube.h5").exists():
        pytest.skip("needs the cube dataset")
    from stable_worldmodel.wm.loss import SIGReg

    import train
    from tasks import get_task

    cfg, task = TrainConfig(), get_task("cube", TrainConfig().img_size)
    for spec in ("lewm", "lewm+idm+sa+obj+arl", "hybrid+idm+sa+obj+arl"):
        sp = Spec.parse(spec)
        ds = train.make_dataset(task, cfg, "cube", state_columns(task, sp))
        m = build_model(sp, task.action_dim, int(ds.get_dim("state")), cfg).train()
        b = torch.utils.data.default_collate([ds[i] for i in range(0, 160, 40)])
        total, parts = train.losses(m, SIGReg(knots=17, num_proj=64), b, cfg, "cpu")
        total.backward()
        assert torch.isfinite(total)
        assert ("arl" in parts) == sp.arl


def test_plan_runner_backs_up_and_restores_parts(tmp_path):
    """A finished part is copied to the backup folder; a later run restores it instead of
    re-running the job, and only missing parts run."""
    import sys

    from colab_plan import Job, run
    out, bak = tmp_path / "data", tmp_path / "drive"
    mk = lambda i: Job(f"collect:p{i}", [sys.executable, "-c",  # noqa: E731
                                         f"import pathlib; pathlib.Path(r'{out}').mkdir(exist_ok=True); "
                                         f"pathlib.Path(r'{out / f'p{i}.h5'}').write_text('part{i}')"],
                       [out / f"p{i}.h5"], backup=bak / f"p{i}.h5")
    state = run([mk(0), mk(1)], workers=2, gpu_slots=1, log_dir=tmp_path / "logs", poll=0.05)
    assert state == {"collect:p0": "done", "collect:p1": "done"}
    assert (bak / "p0.h5").read_text() == "part0" and not list(bak.glob("*.tmp"))
    # new session: local data gone, p0 backed up, p1 backup lost -> p0 restored, p1 re-run
    for f in out.glob("*"):
        f.unlink()
    (bak / "p1.h5").unlink()
    jobs = [mk(0), mk(1)]
    jobs[0].cmd = [sys.executable, "-c", "raise SystemExit(3)"]            # would fail if it ran
    state = run(jobs, workers=2, gpu_slots=1, log_dir=tmp_path / "logs", poll=0.05)
    assert state == {"collect:p0": "done", "collect:p1": "done"}
    assert (out / "p0.h5").read_text() == "part0" and (bak / "p1.h5").exists()


def test_plan_collection_parts_are_small():
    from pathlib import Path

    from colab_plan import build
    jobs = build("free", [1], None, Path("/tmp/plan_parts_check"), workers=2)
    cube = [j for j in jobs if j.name.startswith("collect:cube_p")]
    ev = [j for j in jobs if j.name.startswith("collect:cube_eval_p")]
    assert len(cube) == 24 and len(ev) == 4
    eps = sum(int(j.cmd[j.cmd.index("--episodes") + 1]) for j in cube)
    seeds = {j.cmd[j.cmd.index("--seed") + 1] for j in cube + ev}
    assert eps == 600 and len(seeds) == 28                                  # all parts have distinct seeds


def test_vectorised_cube_oracle_matches_ogbench():
    """oracle.cube_oracle == CubeMarkovOracle.select_action on real dataset states, toward
    random targets (final pose pinned to the current effector pose in both)."""
    import types

    from models import HOME
    path = HOME / "datasets" / "cube_eval.h5"
    if not path.exists():
        pytest.skip("needs the cube_eval dataset")
    import h5py
    from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle

    from oracle import ORACLE_COLS, cube_oracle
    rng = np.random.default_rng(0)
    with h5py.File(path, "r") as f:
        full = {c: f[c][:] for c in ORACLE_COLS}
    n = len(full["privileged_block_0_pos"])
    rows, tgt = rng.choice(n, 600, replace=False), rng.choice(n, 600)
    S = {c: v[rows] for c, v in full.items()}
    T = {c: full[c][tgt] for c in ("privileged_block_0_pos", "privileged_block_0_yaw")}
    # half the targets = the current cube pose (exercises the release / after-placement phases)
    same = np.arange(600) % 2 == 0
    tpos = np.where(same[:, None], S["privileged_block_0_pos"], T["privileged_block_0_pos"])
    tyaw = np.where(same[:, None], S["privileged_block_0_yaw"], T["privileged_block_0_yaw"])
    ours = cube_oracle(S["proprio_effector_pos"], S["proprio_effector_yaw"], S["proprio_gripper_contact"],
                       S["privileged_block_0_pos"], S["privileged_block_0_yaw"], tpos, tyaw)
    env = types.SimpleNamespace(unwrapped=types.SimpleNamespace(_arm_sampling_bounds=([0, 0, 0], [1, 1, 1])))
    orc = CubeMarkovOracle(env=env, min_norm=0.4)
    for i in range(600):
        orc.reset(None, None)
        orc._final_pos, orc._final_yaw = S["proprio_effector_pos"][i], S["proprio_effector_yaw"][i][0]
        info = {"proprio/effector_pos": S["proprio_effector_pos"][i], "proprio/effector_yaw": S["proprio_effector_yaw"][i],
                "proprio/gripper_opening": None, "proprio/gripper_contact": S["proprio_gripper_contact"][i],
                "privileged/target_block": 0, "privileged/block_0_pos": S["privileged_block_0_pos"][i],
                "privileged/block_0_yaw": S["privileged_block_0_yaw"][i],
                "privileged/target_block_pos": tpos[i], "privileged/target_block_yaw": tyaw[i]}
        ref = orc.select_action(None, info)
        assert np.allclose(ours[i], ref, atol=1e-9), (i, ours[i], ref)


def test_oracle_sim_chunk_replays_open_loop():
    """A simulated oracle chunk, replayed open-loop from the same state, reaches the closed-loop
    oracle's end state (up to contact non-determinism)."""
    from models import HOME
    path = HOME / "datasets" / "cube_eval.h5"
    if not path.exists():
        pytest.skip("needs the cube_eval dataset")
    import h5py

    from oracle_sim import OracleSim
    with h5py.File(path, "r") as f:
        r, g = 300, 320
        qpos, qvel, ctrl = f["qpos"][r], f["qvel"][r], f["control"][r]
        tp, ty = f["privileged_block_0_pos"][g], f["privileged_block_0_yaw"][g]
    sim = OracleSim()
    chunk = sim.chunk(qpos, qvel, ctrl, tp, ty, 25)
    assert chunk.shape == (25, 5) and np.all(np.abs(chunk) <= 1)
    end_closed = sim.u.compute_ob_info()["proprio/effector_pos"].copy()
    sim.u.set_state(qpos, qvel)
    sim.u._data.ctrl[:] = ctrl
    for a in chunk:
        sim.u.step(a)
    end_open = sim.u.compute_ob_info()["proprio/effector_pos"]
    assert np.linalg.norm(end_open - end_closed) < 0.01


def test_scorer_selection_picks_lowest_score():
    """select='scorer': the solver executes the candidate with the lowest external score, with no rollout."""
    import types

    from system12 import GatedSolver

    class L(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def encode(self, d):
            return {"emb": torch.zeros(d["pixels"].shape[0], 1, 4)}

    class P:
        def sample(self, z, g, k, generator=None, extra=None, std_scale=1.0):
            return torch.arange(k, dtype=torch.float32).view(1, k, 1, 1).expand(z.shape[0], k, 5, 2).clone()

    GatedSolver.horizon = property(lambda s: 5)
    want = torch.tensor([3, 7])
    scorer = lambda info, z, zg, cand: (torch.arange(cand.shape[1])[None] - want[:, None]).abs().float()  # noqa: E731
    sol = GatedSolver(types.SimpleNamespace(cost=None, horizon=5), L(), P(), k=8, mode="s1", select="scorer",
                      scorer=scorer)
    out = sol.solve({"pixels": torch.zeros(2, 1, 3, 2, 2), "goal": torch.zeros(2, 1, 3, 2, 2)})
    assert out["actions"][:, 0, 0].tolist() == [3.0, 7.0]
    with pytest.raises(ValueError):
        GatedSolver(types.SimpleNamespace(cost=None, horizon=5), L(), P(), k=8, mode="s1", select="scorer")


def test_plan_other_environments():
    from pathlib import Path

    from colab_plan import CONFIGS, build
    tw = build("free", [1], None, Path("/tmp/plan_tw_check"), workers=2, task="tworoom")
    names = {j.name for j in tw}
    assert {n.split(":")[1] for n in names if n.startswith("train:")} == {"C1s0", "C2s0"}   # no object state
    assert not any(":s1_pk" in n or ":cam" in n or n.startswith("diag:") for n in names)  # Cube-only tests
    assert {"eval:C1s0:s1_all", "eval:C1s0:cem_all", "eval:C2s0:s1_all"} <= names
    assert all("tworoom" in " ".join(j.cmd) for j in tw if j.name.startswith(("collect:", "train:")))
    cube = {j.name for j in build("free", [1], None, Path("/tmp/plan_cube_check"), workers=2)}
    assert not any(n.endswith(("s1_all", "cem_all")) for n in cube)        # Cube plan unchanged
    assert "obj" in CONFIGS["C3"]


def test_analysis_labels_authors_protocol():
    from analyze_colab import label_of
    base = {"controller": "s1", "policy_file": "policy_s1.pt", "goal_offset": 25, "k": 1}
    assert label_of({**base, "pairs": "all"}) == "s1_all"
    assert label_of({**base, "pairs": "nontrivial"}) == "s1_nt"
    assert label_of({"controller": "cem", "goal_offset": 25, "pairs": "all"}) == "cem_all"
    assert label_of({**base, "pairs": "all", "k": 16}) is None


def test_plan_detects_outputs_of_a_lost_and_retrained_model(tmp_path):
    """Weights lost from cloud storage and retrained: policies and results built from the old weights must
    re-run (and be kept aside, not deleted); a normal finished model must stay done."""
    import os

    from colab_plan import STALE_TOLERANCE_S, build
    home = tmp_path / "home"
    jobs = {j.name: j for j in build("free", [2], ["C2"], home, workers=2)}
    pol, hier, ev = jobs["policy:C2s1"], jobs["hier:C2s1"], jobs["eval:C2s1:s1_nt"]
    ev.outputs = [tmp_path / "results" / ev.outputs[0].name]         # result paths follow STABLEWM_HOME: keep the
    assert "train:C2s1" in ev.deps                                    # never evaluate without weights
    w = jobs["train:C2s1"].outputs[0]
    for f in (pol.outputs[0], hier.outputs[0], ev.outputs[0]):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("old")
    assert not pol.done() and pol.stale()                             # weights missing
    w.parent.mkdir(parents=True, exist_ok=True)
    w.write_text("w")
    assert pol.done() and hier.done() and ev.done()                   # finished model, weights written first
    t = w.stat().st_mtime
    for f in (pol.outputs[0], hier.outputs[0], ev.outputs[0]):
        os.utime(f, (t - 2 * STALE_TOLERANCE_S, t - 2 * STALE_TOLERANCE_S))
    assert not pol.done() and not hier.done() and not ev.done()       # retrained after them
    os.utime(pol.outputs[0], (t - 60, t - 60))                        # storage timestamp jitter: still fine
    assert pol.done()
    moved = ev.set_aside()
    assert len(moved) == 1 and moved[0].read_text() == "old" and not ev.outputs[0].exists()
    assert moved[0].parent.name == "stale"


def test_plan_warns_but_never_stops_on_low_storage(tmp_path, monkeypatch):
    import shutil

    import colab_plan
    (tmp_path / "checkpoints").mkdir()
    Usage = type(shutil.disk_usage(tmp_path))
    monkeypatch.setattr(colab_plan.shutil, "disk_usage", lambda p: Usage(10**12, 10**12 - 5 * 10**8, 5 * 10**8))
    assert colab_plan.check_space(tmp_path) is False                  # reported as low; main() only warns
    import inspect
    assert "sys.exit" not in inspect.getsource(colab_plan.main)
    monkeypatch.setattr(colab_plan.shutil, "disk_usage", lambda p: Usage(10**12, 10**11, 10**11))
    assert colab_plan.check_space(tmp_path) is True


def test_plan_rejects_unreadable_trained_weights(tmp_path):
    import torch

    from colab_plan import Job, verify_output
    w = tmp_path / "weights.pt"
    torch.save({"a": torch.zeros(2)}, w)
    job = Job("train:C1s0", [], [w])
    assert verify_output(job)
    w.write_bytes(w.read_bytes()[:20])                                 # truncated, as after a lost upload
    assert not verify_output(job)
    assert verify_output(Job("eval:C1s0:s1_nt", [], [w]))              # only training outputs are loaded


def test_contrastive_verifier_learns_and_retrieves():
    """InfoNCE on a toy where the right chunk is a fixed function of (z, z_goal): the verifier must rank the
    true chunk first, prefer it over a perturbed copy, and the retrieval policy must return it from a menu."""
    import torch

    from contrastive_verifier import ContrastiveVerifier, RetrievalPolicy, info_nce
    torch.manual_seed(0)
    D, Hb, A = 8, 5, 4
    W = torch.randn(2 * D, Hb * A)
    make = lambda z, zg: torch.tanh(torch.cat([z, zg], -1) @ W).reshape(-1, Hb, A)  # noqa: E731
    v = ContrastiveVerifier(D, Hb * A, emb=32)
    opt = torch.optim.Adam(v.parameters(), 3e-3)
    g = torch.Generator().manual_seed(0)
    for _ in range(300):
        z, zg = torch.randn(256, D), torch.randn(256, D)
        loss = info_nce(v, z, zg, make(z, zg), 2, g)
        opt.zero_grad()
        loss.backward()
        opt.step()
    z, zg = torch.randn(64, D), torch.randn(64, D)
    true = make(z, zg)
    cand = torch.stack([true, true + 0.5 * torch.randn_like(true)], 1)          # [64, 2, H, A]
    assert v.score(z, zg, cand).shape == (64, 2)
    assert (v.score(z, zg, cand).argmin(1) == 0).float().mean() > 0.9          # lower score = better
    pol = RetrievalPolicy(v.eval(), true)
    out = pol.sample(z, zg, 3)
    assert out.shape == (64, 3, Hb, A)
    assert (out[:, 0] - true).abs().amax(dim=(1, 2)).lt(1e-6).float().mean() > 0.9


def _fake_env(n: int, a_dim: int = 2):
    import gymnasium as gym

    class Env:
        num_envs = n
        single_action_space = gym.spaces.Box(-1e9, 1e9, (a_dim,), np.float32)
        action_space = gym.spaces.Box(-1e9, 1e9, (n, a_dim), np.float32)
    return Env()


class _CountingSolver:
    """Plans whose step j of call c holds the value 100 * c + j, so executed actions show which plan and
    which step of it ran. Remembers the step index it was given (like GCIDMSolver)."""

    def __init__(self):
        self.calls, self.step, self.steps_seen = 0, 0, []

    def configure(self, *, action_space, n_envs, config):
        self.n, self.cfg = n_envs, config

    action_dim, n_envs, horizon = 2, 1, 5

    def __call__(self, info, init_action=None):
        return self.solve(info, init_action)

    def solve(self, info, init_action=None):
        self.calls += 1
        self.steps_seen.append(self.step)
        n = len(info["pixels"])
        steps = self.cfg.horizon * self.cfg.action_block
        v = 100 * self.calls + torch.arange(steps, dtype=torch.float32)
        plan = v[None, :, None].expand(n, steps, 2).reshape(n, self.cfg.horizon, -1)
        return {"actions": plan.clone()}


@pytest.mark.parametrize("exec_steps,receding", [(1, 1), (5, 1), (25, 5), (3, 1)])
def test_step_policy_executes_only_the_first_steps(exec_steps, receding):
    import stable_worldmodel as swm
    from stable_worldmodel.policy import WorldModelPolicy

    from gcidm import step_policy
    solver = _CountingSolver()
    pol = step_policy(WorldModelPolicy)(solver=solver, exec_steps=exec_steps,
                                        config=swm.PlanConfig(horizon=5, receding_horizon=receding, action_block=5))
    pol.set_env(_fake_env(2))
    executed = []
    for _ in range(50):
        a = pol.get_action({"pixels": np.zeros((2, 1, 3, 4, 4), np.float32)})
        executed.append(a[0, 0])
    assert solver.calls == -(-50 // exec_steps)                       # one decision every exec_steps steps
    expect = [100 * (t // exec_steps + 1) + t % exec_steps for t in range(50)]
    assert executed == expect                                          # always the first steps of the newest plan
    assert solver.steps_seen == list(range(0, 50, exec_steps))         # step index handed to the solver


def test_gcidm_horizon_and_shapes():
    from gcidm import GCIDM, training_pairs
    net = GCIDM(8, 5).eval()
    z, zg = torch.randn(4, 8), torch.randn(4, 8)
    a1 = net(z, zg, torch.full((4,), 10.0))
    a2 = net(z, zg, torch.full((4,), 40.0))
    assert a1.shape == (4, 5) and torch.allclose(a1, a2)               # AdaLN-Zero: no horizon effect at init
    with pytest.raises(ValueError):
        net(z, zg)
    assert GCIDM(8, 5, horizon=False)(z, zg).shape == (4, 5)
    cache = {"offsets": torch.tensor([0, 10]), "lengths": torch.tensor([10, 60])}
    rows, hmax = training_pairs(cache)
    assert len(rows) == 9 + 59 and int(hmax[0]) == 9 and int(hmax[9]) == 50 and int(hmax[-1]) == 1
    assert bool((rows + hmax < torch.tensor([10] * 9 + [70] * 59)).all())   # goals stay inside the episode


def test_short_chunk_batch_targets_and_goals():
    from cached_policy import OfflineWindows
    from short_chunk import short_batch
    lengths = torch.tensor([40, 60])
    n_rows = int(lengths.sum())
    cache = {"frameskip": 5, "offsets": torch.tensor([0, 40]), "lengths": lengths,
             "z": torch.arange(n_rows, dtype=torch.float32)[:, None].repeat(1, 3),
             "action": torch.arange(n_rows, dtype=torch.float32)[:, None].repeat(1, 2)}
    win = OfflineWindows(cache)
    for c in (1, 5):
        b = short_batch(cache, win, 64, c, torch.Generator().manual_seed(0))
        r = b["z_t"][:, 0].long()
        assert torch.equal(b["z_g"][:, 0].long(), r + 5 * b["k"])                 # goal k blocks ahead
        assert b["action"].shape == (64, 1, 2 * c)
        assert torch.equal(b["action"][:, 0, ::2].long(), r[:, None] + torch.arange(c)[None])   # next c actions
        ep_end = torch.where(r < 40, 39, 99)
        assert bool((r + 25 <= ep_end).all())                                   # same start rows as System 1


def test_latent_step_solver_pads_the_chunk():
    import gymnasium as gym

    from gcidm import LatentStepSolver

    class Enc(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))

        def encode(self, d):
            return {"emb": d["pixels"].flatten(2)[:, :, :4]}

    class Cem:
        def configure(self, *, action_space, n_envs, config):
            self.n_envs, self.action_dim, self.horizon = n_envs, 5, config.horizon

    import stable_worldmodel as swm
    chunk = torch.arange(3 * 2 * 5, dtype=torch.float32).reshape(3, 2, 5)
    s = LatentStepSolver(Cem(), Enc(), lambda zn, zg, t: chunk)
    s.configure(action_space=gym.spaces.Box(-1, 1, (3, 25)), n_envs=3,
                config=swm.PlanConfig(horizon=5, receding_horizon=1, action_block=5))
    out = s({"pixels": torch.zeros(3, 1, 3, 4, 4), "goal": torch.zeros(3, 1, 3, 4, 4)})["actions"]
    steps = out.reshape(3, 25, 5)
    assert out.shape == (3, 5, 25) and torch.equal(steps[:, :2], chunk)
    assert torch.equal(steps[:, 2:], chunk[:, -1:].expand(3, 23, 5))


def test_horizon_rules_agree():
    """Our reimplementation's h = budget - t + 1 (t 1-based) equals the authors' min(budget - steps_done, H)."""
    budget = 50
    for done in range(budget):
        ours = min(max(budget - (done + 1) + 1, 1), 50)
        assert ours == min(budget - done, 50)


def test_cluster_bootstrap_basic_cases():
    from cluster_bootstrap import cluster_bootstrap, holm
    rng = np.random.default_rng(1)
    a = (rng.random((4, 60)) < 0.5).astype(float)
    same = cluster_bootstrap(a, a, n_boot=2000)
    assert same["diff_pts"] == 0 and same["ci_lo"] == 0 and same["ci_hi"] == 0 and same["p_boot"] == 1.0
    full = cluster_bootstrap(np.ones((4, 60)), np.zeros((4, 60)), n_boot=2000)
    assert full["diff_pts"] == 100 and full["p_boot"] == 2 / 2000
    # duplicating every pair inside one cluster changes nothing; treating copies as independent narrows the CI
    b = (rng.random((4, 60)) < 0.4).astype(float)
    a2, b2 = np.repeat(a, 3, axis=1), np.repeat(b, 3, axis=1)
    by_ep = cluster_bootstrap(a2, b2, np.repeat(np.arange(60), 3), n_boot=4000)
    by_pair = cluster_bootstrap(a2, b2, n_boot=4000)
    ref = cluster_bootstrap(a, b, n_boot=4000)
    assert abs(by_ep["diff_pts"] - ref["diff_pts"]) < 1e-9
    assert (by_pair["ci_hi"] - by_pair["ci_lo"]) < 0.75 * (by_ep["ci_hi"] - by_ep["ci_lo"])
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


# ---- trivial_share.py: pairs already solved at the start (le-wm sampling rule)
def test_trivial_share_counts_and_lewm_draw(tmp_path):
    h5py = pytest.importorskip("h5py")
    import numpy as np

    from trivial_share import lewm_eval_draw, summary, valid_starts
    # two episodes of 30 steps; the cube moves 0.1 m at step 10 of episode 0 and never in episode 1
    ep = np.repeat([0, 1], 30)
    step = np.tile(np.arange(30), 2)
    pos = np.zeros((60, 3))
    pos[10:30, 0] = 0.1
    with h5py.File(tmp_path / "toy.h5", "w") as f:
        f["ep_idx"], f["step_idx"], f["privileged_block_0_pos"] = ep, step, pos
    valid = valid_starts(ep, step, 25)
    assert valid.tolist() == [0, 1, 2, 3, 4, 30, 31, 32, 33, 34]       # step <= 30 - 25 - 1
    s = summary(tmp_path / "toy.h5", goal_offset=25, num_eval=3, seed=42)
    # episode 0: starts 0..4 have the goal at 25..29, after the move -> not solved; episode 1: all solved
    assert s["all_pairs"] == {"pairs": 10, "solved_at_start": 5, "share": 0.5}
    drawn = lewm_eval_draw(valid, 3, 42)
    g = np.random.default_rng(42)                                       # the calls in le-wm eval.py
    assert drawn.tolist() == np.sort(valid[g.choice(len(valid) - 1, size=3, replace=False)]).tolist()
    assert s["lewm_eval_draw"]["rows"] == drawn.tolist()


def test_time_decisions_skips_policies_without_solver():
    from evaluate import _time_decisions

    class NoSolver:                                       # like stable-worldmodel's RandomPolicy
        pass
    out: list = []
    _time_decisions(NoSolver(), out)
    assert out == []
