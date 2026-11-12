# Grounded JEPA World Models for Fast System-1 Control

Code for the paper of the same name (Simone Bonaccorsi, 2026). It studies which parts of a small JEPA world
model (LeWM) make a simulated robot succeed on the OGBench Cube task: grounding the latent on the object,
a fast goal-conditioned policy ("System 1") instead of planning, a slow subgoal network for far goals, the
world model as memory when camera frames are lost, and why choosing among the policy's proposals with the
world model does not help.

## Layout

| Path | What it is |
|---|---|
| `src/hybrid_wm/` | Library: System 1 policies (`control/latent_policy.py`), grounding heads (`grounding.py`), the hybrid Mamba-3/attention predictor (`integrations/lewm.py`, `backbone.py`, `layers/`) |
| `bench/lewm_suite/` | Experiments: data collection, training, caching, policy training, evaluation, analysis |
| `bench/lewm_suite/build_notebook.py` | Builds the Colab notebook that runs the GPU study end to end |
| `bench/lewm_suite/prereg_*.md` | Dated plans for the CPU follow-up studies, written before those runs, with addenda |
| `docs/gpu_study_results.md` | Result tables of the GPU study as produced by `analyze_colab.py` |
| `bench/lewm_suite/gcidm.py`, `official_gcidm.py`, `short_chunk.py`, `run_feedback.sh`, `run_feedback2.sh`, `analyze_feedback.py` | Feedback-frequency study (paper Section 5.8): our GC-IDM-style controller, an adapter that runs the GC-IDM authors' code on our latents, System 1 trained for 5- and 1-step chunks, and execution of any controller for n steps |
| `bench/lewm_suite/cluster_bootstrap.py` | Paired success differences with a bootstrap over evaluation episodes (or pairs) |
| `docs/feedback_study.md` | Its results |
| `results/` | Per-episode outcomes and evaluation pair identities of every reported run, with the summaries |
| `tests/`, `bench/lewm_suite/test_bench_units.py` | Unit tests (`pytest` runs both) |

## Install

Python 3.10+ (tested with 3.13), Linux, MuJoCo rendering through EGL.

```bash
pip install -e .                 # installs hybrid_wm and its dependencies (stable-worldmodel 0.1.1, OGBench 1.2.1)
export MUJOCO_GL=egl
export STABLEWM_HOME=$PWD/bench/lewm_suite/swm_home    # data, checkpoints, caches and results go here
```

## Reproduce the GPU study (Tables 2–4, Figures 2–3)

The main study ran on a free Google Colab T4. Build the notebook and run it there:

```bash
cd bench/lewm_suite && python build_notebook.py      # writes notebooks/hybrid_wm_gpu_study.ipynb
```

The notebook collects the data, trains every configuration, evaluates and analyses, and resumes after a
disconnect. Tiers: `[1]` = seed 0 of C1, C2, C3, C5; `[2]`, `[3]` = seeds 1 and 2 (+ C6/C7 seed 0). Section 12
runs TwoRoom; section 13 the authors' standard protocol (Appendix A). The same plan can run on any machine:

```bash
python colab_plan.py --profile free --tiers 1 --list            # see the jobs
python colab_plan.py --profile free --tiers 1 2 3 --run         # run them (skip-if-done)
python analyze_colab.py --profile free                          # tables and paired tests
python colab_plan.py --profile smoke --tiers 1 --run            # toy-scale check of every job type (minutes)
```

Pipeline per configuration: `collect.py` → `merge_h5.py` → `train.py` (world model) → `latent_cache.py`
(encode every frame once) → `cached_policy.py` (System 1) and `cached_subgoal.py` (subgoal network) →
`evaluate.py` (robot tests) → `probe_diag.py`, `rollout_diag.py` (diagnostics) → `analyze_colab.py`.

Configurations (`colab_plan.py: CONFIGS`): C1 `lewm`; C2 `lewm+idm+sa`; C3 `lewm+idm+sa+obj`;
C5 `hybrid+idm+sa+obj`; C6/C7 add `+arl`. Model specs are parsed by `models.py: Spec`.

## CPU follow-up studies (Section 5.6 and Appendix B)

These ran on a 2-CPU machine at the CPU-study scale (400/80 episodes, 2,000 world-model steps). Each script
skips finished work.

| Paper section | Script | Main programs | Plan file |
|---|---|---|---|
| 5.2 CPU replication of object grounding | `run_next9.sh` | `train.py`, `cached_policy.py`, `evaluate.py`, `analyze.py` | – |
| 5.6 selection vs random candidate | `run_verify.sh` | `evaluate.py`, `analyze_verify.py` | `prereg_verification.md` |
| 5.6 privileged judge and outcome critic (E6) | `run_critic_data.sh`, `run_critic_eval.sh` | `gen_critic_data.py`, `critic.py`, `oracle.py` | `prereg_critic.md` |
| 5.6 contrastive verifier (E10) | `run_verifier.sh` | `contrastive_verifier.py` | `prereg_critic.md` |
| B.1 practice in the simulator | `run_residual.sh`, `run_practice.sh`, `run_awr.sh` | `residual_rl.py`, `practice_rl.py`, `awr_policy.py` | `prereg_critic.md` |
| B.2 flow head, clean labels, DAgger | `run_overnight.sh`, `run_dagger.sh`, `run_dagger2.sh`, `run_flow_budget.sh` | `cached_policy.py`, `oracle_policy.py`, `dagger.py` | `prereg_overnight.md` |
| B.3 action-resolution loss | `run_extra4.sh` | `train.py`, `evaluate.py` | – |

## Feedback-frequency study (Section 5.8)

Uses the CPU-study models C1 and C3 (seeds 0 and 1). The authors' GC-IDM code is licensed CC BY-NC 4.0 and is
not included; clone it at the pinned commit and point `GCIDM_REPO` at it:

```bash
git clone https://github.com/hdnndh/Latent-Geometry-Beyond-Search-Amortizing-Planning-in-World-Models gcidm_repo
git -C gcidm_repo checkout 48c45b1cb2b34dd2c1c61d222c8309de567fde55
cd bench/lewm_suite
./run_feedback.sh                                # System 1 at 25/5/1 steps, our GC-IDM-style controller
GCIDM_REPO=../../gcidm_repo ./run_feedback2.sh   # the authors' GC-IDM (H_max 50 and 1), System 1 for 5/1-step chunks
python analyze_feedback.py                       # tables, episode-clustered bootstrap, Holm correction
```

Plan and dated addendum: `prereg_feedback.md`.

## Checks added after the review

- `check_goal_collisions.py`: confirms that no two evaluation pairs share a goal image (the camera-dropout
  memory recognised episodes by their goal latent; it is now reset for every batch of episodes and refuses
  duplicate goals).
- `analyze_colab.py` refuses duplicate result files and paired comparisons on different pairs.
- `evaluate.py` records the wall-clock time of every decision (`decision_seconds`), with CUDA synchronised.

## Licence

MIT (`LICENSE`). Third-party code and dependencies: `THIRD_PARTY_NOTICES.md`.
