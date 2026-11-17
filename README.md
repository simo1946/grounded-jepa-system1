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
| `paper/` | The paper's source (`paper.md`, `tables.py`), its builder (`build.py`, `docx_builder.js`), `reproduce.sh`, and the built `paper.docx` / `paper.pdf`; `osaudit.toml` (repository root) traces every printed number |
| `notebooks/official_cube_check.ipynb` | Colab check of the LeWM authors' released Cube dataset: pairs already solved at the start, and a random policy with their `eval.py` |
| `tests/`, `bench/lewm_suite/test_bench_units.py` | Unit tests (`pytest` runs both) |

## Install

Python 3.10+ (tested with 3.13), Linux, MuJoCo rendering through EGL.

```bash
pip install -e .                 # installs hybrid_wm and its dependencies (stable-worldmodel 0.1.1, OGBench 1.2.1)
export MUJOCO_GL=egl
export STABLEWM_HOME=$PWD/bench/lewm_suite/swm_home    # data, checkpoints, caches and results go here
```

## Reproduce the GPU study (Tables 2–4, Figures 3–5)

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

## CPU follow-up studies (Sections 5.1, 5.2 and 5.6)

These run on a 2-CPU machine at the CPU-study scale (400/80 episodes, 2,000 world-model steps). Their first result
files were lost when the compute workspace was replaced (November 2026), so they were run again on regenerated data
with `run_rerun.sh`, which follows the original scripts below with two recorded changes: the 400 training episodes
were collected in two parts (seeds 3072 and 3073) and merged, and System 1 is trained with the GPU-study recipe
(`cached_policy.py`, 3,000 steps, batch 256) instead of `train_policy.py`. Regenerating the data with the same
seeds did not reproduce the original episodes, so every number is new. `analyze_rerun.py` computes the numbers the
paper prints from the result files in `results/cpu_rerun/`.

```bash
cd bench/lewm_suite
export STABLEWM_HOME=/path/to/swm_home MUJOCO_GL=egl
python collect.py --task cube --name cube_eval --episodes 80 --seed 777
python collect.py --task cube --name cube_p0 --episodes 200 --seed 3072
python collect.py --task cube --name cube_p1 --episodes 200 --seed 3073
python merge_h5.py --out cube --parts cube_p0 cube_p1
./run_rerun.sh base && ./run_rerun.sh timing && ./run_rerun.sh select && ./run_rerun.sh ceiling && ./run_rerun.sh verifier
```

| Paper section | Stage of `run_rerun.sh` | Original script | Plan file |
|---|---|---|---|
| Introduction: random policy on the standard protocol | `base` | – | – |
| 5.1 single-robot decision latency | `timing` | – | – |
| 5.2 CPU replication of object grounding | `base` | `run_next9.sh` | – |
| 5.6 selection vs a random candidate | `select` | `run_verify.sh` | `prereg_verification.md` |
| 5.6 privileged judge and outcome critic (E6) | `ceiling` | `run_critic_data.sh`, `run_critic_eval.sh` | `prereg_critic.md` |
| 5.6 contrastive verifier (E10) | `verifier` | `run_verifier.sh` | `prereg_critic.md` |

Not in the current paper: practice in the simulator (`run_residual.sh`, `run_practice.sh`, `run_awr.sh`) and clean
labels and DAgger (`run_dagger.sh`, `run_dagger2.sh`). Their result files were lost and they were not run again;
the code is kept.

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

## Paper and audit

Every number in the paper comes from a results file in `results/paper/`, written by `osaudit run` (command, input
checksums and git commit recorded). `paper/build.py` prints the numbers from those files and writes `osaudit.toml`;
[osaudit](https://github.com/ASSERT-KTH/osaudit) (`pip install osaudit==0.7.0`, Python 3.13+) then checks the PDF
against the files and the files against the code:

```bash
paper/reproduce.sh        # regenerate results/paper/*.json, the figures, paper.docx and paper.pdf, then run osaudit
osaudit                   # the audit alone; prints nothing when every number is traced
```

Two results files also read inputs from outside the repository; `reproduce.sh` looks for them in the git-ignored
folder `external/` and skips (keeping the committed file) when they are absent: `external/gcidm_repo`, the GC-IDM
authors' code at commit 48c45b1 (for `model_sizes.json`), and `external/cube_data/cube.h5` and `cube_eval.h5`, the
regenerated Cube data (for `trivial_share_cpu.json`).

## Checks added after the review

- `check_goal_collisions.py`: confirms that no two evaluation pairs share a goal image (the camera-dropout
  memory recognised episodes by their goal latent; it is now reset for every batch of episodes and refuses
  duplicate goals).
- `analyze_colab.py` refuses duplicate result files and paired comparisons on different pairs.
- `evaluate.py` records the wall-clock time of every decision (`decision_seconds`), with CUDA synchronised.

## Licence

MIT (`LICENSE`). Third-party code and dependencies: `THIRD_PARTY_NOTICES.md`.
