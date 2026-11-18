"""Build hybrid_wm_gpu_study.ipynb (the Colab notebook for the GPU study).

    python build_notebook.py            # writes ../../notebooks/hybrid_wm_gpu_study.ipynb
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

SWM_COMMIT = "21446f1ede6d5284e981bd7b47f432b994e6d812"

INTRO = r"""
# Hybrid JEPA world model — GPU study on OGBench-Cube

This notebook runs every comparison left open by the CPU study (see `docs/assessment.md` in the repository).
It also tests the untested pairings of our best parts, with more data than the CPU study, several seeds and paired statistics.
It is **resumable**: models, results and logs live on Google Drive and every job skips when its output exists.
After a disconnect, re-run all cells.

**Configurations (world model)**

| id | spec | question |
|---|---|---|
| C1 | `lewm` | LeWM baseline (authors' recipe) |
| C2 | `lewm+idm+sa` | proprioceptive grounding |
| C3 | `lewm+idm+sa+obj` | + object-aware grounding (training-only target) |
| C4 | `hybrid+idm+sa` | our hybrid Mamba-3/attention predictor |
| C5 | `hybrid+idm+sa+obj` | hybrid × object grounding *(new pair)* |
| C6 | `lewm+idm+sa+obj+arl` | + action-resolution loss *(new)* |
| C7 | `hybrid+idm+sa+obj+arl` | all best parts *(new)* |
| C8 | `lewm+idm` | inverse-dynamics-only ablation |
| C9 | `lewm+idm+sa+obj+enc=mnv4s` | fast encoder (MobileNetV4-S) |

**Each model is evaluated on:**
- System 1 (the goal-conditioned chunk policy) and two-level System 1 (subgoal network, gated by distance), on three protocols:
  - 100 non-trivial pairs at goal +25;
  - H-JEPA's grasp-centred windows;
  - goal +50.
- Best-of-16 selection by the world model, and verify-then-act with a CEM fallback.
- CEM (C1, C3, C6, C7).
- Camera dropout at 50% and 80%, with blank / hold-last / world-model-prediction fill (C3, C5, C6, C7).
- Latent probes and action-identifiability diagnostics.

**Pre-registered hypotheses.** These are fixed before running. The tests:
- paired exact McNemar on start/goal pairs pooled over seeds;
- Wilcoxon signed-rank for H3;
- Holm correction within this family only.

Every other comparison in the output is exploratory.

| # | hypothesis | supported if |
|---|---|---|
| H1 | Object-aware grounding improves System 1 | C3 > C2 on non-trivial pairs |
| H2 | The action-resolution loss (ARL) makes the world model rank nearby actions | C6/C7 beat C3/C5 on perturbed-chunk identifiability (σ = 0.1, 0.3). This is a mechanism check, not evidence: it is close to ARL's own training objective. |
| H3 | With ARL, world-model selection stops hurting | (best-of-16 − mean) is higher with ARL (C6 vs C3, C7 vs C5) |
| H3b | ARL improves System 1 itself | C6 > C3, C7 > C5 |
| H4 | The hybrid predictor is a better state estimator | C5 > C3 and C7 > C6 under camera dropout with prediction fill |
| H5 | Two-level System 1 helps long horizons without hurting short ones | hier > flat at +50 (pooled over configs), and no significant loss on grasp windows |
| H6 | The full combination beats the LeWM baseline | C7 > C1 on both System 1 protocols |

**Profiles and runtime.** Collection and evaluation are CPU-bound (MuJoCo rendering), so CPU count and RAM matter as much as the GPU.

| profile | for | tier 1 | later tiers | rough time for tier 1 |
|---|---|---|---|---|
| `free` (default) | free Colab: T4, 2 CPUs, ~12.7 GB RAM | C1, C2, C3, C5, seed 0; 600 / 100 episodes; 60 evaluation pairs | 2: seed 1 · 3: seed 2, plus the ARL configs C6 and C7 (seed 0) · 4 (optional): C6 and C7 seeds 1–2 | ~4–5 h per tier |
| `standard` | Colab Pro, A100/L4 high-RAM | all 9 configs, seed 0; 2,000 / 200 episodes; 100 pairs | 2: seeds 1–2 for C1–C7 | ~6–10 h |

On free Colab:
- Keep the browser tab open; an idle session is disconnected.
- When the daily GPU quota ends the session, open the notebook again later and run all cells. Finished work is skipped; training resumes from its last checkpoint.
- Tier 1 already answers H1, H4, H5 and the grounding and hybrid comparisons once. Tiers 2–3 add the seeds a paper needs.

**Action-resolution loss (C6, C7): CPU result first.** On the CPU study (seed 0) ARL behaved sanely but did not help:
- System 1: 34 vs 35/50 without it.
- Grasp windows: 45 vs 44.
- Best-of-16 selection: 22 vs 28/50, i.e. worse.
- 5-block action identifiability and rollout error: also worse.

It resolves actions one step ahead (held-out accuracy 0.43 vs 0.20 chance), but that does not transfer to multi-step rollouts. So H2/H3 are low priority on free Colab (tier 3). A multi-step version is the next idea to try.
"""

SETTINGS = r'''
# ---- settings (edit) -----------------------------------------------------------------------------
PROFILE = "free"          # "free" (T4 free Colab), "standard" (A100/L4 high-RAM), "quick", "smoke"
TIERS = [1]               # run [1] first; then [2], then [3] (see the table above), one session each
CONFIGS = None            # None = all, or e.g. ["C3", "C7"]
RUN_PREFLIGHT = True      # ~15-25 min end-to-end toy run on this machine before the long run
WORKERS = "auto"          # parallel jobs (MuJoCo rendering is CPU-bound); "auto" = from CPUs and RAM
GPU_SLOTS = "auto"        # concurrent training jobs: "auto" = 2 on GPUs with >= 40 GB, else 1
IMG_SIZE = 64             # 64 matches the CPU study; 128 is a separate experiment (own data and models)
DRIVE_DIR = "/content/drive/MyDrive/hybrid-wm"     # persistent: checkpoints, results, logs, data backup
REPO_ZIP = f"{DRIVE_DIR}/grounded-jepa-system1.zip"        # upload the repository zip here ...
REPO_GIT = ""                                       # ... or set a git URL (used instead of the zip)
'''

RUNTIME = r'''
import os, sys, json, shutil, subprocess, pathlib, importlib.util

IN_COLAB = "google.colab" in sys.modules or os.path.exists("/content")

def run(cmd, cwd=None, check=True, env=None):
    """Run a command, streaming its output into the cell."""
    print("$", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)), flush=True)
    p = subprocess.Popen(cmd, cwd=cwd, shell=isinstance(cmd, str), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, env=env)
    for line in p.stdout:
        if "UserWarning" in line or "warnings.warn" in line or "logger.warn" in line:
            continue
        print(line, end="", flush=True)
    rc = p.wait()
    if check and rc != 0:
        raise RuntimeError(f"command failed (exit {rc})")
    return rc

cpus = os.cpu_count() or 1
ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
gpu_name, gpu_mem_gb = ("none", 0.0)
try:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True)
    if gpu.returncode == 0 and gpu.stdout.strip():
        name, mem = gpu.stdout.strip().splitlines()[0].rsplit(",", 1)
        gpu_name, gpu_mem_gb = name.strip(), float(mem) / 1024
except FileNotFoundError:                                  # no NVIDIA driver on this machine
    pass
# each evaluation keeps ~10 MuJoCo envs (~0.45 GB each) plus a model; training needs ~4-6 GB
env_chunk = 10 if ram_gb >= 24 else 4                    # MuJoCo envs per evaluation (~0.45 GB each)
per_worker_gb = 1.5 + 0.45 * env_chunk                    # one evaluation process
workers = WORKERS if WORKERS != "auto" else max(1, min(cpus, int((ram_gb - 4) // per_worker_gb)))  # 4 GB for training
gpu_slots = GPU_SLOTS if GPU_SLOTS != "auto" else (2 if gpu_mem_gb >= 40 else 1)
print(json.dumps({"colab": IN_COLAB, "gpu": gpu_name, "gpu_mem_gb": round(gpu_mem_gb, 1), "cpus": cpus,
                  "ram_gb": round(ram_gb, 1), "workers": workers, "gpu_slots": gpu_slots, "env_chunk": env_chunk}))
if gpu_name == "none":
    print("WARNING: no GPU found. Runtime > Change runtime type > GPU (an A100/L4 high-RAM runtime is best).")
'''

DRIVE = r'''
# ---- persistent storage: models/results/logs on Drive; datasets and latent caches on fast local disk
if IN_COLAB and os.path.exists("/content"):
    from google.colab import drive
    drive.mount("/content/drive")
DRIVE = pathlib.Path(DRIVE_DIR)
DRIVE.mkdir(parents=True, exist_ok=True)
tag = f"img{IMG_SIZE}"
HOME = pathlib.Path(f"/content/swm_home_{tag}" if IN_COLAB else f"./swm_home_{tag}").resolve()
(HOME / "datasets").mkdir(parents=True, exist_ok=True)
(HOME / "cache").mkdir(exist_ok=True)
for d in ("checkpoints", "results", "logs"):
    target = DRIVE / f"swm_home_{tag}" / d
    target.mkdir(parents=True, exist_ok=True)
    link = HOME / d
    if link.is_symlink() or not link.exists():
        if link.is_symlink():
            link.unlink()
        link.symlink_to(target, target_is_directory=True)
BACKUP = DRIVE / "datasets_backup" / tag
BACKUP.mkdir(parents=True, exist_ok=True)
for f in BACKUP.glob("*.h5"):                       # restore data collected in an earlier session
    if not (HOME / "datasets" / f.name).exists():
        print("restoring", f.name); shutil.copy2(f, HOME / "datasets" / f.name)
print("HOME =", HOME)
'''

INSTALL = rf'''
# ---- dependencies, pinned to the versions the CPU study used (Colab's own torch is kept)
need = ["stable_worldmodel", "stable_pretraining", "ogbench", "dm_control", "mujoco", "timm", "h5py", "hdf5plugin",
        "zstandard"]
if not all(importlib.util.find_spec(m) for m in need):
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    run(pip + ["git+https://github.com/galilai-group/stable-worldmodel@{SWM_COMMIT}"])
    run(pip + ["stable-pretraining==0.1.8", "mujoco==3.15.0", "timm>=1.0.20", "h5py", "hdf5plugin", "einops",
               "hydra-core", "loguru", "scikit-learn", "scipy", "pygame", "pymunk", "shapely",
               "opencv-python-headless", "dm-env", "dm-tree", "absl-py", "lxml", "PyOpenGL", "glfw", "pytest",
               "zstandard", "huggingface_hub"])
    # ogbench/dm_control without their dependency pins (labmaze is not needed for Cube)
    run(pip + ["--no-deps", "ogbench==1.2.1", "dm_control==1.0.48"])
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
'''

CODE = r'''
# ---- the code: repository zip from Drive (or a git URL)
REPO = pathlib.Path(os.environ.get("HWM_REPO", "/content/grounded-jepa-system1" if IN_COLAB else "../..")).resolve()
if IN_COLAB:
    if REPO_GIT:
        if not REPO.exists():
            run(["git", "clone", "--depth", "1", REPO_GIT, str(REPO)])
        else:
            run(["git", "-C", str(REPO), "pull", "--ff-only"])
    else:                                  # always re-extract, so a newer zip on Drive takes effect
        if not os.path.exists(REPO_ZIP):
            raise FileNotFoundError(f"upload grounded-jepa-system1.zip to {REPO_ZIP} (or set REPO_GIT)")
        run(["unzip", "-q", "-o", REPO_ZIP, "-d", str(REPO.parent)])
if importlib.util.find_spec("hybrid_wm") is None:
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(REPO)])
    sys.path.insert(0, str(REPO / "src"))
SUITE = REPO / "bench" / "lewm_suite"
print("suite:", SUITE)
'''

ENV = r'''
# ---- environment + checks: unit tests, MuJoCo rendering (EGL, falling back to OSMesa), CUDA
os.environ.update({"STABLEWM_HOME": str(HOME), "HWM_IMG_SIZE": str(IMG_SIZE), "HDF5_USE_FILE_LOCKING": "FALSE",
                   "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl"), "PYOPENGL_PLATFORM": os.environ.get("MUJOCO_GL", "egl")})
run([sys.executable, "-m", "pytest", "-q", "test_bench_units.py"], cwd=SUITE)
render_check = """
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import stable_worldmodel as swm
from tasks import get_task, apply_render_profile
t = get_task('cube', int(os.environ['HWM_IMG_SIZE'])); apply_render_profile(t)
w = swm.World(t.env_id, num_envs=1, image_shape=(t.eval_kwargs['width'],) * 2, max_episode_steps=10, **t.collect_kwargs)
w.reset(seed=0)
print('render ok:', w.infos['pixels'].shape, flush=True)
w.close()
os._exit(0)   # skip MuJoCo's harmless EGL teardown messages at interpreter exit
"""
if run([sys.executable, "-c", render_check], cwd=SUITE, check=False) != 0:
    print("EGL rendering failed; switching to OSMesa (software, slower)")
    run("apt-get -qq install -y libosmesa6-dev > /dev/null")
    os.environ.update({"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"})
    run([sys.executable, "-c", render_check], cwd=SUITE)
'''

PREFLIGHT = r'''
# ---- preflight: the whole pipeline at toy scale on THIS machine (catches GPU-only problems in minutes)
if RUN_PREFLIGHT:
    smoke_home = pathlib.Path("/content/smoke_home" if IN_COLAB else "./smoke_home").resolve()
    if smoke_home.exists():
        shutil.rmtree(smoke_home)
    env = dict(os.environ, STABLEWM_HOME=str(smoke_home))
    run([sys.executable, "colab_plan.py", "--profile", "smoke", "--tiers", "1", "--configs", "C7",
         "--home", str(smoke_home), "--workers", str(min(workers, 4)), "--gpu-slots", "1", "--run"], cwd=SUITE, env=env)
    out = subprocess.run([sys.executable, "colab_plan.py", "--profile", "smoke", "--tiers", "1", "--configs", "C7",
                          "--home", str(smoke_home), "--list"], cwd=SUITE, env=env, capture_output=True, text=True).stdout
    pending = [l for l in out.splitlines() if l.startswith("pending")]
    if pending:
        print("\n".join(pending))
        for l in pending:
            log = smoke_home / "logs" / "plan" / (l.split()[-1].replace(":", "_") + ".log")
            if log.exists():
                print(f"--- {log.name} (last lines)\n" + "\n".join(log.read_text().splitlines()[-15:]))
        raise RuntimeError("preflight failed: fix the jobs above before the long run")
    print("preflight passed: every job type runs on this machine")
'''

DATA = r'''
# ---- data: authors' expert collector in parallel processes, merged; backed up to Drive.
# Collection runs in parts of 25 episodes (~20 min each on 2 vCPUs). Each finished part is copied to
# Drive at once, so after a disconnect re-running this cell restores the finished parts and only
# collects the missing ones. The free profile needs ~4 h of collection in total.
plan = [sys.executable, "colab_plan.py", "--profile", PROFILE, "--home", str(HOME), "--workers", str(workers)]
PARTS = BACKUP / "parts"
# collection only renders (no evaluation memory), so it can use one process per CPU
data_plan = [sys.executable, "colab_plan.py", "--profile", PROFILE, "--home", str(HOME), "--workers", str(max(workers, cpus)),
             "--backup-dir", str(PARTS)]
run(data_plan + ["--data-only", "--run"], cwd=SUITE)
for name in ("cube.h5", "cube_eval.h5"):
    src, dst = HOME / "datasets" / name, BACKUP / name
    if src.exists() and not dst.exists():
        print("backing up", name, f"({src.stat().st_size / 2**30:.1f} GB)")
        shutil.copy2(src, dst.with_name(dst.name + ".tmp")); dst.with_name(dst.name + ".tmp").replace(dst)
    if dst.exists():                                     # merged copy is safe: the parts are no longer needed
        for part in PARTS.glob(name.removesuffix(".h5") + "_p*.h5"):
            part.unlink()
'''

MAIN = r'''
# ---- the study: training (GPU slots) + caches, policies and evaluations (worker slots). Re-run to resume.
args = ["--tiers", *map(str, TIERS), "--gpu-slots", str(gpu_slots), "--env-chunk", str(env_chunk)]
if CONFIGS:
    args += ["--configs", *CONFIGS]
run(plan + args + ["--list"], cwd=SUITE)
rc = run(plan + args + ["--run"], cwd=SUITE, check=False)   # failures are listed at the end; re-run to retry
'''

ANALYSIS = r'''
# ---- analysis: success over seeds, paired contrasts (exact McNemar, Holm), diagnostics, figures
from IPython.display import Markdown, Image, display
run([sys.executable, "analyze_colab.py", "--profile", PROFILE, "--home", str(HOME)], cwd=SUITE)
out = HOME / "results" / "colab"
display(Markdown((out / "summary.md").read_text()))
for f in sorted(out.glob("fig_*.png")):
    display(Image(filename=str(f)))
'''

EXPORT = r'''
# ---- export: one zip on Drive with every result JSON, the summary, figures and training logs
import zipfile, time
zpath = DRIVE / f"gpu_study_{tag}_{PROFILE}_{time.strftime('%Y%m%d-%H%M')}.zip"
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for f in (HOME / "results").rglob("*"):
        if f.is_file():
            z.write(f, f.relative_to(HOME))
    for f in (HOME / "checkpoints").rglob("train_info.json"):
        z.write(f, f.relative_to(HOME))
print("wrote", zpath, f"({zpath.stat().st_size / 2**20:.1f} MB)")
'''

TASK2_INTRO = r"""
## 12. Second environment: TwoRoom (optional)

Do the main findings hold beyond Cube? These cells run the same pipeline on a second environment.
They use the same Drive folders; files are named by environment, so nothing from the Cube study is touched.
Run them in their own session, after the Cube tiers (in a new session: run sections 1–6, then section 12 with `RUN_TASK2 = True`).

| environment | what it is | data | configurations | free Colab |
|---|---|---|---|---|
| `tworoom` | a point agent navigating between two rooms through a door (no object) | our expert collector (a few minutes) | C1 LeWM, C2 proprio grounding (object grounding does not apply) | fits: 2 models, less than a Cube tier |

What is evaluated (same code as Cube): System 1 (Gaussian and flow), two-level System 1, best-of-16 selection, CEM.
Grasp windows and camera dropout are Cube-only and are skipped.
Each environment is also evaluated on the authors' own protocol (every start/goal pair, goal +25, budget 50: `s1_all`, `cem_all`), so the numbers can be set beside the published LeWM table.
Our frames are 64 px; LeWM used 224 px, so a gap to their numbers is expected and is not a like-for-like comparison.

"""

TASK2_SETTINGS = r'''
# ---- second environment (edit)
RUN_TASK2 = False          # True to run section 12
TASK2 = "tworoom"
TASK2_TIERS = [1]          # same tiers as Cube: [1] = seed 0; then [2], [3] for more seeds
plan2 = [sys.executable, "colab_plan.py", "--task", TASK2, "--profile", PROFILE, "--home", str(HOME)]
'''

TASK2_DATA = r'''
# ---- data for the second environment (TwoRoom: our collector in parts)
if RUN_TASK2:
    if RUN_PREFLIGHT:                              # toy-scale run of every job type
        smoke2 = pathlib.Path("/content/smoke_tworoom" if IN_COLAB else "./smoke_tworoom").resolve()
        shutil.rmtree(smoke2, ignore_errors=True)
        run([sys.executable, "colab_plan.py", "--task", "tworoom", "--profile", "smoke", "--tiers", "1", "--configs", "C2",
             "--home", str(smoke2), "--workers", str(min(workers, 4)), "--run"], cwd=SUITE,
            env=dict(os.environ, STABLEWM_HOME=str(smoke2)))
    for name in (f"{TASK2}.h5", f"{TASK2}_eval.h5"):              # restore from an earlier session
        b = BACKUP / name
        if b.exists() and not (HOME / "datasets" / name).exists():
            print("restoring", name); shutil.copy2(b, HOME / "datasets" / name)
    run(plan2 + ["--workers", str(max(workers, cpus)), "--backup-dir", str(BACKUP / "parts"), "--data-only", "--run"],
        cwd=SUITE)
    missing = [n for n in (f"{TASK2}.h5", f"{TASK2}_eval.h5") if not (HOME / "datasets" / n).exists()]
    if missing:
        for log in sorted((HOME / "logs" / "plan").glob(f"*{TASK2}*.log")):
            print(f"--- {log.name} (last lines)\n" + "\n".join(log.read_text().splitlines()[-6:]))
        raise RuntimeError(f"{TASK2} data not ready ({missing}); see the log lines above "
                           "")
    for name in (f"{TASK2}.h5", f"{TASK2}_eval.h5"):              # back up the merged data, drop the parts
        src, dst = HOME / "datasets" / name, BACKUP / name
        if not dst.exists():
            print("backing up", name, f"({src.stat().st_size / 2**30:.1f} GB)")
            shutil.copy2(src, dst.with_name(dst.name + ".tmp")); dst.with_name(dst.name + ".tmp").replace(dst)
        for part in (BACKUP / "parts").glob(name.removesuffix(".h5") + "_p*.h5"):
            part.unlink()
'''

TASK2_MAIN = r'''
# ---- the study on the second environment. Re-run to resume.
if RUN_TASK2:
    args2 = ["--tiers", *map(str, TASK2_TIERS), "--gpu-slots", str(gpu_slots), "--env-chunk", str(env_chunk),
             "--workers", str(workers)]
    run(plan2 + args2 + ["--list"], cwd=SUITE)
    run(plan2 + args2 + ["--run"], cwd=SUITE, check=False)
'''

TASK2_ANALYSIS = r'''
# ---- analysis and export for the second environment
if RUN_TASK2:
    import zipfile, time
    from IPython.display import Markdown, Image, display
    run([sys.executable, "analyze_colab.py", "--task", TASK2, "--profile", PROFILE, "--home", str(HOME)], cwd=SUITE)
    out2 = HOME / "results" / f"colab_{TASK2}"
    display(Markdown((out2 / "summary.md").read_text()))
    for f in sorted(out2.glob("fig_*.png")):
        display(Image(filename=str(f)))
    zpath = DRIVE / f"gpu_study_{TASK2}_{tag}_{PROFILE}_{time.strftime('%Y%m%d-%H%M')}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in (HOME / "results").glob(f"{TASK2}__*.json"):
            z.write(f, f.relative_to(HOME))
        for f in out2.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(HOME))
        for f in (HOME / "checkpoints" / TASK2).rglob("train_info.json"):
            z.write(f, f.relative_to(HOME))
    print("wrote", zpath, f"({zpath.stat().st_size / 2**20:.1f} MB)")
'''


APPENDIX_INTRO = r"""
## 13. Authors' standard protocol (Appendix A, optional)

This scores every Cube model on the authors' own protocol (every start/goal pair, for Paper 1's appendix): System 1 for C1, C2, C3, C5 and CEM for C1, C5.
Run it in its own session: sections 1–6, then this one with `RUN_APPENDIX = True`.
After it finishes, the cell lists the collection parts that can be deleted from Drive. Keep `cube.h5`: every evaluation z-scores actions with statistics of the training data.
"""

APPENDIX = r'''
RUN_APPENDIX = False
if RUN_APPENDIX:
    appx = [sys.executable, "colab_plan.py", "--profile", PROFILE, "--home", str(HOME), "--tiers", "1", "2", "3",
           "--configs", "C1", "C2", "C3", "C5", "--extra", "appendix", "--only-extra", "--workers", str(workers),
           "--gpu-slots", str(gpu_slots), "--env-chunk", str(env_chunk)]
    for name in ("cube.h5", "cube_eval.h5"):                      # the Cube data, from the Drive backup
        b = BACKUP / name
        if b.exists() and not (HOME / "datasets" / name).exists():
            print("restoring", name); shutil.copy2(b, HOME / "datasets" / name)
    run(appx + ["--list"], cwd=SUITE)
    run(appx + ["--run"], cwd=SUITE, check=False)
    run([sys.executable, "analyze_colab.py", "--profile", PROFILE, "--home", str(HOME)], cwd=SUITE)   # + appendix columns
    left = subprocess.run(appx + ["--list"], cwd=SUITE, capture_output=True, text=True).stdout
    if any(l.startswith("pending") for l in left.splitlines()):
        print("Some jobs are not finished yet: re-run this cell before deleting any Cube data.")
    else:
        big = [f for f in (BACKUP / "parts",) if f.exists()]
        print("All Cube runs are done. To free Drive space you can delete the collection parts:")
        for f in big:
            size = sum(x.stat().st_size for x in f.rglob("*")) if f.is_dir() else f.stat().st_size
            print(f"  {f}  ({size / 2**30:.2f} GB)")
        print("Keep cube.h5 (evaluations need its action statistics), cube_eval.h5, checkpoints/ and results/.")
'''


CEM_ALL_INTRO = r"""
## 14. CEM on every Cube configuration (follow-up, optional)

The main run planned with CEM only on C1 and C5. This adds CEM on C2 and C3 (three seeds) and on C6 and C7 (seed 0)
with the same settings, so every configuration has a planning baseline. It evaluates the existing models only.

Evaluations z-score actions with statistics of the **training data** (`cube.h5`). If that file is gone from Drive,
this cell first collects it again with the same collector and seeds (24 parts of 25 episodes, about 4 h; each
finished part is saved to Drive, so after a disconnect re-run the cell and it resumes). The collector does not
reproduce the same episodes from a seed, so the new file is a second sample of the same distribution. To measure
what that changes, the cell re-runs the stored CEM evaluation of C1, seed 0, and compares the two results pair by
pair before running the new evaluations.

Run it in its own session: sections 1–6, then this one with `RUN_CEM_ALL = True`. At the end it saves a zip of the
new result files and the check on Drive; send that zip back.
"""

CEM_ALL = r'''
RUN_CEM_ALL = False
if RUN_CEM_ALL:
    import zipfile, time
    for name in ("cube_eval.h5", "cube.h5"):                     # restore from the Drive backup when present
        b = BACKUP / name
        if b.exists() and not (HOME / "datasets" / name).exists():
            print("restoring", name); shutil.copy2(b, HOME / "datasets" / name)
    if not (HOME / "datasets" / "cube_eval.h5").exists():
        raise RuntimeError("cube_eval.h5 is missing from Drive: the evaluation pairs cannot be rebuilt identically")
    recollected = BACKUP / "cube.h5.RECOLLECTED.txt"
    if not (HOME / "datasets" / "cube.h5").exists():             # training data deleted: collect it again
        PARTS = BACKUP / "parts"
        data_plan = [sys.executable, "colab_plan.py", "--profile", PROFILE, "--home", str(HOME),
                     "--workers", str(max(workers, cpus)), "--backup-dir", str(PARTS)]
        run(data_plan + ["--data-only", "--run"], cwd=SUITE)
        src, dst = HOME / "datasets" / "cube.h5", BACKUP / "cube.h5"
        if not src.exists():
            raise RuntimeError("collection not finished: re-run this cell to resume")
        shutil.copy2(src, dst.with_name(dst.name + ".tmp")); dst.with_name(dst.name + ".tmp").replace(dst)
        recollected.write_text(f"cube.h5 collected again on {time.strftime('%Y-%m-%d')} (first copy deleted); "
                               "same collector and seeds, not the same episodes\n")
        for part in PARTS.glob("cube_p*.h5"):
            part.unlink()
    check = HOME / "results" / "checks" / "eval__C1s0__cem.json"
    if recollected.exists() and not check.exists():              # what the new training data changes
        run([sys.executable, "rerun_check.py", "--profile", PROFILE, "--home", str(HOME), "--job", "eval:C1s0:cem"],
            cwd=SUITE)
    cem = [sys.executable, "colab_plan.py", "--profile", PROFILE, "--home", str(HOME), "--tiers", "1", "2", "3",
           "--configs", "C2", "C3", "C6", "C7", "--extra", "cem_all", "--only-extra", "--workers", str(workers),
           "--gpu-slots", str(gpu_slots), "--env-chunk", str(env_chunk)]
    plan = subprocess.run(cem + ["--list"], cwd=SUITE, capture_output=True, text=True).stdout
    print(plan)
    missing = [l.split()[-1] for l in plan.splitlines() if l.startswith("pending") and not l.endswith(":cem")]
    if missing:
        raise RuntimeError(f"not evaluating: these would have to be trained or collected again: {missing}. "
                           "Check that Drive holds the checkpoints of the main run.")
    run(cem + ["--run"], cwd=SUITE, check=False)
    def is_cem(f):                                                # CEM result files carry controller "cem"
        try:
            return json.loads(f.read_text()).get("controller") == "cem"
        except (ValueError, OSError):
            return False
    new = sorted(f for f in (HOME / "results").glob("cube__*.json") if is_cem(f))
    new += sorted((HOME / "results" / "checks").glob("*.json")) + ([recollected] if recollected.exists() else [])
    zpath = DRIVE / f"gpu_study_cem_all_{tag}_{time.strftime('%Y%m%d-%H%M')}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in new:
            z.write(f, f.name if f == recollected else f.relative_to(HOME))
    print("wrote", zpath, f"with {len(new)} files")
'''


OUTRO = r"""
## Reading the results

- **Success by configuration**: mean ± SE over seeds, with the number of seeds in parentheses.
- **Paired contrasts**: A − B in points, the discordant pairs, and the exact McNemar p-value on the pairs pooled over seeds.
  - Bold rows are the pre-registered family. A hypothesis is supported when its row keeps p < 0.05 after Holm correction.
  - The ARL-selection rows (H3) use a Wilcoxon test on the per-pair change in selection gain.
  - Other rows are exploratory: report them as such.
- **Diagnostics**: perturbed-chunk identifiability. Chance is 0.06; the CPU study measured 0.02–0.03 at σ = 0.1. The cube probe is read against the 4 cm success tolerance.

"""


def build(path: Path) -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata = {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}}
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb.cells = [md(INTRO.strip()), md("## 1. Settings"), code(SETTINGS.strip()),
                md("## 2. Runtime"), code(RUNTIME.strip()),
                md("## 3. Storage (Google Drive)"), code(DRIVE.strip()),
                md("## 4. Dependencies"), code(INSTALL.strip()),
                md("## 5. Code"), code(CODE.strip()),
                md("## 6. Checks"), code(ENV.strip()),
                md("## 7. Preflight (toy scale, all job types)"), code(PREFLIGHT.strip()),
                md("## 8. Data"), code(DATA.strip()),
                md("## 9. Run the study"), code(MAIN.strip()),
                md("## 10. Analysis"), code(ANALYSIS.strip()),
                md("## 11. Export"), code(EXPORT.strip()),
                md(TASK2_INTRO.strip()), code(TASK2_SETTINGS.strip()),
                md("### 12a. Data (second environment)"), code(TASK2_DATA.strip()),
                md("### 12b. Run (second environment)"), code(TASK2_MAIN.strip()),
                md("### 12c. Analysis and export (second environment)"), code(TASK2_ANALYSIS.strip()),
                md(APPENDIX_INTRO.strip()), code(APPENDIX.strip()),
                md(CEM_ALL_INTRO.strip()), code(CEM_ALL.strip()),
                md(OUTRO.strip())]
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)
    print("wrote", path)


if __name__ == "__main__":
    build(Path(__file__).resolve().parents[2] / "notebooks" / "hybrid_wm_gpu_study.ipynb")
