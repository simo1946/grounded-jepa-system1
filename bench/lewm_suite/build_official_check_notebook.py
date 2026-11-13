"""Build notebooks/official_cube_check.ipynb: the LeWM authors' Cube dataset, checked on Colab.

What it runs (official_cube_check.py): share of start/goal pairs already solved at the start in the authors'
released dataset (quentinll/lewm-cube), for all pairs and for the 50 pairs their eval.py draws; then their
eval.py, unmodified at a pinned commit, with its default random policy. Needs about 60-100 GB of free disk
(a standard Colab CPU or GPU runtime); no GPU is needed.

    python build_official_check_notebook.py
"""
from __future__ import annotations

from pathlib import Path

import nbformat

DRIVE_DIR = "/content/drive/MyDrive/hybrid-wm"
SWM_COMMIT = "21446f1ede6d5284e981bd7b47f432b994e6d812"     # as the GPU study (build_notebook.py)
LEWM_COMMIT = "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"

INTRO = """# Already-solved pairs in the official LeWM Cube dataset

1. Upload `grounded-jepa-system1.zip` to `MyDrive/hybrid-wm/` (as for the GPU study), replacing the old one.
2. Runtime: the T4 GPU runtime, as for the GPU study (its rendering setup is known to work). Run all cells.
   The download is about 46 GB and is streamed: only the extracted `.h5` file is written.
3. The result is saved to `MyDrive/hybrid-wm/official_cube_check.json` as soon as the already-solved share is
   computed, and again after the authors' random-policy evaluation, which runs 50 environments in parallel and
   needs about 9 GB of RAM. If that step runs out of memory, the file records the error and still holds the share.
   Send the file back either way.
"""

SETUP = rf'''
import os, subprocess, sys, shutil, pathlib
from google.colab import drive
drive.mount("/content/drive")
DRIVE = pathlib.Path("{DRIVE_DIR}")
run = lambda cmd, **kw: subprocess.run(cmd, check=True, **kw)
run(["unzip", "-q", "-o", str(DRIVE / "grounded-jepa-system1.zip"), "-d", "/content"])
SUITE = pathlib.Path("/content/grounded-jepa-system1/bench/lewm_suite")
pip = [sys.executable, "-m", "pip", "install", "-q"]
run(pip + ["git+https://github.com/galilai-group/stable-worldmodel@{SWM_COMMIT}"])
run(pip + ["stable-pretraining==0.1.8", "mujoco==3.15.0", "h5py", "hdf5plugin", "zstandard", "huggingface_hub",
           "hydra-core", "scikit-learn", "imageio[ffmpeg]", "pygame", "pymunk", "shapely", "dm-env", "dm-tree",
           "absl-py", "lxml", "PyOpenGL", "glfw"])
run(pip + ["--no-deps", "ogbench==1.2.1", "dm_control==1.0.48"])
if not pathlib.Path("/content/le-wm").exists():
    run(["git", "clone", "https://github.com/lucas-maes/le-wm", "/content/le-wm"])
run(["git", "-C", "/content/le-wm", "checkout", "-q", "{LEWM_COMMIT}"])
print("free disk GB:", shutil.disk_usage("/content").free // 2**30,
      "| RAM GB:", os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 2**30)
'''

CHECK = r'''
os.environ["MUJOCO_GL"] = "egl"
out = DRIVE / "official_cube_check.json"
run([sys.executable, str(SUITE / "official_cube_check.py"), "--out", str(out), "--random-eval",
     "--lewm-repo", "/content/le-wm", "--data-dir", "/content/lewm_cube", "--home", "/content/swm_home_official"])
print(out.read_text()[:1500])
'''


def build(path: Path) -> Path:
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_markdown_cell(INTRO), nbformat.v4.new_code_cell(SETUP.strip()),
                nbformat.v4.new_code_cell(CHECK.strip())]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, path)
    return path


if __name__ == "__main__":
    print(build(Path(__file__).resolve().parents[2] / "notebooks" / "official_cube_check.ipynb"))
