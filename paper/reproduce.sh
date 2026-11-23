#!/usr/bin/env bash
# Regenerate everything the paper prints, from the committed result files, and audit it.
#
#   pip install "osaudit @ git+https://github.com/ASSERT-KTH/osaudit@79e3bb737266e3162f66e4f213a4e985975bd5fb"   # Python >= 3.13
#   paper/reproduce.sh
#
# 1. Each results file in results/paper/ is written by `osaudit run`, which records the command, the sha256 of
#    every input and the git commit of the code. The instruments declared in the committed osaudit.toml must be
#    unchanged since the last commit; results files may be regenerated in the same session.
# 2. Figures 2 and 3 are drawn from results/paper/gpu_cube.json.
# 3. paper/build.py writes paper/paper.docx, paper/paper.pdf and osaudit.toml (one entry per printed number).
# 4. osaudit checks every printed number against the results files and every results file against the code.
#
# Inputs from outside this repository go in external/ (git-ignored), so the provenance records repository-relative
# paths:
#   external/gcidm_repo      the GC-IDM authors' code at commit 48c45b1 (model_sizes.json):
#     git clone https://github.com/hdnndh/Latent-Geometry-Beyond-Search-Amortizing-Planning-in-World-Models external/gcidm_repo
#     git -C external/gcidm_repo checkout 48c45b1cb2b34dd2c1c61d222c8309de567fde55
#   external/cube_data       cube.h5 and cube_eval.h5, the regenerated Cube data (trivial_share_cpu.json); copy or
#                            hard-link them from $STABLEWM_HOME/datasets
# Steps whose inputs are missing are skipped and keep the committed file.
set -euo pipefail
cd "$(dirname "$0")/.."
OSA=${OSAUDIT:-osaudit}
S=bench/lewm_suite
R=results/paper
mkdir -p "$R"

"$OSA" run --input results/gpu_study/runs -o "$R/gpu_cube.json" -- \
  python3 "$S/paper_values_gpu.py" --runs results/gpu_study/runs --task cube
"$OSA" run --input results/gpu_study/runs -o "$R/gpu_tworoom.json" -- \
  python3 "$S/paper_values_gpu.py" --runs results/gpu_study/runs --task tworoom
"$OSA" run --input results/feedback_study/runs -o "$R/feedback.json" -- \
  python3 "$S/analyze_feedback.py" --runs results/feedback_study/runs --json
if [ -d results/cpu_rerun ]; then
  "$OSA" run --input results/cpu_rerun -o "$R/cpu_rerun.json" -- python3 "$S/analyze_rerun.py" --runs results/cpu_rerun
fi
GCIDM=external/gcidm_repo
if [ -f "$GCIDM/idm/model.py" ]; then
  "$OSA" run --input "$S/models.py" --input "$GCIDM/idm/model.py" -o "$R/model_sizes.json" -- \
    python3 "$S/model_sizes.py" --gcidm-repo "$GCIDM"
fi
CUBE=external/cube_data
if [ -f "$CUBE/cube.h5" ] && [ -f "$CUBE/cube_eval.h5" ]; then
  "$OSA" run --input "$CUBE/cube.h5" --input "$CUBE/cube_eval.h5" -o "$R/trivial_share_cpu.json" -- \
    python3 "$S/trivial_share.py" "$CUBE/cube.h5" "$CUBE/cube_eval.h5"
fi

python3 "$S/paper_figures.py" "$R/gpu_cube.json" paper/figures
python3 paper/build.py
"$OSA"
