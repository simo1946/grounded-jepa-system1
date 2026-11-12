#!/usr/bin/env bash
# Regenerate everything the paper prints, from the committed result files, and audit it.
#
#   pip install osaudit==0.7.0         # Python >= 3.13
#   GCIDM_REPO=/path/to/gcidm_repo STABLEWM_HOME=/path/to/swm_home paper/reproduce.sh
#
# 1. Each results file in results/paper/ is written by `osaudit run`, which records the command, the sha256 of
#    every input and the git commit of the code (the code must be committed first; results may be regenerated).
# 2. Figures 2 and 3 are drawn from results/paper/gpu_cube.json.
# 3. paper/build.py writes paper/paper.docx, paper/paper.pdf and osaudit.toml (one entry per printed number).
# 4. osaudit checks every printed number against the results files and every results file against the code.
#
# GCIDM_REPO (the GC-IDM authors' code at commit 48c45b1) is needed only for model_sizes.json; STABLEWM_HOME (the
# regenerated Cube data) only for trivial_share_cpu.json. Steps whose inputs are missing are skipped and keep the
# committed file.
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
if [ -n "${GCIDM_REPO:-}" ] && [ -d "$GCIDM_REPO" ]; then
  "$OSA" run --input "$S/models.py" --input "$GCIDM_REPO/idm/model.py" -o "$R/model_sizes.json" -- \
    python3 "$S/model_sizes.py"
fi
if [ -n "${STABLEWM_HOME:-}" ] && [ -f "$STABLEWM_HOME/datasets/cube.h5" ]; then
  "$OSA" run --input "$STABLEWM_HOME/datasets/cube.h5" --input "$STABLEWM_HOME/datasets/cube_eval.h5" \
    -o "$R/trivial_share_cpu.json" -- \
    python3 "$S/trivial_share.py" "$STABLEWM_HOME/datasets/cube.h5" "$STABLEWM_HOME/datasets/cube_eval.h5"
fi

python3 "$S/paper_figures.py" "$R/gpu_cube.json" paper/figures
python3 paper/build.py
"$OSA"
