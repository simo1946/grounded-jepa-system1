#!/usr/bin/env bash
# Addendum of prereg_feedback.md (2026-11-09): the authors' GC-IDM code (H_max 50 and their H_max 1 ablation)
# and System 1 trained for 5- and 1-step chunks, on the four CPU-study world models. Needs GCIDM_REPO (a clone of
# the authors' repository at the pinned commit, see official_gcidm.py). Safe to re-run: finished work is skipped.
# Usage: run_feedback2.sh ["spec seed" ...]   (default: the four models)
set -euo pipefail
cd "$(dirname "$0")"
: "${STABLEWM_HOME:?set STABLEWM_HOME}" "${GCIDM_REPO:?set GCIDM_REPO}"
DEV=${DEV:-cpu}
EVAL="--task cube --pairs nontrivial --num-eval 60 --goal-offset 25 --budget 50 --eval-seed 42 --device $DEV --skip-existing"
models=("$@"); [ ${#models[@]} -gt 0 ] || models=("lewm 0" "lewm+idm+sa+obj 0" "lewm 1" "lewm+idm+sa+obj 1")
for model in "${models[@]}"; do
  set -- $model; spec=$1; seed=$2
  dir="$STABLEWM_HOME/checkpoints/cube/$spec/seed$seed"
  npz="$STABLEWM_HOME/cache/gcidm_official/$(echo "$spec" | tr '+' '_')__seed${seed}__cube.npz"
  [ -f "$npz" ] || python3 official_gcidm.py export --spec "$spec" --seed "$seed"
  [ -f "$dir/official_gcidm.pt.done" ] || { python3 official_gcidm.py train --spec "$spec" --seed "$seed" \
      --repo "$GCIDM_REPO" && touch "$dir/official_gcidm.pt.done"; }
  [ -f "$dir/official_gcidm_h1.pt.done" ] || { python3 official_gcidm.py train --spec "$spec" --seed "$seed" \
      --repo "$GCIDM_REPO" --max-goal-horizon 1 && touch "$dir/official_gcidm_h1.pt.done"; }
  for c in 5 1; do
    [ -f "$dir/policy_c$c.pt" ] || python3 short_chunk.py --spec "$spec" --seed "$seed" --chunk-steps $c
  done
  M="--spec $spec --seed $seed --receding 1"
  python3 evaluate.py $EVAL $M --controller gcidm_official --policy-file official_gcidm.pt --exec-steps 1
  python3 evaluate.py $EVAL $M --controller gcidm_official --policy-file official_gcidm_h1.pt --exec-steps 1
  python3 evaluate.py $EVAL $M --controller s1short --policy-file policy_c5.pt --exec-steps 5
  python3 evaluate.py $EVAL $M --controller s1short --policy-file policy_c1.pt --exec-steps 1
done
