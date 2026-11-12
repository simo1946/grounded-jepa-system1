#!/usr/bin/env bash
# Feedback-frequency study (prereg_feedback.md): CPU-study world models C1 (lewm) and C3 (lewm+idm+sa+obj),
# seeds 0 and 1. Per model: latent cache, System 1 (GPU-study recipe), GC-IDM-style with and without the
# horizon input, then five evaluations on the same 60 non-trivial Cube pairs. Safe to re-run (skips work done).
set -euo pipefail
cd "$(dirname "$0")"
: "${STABLEWM_HOME:?set STABLEWM_HOME}"
DEV=${DEV:-cpu}
EVAL="--task cube --pairs nontrivial --num-eval 60 --goal-offset 25 --budget 50 --eval-seed 42 --device $DEV --skip-existing"
for model in "lewm 0" "lewm+idm+sa+obj 0" "lewm 1" "lewm+idm+sa+obj 1"; do
  set -- $model; spec=$1; seed=$2
  dir="$STABLEWM_HOME/checkpoints/cube/$spec/seed$seed"
  cache="$STABLEWM_HOME/cache/$(echo "$spec" | tr '+' '_')__seed${seed}__cube.pt"
  [ -f "$cache" ] || python3 latent_cache.py --spec "$spec" --seed "$seed" --data cube --device "$DEV"
  [ -f "$dir/policy_s1.pt" ] || python3 cached_policy.py --task cube --spec "$spec" --seed "$seed" \
      --out policy_s1.pt --steps 3000 --batch 256 --device "$DEV"
  [ -f "$dir/gcidm.pt.done" ] || { python3 gcidm.py --spec "$spec" --seed "$seed" && touch "$dir/gcidm.pt.done"; }
  [ -f "$dir/gcidm_noh.pt.done" ] || { python3 gcidm.py --spec "$spec" --seed "$seed" --no-horizon \
      && touch "$dir/gcidm_noh.pt.done"; }
  S1="--spec $spec --seed $seed --controller s1 --k 1 --policy-file policy_s1.pt"
  python3 evaluate.py $EVAL $S1 --receding 5 --exec-steps 25
  python3 evaluate.py $EVAL $S1 --receding 1 --exec-steps 5
  python3 evaluate.py $EVAL $S1 --receding 1 --exec-steps 1
  python3 evaluate.py $EVAL --spec "$spec" --seed "$seed" --controller gcidm --policy-file gcidm.pt --receding 1 --exec-steps 1
  python3 evaluate.py $EVAL --spec "$spec" --seed "$seed" --controller gcidm --policy-file gcidm_noh.pt --receding 1 --exec-steps 1
done
python3 analyze_feedback.py
