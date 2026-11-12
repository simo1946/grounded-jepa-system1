#!/usr/bin/env bash
# Early CPU signal for the action-resolution loss (ARL), after the main queue (strictly sequential).
set -uo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl
until grep -q "ALL DONE" logs/run_next8b.log 2>/dev/null; do sleep 120; done
S=lewm+idm+sa+obj+arl
NT=(--pairs nontrivial --skip-existing)
[ -f swm_home/checkpoints/cube/$S/seed0/weights.pt ] || python3 train.py --task cube --spec $S --seed 0 --steps 2000
[ -f swm_home/checkpoints/cube/$S/seed0/policy.pt ] || python3 train_policy.py --task cube --spec $S --seed 0
python3 rollout_diag.py --spec $S --seed 0
python3 probe_diag.py --spec $S --seed 0
python3 evaluate.py --task cube "${NT[@]}" --spec $S --seed 0 --controller s1 --k 1
python3 evaluate.py --task cube "${NT[@]}" --spec $S --seed 0 --controller s1 --k 16
python3 evaluate.py --task cube "${NT[@]}" --spec lewm+idm+sa+obj --seed 0 --controller s1 --k 16
echo "ARL DONE"
