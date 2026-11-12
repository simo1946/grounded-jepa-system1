#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
export MUJOCO_GL=${MUJOCO_GL:-egl}
O=lewm+idm+sa+obj; CK=swm_home/checkpoints/cube/$O/seed0
for r in shaped sparse; do
  [ -f $CK/policy_prac_$r.pt ] || python practice_rl.py --reward $r --out policy_prac_$r.pt
done
for r in shaped sparse; do
  python evaluate.py --task cube --spec $O --seed 0 --pairs nontrivial --controller s1 --k 1 --policy-file policy_prac_$r.pt --skip-existing
done
echo "[$(date +%H:%M:%S)] ALL DONE"
