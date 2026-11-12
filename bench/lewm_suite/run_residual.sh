#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
export MUJOCO_GL=${MUJOCO_GL:-egl}
O=lewm+idm+sa+obj; CK=swm_home/checkpoints/cube/$O/seed0
for r in real latent; do
  [ -f $CK/policy_res_$r.pt ] || python residual_rl.py --spec $O --base policy_gauss6k.pt --reward $r --out policy_res_$r.pt
done
for r in real latent; do
  python evaluate.py --task cube --spec $O --seed 0 --pairs nontrivial --controller s1 --k 1 --policy-file policy_res_$r.pt --skip-existing
done
echo "[$(date +%H:%M:%S)] ALL DONE"
