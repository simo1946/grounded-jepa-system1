#!/usr/bin/env bash
# Exploratory: is the flow-vs-Gaussian result sensitive to the policy step budget (Colab free: 3,000; CPU: 6,000)?
set -u
cd "$(dirname "$0")"
export MUJOCO_GL=${MUJOCO_GL:-egl}
O=lewm+idm+sa+obj; CK=swm_home/checkpoints/cube/$O/seed0
for t in gauss flow; do
  [ -f "$CK/policy_${t}3k.pt" ] || python cached_policy.py --spec $O --seed 0 --out policy_${t}3k.pt --steps 3000 --batch 256 --policy-seed 0 --policy-type $t --device cpu
done
for t in gauss flow; do
  python evaluate.py --task cube --spec $O --seed 0 --pairs nontrivial --controller s1 --k 1 --policy-file policy_${t}3k.pt --skip-existing
done
echo "[$(date +%H:%M:%S)] ALL DONE"
