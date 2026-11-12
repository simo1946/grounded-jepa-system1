#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
export MUJOCO_GL=${MUJOCO_GL:-egl}
O=lewm+idm+sa+obj; CK=swm_home/checkpoints/cube/$O/seed0
[ -f $CK/policy_awr.pt ] || python awr_policy.py --spec $O --init policy_gauss6k.pt --out policy_awr.pt
[ -f $CK/policy_awrctrl.pt ] || python awr_policy.py --spec $O --init policy_gauss6k.pt --out policy_awrctrl.pt --uniform
for p in policy_awr policy_awrctrl; do
  python evaluate.py --task cube --spec $O --seed 0 --pairs nontrivial --controller s1 --k 1 --policy-file $p.pt --skip-existing
done
echo "[$(date +%H:%M:%S)] ALL DONE"
