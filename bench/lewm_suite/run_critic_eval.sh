#!/usr/bin/env bash
# E6 (prereg_critic.md): train the outcome critic, then evaluate selection by critic, true outcome, random.
set -u
cd "$(dirname "$0")"
export MUJOCO_GL=${MUJOCO_GL:-egl}
O=lewm+idm+sa+obj
until grep -q "ALL DONE" logs/run_critic_data.log 2>/dev/null; do sleep 20; done
[ -f swm_home/checkpoints/cube/$O/seed0/critic.pt ] || python critic.py --spec $O
EV="python evaluate.py --task cube --spec $O --seed 0 --pairs nontrivial --controller s1 --k 16 --policy-file policy_gauss6k.pt --skip-existing"
$EV --select oracle
$EV --select critic
$EV --select random
echo "[$(date +%H:%M:%S)] ALL DONE"
