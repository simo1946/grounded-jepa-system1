#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
export MUJOCO_GL=${MUJOCO_GL:-egl}
G="python gen_critic_data.py --spec lewm+idm+sa+obj --policy-file policy_gauss6k.pt"
$G --data cube --n 1500 --part 0 > logs/critic_data_p0.log 2>&1 &
$G --data cube --n 1500 --part 1 > logs/critic_data_p1.log 2>&1 &
wait
$G --data cube_eval --n 300 --part 0 > logs/critic_data_eval.log 2>&1
echo "[$(date +%H:%M:%S)] ALL DONE"
