#!/usr/bin/env bash
# E5 (prereg_overnight.md + addendum): clean oracle labels, then DAgger vs equal-steps control.
# Grounded model, Gaussian System 1. Strictly sequential, skip-if-done.
set -u
cd "$(dirname "$0")"
export MUJOCO_GL=${MUJOCO_GL:-egl}
log() { echo "[$(date +%H:%M:%S)] $*"; }
S=lewm+idm+sa
CK=swm_home/checkpoints/cube/$S/seed0
EV="python evaluate.py --task cube --seed 0 --pairs nontrivial --controller s1 --k 1 --skip-existing"
train() { [ -f "$CK/$1" ] || python oracle_policy.py --spec $S --out "$@"; }

log "wait for the E3 queue"
while pgrep -f run_overnight.sh > /dev/null; do sleep 30; done

log "A1: clean oracle labels (Gaussian and flow), 6000 steps"
train policy_oracle6k.pt --steps 6000 --policy-type gauss
train policy_oracleflow6k.pt --steps 6000 --policy-type flow
$EV --spec $S --policy-file policy_oracle6k.pt
$EV --spec $S --policy-file policy_oracleflow6k.pt
[ -f "swm_home/checkpoints/cube/$S+obj/seed0/policy_oracle6k.pt" ] || \
  python oracle_policy.py --spec $S+obj --out policy_oracle6k.pt --steps 6000 --policy-type gauss
$EV --spec $S+obj --policy-file policy_oracle6k.pt

log "DAgger round 1"
[ -f "$CK/dagger/round1.pt" ] || python dagger.py --spec $S --policy-file policy_oracle6k.pt --out round1.pt --episodes 100 --round 1
train policy_dagger_r1.pt --init policy_oracle6k.pt --steps 2000 --dagger $CK/dagger/round1.pt --mix 0.5
train policy_octrl_r1.pt --init policy_oracle6k.pt --steps 2000
log "DAgger round 2"
[ -f "$CK/dagger/round2.pt" ] || python dagger.py --spec $S --policy-file policy_dagger_r1.pt --out round2.pt --episodes 100 --round 2
train policy_dagger_r2.pt --init policy_dagger_r1.pt --steps 2000 --dagger $CK/dagger/round1.pt $CK/dagger/round2.pt --mix 0.5
train policy_octrl_r2.pt --init policy_octrl_r1.pt --steps 2000

log "evaluate"
for p in policy_dagger_r1 policy_octrl_r1 policy_dagger_r2 policy_octrl_r2; do $EV --spec $S --policy-file $p.pt; done
log "ALL DONE"
