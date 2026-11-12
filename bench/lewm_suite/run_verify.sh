#!/usr/bin/env bash
# Verification-regime study (prereg_verification.md). Strictly sequential, skip-if-done.
set -u
cd "$(dirname "$0")"
export MUJOCO_GL=${MUJOCO_GL:-egl}
G="lewm+idm+sa"; O="lewm+idm+sa+obj"
EV="python evaluate.py --task cube --seed 0 --pairs nontrivial --controller s1 --skip-existing"
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "E1 sanity: std 0 selection must equal the policy mean"
$EV --spec $G --k 16 --std-scale 0

log "E1 spread sweep"
for s in 0.25 0.5 2 4; do
  for spec in $G $O; do $EV --spec $spec --k 16 --std-scale $s; done
done
for s in 0.25 0.5 2; do
  for spec in $G $O; do $EV --spec $spec --k 16 --std-scale $s --select random; done
done
$EV --spec $O --k 16 --select random

log "E2 pool: train members"
CK=swm_home/checkpoints/cube/$G/seed0
for p in 1 2 3 4; do
  [ -f "$CK/policy_p$p.pt" ] || python cached_policy.py --spec $G --seed 0 --out policy_p$p.pt --steps 1500 --policy-seed $p --device cpu
done
log "E2 pool: evaluate"
for p in 1 2 3 4; do $EV --spec $G --k 1 --policy-file policy_p$p.pt; done
$EV --spec $G --k 5 --policy-file policy_cached.pt --pool policy_p1.pt policy_p2.pt policy_p3.pt policy_p4.pt --pool-mode select
$EV --spec $G --k 1 --policy-file policy_cached.pt --pool policy_p1.pt policy_p2.pt policy_p3.pt policy_p4.pt --pool-mode mean
log "ALL DONE"
