#!/usr/bin/env bash
# E5 with simulated clean oracle chunks (prereg_overnight.md, addendum 2). Sequential, skip-if-done.
set -u
cd "$(dirname "$0")"
export MUJOCO_GL=${MUJOCO_GL:-egl}
log() { echo "[$(date +%H:%M:%S)] $*"; }
G=lewm+idm+sa; O=lewm+idm+sa+obj
CK=swm_home/checkpoints/cube/$G/seed0
EV="python evaluate.py --task cube --seed 0 --pairs nontrivial --controller s1 --k 1 --skip-existing"
tr() { local spec=$1 out=$2; shift 2
       [ -f "swm_home/checkpoints/cube/$spec/seed0/$out" ] || python oracle_policy.py --spec $spec --out $out --labels sim "$@"; }

log "wait for the oracle chunks"
# (pgrep -f can match the launching shell's own command text, so wait on the output files instead)
until [ -f swm_home/datasets/cube_oracle_chunks_p0.pt ] && [ -f swm_home/datasets/cube_oracle_chunks_p1.pt ]; do sleep 20; done

log "clean simulated labels: Gaussian and flow, both models"
for spec in $G $O; do
  tr $spec policy_osim6k.pt --steps 6000 --policy-type gauss
  tr $spec policy_osimflow6k.pt --steps 6000 --policy-type flow
done
for spec in $G $O; do
  $EV --spec $spec --policy-file policy_osim6k.pt
  $EV --spec $spec --policy-file policy_osimflow6k.pt
done

log "DAgger round 1"
[ -f "$CK/dagger/sround1.pt" ] || python dagger.py --spec $G --policy-file policy_osim6k.pt --out sround1.pt --episodes 100 --round 1
tr $G policy_sdagger_r1.pt --init policy_osim6k.pt --steps 2000 --dagger $CK/dagger/sround1.pt --mix 0.5
tr $G policy_sctrl_r1.pt --init policy_osim6k.pt --steps 2000
log "DAgger round 2"
[ -f "$CK/dagger/sround2.pt" ] || python dagger.py --spec $G --policy-file policy_sdagger_r1.pt --out sround2.pt --episodes 100 --round 2
tr $G policy_sdagger_r2.pt --init policy_sdagger_r1.pt --steps 2000 --dagger $CK/dagger/sround1.pt $CK/dagger/sround2.pt --mix 0.5
tr $G policy_sctrl_r2.pt --init policy_sctrl_r1.pt --steps 2000
log "evaluate"
for p in policy_sdagger_r1 policy_sctrl_r1 policy_sdagger_r2 policy_sctrl_r2; do $EV --spec $G --policy-file $p.pt; done
log "ALL DONE"
