#!/usr/bin/env bash
# Overnight queue (prereg_overnight.md). Strictly sequential, skip-if-done.
set -u
cd "$(dirname "$0")"
export MUJOCO_GL=${MUJOCO_GL:-egl}
log() { echo "[$(date +%H:%M:%S)] $*"; }
EV="python evaluate.py --task cube --seed 0 --pairs nontrivial --controller s1 --skip-existing"

log "wait for the object-model caches"
while pgrep -f "latent_cache.py --spec lewm\+idm\+sa\+obj" > /dev/null; do sleep 30; done
[ -f swm_home/cache/lewm_idm_sa_obj__seed0__cube_eval.pt ] || { log "object cache missing"; exit 1; }

log "E3: train equal-budget Gaussian and flow System 1"
for spec in lewm+idm+sa lewm+idm+sa+obj; do
  CK=swm_home/checkpoints/cube/$spec/seed0
  for t in gauss flow; do
    [ -f "$CK/policy_${t}6k.pt" ] || python cached_policy.py --spec $spec --seed 0 --out policy_${t}6k.pt \
      --steps 6000 --batch 256 --policy-seed 0 --policy-type $t --device cpu
  done
  python policy_fit.py --spec $spec --policies policy_gauss6k.pt policy_flow6k.pt
done

log "E3: evaluate"
for spec in lewm+idm+sa lewm+idm+sa+obj; do
  $EV --spec $spec --k 1 --policy-file policy_gauss6k.pt
  $EV --spec $spec --k 1 --policy-file policy_flow6k.pt
  $EV --spec $spec --k 16 --policy-file policy_flow6k.pt
  $EV --spec $spec --k 16 --policy-file policy_flow6k.pt --select random
  $EV --spec $spec --k 16 --policy-file policy_gauss6k.pt
done
log "ALL DONE"
