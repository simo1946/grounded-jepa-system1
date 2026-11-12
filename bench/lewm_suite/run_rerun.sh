#!/usr/bin/env bash
# Re-run of the CPU-study experiments whose result files were lost (2026-11-12), on data regenerated with the
# same collector and seeds (collect.py: cube 400 episodes seed 3072, cube_eval 80 episodes seed 777).
# The regenerated episodes are not bit-identical to the lost ones, so every number is new; the paper reports
# these. Commands and settings are those of the original scripts named per stage. Every step skips finished
# work, and train.py resumes from its last checkpoint, so the script can be restarted at any point.
#
#   ./run_rerun.sh base       # data checks, random baseline, world models and System 1 (5.2 replication)
#   ./run_rerun.sh timing     # single-robot decision latency, System 1 vs CEM (5.1)
#   ./run_rerun.sh select     # selection vs a random candidate (5.6; run_verify.sh, E1)
#   ./run_rerun.sh ceiling    # privileged ceiling and outcome critic (5.6; run_critic_*.sh, E6)
#   ./run_rerun.sh verifier   # contrastive verifier (5.6; run_verifier.sh, E10)
set -euo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=${MUJOCO_GL:-egl}
: "${STABLEWM_HOME:?set STABLEWM_HOME to the data folder}"
H=$STABLEWM_HOME
mkdir -p logs "$H/results"
G=lewm+idm+sa          # C2
O=lewm+idm+sa+obj      # C3
NT=(--task cube --pairs nontrivial --skip-existing)

log()    { echo "[$(date +%H:%M:%S)] $*"; }
ck()     { echo "$H/checkpoints/cube/$1/seed$2"; }
train()  { [ -f "$(ck "$1" "$2")/weights.pt" ] || python3 train.py --task cube --spec "$1" --seed "$2" --steps 2000; }
policy() { [ -f "$(ck "$1" "$2")/policy.pt" ]  || python3 train_policy.py --task cube --spec "$1" --seed "$2"; }
cache()  { local d=$3 n
           n=$(python3 -c "from models import Spec; from latent_cache import cache_path; print(cache_path(Spec.parse('$1').name(), $2, '$d'))")
           [ -f "$n" ] || python3 latent_cache.py --spec "$1" --seed "$2" --data "$d"; }

stage_base() {
  log "already-solved pairs in the regenerated data"
  [ -f "$H/results/trivial_share_cpu.json" ] || \
    python3 trivial_share.py "$H/datasets/cube.h5" "$H/datasets/cube_eval.h5" > "$H/results/trivial_share_cpu.json"
  log "random policy on the authors' protocol (50 pairs, all, as the original)"
  python3 evaluate.py --task cube --spec random --num-eval 50 --skip-existing
  log "5.2 CPU replication of object grounding (run_next9.sh): C2 and C3, seeds 0 and 1"
  for seed in 0 1; do
    for spec in "$O" "$G"; do
      train "$spec" "$seed"; policy "$spec" "$seed"
      python3 evaluate.py "${NT[@]}" --spec "$spec" --seed "$seed" --controller s1 --k 1
    done
  done
  log "C1 for the latency benchmark"
  train lewm 0; policy lewm 0
}

stage_timing() {
  log "5.1 latency: one robot (one env), the same 3 non-trivial pairs, System 1 mean action vs CEM"
  python3 evaluate.py "${NT[@]}" --spec lewm --seed 0 --controller s1 --k 1 --num-eval 3 --env-chunk 1 --device cpu
  python3 evaluate.py "${NT[@]}" --spec lewm --seed 0 --num-eval 3 --env-chunk 1 --device cpu
}

stage_select() {
  log "5.6 selection by the world model vs a random candidate (run_verify.sh E1)"
  for s in 0.25 0.5 1 2; do
    for spec in "$G" "$O"; do
      python3 evaluate.py "${NT[@]}" --spec "$spec" --seed 0 --controller s1 --k 16 --std-scale "$s"
      python3 evaluate.py "${NT[@]}" --spec "$spec" --seed 0 --controller s1 --k 16 --std-scale "$s" --select random
    done
  done
}

stage_ceiling() {
  log "5.6 privileged ceiling and outcome critic (run_overnight.sh policy, run_critic_data.sh, run_critic_eval.sh)"
  local CK; CK=$(ck "$O" 0)
  cache "$O" 0 cube; cache "$O" 0 cube_eval
  [ -f "$CK/policy_gauss6k.pt" ] || python3 cached_policy.py --spec "$O" --seed 0 --out policy_gauss6k.pt \
      --steps 6000 --batch 256 --policy-seed 0 --policy-type gauss --device cpu
  local GEN=(python3 gen_critic_data.py --spec "$O" --policy-file policy_gauss6k.pt) D="$H/datasets/critic_lewm_idm_sa_obj"
  [ -f "${D}__cube_p0.pt" ]      || "${GEN[@]}" --data cube --n 1500 --part 0
  [ -f "${D}__cube_p1.pt" ]      || "${GEN[@]}" --data cube --n 1500 --part 1
  [ -f "${D}__cube_eval_p0.pt" ] || "${GEN[@]}" --data cube_eval --n 300 --part 0
  [ -f "$CK/critic.pt" ] || python3 critic.py --spec "$O"
  for sel in oracle critic random cost; do
    python3 evaluate.py "${NT[@]}" --spec "$O" --seed 0 --controller s1 --k 16 --policy-file policy_gauss6k.pt --select "$sel"
  done
  python3 evaluate.py "${NT[@]}" --spec "$O" --seed 0 --controller s1 --k 1 --policy-file policy_gauss6k.pt
}

stage_verifier() {
  log "5.6 contrastive verifier (run_verifier.sh E10)"
  local CK; CK=$(ck "$O" 0)
  [ -f "$CK/verifier_in.pt" ]   || python3 contrastive_verifier.py --spec "$O" --out verifier_in.pt
  [ -f "$CK/verifier_hard.pt" ] || python3 contrastive_verifier.py --spec "$O" --hard 2 --out verifier_hard.pt
  local EV=(python3 evaluate.py "${NT[@]}" --spec "$O" --seed 0 --controller s1)
  "${EV[@]}" --k 16 --policy-file policy_gauss6k.pt --select verifier --critic-file verifier_in.pt
  "${EV[@]}" --k 1 --policy-file verifier_in_retrieval.pt
  "${EV[@]}" --k 16 --policy-file policy_gauss6k.pt --select verifier --critic-file verifier_hard.pt
}

case "${1:-}" in
  base|timing|select|ceiling|verifier) "stage_$1"; log "STAGE $1 DONE" ;;
  *) echo "usage: $0 base|timing|select|ceiling|verifier" >&2; exit 2 ;;
esac
