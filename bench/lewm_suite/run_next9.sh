#!/usr/bin/env bash
# Cube queue, v9 (supersedes run_next8.sh; seed-1 replication of object grounding first; strictly sequential: 8 GB RAM; skip-if-done, so finished work is reused).
#   1b  object-aware grounding (alignment target includes the cube position)
#   C   object-aware planning cost (objectives.py) for CEM, selection and gating
#   1c  H-JEPA's protocol: grasp-centred windows, goal +20, budget 30 (their Table)
#   S   simulated sensors for System 1 (proprioception, noisy object detector,
#       dropout; goal-as-coordinates reported separately)
#   Hs  two-level System 1 (slow subgoal network + fast policy), +25 and +50-step goals
#   L   continual learning from own successes/errors (3 rounds + control)
#   M   partial observability (camera dropout; fill blank / hold / world-model prediction),
#       grounded LeWM vs our hybrid predictor as the state estimator
#   2   our hybrid Mamba-3/attention predictor, full stack
#   3   seed 1;  4  encoder / IDM-only ablations
set -uo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl
H=swm_home
NT=(--pairs nontrivial --skip-existing)
PK=(--pairs pickup --goal-offset 20 --budget 30 --skip-existing)

if [ -n "${WAIT_PID:-}" ]; then while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done; fi

ck() { echo "$H/checkpoints/cube/$1/seed$2"; }
train()  { [ -f "$(ck "$1" "$2")/weights.pt" ] || python3 train.py --task cube --spec "$1" --seed "$2" --steps 2000; }
policy() { [ -f "$(ck "$1" "$2")/policy.pt" ]  || python3 train_policy.py --task cube --spec "$1" --seed "$2"; }
reach()  { [ -f "$(ck "$1" "$2")/reach.pt" ]   || python3 train_reach.py --task cube --spec "$1" --seed "$2"; }
cache()  { local n; n=$(python3 -c "from models import Spec; from latent_cache import cache_path; print(cache_path(Spec.parse('$1').name(), $2, 'cube'))")
           [ -f "$n" ] || python3 latent_cache.py --spec "$1" --seed "$2" --data cube; }
cpol()   { [ -f "$(ck "$1" "$2")/$4" ] || python3 cached_policy.py --spec "$1" --seed "$2" --sensors "$3" --out "$4"; }
ev()     { python3 evaluate.py --task cube "${NT[@]}" "$@"; }
pk()     { python3 evaluate.py --task cube "${PK[@]}" "$@"; }
diag()   {
  local n; n=$(python3 -c "from models import Spec; print(Spec.parse('$1').name())")
  [ -f "$H/results/probe_diag__${n}__seed$2.json" ]   || python3 probe_diag.py --spec "$1" --seed "$2"
  [ -f "$H/results/rollout_diag__${n}__seed$2.json" ] || python3 rollout_diag.py --spec "$1" --seed "$2"
  if [ -f "$(ck "$1" "$2")/reach.pt" ]; then
    [ -f "$H/results/progress_diag__${n}__seed$2.json" ] || python3 progress_diag.py --spec "$1" --seed "$2"
  fi
}
full_eval() {
  local s=$1 d=$2
  ev --spec "$s" --seed "$d" --controller s1 --k 1
  ev --spec "$s" --seed "$d" --controller s1 --cost reach
  ev --spec "$s" --seed "$d" --controller s1s2 --cost reach
  ev --spec "$s" --seed "$d"
  ev --spec "$s" --seed "$d" --cost reach
}
G=lewm+idm+sa
O=lewm+idm+sa+obj

# ---- 1b: object-aware grounding
train "$O" 0 && policy "$O" 0 && { ev --spec "$O" --seed 0 --controller s1 --k 1; ev --spec "$O" --seed 0; diag "$O" 0; }
echo "STAGE 1b DONE"

# ---- C: object-aware cost
for s in "$O" "$G"; do
  ev --spec "$s" --seed 0 --cost probe
  ev --spec "$s" --seed 0 --controller s1 --cost probe
  ev --spec "$s" --seed 0 --controller s1s2 --cost probe
done
echo "STAGE C DONE"

# ---- 1c: H-JEPA protocol
pk --spec random
for s in lewm "$G" "$O"; do pk --spec "$s" --seed 0 --controller s1 --k 1; done
for s in "$G" lewm "$O"; do pk --spec "$s" --seed 0; done
for s in "$O" "$G"; do pk --spec "$s" --seed 0 --cost probe; done
echo "STAGE 1c DONE"

# ---- S: simulated sensors (System 1, grounded model, seed 0)
cache "$G" 0
SENS=("none:policy_cached.pt" "proprio:policy_s-proprio.pt"
      "proprio+obj+objnoise=0.01:policy_s-proprio-obj.pt"
      "proprio+obj+objnoise=0.02+drop=0.2:policy_s-proprio-obj-noisy.pt"
      "proprio+obj+goalobj+objnoise=0.01:policy_s-goalcoords.pt")
for pair in "${SENS[@]}"; do
  sens=${pair%%:*}; f=${pair##*:}
  cpol "$G" 0 "$sens" "$f"
  ev --spec "$G" --seed 0 --controller s1 --k 1 --policy-file "$f"
  pk --spec "$G" --seed 0 --controller s1 --k 1 --policy-file "$f"
done
echo "STAGE S DONE"

# ---- Hs: two-level System 1
[ -f "$(python3 -c "from latent_cache import cache_path; print(cache_path('$G', 0, 'cube_eval'))")" ] || \
  python3 latent_cache.py --spec "$G" --seed 0 --data cube_eval
[ -f "$(ck "$G" 0)/policy_hier.pt" ] || python3 cached_subgoal.py --spec "$G" --seed 0 --base policy_cached.pt --out policy_hier.pt
for f in policy_cached.pt policy_hier.pt; do
  ev --spec "$G" --seed 0 --controller s1 --k 1 --policy-file "$f"
  ev --spec "$G" --seed 0 --controller s1 --k 1 --policy-file "$f" --goal-offset 50 --budget 100
  pk --spec "$G" --seed 0 --controller s1 --k 1 --policy-file "$f"
done
F=policy_hier_gated.pt                  # post-hoc variant: subgoal only for far goals
ev --spec "$G" --seed 0 --controller s1 --k 1 --policy-file "$F"
ev --spec "$G" --seed 0 --controller s1 --k 1 --policy-file "$F" --goal-offset 50 --budget 100
pk --spec "$G" --seed 0 --controller s1 --k 1 --policy-file "$F"
echo "STAGE Hs DONE"

# ---- L: continual learning (policy_cached.pt is the shared starting point)
python3 learn_loop.py --spec "$G" --seed 0 --rounds 3 --episodes 100
echo "STAGE L DONE"

# ---- 2: ours (hybrid predictor + grounding)
# CEM with the hybrid predictor takes ~2.3 h per evaluation on 2 CPU cores (pure-PyTorch Mamba-3 scan) and
# CEM is at the floor for every model, so hybrid CEM-with-reach-cost and CEM on the pickup protocol are skipped.
train hybrid+idm+sa 0 && policy hybrid+idm+sa 0 && reach hybrid+idm+sa 0 && {
  ev --spec hybrid+idm+sa --seed 0 --controller s1 --k 1
  ev --spec hybrid+idm+sa --seed 0 --controller s1 --cost reach
  ev --spec hybrid+idm+sa --seed 0 --controller s1s2 --cost reach
  ev --spec hybrid+idm+sa --seed 0
  pk --spec hybrid+idm+sa --seed 0 --controller s1 --k 1
  diag hybrid+idm+sa 0; }
echo "STAGE 2 DONE"

# ---- M: camera dropout (System 1, decide every block)
for s in "$G" hybrid+idm+sa; do
  [ -f "$(ck "$s" 0)/policy.pt" ] || continue
  ev --spec "$s" --seed 0 --controller s1 --k 1 --receding 1
  for fill in blank hold predict; do
    ev --spec "$s" --seed 0 --controller s1 --k 1 --receding 1 --camera-drop 0.5 --fill "$fill"
  done
done
echo "STAGE M DONE"

# ---- 3: seed 1, replication first (object grounding is the main positive finding)
train "$O" 1 && policy "$O" 1 && { ev --spec "$O" --seed 1 --controller s1 --k 1; pk --spec "$O" --seed 1 --controller s1 --k 1; diag "$O" 1; }
train "$G" 1 && policy "$G" 1 && { ev --spec "$G" --seed 1 --controller s1 --k 1; pk --spec "$G" --seed 1 --controller s1 --k 1; }
train lewm 1 && policy lewm 1 && { ev --spec lewm --seed 1 --controller s1 --k 1; pk --spec lewm --seed 1 --controller s1 --k 1; ev --spec lewm --seed 1; diag lewm 1; }
echo "STAGE 3 DONE"

# ---- 4: encoder and grounding ablations
E="lewm+idm+sa+enc=mnv4s"
train "$E" 0 && policy "$E" 0 && { ev --spec "$E" --seed 0 --controller s1 --k 1; ev --spec "$E" --seed 0; diag "$E" 0; }
train lewm+idm 0 && policy lewm+idm 0 && { ev --spec lewm+idm --seed 0 --controller s1 --k 1; ev --spec lewm+idm --seed 0; diag lewm+idm 0; }
echo "ALL DONE"
