#!/usr/bin/env bash
# E10 (prereg_critic.md): contrastive verifier (CLM / CoVer-VLA style), offline then closed loop.
set -u
cd "$(dirname "$0")"
export MUJOCO_GL=${MUJOCO_GL:-egl}
O=lewm+idm+sa+obj
CK=swm_home/checkpoints/cube/$O/seed0
[ -f $CK/verifier_in.pt ] || python contrastive_verifier.py --spec $O --out verifier_in.pt
[ -f $CK/verifier_hard.pt ] || python contrastive_verifier.py --spec $O --hard 2 --out verifier_hard.pt
EV="python evaluate.py --task cube --spec $O --seed 0 --pairs nontrivial --controller s1 --skip-existing"
$EV --k 16 --policy-file policy_gauss6k.pt --select verifier --critic-file verifier_in.pt
$EV --k 1 --policy-file verifier_in_retrieval.pt
$EV --k 16 --policy-file policy_gauss6k.pt --select verifier --critic-file verifier_hard.pt
echo "[$(date +%H:%M:%S)] ALL DONE"
