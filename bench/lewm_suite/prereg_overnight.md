# Pre-registration: expressive System 1, uncertainty, DAgger

Written 2026-10-14 00:50, before any of these runs. Same protocol as before: 50 non-trivial Cube
pairs, eval seed 42, exact two-sided McNemar on identical pairs. One model seed (0).

## E3. Flow-matching System 1 (expressive proposer)

Motivation: the Gaussian System 1 barely fits the expert (its mean is 0.78 action units from the
executed chunk; the actions' own rms is 0.99) and its spread is close to the data's, consistent
with averaging several ways of acting.

Runs, for the grounded (`lewm+idm+sa`) and object-grounded (`lewm+idm+sa+obj`) models, from the same
latent caches, equal budget (6,000 steps, batch 256, lr 1e-3, policy seed 0):
- `policy_gauss6k.pt` (Gaussian, the control) and `policy_flow6k.pt` (flow, 10 Euler steps);
- each evaluated deterministic (k = 1: Gaussian mean / flow zero-noise sample),
  best-of-16 by the world model, and a random sample of 16 (flow only).
- Offline, on held-out latents: distance of the deterministic chunk and of the best of 16 samples
  to the executed chunk (`policy_fit.py`).

Hypotheses:
- **H4.** Flow deterministic >= Gaussian mean (primary: McNemar per model; pooled count over models).
- **H5.** With flow candidates, world-model selection beats the flow's deterministic output.
  (With the Gaussian it did not.)
- **H5b.** Flow best-of-16 beats flow random sample (the ranking stays informative).

## E4. Does ensemble disagreement predict failure?

At each evaluation pair's start (cached latents of start and goal frames), compute
- disagreement of 5 Gaussian policies (`policy_cached`, `policy_p1-4`): mean pairwise rms between
  their mean chunks;
- disagreement of the flow policy's 16 samples (rms spread around their mean);
- the baseline difficulty score: latent distance from start to goal.

Against the per-pair success of the corresponding System 1 runs, report AUROC for predicting failure
and success on the 50% most-confident pairs.

- **H6.** Ensemble disagreement predicts failure better than chance (AUROC > 0.5, with a
  bootstrap 95% interval) and adds over goal distance (higher AUROC). This is exploratory:
  n = 50 pairs gives wide intervals.

## E5. DAgger with the simulator expert (only if the expert can label arbitrary states)

Round: roll out the current policy on training starts, label the visited states with the
simulator expert, aggregate, retrain. Control: the same number of extra gradient steps on expert
data only. Primary: success after the final round vs control (McNemar).
- **H7.** DAgger beats the equal-steps control.

All outcomes are reported, including nulls.

## Addendum (01:00, before any E5 run and before any E3 evaluation result)

Oracle check: the vectorised oracle (`oracle.py`) equals OGBench's CubeMarkovOracle on 600 dataset
states (test). On the training data, before the cube is placed, the clean oracle explains 98-99% of
the expert's xyz actions; 72% of all expert steps come after placement, when the expert moves the arm
to a random pose.

E5 design, fixed now:
- Labels are block-level: the clean oracle action at each block's first state toward the goal cube pose,
  repeated over the block's 5 steps (the same in all oracle arms).
- A0 = `policy_gauss6k.pt` (noisy expert labels, E3). A1 = `policy_oracle6k.pt`: same budget, clean
  oracle labels toward the hindsight goal, all 5 blocks labelled. Also a flow version and the
  object-grounded model.
- DAgger: 2 rounds x 100 training pairs; the learner commits to its chunk as deployed, its visited
  block states are labelled by the oracle toward the commanded goal; each round fine-tunes 2,000
  steps on a 50/50 mix of DAgger states and offline oracle windows. Control: the same 2,000 steps per
  round on offline oracle windows only (`policy_octrl_r*`).
- Primary (H7): `policy_dagger_r2` vs `policy_octrl_r2`. Secondary: A1 vs A0 (clean labels), which is
  confounded with block-level labels and labelling all blocks.

## Addendum 2 (02:13): block-level labels were invalid; replaced before any DAgger result

The first oracle-label policies (block-level labels) scored 6/50 (grounded) and 12/50 (object), far
below the noisy-label policies (34, 39). Diagnosis by simulation (`oracle_sim_check.py`, 50 eval pairs):
the oracle itself solves 48/50 closed loop but only 8/50 when each action is held for a 5-step block.
Block-level labels teach exactly that held behaviour, so the design was wrong (my error), not clean
labels as such. The DAgger run built on that policy was stopped before any result.

Fix: labels simulated closed-loop. From the window's simulator state (qpos, qvel, ctrl), the oracle
runs 25 steps toward the goal cube pose in a scratch env; its actions are the chunk label
(`oracle_sim.py`, `gen_oracle_chunks.py`: 20,000 training windows). Ceiling check: executing such
oracle chunks open-loop, re-planned every 25 steps, solves 50/50 evaluation pairs.

E5 now (same structure as addendum 1): `policy_osim6k` (Gaussian) and `policy_osimflow6k` (flow), both
models; DAgger on the grounded model from `policy_osim6k`, visited block states labelled by simulated
oracle chunks toward the commanded goal (`sround*.pt`); control `policy_sctrl_r*`.
Primary H7: `policy_sdagger_r2` vs `policy_sctrl_r2`. Block-level results are reported as the failed
first attempt.
