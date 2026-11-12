# Pre-registration: selecting System 1 candidates by their real-world outcome

Written 2026-10-18 13:30, before any of these runs. Object-grounded model (`lewm+idm+sa+obj`, seed 0),
System 1 = `policy_gauss6k.pt` (Gaussian, 6,000 steps; 39/50 deterministic). 50 non-trivial Cube pairs,
eval seed 42, same pairs as every earlier study; exact two-sided McNemar on identical pairs.

## Question

Selection by the world model's imagined outcome (latent distance, object-aware cost, reachability
critic) lowered success in every test. Two possible reasons: (a) no candidate among System 1's samples
is better than its own mean, or (b) the scorer cannot tell which one is. This study separates them and
tests a scorer trained on real outcomes (the author's proposal: measure success in the real world, in
real units, not in imagination).

## Runs

Candidates in every arm: 16 per decision, candidate 0 = the policy mean, 15 samples (std x 1).
- **Ceiling (oracle selection):** each candidate is executed open-loop in a scratch copy of the simulator
  from the current true state; pick the one whose final cube position is closest to the goal cube
  position (metres). Uses privileged state: an upper bound for any selector, not a deployable method.
- **Critic selection (deployable):** an MLP critic f(z_t, z_goal, chunk) -> predicted final cube-goal
  distance (metres) and success probability (<= 4 cm). Trained only on real (simulated) outcomes of
  System 1 candidates executed from training-set states toward hindsight goals (`gen_critic_data.py`);
  at run time it sees only latents and the candidate. Pick the lowest predicted distance.
- Controls: policy mean (k = 1; existing, 39/50), random candidate (`--select random`), latent-MSE
  selection (existing, 30/50).

## Hypotheses

- **H8a (headroom).** Oracle selection beats the policy mean. If not, no selector can help here.
- **H8 (primary).** Critic selection beats the policy mean.
- **H8b.** Critic selection beats latent-MSE selection.
- Offline: on held-out evaluation-set states, the critic's ranking of candidates correlates with
  their real outcomes (Spearman within state), reported alongside the latent-MSE ranking.

All outcomes are reported, including nulls.

## Results (2026-10-18 14:30; seed 0, 50 non-trivial pairs)

| Selection among 16 System 1 candidates | Success |
|---|---|
| Policy mean (no selection) | 39/50 |
| True-outcome ceiling (privileged) | 41/50 (vs mean 4/2, p = 0.69) |
| Outcome critic | 33/50 (vs mean 1/7, p = 0.07; vs latent MSE 33 vs 30, p = 0.55) |
| Latent-MSE world model | 30/50 |
| Random candidate | 28/50 |

- H8a (headroom) not supported in closed loop: perfect one-chunk selection adds 2/50. Offline, on
  held-out states, the true-outcome pick succeeds 93% vs 87% for the mean (100% vs 77% on the 78
  states where the choice matters), but 9 evaluation episodes still fail even when the truly best of the 16 candidates is chosen at every decision.
- H8 not supported: the critic ranks candidates barely above chance (within-state Spearman 0.05; 0.14
  where the choice matters). It learned which situations are hard, not which candidate is better.
- H8b: 33 vs 30, not significant.
- Reading: on Cube at this scale, selecting among System 1's own samples has almost no headroom; gains
  must come from better proposals (perception of the object, more expert data), not a better judge.

## Addendum E7 (2026-10-18 14:05, before any run): use the real outcomes to train, not to judge

The judge had no headroom; the proposals are the limit. So use the same real outcomes to improve
the proposals: reward-weighted fine-tuning (advantage-weighted regression) of `policy_gauss6k.pt` on
the 3,000 x 16 executed candidates, weight exp(-(d - mean_state d) / 0.02 m), clipped at 20, half of
each batch the original expert data, 2,000 steps (`awr_policy.py` -> `policy_awr.pt`).
Control: identical, uniform weights (`policy_awrctrl.pt`).
- **H9 (primary):** `policy_awr` > `policy_awrctrl` (McNemar, 50 non-trivial pairs).
- Secondary: `policy_awr` vs the starting policy (39/50).
Only one round (offline data from the starting policy); on-policy rounds come after, if H9 holds.

### E7 result (14:20)

`policy_awr` 40/50, `policy_awrctrl` 42/50, start 39/50. H9 not supported (40 vs 42, discordant 0/2,
p = 0.5). One offline round of reward-weighted fine-tuning on the policy's own candidates changes
nothing, consistent with the small selection headroom (true-outcome ceiling 41/50).

## Addendum E8 (2026-10-18 14:15, before any run): practice with a real-world success measure

Residual practice loop (after ResFiT, arXiv 2509.19301; RFS, ICLR 2026): System 1 (`policy_gauss6k.pt`,
object-grounded model) stays frozen; a small residual network adds a per-block correction (5 blocks x 5
action dims, bounded) to its deterministic chunk. Practice is in the simulator from training-set states
(start r, hindsight goal g = r + k*5, k in 1..5): each attempt executes the corrected chunk (25 steps)
from the true state at r.
- Reward (real): -10 x final cube-goal distance (m) + 1 if within 4 cm.
- Updates: group-relative policy gradient (4 corrections per situation, advantage = reward minus the
  group mean, divided by the group std), PPO clipping 0.2, 4 epochs, 128 situations per iteration,
  120 iterations; L2 penalty 0.01 on the correction; the correction's last layer starts at zero, so
  training starts exactly at System 1.
- Control (imagined reward): identical, but the reward is the world model's predicted latent distance to
  the goal after the chunk (negative squared distance). Real outcomes are logged for both arms.
Evaluation: corrected deterministic policy, 50 non-trivial evaluation pairs (closed loop, rendered).
- **H10 (primary):** real-reward practice > System 1 alone (39/50).
- **H10b:** real-reward practice > imagined-reward practice.
Amendment (14:30, after a 2-iteration smoke run only, no real run): practice and validation situations are
restricted to non-trivial ones (the cube must move more than 4 cm to reach the goal), matching the
evaluation protocol; in the smoke run 87% of unrestricted situations were already solved by System 1.

### E8 result (2026-10-18 16:10)

Closed loop, 50 non-trivial held-out pairs: real-reward practice 39/50 vs System 1 39/50 (4/4, p = 1);
imagined-reward practice 30/50 (vs System 1 3/12, p = 0.035; vs real 5/14, p = 0.064).
H10 not supported in closed loop; H10b supported in direction (p = 0.064).

Diagnostics (`residual_heldout.py`, 400 held-out non-trivial single-chunk situations, true outcomes):
System 1 51%, real-reward practice 65%, imagined-reward practice 50%. By goal distance (blocks ahead,
1-5): System 1 30/40/56/62/58%; real practice 64/69/69/73/51%; imagined 41/40/64/54/46%.
Reading: practice with the real measure learned corrections that generalise to new episodes for near
goals, but not for the 5-block first move of every evaluation episode, where it is slightly worse;
in closed loop the effects cancel. Practising with the world model's imagined score does not improve
and in closed loop hurts. Next (not yet run): practice matched to deployment, i.e. full on-policy
episodes (both decisions, goal 25 steps ahead), which needs rendering at decision points.

## Addendum E9 (2026-10-18 15:45, before any run beyond a 2-round smoke test): full-episode practice

Practice matched to deployment (`practice_rl.py`): non-trivial training-set starts, goal 25 steps ahead,
two decisions of 25 steps, stop at success; decision 2 sees the rendered, encoded state after decision 1
(renderer checked: frames within 0.2/255 of the dataset, every rendered latent's nearest cached latent is
the right frame). Frozen System 1 (`policy_gauss6k.pt`, object-grounded) + residual correction;
group-relative policy gradient (4 episodes per start, 96 starts per round, 100 rounds), PPO clip 0.2.
Two reward schemes, same everything else:
- **sparse:** episode success (0/1) for both decisions (ResFiT-style);
- **shaped (intermediate checkpoints):** per decision 10 x progress in metres
  (potential-based, Ng et al. 1999) + 0.3 the first time the gripper grasps the cube + 1 at success;
  each decision credited with its reward-to-go.
Evaluation: 50 non-trivial held-out pairs, closed loop.
- **H11 (primary):** shaped practice > System 1 alone (39/50).
- **H11b:** sparse practice > System 1 alone.
- **H11c:** shaped > sparse.

### E9 result (2026-10-18 16:55)

Closed loop, 50 non-trivial held-out pairs: shaped practice 39/50, sparse practice 36/50, System 1 39/50.
H11 not supported (39 vs 39, 5/5); H11b not supported (36 vs 39, 2/5, p = 0.45); H11c 39 vs 36 (8/5,
p = 0.58). On the fixed practice validation episodes (training starts, deterministic correction) System 1
already succeeds 86%; shaped practice ended at 82%, sparse at 88%.
Observations, not tested: exploration noise (std 0.3 per action dimension) cut practice-episode success to
about 52-60%, so most comparisons were among degraded variants; headroom on training starts was small
(86%). 38,400 practice episodes per arm.

## Addendum E10 (2026-10-21 20:55, before any code or run): a contrastive verifier (CLM / CoVer-VLA style)

Motivation: Stanford/NVIDIA's CLM-8B scores candidate actions by embedding similarity between the state and
each candidate; its robotics variant CoVer-VLA (arXiv 2602.12281) trains a verifier with a symmetric InfoNCE
loss (in-batch negatives, no failure labels) and picks the highest-scoring action chunk among a base
policy's samples. Here the same principle, at our scale, on the same 50 non-trivial pairs (eval seed 42),
object-grounded model seed 0, System 1 = `policy_gauss6k.pt` (39/50).

Verifier: state encoder f(z_t, z_goal) and action encoder g(chunk of 5 blocks x 25 dims), both MLPs to a
256-d unit vector, score = cosine x learned temperature. Trained on the cached training latents with
hindsight goals (k = 1..5 blocks ahead, the executed expert chunk is the positive), symmetric InfoNCE,
batch 1024, 6,000 steps (`contrastive_verifier.py`).
- **V-in (primary, CoVer-faithful):** in-batch negatives only.
- **V-hard (secondary):** plus 2 perturbed copies of the positive chunk per state as extra negatives
  (Gaussian noise, sigma log-uniform in [0.1, 1.0] normalised action units).

Runs (closed loop, 50 pairs):
1. Select among 16 System 1 candidates (mean + 15 samples) with V-in; then with V-hard.
2. **Retrieval System 1 (CLM's cached-menu idea):** pick, from a menu of 20,000 expert chunks from the
   training set (embeddings cached), the chunk with the highest score for (z_t, z_goal); no generative policy.

Comparisons (same pairs, existing): System 1 mean 39, true-outcome ceiling 41, outcome critic 33, latent-MSE
selection 30, random candidate 28.
- **H12 (primary):** V-in selection > System 1 mean (exact McNemar). Expected hard to meet: the ceiling is 41.
- **H12b:** V-in selection > outcome-critic selection (33).
- **H12c:** retrieval System 1 vs Gaussian System 1 (two-sided; no direction predicted).
- Offline, before closed loop: within-state Spearman between verifier score and real outcome on the 1,500
  held-out evaluation states x 16 executed candidates (critic: 0.05), and retrieval top-1 of the true chunk.
- Decision latency per step reported for each arm.

### E10 result (2026-10-26)

Closed loop, 50 non-trivial pairs, object-grounded model seed 0 (ViT-tiny encoder, transformer predictor,
192-d latent, 64 px); verifier = two 3-layer MLPs (512 hidden, 256-d output), CLM's bidirectional InfoNCE.

| Arm | Success | vs System 1 mean (39) |
|---|---|---|
| V-in selection among 16 System 1 candidates | 28/50 | 3/14, p = 0.013 |
| V-hard selection (+2 perturbed copies as negatives) | 38/50 | 0/1, p = 1 |
| Retrieval System 1 (20,000-chunk expert menu, no generative policy) | 28/50 | 8/19, p = 0.052 |

- H12 not supported (V-in is worse than the mean); H12b not supported (28 vs critic 33, 7/12, p = 0.36);
  H12c: retrieval is worse than the Gaussian System 1 (borderline, p = 0.052).
- Offline (held-out set: 300 evaluation states x 16 executed candidates; the pre-registration said 1,500,
  which is the training-side count): top-1 of the true chunk among 1,024 = 0.195 (V-in) / 0.189 (V-hard),
  chance 0.001; within-state Spearman with real outcome 0.005 / 0.006 (critic 0.05); pick success 81% / 85%
  vs 87% for the mean candidate.
- Reading: the verifier recognises expert-like chunks but not which System 1 candidate will succeed. With
  perturbed copies as negatives it learns "smooth = good" and almost always keeps the mean (one discordant
  pair). Latency: 0.02 s per decision on CPU for selection and retrieval alike. Consistent with E6: on Cube
  at this scale, choosing among candidates has almost no headroom (true-outcome ceiling 41/50).
