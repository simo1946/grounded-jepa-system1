# Pre-registration: when does world-model verification help System 1?

Written 2026-10-13, before any of these runs.

## Question

Selecting among System 1 samples with the world model lowered success in all
nine earlier tests. Re-ranking a policy's proposals with a world model helps
in Alrasheed et al. (2609.39235; 65 -> 77%). Our explanation: verification helps
only when the candidates differ by more than the world model's action
resolution. System 1 samples at the default spread differ by roughly policy
noise, where the model identifies the true chunk at chance or below.

## Runs (50 non-trivial Cube pairs, eval seed 42, same pairs throughout)

Models: grounded (`lewm+idm+sa`, seed 0) and object-grounded (`lewm+idm+sa+obj`, seed 0).

- **E1, spread sweep.** Best-of-16 selection by latent MSE (`cost`) and a uniformly random
  sample (`random`) for candidate spread `std_scale` in {0.25, 0.5, 1, 2, 4}
  (random: 0.25 to 2). The policy mean (k = 1) is the reference.
  Sanity check: `std_scale = 0` with selection must reproduce the k = 1 result exactly.
- **E2, diverse proposals.** Four extra System 1 policies (grounded model, policy seeds 1-4,
  same data and steps as `policy_cached.pt`). The pool of 5 members' means is
  - selected by the world model (`pool5-select`),
  - averaged (`pool5-mean`, the verifier-free combination),
  - and each member alone (k = 1); the members' average is the random-member control.

## Hypotheses and decision rules

- **H1.** The selection gain over random, (cost - random), grows with spread. Evidence: the gain
  is larger at std 2 than at std 0.25 and 0.5, for both models.
- **H2.** At spreads up to the default (std <= 1), selection does not beat the policy mean.
  Exact McNemar vs k = 1, two-sided.
- **H3.** With diverse proposals (E2), selection beats a random member. Primary test: exact
  McNemar of `pool5-select` vs member 0 (`policy_cached.pt`). Also reported: the average
  member success, McNemar vs each member, and vs averaging (`pool5-mean`).

A positive H3 together with H1 supports "verification helps when proposals differ by more
than the model resolves". A negative H3 means verification does not help on Cube at this
scale even with diverse proposals. Both outcomes are reported.

n = 50 per cell: effects below about 8 points are within noise. One model seed.

## Addendum (2026-10-13 23:40, after the cost-selection sweep, before the random-sample and pool results)

`spread_diag.py` measured the System 1 candidate spread on held-out states (grounded model):
policy std rms 0.83 normalised action units per dimension (median 0.70, p10 0.42, p90 1.17),
close to the expert actions' own rms (0.99). At the default spread, candidates therefore differ by
about sigma 0.8, where the world model identifies the true chunk 84% of the time
(rollout_diag, sigma = 1.0). The premise "System 1 samples differ by policy-noise amounts below
the model's resolution (sigma 0.1)" in docs/assessment.md was not measured and is wrong at the
default spread. The decision rules above are unchanged; the interpretation of E1/E2 must not rely
on that premise.
