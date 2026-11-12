# GPU study: tworoom (free profile)

6 trained models, 48 evaluations; success in % (seeds).

## Success by configuration

| config | System 1, all pairs (authors' protocol) | CEM planner, all pairs (authors' protocol) | System 1, non-trivial +25 | flow System 1, non-trivial +25 | System 1, non-trivial +50 | two-level, +50 | best-of-16 by world model | best-of-16, candidates near the mean (std x 0.25) | CEM planner |
|---|---|---|---|---|---|---|---|---|---|
| C1 LeWM | 96 ± 1 (3) | 97 ± 1 (3) | 97 ± 1 (3) | 98 ± 0 (3) | 93 ± 0 (3) | 99 ± 1 (3) | 98 ± 1 (3) | 98 ± 0 (3) | 97 ± 1 (3) |
| C2 + proprio grounding | 95 ± 0 (3) |  | 97 ± 1 (3) | 98 ± 0 (3) | 93 ± 0 (3) | 98 ± 0 (3) | 99 ± 1 (3) | 99 ± 1 (3) |  |

## Paired contrasts (exact McNemar on pairs pooled over seeds; ARL rows: Wilcoxon signed-rank)

| contrast | eval | A | B | units | A − B (pts) | 95% CI (pair bootstrap) | A-only | B-only | p | p (Holm, primary) | p boot (Holm, primary) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **all configs: two-level vs flat, +50** | hier_50 | 99% | 93% | 6 | +5.3 ± 0.3 | [+0.3, +11.7] | 19 | 0 | 3.81e-06 | 3.81e-06 | 0.0332 |
| proprio grounding | s1_nt | 97% | 97% | 3 | +0.0 ± 0.0 | [+0.0, +0.0] | 0 | 0 | 1 | — | — |
| all configs: best-of-16 near the mean vs mean (exploratory) | sel16n | 99% | 97% | 6 | +1.4 ± 0.3 | [+0.0, +3.9] | 5 | 0 | 0.0625 | — | — |
| all configs: flow vs Gaussian System 1, +25 | s1f_nt | 98% | 97% | 6 | +1.1 ± 0.4 | [+0.0, +3.3] | 4 | 0 | 0.125 | — | — |
| C1 best-of-16 vs mean | sel16 | 98% | 97% | 3 | +1.1 ± 0.6 | [+0.0, +3.3] | 2 | 0 | 0.5 | — | — |
| C1 two-level vs flat, +50 | hier_50 | 99% | 93% | 3 | +5.6 ± 0.6 | [+0.6, +11.7] | 10 | 0 | 0.00195 | — | — |
| C2 best-of-16 vs mean | sel16 | 99% | 97% | 3 | +1.7 ± 1.0 | [-1.1, +5.6] | 4 | 1 | 0.375 | — | — |
| C2 two-level vs flat, +50 | hier_50 | 98% | 93% | 3 | +5.0 ± 0.0 | [+0.0, +11.7] | 9 | 0 | 0.00391 | — | — |
| C1 System 1 vs CEM | cem | 97% | 97% | 3 | +0.0 ± 1.7 | [-3.3, +3.3] | 3 | 3 | 1 | — | — |
| C1 System 1 vs CEM, all pairs | cem_all | 96% | 97% | 3 | -1.7 ± 0.0 | [-7.8, +3.9] | 5 | 8 | 0.581 | — | — |
| C2 + proprio grounding vs LeWM, all pairs | s1_all | 95% | 96% | 3 | -0.6 ± 0.6 | [-2.8, +1.1] | 1 | 2 | 1 | — | — |

Bold rows are the pre-registered family (Holm-corrected together); the rest are exploratory. Units = (config, seed) groups of paired start/goal pairs. McNemar p pools units as if independent (descriptive); the bootstrap resamples the shared pairs with all units together (10,000 draws).

## Diagnostics (seed 0)

(Cube only)
