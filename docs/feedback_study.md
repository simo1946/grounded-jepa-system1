# Feedback-frequency study (paper Section 5.8)

Plan: `bench/lewm_suite/prereg_feedback.md` (2026-11-06) and its dated addendum (2026-11-09).
Runs: `bench/lewm_suite/run_feedback.sh` (first round), `run_feedback2.sh` (addendum).
Analysis: `bench/lewm_suite/analyze_feedback.py`; output and every result file: `results/feedback_study/`.

CPU-study world models (400 training episodes, 2,000 steps at batch 32): C1 = `lewm` (17.92M parameters),
C3 = `lewm+idm+sa+obj` (18.35M), seeds 0 and 1. Cube, 60 non-trivial pairs from 40 evaluation episodes,
goal 25 steps ahead, 50-step budget, evaluation seed 42. Success %:

| Condition | C1 s0 | C1 s1 | C3 s0 | C3 s1 | Pooled |
|---|---|---|---|---|---|
| S1-25 (System 1, 25 steps per decision (default)) | 51.7 | 46.7 | 76.7 | 75.0 | 62.5 |
| S1-5 (System 1, first 5 steps of each chunk) | 46.7 | 38.3 | 63.3 | 68.3 | 54.2 |
| S1-1 (System 1, first step of each chunk) | 25.0 | 30.0 | 40.0 | 48.3 | 35.8 |
| S1c5 (System 1 trained for a 5-step chunk) | 55.0 | 41.7 | 80.0 | 68.3 | 61.3 |
| S1c1 (System 1 trained for a 1-step chunk) | 33.3 | 23.3 | 40.0 | 48.3 | 36.2 |
| IDM-1 (our GC-IDM-style controller, every step) | 58.3 | 51.7 | 66.7 | 63.3 | 60.0 |
| IDM-1-noH (the same without horizon input) | 45.0 | 41.7 | 63.3 | 66.7 | 54.2 |
| OIDM-1 (authors' GC-IDM code (H_max 50), every step) | 51.7 | 46.7 | 56.7 | 61.7 | 54.2 |
| OIDM-1-H1 (authors' code, their H_max = 1 ablation) | 30.0 | 21.7 | 20.0 | 18.3 | 22.5 |

Contrasts (decision rule of the addendum): mean difference A − B in points over the four models, 95% interval
from 10,000 bootstrap resamples of the 40 evaluation episodes, two-sided bootstrap p Holm-corrected over the 12
contrasts. Claimed = Holm p < 0.05 and the same sign in all four models. Per model: pairs solved only by A vs only by B.

| Contrast | C1 s0 | C1 s1 | C3 s0 | C3 s1 | A − B [95% CI] | Holm p | Claimed |
|---|---|---|---|---|---|---|---|
| H1a S1-5 vs S1-25 | 4 vs 7 | 3 vs 8 | 1 vs 9 | 3 vs 7 | -8.3 [-14.8, -2.4] | 0.0352 | yes |
| H1b S1-1 vs S1-25 | 2 vs 18 | 3 vs 13 | 2 vs 24 | 2 vs 18 | -26.7 [-35.1, -17.9] | 0.0024 | yes |
| H2 IDM-1 vs S1-1 | 22 vs 2 | 20 vs 7 | 19 vs 3 | 16 vs 7 | +24.2 [+14.0, +34.1] | 0.0024 | yes |
| H3 IDM-1 vs S1-25 | 14 vs 10 | 16 vs 13 | 5 vs 11 | 5 vs 12 | -2.5 [-10.8, +6.8] | 1.0000 | no |
| H4 IDM-1 vs IDM-1-noH | 15 vs 7 | 15 vs 9 | 6 vs 4 | 6 vs 8 | +5.8 [-2.1, +13.3] | 0.6832 | no |
| H5 OIDM-1 vs IDM-1 | 3 vs 7 | 5 vs 8 | 6 vs 12 | 5 vs 6 | -5.8 [-11.8, +0.4] | 0.4296 | no |
| H6 OIDM-1 vs S1-25 | 12 vs 12 | 14 vs 14 | 4 vs 16 | 4 vs 12 | -8.3 [-17.3, +2.0] | 0.5860 | no |
| H7 OIDM-1 vs OIDM-1-H1 | 17 vs 4 | 21 vs 6 | 28 vs 6 | 28 vs 2 | +31.7 [+21.5, +44.0] | 0.0024 | yes |
| H8 S1c5 vs S1-5 | 6 vs 1 | 4 vs 2 | 11 vs 1 | 7 vs 7 | +7.1 [+0.9, +13.1] | 0.1736 | no |
| H9 S1c1 vs S1-1 | 6 vs 1 | 1 vs 5 | 7 vs 7 | 7 vs 7 | +0.4 [-4.3, +5.8] | 1.0000 | no |
| H10 S1c5 vs S1-25 | 6 vs 4 | 6 vs 9 | 4 vs 2 | 3 vs 7 | -1.2 [-7.1, +4.5] | 1.0000 | no |
| H11 S1c1 vs S1-25 | 4 vs 15 | 3 vs 17 | 3 vs 25 | 3 vs 19 | -26.2 [-35.2, -16.5] | 0.0024 | yes |

Deviations from the plan:
- First round: the run hit the 2-hour limit of the compute session after three of the four models; the remaining
  steps of the fourth model were run separately with the same commands (finished results are skipped, not repeated).
- Addendum round: the four models were run in three separate sessions (two seed-0 models together, then each
  seed-1 model) to stay under the same limit. No setting was changed.

The authors' GC-IDM code was run on our latents and our evaluation, not with the full-scale LeWM encoder it was
designed for, so OIDM-1 is not a reproduction of their published results.
