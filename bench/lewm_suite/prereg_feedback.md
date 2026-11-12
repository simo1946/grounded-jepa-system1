# Pre-registration: how often the controller looks (feedback frequency)

Written 2026-11-06, before any run below. Nothing in this file is changed after the runs start; deviations
are reported separately.

## Question

System 1 executes a 25-step action chunk before it sees a new camera frame. GC-IDM (Latent Geometry Beyond
Search, arXiv 2605.08732) predicts one action, sees a new frame after every step, and is told how many steps
remain. How much of a controller's success comes from how often it looks, and how much from the controller
itself?

## Models (fixed, not retrained)

CPU-study world models on OGBench Cube (single view), 400 training episodes of 201 steps:

| Name | Spec | Parameters | Training |
|---|---|---|---|
| C1 | `lewm` (no grounding) | 17.92M | 2,000 steps, batch 32 |
| C3 | `lewm+idm+sa+obj` (inverse dynamics + state alignment with cube position) | 18.35M | 2,000 steps, batch 32 |

Encoder: ViT-tiny from scratch (12 layers, width 192, 3 heads, patch 8), 64-pixel RGB frames, 192-d latent.
Two training seeds each (0, 1): four world models. These are smaller than the GPU-study models
(600 episodes, 4,000 steps, batch 64); results are reported as a CPU-scale follow-up.

## Controllers (trained on the same frozen latents of each world model)

All controllers read only the encoder: z_now (current frame) and z_goal (goal frame). The world-model
predictor is not used. Training data: the latent cache of the 400 training episodes (latent_cache.py).

1. **System 1** (`ChunkPolicy`, 0.69M parameters). Input [z_now, z_goal, z_goal - z_now]; output a Gaussian
   over 5 blocks x 5 steps x 5 action numbers. Hindsight goals 1..5 blocks ahead, masked Gaussian NLL.
   AdamW lr 1e-3, weight decay 1e-4, cosine, clip 1.0, batch 256, 3,000 steps (the GPU-study recipe).
   Acts with the mean.
2. **GC-IDM-style** (our reimplementation from the paper's description; its code is not released).
   3-layer MLP, width 512, LayerNorm, GELU, dropout 0.1, input [z_now, z_goal]. Remaining horizon h:
   h / 50, 64-d sinusoidal encoding, 2-layer MLP, AdaLN-Zero modulation (zero-initialised) of the backbone
   output, linear head to one action (5 numbers, z-scored like System 1's). Training pairs
   (z_t, z_{t+h}, a_t) with h uniform in 1..50 steps inside an episode; MSE. AdamW lr 1e-3, weight decay
   1e-4, cosine to lr/100, clip 1.0, batch 1,024, 50 epochs. At test time h_t = T - t + 1 with T = 50
   (the step budget), clamped to 1..50, as in the paper.
3. **GC-IDM-style without horizon**: identical, with the horizon branch removed.

## Evaluation (identical for every condition)

Cube, 60 non-trivial start/goal pairs (goal 25 steps after the start, 50-step budget, evaluation seed 42),
the same 60 pairs for every model and controller. CPU, 5 environments simulated at once. Deterministic
actions (no sampling, no world-model selection).

| Condition | Controller | Steps executed between camera frames |
|---|---|---|
| S1-25 | System 1 | 25 (paper default) |
| S1-5 | System 1 | 5 |
| S1-1 | System 1 | 1 (first step of the chunk, then a new decision) |
| IDM-1 | GC-IDM-style | 1 |
| IDM-1-noH | GC-IDM-style without horizon | 1 |

GC-IDM predicts one action, so it has no open-loop 5- or 25-step version without extra machinery; it is
run only at 1 step, as in its paper.

Also recorded: median wall-clock time per decision and per episode (CPU, 2 threads).

## Hypotheses and contrasts (paired, same 60 pairs)

- **H1a** S1-5 vs S1-25. **H1b** S1-1 vs S1-25. Prior evidence: in the CPU study, System 1 deciding every
  block (5 steps) did worse than every 25 steps (C2 seed 0: 30% vs 50%; hybrid: 30% vs 62%). Expectation:
  looking more often does not help this chunk policy; direction for 1 step unknown.
- **H2** IDM-1 vs S1-1: same feedback frequency, different controller.
- **H3** IDM-1 vs S1-25: each controller in its own native mode (the practical comparison).
- **H4** IDM-1 vs IDM-1-noH: effect of the horizon input (the GC-IDM paper reports a drop of 42 points on
  average without it).

## Statistics and decision rule

Exact McNemar test (two-sided binomial on discordant pairs) per model. Pooled over the four models: exact
McNemar on all 240 paired outcomes, Holm-corrected over the five contrasts. Because the same 60 pairs are
reused by every model, pooled p-values overstate the evidence and are descriptive. An effect is claimed
only if (a) its sign is the same in all four models and (b) the pooled Holm-corrected p < 0.05. C1 vs C3
differences are descriptive.

## What would change the paper

If looking more often helps System 1 (H1), the 25-step chunk is a limitation of our controller, and the
paper says so. If IDM-1 beats System 1 at equal feedback (H2), the gain of step-wise inverse dynamics is not
only feedback frequency. All outcomes are reported, including null ones.

---

## Addendum, 2026-11-09 (written before any of the runs below; the plan above is unchanged)

**Correction.** The plan above says GC-IDM's code "is not released". That was wrong: the authors' repository
(github.com/hdnndh/Latent-Geometry-Beyond-Search-Amortizing-Planning-in-World-Models) was public from
May 2026 (commit 48c45b1, 24 May 2026). Compared with that code, our reimplementation (`gcidm.py`) differs in
four ways: it multiplies the horizon fraction by 1,000 before the sinusoidal encoding; it uses PyTorch's default
initialisation (theirs: Kaiming, output layer std 0.01, AdaLN zero); it trains on standardised actions
(theirs: raw); it keeps the final checkpoint (theirs: the best on a random 10% validation split of pairs).
Their "no horizon" ablation trains with H_max = 1 (goal always the next frame), not by removing the horizon
input, so our IDM-1-noH is a different ablation from the one behind their 42-point figure.

**New conditions** (same four world models, same 60 pairs, same evaluation settings as above):

| Condition | Controller | Steps between frames |
|---|---|---|
| OIDM-1 | Authors' GC-IDM code, unmodified (`train_idm.py train`, their Cube settings: 50 epochs, lr 1e-3, batch 1,024, H_max 50, seed 42), trained on our latents of the same 400 training episodes with raw actions | 1 |
| OIDM-1-H1 | The same with H_max = 1 (their ablation) | 1 |
| S1c5 | System 1 trained to output a 5-step chunk (ChunkPolicy with one block; same recipe and goal distances, 5 to 25 steps) | 5 (whole chunk) |
| S1c1 | System 1 trained to output a 1-step chunk (same, action of one step) | 1 (whole chunk) |

The authors' code is licensed CC BY-NC 4.0, so it is not copied into this repository; `official_gcidm.py`
exports our latents to their file format, calls their scripts from a separate clone at commit 48c45b1, and
evaluates their model through a thin adapter (encode frame and goal with our encoder, call their model with
the remaining steps min(50 - steps done, H_max), standardise its raw action for the evaluation pipeline).

**Contrasts added:** H5 OIDM-1 vs IDM-1 (our reimplementation vs theirs); H6 OIDM-1 vs S1-25; H7 OIDM-1 vs
OIDM-1-H1 (their horizon ablation); H8 S1c5 vs S1-5 (trained for 5 steps vs cut to 5 steps); H9 S1c1 vs S1-1;
H10 S1c5 vs S1-25; H11 S1c1 vs S1-25.

**Statistics (replaces the pooled McNemar test as the decision rule, for H1-H11).** The 60 pairs come from
40 evaluation episodes and are shared by the four models, so the unit of resampling is the evaluation
episode: 10,000 bootstrap resamples of the 40 episodes, each keeping all of its pairs and all four models.
Statistic: mean success difference (A - B) over models and pairs, with a 95% percentile interval and a
two-sided bootstrap p (twice the smaller tail), Holm-corrected over H1-H11. Per model: wins and losses and the
same interval over that model's pairs. An effect is claimed only if (a) its Holm-corrected p < 0.05 and (b)
its sign is the same in all four models. Pooled McNemar p-values are still reported, as descriptive.
The same bootstrap (resampling the 60 pairs, which are all that the GPU result files identify) is applied to
the GPU study's pre-registered contrasts; the CPU data show how much clustering by episode widens the intervals.
