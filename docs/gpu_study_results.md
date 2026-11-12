<!-- analyze_colab.py --profile free output, Colab T4, tiers 1-3 complete (3 seeds C1/C2/C3/C5; seed 0 for C6/C7), copied 2026-10-27. -->
14 trained models, 150 evaluations; success in % (seeds).

## Success by configuration

| config | System 1, non-trivial +25 | System 1, grasp windows | flow System 1, non-trivial +25 | flow System 1, grasp windows | System 1, non-trivial +50 | two-level, grasp windows | two-level, +50 | best-of-16 by world model | best-of-16, candidates near the mean (std x 0.25) | CEM planner | System 1, replan every block | camera 50% off, hold last | camera 50% off, world-model prediction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 LeWM | 55 ± 1 (3) | 78 ± 3 (3) | 47 ± 1 (3) | 82 ± 2 (3) | 24 ± 1 (3) | 78 ± 3 (3) | 31 ± 2 (3) | 43 ± 3 (3) | 56 ± 1 (3) | 18 ± 1 (3) |  |  |  |
| C2 + proprio grounding | 66 ± 1 (3) | 95 ± 2 (3) | 65 ± 2 (3) | 84 ± 3 (3) | 27 ± 1 (3) | 95 ± 2 (3) | 31 ± 3 (3) | 56 ± 1 (3) | 66 ± 1 (3) |  |  |  |  |
| C3 + object grounding | 82 ± 1 (3) | 93 ± 1 (3) | 72 ± 1 (3) | 86 ± 5 (3) | 33 ± 1 (3) | 93 ± 1 (3) | 52 ± 1 (3) | 63 ± 1 (3) | 80 ± 2 (3) |  | 73 ± 1 (3) | 58 ± 2 (3) | 75 ± 3 (3) |
| C5 hybrid + object | 82 ± 1 (3) | 96 ± 1 (3) | 72 ± 1 (3) | 91 ± 3 (3) | 33 ± 1 (3) | 96 ± 1 (3) | 54 ± 5 (3) | 68 ± 5 (3) | 81 ± 1 (3) | 17 ± 0 (3) | 75 ± 3 (3) | 57 ± 3 (3) | 78 ± 2 (3) |
| C6 + object + ARL | 80 (1) | 95 (1) | 80 (1) | 93 (1) | 35 (1) | 95 (1) | 45 (1) | 70 (1) | 80 (1) |  |  |  |  |
| C7 hybrid + object + ARL | 83 (1) | 93 (1) | 72 (1) | 90 (1) | 35 (1) | 93 (1) | 47 (1) | 60 (1) | 85 (1) |  |  |  |  |

## Paired contrasts (exact McNemar on pairs pooled over seeds; ARL rows: Wilcoxon signed-rank)

| contrast | eval | A | B | units | A − B (pts) | A-only | B-only | p | p (Holm, primary) |
|---|---|---|---|---|---|---|---|---|---|
| **object grounding** | s1_nt | 82% | 66% | 3 | +16.7 ± 0.0 | 34 | 4 | 6.04e-07 | 6.64e-06 |
| **hybrid predictor (object)** | s1_nt | 82% | 82% | 3 | -0.6 ± 1.1 | 9 | 10 | 1 | 1 |
| **ARL (transformer)** | s1_nt | 80% | 80% | 1 | +0.0 | 5 | 5 | 1 | 1 |
| **ARL (hybrid)** | s1_nt | 83% | 82% | 1 | +1.7 | 4 | 3 | 1 | 1 |
| **all best parts vs LeWM** | s1_nt | 83% | 57% | 1 | +26.7 | 18 | 2 | 0.000402 | 0.00362 |
| **object grounding** | s1_pk | 93% | 95% | 3 | -2.2 ± 1.5 | 4 | 8 | 0.388 | 1 |
| **all best parts vs LeWM** | s1_pk | 93% | 72% | 1 | +21.7 | 13 | 0 | 0.000244 | 0.00244 |
| **all configs: two-level vs flat, +50** | hier_50 | 42% | 30% | 14 | +12.5 ± 2.2 | 118 | 13 | 2.39e-22 | 2.87e-21 |
| **all configs: two-level vs flat, grasp** | hier_pk | 91% | 91% | 14 | -0.1 ± 0.1 | 0 | 1 | 1 | 1 |
| **camera 0.5 + prediction: hybrid vs transformer (object)** | cam0.5 | 78% | 75% | 3 | +2.8 ± 1.5 | 16 | 11 | 0.442 | 1 |
| **ARL makes selection useful (C3 → C6)** | sel16 − s1_nt |  |  | 1 | +5.0 |  |  | 0.405 | 1 |
| **ARL makes selection useful (C5 → C7)** | sel16 − s1_nt |  |  | 1 | -16.7 |  |  | 0.048 | 0.384 |
| proprio grounding | s1_nt | 66% | 55% | 3 | +10.6 ± 2.0 | 31 | 12 | 0.0054 | — |
| proprio grounding | s1_pk | 95% | 78% | 3 | +17.2 ± 1.5 | 32 | 1 | 7.92e-09 | — |
| hybrid predictor (object) | s1_pk | 96% | 93% | 3 | +3.3 ± 1.9 | 7 | 1 | 0.0703 | — |
| ARL (transformer) | s1_pk | 95% | 92% | 1 | +3.3 | 3 | 1 | 0.625 | — |
| ARL (hybrid) | s1_pk | 93% | 95% | 1 | -1.7 | 1 | 2 | 1 | — |
| all configs: best-of-16 near the mean vs mean (exploratory) | sel16n | 72% | 73% | 14 | -0.5 ± 0.6 | 32 | 36 | 0.716 | — |
| all configs: flow vs Gaussian System 1, +25 | s1f_nt | 66% | 73% | 14 | -6.9 ± 1.4 | 56 | 114 | 1.02e-05 | — |
| all configs: flow vs Gaussian System 1, grasp | s1f_pk | 86% | 91% | 14 | -4.5 ± 2.0 | 41 | 79 | 0.000667 | — |
| C1 best-of-16 vs mean | sel16 | 43% | 55% | 3 | -12.2 ± 2.4 | 10 | 32 | 0.000941 | — |
| C1 two-level vs flat, +50 | hier_50 | 31% | 24% | 3 | +6.1 ± 3.1 | 12 | 1 | 0.00342 | — |
| C1 two-level vs flat, grasp | hier_pk | 78% | 78% | 3 | +0.0 ± 0.0 | 0 | 0 | 1 | — |
| C2 best-of-16 vs mean | sel16 | 56% | 66% | 3 | -9.4 ± 2.0 | 11 | 28 | 0.00948 | — |
| C2 two-level vs flat, +50 | hier_50 | 31% | 27% | 3 | +4.4 ± 2.0 | 12 | 4 | 0.0768 | — |
| C2 two-level vs flat, grasp | hier_pk | 95% | 95% | 3 | +0.0 ± 0.0 | 0 | 0 | 1 | — |
| C3 best-of-16 vs mean | sel16 | 63% | 82% | 3 | -19.4 ± 2.2 | 3 | 38 | 1.05e-08 | — |
| C3 two-level vs flat, +50 | hier_50 | 52% | 33% | 3 | +19.4 ± 1.5 | 39 | 4 | 3.11e-08 | — |
| C3 two-level vs flat, grasp | hier_pk | 93% | 93% | 3 | +0.0 ± 0.0 | 0 | 0 | 1 | — |
| C5 best-of-16 vs mean | sel16 | 68% | 82% | 3 | -13.3 ± 5.9 | 9 | 33 | 0.000272 | — |
| C5 two-level vs flat, +50 | hier_50 | 54% | 33% | 3 | +21.1 ± 3.1 | 38 | 0 | 7.28e-12 | — |
| C5 two-level vs flat, grasp | hier_pk | 96% | 96% | 3 | -0.6 ± 0.6 | 0 | 1 | 1 | — |
| C6 best-of-16 vs mean | sel16 | 70% | 80% | 1 | -10.0 | 2 | 8 | 0.109 | — |
| C6 two-level vs flat, +50 | hier_50 | 45% | 35% | 1 | +10.0 | 9 | 3 | 0.146 | — |
| C6 two-level vs flat, grasp | hier_pk | 95% | 95% | 1 | +0.0 | 0 | 0 | 1 | — |
| C7 best-of-16 vs mean | sel16 | 60% | 83% | 1 | -23.3 | 2 | 16 | 0.00131 | — |
| C7 two-level vs flat, +50 | hier_50 | 47% | 35% | 1 | +11.7 | 8 | 1 | 0.0391 | — |
| C7 two-level vs flat, grasp | hier_pk | 93% | 93% | 1 | +0.0 | 0 | 0 | 1 | — |
| C1 System 1 vs CEM | cem | 55% | 18% | 3 | +37.2 ± 1.1 | 82 | 15 | 2.38e-12 | — |
| C3 camera 0.5: prediction vs hold | cam0.5 | 75% | 58% | 3 | +17.2 ± 2.0 | 39 | 8 | 5.54e-06 | — |
| C5 camera 0.5: prediction vs hold | cam0.5 | 78% | 57% | 3 | +21.1 ± 3.1 | 48 | 10 | 4.52e-07 | — |

Bold rows are the pre-registered family (Holm-corrected together); the rest are exploratory. Units = (config, seed) groups of paired start/goal pairs.

## Diagnostics (seed 0)

| config | true chunk vs 15 real (top-1) | vs perturbed σ=0.1 | σ=0.3 | rollout error / copy | cube probe (cm) | cube probe, predicted (cm) | R² arm | R² cube |
|---|---|---|---|---|---|---|---|---|
| C1 LeWM | 0.96 | 0.02 | 0.36 | 0.063 | 5.0 | 6.7 | 0.50 | 0.008 |
| C2 + proprio grounding | 1.00 | 0.08 | 0.49 | 0.032 | 6.0 | 8.0 | 0.32 | 0.028 |
| C3 + object grounding | 1.00 | 0.05 | 0.44 | 0.028 | 2.0 | 2.4 | 0.49 | 0.014 |
| C5 hybrid + object | 1.00 | 0.05 | 0.46 | 0.026 | 2.1 | 2.6 | 0.42 | 0.007 |
| C6 + object + ARL | 0.99 | 0.03 | 0.39 | 0.041 | 2.0 | 3.1 | 0.52 | 0.008 |
| C7 hybrid + object + ARL | 0.99 | 0.22 | 0.72 | 0.032 | 2.2 | 3.3 | 0.43 | 0.005 |

GPU study: cube (free profile)
14 trained models, 150 evaluations; success in % (seeds).

Success by configuration
config	System 1, non-trivial +25	System 1, grasp windows	flow System 1, non-trivial +25	flow System 1, grasp windows	System 1, non-trivial +50	two-level, grasp windows	two-level, +50	best-of-16 by world model	best-of-16, candidates near the mean (std x 0.25)	CEM planner	System 1, replan every block	camera 50% off, hold last	camera 50% off, world-model prediction
C1 LeWM	55 ± 1 (3)	78 ± 3 (3)	47 ± 1 (3)	82 ± 2 (3)	24 ± 1 (3)	78 ± 3 (3)	31 ± 2 (3)	43 ± 3 (3)	56 ± 1 (3)	18 ± 1 (3)			
C2 + proprio grounding	66 ± 1 (3)	95 ± 2 (3)	65 ± 2 (3)	84 ± 3 (3)	27 ± 1 (3)	95 ± 2 (3)	31 ± 3 (3)	56 ± 1 (3)	66 ± 1 (3)				
C3 + object grounding	82 ± 1 (3)	93 ± 1 (3)	72 ± 1 (3)	86 ± 5 (3)	33 ± 1 (3)	93 ± 1 (3)	52 ± 1 (3)	63 ± 1 (3)	80 ± 2 (3)		73 ± 1 (3)	58 ± 2 (3)	75 ± 3 (3)
C5 hybrid + object	82 ± 1 (3)	96 ± 1 (3)	72 ± 1 (3)	91 ± 3 (3)	33 ± 1 (3)	96 ± 1 (3)	54 ± 5 (3)	68 ± 5 (3)	81 ± 1 (3)	17 ± 0 (3)	75 ± 3 (3)	57 ± 3 (3)	78 ± 2 (3)
C6 + object + ARL	80 (1)	95 (1)	80 (1)	93 (1)	35 (1)	95 (1)	45 (1)	70 (1)	80 (1)				
C7 hybrid + object + ARL	83 (1)	93 (1)	72 (1)	90 (1)	35 (1)	93 (1)	47 (1)	60 (1)	85 (1)				
Paired contrasts (exact McNemar on pairs pooled over seeds; ARL rows: Wilcoxon signed-rank)
contrast	eval	A	B	units	A − B (pts)	A-only	B-only	p	p (Holm, primary)
object grounding	s1_nt	82%	66%	3	+16.7 ± 0.0	34	4	6.04e-07	6.64e-06
hybrid predictor (object)	s1_nt	82%	82%	3	-0.6 ± 1.1	9	10	1	1
ARL (transformer)	s1_nt	80%	80%	1	+0.0	5	5	1	1
ARL (hybrid)	s1_nt	83%	82%	1	+1.7	4	3	1	1
all best parts vs LeWM	s1_nt	83%	57%	1	+26.7	18	2	0.000402	0.00362
object grounding	s1_pk	93%	95%	3	-2.2 ± 1.5	4	8	0.388	1
all best parts vs LeWM	s1_pk	93%	72%	1	+21.7	13	0	0.000244	0.00244
all configs: two-level vs flat, +50	hier_50	42%	30%	14	+12.5 ± 2.2	118	13	2.39e-22	2.87e-21
all configs: two-level vs flat, grasp	hier_pk	91%	91%	14	-0.1 ± 0.1	0	1	1	1
camera 0.5 + prediction: hybrid vs transformer (object)	cam0.5	78%	75%	3	+2.8 ± 1.5	16	11	0.442	1
ARL makes selection useful (C3 → C6)	sel16 − s1_nt			1	+5.0			0.405	1
ARL makes selection useful (C5 → C7)	sel16 − s1_nt			1	-16.7			0.048	0.384
proprio grounding	s1_nt	66%	55%	3	+10.6 ± 2.0	31	12	0.0054	—
proprio grounding	s1_pk	95%	78%	3	+17.2 ± 1.5	32	1	7.92e-09	—
hybrid predictor (object)	s1_pk	96%	93%	3	+3.3 ± 1.9	7	1	0.0703	—
ARL (transformer)	s1_pk	95%	92%	1	+3.3	3	1	0.625	—
ARL (hybrid)	s1_pk	93%	95%	1	-1.7	1	2	1	—
all configs: best-of-16 near the mean vs mean (exploratory)	sel16n	72%	73%	14	-0.5 ± 0.6	32	36	0.716	—
all configs: flow vs Gaussian System 1, +25	s1f_nt	66%	73%	14	-6.9 ± 1.4	56	114	1.02e-05	—
all configs: flow vs Gaussian System 1, grasp	s1f_pk	86%	91%	14	-4.5 ± 2.0	41	79	0.000667	—
C1 best-of-16 vs mean	sel16	43%	55%	3	-12.2 ± 2.4	10	32	0.000941	—
C1 two-level vs flat, +50	hier_50	31%	24%	3	+6.1 ± 3.1	12	1	0.00342	—
C1 two-level vs flat, grasp	hier_pk	78%	78%	3	+0.0 ± 0.0	0	0	1	—
C2 best-of-16 vs mean	sel16	56%	66%	3	-9.4 ± 2.0	11	28	0.00948	—
C2 two-level vs flat, +50	hier_50	31%	27%	3	+4.4 ± 2.0	12	4	0.0768	—
C2 two-level vs flat, grasp	hier_pk	95%	95%	3	+0.0 ± 0.0	0	0	1	—
C3 best-of-16 vs mean	sel16	63%	82%	3	-19.4 ± 2.2	3	38	1.05e-08	—
C3 two-level vs flat, +50	hier_50	52%	33%	3	+19.4 ± 1.5	39	4	3.11e-08	—
C3 two-level vs flat, grasp	hier_pk	93%	93%	3	+0.0 ± 0.0	0	0	1	—
C5 best-of-16 vs mean	sel16	68%	82%	3	-13.3 ± 5.9	9	33	0.000272	—
C5 two-level vs flat, +50	hier_50	54%	33%	3	+21.1 ± 3.1	38	0	7.28e-12	—
C5 two-level vs flat, grasp	hier_pk	96%	96%	3	-0.6 ± 0.6	0	1	1	—
C6 best-of-16 vs mean	sel16	70%	80%	1	-10.0	2	8	0.109	—
C6 two-level vs flat, +50	hier_50	45%	35%	1	+10.0	9	3	0.146	—
C6 two-level vs flat, grasp	hier_pk	95%	95%	1	+0.0	0	0	1	—
C7 best-of-16 vs mean	sel16	60%	83%	1	-23.3	2	16	0.00131	—
C7 two-level vs flat, +50	hier_50	47%	35%	1	+11.7	8	1	0.0391	—
C7 two-level vs flat, grasp	hier_pk	93%	93%	1	+0.0	0	0	1	—
C1 System 1 vs CEM	cem	55%	18%	3	+37.2 ± 1.1	82	15	2.38e-12	—
C3 camera 0.5: prediction vs hold	cam0.5	75%	58%	3	+17.2 ± 2.0	39	8	5.54e-06	—
C5 camera 0.5: prediction vs hold	cam0.5	78%	57%	3	+21.1 ± 3.1	48	10	4.52e-07	—
Bold rows are the pre-registered family (Holm-corrected together); the rest are exploratory. Units = (config, seed) groups of paired start/goal pairs.

Diagnostics (seed 0)
config	true chunk vs 15 real (top-1)	vs perturbed σ=0.1	σ=0.3	rollout error / copy	cube probe (cm)	cube probe, predicted (cm)	R² arm	R² cube
C1 LeWM	0.96	0.02	0.36	0.063	5.0	6.7	0.50	0.008
C2 + proprio grounding	1.00	0.08	0.49	0.032	6.0	8.0	0.32	0.028
C3 + object grounding	1.00	0.05	0.44	0.028	2.0	2.4	0.49	0.014
C5 hybrid + object	1.00	0.05	0.46	0.026	2.1	2.6	0.42	0.007
C6 + object + ARL	0.99	0.03	0.39	0.041	2.0	3.1	0.52	0.008
C7 hybrid + object + ARL	0.99	0.22	0.72	0.032	2.2	3.3	0.43	0.005
