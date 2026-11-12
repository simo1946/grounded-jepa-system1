TITLE: Grounded JEPA World Models for Fast System-1 Control

AUTHOR: Simone Bonaccorsi

AFFIL: AI Researcher, Italy · simone.bonaccorsi46@gmail.com


ABSTRACT: Joint-embedding predictive (JEPA) world models such as LeWM usually act by planning: they search for action sequences whose imagined outcome matches a goal image. When such a model is small, what limits goal-conditioned control: what the latent represents, how the controller is structured, or how actions are chosen? We study this in a deliberately small setting: 64-pixel frames, a ViT-tiny encoder, about 18 M parameters and one free GPU, on the Cube manipulation task of OGBench. The standard Cube protocol is dominated by pairs that are already solved at the start (70.5%), so we evaluate on non-trivial pairs, with hypotheses fixed before the runs, paired exact tests and three training seeds, and report each seed separately. Three findings hold in every seed. (1) A goal-conditioned policy over the world model's latent ("System 1") beats planning with CEM at this scale, 55% vs 18%, and decides over 500 times faster in our CPU benchmark. (2) Adding the object's position as a training-only target for the latent raises success from 66% to 82% and makes the cube readable from the latent to 2 cm instead of 6 cm. (3) A slow subgoal network over System 1 raises success on goals twice as far away (52% vs 33% on the object-grounded model), and the gain recurs on a second environment, TwoRoom. When half the camera frames are lost, filling them with the world model's prediction beats holding the last frame (75% vs 58%) and approximately recovers the camera-on score. In contrast, using the world model to choose among System 1's proposals lowered success, and even a privileged judge that executes every candidate adds only 2 of 50 episodes for the candidates tested. A hybrid Mamba-3/attention predictor matches the transformer. In a pre-registered follow-up on smaller models, making System 1 decide after every step lowered its success from 62.5% to 36%, also when it was trained to output a single step, whereas step-wise inverse-dynamics controllers worked at that rate (the GC-IDM authors' code 54%, our reimplementation 60%); we did not detect a difference between them and System 1 committing to 25 steps. In this reduced-scale setting, how well the latent represents the task's object appears to matter more than online action search, and the world model is most useful as a source of training signals, subgoals and memory.

# 1. Introduction

A world model learns how the world changes when an agent acts. JEPA-style world models do this in a learned latent space: an encoder maps each camera frame to a compact vector, and a predictor learns how actions move that vector [1, 2]. The usual way to act with such a model is to plan. A planner such as the cross-entropy method (CEM) samples many action sequences, rolls each one forward in the latent, and executes the sequence whose imagined end state is closest to the goal image's latent. Planning is slow, and it works only if two things hold: the latent must contain the state the task is scored on, and the predictor must be fine enough to tell nearby actions apart. A small model may have neither.

This study started from a question in biology: whether a fast, direct decision step, which we call System 1 after the fast mode of thinking in dual-process accounts of cognition, could make automated experiments faster and more reliable. Robotic control is the most direct place to test such a decision step, so we test it there first; the robotics results do not by themselves establish anything about biological systems.

We take a small LeWM-style world model [1] on the Cube task of OGBench [3], a robot arm that must move a cube to a goal pose, and ask what limits goal-conditioned control when the model is small: what the latent represents, how the controller is structured, or how actions are chosen. We change one component at a time and compare with paired statistics over three training seeds. In this setting, how actions are searched for is not the main lever. A fast policy that maps the current latent and the goal latent directly to actions does better than search, and grounding the latent on the object the task is scored on matters most. Our contributions are:

-   **A complementary evaluation for Cube.** Of all 70,400 start/goal pairs 25 steps apart in 400 episodes from the authors' expert data collector, 70.5% are already solved at the start, and a policy taking uniformly random actions scores 64% on the standard 50 pairs. The standard protocol therefore mostly measures not undoing a solved goal. We add an evaluation on pairs that are not solved at the start and on grasp-centred windows, with exact McNemar tests, pre-specified hypotheses and three seeds reported separately.

-   **Training-only object grounding.** Recent work grounds JEPA latents on the robot's own state [19, 20]; adding the object's position to that target raises success by a further 16 points and makes the cube readable from the latent to 2 cm (from 5--6 cm). At test time the model sees pixels only.

-   **Amortised control beats planning here.** A goal-conditioned policy over the latent ("System 1") beats CEM by 37 points at this scale, with over 500 times lower decision latency in our CPU benchmark, in line with independent recent work that amortises planning in LeWM latents [22, 23].

-   **How often to look.** Deciding after every step lowers System 1 from 62.5% to 36% in every model tested, also when System 1 is trained to output one step. GC-IDM's step-wise inverse dynamics, run with the authors' code and with our reimplementation, works at that rate (54% and 60%), but we detected no gain over System 1's 25-step chunks.

-   **Hierarchy and memory where they help.** A slow subgoal network helps goals 50 steps away on the object-grounded models (+19 and +21 points, in every seed; also on TwoRoom) and is switched off for short goals. The world model's prediction is a better stand-in for a lost camera frame than the last frame (+17 to +21 points).

-   **Negative results with a diagnosis.** World-model selection, a learned outcome critic, a contrastive verifier, a privileged true-outcome judge, reward-weighted fine-tuning and residual practice do not beat System 1's mean action. Even perfect selection among the 16 candidates adds only 2 of 50 episodes, so for these proposals the limit is their quality rather than the judge.

-   **A hybrid Mamba-3/attention predictor** that matches the transformer predictor (82% vs 82%) and can also run step by step as a state estimator.

# 2. Related work

**JEPA world models and latent planning.** LeWM trains an encoder and a predictor end to end from pixels, with a prediction loss and the SIGReg regulariser from LeJEPA [4], and plans with CEM in the latent [1]. DINO-WM plans over frozen DINOv2 features [2]. H-JEPA learns hierarchical JEPA world models end to end for visual planning [5]; we use its grasp-centred Cube windows as a second evaluation protocol. We use LeWM's code, data collector and evaluation through the stable-worldmodel library [6], and change what sits around the latent.

**Grounding the latent.** Inverse dynamics is used as an auxiliary loss for JEPA world models; in SMWM it is the only term that prevents representation collapse [7]. Our state-prediction head adds a different target: the robot's own state and, for object grounding, the object's position. It is used only in training. The novelty is the controlled measurement of what this changes for control, not the head itself.

**Amortised and hierarchical control.** System 1 is goal-conditioned behaviour cloning with hindsight relabelling, as in GCSL [8]. The subgoal network resembles HIQL [9]. We compare a Gaussian action head against an expressive flow-matching head, the family that includes Diffusion Policy [10] and rectified flow [11].

**Model-based verification and practice.** Verifying a policy's proposal with a model before acting, and residual reinforcement learning on top of a frozen policy (ResFiT [12], RFS [13]), motivate our selection and practice experiments. Potential-based reward shaping follows Ng et al. [14]. DAgger [15] is our corrective-label baseline. Our contrastive verifier follows the bidirectional InfoNCE objective of CLM [16], as used to verify robot actions in CoVer-VLA [17].

**Recent and concurrent work.** This study was developed independently of, and in parallel with, several 2025--2026 studies on the same benchmarks that reach compatible conclusions by different routes. On grounding, PSG-JEPA grounds the latent on robot proprioception and joint-angle changes with training-only heads [19]; Liu et al. add inverse dynamics and state alignment to an end-to-end JEPA world model [20]; PhyLatent adds physical-state grounding and a counterfactual action-separation loss [21], related to our action-resolution loss. On control, GC-IDM amortises LeWM's planning into a goal-conditioned inverse-dynamics policy that beats CEM at a fraction of the cost [22]. It differs from our System 1 in ways that likely matter for absolute scores: it acts one step at a time from a fresh observation where System 1 commits to 25 steps, it is told the remaining horizon, and it uses the frozen full-scale LeWM encoder, evaluated on the standard protocol (Section 5.8 tests the first two differences on our latents); LeFlow replaces CEM with flow-based latent planning [23]; LeWAM learns an action head jointly with the world model and finds that refining its proposals through the model helps on Push-T [24], whereas selection did not help in our Cube setting; Terver et al. study which design choices drive planning success with JEPA world models [25]; SpecWM replaces LeWM's anti-collapse regulariser with a loss that aligns the latent geometry with a target similarity kernel (temporal, quasimetric or state-based), raising CEM success on Cube from 52% to 68% [26], a different route to the control-relevant representation that our grounding heads target. At the other end of the scale, RoboJEPA trains action-conditioned JEPA predictors of up to 8B parameters on about 15,000 hours of robot video and plans with CEM [27]; it asks how world models scale, not how a small model is best used for control. What this study adds is a controlled, per-seed comparison of these ingredients in one setting: the object-position target against proprioception-only grounding, the non-trivial-pair evaluation, a privileged-ceiling diagnosis of why selection fails, and the hierarchy and memory results. Most of these works report Cube success on the standard protocol, where about 70% of pairs are solved at the start, which should be kept in mind when comparing numbers.

**State-space predictors.** Our hybrid predictor interleaves Mamba-3 layers [18] with attention, so that it can update the latent one step at a time like a filter.

# 3. Method

The system is LeWM's world model with training-only grounding heads, plus a fast policy that acts in its latent. The world model provides training signals, subgoals and state estimates; it does not choose every action (Figure 1).


FIGURE: figures/fig1_system.png | 6.4 | Figure 1. The system. A ViT-tiny encoder maps each 64-pixel frame to a 192-d latent; a predictor (transformer, or hybrid Mamba-3/attention) predicts the next latent from the last latents and a 5-step action block. Training-only heads (dashed) ground the latent; they are removed at test time. System 1 maps the current and goal latents to the next 25 actions; for far goals a subgoal network first proposes an intermediate latent.

**Units of time.** One *step* is one robot action (hand displacement, rotation and gripper command). One *block* is 5 steps; the world model predicts one latent per block. System 1 outputs a *chunk* of 5 blocks (25 steps). By default the robot executes the whole chunk and then observes and replans, so it decides every 25 steps; in the camera-dropout and memory experiments it decides every block (5 steps), and Section 5.8 also tests deciding after every step.

**World model.** A ViT-tiny encoder (patch size 8 on 64-pixel frames) maps each frame to a 192-dimensional latent. A 6-layer transformer predictor takes the last 3 latents and an embedded block of 5 actions and predicts the next latent. The loss is the mean squared error of the predicted latent plus 0.09 × SIGReg, as in LeWM [1]. Each model has about 18 M parameters.

**Grounding heads (training only).** An inverse-dynamics head predicts the action block from two consecutive latents (weight 1). A state-alignment head predicts the next state and its change from consecutive latents (weight 1). The state is the robot's proprioception (configuration C2) or proprioception plus the cube's position (C3, "object grounding"). The heads are discarded after training, so at test time the model sees pixels only. Inverse dynamics follows SMWM [7]; the state-alignment idea follows [20], with our own loss.

**Hybrid predictor (C4, C5).** The transformer is replaced by 6 layers in a 3:1 pattern of Mamba-3 and attention layers, with a mixture-of-experts feed-forward block (8 routed experts, top-2 routing, 1 shared expert). The recurrent layers let the model update the latent one step at a time.

**System 1.** A goal-conditioned policy, a 3-layer MLP of 0.69 M parameters, maps the current latent, the goal latent and their difference to a Gaussian over the next 5 blocks × 5 steps of actions. It is trained on cached latents with hindsight goals 1 to 5 blocks ahead: for each training window, a later frame of the same episode is used as the goal and the actions actually taken are the target. At test time it executes the mean chunk and replans after the whole chunk (every 25 steps). A flow-matching head (rectified flow, 10 integration steps) is the expressive alternative.

**How System 1 is supervised.** Actions are normalised with a scaler fitted on the training data, and the Gaussian is factorised over every action number with a learned spread. For a training window, a goal k blocks ahead is drawn uniformly from 1 to 5; the target is the 5-block chunk actually executed, and blocks after the goal frame are masked out of the loss (a masked Gaussian negative log-likelihood), since they do not help reach that goal. Evaluation goals 25 steps ahead are exactly 5 blocks, inside this training range; goals 50 steps ahead are beyond it, which is what the subgoal network is for. An episode counts as a success as soon as the goal condition holds at any step within the budget.

**Two-level System 1.** A subgoal network maps the current and goal latents to an intermediate latent, which System 1 then pursues. A gate fixed before evaluation uses the subgoal only when the squared latent distance to the goal exceeds the median distance covered in one policy horizon on training data, so short tasks are left to System 1 alone.

**Selection and verify-then-act.** For selection, 16 candidate chunks (the mean plus 15 samples) are rolled out in the world model, and the one whose imagined end state is closest to the goal latent is executed. Verify-then-act accepts System 1's chunk when its imagined cost falls to 0.6 × the current cost or below one typical block displacement, and otherwise falls back to CEM.

**Memory under camera dropout.** With decisions every block, each frame is lost independently with probability 0.5. A lost frame is replaced by a blank, by the last frame seen, or by the world model's prediction from the last latent and the actions executed since.

**Practice with a real-world measure.** A residual network adds a bounded correction to System 1's chunk and is trained in the simulator by a group-relative policy gradient. The reward is the real cube-to-goal distance plus a success bonus, optionally shaped by progress and a grasp milestone. In the control arm the world model's imagined distance replaces the real one.

# 4. Experimental setup

All runs use the LeWM authors' Cube environment, expert data collector and evaluation code [1, 6], at reduced scale (Table 1). Contrasts inside this study are fair; absolute numbers are not comparable with published ones.

TABLE: scale
<!-- caption: Table 1. Experimental scale. The GPU study is the main study; the CPU study holds the follow-up experiments. -->
<!-- |                              | **GPU study (main)**    | **CPU study (follow-ups)** | **LeWM / H-JEPA papers** | -->
<!-- |----------------------|-----------------|-----------------|-----------------| -->
<!-- | Training / held-out episodes | 600 / 100               | 400 / 80                   | about 10,000             | -->
<!-- | Frames                       | 64 px                   | 64 px                      | 224 px                   | -->
<!-- | World-model training         | 4,000 steps, batch 64   | 2,000 steps, batch 32      | several epochs on GPU    | -->
<!-- | System 1 / subgoal training  | 3,000 steps each        | 2,000--6,000 steps         | --                       | -->
<!-- | Evaluation pairs             | 60 non-trivial per seed | 50 non-trivial             | 50 standard              | -->
<!-- | Seeds                        | 3 (C6, C7: 1)           | 1--2                       | --                       | -->
**Configurations.** C1 is LeWM as published, at our scale. C2 adds inverse dynamics and proprioceptive state alignment. C3 adds the cube's position to the state target (object grounding). C5 is the hybrid predictor with object grounding. C6 and C7 add an action-resolution loss to C3 and C5 (Appendix B.3). C4 (hybrid with proprioception only), C8 (inverse dynamics only) and C9 (a MobileNetV4-S encoder) appear only in the CPU study.

**Protocols.** (a) Non-trivial pairs: a start frame and a goal frame 25 steps later from a held-out episode, kept only if the start does not already satisfy the goal (cube within 4 cm); budget 50 steps. (b) Grasp-centred windows from the H-JEPA code [5]: goal 20 steps ahead, budget 30. (c) Long horizon: goal 50 steps ahead, budget 100. A fixed evaluation seed (42) selects the pairs, so every method is scored on identical pairs. The authors' standard protocol, which includes trivial pairs, is reported in Appendix A.

**Statistics.** Each contrast compares two methods on identical start/goal pairs. Because the same pairs are shared by every seed and model, the main evidence is a cluster bootstrap: the pairs are resampled with replacement, each drawn pair bringing its outcomes for all seeds together (10,000 resamples), giving a 95% interval for the success difference and a two-sided bootstrap p. In the CPU follow-up the pair identities were recorded and several pairs share an evaluation episode, so whole episodes are resampled there (40 episodes behind 60 pairs); an effect is claimed there only if its bootstrap p, Holm-corrected over the study's 12 contrasts, is below 0.05 and it points the same way in all four models. The exact McNemar test, which counts the pairs on which exactly one method succeeds ("discordant" pairs) and pools seeds as if independent, is also reported, as descriptive. The hypotheses and analysis plan were written into the notebook before the GPU run (pre-specified, not publicly registered). The primary contrasts form one family and are Holm-corrected together (the bootstrap p-values were added after the run, replacing the pooled McNemar p as the basis for claims): object grounding (two protocols), the action-resolution loss (two configurations, plus two Wilcoxon tests of whether it makes selection useful), the hybrid predictor, the hybrid as a state estimator under dropout, all best parts vs LeWM (two protocols), and two-level vs flat System 1 (two protocols). Everything else is exploratory. Follow-up experiments on CPU have their own dated plan files, written before those runs, with later additions recorded as addenda.

**Units and dependence.** The paired unit is a start/goal pair. The same 60 pairs are used for every model and seed, so the three seeds are independent training runs evaluated on one shared set of scenarios, not independent evaluation sets. An effect that holds in every seed is robust to training randomness; it does not by itself show that the effect generalises to other scenario sets. A pooled McNemar test treats the outcomes of different seeds (and, where stated, different models) as independent although they share these pairs, so its p-values overstate the evidence; the bootstrap does not. We report every main contrast for each seed separately and with its bootstrap interval (Table 4) and claim only effects whose interval excludes zero and that point the same way in all three seeds. ± values are standard errors of the success rate across the three seeds. A linear (ridge) probe fitted on training frames and scored on held-out frames gives the read-out errors, reported as mean Euclidean distance in cm.

# 5. Results

Object grounding, System 1 over planning, hierarchy on the object-grounded models and prediction-based memory each give large gains that point the same way in every seed, while choosing actions with the world model lowers success (Tables 2--4, Figure 2).

TABLE: hypotheses
<!-- caption: Table 2. Hypotheses fixed before the GPU run (C6 and C7 have one seed). Differences in percentage points with 95% intervals from resampling the 60 shared pairs (all seeds together, 10,000 resamples); p: bootstrap p, Holm-corrected over the primary family (H3: Wilcoxon). Table 4 gives each seed. -->
<!-- | **\#** | **Hypothesis**                                                      | **Result**                                                                                                                                                                   | **Verdict**                         | -->
<!-- |------|-----------------------|----------------------------|-----------------| -->
<!-- | H1     | Object grounding improves System 1                                  | 82% vs 66%: +16.7 [8.3, 25.6], Holm p ≤ 0.003; grasp 93% vs 95% (ceiling), Holm p = 1                                                                                      | Supported on general goals          | -->
<!-- | H2     | The action-resolution loss (ARL) lets the model rank nearby actions | Hybrid: true chunk picked over perturbed copies 0.22 vs 0.05 (σ = 0.1), 0.72 vs 0.46 (σ = 0.3); transformer: no change                                                       | Mechanism works, hybrid only        | -->
<!-- | H3     | With ARL, world-model selection stops hurting                       | Change in (best-of-16 − mean): +5 (C6), −17 (C7); Holm p = 1, 0.38                                                                                                           | Not supported                       | -->
<!-- | H3b    | ARL improves System 1                                               | 80% vs 80% (C6 vs C3); 83% vs 82% (C7 vs C5)                                                                                                                                 | Not supported                       | -->
<!-- | H4     | The hybrid is a better state estimator under camera dropout         | 78% vs 75%: +2.8 [−3.3, 8.9], Holm p = 1                                                                                                                                   | Not supported                       | -->
<!-- | H5     | Two-level System 1 helps long horizons without hurting short ones   | C3: 52% vs 33%, C5: 54% vs 33%, better in every seed; C1, C2: small, not consistent; pooled over 14 models 42% vs 30%: +12.5 [7.4, 17.9], Holm p ≤ 0.003; grasp 91% vs 91% | Supported on object-grounded models | -->
<!-- | H6     | The full combination (C7) beats LeWM (C1)                           | 83% vs 57%: +26.7 [13.3, 40.0]; grasp 93% vs 72%: +21.7 [11.7, 31.7]; Holm p ≤ 0.003 for both                                                                            | Supported (seed 0)                  | -->
TABLE: gpu_success
<!-- caption: Table 3. GPU study: success (%) on 60 non-trivial pairs per seed (grasp: H-JEPA windows), mean ± SE over 3 seeds (C6, C7: 1 seed), and the error of reading the cube's position from the latent (seed 0). -->
<!-- | **Config**               | **System 1, 25 steps** | **Grasp windows** | **Flat, 50 steps** | **Two-level, 50 steps** | **Best-of-16 by world model** | **CEM** | **Cube read-out (cm)** | -->
<!-- |----------------|---------|---------|--------|---------|---------|-------|--------| -->
<!-- | C1 LeWM                  | 55 ± 1                 | 78 ± 3            | 24 ± 1             | 31 ± 2                  | 43 ± 3                        | 18 ± 1  | 5.0                    | -->
<!-- | C2 + proprio grounding   | 66 ± 1                 | 95 ± 2            | 27 ± 1             | 31 ± 3                  | 56 ± 1                        | --      | 6.0                    | -->
<!-- | C3 + object grounding    | 82 ± 1                 | 93 ± 1            | 33 ± 1             | 52 ± 1                  | 63 ± 1                        | --      | 2.0                    | -->
<!-- | C5 hybrid + object       | 82 ± 1                 | 96 ± 1            | 33 ± 1             | 54 ± 5                  | 68 ± 5                        | 17 ± 0  | 2.1                    | -->
<!-- | C6 + object + ARL        | 80                     | 95                | 35                 | 45                      | 70                            | --      | 2.0                    | -->
<!-- | C7 hybrid + object + ARL | 83                     | 93                | 35                 | 47                      | 60                            | --      | 2.2                    | -->
TABLE: per_seed
<!-- caption: Table 4. Main contrasts in each seed separately (success %, A vs B, with the exact McNemar p on the same 60 pairs; uncorrected) and over all three seeds (A − B in points, 95% interval from resampling the shared pairs). -->
<!-- | **Contrast (A vs B)**            | **Seed 0**        | **Seed 1**        | **Seed 2**        | **All seeds [95% CI]** | -->
<!-- |---------------------|-------------|-------------|-------------|-------------| -->
<!-- | Object grounding, C3 vs C2       | 80 vs 63 (0.006)  | 82 vs 65 (0.006)  | 85 vs 68 (0.013)  | +16.7 [8.3, 25.6]      | -->
<!-- | System 1 vs CEM, C1              | 57 vs 18 (3×10⁻⁵) | 53 vs 15 (2×10⁻⁵) | 55 vs 20 (8×10⁻⁴) | +37.2 [23.9, 50.6]     | -->
<!-- | Two-level vs flat, 50 steps, C3  | 53 vs 32 (0.001)  | 53 vs 33 (0.002)  | 50 vs 33 (0.013)  | +19.4 [11.1, 28.3]     | -->
<!-- | Two-level vs flat, 50 steps, C5  | 57 vs 33 (1×10⁻⁴) | 45 vs 30 (0.004)  | 60 vs 35 (6×10⁻⁵) | +21.1 [12.8, 30.0]     | -->
<!-- | Two-level vs flat, 50 steps, C1  | 33 vs 25 (0.06)   | 32 vs 22 (0.03)   | 27 vs 27 (1)      | +6.1 [2.2, 11.1]       | -->
<!-- | Prediction vs hold, 50% lost, C3 | 72 vs 53 (0.013)  | 72 vs 58 (0.057)  | 82 vs 62 (0.004)  | +17.2 [7.8, 26.7]      | -->
<!-- | Prediction vs hold, 50% lost, C5 | 75 vs 60 (0.049)  | 77 vs 52 (7×10⁻⁴) | 82 vs 58 (0.004)  | +21.1 [12.2, 30.6]     | -->
<!-- | Best-of-16 vs mean, C3           | 65 vs 80 (0.012)  | 60 vs 82 (0.001)  | 63 vs 85 (0.001)  | −19.4 [−27.2, −12.2]   | -->
<!-- | Hybrid vs transformer, C5 vs C3  | 82 vs 80 (1)      | 80 vs 82 (1)      | 83 vs 85 (1)      | −0.6 [−6.1, 5.6]       | -->

FIGURE: figures/fig2_success.png | 6.4 | Figure 2. (a) Success on goals 25 steps ahead: System 1's mean action, best-of-16 selection by the world model, and CEM planning (run for C1 and C5). (b) Goals 50 steps ahead: flat vs two-level System 1. Bars: mean over 3 seeds; dots: individual seeds; 60 non-trivial pairs per seed.

## 5.1 A policy over the latent beats planning in it

On LeWM's own latent, System 1 succeeds on 55% of pairs and CEM on 18% (in each seed p \< 0.001); on the hybrid object model, 82% vs 17%. CEM uses 300 samples, 30 iterations, the top 30 kept and a horizon of 5 blocks, as in the LeWM evaluation code. CEM samples action sequences from a Gaussian in the same normalised action space as System 1, scores each by the latent distance between its imagined end state and the goal, and sees the same two images. Timed end to end on a 2-CPU machine for one robot (encoding the frames, choosing the actions, copying them out), a System 1 decision takes a median 0.028 s (7 decisions) against 15.7 s for CEM (6 decisions): about 560 times lower latency than the tested CEM in this single-robot, two-CPU benchmark. Ratios measured on other hardware or with many parallel robots are not directly comparable. Our CEM runs on a much smaller and shorter-trained model than published LeWM, so this shows that System 1 beats planning on the same small model, not that it beats LeWM at full scale.

## 5.2 Grounding on the object is the largest single gain

Object grounding raises System 1 from 66% to 82%, by 17 points in each seed (p ≤ 0.013 in each), the primary hypothesis H1. On grasp windows both are near the ceiling (93% vs 95%, p = 0.39). Proprioceptive grounding alone is what lifts grasping: 95% vs 78% for LeWM (32 vs 1, p \< 10⁻⁸), with a smaller gain on general goals (66% vs 55%; positive in every seed but not significant in any single one).

The mechanism is visible in the latent. A linear probe reads the cube's position to 2.0 cm with object grounding, against 5.0 cm for LeWM and 6.0 cm for proprioceptive grounding, and to 2.4 cm from a latent predicted 5 blocks ahead. Success requires the cube within 4 cm of the goal, so only the object-grounded latent holds the cube's position at a precision comparable to the task; this shows the information is decodable, not that the policy uses it. In a linear regression of the squared latent distance between start and goal on the arm's and the cube's squared displacements (held-out frames, seed 0), the arm alone explains R² = 0.49 and the cube alone R² = 0.01. This is why planning by goal distance keeps aligning the arm instead of the cube. The CPU study replicates the effect over two seeds (70 vs 55 of 100, p = 0.006).

## 5.3 The hybrid predictor matches the transformer

C5 vs C3 over three seeds: 82% vs 82% (p = 1 in every seed; the two predictors have 18.29 M and 18.35 M parameters and the same training budget); grasp windows 96% vs 93% (p = 0.07); camera dropout with prediction fill 78% vs 75% (p = 0.44). The primary hypotheses that the hybrid is better (H4 and the hybrid contrast) are not supported. We claim parity, plus a recurrent form that can run as a filter.

## 5.4 Hierarchy helps far goals on the object-grounded models

On the object-grounded models the gated two-level System 1 clearly helps goals 50 steps away: 52% vs 33% on C3 and 54% vs 33% on C5, significant in every seed. On C1 and C2 the gain is small (+6 and +4 points) and not consistent across seeds. Pooled over all 14 models the rates are 42% vs 30% (pre-specified hypothesis H5; +12.5 points, 95% interval 7.4 to 17.9 when the shared pairs are resampled). On grasp windows the gate leaves behaviour unchanged (91% vs 91%), as intended.

## 5.5 The world model's prediction stands in well for lost frames

With half the frames lost, filling them with the world model's prediction beats holding the last frame on both object models: 75% vs 58% (C3; 39 vs 8, p = 6×10⁻⁶) and 78% vs 57% (C5; 48 vs 10, p = 5×10⁻⁷). The direction holds in every seed. With the camera always on and decisions at the same rate, the same models score 73% and 75% (Figure 3), so under this dropout protocol prediction fill approximately recovers the camera-on score. We did not test an equivalence margin, longer outages or consecutive losses.


FIGURE: figures/fig3_memory.png | 4.0 | Figure 3. Success with the camera always on, and with half the frames lost and replaced either by the last frame or by the world model's prediction. Decisions every block; bars: mean over 3 seeds; dots: individual seeds.

## 5.6 Choosing among candidates with the world model does not help here

Best-of-16 selection by imagined latent distance scored below System 1's mean action in all six configurations and in every seed (object model: 63% vs 82%). Keeping the candidates near the mean (spread × 0.25) makes selection neutral (72% vs 73%, 14 models). On CPU, selection always beats picking a random candidate (by up to 16 of 50), so the world model's ranking carries information; the samples are simply worse than the mean.

Further negative results (practice in the simulator, an expressive flow-matching head, DAgger, and a loss that sharpens the model's action resolution) are in Appendix B.

To separate a bad judge from a lack of headroom, we executed every candidate in a copy of the simulator and kept the one that truly ended closest to the goal (a privileged ceiling; object model, 50 CPU pairs). Perfect choice adds only 2 of 50 episodes (41 vs 39, p = 0.69), and 9 episodes fail even when the truly best of the 16 candidates is chosen at every decision. A critic trained on 48,000 real executed outcomes ranks candidates barely above chance (within-state Spearman correlation 0.05). A contrastive verifier trained with CLM's bidirectional InfoNCE objective [16] (two small MLP encoders on our latents and action chunks, after CoVer-VLA [17]) recognises expert chunks (top-1 among 1,024: 0.19) but ranks System 1's candidates at chance (Spearman 0.005) and selects worse than the mean (28 vs 39 of 50, p = 0.013); with perturbed copies as hard negatives it learns to keep the mean (38 of 50). Used as the policy itself, retrieving the best of 20,000 stored expert chunks, it scores 28 of 50 (p = 0.052 vs System 1). For the candidate sets we tested, the limit is the quality of the proposals, not the judge.

## 5.7 A second environment: TwoRoom

On TwoRoom, a 2-D navigation task, the two-level gain replicates, but only modestly: 99% vs 93% at 50 steps, +5.3 points with a 95% interval of 0.3 to 11.7 when the shared pairs are resampled (bootstrap p = 0.03), while every other contrast sits at the ceiling (Table 5). The two-level policy won 3 or 4 pairs and lost none in each of the six model and seed runs (C1 and C2, three seeds each), but the 19 wins fall on only four distinct pairs (the same three in every run), which is why the pooled McNemar p (4×10⁻⁶ over 360 outcomes) overstates the evidence. The pipeline and scale are the same as for Cube (600 / 100 episodes, 4,000 steps, 60 non-trivial pairs per seed, three seeds); only C1 and C2 apply, since the task has no object. System 1 evaluated 60 pairs in 0.2 min against 2.5 min for CEM. Unlike on Cube, selection by the world model did not lower success. The LeWM model for seed 0 was resumed once at step 3,000 after a disconnect; the run is valid but not bit-identical to an uninterrupted one.

TABLE: tworoom
<!-- caption: Table 5. TwoRoom: paired contrasts pooled over C1 and C2 and three seeds where both apply (up to 360 outcomes of the same 60 pairs). A − B with a 95% interval from resampling the shared pairs; McNemar p pools the outcomes as if independent (descriptive). -->
<!-- | **Contrast**                               | **A** | **B** | **A − B [95% CI]** | **A-only vs B-only** | **McNemar p** | -->
<!-- |-----------------------|--------|--------|-------------|-----------|-----------| -->
<!-- | Two-level vs flat, 50 steps (H5)           | 99%   | 93%   | +5.3 [0.3, 11.7]   | 19 vs 0              | 4×10⁻⁶        | -->
<!-- | Proprio grounding (C2 vs C1), 25 steps     | 97%   | 97%   | 0.0 [0.0, 0.0]     | 0 vs 0               | 1             | -->
<!-- | System 1 vs CEM (C1), 25 steps             | 97%   | 97%   | 0.0 [−3.3, 3.3]    | 3 vs 3               | 1             | -->
<!-- | System 1 vs CEM (C1), all pairs            | 96%   | 97%   | −1.7 [−7.8, 3.9]   | 5 vs 8               | 0.58          | -->
<!-- | Flow vs Gaussian System 1, 25 steps        | 98%   | 97%   | +1.1 [0.0, 3.3]    | 4 vs 0               | 0.13          | -->
<!-- | Best-of-16 near the mean vs mean, 25 steps | 99%   | 97%   | +1.4 [0.0, 3.9]    | 5 vs 0               | 0.06          | -->
## 5.8 How often to look: committing to a chunk or deciding every step

GC-IDM [22] decides after every step and System 1 commits to 25 steps. To separate how often a controller looks from what the controller is, we ran a follow-up fixed in advance (prereg_feedback.md, with a dated addendum for the second round) on four smaller world models from our CPU study: C1 and C3 with two training seeds each (400 training episodes, 2,000 steps at batch 32, 17.92 M and 18.35 M parameters). Every controller was trained on the same frozen latents of the same training episodes and evaluated on the same 60 non-trivial pairs, which come from 40 evaluation episodes; intervals resample these episodes (Section 4). The controllers were:

-   System 1 with the GPU-study recipe (0.69 M parameters), executing 25, 5 or 1 steps of its 25-step chunk before deciding again (S1-25, S1-5, S1-1);

-   System 1 trained to output only a 5-step or a 1-step chunk, with the same recipe and goal distances (S1c5, S1c1; 0.59 M and 0.57 M parameters);

-   GC-IDM run with the authors' code, unmodified (commit 48c45b1, their Cube settings, 1.55 M parameters), told the remaining steps up to 50 (OIDM-1), and their ablation trained only on next-frame goals, H_max = 1 (OIDM-1-H1);

-   our GC-IDM-style controller, implemented independently from the paper's description, not from the authors' code (1.55 M parameters; four differences from it are listed in the plan addendum), with and without its horizon input (IDM-1, IDM-1-noH).

Results (Tables 6 and 7; differences in points with 95% intervals):

-   **Deciding more often lowered System 1's success, in every model.** Every 5 steps 54% vs 62.5% (−8.3, −14.8 to −2.4); every step 36% (−26.7, −35.1 to −17.9).

-   **Training System 1 for the shorter chunk: numerically better at 5 steps, no change at 1.** The 5-step chunk policy reached 61%, numerically above System 1 cut to 5 steps (+7.1, 0.9 to 13.1; not significant after correction, and equal in one model) and indistinguishable from the 25-step default (−1.2, −7.1 to 4.5). The 1-step chunk policy reached 36%, the same as cutting (+0.4, −4.3 to 5.8).

-   **At one step per decision, inverse dynamics did much better than System 1.** Our controller scored 60% vs 36% (+24.2, 14.0 to 34.1). The authors' code scored 54%, lower than our reimplementation in all four models but not significantly (−5.8, −11.8 to 0.4).

-   **Each in its own mode, we did not detect an overall difference from System 1's 25-step chunks.** Our controller 60% vs 62.5% (−2.5, −10.8 to 6.8); the authors' code 54% vs 62.5% (−8.3, −17.3 to 2.0). On the plain LeWM latent (C1) the step-wise controllers were equal or ahead; on the object-grounded latent (C3) System 1 was ahead in all four comparisons, by 10 to 20 points.

-   **The horizon effect depends on the ablation.** The authors' ablation, training only on next-frame goals, lowered their controller from 54% to 22.5% (−31.7, −44.0 to −21.5, every model), in line with the 42-point average drop reported for GC-IDM [22]. Removing the horizon input while keeping goals 1 to 50 steps away lowered our controller from 60% to 54% (−5.8, −13.3 to 2.1): the input numerically improved C1 (by 13 and 10 points) but gave no significant overall gain.

So deciding every step works for a controller trained on single-step inverse dynamics, but not for System 1, even when System 1 is trained to output a single step. The two differ in goal distances (any 1 to 50 steps ahead vs 5 to 25 steps in whole blocks), loss (squared error vs Gaussian likelihood), architecture and the number of training samples; we did not test which of these matters. The two horizon ablations suggest that much of GC-IDM's horizon effect comes from training on distant goals rather than from the horizon input itself, but they were run with different implementations, so this is an indication, not a test. In our setting GC-IDM's higher published Cube scores are not explained by step-wise feedback; the full-scale encoder and the standard protocol remain the likelier differences, which we did not test. Per call the controllers cost about the same (a median 0.05--0.06 s for five robots on 2 CPUs), but step-wise control makes up to 50 decisions per episode where System 1 makes up to 2.

TABLE: feedback
<!-- caption: Table 6. How often the controller looks (Section 5.8; CPU-study models, 60 non-trivial pairs, success %). S1-n: System 1 executes n steps of its 25-step chunk, then decides again from a new frame. S1c5, S1c1: System 1 trained to output a 5- or 1-step chunk, executed whole. IDM-1: our GC-IDM-style controller; OIDM-1: the GC-IDM authors' code; both decide every step. -->
<!-- | **Condition**                      | **C1 seed 0** | **C1 seed 1** | **C3 seed 0** | **C3 seed 1** | **Pooled** | -->
<!-- |-----------------------|----------|----------|----------|----------|----------| -->
<!-- | S1-25 (default)                    | 52            | 47            | 77            | 75            | 62.5       | -->
<!-- | S1-5                               | 47            | 38            | 63            | 68            | 54.2       | -->
<!-- | S1-1                               | 25            | 30            | 40            | 48            | 35.8       | -->
<!-- | S1c5 (trained for 5 steps)         | 55            | 42            | 80            | 68            | 61.3       | -->
<!-- | S1c1 (trained for 1 step)          | 33            | 23            | 40            | 48            | 36.2       | -->
<!-- | IDM-1 (our GC-IDM-style)           | 58            | 52            | 67            | 63            | 60.0       | -->
<!-- | IDM-1 without horizon input        | 45            | 42            | 63            | 67            | 54.2       | -->
<!-- | OIDM-1 (authors' GC-IDM code)     | 52            | 47            | 57            | 62            | 54.2       | -->
<!-- | OIDM-1, H_max = 1 (their ablation) | 30            | 22            | 20            | 18            | 22.5       | -->
TABLE: feedbackc
<!-- caption: Table 7. Pre-registered contrasts of Table 6. Per model: pairs solved only by A vs only by B. All: mean difference A − B in points over the four models, with a 95% interval from resampling the 40 evaluation episodes (10,000 resamples). Holm p: two-sided bootstrap p, Holm-corrected over the 12 contrasts (resolution 0.0002 before correction). * claimed: Holm p \< 0.05 and the same sign in all four models. -->
<!-- | **Contrast (A vs B)**  | **C1 seed 0** | **C1 seed 1** | **C3 seed 0** | **C3 seed 1** | **All [95% CI]**     | **Holm p** | -->
<!-- |------------------|--------|--------|--------|--------|--------------|--------| -->
<!-- | H1a S1-5 vs S1-25      | 4 vs 7        | 3 vs 8        | 1 vs 9        | 3 vs 7        | −8.3 [−14.8, −2.4]   | 0.035*    | -->
<!-- | H1b S1-1 vs S1-25      | 2 vs 18       | 3 vs 13       | 2 vs 24       | 2 vs 18       | −26.7 [−35.1, −17.9] | 0.002*    | -->
<!-- | H2 IDM-1 vs S1-1       | 22 vs 2       | 20 vs 7       | 19 vs 3       | 16 vs 7       | +24.2 [+14.0, +34.1] | 0.002*    | -->
<!-- | H3 IDM-1 vs S1-25      | 14 vs 10      | 16 vs 13      | 5 vs 11       | 5 vs 12       | −2.5 [−10.8, +6.8]   | 1          | -->
<!-- | H4 IDM-1 vs IDM-1-noH  | 15 vs 7       | 15 vs 9       | 6 vs 4        | 6 vs 8        | +5.8 [−2.1, +13.3]   | 0.68       | -->
<!-- | H5 OIDM-1 vs IDM-1     | 3 vs 7        | 5 vs 8        | 6 vs 12       | 5 vs 6        | −5.8 [−11.8, +0.4]   | 0.43       | -->
<!-- | H6 OIDM-1 vs S1-25     | 12 vs 12      | 14 vs 14      | 4 vs 16       | 4 vs 12       | −8.3 [−17.3, +2.0]   | 0.59       | -->
<!-- | H7 OIDM-1 vs OIDM-1-H1 | 17 vs 4       | 21 vs 6       | 28 vs 6       | 28 vs 2       | +31.7 [+21.5, +44.0] | 0.002*    | -->
<!-- | H8 S1c5 vs S1-5        | 6 vs 1        | 4 vs 2        | 11 vs 1       | 7 vs 7        | +7.1 [+0.9, +13.1]   | 0.17       | -->
<!-- | H9 S1c1 vs S1-1        | 6 vs 1        | 1 vs 5        | 7 vs 7        | 7 vs 7        | +0.4 [−4.3, +5.8]    | 1          | -->
<!-- | H10 S1c5 vs S1-25      | 6 vs 4        | 6 vs 9        | 4 vs 2        | 3 vs 7        | −1.2 [−7.1, +4.5]    | 1          | -->
<!-- | H11 S1c1 vs S1-25      | 4 vs 15       | 3 vs 17       | 3 vs 25       | 3 vs 19       | −26.2 [−35.2, −16.5] | 0.002*    | -->
# 6. Discussion and limitations

In this reduced-scale setting the results point one way: success depends strongly on how well the latent places the task's object, and the world model earns its place as a source of training signals, long-horizon subgoals and memory, not as a per-decision judge.

**What this suggests for practitioners.** Make the fast policy the controller, and test it at the rate at which it will act: a chunk policy that decided every step did much worse than one committing to its chunk, even when trained for single steps. Ground the latent on the state the task is scored on (here the cube; for driving, other agents and free space), even if that state is available only in training. Use hierarchy and model-based filling where horizons are long or sensors drop out. Invest in selection or verification only once the proposals vary in ways that matter and the model can resolve them.

**Limitations.**

-   Two simulated environments, one of them near the ceiling (TwoRoom); no physical robot.

-   Three training seeds for the main models and one for C6 and C7; the evaluation pairs are shared across seeds, so pooled tests overstate the evidence (Section 4).

-   Goals come from later frames of held-out expert episodes, so the results concern reaching expert-like goals, not arbitrary ones.

-   Camera dropout was simulated as independent frame loss at 50%; longer or correlated outages were not tested.

-   Reduced scale: 64-pixel frames, 600 training episodes and short training. A fully trained LeWM plans much better than ours, so the gap between System 1 and CEM is likely overstated at full scale.

-   The feedback-frequency study (Section 5.8) used the smaller CPU-study models with two seeds; the authors' GC-IDM code ran on our latents and our evaluation, not with the full-scale LeWM encoder it was designed for, so it is not a reproduction of their results.

-   Object grounding needs privileged state during training (a simulator, motion capture or a detector).

-   The non-trivial filter changes what "success" means relative to published Cube numbers; Appendix A reports the standard protocol for comparison.

-   System 1 is goal-conditioned behaviour cloning and the subgoal network resembles HIQL; the contribution is the controlled measurement and the diagnosis, not new components.

**Next.** Section 5.8 leaves open why a single-step System 1 fails where single-step inverse dynamics works; changing its goal distances, loss and amount of training one at a time is the direct next test, and a policy that predicts how long to commit is a natural extension. Since perception of the object appears to limit success, the next model should add resolution where the action happens, for example a second view cropped around the gripper, with the coarse view feeding the subgoal network and the fine view feeding System 1. Its first test is the cube read-out error, then System 1 success and the selection ceiling.

**Code and data.** The code, the pre-registration and dated plan files, and every reported result file (the outcome of each evaluation pair) are released with the preprint. The CPU follow-up result files also record each pair's evaluation episode and start step; the GPU result files do not, but those pairs are fixed by evaluation seed 42 and are regenerated from the evaluation data by check_goal_collisions.py. The GC-IDM authors' code (CC BY-NC 4.0) is not redistributed; an adapter calls it from a separate clone at a pinned commit.

# Acknowledgements

Code and text were drafted with the assistance of Claude (Anthropic). The author designed the experiments, directed the implementation, and checked the results and references.

# Appendix A. The authors' standard protocol

On the authors' protocol every System 1 scores 84--91% and CEM 72--76%, and the ranking of the models matches the non-trivial results (Table 8). Pairs are drawn from all held-out start/goal pairs (goal 25 steps ahead, budget 50, 60 pairs per seed), so about 70% are solved at the start; mean ± SE over three seeds.

TABLE: standard
<!-- caption: Table 8. Success (%) on the authors' standard protocol, mean ± SE over 3 seeds. -->
<!-- | **Config**             | **System 1** | **CEM** | -->
<!-- |------------------------|--------------|---------| -->
<!-- | C1 LeWM                | 84 ± 2       | 72 ± 1  | -->
<!-- | C2 + proprio grounding | 88 ± 1       | --      | -->
<!-- | C3 + object grounding  | 91 ± 1       | --      | -->
<!-- | C5 hybrid + object     | 90 ± 1       | 76 ± 1  | -->
Paired contrasts on these pairs (exploratory, outside the Holm family): System 1 vs CEM on C1, 84% vs 72% (24 vs 2 discordant pairs, p = 10⁻⁵); object grounding vs LeWM, 91% vs 84% (14 vs 3, p = 0.013); hybrid with object grounding vs LeWM, 90% vs 84% (13 vs 3, p = 0.02); proprioceptive grounding vs LeWM, 88% vs 84% (9 vs 3, p = 0.15). The trivial pairs compress every gap, which is why the main text uses non-trivial pairs.

# Appendix B. Further negative results

## B.1 Practice helps only with a real-world measure, and not yet in closed loop

A residual correction on the frozen System 1, trained in the simulator with the real cube-to-goal distance as reward, raised single-move success on 400 held-out situations from 51% to 65%. The same training with the world model's imagined distance gave 50%. In closed loop on the 50 evaluation pairs, real-reward practice ties System 1 (39 vs 39), while imagined-reward practice hurts (30; p = 0.035 vs System 1, p = 0.064 vs real reward). The real-reward gains are on near goals; the first move of each evaluation episode, 5 blocks ahead, got slightly worse (58% to 51%), and the two effects cancel.

Practice matched to deployment (full two-decision episodes, 38,400 per arm) gave 39 of 50 with a shaped reward (progress per decision, a grasp milestone and a success bonus) and 36 with success only, against 39 for System 1. One offline round of reward-weighted fine-tuning on the policy's own executed candidates gave 40 vs 42 for its control. Exploration noise cut practice success to about 55%, and System 1 already solved 86% of the training starts, so most of the practice budget was spent on degraded variants.

## B.2 An expressive action head is not the bottleneck

On the standard, noisy expert data, the flow-matching System 1 is worse than the Gaussian one: 66% vs 73% pooled over 14 models (56 vs 114 discordant pairs, p = 10⁻⁵), and 86% vs 91% on grasp windows (p = 7×10⁻⁴). It wins clearly only on sharp, phase-switching labels from a clean scripted expert (CPU: 32 vs 11 and 40 vs 24 of 50, both p \< 0.001), and those clean labels were worse than the noisy ones for both heads. DAgger with the same expert did not help either (8 vs 16, then 15 vs 18 of 50).

## B.3 Training the model to resolve nearby actions does not make selection useful

The action-resolution loss (ARL) asks the predictor to rank the executed action above perturbed copies of it. On the hybrid model it works as a mechanism: the model now picks the true chunk over copies perturbed by σ = 0.1 in 22% of cases (5% without; chance 6%) and by σ = 0.3 in 72% (46% without). On the transformer it does not (3% vs 5%). Yet System 1 is unchanged (C6 80% vs C3 80%; C7 83% vs C5 82%), and selection with the sharper hybrid model is worse, not better (best-of-16 60% vs mean 83%). With one seed each, this section is exploratory. Resolving nearby actions is therefore not what selection was missing; consistent with Section 5.6, the candidates simply have little to offer over the mean.

# Appendix C. Implementation details

TABLE: impl
<!-- caption: Table 9. Implementation details (GPU study unless stated). -->
<!-- | **Item**                                  | **Setting**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | -->
<!-- |-----------------------|-------------------------------------------------| -->
<!-- | Frames                                    | 64 × 64 RGB, one camera                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | -->
<!-- | Encoder                                   | ViT-tiny from scratch: 12 layers, width 192, 3 heads, MLP 768, patch 8                                                                                                                                                                                                                                                                                                                                                                                                                                     | -->
<!-- | Predictor                                 | 6-layer transformer (or 3:1 Mamba-3/attention hybrid), last 3 latents + one 5-step action block                                                                                                                                                                                                                                                                                                                                                                                                            | -->
<!-- | Loss                                      | next-latent MSE + 0.09 × SIGReg (1,024 projections); grounding heads weight 1 each                                                                                                                                                                                                                                                                                                                                                                                                                         | -->
<!-- | Parameters                                | C1 17.92 M; C2 18.34 M; C3 18.35 M; C5 18.29 M                                                                                                                                                                                                                                                                                                                                                                                                                                                             | -->
<!-- | World-model optimiser                     | AdamW, lr 3×10⁻⁴, weight decay 10⁻³, 5% warm-up then cosine to 1% of lr; 4,000 steps, batch 64                                                                                                                                                                                                                                                                                                                                                                                                             | -->
<!-- | System 1                                  | MLP 576--512--512--250, LayerNorm, GELU, 0.69 M parameters; AdamW lr 10⁻³, weight decay 10⁻⁴, cosine, gradient clip 1.0, batch 256, 3,000 steps                                                                                                                                                                                                                                                                                                                                                            | -->
<!-- | Subgoal network                           | 3,000 steps on the same cached latents                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | -->
<!-- | CEM                                       | 300 samples, 30 iterations, top 30, horizon 5 blocks                                                                                                                                                                                                                                                                                                                                                                                                                                                       | -->
<!-- | Random policy                             | uniformly random actions (stable-worldmodel RandomPolicy)                                                                                                                                                                                                                                                                                                                                                                                                                                                  | -->
<!-- | Success                                   | OGBench Cube task success: cube within 4 cm of the goal position                                                                                                                                                                                                                                                                                                                                                                                                                                           | -->
<!-- | Evaluation                                | evaluation seed 42; the same 60 pairs for every model and seed                                                                                                                                                                                                                                                                                                                                                                                                                                             | -->
<!-- | Software                                  | stable-worldmodel, MuJoCo 3.15, PyTorch; free Colab T4 GPU                                                                                                                                                                                                                                                                                                                                                                                                                                                 | -->
<!-- | CPU-study models (Section 5.8)            | same architecture; 400 training episodes; 2,000 steps, batch 32; 2 CPUs                                                                                                                                                                                                                                                                                                                                                                                                                                    | -->
<!-- | Short-chunk System 1 (Section 5.8)        | as System 1 with a 1-block output of 5 or 1 steps (25 or 5 numbers); target: the next 5 or 1 actions; goals 5--25 steps ahead; 0.59 M and 0.57 M parameters                                                                                                                                                                                                                                                                                                                                                | -->
<!-- | Authors' GC-IDM (Section 5.8)            | their train_idm.py, unmodified (commit 48c45b1): MLP on [z_now, z_goal], 3 layers of width 512, LayerNorm, GELU, dropout 0.1, remaining-step fraction → sinusoidal features → AdaLN-Zero; MSE on raw actions; goals 1--50 steps ahead (H_max 50, or 1 for their ablation); AdamW lr 10⁻³, weight decay 10⁻⁴, cosine to lr/100, clip 1.0, batch 1,024, 50 epochs, checkpoint with the lowest loss on a random 10% of pairs; 1.55 M parameters; at test time remaining steps = min(50 − steps done, H_max) | -->
<!-- | Our GC-IDM-style controller (Section 5.8) | implemented independently from the description in [22]: MLP on [z_now, z_goal], 3 layers of width 512, LayerNorm, GELU, dropout 0.1; remaining steps / 50 → 64 sinusoidal features → 2-layer MLP → AdaLN-Zero on the backbone output; one action out; MSE on (z_t, z_t+h, a_t), h uniform in 1--50 steps; AdamW lr 10⁻³, weight decay 10⁻⁴, cosine to lr/100, clip 1.0, batch 1,024, 50 epochs; 1.55 M parameters (0.73 M without horizon); at test time h = 50 − t + 1                                | -->

REFERENCES

[1] L. Maes, Q. Le Lidec, D. Scieur, Y. LeCun, R. Balestriero. LeWorldModel: Stable end-to-end joint-embedding predictive architecture from pixels. arXiv:2603.19312, 2026.

[2] G. Zhou, H. Pan, Y. LeCun, L. Pinto. DINO-WM: World models on pre-trained visual features enable zero-shot planning. arXiv:2411.04983, 2024.

[3] S. Park, K. Frans, B. Eysenbach, S. Levine. OGBench: Benchmarking offline goal-conditioned RL. ICLR 2025. arXiv:2410.20092.

[4] R. Balestriero, Y. LeCun. LeJEPA: Provable and scalable self-supervised learning without the heuristics. arXiv:2511.08544, 2025.

[5] W. Zhang, B. Terver, M. Rabbat, Y. LeCun, R. Balestriero. H-JEPA: End-to-end learning of hierarchical world models for visual planning. arXiv:2610.06805, 2026.

[6] L. Maes, Q. Le Lidec, D. Haramati, N. Massaudi, D. Scieur, Y. LeCun, R. Balestriero. stable-worldmodel-v1: Reproducible world modeling research and evaluation. arXiv:2602.08968, 2026.

[7] P. Ivashkov, R. Balestriero, B. Schölkopf. Sensorimotor world models: Perception for action via inverse dynamics. arXiv:2606.20104, 2026.

[8] D. Ghosh, A. Gupta, A. Reddy, J. Fu, C. Devin, B. Eysenbach, S. Levine. Learning to reach goals via iterated supervised learning. ICLR 2021. arXiv:1912.06088.

[9] S. Park, D. Ghosh, B. Eysenbach, S. Levine. HIQL: Offline goal-conditioned RL with latent states as actions. NeurIPS 2023. arXiv:2307.11949.

[10] C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, S. Song. Diffusion Policy: Visuomotor policy learning via action diffusion. RSS 2023. arXiv:2303.04137.

[11] X. Liu, C. Gong, Q. Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. ICLR 2023. arXiv:2209.03003.

[12] L. Ankile, Z. Jiang, R. Duan, G. Shi, P. Abbeel, A. Nagabandi. Residual off-policy RL for finetuning behavior cloning policies. arXiv:2509.19301, 2025.

[13] E. Su, T. Westenbroek, A. Nagabandi, A. Gupta. RFS: Reinforcement learning with residual flow steering for dexterous manipulation. ICLR 2026. arXiv:2602.01789.

[14] A. Y. Ng, D. Harada, S. Russell. Policy invariance under reward transformations: Theory and application to reward shaping. ICML 1999.

[15] S. Ross, G. Gordon, D. Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. AISTATS 2011.

[16] Contrastive-LM. CLM: contrastive language model (code and model release; no accompanying paper found as of November 2026). https://github.com/Contrastive-LM/CLM, 2026.

[17] J. Kwok, X. Zhang, M. Xu, Y. Liu, A. Mirhoseini, C. Finn, M. Pavone. Scaling verification can be more effective than scaling policy learning for vision-language-action alignment. arXiv:2602.12281, 2026.

[18] A. Lahoti, K. Y. Li, B. Chen, C. Wang, A. Bick, J. Z. Kolter, T. Dao, A. Gu. Mamba-3: Improved sequence modeling using state space principles. ICLR 2026. arXiv:2603.15569.

[19] H. Yan, J. Zhu, et al. Is forward prediction enough? Physical state grounding for JEPA world models. arXiv:2608.06799, 2026.

[20] M. Liu, Y. Huang, Z. Liang, X. Gao. Toward physically grounded JEPA world models for goal-conditioned robotic planning. arXiv:2609.03565, 2026.

[21] X. Zeng, H. Ren, Z. Song, Y. Nie, R. Drummond. PhyLatent: Learning dynamics-relevant representations for JEPA world models. arXiv:2608.05720, 2026.

[22] H. Nguyen, X. Xu, X. Huang. Latent geometry beyond search: Amortizing planning in world models. arXiv:2605.08732, 2026.

[23] H.-W. Huang, J. Shangguan, J. Lu, J.-N. Hwang. LeFlow: Generative latent flow planning for world models. arXiv:2608.24855, 2026.

[24] M. Fu, T. Siebert, E. Halicki, R. Balestriero. LeWAM: End-to-end world and action modeling with JEPAs. arXiv:2610.12818, 2026.

[25] B. Terver et al. What drives success in physical planning with joint-embedding predictive world models? arXiv:2512.24497, 2025.

[26] H. Molin, W. Peng, M. Bagatella, R. Balestriero. World modeling through spectral alignment. arXiv:2610.15429, 2026.

[27] A. Zholus, N. Beltran-Velez, J. Yuan, S. Chandar, T. Nagarajan, D. Severo, K. Sinha, M. Drozdzal, A. Romero Soriano, J. Bohg, N. Ballas, M. Assran. RoboJEPA: Scaling robotic latent world models. arXiv:2610.10515, 2026.
