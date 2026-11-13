<!-- The code guide. Build: python paper/build.py --source guide.md --name Paper1_code_guide -->
TITLE: How the Paper 1 Code Works

AUTHOR: A learning guide to the grounded-jepa-system1 repository

AFFIL: Every concept, where it lives in the code, how it is trained and used


ABSTRACT: This guide walks through the Paper 1 code from the camera frame to the statistics. Each section explains one idea in plain words, names the file and function that implement it, shows the few lines that matter, and says how it is trained and how it is used at test time. Read it with the repository open. Paths are relative to the repository root; most experiment code is in bench/lewm_suite/ and the reusable pieces are in src/hybrid_wm/.

# 1. The big picture

A camera frame goes into an encoder, which turns it into a short vector (the latent). A predictor learns how actions move that latent. Together they are the world model. A small policy (System 1) looks at the current latent and the goal latent and outputs the next moves. The robot runs them, looks again, and repeats.

The pipeline that produces every number in the paper, one script per stage:

1. collect.py: drives the simulated robot with the authors' scripted expert and records episodes (frames, actions, robot state, cube position) into HDF5 files; merge_h5.py joins the parts.

2. train.py: trains the world model (and the grounding heads) on those episodes.

3. latent_cache.py: runs the trained encoder once over every frame and saves the latents, so later stages do not touch images.

4. cached_policy.py and cached_subgoal.py: train System 1 and the subgoal network on the cached latents.

5. evaluate.py: puts the robot in a start state with a goal image and counts successes.

6. probe_diag.py and rollout_diag.py: measure what the latent knows and how well the predictor rolls forward.

7. analyze_colab.py: tables and paired statistics.

colab_plan.py chains all of these as jobs with dependencies, skips finished work, and is what the Colab notebook runs.

# 2. Units and vocabulary

- **Step:** one robot action. In Cube it has 5 numbers: hand displacement in x, y, z, a rotation, and the gripper command.

- **Block:** 5 steps. The world model works in blocks: it sees one frame every 5 steps and predicts one latent per block (TrainConfig.frameskip = 5 in models.py).

- **Chunk:** 5 blocks = 25 steps. System 1 outputs one chunk per decision.

- **Latent (z):** the 192-number vector the encoder makes from one frame (TrainConfig.embed_dim = 192).

- **Goal latent (z_goal):** the latent of the goal image, made by the same encoder.

- **Normalised actions:** actions are rescaled to zero mean and unit spread with a scaler fitted on the training data (column_normalizer in train.py, get_scaler in latent_cache.py), so all action numbers are on the same scale.

- **Spec string:** the name of a model configuration, parsed by models.py: Spec. Tokens: lewm or hybrid (which predictor), idm (inverse dynamics), sa (state alignment), obj (add the object to the state target), arl (action-resolution loss). C1 = lewm, C2 = lewm+idm+sa, C3 = lewm+idm+sa+obj, C5 = hybrid+idm+sa+obj.

# 3. The world model (LeWM)

**What it is.** A JEPA (joint-embedding predictive architecture) world model predicts the next latent, not the next image. LeWM is the authors' version: a ViT-tiny encoder and a transformer predictor trained end to end from pixels.

**Where.** models.py: build_model creates the encoder (vit_hf "tiny", patch 8 on 64-pixel frames), the predictor (6 transformer layers), an action encoder (Embedder) and two projectors, and wraps them in the authors' LeWM class from stable-worldmodel. GroundedLeWM adds the optional grounding heads.

**How it is trained.** train.py: losses. For a clip of frames, the encoder gives latents; the predictor takes the last 3 latents and the action blocks and predicts the next latents. The loss is:

CODE: pred = lewm.predict(emb[:, :3], act_emb[:, :3])
CODE: loss = mean((pred - emb[:, 1:])**2) + 0.09 * SIGReg(emb)

**SIGReg** (from LeJEPA) is a regulariser that pushes the latents of a batch towards a standard Gaussian distribution, measured along 1,024 random directions. Without it the encoder could make every latent the same (collapse) and the prediction loss would be trivially zero.

**Training settings:** AdamW, learning rate 3×10⁻⁴, weight decay 10⁻³, 5% warm-up then cosine decay, gradient clipping at 1.0, 4,000 steps at batch 64 on the GPU.

# 4. Grounding

**What grounding means here.** The prediction loss only asks the latent to be predictable. Nothing forces it to keep what control needs, such as where the hand or the cube is. Grounding adds training-only tasks that can be solved only if the latent keeps that information. The extra heads are thrown away after training: at test time the model still sees pixels only.

## 4.1 Inverse dynamics (idm)

src/hybrid_wm/grounding.py: InverseDynamicsHead. A 2-layer MLP gets two consecutive latents and must guess the action block executed between them:

CODE: a_hat = MLP([z_t, z_t+1]) # 25 numbers: 5 steps x 5 action numbers
CODE: loss = mean((a_hat - a_t)**2) # masked where the action is missing

If the latent ignored the hand, the action could not be recovered. This is the same idea as SMWM.

## 4.2 State alignment (sa)

grounding.py: StateAlignmentHead. Another MLP gets two consecutive latents and must predict the next state and the change of state:

CODE: s_hat_next, ds_hat = MLP([z_t, z_t+1])
CODE: loss = mean((s_hat_next - s_t+1)**2) + mean((ds_hat - (s_t+1 - s_t))**2)

The state is normalised like the actions (column_normalizer "state" in train.py). Which columns form the state is decided by models.py: state_columns: the robot's proprioception (hand position, hand rotation, gripper opening, joint angles: Task.state_cols in tasks.py) and, with the obj token, also the object columns (Task.object_cols).

## 4.3 Object grounding (obj)

For Cube, Task.object_cols = ("privileged_block_0_pos",): the cube's x, y, z position in metres, read from the simulator. It is "privileged" because a real robot would need a detector or motion capture to know it; that is why it is used only during training. This single change (C2 to C3) is the paper's largest gain: 66% to 82%.

**How the total loss is formed** (train.py: losses):

CODE: total = pred + 0.09*sigreg + 1.0*idm + 1.0*sa (+ 1.0*arl for the arl spec)

# 5. The hybrid predictor (C4, C5)

**What it is.** A drop-in replacement for the transformer predictor (src/hybrid_wm/integrations/lewm.py: HybridPredictor), built from src/hybrid_wm/backbone.py: HybridBackbone. It has 6 layers in the pattern M M M A (repeating): three Mamba-3 layers, then one attention layer.

**Mamba-3** (src/hybrid_wm/layers/mamba3.py: Mamba3Mixer) is a state-space layer: it carries a small hidden state from one time step to the next and updates it with each new input, like a learned filter. That is why the hybrid can run step by step with constant cost per step. The repository has a pure-PyTorch "reference" backend, ported from the Mamba authors' reference equations (Apache-2.0), and can use the official CUDA kernels (mamba-ssm) on a GPU.

**Window attention** (layers/attention.py: GatedWindowAttention) lets each step look directly at the last few latents.

**AdaLN-zero action conditioning** (backbone.py: Block.ada): the action embedding produces a scale, shift and gate for each layer's normalisation; it starts at zero, so at initialisation each layer is the identity and the action effect is learned gradually.

**Mixture of experts (MoE)** (layers/moe.py: MoEFFN): the feed-forward part of most layers is split into 8 small expert networks plus 1 shared one. For every token a router picks the top 2 experts; only those run. Routing follows DeepSeek-V3: a sigmoid score per expert, plus a bias that is nudged after each step so that no expert is overloaded, and a very small balance loss (returned as aux_loss and added in train.py).

**Size.** The hybrid has 18.29 M parameters against 18.35 M for the transformer model, with the same training budget, so the comparison is fair. In the paper it matches the transformer (82% vs 82%).

The backbone also contains optional modules not used in the paper (an n-gram memory, an episodic memory and a two-timescale stream); they are switched off by default and are left untouched so the released predictor is exactly the code that produced the results.

# 6. The action-resolution loss (arl, C6 and C7)

grounding.py: perturb_actions and action_resolution_loss. For each training clip, 4 noisy copies of the executed action blocks are made (noise scale between 0.05 and 0.5). The predictor runs once over the real and the noisy actions, and a hinge loss asks the prediction for the real actions to be closer to the observed next latent than each noisy prediction, by a margin of 10%:

CODE: loss = mean(max(0, log d_true - log d_noisy + log(1.1)))

The idea was to make the model good at telling nearby actions apart, so that choosing among candidates would work. It sharpened that ability on the hybrid model but did not make selection help (Appendix B.3).

# 7. The latent cache

latent_cache.py encodes every frame of a dataset once with the trained encoder and saves latents, normalised actions, robot state and cube position in one file per model and dataset (cache_path). It then checks that the cache reproduces exactly what the training loader would produce. All policy training reads this cache, which is why System 1 trains in minutes.

# 8. System 1

**What it is.** A small, fast, goal-conditioned policy: given where we are and where we want to be, output the next moves directly, without search. The name comes from the fast, intuitive mode of thinking in dual-process accounts of cognition.

**Where.** src/hybrid_wm/control/latent_policy.py: ChunkPolicy.

CODE: input = [z_t, z_goal, z_goal - z_t] # 3 x 192 = 576 numbers
CODE: MLP 576 -> 512 -> 512 -> 250 (LayerNorm, GELU) # 0.69 M parameters
CODE: output = mean and log-spread for 5 blocks x 25 numbers # a Gaussian per action number

**How it is trained** (cached_policy.py: train, with OfflineWindows). This is behaviour cloning with hindsight goals: from the expert episodes, take a window of 6 frames one block apart; pick k from 1 to 5; use frame k as the goal; the target is the chunk of actions the expert actually executed from frame 0. Blocks after the goal frame are masked, because they do not help reach that goal. The loss is the Gaussian negative log-likelihood (ChunkPolicy.loss):

CODE: nll = gaussian_nll(mean, actions, std**2), averaged over action numbers
CODE: loss = sum(nll * mask) / sum(mask)

Settings: AdamW, learning rate 10⁻³, weight decay 10⁻⁴, cosine schedule, gradient clip 1.0, batch 256, 3,000 steps on the GPU.

**How it is used.** system12.py: GatedSolver with mode "s1": encode the current and goal frames, run ChunkPolicy, take the mean chunk (sample() returns the mean as candidate 0), execute it, then decide again. By default all 5 blocks run before the next decision (receding = 5).

**The flow-matching head** (latent_policy.py: FlowChunkPolicy, cached_policy.py --policy-type flow) learns a velocity field that turns noise into an action chunk in 10 integration steps (rectified flow). It can represent several distinct options instead of one average. On Cube's noisy expert data it was worse than the Gaussian (Appendix B.2).

# 9. The two-level System 1 (subgoal network)

latent_policy.py: SubgoalNet. A second MLP maps (z_t, z_goal) to an intermediate latent about one policy horizon ahead (z_t plus a learned correction). System 1 then acts towards that subgoal instead of the far goal.

**Training** (cached_subgoal.py, subgoal_batch): from a clip, pick a goal k blocks ahead with k up to 10; the target subgoal is the frame min(k, 5) blocks ahead; the loss is the squared distance between predicted and real subgoal latents.

**The gate** (cached_subgoal.py --gate auto): the subgoal is used only when the squared latent distance to the goal is larger than the median distance covered in 5 blocks on the training data. Near goals go straight to System 1. The gate is fixed from training data before any evaluation.

**Result.** On the object-grounded models it raises success on goals 50 steps away from 33% to 52--54%, in every seed.

# 10. Choosing among candidates and verify-then-act

system12.py: GatedSolver. Instead of only the mean chunk, System 1 can propose k = 16 candidates (the mean plus 15 samples). The world model imagines where each would lead, and the candidate whose imagined end latent is closest to the goal latent is executed (best-of-16). In verify-then-act mode (s1s2), System 1's chunk is accepted only if its imagined cost falls to 0.6 × the current distance, otherwise CEM plans instead. Variants replace the world-model score with a learned critic (critic.py), a contrastive verifier (contrastive_verifier.py) or the true simulated outcome (select "oracle": the privileged ceiling). None beat the mean chunk on Cube.

# 11. Memory when camera frames are lost

system12.py: GatedSolver._perceive, used with --camera-drop 0.5 and decisions every block (--receding 1). At each decision the frame is lost with probability 0.5. A lost frame is replaced by a black image (blank), by the last latent seen (hold), or by the world model's prediction from the last latent and the action block executed since (predict). The solver keeps a small memory per episode for this. After the review it is cleared for every new batch of episodes (reset_episodes, called by evaluate.py) and refuses two episodes with the same goal image; check_goal_collisions.py confirms no evaluation pairs share a goal image.

# 12. CEM planning (the baseline)

The authors' cross-entropy method, from stable-worldmodel (CEMSolver), set up in evaluate.py: sample 300 action sequences of 5 blocks from a Gaussian, roll each through the world model, keep the 30 whose imagined end latent is closest to the goal, refit the Gaussian to them, repeat 30 times, and execute the result. On CPU a decision takes about 15.7 s against 0.028 s for System 1.

# 13. Evaluation

evaluate.py: main.

- **Pairs.** A pair is a start frame and a goal frame from the held-out episodes. pick_pairs lists every start whose goal 25 steps later exists, removes the trivial ones (trivial_mask: the start already satisfies the goal, cube within 4 cm), and draws 60 with a fixed seed (42), so every model and seed is scored on the same pairs. --pairs pickup uses the grasp-centred windows (tasks.py: pickup_candidates); --pairs all is the authors' protocol.

- **Running an episode.** The simulator is restored to the start state, the goal image is set, and the controller runs for the budget (50 steps; 100 for goals 50 steps away). Success counts if the goal condition holds at any step.

- **Output.** One JSON file per run in results/ with the success of every pair, the pair identities and the decision times (decision_seconds).

# 14. Diagnostics

- **Probe** (probe_diag.py): a ridge (linear) regression from the latent to the cube's position, fitted on training frames and scored on held-out frames; the error is the mean Euclidean distance in cm (2.0 cm for C3, 5.0 cm for LeWM). The same script regresses the latent distance between start and goal on the arm's and the cube's displacement (the R² values in Section 5.2).

- **Rollout** (rollout_diag.py): how well the predictor rolls forward several blocks, and whether it can pick the true action chunk among perturbed copies (the ARL check).

# 15. Statistics

analyze_colab.py.

- **Paired design.** Two methods are scored on the same pairs, so the right test looks only at pairs where they disagree: a pairs only A solved, b pairs only B solved.

- **Exact McNemar test** (mcnemar_exact): if both methods were equally good, each disagreement would go either way with probability 1/2; the p-value is the binomial probability of a split at least as uneven as a vs b.

- **Holm correction** (holm): when several primary hypotheses are tested, the smallest p-value is multiplied by the number of tests, the next by one fewer, and so on, which keeps the chance of any false positive below 5%.

- **Why pooled p-values overstate the evidence.** The same 60 pairs are reused across seeds and models, so outcomes are not independent. The paper therefore reports every main contrast per seed (Table 4) and trusts only effects that hold in all three seeds.

- **Cluster bootstrap** (cluster_bootstrap.py: cluster_bootstrap). Instead of treating every seed-pair outcome as independent, it resamples whole clusters with replacement 10,000 times. A cluster is an evaluation pair in the GPU study (the result files do not record more) and an evaluation episode in the CPU follow-up (60 pairs come from 40 episodes, and pairs from one episode are alike). Each drawn cluster brings its outcomes for every seed or model together. The spread of the mean difference over the resamples gives a 95% interval; twice the share of resamples on the wrong side of zero gives a p-value (smallest possible 2/10,000 = 0.0002). A unit test checks it on cases with a known answer.

- **Safety checks.** The analysis refuses two result files for the same model, seed and test, and refuses a paired comparison on different pairs.

# 16. The CPU follow-up studies

These answer "why does choosing with the world model not help?" and "can practice in the simulator improve System 1?". Each has a plan file written before the run. The first result files of the selection, ceiling, critic and verifier experiments and of the CPU replication and timing were lost in November 2026; run_rerun.sh ran them again on regenerated data (Section 17, Seeds and data), and the numbers below are from that run.

- **Outcome critic** (critic.py, gen_critic_data.py; prereg_critic.md E6): a network trained on {{int: cpu_rerun.critic.train_outcomes}} real executed outcomes to score candidates. Its within-state Spearman correlation on held-out states is {{num2: cpu_rerun.critic.spearman}} (0 is chance), and choosing with it solves {{int: cpu_rerun.ceiling.critic_solved}} of {{int: cpu_rerun.ceiling.pairs}} against {{int: cpu_rerun.ceiling.mean_solved}} for the mean action.

- **Privileged ceiling** (select oracle in evaluate.py; oracle.py): every candidate is executed in a copy of the simulator and the truly best is chosen. It added {{int: cpu_rerun.ceiling.oracle_gain}} of {{int: cpu_rerun.ceiling.pairs}} episodes ({{int: cpu_rerun.ceiling.oracle_solved}} vs {{int: cpu_rerun.ceiling.mean_solved}} for the mean action), which shows the candidates themselves offer little.

- **Contrastive verifier** (contrastive_verifier.py; E10): two small encoders trained with a bidirectional InfoNCE loss (as in CLM) so that a latent and the action chunk that follows it get similar embeddings. It picks the expert chunk among 1,024 with top-1 {{num2: cpu_rerun.verifier.top1}} but ranks System 1's candidates with Spearman {{num3: cpu_rerun.verifier.spearman}}, and selecting with it solves {{int: cpu_rerun.verifier.select_solved}} of {{int: cpu_rerun.ceiling.pairs}}.

- **Practice** (residual_rl.py, practice_rl.py, awr_policy.py): a small residual network adds a bounded correction to System 1's chunk and is trained in the simulator by a group-relative policy gradient (several attempts from the same start are compared with each other), with a real-distance reward or the world model's imagined distance. Exploratory; its result files were lost and it was not re-run, so the paper does not report it.

- **Clean labels and DAgger** (oracle_policy.py, dagger.py; prereg_overnight.md): a clean scripted expert and corrective labels on the policy's own states. Exploratory; its result files were lost and it was not re-run, so the paper does not report it.

- **How often to look** (gcidm.py, official_gcidm.py, short_chunk.py, run_feedback.sh, run_feedback2.sh, analyze_feedback.py; prereg_feedback.md and its dated addendum; paper Section 5.8).

- **Executing n steps.** step_policy in gcidm.py wraps the library's WorldModelPolicy. That policy keeps a buffer of planned actions and asks the controller for a new plan only when the buffer is empty. The wrapper empties the buffer after n executed steps (--exec-steps n in evaluate.py), so the robot takes only the first n steps of each plan and then decides again from a new frame. A unit test checks that exactly the first n steps of the newest plan are executed.

- **The GC-IDM-style controller.** GCIDM is a 3-layer MLP on [current latent, goal latent] that outputs one action. The number of steps left (50 − t + 1, divided by 50) is turned into 64 sine and cosine features, passed through a small MLP, and used to scale and shift the network's last hidden layer (AdaLN-Zero: the scale and shift start at zero, so at first the horizon has no effect). It is trained with mean squared error on (latent at t, latent h steps later, action at t), with h drawn from 1 to 50. It was written from the paper's description, not from the authors' code.

- **One adapter for every step-wise controller.** LatentStepSolver in gcidm.py encodes the current frame and the goal frame, asks a controller function act(z_now, z_goal, steps done) for c actions, and pads them to the shape the evaluation pipeline expects. Three controllers plug into it: our GC-IDM-style network (gcidm_act), the authors' network (official_act) and the short-chunk System 1 (short_chunk_act).

- **The authors' GC-IDM** (official_gcidm.py). Their code is licensed CC BY-NC 4.0, so it is not copied: the user clones their repository and checks out the pinned commit. export writes our cached latents, the raw actions and episode ids into their .npz format (and checks that the raw actions match the cached standardised ones row by row). train runs their train_idm.py unmodified, with their Cube settings, after checking that the clone is at the pinned commit with no local changes. At test time their load_idm loads the model; it is given min(50 − steps done, H_max) as the remaining steps, as in their policy, and its raw action is standardised for our pipeline. Their ablation (H_max = 1) trains only on next-frame goals.

- **Short-chunk System 1** (short_chunk.py). The same ChunkPolicy, recipe and goal distances (5 to 25 steps ahead) as System 1, but the target is only the next 5 or 1 actions, so the robot executes the whole short chunk and decides again.

- **Result:** deciding every step lowers System 1 from 62.5% to 36%, also when it is trained for single steps (36%); trained for 5-step chunks it matches the 25-step default (61%). The step-wise inverse-dynamics controllers work at one step (ours 60%, the authors' code 54%) but are not distinguishable from System 1's 25-step chunks. The authors' H_max = 1 ablation drops to 22.5%. All result files and the summary are in results/feedback_study/.

# 17. Reliability features of the code

- **Skip-if-done and atomic saves.** Every stage writes its output to a temporary file and renames it when complete, so a disconnect never leaves a half-written file that a later run would mistake for finished work.

- **Stale detection** (colab_plan.py): if a model is retrained, everything computed from the old model is redone.

- **Seeds and data.** Seeds fix the training batches and, given the data, the evaluation pairs. Collecting the data again with the same collector on another machine (November 2026, after the first CPU-study files were lost) gave different episodes: the held-out data used the same seed (777), and the training data was collected in two parts of 200 episodes (seeds 3072 and 3073) to fit the 2-hour compute limit. The data files, not the seeds, are therefore the reference; the CPU-study experiments were run again on the new data (run_rerun.sh), with System 1 trained by the GPU study's recipe (cached_policy.py, 3,000 steps, batch 256).

- **Tests.** Running pytest from the repository root runs both test folders: tests/ (library: Mamba-3 against its reference, policies, grounding heads) and bench/lewm_suite/test_bench_units.py (experiment code: executing n steps, the GC-IDM horizon, short-chunk targets, the step adapter, the cluster bootstrap and others). Tests that need a GPU or the simulator are skipped when those are missing.

# 18. How every number of the paper is produced and checked

The paper is built from text and results, never typed by hand.

- **Results files.** Instruments in bench/lewm_suite read the per-pair result files and print one JSON object: paper_values_gpu.py (GPU study, Tables 2 to 5 and 8), analyze_feedback.py (Section 5.8), analyze_rerun.py (the CPU-study re-run), model_sizes.py (parameter counts, built from the model code) and trivial_share.py (pairs already solved at the start). Each is run through osaudit run, which writes the JSON to results/paper/ with a provenance block: the command, the sha256 of every input and the git commit of the code.
- **The paper source.** paper/paper.md is the text; every number in it is a placeholder such as {pct0: gpu_cube.success.C1.s1_nt.mean} (written with double braces), which names a value in results/paper/gpu_cube.json and how to print it. Settings are marked as settings and numbers from other papers cite their reference. Tables are defined in paper/tables.py the same way.
- **The build.** paper/build.py fills the placeholders, writes the Word file (docx_builder.js) and the PDF, and writes osaudit.toml with one entry per printed number: the value as printed and the expression it came from, or the reason a setting is allowed. A value printed as a plain integer must be an integer, and a negative number is checked through its absolute value, because that is how osaudit reads a PDF.
- **The audit.** osaudit (ASSERT-KTH) reads the PDF, finds every number, and checks it against its entry and the results files; it also checks that each results file was produced by code that has not changed since. It prints nothing when everything is traced. paper/reproduce.sh runs every step; inputs from outside the repository (the GC-IDM authors' code, the regenerated Cube data) go in the git-ignored folder external/, so the provenance records paths inside the repository.
- **Figures.** bench/lewm_suite/paper_figures.py draws all five figures. Figures 3 to 5 come from results/paper/gpu_cube.json; their error bars are 95% intervals from resampling the evaluation pairs, each pair bringing its outcomes for all seeds (paper_values_gpu.py computes them). Figures 1 and 2 use camera frames saved in paper/figures/data: figure_examples.py takes the already-solved pair from the CPU re-run's random-policy result file, and figure_rollout.py re-runs the CPU re-run's System 1 evaluation of C3 (seed 0) with a recorder on the per-step callback, checks that its 50 outcomes equal the re-run's, and keeps the frames of the solved pair whose cube moves farthest.
- **Official data check.** notebooks/official_cube_check.ipynb (built by build_official_check_notebook.py) downloads the LeWM authors' Cube dataset on Colab and counts the pairs already solved at the start, for all pairs and for the 50 their eval.py draws, and runs their eval.py with a random policy. It needs about 100 GB of free disk: the dataset is one 95 GB file, which must be on disk to be read.

# 19. Glossary

- **Amortised control:** learning a policy that outputs actions directly, instead of searching for them at every decision.

- **Behaviour cloning:** learning a policy by imitating recorded actions.

- **CEM:** cross-entropy method, a sampling-based planner.

- **Discordant pair:** a pair on which exactly one of two methods succeeds.

- **Hindsight relabelling:** using a later frame of the same episode as the goal for earlier actions.

- **Holm correction:** adjustment of p-values when several hypotheses are tested.

- **JEPA:** joint-embedding predictive architecture; predicts latents, not pixels.

- **Latent:** the encoder's vector summary of a frame.

- **MoE:** mixture of experts; a layer whose work is split among small expert networks, only some of which run per token.

- **Privileged information:** simulator state a real robot would not observe directly.

- **Probe:** a simple model fitted on the latent to test what information it contains.

- **SIGReg:** the regulariser that keeps latents spread out (prevents collapse).

- **State-space model (Mamba):** a recurrent layer with a learned hidden state, run step by step or in parallel.
