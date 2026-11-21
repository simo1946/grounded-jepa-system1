"""The paper's tables: captions, headers and cells, with every number as a placeholder (see build.py).

Values come from results/paper/*.json: gpu_cube and gpu_tworoom (paper_values_gpu.py), feedback
(analyze_feedback.py), model_sizes (model_sizes.py). Widths are in DXA and sum to the text width.
"""
from __future__ import annotations

TEXT_W = 9306

S = '{{set: %s | %s}}'                                    # a setting printed as is


def _w(*fractions: float) -> list[int]:
    w = [round(TEXT_W * f) for f in fractions]
    w[-1] = TEXT_W - sum(w[:-1])
    return w


def _mean_se(path: str, cell: str) -> str:
    return f"{{{{num1: {path}.{cell}.mean}}}} ± {{{{num1: {path}.{cell}.se}}}}"


def _ci(path: str) -> str:
    return (f"{{{{sgn1: {path}.diff_pts}}}} [{{{{dec1: {path}.ci_lo}}}}, {{{{dec1: {path}.ci_hi}}}}]")


# ---------------------------------------------------------------------------------------------- Table 1
def scale():
    caption = "Table 1. Experimental scale. The GPU study is the main study; the CPU study holds the follow-up experiments."
    header = ["", "GPU study (main)", "CPU study (follow-ups)", "LeWM / H-JEPA papers"]
    ext = '{{ext: %s | [1] | %s}}'
    rows = [
        ["Training / held-out episodes",
         f"{S % ('600', 'GPU-study training episodes')} / {S % ('100', 'GPU-study held-out episodes')}",
         f"{S % ('400', 'CPU-study training episodes')} / {S % ('80', 'CPU-study held-out episodes')}",
         f"about {ext % ('10,000', 'approximate number of training episodes in the published setups')}"],
        ["Frames", f"{S % ('64', 'frame size in pixels')} px", f"{S % ('64', 'frame size in pixels')} px",
         f"{ext % ('224', 'frame size in pixels of the published setups')} px"],
        ["World-model training",
         f"{S % ('4,000', 'GPU-study world-model steps')} steps, batch {S % ('64', 'GPU-study batch size')}",
         f"{S % ('2,000', 'CPU-study world-model steps')} steps, batch {S % ('32', 'CPU-study batch size')}",
         "several epochs on GPU"],
        ["System 1 / subgoal training", f"{S % ('3,000', 'policy training steps')} steps each",
         f"{S % ('3,000', 'policy training steps')}–{S % ('6,000', 'longest CPU-study policy training')} steps", "–"],
        ["Evaluation pairs", f"{S % ('60', 'evaluation pairs per seed, GPU study')} non-trivial per seed",
         f"{S % ('50', 'evaluation pairs, CPU study')} non-trivial",
         f"{ext % ('50', 'evaluation pairs of the published protocol')} standard"],
        ["Seeds",
         f"{S % ('3', 'training seeds of the main configurations')} (C6, C7: {S % ('1', 'training seeds of C6 and C7')})",
         f"{S % ('1', 'fewest CPU-study training seeds')}–{S % ('2', 'most CPU-study training seeds')}", "–"],
    ]
    return caption, header, rows, _w(.25, .25, .25, .25)


# ---------------------------------------------------------------------------------------------- Table 2
def hypotheses():
    c = "gpu_cube.contrasts"
    d = "gpu_cube.diagnostics"
    caption = ("Table 2. Hypotheses fixed before the GPU run (C6 and C7 have one seed). Differences in percentage points "
               "with 95% intervals from resampling the {{set: 60 | evaluation pairs per seed, GPU study}} shared pairs "
               "(all seeds together, {{set: 10,000 | bootstrap resamples}} resamples); p: bootstrap p, Holm-corrected "
               "over the primary family (H3: Wilcoxon); the smallest bootstrap p that "
               "{{set: 10,000 | bootstrap resamples}} resamples can give is "
               "{{set: 0.0002 | resolution of a two-sided bootstrap p with 10,000 resamples}}. Table 4 gives each seed.")
    header = ["#", "Hypothesis", "Result", "Verdict"]
    og, ogp = f"{c}.object_grounding_s1_nt", f"{c}.object_grounding_s1_pk"
    h5, h5p = f"{c}.all_configs_two_level_vs_flat_50_hier_50", f"{c}.all_configs_two_level_vs_flat_grasp_hier_pk"
    h6, h6p = f"{c}.all_best_parts_vs_lewm_s1_nt", f"{c}.all_best_parts_vs_lewm_s1_pk"
    cam = f"{c}.camera_0_5_prediction_hybrid_vs_transformer_object_cam0_5"
    h5c3, h5c5 = f"{c}.c3_two_level_vs_flat_50_hier_50", f"{c}.c5_two_level_vs_flat_50_hier_50"
    rows = [
        ["H1", "Object grounding improves System 1",
         f"{{{{pct0: {og}.a_pct}}}} vs {{{{pct0: {og}.b_pct}}}}: {_ci(og)}, Holm p = {{{{p: {og}.p_holm}}}}; grasp "
         f"{{{{pct0: {ogp}.a_pct}}}} vs {{{{pct0: {ogp}.b_pct}}}} (ceiling), Holm p = {{{{p: {ogp}.p_holm}}}}",
         "Supported on general goals"],
        ["H2", "The action-resolution loss (ARL) lets the model rank nearby actions",
         f"Hybrid: true chunk picked over perturbed copies {{{{num2: {d}.C7.fine_top1_s0_1}}}} vs "
         f"{{{{num2: {d}.C5.fine_top1_s0_1}}}} (σ = {S % ('0.1', 'perturbation scale of the ranking test')}), "
         f"{{{{num2: {d}.C7.fine_top1_s0_3}}}} vs {{{{num2: {d}.C5.fine_top1_s0_3}}}} "
         f"(σ = {S % ('0.3', 'perturbation scale of the ranking test')}); transformer: "
         f"{{{{num2: {d}.C6.fine_top1_s0_1}}}} vs {{{{num2: {d}.C3.fine_top1_s0_1}}}} "
         f"(σ = {S % ('0.1', 'perturbation scale of the ranking test')})",
         "Mechanism works, hybrid only"],
        ["H3", "With ARL, world-model selection stops hurting",
         f"Change in (best-of-16 − mean): {{{{sgn1: {c}.arl_makes_selection_useful_c3_c6.diff_pts}}}} (C6), "
         f"{{{{sgn1: {c}.arl_makes_selection_useful_c5_c7.diff_pts}}}} (C7); Holm p = "
         f"{{{{p: {c}.arl_makes_selection_useful_c3_c6.p_holm}}}}, {{{{p: {c}.arl_makes_selection_useful_c5_c7.p_holm}}}}",
         "Not supported"],
        ["H3b", "ARL improves System 1",
         f"{{{{pct0: {c}.arl_transformer_s1_nt.a_pct}}}} vs {{{{pct0: {c}.arl_transformer_s1_nt.b_pct}}}} (C6 vs C3); "
         f"{{{{pct0: {c}.arl_hybrid_s1_nt.a_pct}}}} vs {{{{pct0: {c}.arl_hybrid_s1_nt.b_pct}}}} (C7 vs C5)", "Not supported"],
        ["H4", "The hybrid is a better state estimator under camera dropout",
         f"{{{{pct0: {cam}.a_pct}}}} vs {{{{pct0: {cam}.b_pct}}}}: {_ci(cam)}, Holm p = {{{{p: {cam}.p_holm}}}}",
         "Not supported"],
        ["H5", "Two-level System 1 helps long horizons without hurting short ones",
         f"C3: {{{{pct0: {h5c3}.a_pct}}}} vs {{{{pct0: {h5c3}.b_pct}}}}, "
         f"C5: {{{{pct0: {h5c5}.a_pct}}}} vs {{{{pct0: {h5c5}.b_pct}}}}, "
         f"better in every seed; C1, C2: small, not consistent; pooled over {{{{int: {h5}.units}}}} models "
         f"{{{{pct0: {h5}.a_pct}}}} vs {{{{pct0: {h5}.b_pct}}}}: {_ci(h5)}, Holm p = {{{{p: {h5}.p_holm}}}}; grasp "
         f"{{{{pct0: {h5p}.a_pct}}}} vs {{{{pct0: {h5p}.b_pct}}}}", "Supported on object-grounded models"],
        ["H6", "The full combination (C7) beats LeWM (C1)",
         f"{{{{pct0: {h6}.a_pct}}}} vs {{{{pct0: {h6}.b_pct}}}}: {_ci(h6)}; grasp {{{{pct0: {h6p}.a_pct}}}} vs "
         f"{{{{pct0: {h6p}.b_pct}}}}: {_ci(h6p)}; Holm p = {{{{p: {h6}.p_holm}}}} and {{{{p: {h6p}.p_holm}}}}",
         "Supported (seed 0)"],
    ]
    return caption, header, rows, _w(.06, .24, .48, .22)


# ---------------------------------------------------------------------------------------------- Table 3
CONFIG_ROWS = [("C1", "C1 LeWM"), ("C2", "C2 + proprio grounding"), ("C3", "C3 + object grounding"),
               ("C5", "C5 hybrid + object"), ("C6", "C6 + object + ARL"), ("C7", "C7 hybrid + object + ARL")]


def gpu_success():
    caption = ("Table 3. GPU study: success (%) on {{set: 60 | evaluation pairs per seed, GPU study}} non-trivial pairs "
               "per seed (grasp: H-JEPA windows), mean ± SE over {{set: 3 | training seeds of the main configurations}} "
               "seeds (C6, C7: {{set: 1 | training seeds of C6 and C7}} seed), and the error of reading the cube's "
               "position from the latent (seed 0).")
    header = ["Config", "System 1, 25 steps", "Grasp windows", "Flat, 50 steps", "Two-level, 50 steps",
              "Best-of-16 by world model", "CEM", "Cube read-out (cm)"]
    rows = []
    for cfg, name in CONFIG_ROWS:
        p = f"gpu_cube.success.{cfg}"
        cells = [name]
        for ev in ("s1_nt", "s1_pk", "s1_50", "hier_50", "sel16", "cem"):
            if cfg in ("C6", "C7"):
                cells.append(f"{{{{num1: {p}.{ev}.mean}}}}")
            else:
                cells.append(_mean_se(p, ev))
        cells.append(f"{{{{num1: gpu_cube.diagnostics.{cfg}.cube_probe_cm}}}}")
        rows.append(cells)
    return caption, header, rows, _w(.24, .11, .11, .10, .11, .12, .11, .10)


# ---------------------------------------------------------------------------------------------- Table 4
PER_SEED = [("Object grounding, C3 vs C2", "object_grounding_s1_nt"),
            ("System 1 vs CEM, C1", "c1_system_1_vs_cem_cem"),
            ("Two-level vs flat, 50 steps, C3", "c3_two_level_vs_flat_50_hier_50"),
            ("Two-level vs flat, 50 steps, C5", "c5_two_level_vs_flat_50_hier_50"),
            ("Two-level vs flat, 50 steps, C1", "c1_two_level_vs_flat_50_hier_50"),
            ("Prediction vs hold, 50% lost, C3", "c3_camera_0_5_prediction_vs_hold_cam0_5"),
            ("Prediction vs hold, 50% lost, C5", "c5_camera_0_5_prediction_vs_hold_cam0_5"),
            ("Best-of-16 vs mean, C3", "c3_best_of_16_vs_mean_sel16"),
            ("Hybrid vs transformer, C5 vs C3", "hybrid_predictor_object_s1_nt")]


def per_seed():
    caption = ("Table 4. Main contrasts in each seed separately (success %, A vs B, with the exact McNemar p on the same "
               "{{set: 60 | evaluation pairs per seed, GPU study}} pairs; uncorrected) and over all three seeds "
               "(A − B in points, 95% interval from resampling the shared pairs).")
    header = ["Contrast (A vs B)", "Seed 0", "Seed 1", "Seed 2", "All seeds [95% CI]"]
    rows = []
    for name, key in PER_SEED:
        p = f"gpu_cube.contrasts.{key}"
        cells = [name.replace("50% lost", f"{S % ('50%', 'share of camera frames lost')} lost")]
        for s in ("s0", "s1", "s2"):
            q = f"{p}.per_seed.{s}"
            cells.append(f"{{{{num1: {q}.a_pct}}}} vs {{{{num1: {q}.b_pct}}}} ({{{{p: {q}.p_mcnemar}}}})")
        cells.append(_ci(p))
        rows.append(cells)
    return caption, header, rows, _w(.21, .19, .19, .19, .22)


# ---------------------------------------------------------------------------------------------- Table 5
TWOROOM = [("Two-level vs flat, 50 steps (H5)", "all_configs_two_level_vs_flat_50_hier_50"),
           ("Proprio grounding (C2 vs C1), 25 steps", "proprio_grounding_s1_nt"),
           ("System 1 vs CEM (C1), 25 steps", "c1_system_1_vs_cem_cem"),
           ("System 1 vs CEM (C1), all pairs", "c1_system_1_vs_cem_all_pairs_cem_all"),
           ("Flow vs Gaussian System 1, 25 steps", "all_configs_flow_vs_gaussian_system_1_25_s1f_nt"),
           ("Best-of-16 near the mean vs mean, 25 steps", "all_configs_best_of_16_near_the_mean_vs_mean_exploratory_sel16n")]


def tworoom():
    caption = ("Table 5. TwoRoom: paired contrasts pooled over C1 and C2 and three seeds where both apply (up to "
               "{{set: 360 | most outcomes in one TwoRoom contrast (2 models x 3 seeds x 60 pairs)}} outcomes of the same "
               "{{set: 60 | evaluation pairs per seed, GPU study}} pairs). A − B with a 95% interval from resampling "
               "the shared pairs; McNemar p pools the outcomes as if independent (descriptive).")
    header = ["Contrast", "A", "B", "A − B [95% CI]", "A-only vs B-only", "McNemar p"]
    rows = []
    for name, key in TWOROOM:
        p = f"gpu_tworoom.contrasts.{key}"
        rows.append([name, f"{{{{pct0: {p}.a_pct}}}}", f"{{{{pct0: {p}.b_pct}}}}", _ci(p),
                     f"{{{{int: {p}.a_only}}}} vs {{{{int: {p}.b_only}}}}", f"{{{{p: {p}.p_mcnemar}}}}"])
    return caption, header, rows, _w(.30, .09, .09, .2, .17, .15)


# ---------------------------------------------------------------------------------------------- Tables 6 and 7
FB_MODELS = ["C1 seed 0", "C1 seed 1", "C3 seed 0", "C3 seed 1"]
FB_ROWS = [("S1-25", "S1-25 (default)"), ("S1-5", "S1-5"), ("S1-1", "S1-1"), ("S1c5", "S1c5 (trained for 5 steps)"),
           ("S1c1", "S1c1 (trained for 1 step)"), ("IDM-1", "IDM-1 (our GC-IDM-style)"),
           ("IDM-1-noH", "IDM-1 without horizon input"), ("OIDM-1", "OIDM-1 (authors' GC-IDM code)"),
           ("OIDM-1-H1", "OIDM-1, H_max = 1 (their ablation)")]


def feedback():
    caption = ("Table 6. How often the controller looks (Section 5.8; CPU-study models, "
               "{{set: 60 | evaluation pairs, feedback study}} non-trivial pairs, success %). S1-n: System 1 executes n "
               "steps of its 25-step chunk, then decides again from a new frame. S1c5, S1c1: System 1 trained to output "
               "a 5- or 1-step chunk, executed whole. IDM-1: our GC-IDM-style controller; OIDM-1: the GC-IDM authors' "
               "code; both decide every step.")
    header = ["Condition", *FB_MODELS, "Pooled"]
    rows = []
    for key, name in FB_ROWS:
        rows.append([name, *[f'{{{{num1: feedback.success_pct["{m}"]["{key}"]}}}}' for m in FB_MODELS],
                     f'{{{{num1: feedback.pooled_pct["{key}"]}}}}'])
    return caption, header, rows, _w(.30, .14, .14, .14, .14, .14)


def feedbackc():
    caption = ("Table 7. Pre-registered contrasts of Table 6. Per model: pairs solved only by A vs only by B. All: mean "
               "difference A − B in points over the four models, with a 95% interval from resampling the "
               "{{set: 40 | evaluation episodes behind the 60 feedback-study pairs}} evaluation episodes "
               "({{set: 10,000 | bootstrap resamples}} resamples). Holm p: two-sided bootstrap p, Holm-corrected over "
               "the {{set: 12 | contrasts of the feedback study}} contrasts (resolution "
               "{{set: 0.0002 | resolution of a two-sided bootstrap p with 10,000 resamples}} before correction). "
               "* claimed: Holm p < {{set: 0.05 | significance level}} and the same sign in all four models.")
    header = ["Contrast (A vs B)", *FB_MODELS, "All [95% CI]", "Holm p"]
    rows = []
    from json import loads
    from pathlib import Path
    fb = loads((Path(__file__).resolve().parents[1] / "results" / "paper" / "feedback.json").read_text())
    for name, con in fb["contrasts"].items():
        p = f'feedback.contrasts["{name}"]'
        e = f"{p}.bootstrap_episode"
        star = "*" if con["claimed"] else ""
        rows.append([f"{name} {con['a']} vs {con['b']}",
                     *[f'{{{{int: {p}.per_model["{m}"].a_only}}}} vs {{{{int: {p}.per_model["{m}"].b_only}}}}'
                       for m in FB_MODELS],
                     f"{{{{sgn1: {e}.diff_pts}}}} [{{{{dec1: {e}.ci_lo}}}}, {{{{dec1: {e}.ci_hi}}}}]",
                     f"{{{{p: {p}.p_boot_holm}}}}{star}"])
    return caption, header, rows, _w(.21, .11, .11, .11, .11, .22, .13)


# ---------------------------------------------------------------------------------------------- Table 8
def standard():
    caption = ("Table 8. Success (%) on the authors' standard protocol, mean ± SE over "
               "{{set: 3 | training seeds of the main configurations}} seeds.")
    header = ["Config", "System 1", "CEM"]
    rows = []
    for cfg, name in CONFIG_ROWS[:4]:
        p = f"gpu_cube.success.{cfg}"
        rows.append([name, _mean_se(p, "s1_all"), _mean_se(p, "cem_all") if cfg in ("C1", "C5") else "–"])
    return caption, header, rows, _w(.4, .3, .3)


# ---------------------------------------------------------------------------------------------- Table 9
def impl():
    m = "model_sizes"

    def mp(path: str) -> str:
        return f"{{{{num2: {m}.{path} / 1000000}}}} M"
    caption = "Table 9. Implementation details (GPU study unless stated)."
    header = ["Item", "Setting"]
    rows = [
        ["Frames", f"{S % ('64', 'frame size in pixels')} × {S % ('64', 'frame size in pixels')} RGB, one camera"],
        ["Encoder", f"ViT-tiny from scratch: {S % ('12', 'encoder layers')} layers, width {S % ('192', 'latent width')}, "
                    f"{S % ('3', 'attention heads of the encoder')} heads, MLP {S % ('768', 'encoder MLP width')}, "
                    f"patch {S % ('8', 'patch size')}"],
        ["Predictor", f"{S % ('6', 'predictor layers')}-layer transformer (or 3:1 Mamba-3/attention hybrid), last "
                      f"{S % ('3', 'latents of history')} latents + one {S % ('5', 'steps per action block')}-step action block"],
        ["Loss", f"next-latent MSE + {S % ('0.09', 'SIGReg weight')} × SIGReg ({S % ('1,024', 'SIGReg projections')} "
                 f"projections); grounding heads weight {S % ('1', 'grounding head loss weight')} each"],
        ["Parameters", f"C1 {mp('world_models.C1.params')}; C2 {mp('world_models.C2.params')}; "
                       f"C3 {mp('world_models.C3.params')}; C5 {mp('world_models.C5.params')}"],
        ["World-model optimiser", f"AdamW, lr {S % ('0.0003', 'world-model learning rate')}, weight decay "
                                  f"{S % ('0.001', 'world-model weight decay')}, {S % ('5%', 'warm-up share')} warm-up "
                                  f"then cosine to {S % ('1%', 'final learning-rate share')} of lr; "
                                  f"{S % ('4,000', 'GPU-study world-model steps')} steps, "
                                  f"batch {S % ('64', 'GPU-study batch size')}"],
        ["System 1", f"MLP {S % ('576', 'System 1 input width')}–{S % ('512', 'System 1 hidden width')}–"
                     f"{S % ('512', 'System 1 hidden width')}–{S % ('250', 'System 1 output width')}, LayerNorm, GELU, "
                     f"{mp('policies.system1')} parameters; AdamW lr {S % ('0.001', 'policy learning rate')}, weight decay "
                     f"{S % ('0.0001', 'policy weight decay')}, cosine, gradient clip {S % ('1.0', 'gradient clip')}, "
                     f"batch {S % ('256', 'policy batch size')}, {S % ('3,000', 'policy training steps')} steps"],
        ["Subgoal network", f"{S % ('3,000', 'policy training steps')} steps on the same cached latents"],
        ["CEM", f"{S % ('300', 'CEM samples')} samples, {S % ('30', 'CEM iterations')} iterations, top "
                f"{S % ('30', 'CEM elites')}, horizon {S % ('5', 'CEM horizon in blocks')} blocks"],
        ["Random policy", "uniformly random actions (stable-worldmodel RandomPolicy)"],
        ["Success", f"OGBench Cube task success: cube within {S % ('4', 'success distance in cm')} cm of the goal position"],
        ["Evaluation", f"evaluation seed {S % ('42', 'evaluation seed')}; the same "
                       f"{S % ('60', 'evaluation pairs per seed, GPU study')} pairs for every model and seed"],
        ["Software", f"stable-worldmodel, MuJoCo {S % ('3.15', 'MuJoCo version')}, PyTorch; free Colab T4 GPU"],
        ["CPU-study models (Section 5.8)", f"same architecture; {S % ('400', 'CPU-study training episodes')} training episodes; "
                                           f"{S % ('2,000', 'CPU-study world-model steps')} steps, batch "
                                           f"{S % ('32', 'CPU-study batch size')}; {S % ('2', 'CPUs of the CPU study')} CPUs"],
        ["Short-chunk System 1 (Section 5.8)",
         f"as System 1 with a 1-block output of {S % ('5', 'short chunk length in steps')} or "
         f"{S % ('1', 'short chunk length in steps')} "
         f"steps ({S % ('25', 'action numbers of a 5-step chunk')} or {S % ('5', 'action numbers per step')} numbers); target: "
         f"the next {S % ('5', 'short chunk length in steps')} or {S % ('1', 'short chunk length in steps')} actions; goals "
         f"{S % ('5', 'nearest training goal in steps')}–{S % ('25', 'farthest training goal in steps')} steps ahead; "
         f"{mp('policies.short_chunk_5')} and {mp('policies.short_chunk_1')} parameters"],
        ["Authors' GC-IDM (Section 5.8)",
         "their train_idm.py, unmodified (commit 48c45b1): MLP on [z_now, z_goal], "
         f"{S % ('3', 'GC-IDM hidden layers')} layers of width {S % ('512', 'GC-IDM width')}, LayerNorm, GELU, dropout "
         f"{S % ('0.1', 'GC-IDM dropout')}, remaining-step fraction → sinusoidal features → AdaLN-Zero; MSE on raw actions; "
         f"goals {S % ('1', 'nearest GC-IDM goal in steps')}–{S % ('50', 'farthest GC-IDM goal in steps')} steps ahead "
         f"(H_max {S % ('50', 'farthest GC-IDM goal in steps')}, or {S % ('1', 'H_max of their ablation')} for their ablation); "
         f"AdamW lr {S % ('0.001', 'policy learning rate')}, weight decay {S % ('0.0001', 'policy weight decay')}, cosine to "
         f"lr/{S % ('100', 'final learning-rate divisor')}, clip {S % ('1.0', 'gradient clip')}, batch "
         f"{S % ('1,024', 'GC-IDM batch size')}, {S % ('50', 'GC-IDM epochs')} epochs, checkpoint with the lowest loss on a "
         f"random {S % ('10%', 'validation share')} of pairs; {mp('official_gcidm.params')} parameters; at test time remaining "
         f"steps = min({S % ('50', 'step budget')} − steps done, H_max)"],
        ["Our GC-IDM-style controller (Section 5.8)",
         "implemented independently from the description in [22]: MLP on [z_now, z_goal], "
         f"{S % ('3', 'GC-IDM hidden layers')} layers of width {S % ('512', 'GC-IDM width')}, LayerNorm, GELU, dropout "
         f"{S % ('0.1', 'GC-IDM dropout')}; remaining steps / {S % ('50', 'step budget')} → "
         f"{S % ('64', 'sinusoidal features')} "
         f"sinusoidal features → 2-layer MLP → AdaLN-Zero on the backbone output; one action out; MSE on (z_t, z_t+h, a_t), "
         f"h uniform in {S % ('1', 'nearest GC-IDM goal in steps')}–{S % ('50', 'farthest GC-IDM goal in steps')} steps; AdamW "
         f"lr {S % ('0.001', 'policy learning rate')}, weight decay {S % ('0.0001', 'policy weight decay')}, cosine to "
         f"lr/{S % ('100', 'final learning-rate divisor')}, clip {S % ('1.0', 'gradient clip')}, batch "
         f"{S % ('1,024', 'GC-IDM batch size')}, {S % ('50', 'GC-IDM epochs')} epochs; {mp('policies.gcidm_ours')} parameters "
         f"({mp('policies.gcidm_ours_no_horizon')} without horizon); at test time h = {S % ('50', 'step budget')} − t + "
         f"{S % ('1', 'offset of the remaining-step count')}"],
    ]
    return caption, header, rows, _w(.26, .74)


TABLES = {"scale": scale, "hypotheses": hypotheses, "gpu_success": gpu_success, "per_seed": per_seed,
          "tworoom": tworoom, "feedback": feedback, "feedbackc": feedbackc, "standard": standard, "impl": impl}


def build(name: str, f) -> dict:
    from build import runs
    caption, header, rows, widths = TABLES[name]()
    return {"type": "table", "name": name, "caption": runs(f.text(caption)),
            "header": [runs(f.text(h)) for h in header],
            "rows": [[runs(f.text(c)) for c in r] for r in rows], "widths": widths}
