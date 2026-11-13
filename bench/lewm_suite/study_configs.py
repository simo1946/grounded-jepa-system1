"""Configurations and run profiles of the GPU study (shared by colab_plan.py and the analysis scripts).

Kept apart from colab_plan.py, which schedules jobs and reads the clock, so that the analysis instruments
depend only on these tables."""

CONFIGS = {
    "C1": "lewm",
    "C2": "lewm+idm+sa",
    "C3": "lewm+idm+sa+obj",
    "C4": "hybrid+idm+sa",
    "C5": "hybrid+idm+sa+obj",
    "C6": "lewm+idm+sa+obj+arl",
    "C7": "hybrid+idm+sa+obj+arl",
    "C8": "lewm+idm",
    "C9": "lewm+idm+sa+obj+enc=mnv4s",
}
TIERS = {1: [(c, 0) for c in ("C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9")],
         2: [(c, s) for s in (1, 2) for c in ("C1", "C2", "C3", "C4", "C5", "C6", "C7")],
         3: []}
# C8/C9 seed 0 sit in tier 1 so the first session already answers every question once.
CEM_SET = {"C1", "C3", "C6", "C7"}
MEMORY_SET = {"C3", "C5", "C6", "C7"}

PROFILES = {
    # ~2000 expert episodes, 64 px: 10x the CPU study's data, 5x its training steps, 2x its eval pairs
    "standard": dict(n_train=2000, n_eval=200, train_steps=10000, batch=128, lr=3e-4, policy_steps=6000,
                     subgoal_steps=6000, num_eval=100, env_chunk=10, diag_clips=256),
    "quick": dict(n_train=800, n_eval=120, train_steps=4000, batch=128, lr=3e-4, policy_steps=3000,
                  subgoal_steps=3000, num_eval=60, env_chunk=10, diag_clips=256),
    "smoke": dict(n_train=8, n_eval=6, train_steps=10, batch=8, lr=3e-4, policy_steps=20, subgoal_steps=20,
                  num_eval=3, env_chunk=3, diag_clips=8, part_size=4),
    # free Colab (T4, 2 vCPUs, ~12.7 GB RAM): fewer models and evaluations, aimed at the hypotheses
    "free": dict(n_train=600, n_eval=100, train_steps=4000, batch=64, lr=3e-4, policy_steps=3000,
                 subgoal_steps=3000, num_eval=60, env_chunk=4, diag_clips=128,
                 evals={"s1_nt", "s1_pk", "s1f_nt", "s1f_pk", "s1_50", "hier_50", "hier_pk", "sel16", "sel16n", "cem", "rec1",
                        "cam0.5-hold", "cam0.5-predict", "s1_all", "cem_all"},
                 cem_set={"C1", "C5"}, memory_set={"C3", "C5"},
                 tiers={1: [(c, 0) for c in ("C1", "C2", "C3", "C5")],
                        2: [(c, 1) for c in ("C1", "C2", "C3", "C5")],
                        3: [(c, 2) for c in ("C1", "C2", "C3", "C5")] + [("C6", 0), ("C7", 0)],
                        4: [(c, s) for s in (1, 2) for c in ("C6", "C7")]}),   # optional: ARL seeds 1-2
}
# 'free' tiers: 1 = LeWM, proprio grounding, object grounding, hybrid + object (seed 0, one ~4-5 h
# session); 2 and 3 = seeds 1 and 2. The action-resolution configs (C6, C7) moved to tier 3: on the
# CPU study (seed 0) ARL did not improve System 1 (34 vs 35/50) and made world-model selection worse
# (22 vs 28/50), so they are a lower priority for scarce GPU time.
