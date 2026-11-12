# Result files

Every result file behind the paper's numbers. One JSON per evaluated (model, seed, controller, test); `successes` is
the 0/1 outcome of each evaluation pair, in pair order.

| Folder | What |
|---|---|
| `gpu_study/runs/` | GPU study (Tables 2–5, 8; Figures 2–3): 255 files, Cube and TwoRoom |
| `gpu_study/runs_superseded/` | 22 earlier files of C2 seed 1 that were replaced by later runs of the same configuration; kept for completeness, not used |
| `gpu_study/summary_cube/`, `summary_tworoom/` | Output of `analyze_colab.py --profile free` on these files (tables, paired contrasts with bootstrap intervals) |
| `feedback_study/runs/` | Feedback-frequency study (Section 5.8, Tables 6–7): 36 files (4 models x 9 conditions) |
| `feedback_study/feedback_summary.json` | Output of `analyze_feedback.py`, including the pair identities (evaluation episode and start step of each pair) |

Pair identities. The feedback-study files record each pair's evaluation episode (`episodes_idx`) and start step
(`start_steps`). The GPU-study files do not; those pairs are fixed by evaluation seed 42 and are regenerated from the
evaluation data by `bench/lewm_suite/check_goal_collisions.py`, which also lists them.

Re-running the summaries:

```bash
cd bench/lewm_suite
mkdir -p /tmp/gpu/results && cp ../../results/gpu_study/runs/*.json /tmp/gpu/results/
python analyze_colab.py --profile free --home /tmp/gpu --task cube      # and --task tworoom; output in /tmp/gpu/results/colab*
python analyze_feedback.py           # reads the feedback-study files from $STABLEWM_HOME/results
```
