"""Re-run one finished evaluation of the plan and compare it with its stored result.

Used after the Cube training data had to be collected again (the first copy was deleted, and the collector does
not reproduce the same episodes from a seed): evaluations z-score actions with statistics of the training data, so
re-running a stored evaluation with the new data shows whether that changes results. The job runs exactly as
colab_plan.py defines it, in a separate home that links the checkpoints and datasets and has its own empty results
folder, so the stored result is never touched. Writes <home>/results/checks/<job>.json.

    python rerun_check.py --profile free --home $STABLEWM_HOME --job eval:C1s0:cem
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def find_job(profile: str, home: Path, name: str, task: str):
    from colab_plan import build
    jobs = build(profile, [1, 2, 3], None, home, 1, None, None, task=task)
    match = [j for j in jobs if j.name == name]
    if not match:
        raise KeyError(f"{name} is not a job of profile {profile!r}")
    return match[0]


def main(profile: str, home: Path, job_name: str, task: str) -> dict:
    job = find_job(profile, home, job_name, task)
    stored = job.outputs[0]
    if not stored.exists():
        raise FileNotFoundError(f"no stored result to compare with: {stored}")
    check_home = home.parent / f"{home.name}__check"
    (check_home / "results").mkdir(parents=True, exist_ok=True)
    for d in ("checkpoints", "datasets", "cache"):
        link = check_home / d
        if not link.exists():
            link.symlink_to((home / d).resolve(), target_is_directory=True)
    new = check_home / "results" / stored.name
    new.unlink(missing_ok=True)                                 # a leftover of an earlier check
    subprocess.run(job.cmd, cwd=HERE, env={**os.environ, "STABLEWM_HOME": str(check_home)}, check=True)
    a, b = json.loads(stored.read_text()), json.loads(new.read_text())
    if a["episodes_idx"] != b["episodes_idx"] or a["start_steps"] != b["start_steps"]:
        raise RuntimeError("the re-run drew different pairs: not comparable")
    sa, sb = a["successes"], b["successes"]
    res = {"job": job_name, "result_file": stored.name, "pairs": len(sa), "stored_solved": int(sum(sa)),
           "rerun_solved": int(sum(sb)), "same_outcome": int(sum(x == y for x, y in zip(sa, sb, strict=True))),
           "stored_only": int(sum(x and not y for x, y in zip(sa, sb, strict=True))),
           "rerun_only": int(sum(y and not x for x, y in zip(sa, sb, strict=True))),
           "training_data": str((home / "datasets" / f"{task}.h5").resolve())}
    out = home / "results" / "checks" / f"{job_name.replace(':', '__')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**res, "rerun": b}, indent=1))
    print(json.dumps(res), flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="free")
    ap.add_argument("--task", default="cube")
    ap.add_argument("--job", default="eval:C1s0:cem", help="a finished evaluation job of colab_plan.py")
    ap.add_argument("--home", type=Path, default=Path(os.environ.get("STABLEWM_HOME", HERE / "swm_home")))
    a = ap.parse_args()
    main(a.profile, a.home, a.job, a.task)
