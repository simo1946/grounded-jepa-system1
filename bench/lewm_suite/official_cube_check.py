"""Check, on the LeWM authors' released Cube dataset, how many evaluation pairs are already solved at the start,
and what a random policy scores on their protocol.

Meant for a Linux machine with internet access and at least 100 GB of free disk (the archive holds one 95 GB .h5
file, measured on 2026-11-13; it must be on disk to be read):

    python official_cube_check.py --out official_cube_check.json            # steps 1-2
    python official_cube_check.py --out official_cube_check.json --random-eval --lewm-repo le-wm   # + step 3

1. Lists the files of the Hugging Face dataset repository ``quentinll/lewm-cube`` (linked from the le-wm README)
   and streams its ``.tar.zst`` archive through zstd and tar, writing only the ``.h5`` member to disk (the
   archive itself is never stored). Before writing, it compares the member's size with the free disk space.
2. Runs ``trivial_share.summary`` on the extracted file: the share of all valid start/goal pairs (goal 25 steps
   ahead) already solved at the start, and the same for the 50 start rows that le-wm ``eval.py`` draws with
   seed 42.
3. Optionally runs the authors' ``eval.py`` unmodified, with its default ``policy=random`` and their
   ``config/eval/cube.yaml``, and records the success rate it prints.

Prints one JSON object (and writes it to --out).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

REPO_ID = "quentinll/lewm-cube"
LEWM_COMMIT = "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"


def find_archive() -> tuple[str, str, int | None]:
    from huggingface_hub import HfApi, hf_hub_url
    api = HfApi()
    files = [f for f in api.list_repo_tree(REPO_ID, repo_type="dataset", recursive=True)
             if getattr(f, "path", "").endswith((".tar.zst", ".h5"))]
    if not files:
        raise RuntimeError(f"no .tar.zst or .h5 file in {REPO_ID}")
    f = max(files, key=lambda x: getattr(x, "size", 0) or 0)
    return f.path, hf_hub_url(REPO_ID, f.path, repo_type="dataset"), getattr(f, "size", None)


def stream_extract(url: str, dest: Path) -> Path:
    """Stream the .tar.zst through zstd and tar; write only the .h5 member to dest."""
    import requests
    import zstandard
    dest.mkdir(parents=True, exist_ok=True)
    headers = {}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        reader = zstandard.ZstdDecompressor(max_window_size=2 ** 31).stream_reader(r.raw)
        with tarfile.open(fileobj=reader, mode="r|") as tf:
            for m in tf:
                if not (m.isfile() and m.name.endswith(".h5")):
                    continue
                free = shutil.disk_usage(dest).free
                print(json.dumps({"member": m.name, "bytes": m.size, "free_bytes": free}), flush=True)
                if m.size > free - 2 * 2 ** 30:
                    raise RuntimeError(f"{m.name} needs {m.size / 2**30:.1f} GB; only {free / 2**30:.1f} GB free")
                out = dest / Path(m.name).name
                src = tf.extractfile(m)
                with open(out, "wb") as w:
                    shutil.copyfileobj(src, w, length=64 * 2 ** 20)
                return out
    raise RuntimeError("no .h5 member in the archive")


def random_eval(h5: Path, lewm_repo: Path, home: Path) -> dict:
    """Run le-wm eval.py (pinned commit, unmodified) with policy=random on the official file."""
    head = subprocess.run(["git", "-C", str(lewm_repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    if head != LEWM_COMMIT:
        raise RuntimeError(f"{lewm_repo} is at {head}, expected {LEWM_COMMIT}")
    target = home / "datasets" / "ogbench" / "cube_single_expert.h5"   # name in config/eval/cube.yaml
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        os.symlink(h5.resolve(), target)
    env = {**os.environ, "STABLEWM_HOME": str(home), "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl")}
    cmd = [sys.executable, "eval.py", "--config-name=cube.yaml", "policy=random",
           f"hydra.run.dir={home / 'hydra_outputs'}"]                  # keep the clone unmodified
    p = subprocess.run(cmd, cwd=lewm_repo, env=env, capture_output=True, text=True)
    if p.returncode != 0:                                            # -9: killed, usually out of memory
        raise RuntimeError(f"eval.py failed ({p.returncode})\nstdout:\n{p.stdout[-3000:]}\nstderr:\n{p.stderr[-3000:]}")
    rate = re.findall(r"'success_rate': ([0-9.]+)", p.stdout)        # eval.py prints the metrics dict
    succ = re.findall(r"'episode_successes': array\(\[([^\]]*)\]", p.stdout, flags=re.S)
    if not rate:
        raise RuntimeError("eval.py printed no metrics:\n" + p.stdout[-3000:])
    flags = re.findall(r"True|False|1\.|0\.", succ[-1]) if succ else []
    return {"success_rate": float(rate[-1]), "episodes": len(flags),
            "successes": sum(f in ("True", "1.") for f in flags),
            "lewm_commit": head, "command": "python eval.py --config-name=cube.yaml policy=random",
            "stdout_tail": p.stdout[-2000:]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("/content/lewm_cube"))
    ap.add_argument("--h5", type=Path, default=None, help="skip the download and use this file")
    ap.add_argument("--random-eval", action="store_true")
    ap.add_argument("--lewm-repo", type=Path, default=Path("le-wm"))
    ap.add_argument("--home", type=Path, default=Path("/content/swm_home_official"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from trivial_share import summary
    res: dict = {"hf_repo": REPO_ID}
    h5 = a.h5
    if h5 is None:
        path, url, size = find_archive()
        res.update({"hf_file": path, "hf_file_bytes": size})
        h5 = stream_extract(url, a.data_dir) if path.endswith(".tar.zst") else None
        if h5 is None:
            from huggingface_hub import hf_hub_download
            h5 = Path(hf_hub_download(REPO_ID, path, repo_type="dataset", local_dir=a.data_dir))
    res["h5_bytes"] = h5.stat().st_size
    res["trivial_share"] = summary(h5)
    a.out.write_text(json.dumps(res, indent=1))           # saved before the slower, memory-heavy random run
    if a.random_eval:
        try:                                              # 50 parallel 224-px environments need about 9 GB of RAM
            res["random_policy"] = random_eval(h5, a.lewm_repo, a.home)
        except RuntimeError as e:
            res["random_policy"] = {"error": str(e)[-3000:]}
        a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "trivial_share"} |
                     {"all_pairs": res["trivial_share"]["all_pairs"],
                      "lewm_eval_draw": {k: v for k, v in res["trivial_share"]["lewm_eval_draw"].items()
                                         if k != "rows"}}, indent=1))
