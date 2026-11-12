"""Health check of a study home (models, results, logs) and of the storage it lives on.

Reports, per trained model: weights present / readable / plausible size, whether training was interrupted
or resumed, and whether policies or results were built from an older copy of the weights; plus free space
and what uses it. Read-only: it never changes or deletes anything.

    python check_storage.py --home /content/swm_home_img64 --drive /content/drive/MyDrive
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import time
from pathlib import Path

GB = 1e9
TOL_S = 3600            # same tolerance as colab_plan.STALE_TOLERANCE_S


def du(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def when(p: Path) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))


def space(home: Path, drive: Path | None) -> list[str]:
    out = ["## Storage"]
    for label, p in (("Google Drive", drive), ("local disk", home)):
        if p and p.exists():
            u = shutil.disk_usage(p)
            flag = "  <-- NEARLY FULL: files can be lost" if u.free < 1 * GB else ""
            out.append(f"- {label}: {u.free / GB:.1f} GB free of {u.total / GB:.0f} GB{flag}")
    for d in ("checkpoints", "results", "logs", "datasets", "cache"):
        p = home / d
        if p.exists():
            out.append(f"- {d}: {du(p) / GB:.2f} GB ({'Drive' if p.is_symlink() else 'local'})")
    if drive and (drive / "hybrid-wm").exists():
        for sub in sorted((drive / "hybrid-wm").iterdir()):
            size = du(sub) if sub.is_dir() else sub.stat().st_size
            out.append(f"- Drive hybrid-wm/{sub.name}: {size / GB:.2f} GB")
    out.append("- Files deleted from a mounted Drive go to the Drive bin and still count against the quota "
               "until the bin is emptied (drive.google.com > Bin).")
    return out


def models(home: Path) -> list[str]:
    import torch
    out, rows = ["## Models"], []
    for w_dir in sorted({p.parent for p in (home / "checkpoints").rglob("*.pt")}):
        if w_dir.name == "stale":
            continue
        run = w_dir.relative_to(home / "checkpoints")
        w = w_dir / "weights.pt"
        rows.append((run, w_dir, w))
    sizes: dict[str, list[int]] = {}
    for run, _, w in rows:
        if w.exists():
            sizes.setdefault(str(run.parent), []).append(w.stat().st_size)
    problems = 0
    for run, d, w in rows:
        notes = []
        pols = sorted(d.glob("policy*.pt"))
        if not w.exists():
            notes.append("WEIGHTS MISSING" + (f" but {len(pols)} policy file(s) exist: weights were lost after "
                                              "training finished" if pols else ""))
            if (d / "resume.pt").exists():
                notes.append("training was interrupted (resume.pt present, will continue)")
        else:
            ref = statistics.median(sizes[str(run.parent)])
            if abs(w.stat().st_size - ref) > 0.05 * ref:
                notes.append(f"weights size {w.stat().st_size / 1e6:.0f} MB vs {ref / 1e6:.0f} MB for the same "
                             "model type: possibly truncated")
            try:
                torch.load(w, map_location="cpu", weights_only=True)
            except Exception as e:                                          # noqa: BLE001
                notes.append(f"weights UNREADABLE ({type(e).__name__})")
            old = [p.name for p in pols if p.stat().st_mtime < w.stat().st_mtime - TOL_S]
            if old:
                notes.append(f"built from an older copy of the weights (will be redone): {', '.join(old)}")
            info = d / "train_info.json"
            if info.exists():
                try:
                    i = json.loads(info.read_text())
                    if i.get("resumed_at_steps"):
                        notes.append(f"training resumed at steps {i['resumed_at_steps']} (valid, not bit-identical)")
                except json.JSONDecodeError:
                    notes.append("train_info.json unreadable")
        if (d / "stale").exists():
            notes.append(f"{len(list((d / 'stale').iterdir()))} older file(s) kept in stale/")
        problems += any(k in n for n in notes for k in ("MISSING", "UNREADABLE", "truncated", "older copy"))
        when_w = when(w) if w.exists() else "-"
        out.append(f"- {run}: weights {when_w}; " + ("; ".join(notes) if notes else "ok"))
    out.append(f"{problems} model(s) with a problem" if problems else "all models ok")
    return out


def results(home: Path) -> list[str]:
    out, bad, n = ["## Results"], [], 0
    for f in sorted((home / "results").glob("*.json")):
        n += 1
        try:
            json.loads(f.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            bad.append(f.name)
    st = home / "results" / "stale"
    out.append(f"- {n} result files, {len(bad)} unreadable" + (f": {', '.join(bad)}" if bad else ""))
    if st.exists():
        out.append(f"- {len(list(st.iterdir()))} older result(s) kept in results/stale/")
    return out


def failed_logs(home: Path) -> list[str]:
    out = ["## Why recent jobs failed (last error line of each failed log)"]
    for log in sorted((home / "logs" / "plan").glob("*.log"), key=lambda p: p.stat().st_mtime)[-200:]:
        lines = [ln for ln in log.read_text(errors="replace").splitlines() if ln.strip()]
        err = next((ln for ln in reversed(lines) if "Error" in ln or "error:" in ln), None)
        if err and "Traceback" in log.read_text(errors="replace"):
            out.append(f"- {log.stem} ({when(log)}): {err.strip()[:160]}")
    return out if len(out) > 1 else out + ["- none"]


def main(home: Path, drive: Path | None) -> str:
    report = "\n".join(space(home, drive) + [""] + models(home) + [""] + results(home) + [""] + failed_logs(home))
    print(report)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--drive", default=None, help="e.g. /content/drive/MyDrive")
    a = ap.parse_args()
    main(Path(a.home), Path(a.drive) if a.drive else None)
