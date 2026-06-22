"""One-shot monitor for the Attack C pipeline (fired every 2h by ScheduleWakeup).

Reads the tails of all training, rollout, and eval logs, scans for NaN / OOM /
stuck step counters, summarizes orbax checkpoint counts, and prints a single
status line. Designed to be called periodically; the parent loop decides
whether to re-schedule the next tick.

Conventions:
  - Logs live under /workspace/vla/attack_c_data/logs/{rollout_*, lam*, eval_lam*}.log
  - Attacked checkpoints live under /workspace/vla/attack_c_data/attacked/<exp>/<step>/
  - Eval rollouts live under /workspace/vla/attack_c_data/eval/lam*/libero_10/

The output is two parts:
  1. A one-liner summary printed first (suitable for log-grepping).
  2. A multi-section breakdown (per-run log tail snippets).
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import math
import os
import pathlib
import re
import shlex
import subprocess
import sys
from typing import Optional

ROOT = pathlib.Path(os.environ.get("ATTACK_C_DATA", "/workspace/vla/attack_c_data"))
LOGS = ROOT / "logs"
ATTACKED = ROOT / "attacked"
EVAL = ROOT / "eval"


def _tail(path: pathlib.Path, n: int = 200) -> str:
    if not path.exists():
        return ""
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            chunk = min(size, max(2048, n * 200))
            f.seek(size - chunk)
            data = f.read().decode("utf-8", errors="replace")
        return "\n".join(data.splitlines()[-n:])
    except Exception as exc:
        return f"<tail error: {exc}>"


def _has_nan(text: str) -> bool:
    return any(tok in text.lower() for tok in ("nan,", "nan ", "loss=nan", "isnan", " inf "))


def _has_oom(text: str) -> bool:
    return any(tok in text.lower() for tok in ("out of memory", "oom-killer", "cuda out of memory", "resourceexhausted"))


def _latest_step_from_log(text: str) -> Optional[int]:
    # Matches "Step 1234: ..." style lines emitted by scripts/train.py.
    matches = re.findall(r"Step (\d+):", text)
    if matches:
        return int(matches[-1])
    return None


def _log_age_minutes(path: pathlib.Path) -> Optional[float]:
    if not path.exists():
        return None
    return (datetime.datetime.now().timestamp() - path.stat().st_mtime) / 60.0


def _gpu_status() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return [line.strip() for line in out.stdout.splitlines() if line.strip()]
    except Exception as exc:
        return [f"<nvidia-smi error: {exc}>"]


def _scan_logs() -> dict:
    """Walk all known log files under LOGS, summarize each."""
    summary: dict[str, dict] = {}
    if not LOGS.exists():
        return summary
    for log_path in sorted(LOGS.glob("*.log")):
        text = _tail(log_path, n=400)
        last = text.splitlines()[-1] if text else ""
        summary[log_path.name] = {
            "tail_path": str(log_path),
            "age_min": _log_age_minutes(log_path),
            "step": _latest_step_from_log(text),
            "has_nan": _has_nan(text),
            "has_oom": _has_oom(text),
            "last_line": last[:300],
        }
    return summary


def _ckpt_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    if not ATTACKED.exists():
        return counts
    for exp_dir in sorted(ATTACKED.iterdir()):
        if not exp_dir.is_dir():
            continue
        # Each step is a numeric subdir under the experiment.
        step_dirs = [d for d in exp_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        counts[exp_dir.name] = len(step_dirs)
    return counts


def _eval_status() -> dict[str, int]:
    """Count episode_*.npz under EVAL/lam*/libero_10/."""
    out: dict[str, int] = {}
    if not EVAL.exists():
        return out
    for lam_dir in sorted(EVAL.glob("lam*")):
        npzs = list(lam_dir.rglob("episode_*.npz"))
        out[lam_dir.name] = len(npzs)
    return out


def _format_headline(logs: dict, ckpts: dict, eval_counts: dict, gpus: list[str]) -> str:
    bits = []
    if logs:
        active = sum(1 for v in logs.values() if v["age_min"] is not None and v["age_min"] < 5)
        nan = sum(1 for v in logs.values() if v["has_nan"])
        oom = sum(1 for v in logs.values() if v["has_oom"])
        bits.append(f"logs={len(logs)} active={active} nan={nan} oom={oom}")
    if ckpts:
        total = sum(ckpts.values())
        bits.append(f"ckpts={total} across {len(ckpts)} exps")
    if eval_counts:
        bits.append(f"eval_npz={sum(eval_counts.values())} across {len(eval_counts)} lams")
    if gpus:
        active_gpus = sum(1 for g in gpus if g and ", 0," not in f", {g.split(',')[1].strip()},")
        bits.append(f"gpus_active={active_gpus}/{len(gpus)}")
    return " | ".join(bits) if bits else "no data"


def main():
    global ROOT, LOGS, ATTACKED, EVAL

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT), help="attack_c_data root dir")
    parser.add_argument("--json", action="store_true", help="emit JSON only (no human-readable section)")
    parser.add_argument("--tail-lines", type=int, default=15)
    args = parser.parse_args()

    ROOT = pathlib.Path(args.root)
    LOGS = ROOT / "logs"
    ATTACKED = ROOT / "attacked"
    EVAL = ROOT / "eval"

    logs = _scan_logs()
    ckpts = _ckpt_counts()
    eval_counts = _eval_status()
    gpus = _gpu_status()

    headline = _format_headline(logs, ckpts, eval_counts, gpus)
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")

    payload = {
        "timestamp": timestamp,
        "root": str(ROOT),
        "headline": headline,
        "logs": logs,
        "ckpts": ckpts,
        "eval_counts": eval_counts,
        "gpus": gpus,
    }

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
        return

    print(f"[attack-c monitor] {timestamp}  {headline}")
    print()
    print("=== GPUs ===")
    for line in gpus:
        print(f"  {line}")
    print()
    print(f"=== Checkpoints under {ATTACKED} ===")
    if not ckpts:
        print("  (none yet)")
    for name, n in sorted(ckpts.items()):
        print(f"  {name}: {n} saved step(s)")
    print()
    print(f"=== Eval NPZ under {EVAL} ===")
    if not eval_counts:
        print("  (none yet)")
    for name, n in sorted(eval_counts.items()):
        print(f"  {name}: {n} episode npz(s)")
    print()
    print(f"=== Log tails (last {args.tail_lines}) ===")
    if not logs:
        print("  (no logs yet)")
    for name, info in sorted(logs.items()):
        print(f"--- {name} ---")
        flags = []
        if info["has_nan"]:
            flags.append("NaN")
        if info["has_oom"]:
            flags.append("OOM")
        if info["age_min"] is not None:
            flags.append(f"age={info['age_min']:.1f}m")
        if info["step"] is not None:
            flags.append(f"step={info['step']}")
        print(f"  flags: {','.join(flags) if flags else 'none'}")
        text = _tail(pathlib.Path(info["tail_path"]), n=args.tail_lines)
        for line in text.splitlines():
            print(f"  {line}")


if __name__ == "__main__":
    main()
