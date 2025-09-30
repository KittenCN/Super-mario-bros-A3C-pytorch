"""Run multiple AsyncVectorEnv trials in subprocesses with timeouts and append results to env_stability_report.jsonl

This parent script spawns a worker process (`async_trial_worker.py`) for each trial and
enforces a per-trial timeout. It captures stdout JSON from the worker and appends it
into `env_stability_report.jsonl`. If a worker times out, the parent kills it and
writes a failure entry.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import random

OUT = Path("env_stability_report.jsonl")
WORKER = Path(__file__).parent / "async_trial_worker.py"


def run_trial(num_envs: int, seed: int, timeout: float) -> dict:
    cmd = [sys.executable, str(WORKER), str(num_envs), str(seed)]
    start = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = proc.stdout.strip()
        if not out:
            # worker produced no JSON; capture stderr
            entry = {
                "timestamp": time.time(),
                "num_envs": num_envs,
                "seed": seed,
                "reset_ok": False,
                "duration": time.time() - start,
                "error": repr(proc.stderr.strip() or "no-output")
            }
            return entry
        try:
            entry = json.loads(out)
            # ensure duration present
            if entry.get("duration") is None:
                entry["duration"] = time.time() - start
            return entry
        except Exception:
            entry = {
                "timestamp": time.time(),
                "num_envs": num_envs,
                "seed": seed,
                "reset_ok": False,
                "duration": time.time() - start,
                "error": "bad-json-output"
            }
            return entry
    except subprocess.TimeoutExpired as exc:
        # timed out -> try to kill and collect partial output
        try:
            proc = exc
        except Exception:
            proc = None
        entry = {
            "timestamp": time.time(),
            "num_envs": num_envs,
            "seed": seed,
            "reset_ok": False,
            "duration": time.time() - start,
            "error": "timeout"
        }
        return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=30.0, help="Seconds per trial before killing worker")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    seed_base = args.seed if args.seed is not None else random.randrange(1 << 30)
    for i in range(args.trials):
        seed = seed_base + i
        print(f"[async-reg] trial {i+1}/{args.trials} seed={seed} num_envs={args.num_envs}")
        entry = run_trial(args.num_envs, seed, timeout=args.timeout)
        try:
            with OUT.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print("Failed to write report entry:", e)
    print("[async-reg] complete")


if __name__ == "__main__":
    main()
