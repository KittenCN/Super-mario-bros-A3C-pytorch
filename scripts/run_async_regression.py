"""Run multiple AsyncVectorEnv trials in subprocesses with timeouts and append results to env_stability_report.jsonl

This parent script spawns a worker process (`async_trial_worker.py`) for each trial and
enforces a per-trial timeout. It captures stdout JSON from the worker and appends it
into `env_stability_report.jsonl`. If a worker times out, the parent kills it and
writes a failure entry.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

OUT = Path("env_stability_report.jsonl")
WORKER = Path(__file__).parent / "async_trial_worker.py"
REPORT_DIR = Path("reports")


def run_trial(num_envs: int, seed: int, timeout: float) -> dict:
    cmd = [sys.executable, str(WORKER), str(num_envs), str(seed)]
    start = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = proc.stdout.strip()
        err = proc.stderr.strip()
        if not out:
            # worker produced no JSON; capture stderr
            entry = {
                "timestamp": time.time(),
                "num_envs": num_envs,
                "seed": seed,
                "reset_ok": False,
                "duration": time.time() - start,
                "error": repr(err or "no-output"),
            }
            # Save diagnostic files for this failed trial
            try:
                _save_diag_for_trial(num_envs, seed)
            except Exception:
                pass
            return entry
        try:
            entry = json.loads(out)
            # ensure duration present
            if entry.get("duration") is None:
                entry["duration"] = time.time() - start
            # if worker emitted stderr, attach it for debugging (sanitize)
            if err:
                if isinstance(err, bytes):
                    try:
                        err = err.decode("utf-8", errors="replace")
                    except Exception:
                        err = repr(err)
                entry.setdefault("worker_stderr", "")
                entry["worker_stderr"] = str(err)
            return _sanitize_entry(entry)
        except Exception:
            entry = {
                "timestamp": time.time(),
                "num_envs": num_envs,
                "seed": seed,
                "reset_ok": False,
                "duration": time.time() - start,
                "error": "bad-json-output",
                "worker_stderr": str(err) if err is not None else None,
            }
            try:
                _save_diag_for_trial(num_envs, seed)
            except Exception:
                pass
            return _sanitize_entry(entry)
    except subprocess.TimeoutExpired as exc:
        # timed out -> try to kill and collect partial output
        out = getattr(exc, "stdout", None) or ""
        err = getattr(exc, "stderr", None) or ""
        if isinstance(out, (bytes, bytearray)):
            try:
                out = out.decode("utf-8", errors="replace")
            except Exception:
                out = repr(out)
        if isinstance(err, (bytes, bytearray)):
            try:
                err = err.decode("utf-8", errors="replace")
            except Exception:
                err = repr(err)
        entry = {
            "timestamp": time.time(),
            "num_envs": num_envs,
            "seed": seed,
            "reset_ok": False,
            "duration": time.time() - start,
            "error": "timeout",
            "worker_stdout_partial": out if out else None,
            "worker_stderr_partial": err if err else None,
        }
        # Save diagnostic files for this timed-out trial
        try:
            _save_diag_for_trial(num_envs, seed)
        except Exception:
            pass
        return entry


def _save_diag_for_trial(num_envs: int, seed: int) -> None:
    """Copy any /tmp/mario_env_diag_* files into a reports subdir for later inspection."""
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    dest_base = REPORT_DIR / f"async_diag_{ts}" / f"seed_{seed}_n{num_envs}"
    dest_base.mkdir(parents=True, exist_ok=True)
    pattern = "/tmp/mario_env_diag_*"
    for path in glob.glob(pattern):
        try:
            if os.path.isdir(path):
                # copy directory tree
                base_name = os.path.basename(path)
                shutil.copytree(path, dest_base / base_name, dirs_exist_ok=True)
            else:
                shutil.copy(path, dest_base)
        except Exception:
            # ignore copy errors; this is diagnostic best-effort
            pass


def _sanitize_entry(entry: dict) -> dict:
    """Ensure all values in entry are JSON serializable; convert bytes to strings and
    repr anything else non-serializable."""
    for k, v in list(entry.items()):
        if isinstance(v, (bytes, bytearray)):
            try:
                entry[k] = v.decode("utf-8", errors="replace")
            except Exception:
                entry[k] = repr(v)
        elif v is None:
            # keep None (json dumps will handle it), but ensure not a complex object
            entry[k] = None
        else:
            try:
                json.dumps(v)
            except Exception:
                entry[k] = repr(v)
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Seconds per trial before killing worker",
    )
    parser.add_argument(
        "--prewarm-count",
        type=int,
        default=0,
        help="Number of times to prewarm in parent process before async trials",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    seed_base = args.seed if args.seed is not None else random.randrange(1 << 30)
    # Ensure report dir exists
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Optional parent prewarm: perform a number of synchronous env constructions
    if args.prewarm_count and args.prewarm_count > 0:
        try:
            print(
                f"[async-reg] parent prewarm: running {args.prewarm_count} synchronous env constructions"
            )
            # local import to avoid importing heavy modules unless needed
            from src.envs import MarioVectorEnvConfig, create_vector_env

            for i in range(args.prewarm_count):
                try:
                    cseed = seed_base + i
                    pcfg = MarioVectorEnvConfig(
                        num_envs=1,
                        asynchronous=False,
                        stage_schedule=((1, 1),),
                        base_seed=cseed,
                    )
                    env = create_vector_env(pcfg)
                    env.reset(seed=cseed)
                    env.close()
                    print(
                        f"[async-reg] prewarm {i+1}/{args.prewarm_count} ok (seed={cseed})"
                    )
                except Exception as e:
                    print(f"[async-reg] prewarm {i+1} failed: {e}")
        except Exception as e:
            print(f"[async-reg] parent prewarm overall failed: {e}")
    for i in range(args.trials):
        seed = seed_base + i
        print(
            f"[async-reg] trial {i+1}/{args.trials} seed={seed} num_envs={args.num_envs}"
        )
        entry = run_trial(args.num_envs, seed, timeout=args.timeout)
        try:
            with OUT.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print("Failed to write report entry:", e)
    print("[async-reg] complete")


if __name__ == "__main__":
    main()
