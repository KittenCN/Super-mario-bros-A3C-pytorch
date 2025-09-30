"""Worker script to run a single AsyncVectorEnv trial and print JSON result to stdout.

Usage (invoked by run_async_regression.py):
  python scripts/async_trial_worker.py <num_envs> <seed>

This script intentionally avoids writing to files so the parent can capture stdout
and append the JSON line to the shared report file.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

try:
    num_envs = int(sys.argv[1])
    seed = int(sys.argv[2])
except Exception:
    print(json.dumps({"error": "bad-args"}))
    sys.exit(2)

# Do local import so parent process can remain lightweight
from src.envs import MarioVectorEnvConfig, create_vector_env

entry = {
    "timestamp": time.time(),
    "num_envs": num_envs,
    "seed": seed,
    "reset_ok": False,
    "duration": None,
    "construct_time": None,
    "reset_time": None,
    "error": None,
}
start = time.time()
try:
    cfg = MarioVectorEnvConfig(num_envs=num_envs, asynchronous=True, stage_schedule=((1,1),), base_seed=seed)
    # attempt construction and reset; leave relatively short so parent can time out
    env = create_vector_env(cfg)
    construct_time = time.time() - start
    entry["construct_time"] = construct_time
    t0 = time.time()
    obs, info = env.reset(seed=seed)
    reset_time = time.time() - t0
    entry["reset_time"] = reset_time
    entry["duration"] = time.time() - start
    entry["reset_ok"] = True
    try:
        env.close()
    except Exception:
        pass
except Exception as e:
    entry["duration"] = time.time() - start
    entry["error"] = repr(e)

# Print JSON to stdout for parent to capture
sys.stdout.write(json.dumps(entry, ensure_ascii=False))
sys.stdout.flush()

if not entry.get("reset_ok"):
    sys.exit(1)
sys.exit(0)
