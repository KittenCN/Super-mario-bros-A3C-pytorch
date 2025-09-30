"""Run a short synchronous vector-env regression and append results to env_stability_report.jsonl

Usage: python scripts/run_sync_regression.py --trials 5 --num-envs 4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import random

from src.envs import MarioEnvConfig, MarioVectorEnvConfig, create_vector_env

OUT = Path("env_stability_report.jsonl")


def run_trial(num_envs: int, seed: int) -> dict:
    cfg = MarioVectorEnvConfig(
        num_envs=num_envs,
        asynchronous=False,
        stage_schedule=((1, 1),),
        base_seed=seed,
    )
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
    return entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    seed_base = args.seed if args.seed is not None else random.randrange(1 << 30)
    results = []
    for i in range(args.trials):
        seed = seed_base + i
        print(f"[reg] trial {i+1}/{args.trials} seed={seed} num_envs={args.num_envs}")
        entry = run_trial(args.num_envs, seed)
        results.append(entry)
        try:
            with OUT.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print("Failed to write report entry:", e)
    print("[reg] complete")


if __name__ == "__main__":
    main()
