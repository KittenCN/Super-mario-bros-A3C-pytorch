#!/usr/bin/env python3
"""Overlap 模式性能基准脚本.

对比 overlap=on/off 在不同 (num_envs, rollout_steps) 组合下的 env_steps_per_sec 与 updates_per_sec.
输出 CSV: benchmarks/bench_overlap_<timestamp>.csv

示例:
  python scripts/benchmark_overlap.py --num-envs 4 8 --rollout-steps 32 64 --updates 300 --warmup-fraction 0.4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List

PYTHON = "python"


def run_case(overlap: bool, num_envs: int, rollout: int, updates: int, log_interval: int, warmup_fraction: float, extra: List[str]) -> dict:
    ts = int(time.time())
    metrics_path = Path("benchmarks") / f"metrics_{'ol' if overlap else 'sync'}_{num_envs}e_{rollout}r_{ts}.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON,
        "train.py",
        "--num-envs", str(num_envs),
        "--rollout-steps", str(rollout),
        "--total-updates", str(updates),
        "--log-interval", str(log_interval),
        "--metrics-path", str(metrics_path),
        "--no-prewarm",
        "--force-sync",
    ]
    if overlap:
        cmd.append("--overlap-collect")
    cmd.extend(extra)
    start = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start
    if res.returncode != 0:
        return {"mode": "overlap" if overlap else "sync", "num_envs": num_envs, "rollout": rollout, "error": res.stderr.strip()[:300]}
    records = []
    if metrics_path.exists():
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    if not records:
        return {"mode": "overlap" if overlap else "sync", "num_envs": num_envs, "rollout": rollout, "error": "no_metrics"}
    skip = int(len(records) * warmup_fraction)
    use = records[skip:] if skip < len(records) else records
    def avg(key: str):
        vals = [r.get(key, 0.0) for r in use if key in r]
        return sum(vals) / len(vals) if vals else 0.0
    return {
        "mode": "overlap" if overlap else "sync",
        "num_envs": num_envs,
        "rollout": rollout,
        "steps_per_sec_mean": round(avg("env_steps_per_sec"), 2),
        "updates_per_sec_mean": round(avg("updates_per_sec"), 2),
        "replay_fill_final": round(use[-1].get("replay_fill_rate", 0.0), 4) if use else 0.0,
        "gpu_util_mean_window": round(use[-1].get("gpu_util_mean_window", 0.0), 2) if use else 0.0,
        "duration_sec": round(duration, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--rollout-steps", type=int, nargs="+", default=[32, 64])
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--warmup-fraction", type=float, default=0.3, help="丢弃前比例作为 warmup")
    ap.add_argument("--no-sync-only", action="store_true", help="仅运行 overlap 模式")
    ap.add_argument("--extra", type=str, nargs="*", default=[], help="透传 train.py 其它参数")
    args = ap.parse_args()
    cases = []
    for ne in args.num_envs:
        for rs in args.rollout_steps:
            for ov in ([True] if args.no_sync_only else [False, True]):
                cases.append((ov, ne, rs))
    results = []
    for i, (ov, ne, rs) in enumerate(cases, 1):
        print(f"[benchmark] ({i}/{len(cases)}) overlap={ov} num_envs={ne} rollout={rs}")
        res = run_case(ov, ne, rs, args.updates, args.log_interval, args.warmup_fraction, args.extra)
        results.append(res)
        print(f"[benchmark] result: {res}")
    out_dir = Path("benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"bench_overlap_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    import csv
    headers = ["mode", "num_envs", "rollout", "steps_per_sec_mean", "updates_per_sec_mean", "replay_fill_final", "gpu_util_mean_window", "duration_sec", "error"]
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=headers)
        writer.writeheader()
        for r in results:
            if "error" not in r:
                r["error"] = ""
            writer.writerow(r)
    print(f"[benchmark] CSV saved -> {csv_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
