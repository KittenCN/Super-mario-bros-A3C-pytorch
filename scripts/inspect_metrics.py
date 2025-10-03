#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""快速检查 metrics.jsonl 中推进/自救相关关键字段的脚本。

用法:
    python scripts/inspect_metrics.py --path tmp_metrics.jsonl --tail 50

输出:
    - 最新 N 条存在的字段聚合: 平均 env_positive_dx_ratio / 最近一次 milestone_count / 是否触发 secondary_script
    - 首次出现 env_distance_delta_sum>0 的 update
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="metrics.jsonl 路径")
    ap.add_argument(
        "--tail", type=int, default=100, help="分析最近 N 条拥有 update 字段的记录"
    )
    return ap.parse_args()


def main():
    args = parse_args()
    p = Path(args.path)
    if not p.exists():
        print(f"[inspect] 文件不存在: {p}")
        return 1
    updates = []
    first_positive = None
    with p.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line or line.startswith("{") is False:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "update" not in obj:
                continue
            updates.append(obj)
            if (
                first_positive is None
                and int(obj.get("env_distance_delta_sum", 0) or 0) > 0
            ):
                first_positive = obj["update"]
    if not updates:
        print("[inspect] 没有可解析的 update 指标行")
        return 0
    recent = updates[-args.tail :]
    pos_ratios = [
        float(o.get("env_positive_dx_ratio", 0.0) or 0.0)
        for o in recent
        if "env_positive_dx_ratio" in o
    ]
    avg_pos_ratio = sum(pos_ratios) / len(pos_ratios) if pos_ratios else 0.0
    last = updates[-1]
    report = {
        "records_analyzed": len(recent),
        "avg_env_positive_dx_ratio_recent": round(avg_pos_ratio, 4),
        "last_milestone_count": int(last.get("milestone_count", 0) or 0),
        "last_milestone_next": int(last.get("milestone_next", -1) or -1),
        "secondary_triggered": int(last.get("secondary_script_triggered", 0) or 0),
        "secondary_remaining": int(last.get("secondary_script_remaining", 0) or 0),
        "auto_bootstrap_triggered": int(last.get("auto_bootstrap_triggered", 0) or 0),
        "first_positive_distance_update": (
            first_positive if first_positive is not None else -1
        ),
        "latest_update": int(last.get("update", -1)),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
