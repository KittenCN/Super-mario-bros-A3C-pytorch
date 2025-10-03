#!/usr/bin/env python3
"""metrics_summary.py

读取 metrics.jsonl (或指定文件) 生成：
1. 末尾若干条聚合表（CSV）
2. 全量时间序列导出（可选 Parquet，需 pyarrow；否则 CSV）
3. 简洁控制台摘要（steps/s、updates/s、avg_return、replay_fill_rate、gpu_util_mean_window）

用法:
  python scripts/metrics_summary.py --path run_dir/metrics.jsonl --out-dir summaries --tail 200

可选:
  --parquet 导出 Parquet
  --fields 指定逗号分隔聚合字段（默认 auto）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

DEFAULT_FIELDS = [
    "update",
    "global_step",
    "avg_return",
    "recent_return_p50",
    "recent_return_p90",
    "recent_return_p99",
    "env_steps_per_sec",
    "updates_per_sec",
    "replay_fill_rate",
    "replay_last_unique_ratio",
    "replay_avg_unique_ratio",
    "replay_priority_mean",
    "replay_priority_p90",
    "gpu_util_mean_window",
    "loss_total",
    "loss_value",
    "loss_policy",
]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            records.append(obj)
        except Exception:
            continue
    return records


def summarise(records: List[Dict[str, Any]], fields: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not records:
        return out
    last = records[-1]
    for f in fields:
        vals = [r.get(f) for r in records if isinstance(r.get(f), (int, float))]
        if not vals:
            continue
        out[f + "_last"] = float(vals[-1])
        out[f + "_mean"] = float(sum(vals) / len(vals))
    # 额外关键指标
    out["updates_total"] = float(last.get("update", 0))
    out["global_step_last"] = float(last.get("global_step", 0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", type=Path, required=True, help="metrics.jsonl 路径")
    ap.add_argument("--out-dir", type=Path, default=Path("summaries"))
    ap.add_argument("--tail", type=int, default=0, help="仅分析最后 N 条 (0=全量)")
    ap.add_argument(
        "--fields", type=str, default="", help="逗号分隔聚合字段; 留空用默认列表"
    )
    ap.add_argument(
        "--parquet", action="store_true", help="导出 Parquet (需要 pandas+pyarrow)"
    )
    args = ap.parse_args()

    recs = load_jsonl(args.path)
    if args.tail > 0:
        recs = recs[-args.tail :]

    fields = (
        [f.strip() for f in args.fields.split(",") if f.strip()]
        if args.fields
        else DEFAULT_FIELDS
    )
    summary = summarise(recs, fields)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 写入 summary JSON
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 导出表格
    if pd is not None:
        try:
            import pandas as _pd

            df = _pd.DataFrame(recs)
            if args.parquet:
                df.to_parquet(args.out_dir / "metrics.parquet", index=False)
            df.to_csv(args.out_dir / "metrics.csv", index=False)
        except Exception as e:  # pragma: no cover
            print(f"[metrics_summary][warn] export failed: {e}")
    else:
        # 简单 CSV 手写
        if recs:
            cols = list({k for r in recs for k in r.keys()})
            lines = [",".join(cols)]
            for r in recs:
                row = [str(r.get(c, "")) for c in cols]
                lines.append(",".join(row))
            (args.out_dir / "metrics.csv").write_text(
                "\n".join(lines), encoding="utf-8"
            )

    # 控制台摘要
    key_print = [
        "avg_return_last",
        "recent_return_p90_last",
        "env_steps_per_sec_mean",
        "updates_per_sec_mean",
        "replay_fill_rate_last",
        "replay_avg_unique_ratio_last",
        "gpu_util_mean_window_last",
    ]
    print("[metrics_summary] summary (subset):")
    for k in key_print:
        if k in summary:
            print(f"  {k}: {summary[k]:.4f}")
    print(f"[metrics_summary] output dir -> {args.out_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
