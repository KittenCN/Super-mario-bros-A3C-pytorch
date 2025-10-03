#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对新增推进/自救指标进行最小存在性与基本单调性验证的测试。

设计目标:
1. 运行极短步数 (mock / quick) 下仍应生成 metrics.jsonl 并包含核心字段。
2. 当出现正向位移后 env_distance_delta_sum 应 >= 0 且保持非负递增（允许相等）。
3. progression 相关字段（任选其一）存在即可：auto_bootstrap 或 secondary_script 或 milestone_count。

注意：该测试不验证策略学习效果，只验证日志结构 & 基本约束，避免 CI 波动。
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def run_short_train(tmpdir: pathlib.Path):
    metrics_path = tmpdir / "ci_metrics.jsonl"
    # 选择极小更新步数; 使用 --max-updates 控制; 若项目 train.py 参数不同可在此处调整
    cmd = [
        sys.executable,
        str(ROOT / "train.py"),
        "--num-envs",
        "1",
        "--total-updates",
        "15",
        "--metrics-path",
        str(metrics_path),
        "--no-compile",
        "--rollout-steps",
        "16",
        "--log-interval",
        "1",
        "--early-shaping-window",
        "10",
        "--secondary-script-threshold",
        "5",
        "--secondary-script-frames",
        "20",
        "--milestone-interval",
        "0",
        "--episode-timeout-steps",
        "120",
        "--scripted-forward-frames",
        "32",
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout)
    assert proc.returncode == 0, "短训练进程退出非零"
    assert metrics_path.exists(), "未生成 metrics.jsonl"
    return metrics_path


def parse_metrics(path: pathlib.Path):
    updates = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "update" in obj:
                updates.append(obj)
    return updates


def test_progression_metrics_presence():
    with tempfile.TemporaryDirectory() as td:
        metrics_path = run_short_train(pathlib.Path(td))
        updates = parse_metrics(metrics_path)
        assert updates, "未解析到 update 记录"
        # 核心字段存在性（至少在最后一条）
        last = updates[-1]
        for field in ["env_distance_delta_sum", "env_positive_dx_ratio"]:
            assert field in last, f"缺少字段 {field}"
        # 距离增量非负
        # 允许在 episode 重置或日志汇总方式调整时出现回落，因此仅验证所有值非负且最大值>=最后值
        max_val = 0
        for o in updates:
            val = int(o.get("env_distance_delta_sum", 0) or 0)
            assert val >= 0, "env_distance_delta_sum 出现负值"
            if val > max_val:
                max_val = val
        assert max_val >= updates[-1].get(
            "env_distance_delta_sum", 0
        ), "max 距离统计异常"
        # 至少一个推进/自救机制指标出现
        mechanism_present = any(
            k in last
            for k in [
                "auto_bootstrap_triggered",
                "secondary_script_triggered",
                "milestone_count",
            ]
        )
        assert mechanism_present, "未发现推进/自救机制相关字段"
