"""测试 episode 结束统计字段写入 metrics.jsonl.

策略: 运行一个短程脚本化前进 + 超时截断配置, 确保产生至少 1 个 episode.
断言: metrics.jsonl 至少一行包含 episode_length_mean 与 episode_end_reason_ 前缀字段之一。
"""
from __future__ import annotations
import json
import pathlib
import subprocess
import sys


def run_short(tmp: pathlib.Path) -> pathlib.Path:
    metrics_path = tmp / "metrics.jsonl"
    cmd = [
        sys.executable,
        "train.py",
        "--num-envs","1",
        "--total-updates","10",
        "--rollout-steps","8",
        "--world","1","--stage","1",
        "--scripted-forward-frames","120",
        "--episode-timeout-steps","60",
        "--stagnation-limit","50",
        "--log-interval","2",
        "--checkpoint-interval","0",
        "--eval-interval","0",
        "--metrics-path", str(metrics_path),
    ]
    subprocess.check_call(cmd, cwd=pathlib.Path(__file__).resolve().parent.parent)
    return metrics_path


def test_episode_metrics_presence(tmp_path: pathlib.Path):
    metrics_path = run_short(tmp_path)
    assert metrics_path.exists(), "metrics.jsonl 未生成"
    has_length = False
    has_reason = False
    has_event_line = False
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if "episode_length_mean" in obj and obj.get("episodes_completed",0) >= 0:
            has_length = has_length or obj["episode_length_mean"] >= 0.0
        if any(k.startswith("episode_end_reason_") for k in obj.keys()):
            has_reason = True
        if obj.get("event") == "episode_end":
            has_event_line = True
        if has_length and has_reason:
            break
    assert has_length, "未发现 episode_length_mean 字段"
    # 可能极短运行内尚无完成 episode; reason 字段允许缺失但理想存在
    # 若 episodes_completed>0 则必须出现至少一个 reason 直方图字段
    if any("episodes_completed" in json.loads(l) and json.loads(l)["episodes_completed"]>0 for l in metrics_path.read_text(encoding="utf-8").splitlines() if l.strip() and '"event"' not in l):
        assert has_reason, "存在完成 episode 但未写入 episode_end_reason_* 字段"
        assert has_event_line, "存在完成 episode 但未记录 episode_end 事件行"
