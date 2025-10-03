import json
import os
import pathlib
import subprocess
import time

# 该 smoke 测试以最小规模运行训练脚本若干步，验证出现非零 shaping / distance。
# 假设 train.py 支持通过 RUN_NAME / ACTION_TYPE / SCRIPTED_SEQUENCE / ENABLE_RAM_X_PARSE 等环境变量脚本映射。
# 若未来动作脚本接口变化，可在此同步调整。

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_2080ti_resume.sh"


def run_short():
    env = os.environ.copy()
    env.update(
        {
            "RUN_NAME": "smoke_shaping",
            "ACTION_TYPE": "extended",
            "BOOTSTRAP": "1",  # 自动注入脚本 & distance 权重
            "TOTAL_UPDATES": "3",  # 多给一次机会产生日志
            "CHECKPOINT_INT": "999999",  # 避免频繁落盘
            "LOG_DIR": "tensorboard/a3c_super_mario_bros",
            "LOG_INTERVAL": "1",  # 每个 update 都写 metrics 行
            "NUM_ENVS": "1",  # 加快启动
            "ROLLOUT": "32",  # 减少单 update 步数
            "GRAD_ACCUM": "1",
            # 显式脚本化序列（即使 BOOTSTRAP 会注入，也覆盖长度更短）
            "SCRIPTED_SEQUENCE": "START:8,RIGHT+B:120",
        }
    )
    # 使用 --dry-run 获取命令确认（不严格断言），然后真实执行
    dry = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"], env=env, capture_output=True, text=True
    )
    if dry.returncode != 0:
        raise RuntimeError("dry-run failed: " + dry.stderr)
    # 真正运行（限制时间）
    proc = subprocess.Popen(
        ["bash", str(SCRIPT)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    start = time.time()
    lines = []
    timeout = 120  # 秒
    dx_detected = False
    while time.time() - start < timeout:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                break
            continue
        lines.append(line)
        # 检测首次位移
        if "[reward][dx] first_positive_dx" in line:
            dx_detected = True
            # 不立即结束，等待最多 2s 看是否写出 metrics
            wait_until = time.time() + 2.0
            while time.time() < wait_until and proc.poll() is None:
                try:
                    extra = proc.stdout.readline()
                except Exception:
                    break
                if extra:
                    lines.append(extra)
                time.sleep(0.05)
            break
    proc.terminate()
    # 聚合已有 tensorboard 下 metrics.jsonl 文件（最新目录）
    tb_root = PROJECT_ROOT / "tensorboard" / "a3c_super_mario_bros"
    if not tb_root.exists():
        raise AssertionError("tensorboard root not created")
    # 找最大修改时间的子目录
    # 若当前目录下不存在 metrics.jsonl，尝试在所有子目录中寻找最近写入的一个
    candidates = []
    for p in tb_root.iterdir():
        if not p.is_dir():
            continue
        mf = p / "metrics.jsonl"
        if mf.exists():
            candidates.append(mf)
    if not candidates:
        raise AssertionError(
            "metrics.jsonl missing under any run dir in " + str(tb_root)
        )
    metrics_file = max(candidates, key=lambda f: f.stat().st_mtime)
    distance_nonzero = dx_detected  # 如果已在 stdout 捕获 dx，则直接视为成功
    shaping_nonzero = False
    if not distance_nonzero or not shaping_nonzero:
        with metrics_file.open() as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if not distance_nonzero and row.get("env_distance_delta_sum", 0) > 0:
                    distance_nonzero = True
                if not shaping_nonzero and row.get("env_shaping_raw_sum", 0) > 0:
                    shaping_nonzero = True
                if distance_nonzero and shaping_nonzero:
                    break
    # 断言至少有一项为真（distance 或 shaping 原始值）
    if not (distance_nonzero or shaping_nonzero):
        tail = "".join(lines[-50:])
        raise AssertionError(
            "No non-zero distance or shaping observed in smoke run. Stdout tail:\n"
            + tail
        )
    return distance_nonzero, shaping_nonzero


def test_shaping_smoke():
    distance, shaping = run_short()
    # 若仅 distance 非零仍接受，但打印提示
    print(f"[test_shaping_smoke] distance_nonzero={distance} shaping_nonzero={shaping}")
