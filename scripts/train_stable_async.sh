#!/usr/bin/env bash
# 简化版训练脚本（异步向量环境 + 预热 + overlap）。
#
# 默认配置来自近期压力测试（async num_envs=4、total_updates=50），可直接启动长程训练：
#   bash scripts/train_stable_async.sh
#
# 通过环境变量自定义常用参数，例如：
#   TOTAL_UPDATES=100 NUM_ENVS=6 RUN_NAME=async_run bash scripts/train_stable_async.sh
#
# 支持 --dry-run 查看最终命令；"--" 之后的参数会原样传给 train.py。

set -euo pipefail

usage() {
  cat <<'EOF'
用法：
  bash scripts/train_stable_async.sh [--dry-run] [-- ...train.py 额外参数]

核心环境变量（可选）：
  RUN_NAME                运行名称（默认 stable_async_时间戳）
  WORLD / STAGE           关卡（默认 1-1）
  NUM_ENVS                向量环境数量（默认 4）
  TOTAL_UPDATES           训练更新次数（默认 50）
  ROLLOUT_STEPS           rollout 步长（默认 64）
  LOG_INTERVAL            日志间隔（默认 10）
  CHECKPOINT_INTERVAL     checkpoint 间隔（默认 2000）
  REWARD_DISTANCE_WEIGHT  距离奖励权重（默认 0.05）
  SCRIPTED_FORWARD_FRAMES 脚本化前进帧数（默认 32，可用 SCRIPTED_SEQUENCE 覆盖）
  SAVE_ROOT / LOG_ROOT    输出目录（默认 trained_models / tensorboard/a3c_super_mario_bros）
  ENABLE_OVERLAP          是否开启 --overlap-collect（默认 1）
  PARENT_PREWARM          是否传入 --parent-prewarm（默认 1）
  RESUME_RELAX_MATCH      放宽恢复匹配（默认 1，设为 0 则传 --no-resume-relax-match）

示例：
  RUN_NAME=async_long TOTAL_UPDATES=200 ENABLE_OVERLAP=0 \
    bash scripts/train_stable_async.sh -- --seed 321 --no-compile
EOF
}

print_cmd() {
  if (($# == 0)); then return; fi
  printf '%q' "$1"
  shift
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

DRY_RUN=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

DEFAULT_RUN_NAME="stable_async_$(date +%Y%m%d-%H%M%S)"
RUN_NAME=${RUN_NAME:-$DEFAULT_RUN_NAME}
WORLD=${WORLD:-1}
STAGE=${STAGE:-1}
NUM_ENVS=${NUM_ENVS:-4}
TOTAL_UPDATES=${TOTAL_UPDATES:-50}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-64}
LOG_INTERVAL=${LOG_INTERVAL:-10}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-2000}
GRAD_ACCUM=${GRAD_ACCUM:-1}
FRAME_SKIP=${FRAME_SKIP:-4}
FRAME_STACK=${FRAME_STACK:-4}
ENTROPY_BETA=${ENTROPY_BETA:-0.01}
VALUE_COEF=${VALUE_COEF:-0.5}
LR=${LR:-0.00025}
PER_SAMPLE_INTERVAL=${PER_SAMPLE_INTERVAL:-4}
REWARD_DISTANCE_WEIGHT=${REWARD_DISTANCE_WEIGHT:-0.05}
SCRIPTED_SEQUENCE=${SCRIPTED_SEQUENCE:-}
SCRIPTED_FORWARD_FRAMES=${SCRIPTED_FORWARD_FRAMES:-32}
PROJECT=${PROJECT:-mario-a3c}
SAVE_ROOT=${SAVE_ROOT:-trained_models}
LOG_ROOT=${LOG_ROOT:-tensorboard/a3c_super_mario_bros}
DEVICE=${DEVICE:-auto}
ENABLE_OVERLAP=${ENABLE_OVERLAP:-1}
PARENT_PREWARM=${PARENT_PREWARM:-1}
RESUME_RELAX_MATCH=${RESUME_RELAX_MATCH:-1}
EPISODE_EVENTS_PATH=${EPISODE_EVENTS_PATH:-}

RUN_SAVE_DIR="${SAVE_ROOT%/}/${RUN_NAME}"
RUN_LOG_DIR="${LOG_ROOT%/}/${RUN_NAME}"
RUN_METRICS_PATH="${RUN_SAVE_DIR}/metrics.jsonl"

mkdir -p "$RUN_SAVE_DIR" "$RUN_LOG_DIR"

CMD=(
  python train.py
  --world "$WORLD"
  --stage "$STAGE"
  --num-envs "$NUM_ENVS"
  --total-updates "$TOTAL_UPDATES"
  --rollout-steps "$ROLLOUT_STEPS"
  --grad-accum "$GRAD_ACCUM"
  --frame-skip "$FRAME_SKIP"
  --frame-stack "$FRAME_STACK"
  --entropy-beta "$ENTROPY_BETA"
  --value-coef "$VALUE_COEF"
  --lr "$LR"
  --checkpoint-interval "$CHECKPOINT_INTERVAL"
  --log-interval "$LOG_INTERVAL"
  --per-sample-interval "$PER_SAMPLE_INTERVAL"
  --save-dir "$RUN_SAVE_DIR"
  --log-dir "$RUN_LOG_DIR"
  --metrics-path "$RUN_METRICS_PATH"
  ${EPISODE_EVENTS_PATH:+--episodes-event-path "$EPISODE_EVENTS_PATH"}
  --project "$PROJECT"
  --device "$DEVICE"
  --reward-distance-weight "$REWARD_DISTANCE_WEIGHT"
  --enable-ram-x-parse
  --async-env
  --confirm-async
)

if [[ ${ENABLE_OVERLAP} -eq 1 ]]; then
  CMD+=(--overlap-collect)
fi

if [[ ${PARENT_PREWARM} -eq 1 ]]; then
  CMD+=(--parent-prewarm)
fi

# 放宽恢复匹配开关（默认开启）
if [[ ${RESUME_RELAX_MATCH} -eq 0 ]]; then
  CMD+=(--no-resume-relax-match)
else
  CMD+=(--resume-relax-match)
fi

if [[ -n "$SCRIPTED_SEQUENCE" ]]; then
  CMD+=(--scripted-sequence "$SCRIPTED_SEQUENCE")
else
  CMD+=(--scripted-forward-frames "$SCRIPTED_FORWARD_FRAMES")
fi

if [[ ${ENABLE_TENSORBOARD:-0} == 1 ]]; then
  CMD+=(--enable-tensorboard)
fi

if [[ -n "${SEED:-}" ]]; then
  CMD+=(--seed "$SEED")
fi

ENV_VARS=(
  "MARIO_SHAPING_DEBUG=${MARIO_SHAPING_DEBUG:-1}"
  "MARIO_ALLOW_CPU_AUTO=${MARIO_ALLOW_CPU_AUTO:-1}"
  "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
)

FULL_CMD=(env "${ENV_VARS[@]}" "${CMD[@]}" "${EXTRA_ARGS[@]}")

echo "[train_stable_async] RUN_NAME=${RUN_NAME}"
echo "[train_stable_async] save_dir=${RUN_SAVE_DIR}"
echo "[train_stable_async] log_dir=${RUN_LOG_DIR}"

if (( DRY_RUN )); then
  print_cmd "${FULL_CMD[@]}"
else
  print_cmd "${FULL_CMD[@]}"
  "${FULL_CMD[@]}"
fi

