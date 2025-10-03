#!/usr/bin/env bash
# 简化版训练脚本（同步向量环境）。
#
# 默认配置基于近期验证（sync num_envs=2、scripted_forward_frames=32、distance_weight=0.05），
# 可直接运行：
#   bash scripts/train_stable_sync.sh
#
# 通过环境变量覆盖常用参数，例如：
#   TOTAL_UPDATES=50 NUM_ENVS=4 RUN_NAME=my_run bash scripts/train_stable_sync.sh
#
# 支持 --dry-run 输出最终命令；使用 "--" 之后的参数将原样传给 train.py，便于追加实验配置。

set -euo pipefail

usage() {
  cat <<'EOF'
用法：
  bash scripts/train_stable_sync.sh [--dry-run] [-- ...train.py 额外参数]

核心环境变量（可选）：
  RUN_NAME                运行名称（默认 stable_sync_时间戳）
  WORLD / STAGE           关卡（默认 1-1）
  NUM_ENVS                向量环境数量（默认 2）
  TOTAL_UPDATES           训练更新次数（默认 20）
  ROLLOUT_STEPS           rollout 步长（默认 64）
  LOG_INTERVAL            日志间隔（默认 5）
  CHECKPOINT_INTERVAL     checkpoint 间隔（默认 1000）
  REWARD_DISTANCE_WEIGHT  距离奖励权重（默认 0.05）
  SCRIPTED_FORWARD_FRAMES 脚本化前进帧数（默认 32）
  SAVE_ROOT               checkpoint 根目录（默认 trained_models）
  LOG_ROOT                日志根目录（默认 tensorboard/a3c_super_mario_bros）
  MARIO_SHAPING_DEBUG     若未显式设置，默认置 1 以输出关键诊断

使用 "--" 将其后的参数直接传给 train.py，例如：
  bash scripts/train_stable_sync.sh -- --seed 123 --no-compile
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

DEFAULT_RUN_NAME="stable_sync_$(date +%Y%m%d-%H%M%S)"
RUN_NAME=${RUN_NAME:-$DEFAULT_RUN_NAME}
WORLD=${WORLD:-1}
STAGE=${STAGE:-1}
NUM_ENVS=${NUM_ENVS:-2}
TOTAL_UPDATES=${TOTAL_UPDATES:-20}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-64}
LOG_INTERVAL=${LOG_INTERVAL:-5}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-1000}
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
  --sync-env
)

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

echo "[train_stable_sync] RUN_NAME=${RUN_NAME}"
echo "[train_stable_sync] save_dir=${RUN_SAVE_DIR}"
echo "[train_stable_sync] log_dir=${RUN_LOG_DIR}"

if (( DRY_RUN )); then
  print_cmd "${FULL_CMD[@]}"
else
  print_cmd "${FULL_CMD[@]}"
  "${FULL_CMD[@]}"
fi

