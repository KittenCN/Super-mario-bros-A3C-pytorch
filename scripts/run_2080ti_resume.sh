#!/usr/bin/env bash
# 运行脚本：针对 11G RTX 2080 Ti + 12 vCPU + 40GB RAM 的推荐训练配置
# 支持自动从上次 checkpoint 继续训练（依赖 train.py 内置自动恢复逻辑）
#
# 特性 / Features:
# 1. 自动选择 save_dir (按 RUN_NAME)；存在匹配 checkpoint 时自动恢复
# 2. 扩大 rollout 批量 (rollout-steps=128, grad-accum=2) 提高 GPU 利用率
# 3. 适中 num-envs=10（预留 CPU 线程给主循环与监控）
# 4. 启用 fused 预处理 + overlap 采集 (--overlap-collect) 减少空转
# 5. PER 以较低频率抽样 (--per-sample-interval=4) 降低开销
# 6. 每 2k updates checkpoint，latest 快照每次都会覆盖
# 7. 可通过环境变量或参数覆盖关键超参
# 8. (新增) 自适应显存模式：AUTO_MEM=1 时按阶梯尝试多组 (NUM_ENVS,ROLLOUT,GRAD_ACCUM) 直到训练启动成功
#
# 用法 / Usage:
#   bash scripts/run_2080ti_resume.sh            # 直接启动
#   bash scripts/run_2080ti_resume.sh --dry-run  # 仅打印最终命令
#   bash scripts/run_2080ti_resume.sh --help     # 查看帮助
#   (可选) 覆盖变量: NUM_ENVS=12 ROLLOUT=96 GRAD_ACCUM=1 bash scripts/run_2080ti_resume.sh
#
# 监控建议：另开终端运行 `tensorboard --logdir tensorboard/a3c_super_mario_bros`，并用 nvidia-smi 或 nvtop 观察 GPU 利用率。
#
# 关键参数背后的考虑 / Rationale:
# - NUM_ENVS=10: 12 vCPU 中保留 ~2 核用于主线程 + 杂项；同步向量环境串行 step 但仍受 Python 调度影响。
# - ROLLOUT=128 & GRAD_ACCUM=2: 有效反向长度 256*10 = 2560 时间步/更新，提升计算密度但仍在 11GB 内可控。
# - --per-sample-interval=4: PER 只在每 4 次更新后做一次采样/优先级更新，减少频繁 host<->device 拷贝。
# - --overlap-collect: 双缓冲线程在学习阶段后台采集下一批，加速总体吞吐。
#
# 失败排查 / Troubleshooting:
# - 若显存 OOM: 降低 ROLLOUT 或去掉 GRAD_ACCUM / 减少 NUM_ENVS
# - 若 CPU 饱和: 降低 NUM_ENVS 或提高 ROLLOUT (增大单批时间占比)
# - 若恢复错误: 确认 world / stage / 模型结构未变；必要时删除冲突的旧 checkpoint JSON
#
set -euo pipefail

show_help() {
  sed -n '1,/^set -euo pipefail/p' "$0" | sed '$d'
  cat <<'EOF'
自定义覆盖环境变量 (override via env vars):
  RUN_NAME=run01          # 存储目录 trained_models/<RUN_NAME>
  WORLD=1                 # 初始 world
  STAGE=1                 # 初始 stage
  ACTION_TYPE=complex     # 动作空间: right|simple|complex
  NUM_ENVS=10             # 环境数量 (建议 8~12)
  ROLLOUT=128             # rollout-steps
  GRAD_ACCUM=2            # 梯度累积步数
  HIDDEN=512              # 隐藏层大小
  FRAME_STACK=4           # 帧堆叠
  FRAME_SKIP=4            # 帧跳
  ENTROPY=0.01            # 熵系数
  VALUE_COEF=0.5          # 价值系数
  LR=0.00025              # 学习率
  CHECKPOINT_INT=2000     # checkpoint 间隔 (updates)
  EVAL_INT=5000           # 评估间隔 (暂未实现可改)
  TOTAL_UPDATES=100000    # 总更新数
  PER_INTERVAL=4          # PER 抽样间隔
  SAVE_ROOT=trained_models # 根保存目录
  LOG_DIR=tensorboard/a3c_super_mario_bros
  AUTO_MEM=0             # 1=开启自动显存降载重试
  MEM_TARGET_GB=1.0      # 预留给系统/碎片的安全余量 (估算用, 不做硬限制)
  MAX_RETRIES=4          # 自动降载最大尝试次数

示例 / Example:
  NUM_ENVS=12 ROLLOUT=96 GRAD_ACCUM=1 PER_INTERVAL=2 bash scripts/run_2080ti_resume.sh

EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  show_help
  exit 0
fi

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

# 默认值 (允许外部覆盖)
RUN_NAME=${RUN_NAME:-run01}
WORLD=${WORLD:-1}
STAGE=${STAGE:-1}
ACTION_TYPE=${ACTION_TYPE:-complex}
NUM_ENVS=${NUM_ENVS:-10}
ROLLOUT=${ROLLOUT:-128}
GRAD_ACCUM=${GRAD_ACCUM:-2}
HIDDEN=${HIDDEN:-512}
FRAME_STACK=${FRAME_STACK:-4}
FRAME_SKIP=${FRAME_SKIP:-4}
ENTROPY=${ENTROPY:-0.01}
VALUE_COEF=${VALUE_COEF:-0.5}
LR=${LR:-0.00025}
CHECKPOINT_INT=${CHECKPOINT_INT:-2000}
EVAL_INT=${EVAL_INT:-5000}
TOTAL_UPDATES=${TOTAL_UPDATES:-100000}
PER_INTERVAL=${PER_INTERVAL:-4}
SAVE_ROOT=${SAVE_ROOT:-trained_models}
LOG_DIR=${LOG_DIR:-tensorboard/a3c_super_mario_bros}
AUTO_MEM=${AUTO_MEM:-0}
MEM_TARGET_GB=${MEM_TARGET_GB:-1.0}
MAX_RETRIES=${MAX_RETRIES:-4}

SAVE_DIR="${SAVE_ROOT}/${RUN_NAME}"
mkdir -p "${SAVE_DIR}" || true

# 设备选择：如果有 CUDA 默认选 GPU
DEVICE=auto

build_cmd() {
  CMD=(python train.py \
  --world "${WORLD}" \
  --stage "${STAGE}" \
  --action-type "${ACTION_TYPE}" \
  --num-envs "${NUM_ENVS}" \
  --frame-stack "${FRAME_STACK}" \
  --frame-skip "${FRAME_SKIP}" \
  --total-updates "${TOTAL_UPDATES}" \
  --rollout-steps "${ROLLOUT}" \
  --hidden-size "${HIDDEN}" \
  --entropy-beta "${ENTROPY}" \
  --value-coef "${VALUE_COEF}" \
  --lr "${LR}" \
  --grad-accum "${GRAD_ACCUM}" \
  --checkpoint-interval "${CHECKPOINT_INT}" \
  --eval-interval "${EVAL_INT}" \
  --per \
  --per-sample-interval "${PER_INTERVAL}" \
  --log-dir "${LOG_DIR}" \
  --save-dir "${SAVE_DIR}" \
  --overlap-collect \
  --parent-prewarm \
  --heartbeat-interval 30 \
  --heartbeat-timeout 300 \
  --step-timeout 15 \
  --device "${DEVICE}" \
  --project mario-a3c-2080ti \
  --enable-tensorboard)
}

build_cmd

# 自动恢复：不显式传 --resume，交由内部逻辑扫描 save_dir + 递归匹配

echo "[run_2080ti_resume] Launch command (initial):"
printf ' %q' "${CMD[@]}"; echo

if [[ ${DRY_RUN} -eq 1 ]]; then
  echo "[run_2080ti_resume] Dry run complete."
  exit 0
fi

# 提示已有的 checkpoint（若存在）
checkpoint_glob="${SAVE_DIR}/a3c_world${WORLD}_stage${STAGE}_*.pt"
if compgen -G "${checkpoint_glob}" > /dev/null; then
  # 使用 compgen + wc -l 统计，避免 ls 在无匹配时报错导致重复 0 行
  EXISTING=$(compgen -G "${checkpoint_glob}" | wc -l | tr -d '[:space:]')
else
  EXISTING=0
fi
if (( EXISTING > 0 )); then
  echo "[run_2080ti_resume] Detected ${EXISTING} existing checkpoint file(s) in ${SAVE_DIR} (auto-resume will engage)."
fi


if (( AUTO_MEM == 0 )); then
  exec "${CMD[@]}"
fi

echo "[run_2080ti_resume][auto-mem] Enabled. Will attempt up to ${MAX_RETRIES} fallback tiers on OOM."

# 阶梯参数 (按优先顺序) : 格式 NUM_ENVS,ROLLOUT,GRAD_ACCUM
TIERS=( \
  "${NUM_ENVS},${ROLLOUT},${GRAD_ACCUM}" \
  "${NUM_ENVS},$((ROLLOUT/2)),${GRAD_ACCUM}" \
  "$((NUM_ENVS-2)),${ROLLOUT},${GRAD_ACCUM}" \
  "$((NUM_ENVS-2)),$((ROLLOUT/2)),1" \
  "$((NUM_ENVS-4)),$((ROLLOUT/2)),1" \
  "$((NUM_ENVS-4)),$((ROLLOUT/3)),1" \
)

attempt=0
for tier in "${TIERS[@]}"; do
  if (( attempt >= MAX_RETRIES )); then
    echo "[run_2080ti_resume][auto-mem] Reached max retries (${MAX_RETRIES}). Aborting." >&2
    exit 1
  fi
  IFS=',' read -r T_NUM T_ROLLOUT T_ACCUM <<<"$tier"
  # 合法性修正
  (( T_NUM < 2 )) && T_NUM=2
  (( T_ROLLOUT < 16 )) && T_ROLLOUT=16
  (( T_ACCUM < 1 )) && T_ACCUM=1
  NUM_ENVS=$T_NUM ROLLOUT=$T_ROLLOUT GRAD_ACCUM=$T_ACCUM build_cmd
  echo "[run_2080ti_resume][auto-mem] Attempt ${attempt} -> num_envs=${T_NUM} rollout=${T_ROLLOUT} grad_accum=${T_ACCUM}" 
  printf '  %q' "${CMD[@]}"; echo
  # 运行并捕获 OOM 关键字
  set +e
  OUTPUT=$( "${CMD[@]}" 2>&1 )
  CODE=$?
  set -e
  if (( CODE == 0 )); then
    echo "[run_2080ti_resume][auto-mem] Training started successfully with tier ${attempt}." >&2
    echo "$OUTPUT"
    exit 0
  fi
  if echo "$OUTPUT" | grep -qi 'CUDA out of memory'; then
    echo "[run_2080ti_resume][auto-mem] Detected OOM. Falling back to next tier." >&2
    attempt=$((attempt+1))
    continue
  else
    echo "[run_2080ti_resume][auto-mem] Non-OOM failure (exit $CODE). Printing output and aborting." >&2
    echo "$OUTPUT"
    exit $CODE
  fi
done

echo "[run_2080ti_resume][auto-mem] Exhausted all fallback tiers without success." >&2
exit 1
