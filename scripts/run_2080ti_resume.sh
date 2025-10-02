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
# 9. (新增) 细粒度采样进度参数：--rollout-progress-* 可控制前若干 update 的每步心跳打印与常规间隔
# 10.(新增) 优化 PER 内存：支持 uint8 压缩与上限自适应 (MARIO_PER_MAX_MEM_GB / MARIO_PER_COMPRESS)，脚本提供便捷变量
# 11.(新增) 可选关闭 torch.compile：通过 NO_COMPILE=1 或 COMPILE=0，便于调试 / 避免首次编译开销
# 12.(新增) 可选关闭 overlap：DISABLE_OVERLAP=1 直接去掉 --overlap-collect
# 13.(新增 2025-10-02) 距离奖励 / 动态缩放 / 脚本化前进支持：
#     REWARD_DISTANCE_WEIGHT / REWARD_SCALE_START / REWARD_SCALE_FINAL / REWARD_SCALE_ANNEAL_STEPS
#     SCRIPTED_SEQUENCE='START:8,RIGHT+B:120' 或 SCRIPTED_FORWARD_FRAMES / FORWARD_ACTION_ID / PROBE_FORWARD
#     ENABLE_RAM_X_PARSE=1 启用 RAM 回退解析（fc_emulator 缺失 x_pos 时）
# 14.(新增 2025-10-02) BOOTSTRAP=1 启动“冷启动推进”模式：若未显式指定相关参数则自动注入一组更激进的
#     初期位移与奖励塑形默认（distance_weight=0.08 + scripted START / RIGHT+B 序列 + RAM 解析），
#     用于 distance/shaping 长期为 0 的场景，减少手动调参往返。
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
  LOG_INTERVAL=             # (可选) 覆盖 --log-interval，控制 metrics 输出频率 (默认100)
  # ---- 奖励 / 前进脚本相关 (均可选) ----
  REWARD_DISTANCE_WEIGHT=0.0125   # 位移奖励权重 (默认 1/80) 建议调高以强化初期推进 (如 0.05)
  REWARD_SCALE_START=0.2          # 动态缩放初始值
  REWARD_SCALE_FINAL=0.1          # 动态缩放结束值
  REWARD_SCALE_ANNEAL_STEPS=50000 # 动态缩放线性退火步数 (env steps)
  SCRIPTED_SEQUENCE=              # 例如: 'START:8,RIGHT+B:120' (优先级高于 SCRIPTED_FORWARD_FRAMES)
  SCRIPTED_FORWARD_FRAMES=0       # 仅早期强制前进的帧数 (乘以 num_envs) 0=禁用
  FORWARD_ACTION_ID=              # 明确指定前进动作 id (extended 动作集合中 RIGHT+B 常见 id 不固定)
  PROBE_FORWARD=0                 # >0 时探测每个动作连续执行该步数, 自动选择 forward_action_id
  ENABLE_RAM_X_PARSE=1            # 1 启用 RAM 解析 x_pos 回退
  RAM_X_HIGH=0x006D               # RAM 高字节地址
  RAM_X_LOW=0x0086                # RAM 低字节地址
  BOOTSTRAP=0                     # 1=自动注入更激进的冷启动推进默认参数（若用户未显式设置相关变量）
  NO_COMPILE=0            # 1=传 --no-compile 禁用 torch.compile (或 COMPILE=0)
  DISABLE_OVERLAP=0       # 1=不添加 --overlap-collect
  AUTO_MEM=0             # 1=开启自动显存降载重试
  MEM_TARGET_GB=1.0      # 预留给系统/碎片的安全余量 (估算用, 不做硬限制)
  MAX_RETRIES=4          # 自动降载最大尝试次数
  AUTO_MEM_STREAM=1      # 自适应模式流式输出日志 (1=是 0=否)
  # ---- 进度打印粒度 (对应 train.py 新增 CLI) ----
  ROLL_PROGRESS_INTERVAL=8          # 常规阶段步进间隔
  ROLL_PROGRESS_WARMUP_UPDATES=2    # 前多少个 update 使用 warmup 间隔
  ROLL_PROGRESS_WARMUP_INTERVAL=1   # warmup 阶段间隔 (1=每步)
  # ---- PER 内存控制（脚本内转发为 MARIO_PER_*）----
  PER_MAX_MEM_GB=2.0     # 观测缓存内存目标上限 GiB (传递为 MARIO_PER_MAX_MEM_GB)
  PER_COMPRESS=1         # 1=uint8 压缩 0=float32 (传递为 MARIO_PER_COMPRESS)
  AUTO_MEM_STREAM=1      # 1=降载重试阶段实时输出日志 (stream); 0=缓冲到结束再输出

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
LOG_INTERVAL=${LOG_INTERVAL:-}
AUTO_MEM=${AUTO_MEM:-0}
MEM_TARGET_GB=${MEM_TARGET_GB:-1.0}
MAX_RETRIES=${MAX_RETRIES:-4}
AUTO_MEM_STREAM=${AUTO_MEM_STREAM:-1}
NO_COMPILE=${NO_COMPILE:-0}
COMPILE=${COMPILE:-1}
DISABLE_OVERLAP=${DISABLE_OVERLAP:-0}
BOOTSTRAP=${BOOTSTRAP:-0}

# 奖励 / 脚本化 / 探测相关可选值读取（若未提供则保持缺省或空）
REWARD_DISTANCE_WEIGHT=${REWARD_DISTANCE_WEIGHT:-}
REWARD_SCALE_START=${REWARD_SCALE_START:-}
REWARD_SCALE_FINAL=${REWARD_SCALE_FINAL:-}
REWARD_SCALE_ANNEAL_STEPS=${REWARD_SCALE_ANNEAL_STEPS:-}
SCRIPTED_SEQUENCE=${SCRIPTED_SEQUENCE:-}
SCRIPTED_FORWARD_FRAMES=${SCRIPTED_FORWARD_FRAMES:-}
FORWARD_ACTION_ID=${FORWARD_ACTION_ID:-}
PROBE_FORWARD=${PROBE_FORWARD:-}
ENABLE_RAM_X_PARSE=${ENABLE_RAM_X_PARSE:-1}
RAM_X_HIGH=${RAM_X_HIGH:-0x006D}
RAM_X_LOW=${RAM_X_LOW:-0x0086}

# 若启用 BOOTSTRAP 且用户未显式提供相关 shaping / scripted / scale 参数，则填充默认值
if [[ "$BOOTSTRAP" == "1" ]]; then
  # 仅在变量为空或未设定时注入，避免覆盖用户显式选择
  if [[ -z "${REWARD_DISTANCE_WEIGHT}" ]]; then REWARD_DISTANCE_WEIGHT=0.12; fi
  if [[ -z "${REWARD_SCALE_START}" ]]; then REWARD_SCALE_START=0.25; fi
  if [[ -z "${REWARD_SCALE_FINAL}" ]]; then REWARD_SCALE_FINAL=0.12; fi
  if [[ -z "${REWARD_SCALE_ANNEAL_STEPS}" ]]; then REWARD_SCALE_ANNEAL_STEPS=80000; fi
  # 若用户未提供脚本化动作 & 未指定脚本化帧数，则注入一段更长的前进，引导产生稳定的正向位移
  if [[ -z "${SCRIPTED_SEQUENCE}" && -z "${SCRIPTED_FORWARD_FRAMES}" ]]; then SCRIPTED_SEQUENCE='START:12,RIGHT+B:240,RIGHT+A+B:60'; fi
  if [[ -z "${PROBE_FORWARD}" ]]; then PROBE_FORWARD=24; fi
  # 强制确保 RAM 解析开启
  ENABLE_RAM_X_PARSE=1
fi

# 细粒度进度相关（若未显式设定使用默认）
ROLL_PROGRESS_INTERVAL=${ROLL_PROGRESS_INTERVAL:-8}
ROLL_PROGRESS_WARMUP_UPDATES=${ROLL_PROGRESS_WARMUP_UPDATES:-2}
ROLL_PROGRESS_WARMUP_INTERVAL=${ROLL_PROGRESS_WARMUP_INTERVAL:-1}

# PER 内存控制（允许空字符串表示不覆盖）
PER_MAX_MEM_GB=${PER_MAX_MEM_GB:-}
PER_COMPRESS=${PER_COMPRESS:-}
if [[ -n "$PER_MAX_MEM_GB" ]]; then export MARIO_PER_MAX_MEM_GB="$PER_MAX_MEM_GB"; fi
if [[ -n "$PER_COMPRESS" ]]; then export MARIO_PER_COMPRESS="$PER_COMPRESS"; fi
AUTO_MEM_STREAM=${AUTO_MEM_STREAM:-1}

SAVE_DIR="${SAVE_ROOT}/${RUN_NAME}"
mkdir -p "${SAVE_DIR}" || true

# 设备选择：如果有 CUDA 默认选 GPU
DEVICE=auto

build_cmd() {
  # -u: 无缓冲标准输出，确保前几轮 rollout/日志即时可见
  CMD=(python -u train.py \
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
  $( [[ -n "${LOG_INTERVAL}" ]] && echo --log-interval "${LOG_INTERVAL}" ) \
  --save-dir "${SAVE_DIR}" \
  $( (( DISABLE_OVERLAP == 1 )) && echo "" || echo "--overlap-collect" ) \
  --parent-prewarm \
  --heartbeat-interval 30 \
  --heartbeat-timeout 300 \
  --step-timeout 15 \
  --rollout-progress-interval "${ROLL_PROGRESS_INTERVAL}" \
  --rollout-progress-warmup-updates "${ROLL_PROGRESS_WARMUP_UPDATES}" \
  --rollout-progress-warmup-interval "${ROLL_PROGRESS_WARMUP_INTERVAL}" \
  --device "${DEVICE}" \
  --project mario-a3c-2080ti \
  --enable-tensorboard)
  if (( NO_COMPILE == 1 )) || (( COMPILE == 0 )); then
    CMD+=(--no-compile)
  fi
  # 距离奖励与动态缩放（仅当变量非空）
  if [[ -n "$REWARD_DISTANCE_WEIGHT" ]]; then CMD+=(--reward-distance-weight "$REWARD_DISTANCE_WEIGHT"); fi
  if [[ -n "$REWARD_SCALE_START" ]]; then CMD+=(--reward-scale-start "$REWARD_SCALE_START"); fi
  if [[ -n "$REWARD_SCALE_FINAL" ]]; then CMD+=(--reward-scale-final "$REWARD_SCALE_FINAL"); fi
  if [[ -n "$REWARD_SCALE_ANNEAL_STEPS" ]]; then CMD+=(--reward-scale-anneal-steps "$REWARD_SCALE_ANNEAL_STEPS"); fi
  # 脚本化前进 / 探测 / RAM 解析
  if [[ -n "$SCRIPTED_SEQUENCE" ]]; then CMD+=(--scripted-sequence "$SCRIPTED_SEQUENCE"); fi
  if [[ -n "$SCRIPTED_FORWARD_FRAMES" && "$SCRIPTED_FORWARD_FRAMES" != "0" ]]; then CMD+=(--scripted-forward-frames "$SCRIPTED_FORWARD_FRAMES"); fi
  if [[ -n "$FORWARD_ACTION_ID" ]]; then CMD+=(--forward-action-id "$FORWARD_ACTION_ID"); fi
  if [[ -n "$PROBE_FORWARD" && "$PROBE_FORWARD" != "0" ]]; then CMD+=(--probe-forward-actions "$PROBE_FORWARD"); fi
  if [[ "$ENABLE_RAM_X_PARSE" == "1" ]]; then CMD+=(--enable-ram-x-parse --ram-x-high-addr "$RAM_X_HIGH" --ram-x-low-addr "$RAM_X_LOW"); fi
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
  # 运行并捕获 OOM 关键字；支持流式或缓冲模式
  set +e
  if (( AUTO_MEM_STREAM == 1 )); then
  LOG_FILE=$(mktemp -t mario_auto_mem_${attempt}_XXXXXX.log 2>/dev/null || mktemp /tmp/mario_auto_mem_${attempt}_XXXXXX.log)
  echo "[run_2080ti_resume][auto-mem] Streaming logs (attempt ${attempt}) -> ${LOG_FILE}" >&2
    if command -v stdbuf >/dev/null 2>&1; then
      stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
      CODE=${PIPESTATUS[0]}
    else
      "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
      CODE=${PIPESTATUS[0]}
    fi
    # 仅保留最后 200 行用于错误分支输出
    if [[ -f "${LOG_FILE}" ]]; then
      OUTPUT=$(tail -n 200 "${LOG_FILE}" || true)
    else
      OUTPUT="(log file missing)"
    fi
  else
    OUTPUT=$( "${CMD[@]}" 2>&1 )
    CODE=$?
  fi
  set -e
  if (( CODE == 0 )); then
    echo "[run_2080ti_resume][auto-mem] Training started successfully with tier ${attempt}." >&2
    if (( AUTO_MEM_STREAM == 0 )); then
      echo "$OUTPUT"
    fi
    exit 0
  fi
  if echo "$OUTPUT" | grep -qi 'CUDA out of memory'; then
    echo "[run_2080ti_resume][auto-mem] Detected OOM. Falling back to next tier." >&2
    attempt=$((attempt+1))
    continue
  else
    echo "[run_2080ti_resume][auto-mem] Non-OOM failure (exit $CODE). Printing output and aborting." >&2
    if (( AUTO_MEM_STREAM == 1 )); then
      echo "---- Tail (last 200 lines) ----" >&2
      echo "$OUTPUT"
      if [[ -f "${LOG_FILE}" ]]; then
        echo "---- Full log file: ${LOG_FILE} ----" >&2
      fi
      echo "[run_2080ti_resume][hint] 若为 state_dict 键不匹配，请检查 run 目录下的 state_dict_load_issues.log" >&2
    else
      echo "$OUTPUT"
      echo "[run_2080ti_resume][hint] 查看 state_dict_load_issues.log 以获取加载差异详情 (如存在)" >&2
    fi
    exit $CODE
  fi
done

echo "[run_2080ti_resume][auto-mem] Exhausted all fallback tiers without success." >&2
exit 1
