#!/usr/bin/env bash
# 自动分阶段训练脚本：Phase1 -> Phase2 -> Phase3
# 目标：按推荐策略依次运行三个训练阶段，并在阶段间复制最新 checkpoint 以实现连续训练参数迁移。
# 可重入：再次运行会跳过已完成阶段（检测 .phase_done 标记），可通过 FORCE_PHASE<N>=1 强制重跑。
# 支持 DRY_RUN=1 仅输出将执行的命令。
#
# 阶段概览（默认值，可通过环境变量覆盖）：
# Phase1 (Bootstrap):  强脚本化 + 较高距离 shaping，快速产生正向位移
# Phase2 (Main):       降低 shaping 权重，策略逐渐主导
# Phase3 (Refine):     进一步衰减或关闭距离奖励，偏真实回报
#
# 依赖：使用现有 scripts/run_2080ti_resume.sh 作为底层执行器。
#
# 关键环境变量 (全部可选)：
#  WORLD / STAGE                     关卡 (默认 1 / 1)
#  BASE_SAVE_ROOT=trained_models     保存根目录
#  BASE_LOG_DIR=tensorboard/a3c_super_mario_bros  日志根目录
#  NUM_ENVS=8 ROLLOUT=64 GRAD_ACCUM=1  基线批量配置（各阶段可继承）
#  PHASE1_RUN=exp_boot_phase1  PHASE1_UPDATES=2000  PHASE1_DISTANCE_WEIGHT=0.08  PHASE1_SCALE_START=0.2  PHASE1_SCALE_FINAL=0.1
#  PHASE2_RUN=exp_main_phase2  PHASE2_UPDATES=30000 PHASE2_DISTANCE_WEIGHT=0.04  PHASE2_SCALE_START=0.15 PHASE2_SCALE_FINAL=0.08
#  PHASE3_RUN=exp_refine_phase3 PHASE3_UPDATES=50000 PHASE3_DISTANCE_WEIGHT=0.02 PHASE3_SCALE_START=0.12 PHASE3_SCALE_FINAL=0.06
#  PHASE3_DISABLE_DISTANCE=0 (1=关闭距离奖励: distance_weight=0)
#  LOG_INTERVAL=50 CHECKPOINT_INT=2000 PER_SAMPLE_INTERVAL=4
#  ENTROPY_BETA=0.01 VALUE_COEF=0.5 LR=0.00025
#  OVERLAP=1   PER=1
#  AUTO_MEM=1  发生 OOM 时底层脚本自动降载
#  DRY_RUN=1   仅打印即将执行的命令
#  SKIP_PHASE1 / SKIP_PHASE2 / SKIP_PHASE3 =1 跳过指定阶段
#  FORCE_PHASE1 / FORCE_PHASE2 / FORCE_PHASE3 =1 即使有 .phase_done 仍重新执行
#
# 结果判断：阶段完成后写入 <save_dir>/.phase_done 以及记录‘completed_updates=<target>’。
#
# 使用示例：
#   bash scripts/auto_phase_training.sh                  # 正常执行
#   DRY_RUN=1 bash scripts/auto_phase_training.sh        # 预览命令
#   FORCE_PHASE2=1 bash scripts/auto_phase_training.sh   # 只强制重跑第二阶段（前提第一阶段已完成）
#
set -euo pipefail

usage() {
  cat <<'EOF'
自动分阶段训练脚本
Usage:
  bash scripts/auto_phase_training.sh                # 顺序执行 3 个阶段
  DRY_RUN=1 bash scripts/auto_phase_training.sh      # 仅打印每阶段最终命令 (不创建环境)
  SKIP_PHASE2=1 SKIP_PHASE3=1 bash scripts/auto_phase_training.sh  # 只跑 Phase1

可通过 环境变量 或 传参 KEY=VALUE 覆盖（脚本会解析 KEY=VALUE 形式参数并 export）。
支持的额外自定义：PHASE1_EXTRA / PHASE2_EXTRA / PHASE3_EXTRA 可附加自定义 cli 片段 (例如 "NO_COMPILE=1")。
强制重跑：FORCE_PHASE1=1 等；跳过：SKIP_PHASE2=1 等。
帮助：-h / --help。
EOF
}

# 解析 KEY=VALUE 形式的参数（兼容用户直接传 DRY_RUN=1 等）
for arg in "$@"; do
  case "$arg" in
    -h|--help) usage; exit 0;;
    *=*) export "$arg";;
    *) echo "[auto-phases][warn] 未识别参数: $arg" >&2;;
  esac
done

log() { echo "[auto-phases] $*"; }
warn() { echo "[auto-phases][warn] $*" >&2; }
err() { echo "[auto-phases][error] $*" >&2; exit 1; }

WORLD=${WORLD:-1}
STAGE=${STAGE:-1}
BASE_SAVE_ROOT=${BASE_SAVE_ROOT:-trained_models}
BASE_LOG_DIR=${BASE_LOG_DIR:-tensorboard/a3c_super_mario_bros}

# 通用运行基线
NUM_ENVS=${NUM_ENVS:-8}
ROLLOUT=${ROLLOUT:-64}
GRAD_ACCUM=${GRAD_ACCUM:-1}
LOG_INTERVAL=${LOG_INTERVAL:-50}
CHECKPOINT_INT=${CHECKPOINT_INT:-2000}
PER_SAMPLE_INTERVAL=${PER_SAMPLE_INTERVAL:-4}
ENTROPY_BETA=${ENTROPY_BETA:-0.01}
VALUE_COEF=${VALUE_COEF:-0.5}
LR=${LR:-0.00025}
PER=${PER:-1}
OVERLAP=${OVERLAP:-1}
AUTO_MEM=${AUTO_MEM:-1}
ACTION_TYPE=${ACTION_TYPE:-extended}
DEVICE=${DEVICE:-auto}

# Phase configs
PHASE1_RUN=${PHASE1_RUN:-exp_boot_phase1}
PHASE1_UPDATES=${PHASE1_UPDATES:-2000}
PHASE1_DISTANCE_WEIGHT=${PHASE1_DISTANCE_WEIGHT:-0.08}
PHASE1_SCALE_START=${PHASE1_SCALE_START:-0.2}
PHASE1_SCALE_FINAL=${PHASE1_SCALE_FINAL:-0.1}
PHASE1_SCALE_ANNEAL_STEPS=${PHASE1_SCALE_ANNEAL_STEPS:-50000}
PHASE1_SCRIPTED_SEQUENCE=${PHASE1_SCRIPTED_SEQUENCE:-START:8,RIGHT+B:180}

PHASE2_RUN=${PHASE2_RUN:-exp_main_phase2}
PHASE2_UPDATES=${PHASE2_UPDATES:-30000}
PHASE2_DISTANCE_WEIGHT=${PHASE2_DISTANCE_WEIGHT:-0.04}
PHASE2_SCALE_START=${PHASE2_SCALE_START:-0.15}
PHASE2_SCALE_FINAL=${PHASE2_SCALE_FINAL:-0.08}
PHASE2_SCALE_ANNEAL_STEPS=${PHASE2_SCALE_ANNEAL_STEPS:-50000}

PHASE3_RUN=${PHASE3_RUN:-exp_refine_phase3}
PHASE3_UPDATES=${PHASE3_UPDATES:-50000}
PHASE3_DISTANCE_WEIGHT=${PHASE3_DISTANCE_WEIGHT:-0.02}
PHASE3_SCALE_START=${PHASE3_SCALE_START:-0.12}
PHASE3_SCALE_FINAL=${PHASE3_SCALE_FINAL:-0.06}
PHASE3_SCALE_ANNEAL_STEPS=${PHASE3_SCALE_ANNEAL_STEPS:-30000}
PHASE3_DISABLE_DISTANCE=${PHASE3_DISABLE_DISTANCE:-0}

# Control flags
DRY_RUN=${DRY_RUN:-0}
SKIP_PHASE1=${SKIP_PHASE1:-0}
SKIP_PHASE2=${SKIP_PHASE2:-0}
SKIP_PHASE3=${SKIP_PHASE3:-0}
FORCE_PHASE1=${FORCE_PHASE1:-0}
FORCE_PHASE2=${FORCE_PHASE2:-0}
FORCE_PHASE3=${FORCE_PHASE3:-0}

RUN_SCRIPT="scripts/run_2080ti_resume.sh"
if [[ ! -x ${RUN_SCRIPT} ]]; then
  warn "${RUN_SCRIPT} 不可执行，尝试 chmod +x"
  chmod +x "${RUN_SCRIPT}" 2>/dev/null || true
fi
[[ -x ${RUN_SCRIPT} ]] || err "底层执行脚本 ${RUN_SCRIPT} 不存在或不可执行"

# 复制 checkpoint 到下一阶段目录（如果下一阶段目录不存在或为空）
# 参数: src_run dst_run target_updates
copy_checkpoint() {
  local src_run=$1 dst_run=$2
  local src_dir="${BASE_SAVE_ROOT}/${src_run}"
  local dst_dir="${BASE_SAVE_ROOT}/${dst_run}"
  [[ -d "$src_dir" ]] || { warn "源目录 $src_dir 不存在，跳过 checkpoint 复制"; return 0; }
  mkdir -p "$dst_dir"
  # 查找 latest 或时间最新 .pt
  local latest_pt
  if [[ -f ${src_dir}/a3c_world${WORLD}_stage${STAGE}_latest.pt ]]; then
    latest_pt=${src_dir}/a3c_world${WORLD}_stage${STAGE}_latest.pt
  else
    latest_pt=$(ls -1t ${src_dir}/a3c_world${WORLD}_stage${STAGE}_*.pt 2>/dev/null | head -n1 || true)
  fi
  [[ -n "$latest_pt" ]] || { warn "未找到 Phase 源 checkpoint (*.pt) 于 ${src_dir}"; return 0; }
  local meta_json="${latest_pt%.pt}.json"
  log "复制 checkpoint -> ${dst_dir}"; cp -f "$latest_pt" "$dst_dir/" || true
  if [[ -f "$meta_json" ]]; then cp -f "$meta_json" "$dst_dir/"; fi
}

# 运行单阶段；参数：phase_name run_name total_updates distance_weight scale_start scale_final anneal scripted_sequence bootstrap_flag disable_distance
run_phase() {
  local pname=$1 run_name=$2 total_updates=$3 dist_w=$4 scale_start=$5 scale_final=$6 anneal=$7 script_seq=$8 bootstrap=$9 disable_distance=${10}
  local save_dir="${BASE_SAVE_ROOT}/${run_name}"; mkdir -p "$save_dir"
  local done_flag="${save_dir}/.phase_done"
  if [[ -f "$done_flag" && ${!11} -eq 0 ]]; then
    log "${pname} 已完成，跳过 (标记存在 ${done_flag})"
    return 0
  fi
  if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "--- DRY RUN ${pname} ---"
  else
    log "开始 ${pname}: run=${run_name} updates=${total_updates} distance_weight=${dist_w}"
  fi
  local bootstrap_env=0
  if [[ $bootstrap -eq 1 ]]; then bootstrap_env=1; fi
  local dw=$dist_w
  if [[ $disable_distance -eq 1 ]]; then dw=0; fi

  # 通过 env 注入变量后执行底层脚本；使用单个 bash -c 保证 DRY_RUN 输出统一
  local env_export=(
    "RUN_NAME=$run_name" "WORLD=$WORLD" "STAGE=$STAGE" "ACTION_TYPE=$ACTION_TYPE" \
    "NUM_ENVS=$NUM_ENVS" "ROLLOUT=$ROLLOUT" "GRAD_ACCUM=$GRAD_ACCUM" \
    "TOTAL_UPDATES=$total_updates" "LOG_INTERVAL=$LOG_INTERVAL" "CHECKPOINT_INT=$CHECKPOINT_INT" \
    "PER_SAMPLE_INTERVAL=$PER_SAMPLE_INTERVAL" "ENTROPY=$ENTROPY_BETA" "VALUE_COEF=$VALUE_COEF" "LR=$LR" \
    "PER_INTERVAL=$PER_SAMPLE_INTERVAL" \
    "REWARD_DISTANCE_WEIGHT=$dw" "REWARD_SCALE_START=$scale_start" "REWARD_SCALE_FINAL=$scale_final" "REWARD_SCALE_ANNEAL_STEPS=$anneal" \
    "ENABLE_RAM_X_PARSE=1" \
    "AUTO_MEM=$AUTO_MEM" "OVERLAP_COLLECT=$OVERLAP" "PER=$PER" \
    "LOG_DIR=$BASE_LOG_DIR" \
    "BOOTSTRAP=$bootstrap_env"
  )
  if [[ -n "$script_seq" ]]; then env_export+=("SCRIPTED_SEQUENCE=$script_seq"); fi
  local cmd=(env "${env_export[@]}" bash "$RUN_SCRIPT")

  # 附加用户自定义额外参数 (作为环境变量对底层脚本生效或直接添加 CLI 需要自定义脚本支持)
  case "$pname" in
    Phase1) [[ -n "${PHASE1_EXTRA:-}" ]] && cmd=(env "${env_export[@]}" ${PHASE1_EXTRA} bash "$RUN_SCRIPT");;
    Phase2) [[ -n "${PHASE2_EXTRA:-}" ]] && cmd=(env "${env_export[@]}" ${PHASE2_EXTRA} bash "$RUN_SCRIPT");;
    Phase3) [[ -n "${PHASE3_EXTRA:-}" ]] && cmd=(env "${env_export[@]}" ${PHASE3_EXTRA} bash "$RUN_SCRIPT");;
  esac

  if [[ -n "$script_seq" ]]; then
    cmd+=( SCRIPTED_SEQUENCE="$script_seq" )
  fi

  if [[ ${DRY_RUN} -eq 1 ]]; then
    # 仅运行底层脚本 dry-run 获取最终命令，不进入自动显存尝试逻辑
    printf ' %q' "${cmd[@]}"; echo ' --dry-run'
    # 捕获其输出中的 Launch command 行
    local dry_output
    dry_output=$(env "${env_export[@]}" bash "$RUN_SCRIPT" --dry-run 2>/dev/null || true)
    echo "$dry_output" | grep -E '^ \s*python' || true
    return 0
  fi

  # 实际训练
  if ! "${cmd[@]}"; then
    err "${pname} 运行失败 (run=${run_name})"
  fi
  {
    echo "phase=$pname"
    echo "completed_updates=$total_updates"
    echo "distance_weight=$dw"
    echo "scale_start=$scale_start"
    echo "scale_final=$scale_final"
    echo "anneal_steps=$anneal"
    [[ -n "$script_seq" ]] && echo "scripted_sequence=$script_seq" || true
    echo "timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "$done_flag"
  PHASE_SUMMARY+=("$pname|$run_name|$total_updates|$dw|$scale_start->$scale_final|$anneal|$script_seq")
  log "${pname} 完成，标记写入 ${done_flag}"
}

# Phase1
PHASE_SUMMARY=()

if [[ $SKIP_PHASE1 -eq 0 ]]; then
  run_phase Phase1 "$PHASE1_RUN" "$PHASE1_UPDATES" "$PHASE1_DISTANCE_WEIGHT" \
    "$PHASE1_SCALE_START" "$PHASE1_SCALE_FINAL" "$PHASE1_SCALE_ANNEAL_STEPS" "$PHASE1_SCRIPTED_SEQUENCE" 1 0 FORCE_PHASE1
else
  log "跳过 Phase1 (SKIP_PHASE1=1)"
fi

# Phase2 (复制 Phase1 checkpoint -> Phase2)
if [[ $SKIP_PHASE2 -eq 0 ]]; then
  copy_checkpoint "$PHASE1_RUN" "$PHASE2_RUN"
  run_phase Phase2 "$PHASE2_RUN" "$PHASE2_UPDATES" "$PHASE2_DISTANCE_WEIGHT" \
    "$PHASE2_SCALE_START" "$PHASE2_SCALE_FINAL" "$PHASE2_SCALE_ANNEAL_STEPS" "" 0 0 FORCE_PHASE2
else
  log "跳过 Phase2 (SKIP_PHASE2=1)"
fi

# Phase3 (复制 Phase2 -> Phase3)
if [[ $SKIP_PHASE3 -eq 0 ]]; then
  copy_checkpoint "$PHASE2_RUN" "$PHASE3_RUN"
  run_phase Phase3 "$PHASE3_RUN" "$PHASE3_UPDATES" "$PHASE3_DISTANCE_WEIGHT" \
    "$PHASE3_SCALE_START" "$PHASE3_SCALE_FINAL" "$PHASE3_SCALE_ANNEAL_STEPS" "" 0 "$PHASE3_DISABLE_DISTANCE" FORCE_PHASE3
else
  log "跳过 Phase3 (SKIP_PHASE3=1)"
fi

if [[ ${DRY_RUN} -eq 1 ]]; then
  log "Dry run 完成。"
  exit 0
fi

log "全部阶段完成。"
if ((${#PHASE_SUMMARY[@]} > 0)); then
  echo "[auto-phases] 摘要:"
  printf '  %-7s | %-16s | %-8s | %-8s | %-13s | %-11s | %s\n' Phase Run Updates DistW Scale Anneal Script
  printf '  %s\n' '---------------------------------------------------------------------------------------------'
  for row in "${PHASE_SUMMARY[@]}"; do
    IFS='|' read -r pn rn upd dw sc an scseq <<<"$row"
    printf '  %-7s | %-16s | %-8s | %-8s | %-13s | %-11s | %s\n' "$pn" "$rn" "$upd" "$dw" "$sc" "$an" "${scseq:-}" 
  done
fi
