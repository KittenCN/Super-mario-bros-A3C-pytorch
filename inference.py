"""inference.py
推理脚本 - 加载训练好的 A3C 模型进行 Super Mario Bros 游戏推理。

增强特性 (2025-10-02 更新):
1. 自适应 checkpoint 目录解析：若 --checkpoint 指向目录，自动选择最新 (latest 优先其后按 global_update 排序)。
2. 灵活 state_dict 加载：兼容保存于未编译模型或 torch.compile 后 (含/不含 `_orig_mod.` 前缀) 的权重。
3. 可选使用训练时的完整 stage_schedule (--full-schedule)，而非单关卡固定。
4. 性能统计：每个 episode 输出 steps/sec，并在汇总中提供平均推理速度。
5. 详细缺失 / 多余权重键提示（截断显示 + 文件落盘）。

Usage:
    python inference.py --checkpoint trained_models/run01/a3c_world1_stage1_latest.pt --episodes 5 --deterministic
    python inference.py --checkpoint trained_models/run01 --full-schedule --episodes 8
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from src.config import ModelConfig
from src.envs.mario import MarioEnvConfig, MarioVectorEnvConfig, create_vector_env
from src.models import MarioActorCritic

ISSUES_LOG_NAME = "inference_state_dict_issues.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="推理 Super Mario Bros A3C 模型 (支持目录自动解析 / 前缀自适应加载)"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="检查点文件路径 (.pt)"
    )
    parser.add_argument("--episodes", type=int, default=5, help="推理 episode 数量")
    parser.add_argument(
        "--render", action="store_true", help="是否显示游戏画面（需要显示器）"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="使用确定性策略（选择概率最高的动作）而非采样",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="推理设备",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--max-steps", type=int, default=10000, help="单个 episode 最大步数"
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="录制推理视频到 output/inference_videos/",
    )
    parser.add_argument("--verbose", action="store_true", help="详细输出每步信息")
    parser.add_argument(
        "--full-schedule",
        action="store_true",
        help="使用训练 metadata 中的完整 stage_schedule (向量环境按顺序跑多关卡)",
    )
    parser.add_argument(
        "--max-missing-keys",
        type=int,
        default=20,
        help="控制打印缺失/多余权重键的最大数量 (默认 20)",
    )
    parser.add_argument(
        "--issues-log-dir",
        type=str,
        default=".",
        help="保存 state_dict 加载差异日志的目录 (默认当前)",
    )
    return parser.parse_args()


def _candidate_checkpoints_in_dir(directory: Path) -> List[Path]:
    return sorted(directory.glob("a3c_world*_stage*.pt"))


def _select_best_checkpoint_in_dir(directory: Path) -> Path:
    candidates = _candidate_checkpoints_in_dir(directory)
    if not candidates:
        raise FileNotFoundError(f"目录中未找到 *.pt: {directory}")
    # 优先 latest，再按 global_update 解析 json 元数据排序
    latest = [c for c in candidates if c.name.endswith("latest.pt")]
    if latest:
        return latest[0]
    scored: List[Tuple[int, int, Path]] = []  # (global_update, global_step, path)
    for ckpt in candidates:
        meta_p = ckpt.with_suffix(".json")
        if not meta_p.exists():
            continue
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            save_state = meta.get("save_state", {})
            upd = int(save_state.get("global_update", 0))
            step = int(save_state.get("global_step", 0))
            scored.append((upd, step, ckpt))
        except Exception:
            continue
    if not scored:
        return candidates[-1]
    scored.sort(reverse=True)
    return scored[0][2]


def resolve_checkpoint_path(arg_path: str) -> Path:
    p = Path(arg_path)
    if p.is_dir():
        return _select_best_checkpoint_in_dir(p)
    return p


def load_checkpoint_metadata(checkpoint_path: Path) -> dict:
    """加载检查点的元数据配置 (要求与 .pt 同名 .json)。"""
    metadata_path = checkpoint_path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"找不到模型元数据文件: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def create_inference_env(
    metadata: dict,
    render: bool = False,
    record_video: bool = False,
    full_schedule: bool = False,
):
    """根据模型元数据创建推理环境。

    参数:
        metadata: checkpoint json 内容。
        render: 是否开启渲染。
        record_video: 是否录制视频。
        full_schedule: 若 True 且 metadata.vector_env.stage_schedule 存在, 使用其序列而非单关卡。
    """
    schedule: List[Tuple[int, int]]
    if full_schedule:
        ve = metadata.get("vector_env", {})
        raw_sched = ve.get("stage_schedule") or []
        schedule = (
            [(int(w), int(s)) for (w, s) in raw_sched]
            if raw_sched
            else [(metadata["world"], metadata["stage"])]
        )
    else:
        schedule = [(metadata["world"], metadata["stage"])]

    env_cfg = MarioEnvConfig(
        world=metadata["world"],
        stage=metadata["stage"],
        action_type=metadata["action_type"],
        frame_skip=metadata["frame_skip"],
        frame_stack=metadata["frame_stack"],
        video_dir="output/inference_videos" if record_video else None,
        record_video=record_video,
    )
    vector_cfg = MarioVectorEnvConfig(
        num_envs=1,
        asynchronous=False,
        stage_schedule=tuple(schedule),
        random_start_stage=False,
        base_seed=42,
        env=env_cfg,
    )
    return create_vector_env(vector_cfg)


def _flex_load_state_dict(model: torch.nn.Module, state: dict) -> List[str]:
    """灵活加载 (兼容 _orig_mod. 前缀); 返回问题键列表。"""
    model_keys = set(model.state_dict().keys())
    state_keys = set(state.keys())
    model_prefixed = any(k.startswith("_orig_mod.") for k in model_keys)
    state_prefixed = any(k.startswith("_orig_mod.") for k in state_keys)
    adj = state
    if model_prefixed and not state_prefixed:
        adj = {f"_orig_mod.{k}": v for k, v in state.items()}
    elif not model_prefixed and state_prefixed:
        adj = {
            k[len("_orig_mod.") :]: v
            for k, v in state.items()
            if k.startswith("_orig_mod.")
        }
    missing, unexpected = model.load_state_dict(adj, strict=False)
    issues: List[str] = list(missing) + [f"unexpected:{u}" for u in unexpected]
    return issues


def load_model(
    checkpoint_path: Path,
    metadata: dict,
    device: torch.device,
    max_missing: int,
    issues_log_dir: Path,
) -> MarioActorCritic:
    """加载训练好的模型 (含自适应键前缀 + 问题日志)。"""
    model_cfg = ModelConfig(**metadata["model"])
    model = MarioActorCritic(model_cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    issues = _flex_load_state_dict(model, checkpoint["model"])
    model.eval()
    if issues:
        head = issues[:max_missing]
        more = (
            ""
            if len(issues) <= max_missing
            else f" ... (+{len(issues)-max_missing} more)"
        )
        print(f"[inference][warn] state_dict 差异: {len(issues)} -> {head}{more}")
        try:
            issues_log_dir.mkdir(parents=True, exist_ok=True)
            (issues_log_dir / ISSUES_LOG_NAME).write_text(
                "\n".join(issues), encoding="utf-8"
            )
            print(
                f"[inference] 差异已写入: {(issues_log_dir / ISSUES_LOG_NAME).as_posix()}"
            )
        except Exception as e:  # pragma: no cover
            print(f"[inference][warn] 写入差异日志失败: {e}")
    print(f"[inference] 成功加载模型: {checkpoint_path}")
    print(f"[inference] 模型配置: {model_cfg}")
    if "global_step" in checkpoint:
        print(f"[inference] 训练步数: {checkpoint['global_step']}")
    if "global_update" in checkpoint:
        print(f"[inference] 训练轮次: {checkpoint['global_update']}")
    return model


def _extract_single_info(raw_info: Any) -> dict:
    """Normalize vector env info (list[dict] | dict | other) -> dict.

    fc_emulator backend + SyncVectorEnv 下可能返回单个 dict (非 list)。
    兼容处理：
      - list 且首元素为 dict -> 返回首元素
      - dict -> 直接返回
      - 其它 -> {}
    """
    if isinstance(raw_info, list):
        if raw_info and isinstance(raw_info[0], dict):
            return raw_info[0]
        return {}
    if isinstance(raw_info, dict):
        return raw_info
    return {}


def _coerce_scalar(x):  # noqa: D401 - simple helper
    """将 numpy 标量 / ndarray[0] / tensor 转为 Python 标量 int/float。"""
    try:
        import numpy as _np  # local
        import torch as _t

        if isinstance(x, _t.Tensor):
            if x.numel() == 1:
                return x.item()
            return x
        if isinstance(x, _np.ndarray):
            if x.size == 1:
                return x.flatten()[0].item()
            return x
        if hasattr(x, "item"):
            return x.item()
    except Exception:
        pass
    return x


def run_inference_episode(
    env,
    model: MarioActorCritic,
    device: torch.device,
    max_steps: int = 10000,
    deterministic: bool = False,
    verbose: bool = False,
) -> tuple[float, int, dict, float]:
    """运行单个推理 episode"""
    obs_np, info = env.reset()
    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)

    # 初始化隐藏状态
    hidden_state, cell_state = model.initial_state(1, device)

    total_reward = 0.0
    steps = 0
    episode_info = {"x_pos": [], "stage": [], "flag_get": False, "life": 2}

    start_time = time.time()
    for step in range(max_steps):
        with torch.no_grad():
            output = model(obs, hidden_state, cell_state)
            logits = output.logits
            value = output.value

            if deterministic:
                # 确定性策略：选择概率最高的动作
                action = torch.argmax(logits, dim=-1)
            else:
                # 随机策略：按概率采样
                dist = Categorical(logits=logits)
                action = dist.sample()

        action_np = action.cpu().numpy()
        obs_np, reward_np, terminated, truncated, raw_info = env.step(action_np)
        info_item = _extract_single_info(raw_info)
        # 兼容 reward_np shape=[1] 或标量
        try:
            reward = float(reward_np[0] if hasattr(reward_np, "__len__") else reward_np)
        except Exception:
            reward = float(reward_np)
        total_reward += reward
        steps += 1

        # 记录游戏信息
        if info_item:
            # 兼容 fc_emulator metrics
            x_pos_val = info_item.get("x_pos")
            if x_pos_val is None:
                metrics = info_item.get("metrics")
                if isinstance(metrics, dict):
                    x_pos_val = metrics.get("mario_x") or metrics.get("x_pos")
            if x_pos_val is not None:
                x_pos_val = _coerce_scalar(x_pos_val)
                try:
                    episode_info["x_pos"].append(int(x_pos_val))
                except Exception:
                    pass
            episode_info["stage"].append(info_item.get("stage", 1))
            if info_item.get("flag_get", False):
                episode_info["flag_get"] = True
            if "life" in info_item:
                try:
                    episode_info["life"] = int(_coerce_scalar(info_item["life"]))
                except Exception:
                    pass

        if verbose and (step + 1) % 100 == 0:
            action_id = int(action_np[0])
            prob = F.softmax(logits, dim=-1)[0, action_id].item()
            print(
                f"  步数 {step+1:4d}: 动作={action_id:2d} 概率={prob:.3f} 奖励={reward:6.1f} 价值={value.item():6.2f}"
            )

        # 更新观测和隐藏状态
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        if output.hidden_state is not None:
            hidden_state = output.hidden_state
            if terminated[0] or truncated[0]:
                hidden_state = hidden_state * 0.0  # 重置隐藏状态
            if output.cell_state is not None:
                cell_state = output.cell_state
                if terminated[0] or truncated[0]:
                    cell_state = cell_state * 0.0

        # 检查 episode 结束
        # terminated / truncated 兼容标量或 ndarray
        term_flag = terminated[0] if hasattr(terminated, "__len__") else terminated
        trunc_flag = truncated[0] if hasattr(truncated, "__len__") else truncated
        if term_flag or trunc_flag:
            break
    duration = time.time() - start_time
    return total_reward, steps, episode_info, duration


def main():
    args = parse_args()

    # 设备配置
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[inference] 使用设备: {device}")

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 加载检查点和元数据
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

    print(f"[inference] 加载检查点: {checkpoint_path}")
    metadata = load_checkpoint_metadata(checkpoint_path)

    # 创建环境
    print(
        f"[inference] 创建推理环境 World {metadata['world']}-{metadata['stage']} ({metadata['action_type']})"
    )
    env = create_inference_env(
        metadata,
        render=args.render,
        record_video=args.record_video,
        full_schedule=args.full_schedule,
    )

    try:
        # 加载模型
        model = load_model(
            checkpoint_path,
            metadata,
            device,
            max_missing=args.max_missing_keys,
            issues_log_dir=Path(args.issues_log_dir),
        )

        # 运行推理
        print(f"[inference] 开始推理 {args.episodes} 个 episode...")
        print("=" * 80)

        results = []
        total_start_time = time.time()
        total_env_steps = 0
        total_env_time = 0.0
        for episode in range(args.episodes):
            print(f"Episode {episode + 1}/{args.episodes}")

            episode_start_time = time.time()
            reward, steps, info, epi_time = run_inference_episode(
                env,
                model,
                device,
                max_steps=args.max_steps,
                deterministic=args.deterministic,
                verbose=args.verbose,
            )
            episode_duration = time.time() - episode_start_time
            steps_per_sec = steps / epi_time if epi_time > 0 else 0.0
            total_env_steps += steps
            total_env_time += epi_time

            results.append(
                {
                    "episode": episode + 1,
                    "reward": reward,
                    "steps": steps,
                    "duration": episode_duration,
                    "flag_get": info["flag_get"],
                    "max_x_pos": max(info["x_pos"]) if info["x_pos"] else 0,
                    "final_life": info["life"],
                }
            )

            status = (
                "🏁 通关!"
                if info["flag_get"]
                else f"❌ 失败 (最远位置: {max(info['x_pos']) if info['x_pos'] else 0})"
            )
            print(f"  结果: {status}")
            print(
                f"  奖励: {reward:8.1f} | 步数: {steps:5d} | 时长: {episode_duration:6.2f}s | 速率: {steps_per_sec:6.1f} steps/s"
            )
            print("-" * 80)

        # 汇总统计
        total_duration = time.time() - total_start_time
        avg_reward = np.mean([r["reward"] for r in results])
        avg_steps = np.mean([r["steps"] for r in results])
        success_rate = np.mean([r["flag_get"] for r in results]) * 100
        max_reward = max([r["reward"] for r in results])

        print("\n📊 推理汇总统计:")
        print(
            f"  成功率: {success_rate:5.1f}% ({sum(r['flag_get'] for r in results)}/{args.episodes})"
        )
        print(f"  平均奖励: {avg_reward:8.1f} (最高: {max_reward:8.1f})")
        print(f"  平均步数: {avg_steps:8.1f}")
        print(f"  总耗时: {total_duration:6.2f}s")
        if total_env_time > 0:
            print(
                f"  平均推理速率: {total_env_steps/total_env_time:6.1f} steps/s (基于无渲染计时)"
            )
        print(f"  策略模式: {'确定性' if args.deterministic else '随机采样'}")

        if args.record_video:
            print("  视频已保存至: output/inference_videos/")
        print("[inference] 推理完成")

    finally:
        env.close()


if __name__ == "__main__":
    main()
