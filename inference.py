"""
推理脚本 - 加载训练好的 A3C 模型进行 Super Mario Bros 游戏推理
Usage: python inference.py --checkpoint path/to/checkpoint.pt [options]
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from src.config import ModelConfig
from src.envs.mario import MarioEnvConfig, create_vector_env, MarioVectorEnvConfig
from src.models import MarioActorCritic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="推理 Super Mario Bros A3C 模型")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True, 
        help="检查点文件路径 (.pt)"
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=5, 
        help="推理 episode 数量"
    )
    parser.add_argument(
        "--render", 
        action="store_true", 
        help="是否显示游戏画面（需要显示器）"
    )
    parser.add_argument(
        "--deterministic", 
        action="store_true", 
        help="使用确定性策略（选择概率最高的动作）而非采样"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto", 
        choices=["auto", "cpu", "cuda"], 
        help="推理设备"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="随机种子"
    )
    parser.add_argument(
        "--max-steps", 
        type=int, 
        default=10000, 
        help="单个 episode 最大步数"
    )
    parser.add_argument(
        "--record-video", 
        action="store_true", 
        help="录制推理视频到 output/inference_videos/"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="详细输出每步信息"
    )
    return parser.parse_args()


def load_checkpoint_metadata(checkpoint_path: Path) -> dict:
    """加载检查点的元数据配置"""
    metadata_path = checkpoint_path.with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"找不到模型元数据文件: {metadata_path}")
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_inference_env(metadata: dict, render: bool = False, record_video: bool = False):
    """根据模型元数据创建推理环境"""
    # 从元数据构建环境配置
    env_cfg = MarioEnvConfig(
        world=metadata["world"],
        stage=metadata["stage"], 
        action_type=metadata["action_type"],
        frame_skip=metadata["frame_skip"],
        frame_stack=metadata["frame_stack"],
        video_dir="output/inference_videos" if record_video else None,
        record_video=record_video,
    )
    
    # 单环境推理配置
    vector_cfg = MarioVectorEnvConfig(
        num_envs=1,
        asynchronous=False,
        stage_schedule=tuple([(metadata["world"], metadata["stage"])]),
        random_start_stage=False,
        base_seed=42,
        env=env_cfg,
    )
    
    return create_vector_env(vector_cfg)


def load_model(checkpoint_path: Path, metadata: dict, device: torch.device) -> MarioActorCritic:
    """加载训练好的模型"""
    model_cfg = ModelConfig(**metadata["model"])
    model = MarioActorCritic(model_cfg).to(device)
    
    # 加载检查点 - 设置 weights_only=False 以兼容旧版本checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()  # 设置为推理模式
    
    print(f"[inference] 成功加载模型: {checkpoint_path}")
    print(f"[inference] 模型配置: {model_cfg}")
    
    if "global_step" in checkpoint:
        print(f"[inference] 训练步数: {checkpoint['global_step']}")
    if "global_update" in checkpoint:
        print(f"[inference] 训练轮次: {checkpoint['global_update']}")
    
    return model


def run_inference_episode(
    env, 
    model: MarioActorCritic, 
    device: torch.device, 
    max_steps: int = 10000,
    deterministic: bool = False,
    verbose: bool = False
) -> tuple[float, int, dict]:
    """运行单个推理 episode"""
    obs_np, info = env.reset()
    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    
    # 初始化隐藏状态
    hidden_state, cell_state = model.initial_state(1, device)
    
    total_reward = 0.0
    steps = 0
    episode_info = {"x_pos": [], "stage": [], "flag_get": False, "life": 2}
    
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
        obs_np, reward_np, terminated, truncated, info_dict = env.step(action_np)
        
        reward = float(reward_np[0])
        total_reward += reward
        steps += 1
        
        # 记录游戏信息
        if info_dict and len(info_dict) > 0 and "x_pos" in info_dict[0]:
            episode_info["x_pos"].append(info_dict[0]["x_pos"])
            episode_info["stage"].append(info_dict[0].get("stage", 1))
            if info_dict[0].get("flag_get", False):
                episode_info["flag_get"] = True
            episode_info["life"] = info_dict[0].get("life", 2)
        
        if verbose and (step + 1) % 100 == 0:
            action_id = int(action_np[0])
            prob = F.softmax(logits, dim=-1)[0, action_id].item()
            print(f"  步数 {step+1:4d}: 动作={action_id:2d} 概率={prob:.3f} 奖励={reward:6.1f} 价值={value.item():6.2f}")
        
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
        if terminated[0] or truncated[0]:
            break
    
    return total_reward, steps, episode_info


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
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    print(f"[inference] 加载检查点: {checkpoint_path}")
    metadata = load_checkpoint_metadata(checkpoint_path)
    
    # 创建环境
    print(f"[inference] 创建推理环境 World {metadata['world']}-{metadata['stage']} ({metadata['action_type']})")
    env = create_inference_env(metadata, render=args.render, record_video=args.record_video)
    
    try:
        # 加载模型
        model = load_model(checkpoint_path, metadata, device)
        
        # 运行推理
        print(f"[inference] 开始推理 {args.episodes} 个 episode...")
        print("=" * 80)
        
        results = []
        total_start_time = time.time()
        
        for episode in range(args.episodes):
            print(f"Episode {episode + 1}/{args.episodes}")
            
            episode_start_time = time.time()
            reward, steps, info = run_inference_episode(
                env, model, device, 
                max_steps=args.max_steps,
                deterministic=args.deterministic,
                verbose=args.verbose
            )
            episode_duration = time.time() - episode_start_time
            
            results.append({
                "episode": episode + 1,
                "reward": reward,
                "steps": steps,
                "duration": episode_duration,
                "flag_get": info["flag_get"],
                "max_x_pos": max(info["x_pos"]) if info["x_pos"] else 0,
                "final_life": info["life"]
            })
            
            status = "🏁 通关!" if info["flag_get"] else f"❌ 失败 (最远位置: {max(info['x_pos']) if info['x_pos'] else 0})"
            print(f"  结果: {status}")
            print(f"  奖励: {reward:8.1f} | 步数: {steps:5d} | 时长: {episode_duration:6.2f}s")
            print("-" * 80)
        
        # 汇总统计
        total_duration = time.time() - total_start_time
        avg_reward = np.mean([r["reward"] for r in results])
        avg_steps = np.mean([r["steps"] for r in results])
        success_rate = np.mean([r["flag_get"] for r in results]) * 100
        max_reward = max([r["reward"] for r in results])
        
        print(f"\n📊 推理汇总统计:")
        print(f"  成功率: {success_rate:5.1f}% ({sum(r['flag_get'] for r in results)}/{args.episodes})")
        print(f"  平均奖励: {avg_reward:8.1f} (最高: {max_reward:8.1f})")
        print(f"  平均步数: {avg_steps:8.1f}")
        print(f"  总耗时: {total_duration:6.2f}s")
        print(f"  策略模式: {'确定性' if args.deterministic else '随机采样'}")
        
        if args.record_video:
            print(f"  视频已保存至: output/inference_videos/")
            
    finally:
        env.close()
        print("[inference] 推理完成")


if __name__ == "__main__":
    main()