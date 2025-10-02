"""
快速推理脚本 - 直接运行指定模型
Usage: python quick_inference.py
"""

import torch
import numpy as np
from pathlib import Path
import json
import time

from src.config import ModelConfig
from src.envs.mario import MarioEnvConfig, create_vector_env, MarioVectorEnvConfig
from src.models import MarioActorCritic
from torch.distributions import Categorical


def main():
    # 配置
    checkpoint_path = Path("trained_models/run01/a3c_world1_stage1_0008000.pt")
    metadata_path = checkpoint_path.with_suffix(".json")
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载元数据
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    print(f"加载模型: {checkpoint_path}")
    print(f"World {metadata['world']}-{metadata['stage']} | 动作类型: {metadata['action_type']}")
    
    # 创建模型
    model_cfg = ModelConfig(**metadata["model"])
    model = MarioActorCritic(model_cfg).to(device)
    
    # 加载权重 - 设置 weights_only=False 以兼容旧版本checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    print(f"训练步数: {checkpoint.get('global_step', 0)}")
    print(f"训练轮次: {checkpoint.get('global_update', 0)}")
    
    # 创建环境
    env_cfg = MarioEnvConfig(
        world=metadata["world"],
        stage=metadata["stage"],
        action_type=metadata["action_type"],
        frame_skip=metadata["frame_skip"],
        frame_stack=metadata["frame_stack"],
    )
    
    vector_cfg = MarioVectorEnvConfig(
        num_envs=1,
        asynchronous=False,
        stage_schedule=tuple([(metadata["world"], metadata["stage"])]),
        random_start_stage=False,
        base_seed=42,
        env=env_cfg,
    )
    
    env = create_vector_env(vector_cfg)
    
    try:
        # 运行推理
        for episode in range(3):  # 运行3个episode
            print(f"\n=== Episode {episode + 1} ===")
            
            obs_np, info = env.reset()
            obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
            
            hidden_state, cell_state = model.initial_state(1, device)
            
            total_reward = 0.0
            steps = 0
            max_x_pos = 0
            
            for step in range(5000):  # 最多5000步
                with torch.no_grad():
                    output = model(obs, hidden_state, cell_state)
                    
                    # 使用确定性策略（选最高概率动作）
                    action = torch.argmax(output.logits, dim=-1)
                    
                action_np = action.cpu().numpy()
                obs_np, reward_np, terminated, truncated, info_dict = env.step(action_np)
                
                reward = float(reward_np[0])
                total_reward += reward
                steps += 1
                
                # 更新状态
                obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
                if output.hidden_state is not None:
                    hidden_state = output.hidden_state
                    if output.cell_state is not None:
                        cell_state = output.cell_state
                
                # 获取游戏信息
                current_x_pos = 0
                if info_dict and len(info_dict) > 0:
                    # info_dict可能是list或dict，需要适配
                    if isinstance(info_dict, list) and len(info_dict) > 0:
                        game_info = info_dict[0]
                    elif isinstance(info_dict, dict):
                        game_info = info_dict
                    else:
                        game_info = {}
                    
                    current_x_pos = game_info.get("x_pos", current_x_pos)
                    max_x_pos = max(max_x_pos, current_x_pos)
                    
                    # 每100步打印一次进度
                    if (step + 1) % 100 == 0:
                        print(f"  步数: {step+1:4d} | X位置: {current_x_pos:4d} | 累计奖励: {total_reward:7.1f}")
                    
                    # 检查通关
                    if game_info.get("flag_get", False):
                        print(f"🏁 通关成功! 步数: {steps}, 奖励: {total_reward:.1f}")
                        break
                else:
                    # 没有info时的进度输出
                    if (step + 1) % 100 == 0:
                        print(f"  步数: {step+1:4d} | 累计奖励: {total_reward:7.1f}")
                
                if terminated[0] or truncated[0]:
                    print(f"❌ Episode结束 | 最远位置: {max_x_pos} | 总奖励: {total_reward:.1f} | 总步数: {steps}")
                    break
            
            time.sleep(1)  # 短暂暂停
    
    finally:
        env.close()
        print("\n推理完成!")


if __name__ == "__main__":
    main()