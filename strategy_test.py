"""
改进推理脚本 - 使用随机采样策略
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from src.app_config import ModelConfig
from src.envs.mario import MarioEnvConfig, MarioVectorEnvConfig, create_vector_env
from src.models import MarioActorCritic


def run_episode_with_sampling():
    # 配置
    checkpoint_path = Path("trained_models/run01/a3c_world1_stage1_0008000.pt")
    metadata_path = checkpoint_path.with_suffix(".json")

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载元数据
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"加载模型: {checkpoint_path.name}")
    print(
        f"World {metadata['world']}-{metadata['stage']} | 动作类型: {metadata['action_type']}"
    )

    # 创建模型
    model_cfg = ModelConfig(**metadata["model"])
    model = MarioActorCritic(model_cfg).to(device)

    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

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
        results = []

        # 运行多个episodes，比较不同策略
        strategies = [
            ("贪婪策略", "greedy"),
            ("随机采样", "sample"),
            ("温度采样", "temperature"),
        ]

        for strategy_name, strategy_type in strategies:
            print(f"\n=== {strategy_name} ===")

            obs_np, info = env.reset()
            obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
            hidden_state, cell_state = model.initial_state(1, device)

            total_reward = 0.0
            steps = 0
            max_x_pos = 0
            positions = []

            for step in range(2000):  # 限制2000步
                with torch.no_grad():
                    output = model(obs, hidden_state, cell_state)
                    logits = output.logits

                    # 不同策略选择动作
                    if strategy_type == "greedy":
                        action = torch.argmax(logits, dim=-1)
                    elif strategy_type == "sample":
                        dist = Categorical(logits=logits)
                        action = dist.sample()
                    else:  # temperature sampling
                        # 降低温度使策略更随机
                        temperature = 2.0
                        dist = Categorical(logits=logits / temperature)
                        action = dist.sample()

                action_np = action.cpu().numpy()
                obs_np, reward_np, terminated, truncated, info_dict = env.step(
                    action_np
                )

                reward = float(reward_np[0])
                total_reward += reward
                steps += 1

                # 更新状态
                obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
                if output.hidden_state is not None:
                    hidden_state = output.hidden_state
                    if output.cell_state is not None:
                        cell_state = output.cell_state

                # 记录位置信息
                if info_dict and len(info_dict) > 0:
                    if isinstance(info_dict, list):
                        game_info = info_dict[0] if len(info_dict) > 0 else {}
                    else:
                        game_info = info_dict

                    x_pos = game_info.get("x_pos", 0)
                    if x_pos > 0:
                        positions.append(x_pos)
                        max_x_pos = max(max_x_pos, x_pos)

                    # 检查通关
                    if game_info.get("flag_get", False):
                        print(
                            f"🏁 通关成功! 步数: {steps}, 奖励: {total_reward:.1f}, 位置: {x_pos}"
                        )
                        break

                # 每200步输出进度
                if (step + 1) % 200 == 0:
                    print(
                        f"  步数: {step+1:4d} | 最远位置: {max_x_pos:4d} | 累计奖励: {total_reward:7.1f}"
                    )

                if terminated[0] or truncated[0]:
                    break

            # 计算移动统计
            if positions:
                avg_pos = np.mean(positions)
                pos_progress = max_x_pos - (positions[0] if positions else 0)
            else:
                avg_pos = 0
                pos_progress = 0

            result = {
                "strategy": strategy_name,
                "reward": total_reward,
                "steps": steps,
                "max_x_pos": max_x_pos,
                "avg_x_pos": avg_pos,
                "progress": pos_progress,
                "success": max_x_pos > 100,  # 简单成功判断
            }
            results.append(result)

            print(
                f"结果 - 奖励: {total_reward:7.1f} | 最远: {max_x_pos:4d} | 步数: {steps:4d}"
            )

        # 总结对比
        print("\n=== 策略对比总结 ===")
        for result in results:
            success_mark = "✓" if result["success"] else "✗"
            print(
                f"{success_mark} {result['strategy']:8s}: 奖励={result['reward']:7.1f} 最远={result['max_x_pos']:4d} 进度={result['progress']:4d}"
            )

        # 找出最好的策略
        best_result = max(results, key=lambda x: x["max_x_pos"])
        print(
            f"\n🎯 最佳策略: {best_result['strategy']} (最远位置: {best_result['max_x_pos']})"
        )

        return results
    finally:
        env.close()


if __name__ == "__main__":
    print("=== Super Mario Bros A3C 模型推理测试 ===")
    print("测试不同策略的表现...\n")
    run_episode_with_sampling()
