"""
调试推理脚本 - 分析模型行为
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from src.config import ModelConfig
from src.envs.mario import MarioEnvConfig, MarioVectorEnvConfig, create_vector_env
from src.models import MarioActorCritic


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
        # 运行调试推理
        print("\n=== 调试推理 ===")

        obs_np, info = env.reset()
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)

        print(f"观测形状: {obs.shape}")  # 应该是 [1, 4, 84, 84]
        print(f"观测数据范围: [{obs.min():.3f}, {obs.max():.3f}]")
        # 使用 numpy 打印一次均值以避免未使用告警
        print(
            f"观测像素均值 (numpy): {float(np.asarray(obs.cpu().numpy()).mean()):.3f}"
        )

        hidden_state, cell_state = model.initial_state(1, device)
        print(
            f"隐藏状态形状: {hidden_state.shape if hidden_state is not None else 'None'}"
        )

        # 分析前几步的动作分布
        for step in range(20):  # 只分析前20步
            with torch.no_grad():
                output = model(obs, hidden_state, cell_state)
                logits = output.logits
                values = output.value

                # 分析动作概率分布
                probs = F.softmax(logits, dim=-1)

                # 使用不同策略
                greedy_action = torch.argmax(logits, dim=-1)
                dist = Categorical(logits=logits)
                sampled_action = dist.sample()

                print(f"\n步数 {step+1}:")
                print(f"  价值估计: {values.item():.3f}")
                print(f"  贪婪动作: {greedy_action.item()}")
                print(f"  采样动作: {sampled_action.item()}")
                print(f"  熵: {dist.entropy().item():.3f}")

                # 显示动作概率分布（前5个最高概率）
                top5_probs, top5_actions = torch.topk(probs[0], 5)
                prob_str = " | ".join(
                    [
                        f"动作{a.item()}:{p.item():.3f}"
                        for a, p in zip(top5_actions, top5_probs)
                    ]
                )
                print(f"  Top5概率: {prob_str}")

                # 执行动作（使用贪婪策略）
                action_np = greedy_action.cpu().numpy()
                obs_np, reward_np, terminated, truncated, info_dict = env.step(
                    action_np
                )

                reward = float(reward_np[0])
                print(f"  执行动作 {action_np[0]} -> 奖励: {reward:.1f}")

                # 更新状态
                obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
                if output.hidden_state is not None:
                    hidden_state = output.hidden_state
                    if output.cell_state is not None:
                        cell_state = output.cell_state

                # 检查游戏状态
                if info_dict and len(info_dict) > 0:
                    if isinstance(info_dict, list):
                        game_info = info_dict[0] if len(info_dict) > 0 else {}
                    else:
                        game_info = info_dict

                    x_pos = game_info.get("x_pos", 0)
                    if x_pos > 0:
                        print(f"  游戏位置: {x_pos}")

                if terminated[0] or truncated[0]:
                    print(f"  Episode结束 (步数: {step+1})")
                    break

                # 短暂暂停以便观察
                if step < 5:
                    time.sleep(0.5)

        print("\n=== 分析总结 ===")
        print("1. 检查模型是否输出合理的动作概率分布")
        print("2. 确认价值函数是否有意义的输出")
        print("3. 观察动作选择是否过于集中在某几个动作上")
        print("4. 如果模型表现不佳，可能需要更多训练或检查训练过程")

    finally:
        env.close()
        print("\n调试推理完成!")


if __name__ == "__main__":
    main()
