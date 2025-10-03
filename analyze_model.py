"""
模型状态分析脚本
"""

import json
from pathlib import Path

import torch

from src.config import ModelConfig
from src.models import MarioActorCritic


def analyze_checkpoint():
    checkpoint_path = Path("trained_models/run01/a3c_world1_stage1_0008000.pt")
    metadata_path = checkpoint_path.with_suffix(".json")

    # 加载元数据
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print("=== 模型状态分析 ===")
    print(f"训练轮次: {metadata['save_state']['global_update']}")
    print(f"训练步数: {metadata['save_state']['global_step']}")
    print(f"模型类型: {metadata['save_state']['type']}")

    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    print("\n=== Checkpoint 内容 ===")
    print(f"全局步数: {checkpoint.get('global_step', 'N/A')}")
    print(f"全局更新: {checkpoint.get('global_update', 'N/A')}")

    # 分析模型权重
    model_state = checkpoint["model"]

    print("\n=== 模型权重分析 ===")
    print(f"总参数层数: {len(model_state)}")

    # 检查一些关键层的权重统计
    key_layers = ["stem.0.weight", "policy_head.weight", "value_head.weight"]

    for layer_name in key_layers:
        if layer_name in model_state:
            weight = model_state[layer_name]
            print(f"{layer_name}:")
            print(f"  形状: {weight.shape}")
            print(f"  均值: {weight.mean().item():.6f}")
            print(f"  标准差: {weight.std().item():.6f}")
            print(f"  最小值: {weight.min().item():.6f}")
            print(f"  最大值: {weight.max().item():.6f}")

    # 检查是否有优化器状态
    if "optimizer" in checkpoint:
        print("\n=== 优化器状态 ===")
        print("✓ 包含优化器状态")
        opt_state = checkpoint["optimizer"]
        if "state" in opt_state and opt_state["state"]:
            print(f"优化器参数组数: {len(opt_state['state'])}")
        else:
            print("⚠️  优化器状态为空")
    else:
        print("\n=== 优化器状态 ===")
        print("❌ 不包含优化器状态")

    # 创建一个新的随机初始化模型做对比
    print("\n=== 与随机初始化对比 ===")
    model_cfg = ModelConfig(**metadata["model"])
    random_model = MarioActorCritic(model_cfg)

    # 比较第一层权重
    if "stem.0.weight" in model_state:
        trained_weight = model_state["stem.0.weight"]
        random_weight = random_model.state_dict()["stem.0.weight"]

        weight_diff = torch.norm(trained_weight - random_weight).item()
        print(f"第一层权重与随机初始化的差异: {weight_diff:.6f}")

        if weight_diff < 0.001:
            print("⚠️  权重几乎未变化，可能未经过有效训练")
        else:
            print("✓ 权重已发生变化，经过了训练")

    # 建议
    print("\n=== 建议 ===")
    if checkpoint.get("global_step", 0) == 0:
        print("1. 模型显示训练步数为0，可能是元数据错误或模型未训练")
        print("2. 建议检查其他checkpoint文件或重新训练模型")

    if metadata["save_state"]["global_update"] > 0:
        print("3. 元数据显示有8000次更新，但global_step为0可能是已知bug")
        print("4. 尝试使用latest快照或其他checkpoint文件")


if __name__ == "__main__":
    analyze_checkpoint()
