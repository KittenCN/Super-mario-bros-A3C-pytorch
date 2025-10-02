"""
最小依赖推理脚本 - 仅需torch
"""

def minimal_inference():
    import sys
    import json
    from pathlib import Path
    
    try:
        import torch
        print("✓ torch imported successfully")
    except ImportError:
        print("❌ torch not found. Please install: pip install torch")
        return
    
    try:
        import numpy as np
        print("✓ numpy imported successfully") 
    except ImportError:
        print("❌ numpy not found. Please install: pip install numpy")
        return
    
    # 检查模型文件
    checkpoint_path = Path("trained_models/run01/a3c_world1_stage1_0008000.pt")
    metadata_path = checkpoint_path.with_suffix(".json")
    
    if not checkpoint_path.exists():
        print(f"❌ 模型文件不存在: {checkpoint_path}")
        return
    
    if not metadata_path.exists():
        print(f"❌ 元数据文件不存在: {metadata_path}")
        return
    
    # 加载元数据
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    print(f"✓ 找到模型文件: {checkpoint_path}")
    print(f"✓ World {metadata['world']}-{metadata['stage']} | 动作类型: {metadata['action_type']}")
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ 使用设备: {device}")
    
    try:
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        print(f"✓ 成功加载检查点")
        print(f"  - 训练步数: {checkpoint.get('global_step', 0)}")
        print(f"  - 训练轮次: {checkpoint.get('global_update', 0)}")
        
        # 检查模型状态字典
        model_state = checkpoint.get("model", {})
        print(f"  - 模型参数数量: {len(model_state)} 个层")
        
        # 列出一些关键层
        key_layers = [k for k in model_state.keys() if any(x in k for x in ['conv', 'fc', 'gru', 'policy', 'value'])][:5]
        if key_layers:
            print(f"  - 关键层示例: {', '.join(key_layers)}")
        
        print("\n🎉 模型加载验证成功!")
        print("   要完成完整推理，请安装游戏环境依赖:")
        print("   pip install gymnasium gym-super-mario-bros nes-py")
        
        return True
        
    except Exception as e:
        print(f"❌ 加载检查点失败: {e}")
        return False

if __name__ == "__main__":
    print("=== 最小依赖模型验证 ===\n")
    minimal_inference()