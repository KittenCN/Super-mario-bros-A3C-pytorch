# 模型推理使用指南

## 快速开始

1. **激活环境**（选择一种方式）：
   ```bash
   # 方式A: 使用conda环境
   conda activate mario-a3c
   
   # 方式B: 如果没有环境，创建一个
   conda env create -f environment.yml
   conda activate mario-a3c
   
   # 方式C: 使用pip安装依赖
   pip install -r requirements.txt
   ```

2. **运行推理**：
   ```bash
   # 快速推理（推荐新手）
   python quick_inference.py
   
   # 或直接双击运行
   run_inference.bat
   ```

## 高级推理选项

### 完整功能推理脚本
```bash
# 基本推理 - 3个episode，随机策略
python inference.py --checkpoint trained_models/run01/a3c_world1_stage1_0008000.pt --episodes 3

# 确定性策略推理 - 选择概率最高的动作
python inference.py --checkpoint trained_models/run01/a3c_world1_stage1_0008000.pt --episodes 5 --deterministic

# 详细输出推理 - 显示每步详情
python inference.py --checkpoint trained_models/run01/a3c_world1_stage1_0008000.pt --episodes 2 --deterministic --verbose

# 录制推理视频
python inference.py --checkpoint trained_models/run01/a3c_world1_stage1_0008000.pt --episodes 1 --deterministic --record-video

# CPU推理（如果GPU不可用）
python inference.py --checkpoint trained_models/run01/a3c_world1_stage1_0008000.pt --episodes 3 --device cpu
```

### 参数说明
- `--checkpoint`: 模型文件路径 (.pt文件)
- `--episodes`: 推理轮数（默认5）
- `--deterministic`: 使用确定性策略而非随机采样
- `--verbose`: 显示详细的每步信息
- `--record-video`: 录制游戏视频
- `--device`: 指定设备 (auto/cpu/cuda)
- `--max-steps`: 单轮最大步数（默认10000）
- `--seed`: 随机种子

## 输出说明

推理完成后会显示：
- 成功率（通关比例）
- 平均奖励和最高奖励
- 平均步数
- 每个episode的详细结果

## 可用的训练模型

当前目录中的模型：
- `trained_models/run01/a3c_world1_stage1_0008000.pt` - 8000轮训练的模型
- `trained_models/run01/a3c_world1_stage1_latest.pt` - 最新快照（如果存在）

## 疑难解答

1. **ModuleNotFoundError**: 确保激活了正确的Python环境
2. **CUDA错误**: 使用 `--device cpu` 强制CPU推理
3. **环境构建失败**: 检查gym-super-mario-bros和nes-py是否正确安装
4. **视频录制问题**: 确保ffmpeg已安装且在PATH中

## 自定义推理

如需修改推理行为，可编辑 `quick_inference.py` 或 `inference.py`：
- 修改episode数量
- 调整策略（确定性 vs 随机）
- 更改世界/关卡设置
- 添加自定义输出格式