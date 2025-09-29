# Super Mario Bros A3C 项目优化迭代计划

## 目标概述

- 提升训练效率与稳定性，将单机多进程 A3C 升级为可扩展的分布式/向量化训练框架。
- 提高模型表现，引入更强的特征提取、序列建模与探索机制，缩短收敛时间。
- 完善工程化能力，便于复现实验、自动化评估与可视化监控。

## 核心技术升级方向

- **环境与框架更新**
  - 迁移至 `Gymnasium` 与 `NES-Py` 最新版本，确保与 2024 年生态兼容。
  - 采用 `gymnasium.vector.AsyncVectorEnv` 或 `envpool` 构建向量化环境，取代手工多进程以提升吞吐。
  - 引入 `SuperMarioBrosRandomStages-v0` 等随机化环境，增强泛化能力。

- **模型与算法改进**
  - 使用 `torch.compile` (PyTorch 2.1+) 对模型进行编译优化，减少前向与反向开销。
  - 升级特征提取器：引入 `ImpalaResBlock` 或 `ConvNeXt` 风格模块，结合 `LayerNorm` 稳定训练。
  - 将 LSTM 替换为 `GRU` 或 `GTrXL`（基于注意力的记忆模块）以更好建模长程依赖。
  - 结合 `V-trace` (IMPALA) 或 `R2D2` 的截断反向传播策略，提升多步返回的稳健性。
  - 在策略头加入参数化噪声（NoisyNet）或使用 `A3C + UCB-Exploration` 改善探索。

- **训练策略优化**
  - 使用 `AdamW` + `GradClip`，配合 `lr cosine decay` 与 `warmup` 控制学习率。
  - 实现 `Mixed Precision (AMP)` 训练与梯度累积，提升 GPU 利用率。
  - 引入 `Prioritized Experience Replay (PER)` 的离线缓冲，混合 on-policy/off-policy 经验加速收敛。
  - 利用 `Population Based Training (PBT)` 或 `Optuna` 超参搜索自动调优。

- **监控与评估增强**
  - 切换至 `TensorBoard` 原生记录或 `Weights & Biases`，实现训练曲线、超参与视频统一可视化。
  - 增加评估基线（随机、右移、基线 A2C）与 `SPS/ETA` 指标跟踪。
  - 引入单元测试与集成测试覆盖环境封装、模型初始化共享内存等关键逻辑。

- **工程与部署**
  - 构建基于 `Lightning Fabric` 或 `TorchRL` 的训练脚手架，整合分布式策略与日志。
  - 提供 Docker 镜像与 `conda`/`uv` 环境文件，支持跨平台部署。
  - 支持在云端（AWS EC2 G5/G6、Lambda Labs、Paperspace）一键启动训练，并配置自动保存与告警。

## 分阶段迭代路线

1. **基础设施升级（第 0-2 周）**
   - 完成依赖升级与环境迁移至 `Gymnasium`/`torch>=2.1`。
   - 替换自定义多进程实现，接入向量化环境与共享内存缓存。
   - 重构日志体系为统一的可视化平台（TensorBoard/W&B）。

2. **算法增强（第 3-6 周）**
   - 逐步引入编译优化、混合精度与学习率调度策略。
   - 实验不同特征提取器与序列建模结构（ResNet、GRU、GTrXL）。
   - 实现 V-trace 校正与熵调度，验证收敛速度与稳定性提升幅度。

3. **探索与超参（第 7-9 周）**
   - 集成 PBT/Optuna，通过云端并行搜索关键超参组合。
   - 测试 NoisyNet、参数空间噪声及对比性奖励塑形策略。
   - 增加随机化关卡与数据增强（观测翻转、亮度扰动）评估泛化表现。

4. **评估与部署（第 10-12 周）**
   - 完成回归测试与稳定性实验，输出多关卡对比报告。
   - 打包 Docker 镜像与 CI/CD 流水线，实现自动训练与模型发布。
   - 上线云端训练脚本，支持断点续训与模型版本管理。

## 风险与缓解

- **依赖兼容性**：升级至 Gymnasium/NES-Py 可能引入 API 变化，需通过适配层与单元测试保障。
- **训练不稳定**：高级模型（注意力、PER）可能带来爆炸梯度；需配合 `grad clip`、`ema` 监控数值稳定性。
- **资源成本**：分布式/超参搜索增加云成本，可预设预算与自动停机策略。
- **复杂度上升**：工程重构需保持模块化与文档同步更新，避免知识传递断层。

## 预期产出

- 新版代码框架（TorchRL/Lightning Fabric + 向量化环境）。
- 多关卡基准实验结果（旧版 vs 新版 A3C、V-trace、GTrXL 等）。
- 自动化训练/评估管线与可视化仪表盘。
- 面向研究与产品化的部署手册与容器镜像。

