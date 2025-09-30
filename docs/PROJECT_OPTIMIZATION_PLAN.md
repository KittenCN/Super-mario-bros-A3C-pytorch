# Super Mario Bros A3C 项目优化计划 | Project Optimisation Plan

本计划概述 12 周内让 Super Mario Bros A3C 管线更高效、更稳定、更易复现的关键举措。<br>This plan outlines the 12-week roadmap to make the Super Mario Bros A3C pipeline faster, more stable, and easier to reproduce.

## 核心目标 | Core Objectives
- 提升训练吞吐与稳定性，使单机多进程方案可平滑扩展到分布式或更高并发。<br>Increase training throughput and stability so the single-machine multi-process design can scale to distributed or higher-concurrency setups.
- 增强模型表现，引入更强特征提取、序列建模与探索策略，加快收敛。<br>Boost policy performance with stronger feature extractors, sequence models, and exploration strategies to speed up convergence.
- 完善工程化能力，覆盖环境搭建、监控、测试、部署的全链路。<br>Harden engineering practices spanning environment setup, monitoring, testing, and deployment.

## 技术升级方向 | Technical Upgrades
- **环境与框架**：迁移至最新 `Gymnasium`/`nes-py`，深入利用 `gymnasium.vector.AsyncVectorEnv` 或 `envpool`，并引入随机关卡增强泛化。<br>**Environment & framework**: migrate to up-to-date `Gymnasium`/`nes-py`, leverage `gymnasium.vector.AsyncVectorEnv` or `envpool`, and add randomised stages for generalisation.
- **模型与算法**：结合 `torch.compile`、AMP、IMPALA 残差骨干、GTrXL/GRU/LSTM 序列模块、NoisyNet 探索及 V-trace/R2D2 截断回传。<br>**Model & algorithm**: combine `torch.compile`, AMP, IMPALA residual backbones, GTrXL/GRU/LSTM modules, NoisyNet exploration, and V-trace/R2D2 truncated BPTT.
- **训练策略**：采用 `AdamW`、梯度裁剪、余弦退火 + warmup、梯度累积、优先经验回放 (PER)、Optuna/PBT 超参搜索。<br>**Training strategy**: apply `AdamW`, gradient clipping, cosine decay with warmup, gradient accumulation, PER, and Optuna/PBT hyper-parameter search.
- **监控与评估**：统一到 TensorBoard/W&B，补充基线评估、SPS/ETA 指标与自动化回归测试。<br>**Monitoring & evaluation**: standardise on TensorBoard/W&B, add baseline evaluations, SPS/ETA metrics, and automated regression tests.
- **工程体系**：引入 Lightning Fabric/TorchRL 脚手架、Docker/conda/uv 环境、云端训练脚本与 CI/CD 流水线。<br>**Engineering**: employ Lightning Fabric/TorchRL scaffolding, provide Docker/conda/uv environments, cloud-training scripts, and CI/CD pipelines.

## 分阶段路线 | Phased Roadmap
1. **第 0–2 周：基础设施升级**<br>**Weeks 0–2: Infrastructure upgrades**
   - 完成依赖升级与环境迁移，并重构日志/配置入口。<br>   Upgrade dependencies, complete the environment migration, and refactor logging/config entry points.
   - 以向量化环境替换手写多进程实现，验证稳定性。<br>   Replace the custom multiprocessing implementation with vector environments and validate stability.
2. **第 3–6 周：算法增强**<br>**Weeks 3–6: Algorithm enhancements**
   - 集成编译优化、AMP、学习率调度；实验 ResNet/Impala/GTrXL 组合。<br>   Integrate compile optimisations, AMP, LR scheduling; experiment with ResNet/Impala/GTrXL combinations.
   - 引入 V-trace、熵调度、PER，量化对收敛速度的影响。<br>   Add V-trace, entropy scheduling, and PER, measuring their impact on convergence speed.
3. **第 7–9 周：探索与超参**<br>**Weeks 7–9: Exploration & hyper-parameters**
   - 启动 Optuna/PBT 搜索，覆盖学习率、隐藏维度、循环类型、熵系数等。<br>   Launch Optuna/PBT sweeps covering learning rate, hidden sizes, recurrent types, entropy coefficients, and more.
   - 实验随机关卡、观测扰动、参数噪声等泛化策略。<br>   Test random stage schedules, observation augmentations, and parameter-space noise strategies.
4. **第 10–12 周：评估与交付**<br>**Weeks 10–12: Evaluation & delivery**
   - 完成回归测试与多关卡基准，对比旧版/新版性能并生成报告。<br>   Finish regression tests and multi-stage benchmarks, comparing legacy vs. upgraded performance with reports.
   - 打包 Docker 镜像与部署脚本，接入 CI/CD 并配置云端训练。<br>   Package Docker images and deployment scripts, wire them into CI/CD, and enable cloud training workflows.

## 风险与缓解 | Risks & Mitigations
- **依赖兼容性**：API 变化可能破坏现有流程；需配合适配层与单元测试。<br>**Dependency compatibility**: API changes may break flows; mitigate with adapters and unit tests.
- **训练不稳定**：更复杂的模型/探索可能导致梯度爆炸；使用梯度裁剪、EMA、监控告警。<br>**Training instability**: richer models/exploration can cause gradient blow-ups; contain with gradient clipping, EMA, and alerting.
- **资源成本**：分布式搜索耗费算力；设置预算上限与自动停机策略。<br>**Resource cost**: distributed sweeps are resource-intensive; enforce budget caps and auto-shutdown policies.
- **工程复杂度**：大幅重构增加维护门槛；保持模块化设计并同步更新文档。<br>**Engineering complexity**: broad refactors raise maintenance overhead; keep modules modular and update documentation in lockstep.

## 预期交付物 | Expected Deliverables
- 新版训练框架及脚手架示例，含最小可运行案例。<br>The upgraded training framework with scaffold examples and a minimal runnable demo.
- 多关卡性能对比报告与指标仪表盘。<br>Multi-stage performance comparison reports plus metrics dashboards.
- 自动化测试、部署脚本、容器镜像与环境说明。<br>Automated tests, deployment scripts, container images, and environment guides.
- 更新后的文档体系（FAQ、故障排查、运维手册）。<br>An updated documentation suite covering FAQ, troubleshooting, and operations guides.
