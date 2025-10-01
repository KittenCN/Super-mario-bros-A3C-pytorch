# 文档与归档策略 | Documentation & Archiving Policy

本项目的 Markdown 文档管理策略如下：

## 目录归属 | Location Rules
1. `README.md`：仅保留项目当前最新使用说明、特性、快速开始、调试要点，不在其中累积历史变更记录。
2. `AGENTS.md`：仅保留最新的自动执行/代理协作规范，不堆叠历史版本；若需比较旧规范，单独建立对比文档（可选）。
3. 其余所有 Markdown（运行报告、调试分析、实验记录等）统一放置于 `docs/` 目录。根目录中不再新增新的 `.md` 文件，除 `README.md` 与 `AGENTS.md`。
4. 归档/历史版本采用 `docs/archive/` 子目录（需要时再创建），命名格式：`<slug>_YYYYMMDD.md`。

## 变更记录 | Change Tracking
1. 功能/接口层面的显式变更在 `CHANGELOG.md`（若尚未创建，可后续补充）记录，遵循 Keep a Changelog + SemVer。
2. 调试、性能、环境稳定性一类过程性报告放在 `docs/`：例如 `ENV_DEBUGGING_REPORT.md`、`GPU_UTILIZATION_ANALYSIS.md`。
3. 对 `README.md` 或 `AGENTS.md` 的更新，不在文件内写“更新历史”段落；若需说明重大规范切换，在 `docs/DOCUMENTATION_POLICY.md` 增补一条“版本里程碑”即可。
 4. 跨文件的多项相关变更以合并条目形式汇总在 `docs/decision_record.md`，集中描述动机/备选/决策/验证，避免重复叙述。

## Checkpoint / 训练恢复 | Checkpoint & Resume
（详见 `README.md` 的训练恢复章节）
1. 训练会在 `save_dir`（默认 `trained_models/runXX/`）周期性存储：
   - 序号化 checkpoint：`a3c_world{W}_stage{S}_<update>.pt` + 同名 `.json` 元数据。
   - 最新快照：`a3c_world{W}_stage{S}_latest.pt` + `.json`。
2. 自动恢复逻辑：若用户未提供 `--resume-from`，程序会扫描 `save_dir`，寻找与当前配置匹配的最高 `update` checkpoint 并加载（模型 + 优化器 + 调度器 + AMP scaler + 进度）。
3. 完整恢复包括：`model.state_dict`、`optimizer`、`scheduler`、`scaler`、`global_step`、`global_update`；缺失字段将被跳过并打印警告（未来改进：严格校验开关）。
4. 若配置改变（阶段、动作集合、模型结构等）与元数据不匹配，则拒绝恢复并提示用户手动确认。

## 命名规范 | Naming Conventions
1. 报告类：`<topic>_REPORT.md`（如 `ENV_DEBUGGING_REPORT.md`）。
2. 分析类：`<topic>_ANALYSIS.md`（如 `GPU_UTILIZATION_ANALYSIS.md`）。
3. 方案/决策：`decision_record.md`（未来可拆成 ADR 形式：`adr-XXXX-<slug>.md`）。

## 贡献流程 | Contribution Workflow
1. 新增文档：直接置于 `docs/`；若为历史归档则进入 `docs/archive/`。
2. 修改规范：更新对应主文档（`README.md` / `AGENTS.md` / 本文件），不嵌入“旧版本”文本。
3. 删除/合并：在 PR 说明中交代迁移原因，在 `docs/archive/` 保留原文件（可选）。

## 后续改进 | Next Improvements
- 若文档规模继续增长，可引入 `mkdocs` / `sphinx` 统一站点化。
- 添加 `CHANGELOG.md` 以结构化记录版本要点。
- 自动检查（pre-commit hook）避免在根目录新增非允许的 `.md`。

---
最后更新 | Last Updated: 2025-10-01
