# Changelog

All notable changes to this project will be documented in this file.

The format roughly follows Keep a Changelog and SemVer (future releases). Date format: YYYY-MM-DD.

## [Unreleased]
### Added
 - 新增 `tests/test_adaptive_injection_integration.py` 验证自适应 distance_weight 运行期写回对 shaping raw 的放大效果。
 - MarioRewardWrapper 增加 runtime setter: `set_distance_weight()` 与 `get_diagnostics()`，支持自适应调度即时注入与外部诊断。
 - metrics 新增字段：`adaptive_distance_weight_effective`（已存在） + `wrapper_distance_weight`（真实 wrapper 当前值，便于对比 shadow / effective 差异）。

### Changed
- 统一回退到单一配置模块 `src/config.py`，移除中间 shim（config.py / app_config.py），消除 mypy duplicate module 根因。
- 调整测试路径加载逻辑，在 `tests/conftest.py` 增加项目根路径注入，确保 `import src.*` 稳定。
 - 自适应 distance_weight 逻辑从仅记录 shadow 改为实时写回所有底层 `MarioRewardWrapper`；README / decision_record 文档同步更新。

### Known Issues
 - 在 fc_emulator backend 场景下 `env_positive_dx_ratio` 可能长期为 0（stagnation_steps 未清零），导致自适应调度持续向上；已列入 ROADMAP 后续修复（预期通过明确 dx>0 时重置 stagnation_steps & 直接统计正增量 env）。
 - `env_shaping_raw_sum` 在 fc_emulator 模式返回 0 的场景需追加 dx 诊断（计划添加 debug 标志 `MARIO_SHAPING_DEBUG=1` 输出 dx 解析路径）。

### Technical Debt / Next
- 仍有大量 env / utils / models 类型注解待补；当前保持功能优先，计划后续分阶段加注并拆分超长文件。

### Added
- 新分析辅助脚本：`scripts/inspect_metrics.py` 用于快速最近进展摘要。
- 新 CI 烟雾测试：`tests/test_progression_metrics.py` 验证关键进展指标的存在性与单调性。

### Changed
- `scripts/run_2080ti_resume.sh` 支持 `LOG_INTERVAL` 环境变量，利于高频 metrics 输出与测试。
- `metrics_schema.md` 增加 smoke 测试自动化验证段落。
- `train.py` 新增自动前进回退机制（`--auto-bootstrap-*`），`scripts/run_2080ti_resume.sh`/`auto_phase_training.sh` 默认注入更强冷启动参数，处理距离增量始终为 0 的场景。
- 扩展 RewardConfig 支持 distance_weight 与 death_penalty 线性退火；日志统计增加推进覆盖率与 plateau 诊断输出；加入二次脚本注入与里程碑奖励逻辑。

### Fixed
- 抑制与项目逻辑无关之 `pkg_resources` / `declare_namespace` DeprecationWarning（`pytest.ini`）。

### Security
- N/A

## [2025-10-02]
初始结构化变更前的改动详见 `docs/` 目录各报告（未按版本归档）。
