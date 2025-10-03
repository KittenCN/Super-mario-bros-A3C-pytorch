# Changelog

All notable changes to this project will be documented in this file.

The format roughly follows Keep a Changelog and SemVer (future releases). Date format: YYYY-MM-DD.

## [Unreleased]
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
