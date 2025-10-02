# Changelog

All notable changes to this project will be documented in this file.

The format roughly follows Keep a Changelog and SemVer (future releases). Date format: YYYY-MM-DD.

## [Unreleased]
### Added
- `tests/test_shaping_smoke.py` 最小奖励塑形 & 距离增量烟雾测试，捕获首个正向位移，防止回归。
- 文档：`docs/WARNINGS_POLICY.md` 记录测试告警治理策略。
- README 增补 smoke 测试运行示例与告警抑制说明。

### Changed
- `scripts/run_2080ti_resume.sh` 支持 `LOG_INTERVAL` 环境变量，利于高频 metrics 输出与测试。
- `metrics_schema.md` 增加 smoke 测试自动化验证段落。

### Fixed
- 抑制与项目逻辑无关之 `pkg_resources` / `declare_namespace` DeprecationWarning（`pytest.ini`）。

### Security
- N/A

## [2025-10-02]
初始结构化变更前的改动详见 `docs/` 目录各报告（未按版本归档）。
