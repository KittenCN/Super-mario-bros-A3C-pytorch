# Changelog

All notable changes to this project will be documented in this file.

The format roughly follows Keep a Changelog and SemVer (future releases). Date format: YYYY-MM-DD.

## [Unreleased]
### Added
- 新增 `tests/test_shaping_smoke.py` 覆盖 fc_emulator 聚合路径，确保奖励塑形产生非零 `shaping_raw_sum` 与推进占比；`tests/test_adaptive_injection_integration.py` 继续验证 distance_weight 运行期注入。
- `MarioRewardWrapper` 增添 `set_distance_weight()`、`get_diagnostics()`、`_coerce_scalar()` 等接口，并在 `MARIO_SHAPING_DEBUG=1` 下限频输出 `[reward][warn] dx_missed ...` 提示。
- 新稳态训练脚本：`scripts/train_stable_sync.sh`（同步调试配置）与 `scripts/train_stable_async.sh`（异步 + overlap 压力测试），支持 `--dry-run` 与环境变量覆盖。

### Changed
- `train.py` 引入 `env_progress_dx`、`stagnation_envs`、`stagnation_mean_steps`、`adaptive_ratio_fallback` 等指标，修复 `env_positive_dx_ratio` 在 fc_emulator 场景恒为 0 的问题。
- metrics JSONL/TensorBoard 现同步输出 `wrapper_distance_weight`、`adaptive_distance_weight_effective`，便于对比 shadow 与实际写回的距权。
- README、docs/decision_record.md、docs/ROADMAP.md、docs/PROJECT_ANALYSIS.md、docs/ENV_DEBUGGING_REPORT.md 更新以反映推进修复、诊断管线与新脚本。

### Fixed
- `env_shaping_raw_sum` / `env_shaping_scaled_sum` 在 fc_emulator 模式保持 0 的问题，通过统一标量拉平与推进指标修复。
- YAML/JSON 阶段脚本中遗漏的 GPU 可用性提示，`--device auto` 无 CUDA 时会警示或阻断。

### Known Issues
- `AsyncVectorEnv` 在部分平台仍可能初始化超时（详见 `docs/ENV_DEBUGGING_REPORT.md`），需继续跟踪 NES 原生阻塞原因。

### Technical Debt / Next
- 仍有大量 env / utils / models 类型注解待补；当前保持功能优先，计划后续分阶段加注并拆分超长文件。

### Security
- N/A

## [2025-10-02]
初始结构化变更前的改动详见 `docs/` 目录各报告（未按版本归档）。
