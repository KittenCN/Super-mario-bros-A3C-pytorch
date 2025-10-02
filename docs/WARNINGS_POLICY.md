# 测试警告治理策略 | Test Warning Governance Policy

## 目标 | Goals
- 保持 `pytest` 输出聚焦功能性/逻辑失败，避免被第三方库的弃用告警淹没。
- 记录被抑制的告警来源与理由，方便后续升级依赖时回溯。

## 当前抑制 | Current Suppressions
| Warning Type | Source Package | Pattern | Reason | Action Plan |
|--------------|----------------|---------|--------|------------|
| DeprecationWarning | pkg_resources (setuptools) | `pkg_resources is deprecated` | 外部依赖 pygame 间接使用，短期不可控，对训练逻辑无影响 | 后续依赖升级或 pygame 移除后复查 |
| DeprecationWarning | pkg_resources.declare_namespace | `Deprecated call to \`pkg_resources.declare_namespace` | 同上 | 与上条合并复查 |

详见 `pytest.ini` 中 `filterwarnings` 配置。

## 原则 | Principles
1. 仅抑制可确认与本项目功能正确性无关、且短期无法由我们直接修复的第三方弃用告警。
2. 不抑制自研模块发出的警告；若出现应视为潜在技术债并开 issue。
3. 每次新增抑制需在此文件追加记录并说明后续处置计划。
4. 若上游版本发布修复（告警消失），应移除对应过滤条目保持最小配置。

## 未来工作 | Future Work
- 监控 pygame / nes-py 版本演进，评估是否去除对 `pkg_resources` 的依赖链。
- 引入 `warnings.simplefilter("error", SomeCategory)` 针对关键路径自定义告警（如性能退化指标）以提前捕获回归。

