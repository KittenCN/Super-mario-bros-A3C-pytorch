# 文档迁移记录 | Documentation Relocation Log

**更新时间 | Update Date**: 2025-09-29

为保持仓库根目录整洁，所有松散的 Markdown 文档已集中到 `docs/` 目录，并在此保留迁移说明，便于团队追溯历史路径。<br>To keep the repository root tidy, previously scattered Markdown documents were consolidated into the `docs/` directory; this note preserves their relocation history for future reference.

## 已迁移文件 | Files Relocated
- `PROJECT_ANALYSIS.md` → `docs/PROJECT_ANALYSIS.md`：汇总项目工作原理分析内容。<br>`PROJECT_ANALYSIS.md` → `docs/PROJECT_ANALYSIS.md`: centralises the project working-principle analysis.
- `PROJECT_OPTIMIZATION_PLAN.md` → `docs/PROJECT_OPTIMIZATION_PLAN.md`：集中优化计划与路线图。<br>`PROJECT_OPTIMIZATION_PLAN.md` → `docs/PROJECT_OPTIMIZATION_PLAN.md`: groups optimisation plans and roadmap materials.
- `GPU_UTILIZATION_ANALYSIS.md` → `docs/GPU_UTILIZATION_ANALYSIS.md`：统一 GPU 利用率调研报告。<br>`GPU_UTILIZATION_ANALYSIS.md` → `docs/GPU_UTILIZATION_ANALYSIS.md`: keeps the GPU utilisation study alongside other documentation.
- `AGENT.md` → `docs/AGENT.md`：为代理执行规范提供独立文档空间。<br>`AGENT.md` → `docs/AGENT.md`: dedicates space to the agent execution guidelines.

## 迁移目的 | Rationale
- 减少根目录文件数量，提升仓库导航效率。<br>Reduce the number of top-level files to make repository navigation easier.
- 让项目文档集中呈现，方便批量更新与版本化管理。<br>Keep all project documentation co-located to simplify bulk updates and version tracking.
- 明确告知历史路径，避免脚本或引用因路径改变而失效。<br>Document historic locations so that scripts or references can be updated without confusion.

## 后续提示 | Follow-up Notes
- 如自动化脚本仍引用旧路径，请同步修改为 `docs/` 下的新位置。<br>If automation scripts still reference the previous paths, update them to point to the new `docs/` locations.
- 对于新增 Markdown 文件，建议直接放置在 `docs/` 中并在此文档补充记录。<br>For any new Markdown artefacts, place them under `docs/` and append an entry here.
