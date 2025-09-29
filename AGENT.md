# agent.md — Codex 执行代理规范（中文版）

> 目标：让 Codex（或同类代码生成模型）**全程用中文沟通**，在**最少追问**的前提下**自动完成需求**、**自测并保证可运行**、**交付可维护成果**。

---

## 0. 沟通与产出原则

1. **语言**：所有交流、注释、提交信息、README 一律使用**简体中文**（必要时可附英文术语）。
2. **默认动作**：遇到可合理假设的细节，**直接做出专业默认值并继续实现**；将假设写入 `ASSUMPTIONS.md`。
3. **最小打扰**：除非风险极高，**不等待用户确认**；完成后在交付物中标注可选项与替代方案。
4. **可运行为王**：你生成的一切代码、脚本、配置，必须**能在本地一键运行与测试通过**。
5. **可复现**：输出**固定的环境说明与锁定依赖**（如 `requirements.txt`/`poetry.lock`、`package-lock.json` 等）。
6. **安全优先**：默认**不开启**危险权限（写系统目录、外网删除/高危命令等）；对外部调用使用**显式白名单**与可配置开关。

---

## 1. 交付物目录结构（通用模板）

```
project-root/
├─ src/                         # 业务源码
├─ tests/                       # 单元/集成测试
├─ examples/                    # 最小可运行示例、用法样例
├─ scripts/                      # 任务脚本（启动/构建/发布/数据等）
├─ config/                       # 配置模板：*.yaml / *.json / .env.example
├─ docs/                         # 设计说明、API 文档、图表
├─ .github/workflows/            # CI（pytest/linters/build）
├─ Dockerfile                     # 可选：容器化运行
├─ docker-compose.yml             # 可选：本地依赖编排（DB/Cache/Bus）
├─ pyproject.toml / package.json / requirements.txt
├─ Makefile                        # 一键任务入口（见下）
├─ README.md                       # 使用说明（中文）
├─ ASSUMPTIONS.md                   # 实现时的假设与权衡
├─ CHANGELOG.md                     # 语义化变更日志
└─ agent_report.md                  # 本次自动执行报告（见 §8）
```

---

## 2. 一键任务（Makefile 约定）

提供以下**标准目标**（任意语言/栈均保持同名入口；内部调用对应工具）：

```
make setup        # 安装依赖、初始化钩子、生成本地配置(.env)
make fmt          # 代码格式化（black/ruff 或 prettier/eslint --fix）
make lint         # 静态检查（ruff/mypy/eslint/typescript/sonarlint）
make test         # 运行全部测试（含覆盖率报告）
make run          # 启动应用或示例（examples/ 下的最小可运行案例）
make build        # 构建产物（wheel、docker image、前端打包等）
make ci           # 本地模拟 CI：lint + test + build
```

> 要求：**所有 PR/交付前必须通过 `make ci`**。如无需 Docker，则 `build` 可为可分发包或可执行文件。

---

## 3. 需求到实现的自动流程（你应严格遵循）

1. **解析需求**  
   - 抽取功能点、接口、数据结构、约束、非功能需求（性能/安全/合规/可观测）。  
   - 识别**歧义**并**自拟合理默认值**（写入 `ASSUMPTIONS.md`）。
2. **制定方案**  
   - 产出简要架构：模块边界、依赖、关键数据流/时序图（可用 Mermaid）。  
   - 列出替代方案（2–3 个），并在 `docs/decision_record.md` 中给出取舍。
3. **脚手架落地**  
   - 生成目录与基础文件（§1），补齐 `README.md` 的**一键运行**指引。  
   - 写好 `config/*` 模板和 `.env.example`，**禁止把真实密钥入库**。
4. **实现编码**  
   - 遵循**整洁代码**与**中文注释**；公共函数写**docstring**与示例。  
   - 提供**最小可运行示例**于 `examples/`。
5. **自测优先**  
   - 为每个模块编写**单元测试**（边界/异常/典型路径）。  
   - 为关键流程编写**集成测试**（含假外部依赖的 mock）。  
   - 覆盖率**≥80%**（允许对不可测分支豁免并在测试中注明原因）。
6. **质量闸门**  
   - 通过 `make fmt && make lint && make test`。  
   - 产生**覆盖率报告**与**构建产物**（wheel/docker/前端包）。  
   - 产出 `agent_report.md`（§8）。
7. **交付收尾**  
   - 更新 `CHANGELOG.md`（遵循**语义化版本**）。  
   - 在 `README.md` 增补**常见问题**与**故障排查**。

---

## 4. 代码与文档规范

- **风格**：  
  - Python：`ruff` + `black` + `mypy`（启用严格模式）。  
  - TS/JS：`eslint`（typescript-eslint）+ `prettier`。  
  - Go：`gofmt` + `golangci-lint`。  
- **日志**：统一 `logger` 封装，分级输出，默认**不打印敏感信息**。  
- **配置**：优先 `config.yaml` + `.env`；代码中读取**必须带默认值**与**类型校验**。  
- **错误处理**：外层统一错误边界，返回**中文可读**的错误信息与建议。  
- **注释**：中文为主，必要处附英文术语；公开 API 必有使用示例。  
- **提交信息**：遵循 Conventional Commits（`feat:`、`fix:`、`docs:` …）。

---

## 5. 测试策略（必须执行）

1. **单元测试**：核心算法、数据转换、边界/异常。  
2. **集成测试**：端到端最小流（含 I/O、DB/Cache、外部 API 的 mock）。  
3. **回归样例**：对已修复的 bug 补充最小复现测试。  
4. **性能冒烟**（可选）：关键路径做小规模基准，记录指标与阈值。  
5. **覆盖率目标**：`--cov=src --cov-report=xml`，行覆盖 ≥80%。

---

## 6. CI/CD（最小流水线）

- 触发：`push` 与 `pull_request`。  
- 作业：  
  1. **Setup**：拉依赖、缓存。  
  2. **Static**：`make fmt && make lint`。  
  3. **Test**：`make test`（上传覆盖率）。  
  4. **Build**：`make build`。  
- 产物：上传构建包/镜像、测试报告、覆盖率。  
- 失败门槛：任一作业失败则阻断合入。

---

## 7. 安全与合规

- 严禁硬编码密钥、Token、私钥；使用 `.env` + 密钥管理（如 GitHub Secrets）。  
- 外部请求默认**超时/重试/熔断**；域名/地址走白名单配置。  
- 文件操作限制在 `project-root/` 内；删除/覆盖前先备份。  
- 记录必要审计日志（变更、关键操作、失败栈）。  

---

## 8. 自动执行报告（`agent_report.md` 模板）

```
# 本次自动执行报告

## 需求摘要
- 背景与目标：
- 核心功能点：

## 关键假设（详见 ASSUMPTIONS.md）
- …

## 方案概览
- 架构与模块：
- 选型与权衡：

## 实现与自测
- 一键命令：`make setup && make ci && make run`
- 覆盖率：xx%
- 主要测试清单：单测 N 项、集成 M 项
- 构建产物：…

## 风险与后续改进
- 已知限制：
- 建议迭代：
```

---

## 9. 与外部系统/数据交互（约定）

- **HTTP**：统一 `client` 封装（重试、超时、重定向、错误码翻译、指标采集）。  
- **数据库**：迁移脚本（如 Alembic / Prisma），**本地容器化**依赖（`docker-compose`）。  
- **消息/缓存**：提供本地 mock 或容器服务（Kafka/Redis/RabbitMQ）。  
- **文件**：输入输出路径以 `config/` 和 `.env` 可配置；**不要硬编码绝对路径**。

---

## 10. 典型命令与脚本（示例）

- `scripts/bootstrap.sh`：环境检查、依赖安装、生成 `.env`。  
- `scripts/dev_run.sh`：本地热重载启动（后端/前端）。  
- `scripts/seed_data.py`：样例数据灌入。  
- `scripts/release.sh`：版本号、CHANGELOG、打包、推镜像。

所有脚本：  
- 具备 `-h/--help`；  
- 失败时**非零退出码**；  
- 打印**中文提示**与下一步建议。

---

## 11. 文档要求

- `README.md` 必含：  
  - 项目简介、功能清单、快速开始（3 条以内命令）、配置说明、常见问题。  
- `docs/`：  
  - `architecture.md`（Mermaid 图）、`api.md`（接口定义）、`ops.md`（监控/告警/日志）。  
- 变更：更新 `CHANGELOG.md`（Keep a Changelog + SemVer）。

---

## 12. 定义“完成”（Definition of Done）

- [ ] `make ci` 全绿（格式、静态检查、测试、构建）。  
- [ ] 本地 `make run` 可运行最小示例。  
- [ ] 覆盖率 ≥80%，关键路径有集成测试。  
- [ ] README/ASSUMPTIONS/CHANGELOG/agent_report 已更新。  
- [ ] 配置可通过 `.env` 切换环境，无敏感信息入库。  
- [ ] 日志、错误信息**中文可读**，含排错建议。  

---

## 13. 你应如何回应用户的新需求（回复模板）

> **始终用中文简洁回答**，并在需要时直接给出可运行的命令/代码块。

1. **概述**：复述需求要点 + 关键假设（如有）。  
2. **交付**：直接给出新增文件/补丁（`diff` 或完整文件），同步更新测试与文档。  
3. **运行**：提供 1–3 条命令即可验证。  
4. **结果**：说明自测范围、覆盖率或关键截图/日志。  
5. **后续**：可选优化项与影响评估。

---

## 14. 最小示例（占位，按项目替换）

- 运行：  
  ```bash
  make setup
  make ci
  make run
  ```
- 若失败：查看 `agent_report.md` 的**故障排查**章节，并执行 `make test -k failing_case` 复现。

---

> **执行承诺**：除非明确要求暂停，Codex 将按本规范**默认推进至“可运行 + 已自测 + 可交付”状态**再输出结果。
