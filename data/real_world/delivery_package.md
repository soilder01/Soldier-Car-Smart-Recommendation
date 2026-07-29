# G7 版本说明与交付包

生成时间：2026-06-22 19:24:51
交付结论：可交付
交付评分：100.0%
验收评分：92.5%，阶段通过：6/6

## 版本说明

- **A-E 基础推荐、Obsidian Vault、真实数据与 Agent 主链路**：完成推荐工作台、RAG/Obsidian、真实数据采集和 Agent 编排基础能力
- **F 反馈复盘反向影响推荐**：人工反馈沉淀后参与推荐排序和解释
- **G1 反馈策略稳定性**：样本阈值、时间衰减、置信度和策略解释面板
- **G2 Agent 端到端回归**：候选池、Trace、FeedbackPolicyTool、Obsidian 写入回归
- **G3 真实数据治理**：重复、缺失、异常、来源可信度和治理动作
- **G4 工程化健康检查**：系统就绪评分、关键文件、RAG、配置和发布风险
- **G5 Agent 发布门禁**：工程就绪、回归、治理、样本和反馈统一门禁
- **G6 自动化验收报告**：汇总 F-G5 评估和门禁，生成发布前验收报告

## 关键文件

- ✅ backend/app/services/agent_orchestrator.py（38416 bytes）
- ✅ backend/app/services/feedback_policy.py（8962 bytes）
- ✅ backend/app/services/evaluation.py（16491 bytes）
- ✅ backend/app/services/real_data_governance.py（6990 bytes）
- ✅ backend/app/services/system_readiness.py（2943 bytes）
- ✅ backend/app/services/release_gate.py（4763 bytes）
- ✅ backend/app/services/acceptance_report.py（5043 bytes）
- ✅ frontend/src/App.vue（108979 bytes）
- ✅ frontend/src/api/client.ts（6372 bytes）
- ✅ frontend/src/style.css（34378 bytes）
- ✅ data/real_world/acceptance_report.md（1232 bytes）

## 交付检查

- ✅ 自动化验收通过：G6 acceptance_report accepted=true
- ✅ 后端接口可用：/api/system/acceptance-report、/api/system/release-gate、/api/system/readiness
- ✅ 前端可视化入口：系统设置页包含 G6/G5/G4 面板
- ✅ 真实数据沉淀：data/real_world/real_ev_specs.csv
- ✅ Obsidian 长期记忆：obsidian-vault
- ✅ 发布报告落盘：data/real_world/acceptance_report.md

## 使用说明

- 进入系统设置页，先查看 G6 自动化验收报告，再查看 G5 发布门禁和 G4 工程化健康检查。
- 进入 Agent 工作台输入购车需求，检查候选池选择、推荐解释、反馈策略和 Obsidian 写入。
- 进入真实数据页查看 200+ 真实车型样本、治理评分、重复/异常记录和融合推荐结果。
- 发布前执行后端编译、前端 lint、TSC、build，并确认预览页面控制台无错误。

## 后续建议

- 交付包已准备完成，可进行代码提交前最终 diff 审核
- 建议补充 README 中的阶段交付说明和本地运行步骤