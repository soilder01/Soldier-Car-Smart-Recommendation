# G6 自动化验收报告与发布前检查

生成时间：2026-06-22 19:24:51
验收结论：通过，可进入人工验收
综合评分：92.5%
阶段通过：6/6，发布门禁：warn

## 阶段评估汇总

- ✅ F 反馈策略闭环：100.0%，报告 data/real_world/feedback_policy_evaluation.json
- ✅ G1 反馈策略稳定性：100.0%，报告 data/real_world/feedback_policy_stability_evaluation.json
- ✅ G2 Agent端到端回归：100.0%，报告 data/real_world/agent_regression_evaluation.json
- ✅ G3 真实数据治理：100.0%，报告 data/real_world/real_data_governance_evaluation.json
- ✅ G4 工程化健康检查：100.0%，报告 data/real_world/system_readiness_evaluation.json
- ✅ G5 发布门禁：100.0%，报告 data/real_world/release_gate_evaluation.json

## 发布门禁

- ✅ 工程化就绪评分：当前 100.0，阈值 >= 95
- ✅ Agent端到端回归：当前 100.0，阈值 = 100
- ✅ 真实数据治理评分：当前 93.8，阈值 >= 90
- ✅ 真实数据样本量：当前 227，阈值 >= 200
- ✅ 发布阻断风险：当前 0，阈值 = 0
- ⛔ 人工反馈正向率：当前 51.5，阈值 >= 60% 或样本<3

## 下一步动作

- 继续收集推荐反馈并复盘负反馈车型