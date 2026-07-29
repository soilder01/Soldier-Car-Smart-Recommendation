# 阶段F：反馈策略反向影响 Agent 推荐评估

生成时间：2026-06-22 17:29:48
通过率：100.0%（7/7）
真实 Agent 链路应用规则数：2

## 检查明细

- ✅ 反馈策略生成正负车型规则：['小鹏 G6', '特斯拉 Model Y', '理想 L6', '比亚迪 宋PLUS DM-i', 'Hyundai IONIQ 5']
- ✅ 策略函数可对负反馈车型降权：-16.0
- ✅ 策略函数可对正反馈车型加权：1.0
- ✅ Agent Trace 包含 FeedbackPolicyTool：['ProfileParserTool', 'CandidatePoolSelectorTool', 'RankTool', 'FeedbackPolicyTool', 'EvidenceRetrievalTool', 'RiskCheckTool', 'LLMReportAgent', 'ObsidianCaseWriterTool']
- ✅ 真实 Agent 推荐结果已被反馈策略改分：5
- ✅ API 返回反馈策略应用规则：2
- ✅ 解释性报告包含反馈策略：{'applied_rules': [{'target': '理想 L6', 'delta': -0.4, 'score_before': 93.7}, {'target': '智界 R7', 'delta': -4.0, 'score_before': 94.5}, {'target': '腾势 N7', 'delta': -4.0, 'score_before': 94.4}, {'target': '问界 M5', 'delta': -4.0, 'score_before': 93.6}, {'target': 'Faraday Future 91', 'delta': -4.0, 'score_before': 82.4}]}

## 策略函数分数变化

- 小鹏 G6：84.0 → 85.0（+1.0）
- 理想 L6：82.0 → 81.6（-0.4）
- 特斯拉 Model Y：88.0 → 72.0（-16.0）

## 真实 Agent 推荐分变化

- 理想 L6：93.7 → 93.3（-0.4）
- 智界 R7：94.5 → 90.5（-4.0）
- 腾势 N7：94.4 → 90.4（-4.0）
- 问界 M5：93.6 → 89.6（-4.0）
- Faraday Future 91：82.4 → 78.4（-4.0）