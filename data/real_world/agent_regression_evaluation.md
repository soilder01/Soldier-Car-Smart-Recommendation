# 阶段G2：Agent 端到端回归评估

生成时间：2026-07-23 15:11:45
回归通过率：100.0%
检查通过率：100.0%（6/6）
Obsidian回归报告：07-测试样例/Agent端到端回归评估-20260723-151145.md

## 检查明细

- ✅ 端到端用例不少于4条：4
- ✅ Agent回归通过率达到100%：100.0
- ✅ 所有用例覆盖候选池选择：['local', 'fused', 'real', 'fused']
- ✅ 所有用例覆盖FeedbackPolicyTool：[['ProfileParserTool', 'CandidatePoolSelectorTool', 'RankTool', 'FeedbackPolicyTool', 'EvidenceRetrievalTool', 'RiskCheckTool', 'LLMReportAgent', 'ObsidianCaseWriterTool'], ['ProfileParserTool', 'CandidatePoolSelectorTool', 'RankTool', 'FeedbackPolicyTool', 'EvidenceRetrievalTool', 'RiskCheckTool', 'LLMReportAgent', 'ObsidianCaseWriterTool'], ['ProfileParserTool', 'CandidatePoolSelectorTool', 'RankTool', 'FeedbackPolicyTool', 'EvidenceRetrievalTool', 'RiskCheckTool', 'LLMReportAgent', 'ObsidianCaseWriterTool'], ['ProfileParserTool', 'CandidatePoolSelectorTool', 'RankTool', 'FeedbackPolicyTool', 'EvidenceRetrievalTool', 'RiskCheckTool', 'LLMReportAgent', 'ObsidianCaseWriterTool']]
- ✅ 所有用例写入Obsidian长期记忆：['08-推荐案例/推荐案例-20260723-151145-比亚迪-宋PLUS-DM-i.md', '08-推荐案例/推荐案例-20260723-151145-理想-L6.md', '08-推荐案例/推荐案例-20260723-151145-Hyundai-IONIQ-5-Base.md', '08-推荐案例/推荐案例-20260723-151145-智界-R7.md']
- ✅ 回归报告写入Obsidian：07-测试样例/Agent端到端回归评估-20260723-151145.md

## 回归用例

- Agent家庭通勤本地池：100.0 分，候选池 local，Top：比亚迪 宋PLUS DM-i、理想 L6、问界 M7
- Agent无家充复杂场景：100.0 分，候选池 fused，Top：理想 L6、岚图 FREE、零跑 C11
- Agent真实数据扩展池：100.0 分，候选池 real，Top：Hyundai IONIQ 5 Base、Hyundai IONIQ 5 Base、Hyundai IONIQ 5 AWD
- Agent点名车型对比：100.0 分，候选池 fused，Top：智界 R7、腾势 N7、理想 L6