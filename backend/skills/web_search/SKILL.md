# Web Search Skill

## 目标

为 DeepSearch 和智能客服补充公开网页搜索结果，增强实时资料覆盖能力。

## 输入

- 搜索 query
- top_k 搜索数量

## 输出

- 网页标题
- 网页链接
- 来源类型
- 证据片段

## 运行入口

```text
backend/app/services/skills.py
SkillRegistry.web_search()
```

## 降级策略

联网失败时返回空结果，并由 RAG 本地知识库继续支撑回答，保证系统可用。
