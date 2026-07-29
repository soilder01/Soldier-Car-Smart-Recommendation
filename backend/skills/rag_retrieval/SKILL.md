# RAG Retrieval Skill

## 目标

从新能源汽车知识库中检索可追溯证据，为推荐报告和客服回答提供依据。

## 输入

- 用户问题
- top_k 检索数量

## 输出

- 来源文档
- 知识领域
- 证据片段
- 语义相似度
- 关键词命中分
- 融合排序分

## 运行入口

```text
backend/app/services/skills.py
SkillRegistry.rag_retrieve()
```

## 知识源

```text
data/knowledge_base
```
