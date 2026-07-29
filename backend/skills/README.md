# NEV Insight Skills

本目录用于描述项目内置 Skills 的标准化能力边界。每个技能独立放在一个文件夹中，并提供 `SKILL.md` 说明文件。

后端运行时代码位于：

```text
backend/app/services/skills.py
```

`skills.py` 中的 `SkillRegistry` 是实际执行入口，本目录中的 `SKILL.md` 用于教学、交付、面试讲解和后续插件化扩展。

当前 Skills：

- `profile_extraction`
- `vehicle_recommendation`
- `rag_retrieval`
- `web_search`
- `customer_service`
- `compliance_guard`
