# Compliance Guard Skill

## 目标

检查并修正新能源汽车销售和客服回答中的风险表达。

## 检查重点

- 避免“绝对安全”
- 避免“自动驾驶”
- 避免“续航不打折”
- 避免“保证最低价”
- 避免贬低竞品
- 避免夸大政策和权益

## 运行入口

```text
backend/app/services/skills.py
SkillRegistry.compliance_check()
```

## 输出

合规修正后的文本，以及 Skill Trace 中的检查记录。
