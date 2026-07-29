# Vehicle Recommendation Skill

## 目标

基于用户画像和 SQLite 车型库完成车型召回、评分排序和推荐理由生成。

## 输入

- 结构化用户画像
- 车型数据库
- 用户点名车型

## 输出

- 推荐车型 Top N
- 总推荐分
- 预算匹配分
- 续航匹配分
- 空间匹配分
- 补能匹配分
- 智驾匹配分
- 安全分
- 推荐理由
- 风险提示

## 运行入口

```text
backend/app/services/skills.py
SkillRegistry.vehicle_recall()
SkillRegistry.vehicle_rank()
```

## 设计说明

当用户明确提到车型时，系统优先召回点名车型，避免被默认预算、SUV、能源类型等表单条件带偏。
