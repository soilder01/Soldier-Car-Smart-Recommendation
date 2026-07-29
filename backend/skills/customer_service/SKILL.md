# Customer Service Skill

## 目标

面向客户咨询场景生成专业、合规、可执行的智能客服回答。

## 输入

- 客户问题
- RAG 检索证据
- Web Search 结果

## 输出

- 直接结论
- 分点解释
- 证据引用
- 风险提示
- 合规表达

## 运行入口

```text
backend/app/services/customer_service.py
CustomerServiceAgent.answer()
```

## 适用场景

- 车型参数咨询
- 新能源选购咨询
- 电池安全解释
- 充电和续航解释
- 价格和权益提醒
- 辅助驾驶边界说明
