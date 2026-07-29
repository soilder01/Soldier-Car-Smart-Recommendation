# 模型层最终对比报告

## 评估隔离

- reward-visible cases：20（仅训练期奖励、调参和诊断可见）
- held-out cases：40（仅 Phase 6 最终验收读取）
- 本报告主要结论只基于 held-out cases。
- 当前状态：模板已建立，四后端真实评测尚未执行。

## 四列对比

| 后端 | 工具选择准确率 | 参数合法率 | 推荐命中率 | 幻觉率 | 端到端时延 | agent regression | release gate |
|---|---:|---:|---:|---:|---:|---:|---|
| Ark baseline | 未评估 | 未评估 | 未评估 | 未评估 | 未评估 | 未评估 | 未评估 |
| 原生 Qwen2.5-7B via vLLM | 未评估 | 未评估 | 未评估 | 未评估 | 未评估 | 未评估 | 未评估 |
| QLoRA SFT Qwen2.5-7B | 未评估 | 未评估 | 未评估 | 未评估 | 未评估 | 未评估 | 未评估 |
| GRPO Qwen2.5-7B | 未评估 | 未评估 | 未评估 | 未评估 | 未评估 | 未评估 | 未评估 |

## 计分口径

- 每个指标必须记录分子、分母、ratio 和 percentage。
- `tool_selection_accuracy`、`argument_validity`、`recommendation_hit_rate`
  和 `hallucination_rate` 使用 `scripts/evaluate_model_outputs.py` 的
  held-out 输出。
- `hallucination_rate` 只覆盖终态已声明车型实体，不夸大为自由文本全量检测。
- 任一 runner error 都必须保留在分母中，不能让失败 case 消失。

## 结论规则

未完成 held-out 四列对比前，不声明本地模型优于云端，也不声明 SFT/GRPO 有
最终模型提升。训练期 reward-visible 结果只能作为调试信号，不得替代最终验收。
