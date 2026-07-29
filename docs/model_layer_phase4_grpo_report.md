# Phase 4 GRPO/RLVR 报告

## 状态

- 当前状态：blocked
- 阻断原因：SFT 晋级门禁未完成；vLLM/训练运行时门禁未 ready；GRPO 数据与
  reward 仅有骨架。

## 前置门禁

- Phase 0 reward-visible / held-out 划分：已物理拆分，训练不得读取 held-out。
- Phase 3 SFT 改进证据：未完成。
- 运行时兼容：`overall_gate=blocked_until_verified`。

## Rollout

- backend：vLLM
- num_generations：8
- temperature：0.8
- top_p：0.95

## Reward

- answer reward：primary，当前只允许 reward-visible ID，held-out 返回 0。
- format reward：封顶 1.0，仅校验 OpenAI tool-call 结构和 approved schema。
- tool execution reward：必须有一一对应、无 error 的 tool result。
- reward cache：stable SHA256 key，防外部 mutation。
- reward 吞吐：正式训练前必须记录缓存命中率、批处理规模和每秒样本数。

## 显存策略

- 同卡并发 / 分时执行：优先 vLLM `gpu_memory_utilization=0.4-0.5`，OOM 时
  切换 generation/training alternating。
- OOM 处理：不得降级为 mock reward 或跳过真实 rollout。

## Held-out 隔离确认

held-out cases 未用于 reward、调参、early stopping。最终 held-out 只允许
Phase 6 四后端对比读取。

## 结论

未完成 held-out 对比前，不得声明最终模型提升。
