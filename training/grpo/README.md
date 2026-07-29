# GRPO/RLVR

## 当前状态

GRPO 当前为 blocked。只有 SFT 门禁产生可度量改进、运行时兼容门禁为 ready、
reward-visible 与 held-out 物理隔离确认后，才能启动训练。

硬约束：

- rollout 使用 vLLM backend。
- `num_generations >= 8`。
- `temperature` 保持在 `0.7-1.0`。
- KL 锁定非零初值 `beta=0.01`，使用共享 frozen base 的
  `disable_adapter()` logits 作为 reference；禁止 `beta=0`。
- 使用 `clip-higher` 或等价高方差控制。
- answer reward 权重最高；format reward 必须封顶，不能主导训练。
- reward 计算必须启用缓存、批处理或离线 scoring，避免吞吐低于 rollout。
- held-out cases 不得进入 reward、调参或 early stopping。

## 显存策略

目标硬件为 Tesla V100-SXM2-32GB。单卡同时放置训练侧 QLoRA policy 和 vLLM
rollout 时，vLLM `gpu_memory_utilization` 初始控制在 `0.4-0.5`；若仍 OOM，
采用 generation/training 分时执行，不在服务 `.venv` 中安装训练依赖。

## 入口

`train_grpo.py` 当前故意 fail closed。它只在门禁未满足时退出，不启动模型、
不连接 vLLM、不读取 held-out。
