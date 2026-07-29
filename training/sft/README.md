# QLoRA SFT

## 当前状态

首次正式 QLoRA SFT 已完成，3 个 epoch checkpoint 与最优 adapter 已保存。
冻结 held-out harness 已且仅执行一次：严格总分 `0/40=0.0%`，低于原生基线
`6/40=15.0%`，因此本轮模型不得晋级。

## 环境隔离

训练依赖来自根目录 `requirements-train.txt`，必须安装在独立训练环境中。
服务 `.venv` 必须继续保持无 CUDA torch、vLLM 和 bitsandbytes 的干净状态。

当前主机是 Tesla V100-SXM2-32GB。QLoRA compute dtype 使用 `float16`，
`fp16=true`、`bf16=false`，不以 V100 不支持的 `bfloat16` 作为训练精度。
已验证的独立环境为 `.venv-train`；关键版本锁定在
`training/sft/requirements-cu118.lock.txt`。该环境使用
`torch 2.3.1+cu118`，不修改服务 `.venv`。

## 输入输出

输入：

- `data/model_training/sft_train.jsonl`（active 1,810 条）
- `data/model_training/sft_val.jsonl`（active 203 条）

监督截断隔离：

- 原 train 中 440/440 条 over-length 样本都会截断 assistant labels，已隔离。
- validation 中额外 47 条同类样本已隔离，避免 checkpoint selection 受污染。
- 隔离行：`data/model_training/truncated_excluded.jsonl`
- 审计：`data/model_training/truncation_supervision_audit.json`

输出：

- `checkpoints/sft/`
- `docs/model_layer_phase3_sft_report.md`
- `data/model_training/sft_training_steps.jsonl`
- `data/model_training/sft_training_report.json`
- `data/model_training/sft_heldout_report.json`
- `data/model_training/sft_heldout_failures.jsonl`

配置：

- 4-bit NF4 QLoRA，double quant
- LoRA `r=16`、`alpha=32`、`dropout=0.05`
- `max_seq_len=5632`
- `gradient_checkpointing=true`
- 每 epoch 保存 checkpoint，并以 active validation `eval_loss` 最低者为候选
- 40 条 held-out 不参与 checkpoint selection，只在候选锁定后评一次

训练配置、冻结数据及评测 harness 在训练全程保持哈希不变。三个 epoch 的
active-validation eval_loss 分别为 `0.5016344750`、`0.4639921805`、
`0.4557686771`，因此选择 epoch 3。

## Dry-run

仅允许执行一次真实的本地运行时探针：

```bash
PYTHONPATH=backend:. .venv-train/bin/python \
  training/sft/train_qlora_sft.py --dry-run \
  --log-path data/model_training/sft_dry_run_gpu_report.json
```

该命令强制 `local_files_only=True`，从冻结后的训练集读取一条完整多轮工具轨迹，
加载 NF4 QLoRA 并执行一次 forward/backward。结果会写入
`data/model_training/sft_dry_run_report.json`；dry-run 不保存 adapter 或 checkpoint。

GPU dry-run 已通过：一次 forward/backward 的 loss 为有限值，峰值分配显存
`19181.7 MiB`、峰值保留显存 `21208.0 MiB`、单 step `17.4041s`。报告见
`data/model_training/sft_gpu_runtime_report.json`。冻结文本与真实
`tokenizer.apply_chat_template()` 已逐字节一致，报告见
`data/model_training/qwen_template_verification.json`。

训练启动前的最终长度 profile 将 `max_seq_len` 锁定为 `5632`：最长样本 probe
的 peak reserved 为 `28336.0 MiB`，保留约 `4174.5 MiB` 余量；p95 微步均值为
`6.2521s`。active 数据按该保守微步估算 3 epochs 为 `10.84h`。held-out
评测设置冻结在
`data/model_training/eval/frozen_qwen_heldout_harness.json`。训练启动计划见
`docs/sft_training_launch_plan.md`。

真实训练耗时约 `7.07h`，峰值 allocated/reserved 为
`24895.9/31152.0 MiB`，OOM 与 NaN/inf 均为 0。最优 adapter 位于
`checkpoints/sft/best_adapter`。

复评中 40/40 均因终态 format/protocol 失败；active train 1810 条和
validation 203 条的监督终态全部是自然语言，严格 JSON 终态样本为 0。该结果
禁止包装成模型提升、决策能力提升或优于云端。服务 `.venv` 未被训练依赖污染。
