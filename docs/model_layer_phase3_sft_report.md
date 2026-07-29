# Phase 3 QLoRA SFT 报告

## 状态

- 当前状态：completed
- 正式 SFT：已完成 3 epochs
- adapter reload：已验证（冻结 held-out harness 单次复评）
- 晋级结论：不通过；严格 held-out 从 6/40=15.0% 回退到 0/40=0.0%

## 前置条件

- Phase 1 原生 Qwen 基线：已完成，held-out 严格合约接地总分 6/40=15.0%；
  reward-visible 未参与。
- 教师数据：每个 intent 500 条有效正样本，五意图共 2,500 条；全量审计失败 0。
- SFT 初始固定切分为 train=2,250、eval=250；监督 token offset 审计发现
  over-length train 440/440 和 eval 47/47 均截断 assistant labels，已全部
  隔离。active train=1,810、eval=203，监督截断样本均为 0；与
  reward-visible/held-out 保留清单的 ID/query 交集仍为 0。
- active train 序列长度 p50/p95/p99/max=3864/5544/5617/5632；
  max_seq_len=5632，最长 active 样本不再截断。原最长样本 capacity probe
  仍证明 5632-token batch 在 V100 上可运行。
- 训练运行时：Python 3.11.2、PyTorch 2.3.1+cu118、CUDA 11.8、
  transformers 4.44.2、peft 0.12.0、bitsandbytes 0.43.3。

## 训练配置

- base model：`models/Qwen2.5-7B-Instruct`
- output：`checkpoints/sft`
- QLoRA：4-bit NF4、double quant、compute dtype `float16`
- 精度：`fp16=true`、`bf16=false`
- LoRA：`r=16`、`alpha=32`、`dropout=0.05`
- max sequence length：5632
- gradient checkpointing：true
- checkpoint：每 epoch 保存；按 active SFT validation `eval_loss` 最低者选择
- held-out：不参与 checkpoint selection；候选锁定后用冻结 harness 评一次

## 训练证据

- 小规模过拟合 loss：未运行
- 全量训练：5,430 micro-steps、342 optimizer updates，耗时
  25,453.6691s（约 7.07h）。
- epoch 1：train mean loss=0.6485811377，eval_loss=0.5016344750。
- epoch 2：train mean loss=0.4468086456，eval_loss=0.4639921805。
- epoch 3：train mean loss=0.3819883886，eval_loss=0.4557686771。
- 最优 checkpoint：`checkpoints/sft/checkpoint-epoch-3`，选择依据仅为最低
  active-validation eval_loss；held-out 未参与选择。
- adapter：`checkpoints/sft/best_adapter/adapter_model.safetensors`，已保存并
  重新加载完成一次 held-out 复评。
- 峰值显存：allocated=24895.9 MiB，reserved=31152.0 MiB。
- OOM：0；NaN/inf：0；训练失败日志不存在。
- 逐 micro-step 日志：`data/model_training/sft_training_steps.jsonl`（5,430 行）。
- 汇总：`data/model_training/sft_training_report.json`。
- dry-run：ready，隔离 `.venv-train` 在 V100 上完成 1 次 NF4 QLoRA
  forward/backward；loss=0.570825457572937（有限）、step=17.4041s、
  peak allocated=19181.7 MiB、peak reserved=21208.0 MiB；未保存 checkpoint。
  详细报告：`data/model_training/sft_gpu_runtime_report.json`。
- 训练前保守估算为 10.84h；真实训练主体为 7.07h。

## 评估

- 冻结 harness 评测次数：1。
- 严格总分：0/40=0.0%，相对原生基线 6/40=15.0% 回退 15.0 个百分点。
- recommend：0/10；compare：0/10；knowledge：0/10；sales：0/10。
- response schema validity：0/40。
- 40/40 失败的互斥主因均为 format/protocol；其中 invalid terminal JSON 22、
  invalid tool-call XML 7、无终态 11。
- evaluator 未发现 mandatory tool 顺序错误或参数 schema 错误，但严格终态协议
  全部失败，因此不能将工具调用子项表现包装成整体合约成功。
- 失败样本：`data/model_training/sft_heldout_failures.jsonl`（40 行）。
- 完整报告：`data/model_training/sft_heldout_report.json`；
  归因：`data/model_training/sft_heldout_failure_taxonomy.json`。
- 冻结 SFT 终态审计显示 train 1810/1810、validation 203/203 都是自然语言
  非 JSON，严格 `answer/mentioned_models` 终态监督为 0。该契约错配与本次
  schema 回退一致，但不宣称已完成唯一因果证明。

## 结论

本轮不得声称模型能力提升，也不得写“工具调用契约/结构化协议合规提升”：
冻结严格总分实际从 15.0% 降至 0.0%。可以如实表述为完成了可复现 QLoRA
训练与独立 held-out 审计，并发现训练终态监督与生产评测协议不一致导致的回退。
不得声称决策能力提升，不得声称优于云端。
