# Agentic Agent: 新能源车多 Agent 本地微调闭环

本项目把新能源汽车推荐、销售话术和复合对比问答链路，从云端 OpenAI-compatible API 迁移到本地微调的 Qwen2.5-7B-Instruct 模型，并完成数据合成、SFT、GRPO/RLVR、冻结评估、OpenAI 兼容服务化和后端切换。

核心目标不是只做一个推荐 Demo，而是证明一条可复现的模型层工程闭环：

```text
工具调用数据合成
→ QLoRA SFT
→ deterministic reward 的 GRPO/RLVR
→ 冻结 held-out / final 评估
→ adapter-on-base 本地 OpenAI-compatible 服务
→ 后端 CHAT_* 零代码切换
→ 端到端多 Agent 验证
```

## 关键结论

- 三类业务意图：`recommend`、`sales`、`composite/compare`。
- 云端基线：SeedPro ARK endpoint（仓库不包含真实 endpoint 或 key）。
- 最终上线方式：adapter-on-base 服务，不使用 merged FP16 模型。
- held-out-40：冻结、只燃烧一次，当前 `accessed=true`。
- 相对云端 held-out-40 提升：
  - recommend `+0.2983`
  - sales `+0.0769`
  - composite `+0.1876`
- 服务保真：adapter harness 贪心分与 adapter endpoint 贪心分完全一致，diff = `0.000`。

## 数据

### SFT

正式 SFT 阶段冻结 5 个意图的教师轨迹，每类 500 条：

| Source intent | Rows |
|---|---:|
| recommend | 500 |
| compare | 500 |
| customer_service | 500 |
| deep_search | 500 |
| sales | 500 |
| total | 2,500 |

经过监督截断审计后，实际输入为：

| Split | Rows |
|---|---:|
| train | 1,810 |
| validation | 203 |
| excluded by supervision truncation | 487 |

### GRPO/RLVR

- 初始 reward-visible：20 条，按意图均分为 16 train + 4 dev。
- formal v4 扩集：800 条 train + 固定 dev-4。
- sales dense v2：640 条训练混样，其中 sales dense 512 条 + recommend rehearsal 128 条。
- 冻结 held-out-40：40 条，recommend / compare / knowledge / sales 各 10 条。
- final-40：40 条，仅用于一次性选点确认。

数据和协议 SHA 见 [`docs/PROJECT_MANIFEST.md`](docs/PROJECT_MANIFEST.md)。

## 训练配置

### SFT

- Base model: `Qwen2.5-7B-Instruct`
- Method: QLoRA
- Quantization: NF4
- LoRA rank: `r=16`
- LoRA alpha: `32`
- Learning rate: `2e-4`
- Epochs: `3`
- Precision: fp16; V100/Volta 不支持 bf16

### GRPO/RLVR

- Group size: `G=8`
- KL beta: `0.01`
- Reward core:

```text
0.6 * factual_precision + 0.4 * required_coverage
```

完整 reward 函数固定在：

```text
training/grpo/reward_fn.py
SHA256: 325ad44feb83ec37c35babfed4bddb928cf400788e07735eb4631fc4af6962c8
```

## 运行环境

硬件约束：

- GPU: Tesla V100-SXM2-32GB
- Compute capability: sm_70
- Driver: 470.129.06
- `nvidia-smi` CUDA capability: 11.4
- bf16 disabled; use fp16 / NF4
- vLLM 官方 wheel 依赖 CUDA 12 runtime，本机 driver 470 无法运行，因此 vLLM 路径被排除

主要虚拟环境：

| Env | Purpose | Key versions |
|---|---|---|
| `.venv` | backend API | FastAPI, uvicorn, OpenAI client |
| `.venv-train` | QLoRA SFT | torch `2.3.1+cu118`, transformers `4.44.2`, peft `0.12.0`, bitsandbytes `0.43.3` |
| `.venv-grpo` | GRPO + adapter serving | torch `2.3.1+cu118`, transformers `4.46.3`, peft `0.12.0`, bitsandbytes `0.43.3`, FastAPI/uvicorn |
| `.venv-serve` | failed vLLM / merged-model audit env | retained for reproducibility notes, not the final serving path |

## Serving

The production local path is adapter-on-base:

```bash
PYTHONPATH=. \
.venv-grpo/bin/python -m uvicorn backend.serve_adapter:app \
  --host 127.0.0.1 --port 8000
```

Start backend with local model override:

```bash
PYTHONPATH=backend \
CHAT_BASE_URL=http://127.0.0.1:8000/v1 \
CHAT_MODEL=car-7b \
CHAT_API_KEY=<placeholder> \
.venv/bin/python -m uvicorn app.main:app \
  --host 127.0.0.1 --port 8001
```

`CHAT_*` has higher priority than `ARK_*`, so the cloud configuration can stay in the local environment for rollback while backend traffic goes to the local model. Do not commit real `CHAT_*`, `ARK_*`, `OPENAI_*`, or `.env` values.

Smoke tests:

```bash
curl http://127.0.0.1:8000/v1/models

curl http://127.0.0.1:8001/api/health

curl http://127.0.0.1:8001/api/recommend \
  -H 'Content-Type: application/json' \
  -d '{"query":"预算 25 万，家用为主，想要空间大续航长的 SUV","top_k":5,"use_deep_search":false}'
```

## 合并劣化反例

`merged_model/` 保留但不作为最终 serving path。

原因：

- 将 LoRA merge 成 FP16 全量模型后，endpoint 贪心对照与 adapter harness 不一致。
- recommend / sales 的同口径贪心分出现明显差异。
- 改回 adapter-on-base 后，adapter harness 与 endpoint adapter 贪心分完全一致：

| Path | recommend | sales | composite |
|---|---:|---:|---:|
| adapter harness | 0.900 | 0.755 | 0.8275 |
| adapter endpoint | 0.900 | 0.755 | 0.8275 |
| diff | 0.000 | 0.000 | 0.000 |

因此最终部署选择 NF4 base + adapter runtime，而不是 merge 后 FP16 模型。

## Evaluation

Held-out-40 terminal evaluation:

| Object | recommend | sales | composite |
|---|---:|---:|---:|
| `sales_dense_v2_checkpoint_150` | 0.8458 | 0.6563 | 0.7510 |
| `checkpoint-300` | 0.6582 | 0.5756 | 0.6169 |
| `cloud_seedpro_ark_ep_masked` | 0.5475 | 0.5794 | 0.5634 |

Delta versus cloud:

| Metric | Delta |
|---|---:|
| recommend | +0.2983 |
| sales | +0.0769 |
| composite | +0.1876 |

Endpoint retest through adapter service:

| Protocol | recommend | sales | composite |
|---|---:|---:|---:|
| greedy parity | 0.9000 | 0.7550 | 0.8275 |
| sampling retest | 0.7942 | 0.6531 | 0.7237 |

The sampling retest differs from the original sampled held-out score because endpoint retest resets seed per prompt request, while the original adapter-local evaluation consumes one continuous RNG stream across the evaluation run. Greedy parity is the serving-path equivalence check.

## Directory Layout

```text
backend/                 FastAPI backend, Agent graph, OpenAI-compatible local serving
data_synth/              Synthetic data generation, schema and freeze helpers
scripts/                 Freeze/eval/deployment helper scripts
training/sft/            QLoRA SFT training and SFT held-out scripts
training/grpo/           Reward function, GRPO runs, endpoint retests
docs/                    Design docs, reports, reproducibility manifest
tests/                   Model-layer and backend tests
data/                    Vehicle DB, knowledge base, model-training artifacts
checkpoints/             Local adapters/checkpoints; not for ordinary Git
models/                  Base model; not for ordinary Git
merged_model/            Retained merge-degradation audit artifact; not for ordinary Git
```

## Security And Repository Hygiene

Do not commit:

- `.env`
- real API keys / tokens / endpoint IDs
- `models/`
- `checkpoints/`
- `merged_model/`
- virtual environments
- raw logs or generated answer text
- large `data/model_training/` intermediates

Commit only source code, docs, small sanitized manifests/protocols, and explicit SHA manifests.

Before commit:

```bash
git status --ignored --short
```

Confirm `.env`, model weights, checkpoints, merged weights, venvs, caches, logs, and large training artifacts are ignored.

## Frozen Benchmark Use

`held_out_40_frozen_eval.jsonl`, `grpo_final_held_out.jsonl`, and the
corresponding final-40 files are terminal frozen benchmarks for this project.
They stayed read-only during training and were burned exactly once in terminal
evaluation (`held_out_40_accessed=true`). Reproducers may use them only for
terminal verification, not for training, hyperparameter selection, checkpoint
selection, or early stopping, so the benchmark remains fair.
