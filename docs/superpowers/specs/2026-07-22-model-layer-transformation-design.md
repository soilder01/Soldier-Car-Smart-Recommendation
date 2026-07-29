# 模型层改造设计

日期：2026-07-22

状态：已确认方向：方案 A

## 1. 目标

将当前 `Soldier-Car-Smart-Recommendation` 项目从单纯的 LangGraph 应用系统，升级为“应用层 + 本地模型层”的完整项目。

应用层必须保持稳定。本次改造新增本地 `Qwen2.5-7B` 模型链路，包括 vLLM 服务、工具调用数据合成、QLoRA SFT、GRPO/RLVR 强化、模型后端切换和对比评估。

核心工程原则是渐进证明：

```text
当前应用基线
  -> 原生 Qwen2.5-7B 通过 vLLM 接入
  -> SFT 数据合成
  -> QLoRA SFT
  -> GRPO/RLVR
  -> 模型后端对比报告
```

Phase 1 必须在训练开始前产出一个可演示里程碑。

## 2. 当前项目事实

当前项目是一个真实的新能源汽车推荐系统，技术栈包括 FastAPI、LangGraph、Vue3、SQLite、RAG 和 Obsidian。

当前应用栈：

- 后端：FastAPI、LangGraph、LangChain-compatible tools、SQLite、scikit-learn。
- 前端：Vue3、Element Plus、ECharts、animejs、GSAP。
- RAG：基于 `data/knowledge_base/` 和 `obsidian-vault/` 的本地 TF-IDF + BM25-like 检索。
- 长期知识库：Obsidian Markdown vault。
- 评估：多个 `scripts/evaluate_*.py` 脚本。
- 现有对话客户端：`backend/app/services/llm_client.py` 中的 OpenAI-compatible client。

当前模型集成事实：

- `backend/app/config.py` 暴露 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`CHAT_MODEL`、`EMBEDDING_MODEL`、`TEMPERATURE` 和 `TIMEOUT`。
- `backend/app/services/llm_client.py` 使用 `OPENAI_BASE_URL` 构造单一 OpenAI-compatible client。
- 当前没有独立 embedding client。
- 当前 RAG 不调用 embedding API，而是使用本地 scikit-learn 向量化。
- 后续模型/后端切换仍必须拆分 chat 与 embedding 配置，避免 vLLM chat endpoint 误覆盖未来的 embedding 服务。

`backend/app/services/agent_graph.py` 中当前真实工具为：

1. `extract_user_profile(query: str, budget_max: int = 0) -> str`
2. `search_and_rank_vehicles(budget_max: int = 0, preferred_type: str = "", preferred_energy: str = "", concerns: str = "", top_k: int = 5) -> str`
3. `retrieve_knowledge_base(query: str) -> str`
4. `search_web_info(query: str) -> str`
5. `generate_sales_talk(budget_max: int = 0, concerns: str = "", top_model: str = "") -> str`

除非项目后续明确新增工具，否则 SFT 和 GRPO 的工具调用 schema 只能围绕这 5 个工具。

## 3. 设计决策

采用方案 A：完整 Phase 0-6 改造，并将 Phase 1 作为第一个可对外演示的里程碑。

已拒绝的方案：

- 训练优先路线：拒绝。原因是缺少原生 Qwen 基线，并且如果 vLLM tool calling 后续不通，会浪费训练算力。
- 纯工程切换路线：拒绝。原因是无法支撑 QLoRA + GRPO 本地模型层的目标。

最终设计保留完整长期目标，同时强制先完成早期集成证明。

## 4. 非目标

本设计不重写推荐业务逻辑。

本设计首轮不替换本地 TF-IDF RAG 管线。

本设计不要求将所有模型资产纳入 git。

本设计不为展示效果虚构更大的项目数字。文档和简历必须使用真实数字，除非代码或数据确实被扩展。

## 5. 全局约束

- 禁止提交私密文件 `backend/config/config.yaml`。
- 优先使用本地模型路径，不默认从 HuggingFace 下载。
- 所有 chat 模型后端统一使用 OpenAI-compatible API。
- 服务依赖与训练依赖必须分离。
- 除非提供明确兼容层，否则保持现有应用 API 稳定。
- 工具 schema 必须与真实工具名、参数名逐字对齐。
- 训练前必须记录基线指标。
- SFT 产生可度量工具调用改进前，不允许启动 GRPO。
- 格式奖励不得压过答案质量奖励。
- 训练奖励与最终评估必须隔离。GRPO 可以使用 reward-specific evaluation subsets，但 Phase 6 必须使用训练期间策略从未接触过的 held-out 测试集和独立评估路径。
- 不允许将同一批 `evaluate_*.py` 用例同时作为 GRPO answer reward 和最终项目证明。
- 现有 `scripts/evaluate_*.py` 主要作为应用层与工程门禁；模型质量最终证明必须使用新增的独立模型输出 held-out 数据集和专用评测脚本。
- 当前硬件是 Tesla V100-SXM2-32GB，训练计算精度使用 FP16；不得使用 V100 不支持的 BF16 作为默认训练配置。
- 没有对比评估证据，不得声称模型提升。

## 6. 目标架构

改造后的系统分为三层。

应用层：

- 现有 FastAPI endpoints。
- 现有 LangGraph agents。
- 现有 5 个工具函数。
- 现有 RAG、Obsidian、反馈策略和评估脚本。

模型服务层：

- Ark 后端，用作当前云端基线。
- 原生 `Qwen2.5-7B` vLLM 后端。
- SFT 后 `Qwen2.5-7B` vLLM 后端。
- GRPO 后 `Qwen2.5-7B` vLLM 后端。
- 通过显式配置或脚本完成后端切换。

训练层：

- 工具 schema 导出器。
- SFT 合成数据生成器。
- 工具数据校验器。
- QLoRA SFT 训练配置。
- GRPO/RLVR 奖励函数和训练入口。
- 评估与对比报告。

后端切换必须可见、可复现。用户应能在不修改业务代码的前提下，对比 `ark`、`qwen_base_vllm`、`qwen_sft_vllm` 和 `qwen_grpo_vllm`。

## 7. 配置设计

当前单一 `OPENAI_BASE_URL` 对旧的云端单后端配置足够，但对模型层改造过粗。

本设计引入明确的 chat 与 embedding 配置概念：

- Chat backend：用于 LangGraph agents 和 OpenAI-compatible chat completions。
- Embedding backend：预留给未来 embedding API 或独立 bge 服务。

实施计划需要定义具体变量名和迁移行为。目标形态为：

```text
CHAT_API_KEY
CHAT_BASE_URL
CHAT_MODEL
EMBEDDING_API_KEY
EMBEDDING_BASE_URL
EMBEDDING_MODEL
```

必须保留对现有变量的兼容：

- 现有 `ARK_API_KEY`、`ARK_BASE_URL` 和 `ARK_CHAT_MODEL` 继续作为 chat fallback。
- 现有 `ARK_EMBEDDING_MODEL` 继续作为 embedding model 选择 fallback。
- `OPENAI_API_KEY`、`OPENAI_BASE_URL` 和 `OPENAI_CHAT_MODEL` 继续作为 legacy chat fallback。

Phase 1 不得假设 vLLM 提供 embeddings。如果后续需要 embedding，则继续使用云端 embedding 或单独部署 embedding 服务。

当前设计中的 `EMBEDDING_*` 设置是预留项，因为现有 RAG 走本地 TF-IDF/BM25-like 检索，不调用 embedding API。配置样例必须标注 embedding 设置为“预留”或“可选”，避免执行者在 Phase 1 误引入不必要的 embedding 服务依赖。

## 8. 阶段设计

### Phase 0：基线与项目状态锁定

目标：在任何模型训练前建立可辩护的基线。

交付物：

- `docs/model_layer_baseline.md`
- 依赖状态汇总
- 当前工具 schema 汇总
- 当前评估结果
- 现有评估脚本的机器可读奖励可行性报告
- 独立模型质量 reward-visible/held-out 数据协议与划分定义
- release gate 阻断项

必跑检查：

- 后端导入检查。
- `/api/health` 检查。
- `scripts/evaluate_agent_regression.py`
- `scripts/evaluate_knowledge_fusion.py`
- `scripts/evaluate_release_gate.py`
- 确认哪些 `scripts/evaluate_*.py` 输出是确定性的、机器可解析的数值，适合作为奖励。
- 定义新增模型质量评测数据中哪些用例 reward-visible，哪些用例 held out 给 Phase 6 最终评估。
- 检查 Ark 或等价 OpenAI-compatible 强教师模型端点是否可用；如果不可用，则将 Phase 2 数据合成标记为“仅可完成 schema/validator，合成阻断”。

报告必须明确说明哪些检查通过、哪些仍被阻断。由于缺少模型 key 导致 release gate blocked，可以作为基线事实记录。

Phase 0 必须将现有评估脚本分成三类：

1. Reward-compatible：输出确定性、机器可解析的数值。
2. Report-only：适合人类阅读报告，但不适合在线 reward。
3. Engineering-gate：验证 RAG、Obsidian、release gate 等工程能力，不作为模型质量证明。

如果没有稳定的 numeric reward-compatible 子集，则 Phase 4 不得启动，直到实现确定性 reward adapter。

Phase 0 同时必须新增独立模型质量评测协议：

- `data/model_training/eval/reward_visible.jsonl`：可用于 reward adapter 开发和训练期诊断。
- `data/model_training/eval/held_out.jsonl`：仅用于 Phase 6 四模型最终对比。
- `scripts/evaluate_model_outputs.py`：读取统一模型输出并计算工具选择准确率、参数合法率、推荐命中率和幻觉率。
- held-out 样本的 query、期望工具、期望参数约束和期望车型集合不得进入 SFT/GRPO 数据、prompt、reward、调参或 early stopping。

### Phase 1：原生 Qwen2.5-7B vLLM 接入

目标：在任何微调前，先让现有应用通过 vLLM 使用原生 `Qwen2.5-7B-Instruct`。

本阶段是第一个最小可演示里程碑。

vLLM tool calling 必需参数：

```bash
--enable-auto-tool-choice
--tool-call-parser hermes
--max-model-len 8192
--gpu-memory-utilization 0.90
--served-model-name qwen7b-nev
```

启动脚本必须优先使用本地模型路径：

```text
models/Qwen2.5-7B-Instruct
```

如果本地路径不存在，脚本必须给出清晰错误并退出，不允许静默下载模型。

交付物：

- `docs/deploy_vllm.md`
- `backend/config/config.vllm.example.yaml`
- `scripts/start_vllm_qwen7b.sh`
- `scripts/switch_llm_backend.py`
- `docs/model_layer_phase1_vllm_baseline.md`

验收标准：

- vLLM 启动时已启用 tool calling。
- 后端可以指向 vLLM chat endpoint。
- `/api/agent/recommend` 能在原生 Qwen 后端上运行。
- `/api/recommend-stream` 能在原生 Qwen 后端上运行。
- SSE trace 至少出现一条成功的工具调用路径。
- 至少一个评估脚本记录原生 Qwen 基线表现。

即使后续训练延迟，Phase 1 也必须产出一个可打 tag 的演示状态。

### Phase 2：工具调用数据合成

目标：生成与真实 5 个工具对齐且有 grounding 的 SFT 和 GRPO 数据。

交付物：

- `data_synth/tool_schemas.py`
- `data_synth/generate_sft_data.py`
- `data_synth/validate_tool_data.py`
- `data/model_training/sft_train.jsonl`
- `data/model_training/sft_val.jsonl`
- `data/model_training/grpo_prompts.jsonl`
- `data/model_training/data_synth_report.md`

数据要求：

- 工具 schema 名称和参数必须匹配 `agent_graph.py`。
- 工具输出应尽量通过执行真实本地工具生成。
- 车型名称必须来自真实车辆数据。
- 每个主要 intent 至少 500 条正向 SFT 样本，除非 Phase 0 明确记录为更小规模 pilot。
- 数据必须包含少量已标注负样本。
- 合成数据生成依赖可用强教师模型：Ark 或等价 OpenAI-compatible endpoint。如果没有教师端点，Phase 2 在 schema export 和 validator 之后停止。
- 使用 prompt variation、temperature variation 和多种 seed templates，降低单一教师模型导致的分布坍缩风险。

负样本类型：

- 选择错误工具。
- 缺少必要语义参数。
- 参数类型错误。
- 幻觉车型。
- 跳过必要的 retrieval/ranking 工具。
- 最终答案与工具输出矛盾。

负样本不是为了让模型学习错误输出，而是用于校验、奖励塑形和边界分析。

验收标准：

- 工具参数合法率 100%。
- 每个主要 intent 至少 500 条正向 SFT 样本，或者报告明确标记该轮为 pilot dataset，不具备完整 SFT 结论资格。
- 随机抽样 50 条人工检查，可读且 grounded。
- 数据报告包含 intent 分布、工具分布、正/负样本数量和 schema 校验汇总。

### Phase 3：QLoRA SFT

目标：让 `Qwen2.5-7B` 冷启动学习本项目的工具调用格式、画像抽取和 grounded response 风格。

交付物：

- `requirements-train.txt`
- `training/sft/` 下的 SFT 配置。
- 训练入口或明确的 LLaMA-Factory 命令。
- `checkpoints/sft/`
- `docs/model_layer_phase3_sft_report.md`

推荐训练约束：

- 4-bit NF4 QLoRA。
- `bnb_4bit_compute_dtype=float16`。
- LoRA `r=16`、`alpha=32`、`dropout=0.05`。
- target modules 覆盖 attention 和 MLP projection layers。
- 初始 `max_seq_len=4096`。
- `gradient_checkpointing=True`。
- `lr=2e-4`。
- `epochs=2-3`。

验收标准：

- 工具选择准确率高于原生 Qwen。
- 工具参数合法率高于原生 Qwen。
- 应用评估不低于 Phase 1 基线。
- SFT 报告同时包含训练曲线和应用层指标。
- 只有完成真实数据加载、真实 adapter 保存和真实验证推理，才能将 Phase 3 标记为完成；配置模板或训练骨架不算完成。

### Phase 4：GRPO/RLVR

目标：通过可验证奖励提升工具调用正确性、可执行推理和推荐质量。

交付物：

- `training/grpo/reward_fn.py`
- `training/grpo/train_grpo.py`
- GRPO 配置文件。
- `checkpoints/grpo/`
- `docs/model_layer_phase4_grpo_report.md`

奖励组件：

1. Answer reward：将 Phase 0 中选出的 reward-compatible evaluation subset 映射为有界分数。
2. Format reward：校验 JSON/tool-call 结构、工具名、参数名和参数类型。
3. Tool execution reward：真实执行工具调用，奖励非空、无异常、grounded 的输出。

奖励/评估隔离：

- Answer reward 只能使用 Phase 0 中选定的 reward-compatible cases。
- Held-out cases 不得进入 SFT 数据生成、GRPO prompt 构造、reward functions、超参选择或 early stopping。
- Phase 4 报告可以展示 reward-set 表现，但不能作为最终模型质量证据。
- 只有 Phase 6 可以报告最终模型质量结论，并且必须使用 held-out set。
- GRPO 必须从通过 Phase 3 门禁的真实 SFT adapter 启动；只有训练产出真实 adapter 并完成 held-out 推理后，才能将 Phase 4 标记为完成。

GRPO 稳定性要求：

- Rollout 必须使用 vLLM backend，或明确记录等价高吞吐 rollout 服务。
- 不允许依赖普通 HF `generate` 跑大规模 rollout。
- `num_generations >= 8`。
- 采样 temperature 在 `0.7-1.0` 范围。
- KL 使用小 `beta`，如出现不稳定则移除 KL。
- 使用 clip-higher 或等价高方差控制策略。
- 记录每项 reward 曲线，不只记录总 reward。

奖励吞吐要求：

- reward functions 必须在安全情况下，按 prompt/completion/tool-call hash 缓存确定性的工具执行和评估结果。
- 如果 evaluation adapter 支持批处理，应使用批量 reward 计算。
- 重型 report-only 评估脚本不得进入每一步 GRPO。
- 如果 reward 计算慢于 rollout 生成，必须将 `answer_reward` 替换为 reward-compatible 子集中的轻量代理指标，或采用离线批量 reward scoring。

单卡 32G 显存方案：

- 训练侧使用 QLoRA 和 gradient checkpointing。
- 如果训练和 rollout 共用一张 GPU，rollout 侧 vLLM 初始使用较低显存占比：`--gpu-memory-utilization 0.4-0.5`。
- 如果并发训练 + rollout OOM，则采用生成/训练交替的分时工作流，并在报告中记录。

Reward hacking 防御：

- Format reward 封顶。
- Answer reward 拥有最高有效权重。
- Tool execution reward 惩罚幻觉车型和空输出。
- 与工具输出矛盾的最终答案给予惩罚。
- 必须人工审计 30 条轨迹。

验收标准：

- GRPO 在 held-out model-layer evaluation 上优于 SFT；reward-set 提升不能作为充分证据。
- 工具调用成功率提升或保持稳定。
- 人工审计中“格式正确但答案错误”的轨迹比例低于 5%。
- 报告包含 OOM 策略、rollout 吞吐、reward 吞吐，以及 held-out evaluation 未参与训练的确认。

### Phase 5：部署与后端切换

目标：让云端、原生本地、SFT 本地、GRPO 本地后端都能在不修改业务代码的情况下切换。

交付物：

- `scripts/start_vllm_sft.sh`
- `scripts/start_vllm_grpo.sh`
- 最终版 `scripts/switch_llm_backend.py`
- `docs/deploy_model_backends.md`

支持的后端名称：

- `ark`
- `qwen_base_vllm`
- `qwen_sft_vllm`
- `qwen_grpo_vllm`

后端切换必须保留 chat/embedding 分离。本地 chat 模型切换不得覆盖 embedding 设置，除非用户明确要求。

验收标准：

- 每个后端都能通过可复现命令或配置 profile 选择。
- 切换后 `/api/agent/recommend` 可运行。
- 切换后 `/api/recommend-stream` 可运行。
- 健康检查或配置诊断能展示后端身份，且不暴露密钥。

### Phase 6：对比评估与文档对齐

目标：用证据证明或否定模型层价值。

交付物：

- `docs/model_layer_report.md`
- README/model-layer section 更新。
- 修正后的简历/项目话术。
- 最终评估产物。

对比列：

- Ark baseline
- 原生 Qwen2.5-7B via vLLM
- QLoRA SFT Qwen2.5-7B
- GRPO Qwen2.5-7B

指标：

- 工具选择准确率
- 工具参数合法率
- 推荐命中率
- 幻觉率
- 端到端时延
- agent regression pass rate
- knowledge fusion status
- release gate status

Held-out evaluation 规则：

- 四列对比必须使用 Phase 0 定义的 held-out evaluation set。
- 四个后端必须对同一份 `data/model_training/eval/held_out.jsonl` 生成原始输出，再由 `scripts/evaluate_model_outputs.py` 统一评分。
- Reward-set 表现只能作为诊断附录。
- 报告必须说明哪些 scripts/cases 是 reward-visible，哪些是 held out。
- “GRPO improves over SFT”等结论只有在 held-out 结果支持时才有效。

文档真实性规则：

- 工具数量写 5 个，除非真的新增更多工具。
- 知识文档数量写实际数量，除非真的新增文档。
- 车辆库规模写实际数量。

推荐的简历/文档路线是使用真实数字，不为了观感夸大。

## 9. 风险登记表

### vLLM 启动但 tool calls 为空

原因：缺少 vLLM tool-calling 参数或 parser 不兼容。

缓解：启动脚本强制包含 `--enable-auto-tool-choice --tool-call-parser hermes`，并在应用测试前先做带 tools 的 chat completion 测试。

### Chat 后端切换破坏 embedding

原因：chat 与 embedding 共用单一 base URL。

缓解：Phase 1 拆分 chat 与 embedding 配置。

### GRPO 训练不稳定

原因：KL 高方差、采样多样性不足或 reward 权重失衡。

缓解：小 KL 或去 KL、clip-higher、`num_generations >= 8`、temperature `0.7-1.0`，并记录分项 reward 曲线。

### 最终评估过拟合 reward 脚本

原因：同一批评估用例同时用于 GRPO reward 和 Phase 6 最终证明。

缓解：Phase 0 定义 reward-visible 与 held-out evaluation split。最终对比只使用 held-out-only cases。

### GRPO rollout 太慢

原因：使用普通 HF generation 跑 grouped rollout。

缓解：计划内 GRPO 必须使用 vLLM rollout backend。

### GRPO reward 计算太慢

原因：每个 rollout completion 都触发工具执行和重型评估脚本。

缓解：缓存确定性 reward 结果、批处理 reward adapters，并将重型 report-only 脚本保留给评估阶段而不是每步训练。

### 单卡 OOM

原因：QLoRA 训练和 vLLM rollout 争用同一张 32G GPU。

缓解：降低 vLLM 显存占比、使用 gradient checkpointing，或拆成生成/训练交替的分时流程。

### 合成数据偏离真实工具

原因：手写 schema 与 `agent_graph.py` 漂移。

缓解：schema exporter 和 validation gate 必须对照真实工具定义校验生成数据。

### 教师模型不可用

原因：缺少 Ark 或等价 OpenAI-compatible 教师端点。

缓解：Phase 2 必须先做教师可用性检查。没有教师时，只完成 schema export/validator，不声明 synthetic-data readiness。

### 合成数据分布坍缩到教师风格

原因：单一教师 prompt 风格主导所有 SFT 样本。

缓解：使用多 intent、多 temperature、多用户表达方式、多真实 trace seed 的模板组合。

### Reward hacking

原因：格式奖励压过答案质量。

缓解：封顶 format reward，以 answer reward 为主，惩罚矛盾答案和幻觉车型。

## 10. 评审门禁

设计文档未评审前，不开始实现。

Phase 1 原生 Qwen 基线未记录前，不开始 SFT。

Phase 0 未识别 reward-compatible 确定性数值评估输出和 held-out evaluation cases 前，不开始 GRPO reward 实现。

SFT 未产生可度量改进或明确失败报告前，不开始 GRPO。

Phase 6 最终对比不得将 reward-visible evaluation cases 作为主要证明。

没有四列对比报告，不得做最终项目提升声明。

## 11. 已解决的开放决策

计划范围：完整 Phase 0-6 改造。

实施优先级：Phase 0 和 Phase 1 优先。

本地模型策略：本地路径优先，不静默下载。

展示策略：使用真实项目数字，不虚报。

GRPO rollout 策略：vLLM backend 或有明确记录的等价方案；不默认使用大规模 HF rollout。
