
明白。你要的是一份**可直接投喂给另一个大模型执行的施工计划书(Planner)**——它得自包含(执行模型看不到我们这段对话,也没读过你的代码),把项目真实现状、改造目标、分阶段任务、每步产出和验收标准全部写死。我不动手,只出图纸。

下面这份直接复制给执行模型即可。

---

# 施工计划书:新能源汽车智能推荐平台 —— 增设"大模型层"(QLoRA微调 + GRPO强化 + vLLM部署)

## 0. 给执行模型的前置说明(必读)

你将改造一个**已完成的、真实的 LangGraph 多智能体项目**,任务是给它嫁接"模型训练/部署"的另一半,使其成为"应用层 + 模型层"全链路项目。**不要重写现有业务逻辑**,只做增量嫁接。所有涉及真实 API 密钥的文件(`backend/config/config.yaml`)保持私密,禁止提交到 git。

### 0.1 硬件与资产

- 单卡 32G 显存
- 基座模型:Qwen 7B(建议 Qwen2.5-7B-Instruct)
- 现有 LLM 后端:火山方舟(OpenAI 兼容),可继续作为"教师模型"用于数据合成

### 0.2 项目真实现状(执行前把这些当作事实,不要臆测)

**技术栈**:LangGraph 1.1.9 + LangChain + FastAPI + SSE + Vue3 + SQLite + scikit-learn。

**LLM 接入方式(关键嫁接点)**:

- `backend/app/services/llm_client.py` 里是标准 OpenAI 兼容客户端:`OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, http_client=httpx.Client(trust_env=False, timeout=TIMEOUT))`
- `backend/app/config.py` 读取环境变量 `ARK_API_KEY / ARK_BASE_URL / ARK_CHAT_MODEL / ARK_EMBEDDING_MODEL`,回退到 `config.yaml`。核心变量:`OPENAI_API_KEY / OPENAI_BASE_URL / CHAT_MODEL / EMBEDDING_MODEL / TEMPERATURE=0.2 / TIMEOUT=60`
- **结论:切换到自训练模型只需改 `base_url` + `CHAT_MODEL` 两个值,不改任何业务代码。**

**多智能体架构**:

- `backend/app/services/multi_agent.py`:Supervisor + Worker 结构。`START → Supervisor → [RecommendationWorker | KnowledgeWorker | SalesWorker] → Supervisor → END`。每个 Worker 是独立编译的 StateGraph,含 `agent_node + ToolNode + tools_condition`。`supervisor_graph` 用 `MemorySaver()` 做 checkpointer。
- `backend/app/services/agent_graph.py`:ReAct tool agent。含两张图:`recommend_graph`(线性 8 节点流水线)和 `tool_agent_graph`(route_intent → agent → tools/finalize,带 MemorySaver)。

**5 个真实工具(微调必须对齐这些 schema,注意不是 7 个)**,全部在 `agent_graph.py` 中以 `@langchain_tool` 定义:

1. `extract_user_profile(query: str, budget_max: int = 0) -> str`
2. `search_and_rank_vehicles(budget_max, preferred_type, preferred_energy, concerns, top_k=5) -> str`(本地 38 台车库)
3. `retrieve_knowledge_base(query: str) -> str`(知识库检索)
4. `search_web_info(query: str) -> str`
5. `generate_sales_talk(budget_max, concerns, top_model) -> str`

意图路由:`_llm_route_intent()` 让 LLM 返回 JSON `{"intent":..,"reason":..}`,失败回退 `_keyword_route()`。`TOOLS_BY_INTENT` 按意图(recommend/compare/knowledge/sales)过滤工具,`PROMPTS_BY_INTENT` 给每意图不同 system prompt。

**已有工程资产(直接复用,别重造)**:

- `scripts/` 下有 18 个 `evaluate_*.py` 评估脚本 —— **这是 GRPO 可验证奖励(RLVR)的现成打分器**。
- 系统有 SSE 流式 Trace(`_build_detailed_trace()` / `_format_tool_call()`)—— **这是真实工具调用轨迹日志,用作数据合成的种子**。
- Obsidian 自增长知识库、反馈策略(`apply_feedback_policy`)、数据治理、发布门禁、候选池选择(`select_candidate_pool`:local/real/fused)。

### 0.3 已知的"简历 vs 代码"不一致(改造收尾时必须修正)

- `backend/requirements.txt` **缺失 langgraph / langchain / tavily**,但代码在 import、简历在写 —— 必须补齐。
- 简历写"7 个工具",代码实际 **5 个** —— 二选一:改简历为 5,或真补 2 个工具(见 Phase 5 可选项)。
- 简历写"50 篇知识文档",代码注释是 **12 篇** —— 数量对齐。
- 车辆库 **38 台** —— 与代码一致,无需改。

---

## 1. 改造总目标

把 Qwen2.5-7B 用 QLoRA 微调 + GRPO 强化,训练成一个**专精本项目 5 个工具调用 + 用户画像抽取**的 Agent 模型,用 vLLM 部署为 OpenAI 兼容服务,替换现有火山方舟后端,使整套多智能体系统跑在自训练模型上,并用现有评估脚本证明"自训练 7B ≈ 或优于原云端大模型"在本业务上的表现。

**最终交付物**:

1. vLLM 部署脚本 + 一键切换配置
2. 数据合成管线(产出 SFT 数据 + GRPO 环境)
3. QLoRA SFT 训练脚本 + 产出 LoRA adapter
4. GRPO 训练脚本(含 3 个可验证奖励)+ 产出强化后模型
5. 回归评估报告(自训练模型 vs 基线)
6. 修正后的 requirements.txt + 简历话术

---

## 2. 分阶段执行计划

> 执行顺序建议:**Phase 1 先跑通最小闭环**(最快拿到可演示成果),再回头做 2→3→4 的训练链路。每个 Phase 完成后必须通过"验收"才进入下一步。

### Phase 1 — vLLM 部署 + LangGraph 切换(最小可跑闭环)

**目标**:先用**未微调**的原生 Qwen2.5-7B-Instruct 起 vLLM,让现有系统直接跑起来,验证工具调用链路在开源模型上能通。

**步骤**:

1. 安装 vLLM(建议 `pip install vllm --user --break-system-packages`,版本 ≥ 0.6.x 以支持 tool calling)。
2. 启动命令(关键参数,32G 显存):
   ```
   vllm serve Qwen/Qwen2.5-7B-Instruct \
     --enable-auto-tool-choice \
     --tool-call-parser hermes \
     --max-model-len 8192 \
     --gpu-memory-utilization 0.90 \
     --served-model-name qwen7b-nev
   ```

   > `--enable-auto-tool-choice --tool-call-parser hermes` 是 Qwen 系列在 vLLM 上启用工具调用的必需组合。
   >
3. 在 `backend/config/config.yaml`(或环境变量)中改:
   - `OPENAI_BASE_URL` → vLLM 地址 `/v1`
   - `OPENAI_API_KEY` → 任意占位串(vLLM 默认不校验)
   - `CHAT_MODEL` → `qwen7b-nev`
   - `EMBEDDING_MODEL` 保持原云端(vLLM 这台只服务 chat),或另起 embedding 服务。
4. 不改任何业务代码,重启后端。

**产出**:能跑的系统 + 一份 `docs/deploy_vllm.md`。
**验收**:调用 `/api/agent/recommend` 与 `/api/recommend-stream`,SSE Trace 里能看到至少一次成功的工具调用;`scripts/evaluate_*.py` 至少一个能跑出基线分数(记录下来作为"微调前基线")。

---

### Phase 2 — 训练数据合成(不爬取,用教师模型合成)

**目标**:产出两类数据:(A) SFT 数据(工具调用轨迹 + 画像抽取);(B) GRPO 用的 prompt 集 + 奖励环境。

**数据来源**:

- 主力:用火山方舟(教师模型)按 5 个工具 schema 合成多轮工具调用轨迹。
- 种子:导出系统真实 SSE Trace 日志作为 few-shot 范例喂给教师模型,保证轨迹分布贴近真实。

**步骤**:

1. 写 `data_synth/schemas.py`,把 `agent_graph.py` 的 5 个工具 schema 以 OpenAI function-calling 格式固化(参数名/类型必须逐字对齐真实代码)。
2. 写合成 prompt 模板(见附录 A),让教师模型对每个意图(recommend/compare/knowledge/sales)生成:
   - 用户 query(覆盖预算/车型/能源/顾虑等多样组合)
   - 正确的工具调用序列(含参数)
   - 工具返回(可用真实工具跑一遍拿真实返回,强烈建议这样保证 grounding)
   - 最终自然语言答复
3. **画像抽取子集**:单独合成 `extract_user_profile` 的 (query → 结构化 profile) 对。
4. **负样本**:合成"错误工具选择/漏调工具/参数缺失/幻觉车型"的负例,供 GRPO 拉开奖励差。
5. 数据清洗:去重、schema 校验(参数名/类型必须合法)、按意图分层。
6. 划分:SFT 训练/验证集;GRPO prompt 集(只留 query,答案由奖励函数在线判)。

**产出**:`data/sft_train.jsonl`、`data/sft_val.jsonl`、`data/grpo_prompts.jsonl`,以及一份数据统计报告(条数/意图分布/正负比)。
**验收**:随机抽样 50 条人工可读,工具参数 100% 合法;每个意图 ≥ N 条(建议每意图 ≥ 500 SFT 条)。

---

### Phase 3 — QLoRA SFT 冷启动

**目标**:让 7B 先学会本项目的工具调用格式和画像抽取,为 GRPO 打底(2026 工具 Agent 标准配方:SFT 冷启动 → GRPO)。

**配置(32G 显存跑 7B QLoRA 充裕)**:

- 量化:NF4 4bit(bitsandbytes),`bnb_4bit_compute_dtype=bfloat16`,double quant。
- LoRA:`r=16, alpha=32, dropout=0.05`,target 全部 attention + MLP 线性层(`q,k,v,o,gate,up,down`)。
- 训练:`lr=2e-4`,cosine,`warmup_ratio=0.03`,`epochs=2~3`,`max_seq_len=4096`,梯度累积按显存调,`gradient_checkpointing=True`。
- 框架建议:LLaMA-Factory 或 trl 的 `SFTTrainer`(任选,LLaMA-Factory 配置最省事)。
- 数据格式:多轮对话 + tool 角色,与 Qwen chat template 对齐。

**步骤**:

1. 写训练配置文件(YAML/命令)。
2. 训练,保存 LoRA adapter 到 `checkpoints/sft/`。
3. 合并或以 adapter 形式加载,用 vLLM 起服务(vLLM 支持 `--enable-lora` 加载 adapter)。

**产出**:`checkpoints/sft/` + 训练 loss 曲线。
**验收**:在 SFT 验证集上工具选择准确率、参数合法率显著高于 Phase 1 原生模型;跑 `scripts/evaluate_*.py` 分数不低于 Phase 1 基线。

---

### Phase 4 — GRPO 强化(RLVR,核心亮点)

**目标**:用可验证奖励把工具调用的**正确性、格式、可执行性**进一步拉高。GRPO 无 critic 网络(相比 PPO 省约一半显存),适合单卡。

**奖励设计(三项可验证奖励,复用现有评估脚本)**:

1. **答案奖励**:调用 `scripts/evaluate_*.py` 对模型最终推荐结果打分(如推荐车型是否命中期望集合、七维评分是否合理),映射到 [0,1]。
2. **格式奖励**:工具调用 JSON 是否合法、参数名/类型是否匹配 schema、是否按意图选对工具集(对照 `TOOLS_BY_INTENT`)。
3. **工具执行奖励**:实际执行工具是否成功返回(无异常、非空、车型真实存在于 38 台库中,惩罚幻觉)。

**配置**:

- 框架:trl `GRPOTrainer` 或 verl(单卡用 trl 更简单)。
- `num_generations`(每 prompt 采样组大小)= 8;`beta`(KL 系数)先设小值甚至趋 0。
- **已知坑与对策**(务必落实):
  - KL 高方差估计器 → 用 `clip-higher` 策略,或直接**移除 KL 项**(2026 多篇工具 Agent RL 实践证明去 KL 更稳)。
  - **reward hacking** → 三奖励加权且设上限,格式奖励不能盖过答案奖励;对"只输出格式正确但答非所问"的样本设惩罚。
  - 采样温度 0.7~1.0 保证组内多样性,否则 advantage 归零。
- 训练在 vLLM 上做推理采样(GRPO 需要高吞吐 rollout),trl 支持 vLLM backend。

**步骤**:

1. 把三个奖励封装成 `reward_fn(prompt, completion) -> float`,内部调用工具执行 + evaluate 脚本。
2. 用 Phase 3 的 SFT 模型作为 GRPO 初始 policy。
3. 训练,保存到 `checkpoints/grpo/`。

**产出**:`checkpoints/grpo/` + 奖励曲线 + 每类奖励分项曲线。
**验收**:综合评估分数高于 Phase 3;无明显 reward hacking(人工抽检 30 条轨迹,格式对但答非所问的比例 < 5%)。

---

### Phase 5 — 回归评估 + 简历/仓库对齐修正

**目标**:出对比报告,并修掉 0.3 的所有不一致。

**步骤**:

1. 跑全套 `scripts/evaluate_*.py`,产出三列对比表:**火山方舟基线 / 原生Qwen7B / 微调后Qwen7B**。指标:工具选择准确率、参数合法率、推荐命中率、端到端时延、幻觉率。
2. 修 `backend/requirements.txt`:补 `langgraph==1.1.9`、`langchain`、`tavily-python`(以及训练相关放独立 `requirements-train.txt`,不污染服务依赖)。
3. 工具数量对齐:**推荐**改简历为"5 个工具";或(可选增强)真补 2 个工具(如 `compare_vehicles` 显式对比工具、`estimate_total_cost` 购车总成本测算工具),补到 7 个并纳入训练数据。
4. 知识文档数量对齐(12 vs 50):要么补文档到 50,要么简历改 12。
5. 出一份 `docs/model_layer_report.md` 总结全链路。

**产出**:对比评估报告 + 修正后的仓库 + 升级版简历话术(见附录 B)。
**验收**:仓库 `pip install -r requirements.txt` 后能干净启动;简历每条数字都能在代码里找到对应。

---

## 3. 关键风险清单(执行时盯紧)

| 风险                 | 触发点                                      | 对策                                                               |
| -------------------- | ------------------------------------------- | ------------------------------------------------------------------ |
| vLLM 工具调用不生效  | 缺`--enable-auto-tool-choice`/parser 不对 | 用`hermes` parser;先用 curl 测 `/v1/chat/completions` 带 tools |
| 合成数据 schema 漂移 | 参数名与真实代码不符                        | Phase 2 步骤1 逐字对齐,加 schema 校验闸门                          |
| GRPO 崩溃/不收敛     | KL 高方差、组内无多样性                     | 去/降 KL、温度 0.7~1.0、`num_generations≥8`                     |
| reward hacking       | 格式奖励过重                                | 三奖励加权+上限,答案奖励主导                                       |
| 显存 OOM             | seq_len 过长                                | QLoRA + gradient_checkpointing,`max_len` 从 4096 起调            |
| embedding 服务缺失   | vLLM 只服务 chat                            | RAG 的 embedding 继续用云端或另起 bge 服务                         |

---

## 附录 A:工具调用轨迹合成 Prompt 模板(给教师模型)

```
你是数据合成器。基于以下工具定义,生成一条【意图={intent}】的多轮工具调用训练样本。
可用工具(严格按此 schema,参数名/类型不得更改):
{tool_schemas_json}

要求:
1. 先生成一个真实自然的用户 query(预算/车型/能源/顾虑随机组合,口语化)。
2. 生成正确的工具调用序列(assistant 的 tool_calls,参数合法)。
3. 每个工具调用后给出 tool 角色的返回(若提供真实执行结果则用真实结果)。
4. 最后生成 assistant 的自然语言终答,必须基于工具返回,不得虚构车型。
5. 车型只能来自本项目 38 台车库,禁止编造。
输出为 OpenAI messages 格式的 JSON。
```

负样本变体:在上述基础上追加"故意制造一处错误(错误工具/缺参/幻觉车型),并在 label 字段标注错误类型",供 GRPO 对比。

---

## 附录 B:升级版简历话术(改造完成后替换)

> **新能源汽车智能推荐平台｜AI Agent 应用 + 大模型全链路**技术栈:LangGraph / LangChain / QLoRA / GRPO(RLVR) / vLLM / Qwen2.5-7B / RAG / FastAPI / SSE / Vue3
>
> - 基于 LangGraph 构建 Supervisor + Worker 多智能体 + ReAct 工具代理(5 个业务工具),支持意图路由与 SSE 流式 Trace。
> - **自训练模型层**:用数据合成产出工具调用轨迹,QLoRA 微调 Qwen2.5-7B 冷启动 + GRPO 强化(答案/格式/工具执行三项可验证奖励),vLLM 部署为 OpenAI 兼容服务替换云端大模型。
> - 单卡 32G 完成 4bit QLoRA + GRPO,自训练 7B 在业务评估上达到/超过云端基座(附对比数据 X→Y)。
> - 复用 18 个评估脚本作为 RLVR 奖励环境,解决 reward hacking 与 KL 高方差(clip-higher / 去 KL)。

---

这份计划是**自包含**的:执行模型不需要看代码就能开工,所有真实结构、文件路径、参数 schema、坑和对策都写进去了。

要不要我再补两样让它更"傻瓜可执行":①把附录 A 的 schema JSON 按你 5 个工具**逐字写全**(执行模型直接抄);②给 Phase 3/4 各写一份**具体到能跑的训练配置文件**(LLaMA-Factory YAML + trl GRPO 脚本骨架)?你说要哪个我就往里填。
