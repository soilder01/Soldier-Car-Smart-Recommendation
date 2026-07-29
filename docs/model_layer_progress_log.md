
## 2026-07-23 09:42:33 Phase 0

- 已生成基线报告。
- 当前工作区不是 Git 仓库时，以本文件记录进度。
- Git unavailable: root workspace is not a git repository. Phase 0 artifacts written without commit.

## 2026-07-23 09:43:36 Phase 0

- 已生成基线报告。
- 当前工作区不是 Git 仓库时，以本文件记录进度。
- Git unavailable: root workspace is not a git repository. Phase 0 artifacts written without commit.

## 2026-07-23 09:49:38 Phase 0

- 已生成基线报告。
- 当前工作区不是 Git 仓库时，以本文件记录进度。
- Git unavailable: root workspace is not a git repository. Phase 0 artifacts written without commit.
- 评估脚本副作用（overwritten）：
  - `data/real_world/agent_regression_evaluation.json`
  - `data/real_world/agent_regression_evaluation.md`
  - `data/real_world/knowledge_fusion_evaluation.json`
  - `data/real_world/knowledge_fusion_evaluation.md`
  - `data/real_world/real_data_governance_report.json`
  - `data/real_world/release_gate_evaluation.json`
  - `data/real_world/release_gate_evaluation.md`
- 评估脚本副作用（created）：
  - `obsidian-vault/07-测试样例/Agent端到端回归评估-20260723-094933.md`
  - `obsidian-vault/08-推荐案例/推荐案例-20260723-094933-Hyundai-IONIQ-5-Base.md`
  - `obsidian-vault/08-推荐案例/推荐案例-20260723-094933-智界-R7.md`
  - `obsidian-vault/08-推荐案例/推荐案例-20260723-094933-比亚迪-宋PLUS-DM-i.md`
  - `obsidian-vault/08-推荐案例/推荐案例-20260723-094933-理想-L6.md`
  - `obsidian-vault/08-推荐案例/推荐案例-20260723-094938-Hyundai-IONIQ-5-Base.md`
  - `obsidian-vault/08-推荐案例/推荐案例-20260723-094938-智界-R7.md`
  - `obsidian-vault/08-推荐案例/推荐案例-20260723-094938-比亚迪-宋PLUS-DM-i.md`
  - `obsidian-vault/08-推荐案例/推荐案例-20260723-094938-理想-L6.md`

## Phase 1 config split

- 已增加 chat/embedding 配置分离。
- Chat 配置按完整 provider 三元组原子选择，禁止跨 provider 拼接。
- 当前 RAG 仍使用本地 TF-IDF/BM25-like，embedding 配置为预留项，不调用 embedding API。

## 2026-07-23 Task 4 review 修复

- 严格按 TDD 先补 review 回归测试；实现修改前 focused pytest 收集 21 项，
  结果为 `15 failed, 6 passed`，确认 RED。
- 最小实现后同一 focused pytest 结果为 `21 passed`，确认 GREEN。
- `build_backend_env("ark")` 现在只返回稳定的 `${ARK_API_KEY}`、
  `${ARK_BASE_URL}`、`${ARK_CHAT_MODEL}` 符号映射，不读取或泄露当前环境值。
- `render_exports()` 仅对白名单 Ark 三元组生成 eval-time block；其他字符串
  继续使用 `shlex.quote()`，不开放任意 shell expansion。
- Ark block 在普通 Bash 与 `set -u` 下均拒绝 unset、空字符串和纯空白值；
  失败时返回非零并保持三个既有 `CHAT_*` 全部不变，完整时通过单个
  `export` 命令统一提交。
- 特殊字符可无损传递且不会二次执行；CLI 已复用
  `build_backend_env()` + `render_exports()`，API/CLI 行为一致。
- Focused tests：
  `PYTHONPATH=. .venv/bin/python -m pytest tests/model_layer/test_switch_llm_backend.py -v`。

## 2026-07-23 Task 7 教师配置门禁

- 新增 `data_synth/generate_sft_data.py`，按完整同源三元组原子选择
  `CHAT > ARK > OPENAI`，拒绝空白值和跨 provider 混配。
- 状态输出和报告只包含布尔配置证据，不写入 API key、base URL 或模型实值。
- `config_ready` 只代表环境变量完整，始终保持
  `endpoint_verified=false`；未运行网络探测或数据合成。
- 当前真实状态：
  `blocked` / `provider=none` / `endpoint_verified=false` /
  `data_generation=not_started`。
- TDD：RED `7 failed, 92 deselected`；GREEN `7 passed, 92 deselected`。
- 关联回归：`107 passed`；独立复审：`Approved`。

## 2026-07-23 Task 8 V100 FP16 QLoRA 配置模板

- 新增独立 `requirements-train.txt`；未安装依赖，服务 `.venv` 未引入
  CUDA torch、vLLM 或 bitsandbytes。
- 新增结构化 QLoRA 配置：4-bit NF4、double quant、compute dtype
  `float16`、`fp16=true`、`bf16=false`、完整 7 个 Qwen LoRA target。
- SFT README 和 Phase 3 报告保持真实状态 `blocked`，adapter reload 与
  模型提升均为未验证。
- TDD：RED `3 failed`；GREEN `3 passed`；独立复审：`Approved`。

## 2026-07-23 Task 9 GRPO reward 骨架

- 新增 `training/grpo/reward_fn.py`，包含 `RewardContext`、`RewardCache`、
  `compute_format_reward()`、`compute_tool_execution_reward()` 和
  held-out fail-closed 的 answer reward 占位。
- Reward split 使用 NFKC、空白压缩和 casefold 规范化；reward-visible 与
  held-out 有交集时直接拒绝。
- format reward 复用训练数据 validator 的 approved tool schema，不执行工具；
  tool execution reward 必须有一一对应且无 error 的 tool result，不能仅凭
  格式给分。
- GRPO 配置锁定 vLLM rollout、`num_generations=8`、`temperature=0.8`、
  `beta=0.0`、`clip-higher` 和 32G V100 显存策略。
- `train_grpo.py` 当前 fail-closed，不启动训练、不读取 held-out、不连接 vLLM。
- TDD：RED `1 collection error`；GREEN `17 passed`；关联回归 `192 passed`；
  独立复审：`Approved`。

## 2026-07-23 Task 10 SFT/GRPO vLLM 启动脚本

- 新增 `scripts/start_vllm_sft.sh` 和 `scripts/start_vllm_grpo.sh`。
- 两个脚本均检查本地 base model 与对应 LoRA adapter，缺失时 exit 1；
  不静默下载、不继续启动。
- SFT 默认 `port=8001`、`served_model=qwen7b-sft`、adapter
  `checkpoints/sft`；GRPO 默认 `port=8002`、`served_model=qwen7b-grpo`、
  adapter `checkpoints/grpo`，与 `switch_llm_backend.py` 保持一致。
- 均包含 `--enable-lora`、`--lora-modules`、`--enable-auto-tool-choice`、
  `--tool-call-parser hermes`、`--max-model-len 8192`。
- TDD：RED `2 failed`；GREEN `8 passed`；联合后端切换回归 `29 passed`；
  `bash -n` 通过；独立复审：`Approved`。

## 2026-07-23 Task 11 Phase 6 报告模板与 README

- 新增 `docs/model_layer_report.md`，四后端均保持“未评估”，最终结论规则
  绑定 held-out 四列对比。
- README 新增模型层改造规划：5 个真实业务工具、本地 Qwen2.5-7B、
  vLLM、QLoRA SFT、GRPO/RLVR，最终只用 held-out 声明提升。
- 门禁检查无 “7 个工具” 或 “50 篇” 虚假数字；独立复审：`Approved`。

## 2026-07-23 Task 12 脚手架阶段总验证

- `PYTHONPATH=backend:. .venv/bin/python -m pytest tests/model_layer -q`：
  `282 passed`，2 个 FastAPI `on_event` deprecation warnings。
- `scripts/evaluate_agent_regression.py`：4/4，通过率 100.0%，平均分 100.0。
- `scripts/evaluate_knowledge_fusion.py`：7/7，RAG chunks 2061，
  RAG documents 12，Obsidian nodes 492，Obsidian chunks 2046。
- `scripts/evaluate_release_gate.py`：exit 0，但业务状态仍 `blocked`；
  `release_allowed=false`，gate score 66.7%，2 个 blocker：
  工程化就绪评分 90.9 < 95、发布阻断风险 1 > 0。
- `/api/health` smoke：HTTP 200，vehicle_count 38，rag_chunks 2086，
  `openai_configured=false`。
- 本地 `models/Qwen2.5-7B-Instruct` 存在，明确不提交；未发现
  `backend/config/config.yaml` 或 `.env` 私密配置文件。

## 2026-07-23 外部门禁复核

- 当前目录即执行目标仓库，不再做 Windows 路径迁移假设。
- 教师模型三元组：`.env` 已配置 `ARK_API_KEY + SEEDPRO_EP`。已更新
  teacher gate，使其直接读取 `.env`，并将 SeedPro endpoint 映射为
  Ark teacher：`ARK_BASE_URL=<ARK_BASE_URL_PLACEHOLDER>`、
  `ARK_CHAT_MODEL=SEEDPRO_EP`。
- 教师连通性：SeedPro Ark chat smoke 通过，返回 `finish_reason=stop`；
  未打印响应正文、base URL、模型实值或 API key。
- 固定调用 smoke：已生成 `data/model_training/pilot_sft.jsonl`，1 条
  `recommend` 样本。该文件只证明“本地工具结果 + SeedPro 最终回答”管道
  连通，不作为后续训练数据生成方式。
- 方案 B teacher-decision pilot：已生成
  `data/model_training/teacher_decision_pilot_sft.jsonl`，1 条 `recommend`
  样本。SeedPro 接收生产 intent prompt、用户 query 和 5 个工具 schema，
  自主发起 tool calls；本地真实执行 `.invoke()` 并回喂 tool messages；
  教师多轮收敛到 final answer。最终轨迹包含
  `system -> user -> assistant(tool_calls) -> tool -> ... -> assistant(final)`。
- 方案 B pilot 工具序列：
  `extract_user_profile -> search_and_rank_vehicles -> retrieve_knowledge_base
  -> search_web_info -> generate_sales_talk`；通过 `validate_record()`、
  held-out ID/query 泄漏防线和 intent mandatory tool 审计。
- 新增 `audit_teacher_decision_record()`，在 validator 之外检查教师工具决策质量：
  mandatory 工具有序完整、forbidden/unknown tool 为错误。该门禁用于阻止
  “validator 通过但工具决策错误”的样本进入扩量。
- 根据人工审计反馈，新增接地约束与自动 `audit_answer_grounding()`：
  最终回答里的价格区间、续航、油耗、轴距、保修、装载能力等硬指标必须能在
  user/tool evidence 中找到；伪造的《来源标题》直接判为不合格。该门禁已接入
  `generate_teacher_decision_record()`，不合格样本不会写入 JSONL。
- 重新生成 1 条方案 B grounded pilot：工具序列为
  `extract_user_profile -> search_and_rank_vehicles -> retrieve_knowledge_base
  -> generate_sales_talk`。最终答案不再包含 12.98/16.98、4.4L/100km、
  2920mm、三电终身保修或伪造《来源标题》；纯电和增程候选只作为与插混硬约束
  不匹配的过滤项出现，未进入推荐表。`validate_record()`、held-out 防线、
  mandatory tool 审计和接地审计均为空错误。
- 当前待办（不阻塞 1 条 pilot，但扩量前需要修）：`extract_user_profile`
  漏掉用户明说的“能耗”，并凭空加入“性价比”；`search_and_rank_vehicles`
  在 `preferred_energy=插混` 时仍返回纯电/增程候选，且多个候选 `score=100`
  无区分度。
- `data/model_training/data_synth_report.md` 当前状态：
  `config_ready` / `teacher_decision_pilot_generated` /
  固定调用 smoke 记录数 1 / 教师决策 pilot 记录数 1。pilot 不等于全量训练集，
  不得宣称完成 Task 15。
- GPU：`Tesla V100-SXM2-32GB`，driver `470.129.06`，显存 `32510 MiB`。
- 模型文件：`models/Qwen2.5-7B-Instruct` valid，4 个 safetensors 分片完整，
  `model_gate=ready`。
- 运行时门禁：`runtime_gate=blocked_until_verified`，
  `overall_gate=blocked_until_verified`；缺少独立环境、torch CUDA、
  bitsandbytes 4-bit、vLLM model load 和 vLLM tool-call smoke 证据。
- 执行策略调整：教师 endpoint 门已通；下一步先跑最小 teacher pilot 数据合成。
  vLLM/CUDA 隔离运行时仍需继续打通。

## 2026-07-23 Task 15 Grounded 多候选黄金样本修正

- 保留并加严 grounding validator：硬指标幻觉、保修类短语、伪造《来源标题》、
  `[1]` bracket 来源行、未接地“第一梯队/同级领先”和能耗软幻觉均一票否决；
  `generate_teacher_decision_record()` 支持有界接地重写，但失败样本仍不落盘。
- 修复 `search_and_rank_vehicles` 质量问题：`preferred_energy=插混` 时工具内部
  先返回精确插混候选，同时允许增程作为“广义插电备选”；纯电不进入插混候选；
  评分加入预算距离和能源备选惩罚，避免全 100 分无区分度。
- 固化增程口径：增程算作广义插电范畴，可作为插混需求备选；如用户严格要求
  PHEV，则进一步筛除增程。该口径已写入推荐系统提示。
- 修复 `extract_user_profile` 关注点抽取：用户明说“能耗”会进入 concerns；
  普通“预算”不再被错误升级为“性价比”关注点。
- 重新生成 1 条插混 grounded golden pilot：
  `data/model_training/teacher_decision_pilot_sft.jsonl`。工具序列为
  `extract_user_profile -> search_and_rank_vehicles -> retrieve_knowledge_base
  -> search_web_info -> generate_sales_talk`；画像 concerns 为 `能耗、空间`；
  推荐候选为 2 款插混 + 3 款增程备选，具备多候选对比。
- 验证：`validate_record()`、held-out 防线、mandatory tool 审计和接地审计均为空；
  `PYTHONPATH=backend:. .venv/bin/python -m pytest
  tests/model_layer/test_sft_generation.py tests/model_layer/test_agent_tool_quality.py`
  结果 `20 passed`。尚未扩到 20/intent，等待人工确认该黄金样本。

## 2026-07-23 Task 15 Validator 泛化与工具喂饱修正

- 扩量仍暂停；本轮只做 validator 泛化、工具证据增强、1 条真实教师样本和
  20 条本地 pressure test。
- `search_and_rank_vehicles` 返回从简化卡片升级为 grounded evidence：
  `specs.price_range / match_score / cltc_range / battery / fast_charge /
  seats / wheelbase / trunk_volume / safety_score / monthly_sales`，并返回
  `energy_evidence` 原文与 `known_missing_specs`，明确油耗/电耗/保修政策
  当前无真实字段。
- `audit_answer_grounding()` 从短语黑名单转为带单位数值/规格反查：
  答案里的万元、km、mm、kWh、分钟、座、L、分等声明必须能在 tool evidence
  中找到同一数值和归一化单位；来源行必须逐字可追溯；政策类具体承诺需有证据；
  `finish_reason=length` 的截断回答直接拒绝落盘。
- 更新生成提示：有 `specs` 就给具体建议；油耗/电耗/保修未返回时只说明需官方
  核验；能耗类非数值判断只能复述 `energy_evidence` 原文，不扩写成未返回的
  成本比较。
- 真实 spec-fed 插混样本已重跑并覆盖
  `data/model_training/teacher_decision_pilot_sft.jsonl`：
  `validate_record()`、mandatory tool 审计、grounding 审计均为空；最终回答
  非截断，包含 5 候选与价格、CLTC、电池、快充、轴距、后备箱、座位数、
  匹配分等具体字段。
- 新增 `data_synth/grounding_pressure_test.py`，输出
  `data/model_training/grounding_pressure_report.md/json`。20 条差异 query
  pressure test 结果：误杀 0、漏网 0、PASS。该 pressure test 只验证
  validator 泛化，不等同于 20/intent 教师数据扩量。
- 回归：`tests/model_layer/test_sft_generation.py`
  `tests/model_layer/test_agent_tool_quality.py`
  `tests/model_layer/test_tool_schemas.py` 共 `25 passed`；`py_compile` 通过。

## 2026-07-23 Task 15 20/intent 教师决策审计批次

- 已按 Plan B 生成 5 个意图各 20 条成功教师决策轨迹，共 100 条，写入
  `data/model_training/teacher_decision_20perintent_sft.jsonl`；未污染黄金样本
  `data/model_training/teacher_decision_pilot_sft.jsonl`，未启动 500/intent。
- 意图覆盖：`recommend / compare / customer_service / deep_search / sales`。
  第 5 类沿用现有 pipeline 中的 `sales`，未为凑数新造意图。
- 每条成功样本均带 `validate_record=[]`、`decision_audit=[]`、
  `grounding_audit=[]`、`finish_reason`、`bounded_rewrite_triggered`、
  `tool_call_rounds`、`tool_call_count` 和 `tool_names`；100 条成功样本
  `finish_reason` 均为 `stop`。
- 统计报告已写入 `data/model_training/20perintent_audit_report.md/json`：
  全局 attempted=105，accepted=100，截断 1，截断率 1/105=1.0%，
  grounding rewrite 33/100=33.0%，平均 tool_call 轮次 2.97。
- 分意图结果：`recommend` 20/20，`compare` 20/20，
  `customer_service` 20/20，`deep_search` 20/25，`sales` 20/20。
  `deep_search` 历史失败 5 条均 fail-closed 未落盘，其中 3 条为旧提示下
  mandatory tool 顺序问题，1 条为旧提示下价格硬指标 grounding 失败，
  1 条为 `max_tokens=1800` 截断；后续已将 deep_search 提示改为
  先画像 -> 候选/specs -> 知识库 -> 实时信息，并将 20/intent 生成
  `max_tokens` 提至 2600、timeout 提至 240 秒，重跑补齐成功样本。
- 抽样轨迹：报告中已按每意图前 3 条原样贴出完整 messages 轨迹，包含
  assistant tool_call、tool observation 和 final answer，用于人工判断
  compare 是否并列多车、customer_service 是否克制、deep_search 是否多轮
  observe -> decide、recommend 是否保持 specs 接地、sales 是否符合销售话术行为。
- 验证：最终字段/空审计/意图计数本地校验通过；`PYTHONPATH=backend:.
  .venv/bin/python -m pytest tests/model_layer/test_agent_tool_quality.py
  tests/model_layer/test_sft_generation.py -q` 结果 `28 passed`。

## 2026-07-23 Task 15 决策质量审计材料与 rewrite 归因

- 已生成外部审计用纯文本 JSON：
  `data/model_training/decision_quality_audit_materials.json`，包含每意图 3 条
  抽样轨迹，共 15 条。每条包含 query、教师工具决策、tool_call 名与参数、
  observation 摘要、最终 answer、finish_reason、bounded_rewrite_triggered、
  tool_call_rounds 和 tool_call_count；不包含 key/base_url/model 实值。
- `20perintent_audit_report.md/json` 已追加 `Rewrite 归因审计与 sales 重跑结果`
  章节。归因口径：对 `bounded_rewrite_triggered=true` 的样本，取重写前
  assistant final answer 重新执行 `audit_answer_grounding()` 并按主因/错误项
  分类。
- 原始高 rewrite 意图归因：
  `recommend` 为分散型（主因：不可反查价格/记忆价格 7/11，规格/评分 3/11，
  价格粗略改写 1/11）；`deep_search` 为分散型（不可反查价格/记忆价格 5/9，
  规格/评分 3/9，价格粗略改写 1/9）；`sales` 为单一系统性模式
  （权益/保修/政策承诺不可反查 8/10）。
- 仅对 `sales` 做定向修正和重跑，未触碰 compare/customer_service，未启动
  500/intent、SFT 或 GRPO。修正包括：sales 首轮提示禁止未接地保修/权益/
  补贴/赠品/免费服务/最低价/锁价/交付周期等承诺；`generate_sales_talk`
  工具改为引导核验官方公开资料；二次收紧 sales 提示，禁止无证据分钟/公里/
  百分比/金额/年限/次数等数字化话术。
- `sales` v1 重跑未达标：19/20 成功，1 条 429 服务过载未落盘，
  rewrite 5/19=26.3%，截断 0。v2 重跑达标：
  `data/model_training/teacher_decision_20perintent_sales_rerun_v2.jsonl`
  20/20 成功，rewrite 2/20=10.0%，截断 0/20=0.0%，无越界推荐/联网工具。
- 验证：`PYTHONPATH=backend:. .venv/bin/python -m pytest
  tests/model_layer/test_agent_tool_quality.py tests/model_layer/test_sft_generation.py -q`
  结果 `28 passed`。

## 2026-07-24 Task 15 放大前硬提示门禁与并发生成

- 仅收紧 `recommend` / `deep_search` 首轮系统提示，未修改
  compare/customer_service/sales 提示。硬约束要求所有车型硬指标逐字复制
  `search_and_rank_vehicles.specs`，价格只允许完整 `price_range`，禁止模型
  记忆、区间端点、单位换算、阈值/百分比推导；车型名数字不得推导座位数。
- 工具证据卡移除 `price_min_yuan / price_max_yuan` 原始元字段，避免教师把
  原始价格换算成未经 evidence 支持的单边“起售价”。
- 单意图生成器新增分批并发（CLI `--concurrency`，本轮使用 20）、主线程串行
  审计/落盘、断点续跑、逐条统计刷新和 429 指数退避（5/10/20/40 秒）。
- `recommend` 最终达标批次：
  `teacher_decision_20perintent_recommend_hardprompt_rerun_v3.jsonl`，
  accepted=20，rewrite=1/20=5.0%，truncated=0，平均轮次=3.60，
  成功样本三审计全空。历史 2 次失败均保留在 failure 日志。
- `deep_search` 最终达标批次：
  `teacher_decision_20perintent_deep_search_hardprompt_rerun_v2.jsonl`，
  accepted=20，rewrite=3/20=15.0%，truncated=0，平均轮次=4.10，
  成功样本三审计全空。
- compare 20 条覆盖审计写入
  `data/model_training/compare_20_coverage_audit.md/json`：
  双车均在 38 车库 17/20=85.0%，真实库缺口 3/20=15.0%；
  但工具轨迹实际同时召回两台点名车仅 4/20=20.0%，邻近替代或召回不完整
  16/20=80.0%。这说明仅从 38 车库构造 500 条 query 不能保证“可真对比”，
  500/intent 启动前仍需补齐点名车型定向召回能力。
- 尚未启动 500/intent、SFT 或 GRPO；等待用户确认前置门禁及 compare 召回
  修复方向。

## 2026-07-24 Task 15 compare 点名检索修复与 500/intent 完成

- compare 根因已确认并修复：原 `search_and_rank_vehicles` 仅按画像排序 top-k，
  不能保证返回用户点名的两款车型，导致 20/intent 库内点名对实际同时召回率仅
  4/20=20.0%。工具现新增 `model_names` 定向查库路径；compare 必须先用该
  入参获得 `named_vehicles` 的真实 specs，排序结果仅作为
  `supplemental_vehicles`。
- 工具返回 `named_vehicle_lookup`（请求、已解析、缺失车型名和
  `named_vehicle_missing`）；compare 成功样本在顶层持久化该字段。双车均在库时
  强制并列比较；任一点名车缺失时，明确“库中无此车规格”并保留诚实的邻近对比。
- 修复同品牌前缀碰撞：`享界S9增程` 不再被 `享界S9` 的前缀匹配误判为缺车。
  解析器按完整品牌车型名、完整车型名、部分匹配分层选择；库内全车型名称
  round-trip 回归覆盖已加入测试。
- compare 独立 20 条验收批次
  `teacher_decision_20perintent_compare_named_lookup_rerun_v5.jsonl`：
  库内点名对 17/17 同时召回=100.0%，库外对 3/3 诚实拒绝并标记缺失，
  rewrite=10.0%，截断=0，三审计均为空。
- 500/intent 生成器支持独立 output label、20 并发、断点续跑、429 指数退避、
  fail-closed failure 日志和分意图实时熔断。compare 分母以生成 query 的期望
  双车为准；错配必须计入分母且首次错配即停止，避免“错误缺失标记”掩盖召回率。
- 最终 500/intent 产物均为独立 JSONL，未启动 SFT/GRPO：

  | intent | 成功/尝试 | 截断率 | rewrite率 | 平均工具轮次 | 关键门禁 |
  |---|---:|---:|---:|---:|---|
  | recommend | 500/517=96.7% | 0.2% | 2.0% | 4.39 | 未触发 |
  | compare | 500/504=99.2% | 0.2% | 11.6% | 1.82 | 双车召回 500/500=100.0%，错配 0 |
  | customer_service | 500/501=99.8% | 0.0% | 0.0% | 1.09 | 未触发 |
  | deep_search | 500/502=99.6% | 0.2% | 6.6% | 4.14 | 未触发 |
  | sales | 500/507=98.6% | 0.2% | 10.6% | 1.64 | 未触发 |

- `deep_search` 与 `sales` 分别完成后均重跑了成功集全量三审计；
  最终对五个 intent 全部 2,500 条成功样本复跑
  `validate_record`、意图决策审计、grounding 审计、落盘审计字段校验，compare
  额外复核点名双车实际召回。结果为 0 失败，见
  `data/model_training/500perintent_final_full_audit.json`。
- 服务过载、截断、无依据政策/规格等 500 批次失败均仅进入各自
  `*_failures.jsonl`，未写入成功集；未打印任何 key、base_url 或 model 实值，
  未迁移产物到 Windows 仓库。
- 回归：`PYTHONPATH=backend:. .venv/bin/python -m pytest tests/model_layer -q`
  为 `324 passed`（仅 FastAPI 生命周期弃用警告）。下一步仅等待用户审阅统计并
  决定是否进入训练准备；不得自行启动 SFT 或 GRPO。

## 2026-07-24 训练前准备：数据冻结与 QLoRA dry-run

- 数据合成阶段完成：五意图 2,500 条成功轨迹已冻结为 Qwen2.5 工具调用 SFT
  资产。源分片保留在 `data/model_training/sft_freeze/shards/`，每个 intent
  500 条；合并后的 `sft_train.jsonl` 为 2,250 条，`sft_val.jsonl` 为 250 条。
- 固定切分已写入代码和 manifest：按 intent 分层，eval 比例 10%，固定种子
  `20260724`。每个 intent 固定为 450 train / 50 eval；train/eval 的规范化
  ID 与 query 均无交集。
- 数据保留原始 `messages` 多轮轨迹，并生成 Qwen2.5 工具调用 ChatML 文本；
  `system/user/assistant/tool` 角色、tool_calls、真实 observation 与后续
  assistant 决策均未压平。训练标签只覆盖 assistant 工具调用和最终回答段，
  不将 system/user/tool observation 作为监督目标。
- 已建立 `sft_freeze/reward_reservation_manifest.json`：保留
  reward-visible 20 条和最终 held-out 40 条的 ID、规范化 query 哈希。冻结器对
  2,500 条 SFT 源执行 ID 与 query 双重硬校验；本次交集均为 0，任一交集会
  fail-closed。完整数据说明见 `data/model_training/sft_dataset_card.md`。
- 新增 `training/sft/train_qlora_sft.py`。入口仅支持 `--dry-run`：
  强制本地模型目录和 `local_files_only=True`，配置 NF4 4-bit/double quant/
  float16，挂载 LoRA，读取冻结集一条 batch 并执行一次 forward/backward；
  不带 `--dry-run` 会拒绝，且任何路径均不保存 adapter/checkpoint。
- 真实 dry-run 已执行并如实阻断：
  `training_dependency_missing`，`ModuleNotFoundError: No module named 'torch'`。
  结果记录在 `data/model_training/sft_dry_run_report.json`。未安装训练依赖到
  服务 `.venv`，未伪造 CUDA、显存或 loss 结果。
- 验证：冻结/隔离/QLoRA dry-run 定向测试 `7 passed`，全量冻结不变量为
  2,500 条、每 intent 450/50、train/eval ID/query 交集 0、SFT/reserved
  ID/query 交集 0、supervision span 无效数 0。
- 当前状态：数据合成阶段完成；SFT 脚手架就绪但 dry-run blocked；尚无 LoRA
  权重、尚无 held-out 模型对比、未启动 SFT 或 GRPO。

## 2026-07-24 独立 CUDA 运行时与 Qwen 模板验证

- 新建隔离训练环境 `.venv-train`，未修改后端服务 `.venv`。本地未发现可直接
  复用的训练 wheel 后，仅下载依赖包；未下载任何模型权重。已锁定的核心依赖见
  `training/sft/requirements-cu118.lock.txt`：PyTorch `2.3.1+cu118`、
  transformers `4.44.2`、peft `0.12.0`、accelerate `0.33.0` 和
  bitsandbytes `0.43.3`；`pip check` 通过。
- 真实 CUDA 探测：`torch.cuda.is_available()=True`，设备为
  `Tesla V100-SXM2-32GB`，compute capability `7.0`，CUDA runtime `11.8`；
  探测时可用显存 `31730.8 MiB`。
- 首次官方模板对比发现冻结器将 tools schema 编码为紧凑 JSON，而 Qwen tokenizer
  的 `tojson` 采用含空格的 JSON 格式。已以 tokenizer 输出为准修复冻结渲染器、
  新增格式回归测试并重建所有 2,500 条冻结资产。
- 修复后以本地 `tokenizer.apply_chat_template()` 复核同一冻结样本
  `500pi-compare-0001`：字节长度均为 `17626`，SHA-256 相同，
  `first_difference_offset=null`，结果为 `matched`。报告：
  `data/model_training/qwen_template_verification.json`。
- 真实 GPU dry-run 已完成：本地 Qwen2.5-7B-Instruct 以 NF4 4-bit/double
  quant/float16 加载，LoRA 注入后从冻结集取 1 条样本，执行且仅执行 1 次
  forward/backward。loss=`0.570825457572937` 且有限，单 step=`17.4041s`，
  峰值 allocated=`19181.7 MiB`、peak reserved=`21208.0 MiB`；退出后 GPU
  已释放为 `0 MiB` 使用。报告：
  `data/model_training/sft_dry_run_gpu_report.json`。
- `--dry-run` 未保存任何 checkpoint/adapter；`checkpoints/sft` 不存在。
  这证明训练运行时 ready，但不代表正式 SFT 可启动：Phase 1 原生模型基线和
  后续训练晋级门禁仍未完成。尚无 LoRA 权重、尚无 held-out 模型对比，未启动
  SFT 或 GRPO。
- reward 保留集来源已澄清：`eval/reward_visible.jsonl` 的 20 条和
  `eval/held_out.jsonl` 的 40 条均为独立维护、人工编写的结构化 agent 合约
  评测用例，带真实 query、工具契约和本地车型约束；它们不是教师生成轨迹或
  占位符。完整说明见 `data/model_training/eval/README.md`。

## 2026-07-24 Phase 1 原生基线与 SFT 启动计划

- Phase 1 基线已使用本地未微调 Qwen 在 canonical held-out 40 条上完成。运行器
  直接复用生产 intent prompt、工具 schemas、真实工具执行和既有
  `evaluate_model_outputs` 合约评分；Qwen 的完整 XML tool call 会在闭合标签处
  结束当前轮并回灌真实 observation。无效 XML、参数或终态 JSON 均按失败记录，
  不由运行器补写。
- 基线只读取 `eval/held_out.jsonl`，对 `reward_visible.jsonl` 做了 ID/query
  双重隔离校验且未读取其样本。严格总分
  `contract_grounding_pass_rate=6/40=15.0%`：recommend 0/10、compare 0/10、
  knowledge 1/10、sales 5/10。原始轨迹在
  `baseline_qwen_heldout_outputs.jsonl`，逐样本通过/失败与既有指标在
  `baseline_qwen_heldout.json`。
- 全部 2,250 条 train 轨迹的真实 Qwen 渲染 token 分布：
  p50=5065、p95=6134、p99=6450、max=8499。候选验证结果：
  4096 截断 58.8%，5632 截断 19.6% 且最长样本 peak reserved=28336.0 MiB，
  6144 虽截断 4.5% 但只剩约 2433 MiB reserved-memory 余量，未选用。
- 训练配置现锁定：`max_seq_len=5632`、micro batch=1、gradient accumulation=16
  （effective batch=16）、epochs=3、learning rate=2e-4、warmup ratio=0.03、
  NF4/double quant/float16、`fp16=true`、`bf16=false`、CUDA autocast=float16、
  gradient checkpointing=true。该配置的回归测试明确锁定 FP16 和 max_seq_len。
- 最终 transient profile：最长样本 5632-token forward/backward loss 有限；
  p95 样本在 1 warmup 后 4 个稳态微步均值=6.2521s，3 epoch 的 6,750 微步/
  422 optimizer updates 估算为 13.48h（含15%余量）。未执行 optimizer step，
  未保存 adapter/checkpoint。报告：
  `sft_token_length_report.json`、`sft_step_profile_report.json`、
  `sft_sequence_capacity_probes.json`。
- 最终 `train_qlora_sft.py --dry-run` 在锁定 `max_seq_len=5632` 与
  `autocast_dtype=float16` 下复核通过：loss=0.8078994154930115（有限）、
  peak reserved=27540.0 MiB，仍无 checkpoint。报告：
  `sft_dry_run_gpu_final_report.json`。
- 启动计划已写入 `docs/sft_training_launch_plan.md`。当前 `--train` 命令仍会
  fail-closed，直到用户明确批准；正式 SFT、GRPO、LoRA adapter 和 held-out
  对比仍未启动。

## 2026-07-24 SFT 启动前监督截断与评测预检

- 使用与 `train_qlora_sft._masked_batch` 相同的本地 Qwen tokenizer offset
  口径逐条审计原 train 中超过 5632 tokens 的 440 条。结果为：
  监督完整 0、部分截断 440、全部截断 0；即 440/440 都会把 assistant
  tool-call 或最终答案 labels 截断，原配置不可直接训练。
- 已 fail-closed 隔离全部 440 条 train 毒性样本：
  compare 58（剩 392）、deep_search 139（剩 311）、recommend 243
  （剩 207），customer_service/sales 各排除 0（各剩 450）。active train
  从 2250 降至 1810。为避免 checkpoint selection 的 validation loss 同样被
  不完整 labels 污染，额外隔离 validation 47 条，active validation=203。
  active 两个 split 的监督截断样本均为 0。逐条证据：
  `truncation_supervision_audit.json`；隔离行：
  `truncated_excluded.jsonl`。
- 过滤后 active train token p50/p95/p99/max 为
  3864/5544/5617/5632；validation 为 4043/5502/5572/5626。复用已测
  5632-token 微步 6.2521s 作为保守上界，3 epochs 现为 5430 微步/
  340 optimizer updates，wall-clock 估算更新为 10.84h（含15%余量）。
- held-out harness 已冻结为 `qwen-heldout-contract-v1`，manifest 固定
  40 条用例 SHA-256、`task14.v3`、max_steps=8、greedy、max_new_tokens=512、
  `</tool_call>` 停止边界、严格 XML parser、真实工具执行/observation 回灌和
  contract+grounding 评分。运行时与 manifest 任一漂移均 fail-closed。
- 基线 34 个失败的重叠归因：格式/协议解析 28、选错工具 8、参数错 10、
  实质 grounding 3；“最终答案缺失”只归协议失败，不重复计入 grounding。
  互斥主因：格式/协议 28、参数 6。recommend 10/10 均以
  格式/协议为主因；compare 为格式/协议 7、参数 3。报告：
  `baseline_failure_taxonomy.json`。
- 训练计划已锁定每 epoch 保存 checkpoint 并按 active SFT validation loss
  选择最低点；40 条 held-out 只在 checkpoint 选定后用冻结 harness 评一次。
  不按 held-out 选 checkpoint，否则会把科学锚点污染成 development set。
- 正式 SFT、GRPO 仍未启动；未生成 adapter 或 checkpoint，`--train` 继续
  fail-closed。

## 2026-07-24 首次正式 QLoRA SFT 与单次 held-out 复评

- 用户明确批准后，以授权清单
  `data/model_training/sft_training_authorization.json` 启动正式训练。
  训练入口在模型加载前锁定并复核 config/train/validation/held-out/
  reward-visible/cu118 lock/harness SHA-256、`local_files_only=True` 和
  `.venv-train` 解释器；backend `.venv` 未安装或修改训练依赖。
- 正式训练完成 3 epochs：5,430 micro-steps、342 optimizer updates，
  总耗时 25,453.6691s（约7.07h）。峰值显存 allocated=24895.9 MiB、
  reserved=31152.0 MiB。全程 OOM=0、NaN/inf=0，失败日志未生成。
- 三个 active-validation eval_loss：
  epoch1=`0.5016344750456034`、epoch2=`0.4639921804587242`、
  epoch3=`0.4557686771078063`。仅按最低 validation loss 选择
  `checkpoint-epoch-3`，并固化 `checkpoints/sft/best_adapter`；40 条
  held-out 未参与 checkpoint 选择。
- 最优 adapter 锁定后，位级冻结 harness 对 40 条 held-out 已且仅执行 1 次。
  严格 contract+grounding 总分 `0/40=0.0%`，相对原生基线
  `6/40=15.0%` 回退15个百分点。recommend、compare、knowledge、sales
  均为 `0/10`；recommend 未获得提升。
- 40/40 失败的互斥主因均为 format/protocol：invalid terminal JSON 22、
  invalid tool-call XML 7、无终态 11。离线 evaluator 未发现 mandatory tool
  顺序或参数 schema 错误，但 strict terminal schema validity 为0/40，
  所以不能把工具调用子项包装成整体合约成功。
- 终态监督审计确认 active train 1810/1810、validation 203/203 的最终
  assistant 都是自然语言，严格 `{"answer","mentioned_models"}` JSON 样本为0。
  这与复评终态协议全面失败一致，但不宣称已完成唯一因果证明。
- 证据：`sft_training_steps.jsonl`、`sft_training_report.json`、
  `sft_heldout_outputs.jsonl`、`sft_heldout_failures.jsonl`、
  `sft_heldout_report.json`、`sft_heldout_failure_taxonomy.json`、
  `sft_terminal_contract_audit.json`。
- 简历口径：本轮不得声称“工具调用契约/结构化协议合规提升”、决策能力提升或
  优于云端。可如实描述为完成可复现 QLoRA 训练与独立 held-out 审计，并发现
  SFT 终态监督与生产评测协议错配导致严格合约回退。GRPO 未启动。
