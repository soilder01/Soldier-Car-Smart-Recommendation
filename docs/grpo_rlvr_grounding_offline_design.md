# GRPO/RLVR Grounding 离线设计

## 0. 状态与非目标

- 状态：`DESIGN_ONLY / BLOCKED`。
- 本文不是训练授权，不解锁 `training/grpo/train_grpo.py`。
- 本阶段不训练、不推理、不执行工具，不修改 SFT 数据、权重、held-out、harness 或
  已冻结 v2 manifest。
- SFT 在生产 prompt + v2 执行契约上已经达到 `40/40`。GRPO 不再为格式、工具顺序
  或停机提供正向分数，只把这些指标作为硬门禁。
- GRPO 的唯一优化目标是：终答中的可验证事实能否由本轮真实工具 observation
  确定性支持。
- 不优化“读起来好”“像专家”“更有说服力”等不可程序核验的主观指标。

## 1. Grounding reward 的输入与信任边界

每个 reward 样本必须包含完整、不可变的本轮轨迹：

1. `prompt_id`、规范化 query、intent。
2. assistant 工具调用。
3. 按 `tool_call_id` 对齐的真实 tool observation。
4. terminal assistant answer。
5. frozen reward spec SHA、工具 schema SHA、证据源快照 SHA。

### 1.1 可以作为事实证据的内容

- `search_and_rank_vehicles` 的结构化返回：
  `full_name / energy / vehicle_type / specs / reasons / cautions /
  highlights / weaknesses / energy_evidence`。
- `retrieve_knowledge_base` 返回的 `source / domain / content` 原始片段。
- `search_web_info` 返回的非空 `title / url / content`。必须保存当轮原始响应与
  SHA，不能在 reward 时重新联网。
- 用户输入只能证明用户自身约束，例如预算、家庭人数、使用场景；不能证明车型属性。

### 1.2 明确不作为事实证据的内容

- 工具调用参数。模型把答案写进 `query` 或其他参数不构成 grounding。
- assistant 的中间文本或最终答案本身。
- `extract_user_profile` 对车型事实的推断。
- `generate_sales_talk` 返回的车型属性或政策表述。该工具是派生话术，不是独立事实源。
- 工具错误对象、空结果、未实际执行的调用。
- reward 运行时重新检索到的内容。只能使用答案生成前已经存在于本轮轨迹中的证据。

## 2. 确定性 evidence ledger

reward 前先把 observation 编译为只读账本：

```text
EvidenceKey = (
    canonical_entity,     # 例如 "小鹏 G6"
    canonical_attribute,  # 例如 "specs.cltc_range"
    canonical_value,      # 例如 "755km"
    source_tool,
    source_call_id,
    source_locator,       # JSON path 或 source/url + span
)
```

规则：

- 车型名用本地 catalog 的唯一 full brand-model alias 解析；歧义时不入账。
- JSON 数值使用 `Decimal`，单位使用白名单规范化；不做推测性单位换算。
- `price_range` 必须整段匹配，不能用区间端点反推“起售价”。
- 同一实体、属性出现冲突值时标记 `conflict`，不能任选一个给分。
- KB/web 文本保留来源和字符 span。只有同一片段中的精确规范化文本或批准的
  field alias 才能支持 claim。
- observation 去重按 `(source, url, content_sha256)`，重复调用不会增加证据量。
- 证据账本、终答和 reward spec 一起进入 cache key：

```text
SHA256(
  reward_spec_sha256
  + prompt_id
  + canonical_evidence_ledger
  + terminal_answer
)
```

现有只按 prompt/completion 缓存的设计不足以防止 observation 漂移，正式实现时必须
升级，不能沿用旧 key。

## 3. 可验证 claim 提取

### 3.1 默认纳入 reward 的 claim

1. 带单位数值：价格、续航、电池、快充、轴距、后备箱、销量等。
2. 结构化枚举：能源类型、车型类型、座位数、驱动形式、ADAS 等级、座舱名称。
3. 明确政策/权益/保修/交付声明。
4. 明确来源引用：`[n]`、source、title、URL。
5. 明确车型归因：某个属性属于哪台车。
6. query 指定的车型、维度和对比对象是否得到覆盖。

提取顺序：

1. 解析 Markdown 表格，建立列车型与行属性绑定。
2. 解析项目列表和普通句子中的车型 mention。
3. 用字段 label、单位和 catalog alias 提取
   `(entity, attribute, value, answer_span)`。
4. 扫描所有剩余数字/单位；无法绑定实体或属性的 hard claim 仍记为未支持，不能忽略。
5. 重复 claim 按唯一 `(entity, attribute, value)` 计一次，并单独计算重复率。

### 3.2 确定性支持条件

- 数值/枚举 claim：实体、属性、规范化值必须同时命中同一 ledger entry。
- 文本 claim：必须是某个 KB/web evidence span 的规范化精确子串，或命中冻结的
  单义 field alias。
- 引用：引用编号必须映射到答案中的来源表；该来源必须存在于当轮 observation，
  且被引用 claim 必须由同一来源支持。
- 政策 claim：必须逐字受 observation 支持；否则只能使用冻结的延期核验模板。
- 同一数值只出现在另一车型下时，当前 claim 判错，禁止跨车型“撞数值”。

### 3.3 默认不纳入 reward 的软维度

以下内容单独记录但不给 reward：

- “更舒适”“更高级”“更适合家庭”等没有结构化依据的主观判断。
- 需要常识推理、多跳因果或语义改写才能判断的 entailment。
- 文风、礼貌、说服力、销售感染力。
- 复杂场景权衡是否“合理”。
- 未经确定性规则覆盖的同义改写。

如未来加入 NLI 或人工评审，必须作为独立实验列，不能悄悄混入主 reward。另一个
LLM 不得作为默认裁判。

## 4. Required obligations，防止靠少说获高分

每条 reward-visible prompt 在训练前必须人工冻结一份
`grounding_obligation_spec`，只描述可程序验证的槽位，不写参考答案。

### 4.1 intent 默认 obligation

- recommend：
  - 覆盖首次有效 `search_and_rank_vehicles` 返回的前 `min(3, N)` 个候选名。
  - 对 query 点名关注维度，覆盖证据中存在的对应字段。
- compare：
  - 覆盖两个 `named_vehicles`。
  - 对每台车覆盖同一组 query 关注字段，不能只写一边。
- knowledge：
  - 至少两个不重复、受 KB/web span 支持的命题。
  - 使用 KB/web 事实时必须有可回指来源。
- sales：
  - 至少一个受证据支持的产品/技术事实。
  - 涉及证据缺失的政策、价格、权益时必须使用延期核验表达。

query 到字段的冻结映射示例：

| query 关键词 | 可验证字段 |
|---|---|
| 价格、预算 | `specs.price_range` |
| 续航 | `specs.cltc_range` |
| 电池 | `specs.battery` |
| 快充、补能 | `specs.fast_charge` 或带来源 KB/web span |
| 空间 | `specs.wheelbase / trunk_volume / seats` |
| 智驾 | `specs.adas_level` |
| 座舱 | `specs.smart_cockpit` |
| 安全 | `specs.safety_score` 或带来源 KB/web span |

obligation 只在对应 evidence 实际存在时进入分母。证据为空的 prompt group 不产生
梯度，不能把“拒答”奖励为正确答案。

### 4.2 Intent-response 硬门禁

新增 `G_intent_response`，不作为低权重尾项。失败直接 `reward=0`：

- compare：
  - 从 query 和首次有效 `model_names` 确定两个目标车型。
  - 两个车型都必须作为正文主体出现，不能只出现在来源列表。
  - 每个车型至少绑定一个受证据支持的 claim 或一个并列表格列。
  - `named_vehicle_missing=true` 时必须逐一说明缺失车型，不能用邻近车型冒充。
- recommend：
  - 必须出现冻结词表中的明确决策操作词，例如“推荐、首选、建议优先、备选”。
  - 操作词必须绑定至少一个首次有效 search 返回的 canonical 车型。
  - 该车型至少有一个受证据支持的推荐理由；纯参数罗列不通过。
- knowledge：
  - 每条 reward-visible spec 预先冻结 1 至 3 个 query 核心 anchor。
  - 答案必须命中 anchor，并包含至少两个不同的 evidence-backed proposition。
- sales：
  - 必须命中 query concern anchor。
  - 必须包含至少一个冻结的沟通动作类别，例如澄清、核验、试驾、下一步跟进。
  - 只讲车型参数但不回应客户异议不通过。

该门禁不判断“建议是否高明”。它只检查预先冻结的实体、query anchor、结构标记和
evidence-backed claim 数量，因此是确定性检查，不引入 LLM 主观裁判。anchor 与
操作词表必须在训练前冻结 SHA，不能根据 dev 或 held-out 表现修改。

#### 4.2.1 可执行的证据确定性绑定

当前实现位于 `training/grpo/reward_fn.py`，不再使用“实体出现 + 至少两句”的
结构代理。可计分证据字段固定为：

```text
EvidenceClaim(
    canonical_entity,
    canonical_attribute,
    canonical_value,
    source_tool,
    source_locator,
    entity_aliases,
    attribute_aliases,
    anchor_tokens,
)

IntentResponseSpec(
    prompt_id,
    intent,
    target_entities,
    query_anchor_tokens,
    query_attribute_anchors,
    minimum_supported_claims,
    decision_tokens,
    communication_action_tokens,
)
```

一条答案命题只有同时满足以下条件才计入 `G_intent_response`：

1. 同一答案命题片段精确命中 evidence 的 canonical 车型或冻结 alias；
2. 同一片段精确命中 canonical 字段或冻结 field alias；
3. 同一片段精确命中 evidence 的 canonical value，不做近似、四舍五入或语义猜测；
4. evidence 的车型属于 `target_entities`；
5. `canonical_attribute` 命中 `query_attribute_anchors`，或 evidence 的
   `anchor_tokens` 与 `query_anchor_tokens` 有确定性交集且该 anchor 出现在命题中。

因此，任意两条格式合法但与 query anchor 无关的证据句不计数。compare 必须对
两个 `target_entities` 分别找到至少一条上述真值匹配；目标车型 + query 字段片段
中出现不属于冻结 evidence 的带单位数值时，直接以
`unsupported_target_value` 失败，即使答案同时包含另一条正确数值也不能通过。
recommend 的决策词还必须在同一原子分句中绑定真实 evidence candidate；
knowledge 的每条计数命题必须与冻结 KB anchor 相交。

确定性正反测试位于
`tests/model_layer/test_grpo_intent_response.py`，覆盖 compare 真值、错数值、
safe-but-vacuous、knowledge 非锚定事实和假推荐对象。

## 5. Reward 公式与硬门禁

### 5.1 硬门禁

```text
G_protocol = frozen_v2_protocol_pass
G_evidence = evidence_ledger_has_verifiable_content
G_answer = terminal_answer_nonempty
           and not_generic_refusal
           and not_tool_json_dump
           and within_length_budget
G_intent_response = deterministic_intent_response_check
G_critical = no_unsupported_numeric_or_enum_claim
             and no_entity_attribute_mismatch
             and no_false_policy_claim
             and no_fake_citation
             and no_evidence_conflict_used_as_fact
```

任一门禁失败：

```text
reward = 0.0
```

协议只做 gate，不再提供加分。这样 GRPO 无法为了 grounding 牺牲已经达到 100% 的
工具顺序、参数和停机能力。

### 5.2 分量

在全部 gate 通过后：

```text
P_ground = supported_unique_claims / all_unique_verifiable_claims
C_required = satisfied_required_obligations / available_required_obligations
C_source = valid_claim_source_links / source_links_required
E_concise = deterministic_concision_score

penalty =
    0.10 * duplicate_claim_ratio
  + 0.15 * excessive_nonfact_copy_ratio
  + 0.10 * verbosity_over_budget_ratio

reward = clamp(
    0.0,
    1.0,
    0.45 * P_ground
  + 0.30 * C_required
  + 0.15 * C_source
  + 0.10 * E_concise
  - penalty
)
```

零分规则：

- `all_unique_verifiable_claims == 0` 时 `P_ground=0`，不能把空话当满分。
- `available_required_obligations == 0` 时该 group 标为不可训练，不用常数 1 填充。
- 所有 8 个 generation reward 相同的 group 跳过 update，避免零方差伪梯度。

### 5.3 伪代码

```python
def grounding_reward(prompt_case, trajectory, frozen_specs):
    assert prompt_case.id in frozen_reward_visible_ids
    assert prompt_case.id not in held_out_ids
    assert prompt_case.id not in grpo_final_ids

    protocol = frozen_v2_protocol_check(prompt_case, trajectory)
    if not protocol.passed:
        return Reward(total=0.0, gate="protocol")

    answer = terminal_answer_only(trajectory)
    ledger = build_evidence_ledger(
        successful_tool_observations_before_terminal(trajectory),
        exclude_tool_arguments=True,
        exclude_derived_sales_talk=True,
    )
    if ledger.verifiable_entries == 0:
        return NonTrainable("no verifiable evidence")

    claims = extract_all_verifiable_claims(answer)
    obligations = instantiate_frozen_obligations(
        frozen_specs[prompt_case.id],
        query=prompt_case.query,
        ledger=ledger,
    )
    matches = match_claims_to_same_entity_attribute_evidence(claims, ledger)

    critical_errors = find_critical_errors(
        claims=claims,
        matches=matches,
        citations=parse_citations(answer),
        conflicts=ledger.conflicts,
    )
    if critical_errors:
        return Reward(total=0.0, gate="critical_grounding")

    if is_generic_refusal(answer) or is_tool_dump(answer):
        return Reward(total=0.0, gate="non_answer")

    if not deterministic_intent_response_check(
        prompt_case,
        answer,
        obligations,
        matches,
    ):
        return Reward(total=0.0, gate="intent_response")

    parts = {
        "ground_precision": supported_unique(matches) / unique(claims),
        "required_coverage": satisfied(obligations, matches) / available(obligations),
        "source_integrity": valid_source_links(answer, matches, ledger),
        "concision": deterministic_concision(answer, matched_claim_spans(matches)),
    }
    penalties = anti_hacking_penalties(answer, claims, ledger)
    total = clamp(
        0.45 * parts["ground_precision"]
        + 0.30 * parts["required_coverage"]
        + 0.15 * parts["source_integrity"]
        + 0.10 * parts["concision"]
        - penalties,
        0.0,
        1.0,
    )
    return Reward(total=total, parts=parts, penalties=penalties)
```

## 6. Reward hacking 防御矩阵

| 攻击 | 表现 | 防御 |
|---|---|---|
| 整段复制证据 | 粘贴 tool JSON/检索片段但不回答 | tool dump gate；去除已匹配事实 span 后计算 5-gram copy ratio；高于阈值零分 |
| 重复复述同一事实 | 同一规格写很多次刷匹配数 | claim 按 `(entity, attribute, value)` 去重；重复率扣分 |
| 拒答规避错误 | 全文“请以官方为准” | generic refusal gate；required obligations coverage；零 claim 不得满分 |
| 把答案塞进工具参数 | query 参数含希望 reward 命中的文字 | 工具参数永不进入 evidence ledger；只认真实 observation |
| 超长啰嗦稀释错误 | 大量正确文本掩盖少量幻觉 | 任一 unsupported hard claim 触发 critical gate；错误不按长度稀释 |
| 跨车型撞数值 | A 车套用 B 车相同数值 | claim 必须同时绑定 entity、attribute、value 与同一 evidence entry |
| 四舍五入/单位改写 | 将区间、续航或销量近似化 | Decimal + 单位白名单；默认要求 exact canonical value，不做近似匹配 |
| 伪造引用 | 堆 `[1][2]` 或真实 URL 但内容不支持 | 引用必须映射当轮 source，且 claim 必须由同一 source span 支持 |
| 重复调用工具刷证据 | 多次检索扩大可匹配文本 | observation 按 source/content SHA 去重；重复调用不增加 reward 分母或分子 |
| 操纵检索 query 回显答案 | 在搜索词中写入欲声称事实 | 参数不入账；web/KB 只取非空 source content，query/title 回显不能单独支持 hard claim |
| 只选最容易写的车型 | 回避 query 指定车型或 top 候选 | obligation 预先绑定 named vehicles 或首次有效 top candidates |
| 利用证据冲突 | 从冲突来源中挑有利值 | ledger 标记冲突；未由高优先级结构化源消解前不得作为事实给分 |
| 空证据下模板化拒答 | 工具无结果时稳定拿保守分 | 整个 prompt group 标为 non-trainable，不给正 reward，也不制造负偏好 |
| 在非终态藏答案 | 工具调用前后输出奖励文本 | reward 只读最后一个正常 terminal answer |

阈值必须在 ignition 前用手工构造的正负 fixtures 冻结，不得看 held-out 或 GRPO final
结果后调整。

## 7. 训练健康诊断与自动 abort

### 7.1 记录频率与原始指标

- 每个 optimizer step 写一行不可覆盖 JSONL。
- 每 `N=5` 个 optimizer step 形成一个 health window。
- step 0 在任何更新前对 train 16 和 dev 4 各运行一次，冻结健康基线。
- 每完成一轮 train 16 后运行一次 dev 4；dev 使用冻结 seed，无梯度、无 optimizer
  state 变更。

每个 window 必须记录：

1. reward：
   - total reward 的 mean/std/p10/p50/p90；
   - `P_ground / C_required / C_source / E_concise` 分项均值；
   - duplicate、nonfact-copy、verbosity 各 penalty 均值；
   - protocol、intent-response、critical-grounding、non-answer gate 通过率。
2. 长度：
   - completion token length 的 min/p10/p50/p90/p95/max；
   - 空答案率、refusal 率、触达 max completion length 的比例。
3. 多样性：
   - 每个 8-generation group 的 normalized unique completion ratio；
   - distinct-2、distinct-4；
   - 完全重复 group 比例。
4. reward 区分度：
   - 每组 reward std；
   - `std < 1e-4` 的 group 比例；
   - advantage mean/std 和 non-finite 数。
5. KL：
   - token 级 `policy_logp - ref_logp` signed mean；
   - 非负 estimator
     `exp(ref_logp-policy_logp) - (ref_logp-policy_logp) - 1`
     的 mean/p50/p95/max；
   - step 0 的 `KL0_mean / KL0_p95`。
6. 运行时：
   - loss、grad norm、learning rate、step time；
   - current/peak allocated/reserved VRAM；
   - OOM、NaN/inf、reward cache hit/miss、evidence SHA drift。

### 7.2 自动 abort 阈值

所有“连续两个 window”均指两个完整的 5-step window。达到 abort 后不允许自动调参
或自动续跑。

| 风险 | 自动 abort 条件 |
|---|---|
| 非有限或 OOM | 任一 loss、grad、reward、KL 出现 NaN/inf，或任一 OOM，立即 abort |
| 协议回退 | protocol pass rate 单 window `<90%` 立即 abort；连续两个 window `<95%` abort |
| 安全废话/答非所问 | 基线固定为 signal probe 的 train_16 共 128 completions，`B_intent=0.6953125`；fail rate `>B_intent+0.10=0.7953125` 连续两个完整 window。绝对 `>10%` 触发器已删除 |
| 伪 reward 上升 | total reward 相对 step 0 提升 `>=0.08`，但 `P_ground` 提升 `<0.01`，连续两个 window |
| 事实质量回退 | `P_ground` 相对 step 0 下降 `>=0.03`，连续两个 window；下降 `>=0.08` 立即 abort |
| 靠简洁度刷分 | total reward 上升部分中，`E_concise` 增益加 penalty 减少贡献 `>60%`，且 `P_ground/C_required` 均未提升，连续两个 window |
| 长度塌缩 | 基线固定为 signal probe train-only `B_length_p50=73.5`；p50 `<0.5*B=36.75` 连续两个 window。绝对 64-token 地板已删除；空答案或 generic refusal `>10%` 仍立即 abort |
| 长度爆炸 | p95 `>486` token，即 512 cap 的 95%，连续两个 window；同时 nonfact-copy ratio `>0.60` 时立即 abort |
| 多样性塌缩 | 超过 50% group 的 unique ratio `<0.25`，或 window distinct-2 `<0.10`，连续两个 window |
| reward 零方差 | 基线固定为 signal probe train-only `B_zero=0.25`；`std <1e-4` 的 group 比例 `>=B_zero+0.20=0.45` 连续两个 window；完整 train sweep `>=B_zero+0.50=0.75` 时立即 abort。绝对 50% 触发器已删除 |
| KL 失控 | mean KL 连续两个 window超过 `max(KL0_mean+0.05, 1.5*KL0_mean)`；单 window 超过 `max(KL0_mean+0.10, 2*KL0_mean)` 立即 abort |
| KL 长尾 | p95 KL 连续两个 window超过 `max(KL0_p95+0.10, 0.30)` |
| dev 反向分叉 | train total reward 上升，同时 dev grounding core 从历史 best 下降 `>=0.05`，连续两次 dev evaluation |
| 数据/证据漂移 | prompt ID/query SHA、reward spec SHA、tool schema SHA、evidence source SHA 或 cache evidence SHA 任一漂移，立即 abort |

其中 `grounding core = 0.6 * P_ground + 0.4 * C_required`，不含简洁度。KL 阈值相对
step 0 定义，是因为 adapter-off reference 为 base model，初始 SFT policy 本身已有
非零 KL。

上述三个相对基线均来自冻结文件
`data/model_training/grpo/grpo_signal_probe_raw.jsonl`，SHA
`1a0dc606c0800c0531e4c4c375375aaebd3141b0542b176bd71c3366e72da07a`，
只统计 train_16，不混入 dev。formal_v1 首窗口 intent-response fail rate 为 0.60，
低于冻结基线 0.6953125，属于改善而非退化；其 absolute 10% abort 被定性为安全
阈值定标错误。该修复只控制是否停止训练，不改变 reward、评分或任何上报指标。

### 7.3 abort 动作

触发任一安全 abort 时必须原子执行：

1. 停止当前 optimizer step；未完成 step 不落 checkpoint。
2. flush 原始 step JSONL、最近两个 health window、四条 dev 明细。
3. 写 `abort_report.json`，包含触发规则、首个违规 step、配置/数据/reward/代码 SHA、
   RNG state、显存与最近 last-good checkpoint 指针。
4. 当前 checkpoint 标记 `quarantined_not_selectable`，不得自动晋级。
5. 释放 rollout/update GPU 进程并以非零状态退出。
6. 写 `resume_allowed=false`；禁止自动降 beta、改长度、改 reward 权重或重启。
7. 只有人工根因审计和新的显式批准才能创建下一次 run ID。

正常 early stop 与安全 abort 分开：

- dev grounding core `min_delta=0.01`、`patience=2`。
- 正常 early stop 只能选择已经保存且全部 gate 健康的 best dev checkpoint。
- safety abort 不能回退后自动续训。

## 8. 数据边界与 checkpoint 选择

### 8.1 唯一可用于 GRPO 的池

- eligible pool：`reward_visible.jsonl` 20 条。
- 不允许从 SFT train/validation、旧 held-out 或新 GRPO final eval 增补 rollout。
- 每个 rollout 必须验证 prompt 的 ID SHA 和 query SHA 均属于 frozen eligible pool。

训练/验证方案已定死为 `16 train + 4 dev`：

- train：
  `data/model_training/grpo/reward_train_16.jsonl`
  - SHA：`0390c0ee32156c02c84b08d0bc96191b0a7040a57fcaab969112f998b4539cc7`
  - 每 intent 4 条，只允许这 16 条产生梯度。
- dev：
  `data/model_training/grpo/reward_dev_4.jsonl`
  - SHA：`bb90df011ff62cc10daf44f6eb17bbca9232d356c2695b3e2af43dcc348ae790`
  - 每 intent 1 条，禁止梯度更新。
  - dev ID：
    `reward-recommend-002 / reward-compare-004 /
    reward-knowledge-003 / reward-sales-004`。
- split manifest：
  `data/model_training/grpo/reward_train_dev_manifest.json`
  - SHA：`d30eedc42ae60ff462680bcf55eb4ca6ce798ebcb76b24ccd6a6314934ad09cb`
- dev 授权 manifest：
  `data/model_training/grpo/grpo_dev_authorization_manifest.json`
  - SHA：`30631cad7a8733739eafef6d285e4e0b3f007e38c1e898c064ebbf0432315bff`
  - 只允许三种只读用途：单向 early stop、触发 abort、训练结束且全部超参锁死后
    checkpoint 选点。
  - early stop 只能减少本次训练，禁止事后回捞更早的差 checkpoint 并重新宣称最优。
  - 任何 dev 指标、分项、gate、轨迹或聚合结果均不得修改 `beta`、学习率、计划/
    最大步数、epoch、generation 参数、optimizer、reward 权重/阈值、abort 阈值、
    数据选择、prompt 或任何其他超参。
  - 违反上述条款时 run 立即失效，禁止 promotion，必须重新授权；不得自动续跑。

选择算法在每个 intent 内取
`SHA256("grpo-reward-split-v1:intent:normalized_id:normalized_query")`
最小者为 dev，其余为 train。16+4 的 ID/query SHA union 必须精确重构冻结
reward-visible 20。

checkpoint 排序使用：

1. dev protocol、intent-response、critical-grounding gate 全通过；
2. 最大 dev grounding core；
3. 更低 mean KL；
4. 更早 step。

上述排序只允许在训练完全结束且所有超参不可更改后执行。total reward 不能单独
决定 checkpoint，因为它含简洁度与 penalty；排序结果也不能反向触发任何超参修改。

不得用旧 held-out 40 或新 GRPO final 40 做 early stopping、超参选择或 reward 阈值
调试。

### 8.2 只读评测集

- `held_out.jsonl` 40 条：已观测科学锚点，只作协议回归记录，不再调参。
- `grpo_final_held_out.jsonl` 40 条：全新最终锚点，训练与选择完成后只运行一次。
- 三套评测集物理分文件，ID SHA 与规范化 query SHA 两两交集必须为 0。

### 8.3 新 GRPO final 来源

- 来源：`generate_500perintent_sft.py` 的确定性候选空间。
- 历史所有生成 manifest 的 `max_candidates=650`。
- 新集只从 `candidate_index > 650` 的未物化尾池选择。
- 选择键：
  `SHA256("grpo-final-v1:intent:index:normalized_query")`。
- recommend / compare / knowledge / sales 各 10 条。
- knowledge 取同工具契约的 customer-service 技术/政策候选并映射为 canonical
  knowledge intent。
- 这些 query 未进入 teacher 请求、SFT 源、SFT train/validation、截断隔离、
  reward-visible、旧 held-out 或历史 query 文件。

冻结文件：

- 数据：`data/model_training/eval/grpo_final_held_out.jsonl`
- 数据 SHA：`0fd611ffdfed27adb50615e76a1fd8b0f43a5330676e22f19a27b764d2c4678c`
- 来源 manifest：`data/model_training/eval/grpo_final_held_out_manifest.json`
- manifest SHA：`685f81078150dfacf1ec62d880bebf3df39cd379698f4ff201b2cc270c21c4fe`

## 9. V100 32GB 可行性初判

结论：**条件可行，但单卡 policy + 独立 reference + vLLM 并发不可行，当前仍未 ready。**

### 9.1 已有实测锚点

- SFT NF4 QLoRA、`max_seq_len=5632`、micro-batch 1 的峰值 reserved：
  `31152 MiB / 32510 MiB`。
- SFT 5,430 micro-step 共约 7.07 小时，均值约 4.7 秒/micro-step。
- 当前 SFT 生产 prompt 推理 40 条耗时约 0.56 小时，均值约 50.7 秒/完整 agent
  trajectory。
- `.venv-train` 当前未安装 `trl` 和 `datasets`；当前
  `transformers 4.44.2 / accelerate 0.33.0` 低于 GRPO 所需版本。

### 9.2 模型驻留方案

禁止方案：

- 同一 V100 同时驻留训练 policy、独立 7B reference 和 vLLM rollout engine。
- 为通过 OOM 自动降低 generation 数、截断监督区或静默 CPU offload。

推荐方案：

1. rollout phase：只加载当前 policy，生成一个 on-policy batch，保存 old logprob、
   完整 trajectory 和 evidence SHA。
2. 卸载 rollout engine。
3. CPU 上批量计算确定性 reward。
4. update phase：加载 NF4 base + trainable LoRA，micro-batch 1，gradient
   checkpointing。
5. 每个 rollout batch 最多做一次预注册 update，然后重新 rollout，保持近似 on-policy。

reference 处理：

- `beta=0` 被禁止。初值锁定为 `beta=0.01`，训练中不得自动调 beta。
- reference 是同一冻结 NF4 base 在 `disable_adapter()` 下的 logits。
- reference pass 使用 `torch.inference_mode()`，只保留 completion token 的
  selected logprob。
- 不创建第二份 base/reference 参数。TRL 0.14.0 的 PEFT 路径也采用
  `ref_model=None + disable_adapter()`；正式入口必须用单测确认没有意外复制模型。
- adapter-off reference 是 pre-SFT base，因此 step 0 本身已有非零 KL；健康阈值
  相对 `KL0` 定义，避免把正常初始差异误判为漂移。
- old policy logprob 在 rollout 时保存，不再驻留一份 old policy。

执行顺序修正：stock TRL 0.14.0 先计算带梯度的 policy logprob，再在该 graph
仍存活时进入 `disable_adapter()` reference forward。原先“reference-first 并释放
临时 logits 后再做 policy forward”的顺序不是 stock 实现，不能继续作为显存实测
口径。若未来覆盖 `compute_loss` 改顺序，必须作为独立实现评审，不能静默宣称仍是
原生 GRPOTrainer。

静态显存核算，单位 GiB：

| 项目 | 保守预算 |
|---|---:|
| NF4 base、量化 metadata、模型 runtime | 5.5 |
| 40,370,176 个 LoRA 参数、gradient、FP32 optimizer state | 0.8 |
| policy activation，micro-batch 1、gradient checkpointing、总序列上限 3,072 | 16-18 |
| adapter-off reference 临时 completion logits/logprob，上限 512 token | 0.6 |
| CUDA context、allocator fragmentation、临时 buffer | 3.0 |
| 合计 | 25.9-27.9 |

V100 可用约 31.7 GiB，静态余量约 3.8-5.8 GiB。该核算成立的前提：

- rollout engine 与 update model 分时，不并发驻留。
- `max_prompt_length<=2560`、`max_completion_length=512`、总序列 `<=3072`。
- 超长完整 trajectory 直接触发预检失败，禁止截断监督区或静默缩短。
- reference 和 policy 顺序 forward，不保留 reference activation。

这是静态可行性证明，不替代 GPU dry-run。

### 9.3 TRL/CUDA 兼容性只读核验

`.venv-train` 上执行了 wheel metadata 检查和 `pip --dry-run`，没有安装包：

- `trl 0.13.0` 不含 `GRPOTrainer`，拒绝。
- 锁定 `trl==0.14.0`，wheel 包含 GRPOTrainer 和 PEFT
  `disable_adapter()` reference 路径。
- 依赖下限要求：
  `transformers>=4.46.0 / accelerate>=0.34.0 / datasets>=2.21.0`。
- dry-run 通过的设计锁：
  - `torch==2.3.1+cu118`
  - `transformers==4.46.3`
  - `tokenizers==0.20.3`
  - `accelerate==0.34.2`
  - `peft==0.12.0`
  - `trl==0.14.0`
  - `datasets==2.21.0`
  - `bitsandbytes==0.43.3`

文件：

- `training/grpo/requirements-cu118-design.lock.txt`
- `data/model_training/grpo/trl_compatibility_static_report.json`

现已按授权创建隔离 `.venv-grpo` 并安装上述锁；`pip check` 通过。本地 tokenizer
在 `transformers 4.46.3 / tokenizers 0.20.3` 下仍能生成正确 Qwen ChatML。
`.venv-train` 的 `pip freeze` SHA 和文件 metadata SHA 在安装前后完全一致，未被
修改。

### 9.4 一步 no-checkpoint smoke 实测

冻结 smoke case：

- 文件：
  `data/model_training/grpo/grpo_one_step_smoke_case.json`
- SHA：
  `341141379dccf0a43b88afae3ba0fe86983d250ca2dee8a084758ddbdec62351`
- prompt：train_16 中的 `reward-compare-001`
- 一组 prompt、8 个 generation、`max_steps=1`、`beta=0.01`、
  `fp16=true / bf16=false`、NF4、SFT adapter 初始化、stock TRL
  `disable_adapter()` reference。

范围限制：TRL 0.14.0 GRPOTrainer 本身只生成 terminal completion，不执行项目的
多轮工具环境。本 smoke 将本地车型库中已经确定性取得的 observation 放入 prompt，
只验证“冻结 evidence 后的终答 rollout -> reward -> KL -> update”，不声称验证
在线工具编排。

实测结果：

| 项目 | 实测 |
|---|---:|
| optimizer step | 1 |
| 8 completion 平均长度 | 24 token |
| train loss | 0.0031548748 |
| gradient norm | 0.0393471 |
| mean KL | 0.3154874444 |
| rollout peak | 9491 MiB / 9.269 GiB |
| rollout headroom | 23019 MiB / 22.479 GiB |
| policy/reference/update phase peak | 12239 MiB |
| 全过程 peak | 12391 MiB |
| checkpoint | 0 |

四个 reward 子项均为有限值，8 个 completion 的
`P_ground / C_required / C_source / E_concise` 均为 `1.0`；总 reward mean
`1.0`，group std `0.0`。因此该 smoke 的 policy advantage 为 0，本次非零 update
只来自 `beta=0.01` 的 KL 项。这个结果足以证明 dtype、autocast、NF4、
adapter-off reference、KL、backward 和 optimizer 工程链路可运行，但不能证明
当前单 prompt 具备有用的 group-relative 学习信号；正式健康门禁仍必须对零方差
group 执行 skip/abort 规则。

进程在 `trainer.train()` 返回后，因报告层读取部分 `Linear4bit.compute_dtype`
属性触发 `AttributeError` 并返回 1。原始失败报告和 terminal log 均保留；该只读
报告 bug 已修复，未重跑第二个 optimizer step。恢复报告明确携带此限定：

- 报告：
  `data/model_training/grpo/grpo_one_step_smoke_report.json`
- 报告 SHA：
  `ab9e206ade775273e9b2b6a5020e8e8903a5d34657d4137c5b6d8d7f3a91b47f`
- terminal log SHA：
  `75162b37e485ccf042ae5074f35feb6591a62af15a531c4e6d94d96dd3c293df`

### 9.5 成本量级

以 `16 train prompts x 8 generations = 128 trajectories` 为一个 train sweep：

- 串行 rollout：按现有 50.7 秒/trajectory，理论约 1.8 小时；考虑 sampling、
  长尾和缓存，估计 `2-3.5 小时`。
- reward：纯 CPU 确定性解析，预计分钟级，不应成为瓶颈。
- policy update：每 trajectory 需要 current/old 相关 logprob 和 backward，
  以及顺序 adapter-off reference forward，估计 `1-2 小时/sweep`。
- dev 4 x 8 固定评估约 `0.4-0.8 小时`，不反向传播。
- 合计约 `3.5-6 小时/sweep`；3 个预注册 sweep 约 `10.5-18 小时`。

以上只是量级估算，不是运行证明。正式训练前仍需独立 `.venv-grpo`、版本锁、
单 trajectory GPU dry-run、8-generation 小组 dry-run 和显存闸门。任何 OOM、
NaN/inf 或 evidence/reward cache SHA 漂移立即停止，不自动降配。

## 10. 点火前必须补齐的静态/运行时门禁

1. 冻结 reward spec、claim parser 规则、单位表、字段 alias、反作弊阈值及 SHA。
2. 为 reward-visible 20 冻结 obligation specs；不能读取两套 held-out。
3. 至少准备以下 deterministic fixtures：
   - exact supported claim；
   - wrong value；
   - right value wrong vehicle；
   - unsupported policy；
   - fake citation；
   - copied tool dump；
   - generic refusal；
   - duplicated facts；
   - evidence conflict；
   - empty evidence。
4. fixtures 必须证明误杀与漏网都在预注册阈值内。
5. 新建 `.venv-grpo` 并锁定 torch/transformers/peft/trl/bitsandbytes 兼容组合。
6. GPU dry-run 必须记录 allocated/reserved、step time、reward parts、KL、advantage
   variance；不得保存 adapter。
7. 校验冻结的 `16 train + 4 dev` union 精确等于 reward-visible 20，dev 无梯度。
8. 训练入口继续 fail-closed，直到用户单独批准点火。

## 11. 当前静态校验结果

- reward-visible 20 vs SFT held-out 40：
  ID SHA overlap 0，normalized query SHA overlap 0。
- reward-visible 20 vs GRPO final 40：
  ID SHA overlap 0，normalized query SHA overlap 0。
- SFT held-out 40 vs GRPO final 40：
  ID SHA overlap 0，normalized query SHA overlap 0。
- GRPO final 40 vs 完整 SFT source surface 2,500：
  ID SHA overlap 0，normalized query SHA overlap 0。
- GRPO final 40 vs 122 个历史 JSON/JSONL、2,761 个唯一 query SHA：
  normalized query SHA overlap 0。
- reward train 16 vs dev 4：
  ID SHA overlap 0，normalized query SHA overlap 0。
- train 16 + dev 4 的 ID/query SHA union：
  精确等于冻结 reward-visible 20。
- train 16 与 dev 4 分别对 SFT held-out 40、GRPO final 40：
  ID SHA overlap 0，normalized query SHA overlap 0。
- `.venv-train` 依赖 dry-run：
  resolver PASS；安装前后 `trl=MISSING / datasets=MISSING`，未修改环境。

静态校验器：

```bash
PYTHONPATH=backend:. .venv/bin/python \
  scripts/validate_grpo_eval_isolation.py \
  --report data/model_training/eval/grpo_final_isolation_report.json
```

该命令只读 JSON/JSONL 并计算 SHA，不导入 torch，不访问 GPU，不执行工具或模型。
