# GRPO Data Asset Inventory

Status: read-only inventory. No data was generated, no split was changed, no
manifest was frozen, no training was started, `reward_fn.py` was not read or
touched, and held-out/final isolation logic was not changed.

## 1. Executive Answer

The formal SFT stage froze **2,500** externally audited teacher trajectories:

| SFT source intent | frozen source rows | active train | active eval | excluded for supervision truncation |
|---|---:|---:|---:|---:|
| recommend | 500 | 207 | 22 | 271 |
| compare | 500 | 392 | 46 | 62 |
| customer_service | 500 | 450 | 50 | 0 |
| deep_search | 500 | 311 | 35 | 154 |
| sales | 500 | 450 | 50 | 0 |
| **total** | **2500** | **1810** | **203** | **487** |

Actual SFT gradient/eval inputs:

- SFT train: `data/model_training/sft_train.jsonl`, 1,810 rows.
- SFT validation: `data/model_training/sft_val.jsonl`, 203 rows.
- Truncation-excluded rows: `data/model_training/truncated_excluded.jsonl`,
  487 rows. These were not fed to SFT.

Current GRPO reward-visible set:

- Source pool: `data/model_training/eval/reward_visible.jsonl`, 20 rows.
- Split: 16 train + 4 dev.
- Current unused rows inside that 20-row reward-visible file: **0**.

Potential GRPO expansion pool:

- The deterministic GRPO/SFT query generator can enumerate 51,938 canonical
  prompt candidates across recommend/compare/knowledge/sales.
- To avoid the historical `candidate_index <=650` surface, the conservative
  tail-only pool is 49,338 candidates.
- The frozen GRPO final-40 is selected from that tail and must be excluded.
- Therefore the conservative "could be used for GRPO but is currently idle"
  count is:

```text
49,338 tail candidates - 40 GRPO final = 49,298 idle candidates
```

This is the key inventory number for future GRPO expansion: **49,298**.

If using the full generator surface rather than the conservative tail-only
surface, the corresponding count is `51,938 - 40 = 51,898`; this document
recommends using the conservative 49,298 count because it avoids reusing the
historical first-650 candidate surface.

## 2. Primary Frozen Assets

### 2.1 SFT Frozen Source Shards

These are the five formal SFT source shards. They are Qwen tool-call JSONL
trajectories and are also duplicated into active train/eval/excluded files after
split and truncation audit.

| path | format | rows | intent distribution | sha256 |
|---|---|---:|---|---|
| `data/model_training/sft_freeze/shards/recommend.jsonl` | JSONL | 500 | recommend 500 | `8d3935aca8b64ae0b6713004473e0d9b8c4c846ef9f8950f8a9a4067d81a4f3c` |
| `data/model_training/sft_freeze/shards/compare.jsonl` | JSONL | 500 | compare 500 | `0abfcf7b2382ad1703ddeb571f5bf01119b46c2399a572556ad0b53e7b6cff37` |
| `data/model_training/sft_freeze/shards/customer_service.jsonl` | JSONL | 500 | customer_service 500 | `937e6b0e8eaf2f30fe854175e05d1f376d66a949725cde932626099b36be7abc` |
| `data/model_training/sft_freeze/shards/deep_search.jsonl` | JSONL | 500 | deep_search 500 | `da76d264ac6125fe38644c426e4a30692ea738dc6e2bd5e50aa1272ead201989` |
| `data/model_training/sft_freeze/shards/sales.jsonl` | JSONL | 500 | sales 500 | `8f07643b64fe8c1662585799719362464dbc0df4fa7feb513137e02d6414ad6e` |

Formal SFT accepted-source files recorded by
`data/model_training/500perintent_final_full_audit.json`:

| source intent | path | rows | sha256 |
|---|---|---:|---|
| recommend | `data/model_training/teacher_decision_500perintent_recommend_sft.jsonl` | 500 | `885e112261fad95920bfd5c1ff4f311fe42fb42d16eddf54f81f7b9381b8780e` |
| compare | `data/model_training/teacher_decision_500perintent_compare_named_lookup_v3_sft.jsonl` | 500 | `d2cdaad21bc470d8f2701b0fb58bd222f1720d74bb850a24d93f6100f3516610` |
| customer_service | `data/model_training/teacher_decision_500perintent_customer_service_rerun_v3_sft.jsonl` | 500 | `c9c19ef5b1d41e66a54dc725b654485e8ff21fb1398ea5936c7f2eaed7713d94` |
| deep_search | `data/model_training/teacher_decision_500perintent_deep_search_rerun_v2_sft.jsonl` | 500 | `e94b2e51d5948eba3bfbb76cc4a2d7fc11b11643bdf77ac16cf75323d0d62a00` |
| sales | `data/model_training/teacher_decision_500perintent_sales_policy_rerun_v3_sft.jsonl` | 500 | `d3c173a5daabf4d5f67a23a41924eb4167eb74d6f74afe832b5437849895118c` |

### 2.2 Active SFT Files

| path | format | rows | intent distribution | sha256 |
|---|---|---:|---|---|
| `data/model_training/sft_train.jsonl` | JSONL | 1810 | compare 392; customer_service 450; deep_search 311; recommend 207; sales 450 | `f6bd8e587a79abaca62d8159521057e34da87c2eb21865cf0d771c131d31ebd3` |
| `data/model_training/sft_val.jsonl` | JSONL | 203 | compare 46; customer_service 50; deep_search 35; recommend 22; sales 50 | `668f8bf3653256a6b51e149938dad850cfbb9a91813de7b35f2a88861ba748cf` |
| `data/model_training/truncated_excluded.jsonl` | JSONL | 487 | compare 62; deep_search 154; recommend 271 | `918ecfb06116c3269b0a25dba329977b0445b7ff06d1e9315a635473d3781171` |

SFT split manifest:

- `data/model_training/sft_freeze/split_manifest.json`
- Split seed: `20260724`
- Eval fraction: `0.1`
- Source total: 2,500
- Pre-audit train/eval: 2,250 / 250
- Active train/eval after supervision-truncation exclusion: 1,810 / 203
- Excluded train/eval: 440 / 47

### 2.3 Reward, Held-Out, and Final Eval Files

| path | format | rows | intent distribution | sha256 |
|---|---|---:|---|---|
| `data/model_training/eval/reward_visible.jsonl` | JSONL | 20 | recommend 5; compare 5; knowledge 5; sales 5 | `98117983647ab5f3618f96831612bca7984af09db85ef434485f42901b391c5e` |
| `data/model_training/grpo/reward_train_16.jsonl` | JSONL | 16 | recommend 4; compare 4; knowledge 4; sales 4 | `0390c0ee32156c02c84b08d0bc96191b0a7040a57fcaab969112f998b4539cc7` |
| `data/model_training/grpo/reward_dev_4.jsonl` | JSONL | 4 | recommend 1; compare 1; knowledge 1; sales 1 | `bb90df011ff62cc10daf44f6eb17bbca9232d356c2695b3e2af43dcc348ae790` |
| `data/model_training/eval/held_out.jsonl` | JSONL | 40 | recommend 10; compare 10; knowledge 10; sales 10 | `964fc352d1c83fa2738042d377c8070d6e355c51a5ddbb36c1fc9a9b99771a79` |
| `data/model_training/eval/grpo_final_held_out.jsonl` | JSONL | 40 | recommend 10; compare 10; knowledge 10; sales 10 | `0fd611ffdfed27adb50615e76a1fd8b0f43a5330676e22f19a27b764d2c4678c` |
| `data/model_training/grpo/grpo_signal_probe_input_manifest.json` | JSON | 20 cases | recommend 5; compare 5; knowledge 5; sales 5 | `bf1f894cc90c3eb503e08dcd06b661315388262f11a413ea2e00fd0604d161ef` |
| `data/model_training/grpo/grpo_signal_probe_raw.jsonl` | JSONL | 20 groups | recommend 5; compare 5; knowledge 5; sales 5 | `1a0dc606c0800c0531e4c4c375375aaebd3141b0542b176bd71c3366e72da07a` |

## 3. Split and Provenance

### 3.1 SFT Train / Eval

Actual SFT inputs:

- Train: `data/model_training/sft_train.jsonl`, 1,810 rows.
- Validation: `data/model_training/sft_val.jsonl`, 203 rows.

Training report confirms:

- `train_rows`: 1,810
- `validation_rows`: 203
- SFT ran 3 epochs over 1,810 micro-steps per epoch.
- Validation used 203 examples per epoch.

The 487 rows in `truncated_excluded.jsonl` were excluded because assistant
supervision tokens would be truncated at `max_seq_len=5632`. They are frozen
source rows but were not fed to SFT.

### 3.2 Reward 16/4 Split

The reward set was cut from:

- Source: `data/model_training/eval/reward_visible.jsonl`
- Source SHA: `98117983647ab5f3618f96831612bca7984af09db85ef434485f42901b391c5e`
- Source rows: 20

Split rule from `data/model_training/grpo/reward_train_dev_manifest.json`:

```text
Within each intent, select the row with the lowest
SHA256(seed:intent:normalized_id:normalized_query) as dev;
all other rows are train.
```

Seed: `grpo-reward-split-v1`

Dev rows:

- `reward-recommend-002`
- `reward-compare-004`
- `reward-knowledge-003`
- `reward-sales-004`

All other 16 rows became gradient-eligible train rows. The train/dev union
exactly reconstructs reward-visible 20, so no row in the current
reward-visible 20 is unused.

### 3.3 Held-Out 40

`data/model_training/eval/held_out.jsonl` is the SFT/product held-out set:

- Rows: 40
- Distribution: 10 per recommend/compare/knowledge/sales
- SHA: `964fc352d1c83fa2738042d377c8070d6e355c51a5ddbb36c1fc9a9b99771a79`
- Source: manually maintained structured agent evaluation cases, independent
  from SeedPro teacher SFT trajectories.
- Use: read-only held-out evaluation; not reward, tuning, training, or early
  stopping.

### 3.4 GRPO Final 40

`data/model_training/eval/grpo_final_held_out.jsonl` is the GRPO final set:

- Rows: 40
- Distribution: 10 per recommend/compare/knowledge/sales
- Dataset SHA: `0fd611ffdfed27adb50615e76a1fd8b0f43a5330676e22f19a27b764d2c4678c`
- Manifest: `data/model_training/eval/grpo_final_held_out_manifest.json`
- Manifest SHA: `685f81078150dfacf1ec62d880bebf3df39cd379698f4ff201b2cc270c21c4fe`
- Selection: deterministic generator candidates with `candidate_index > 650`,
  sorted by SHA256 seed `grpo-final-v1`, first 10 per intent.
- Use: final evaluation only; forbidden for rollout, reward, tuning,
  checkpoint selection, or early stopping.

## 4. Intent Distribution

### 4.1 SFT Formal Pool

SFT formal source labels are five-way, not the four GRPO canonical intents:

| label | source | active train | active eval | excluded |
|---|---:|---:|---:|---:|
| recommend | 500 | 207 | 22 | 271 |
| compare | 500 | 392 | 46 | 62 |
| customer_service | 500 | 450 | 50 | 0 |
| deep_search | 500 | 311 | 35 | 154 |
| sales | 500 | 450 | 50 | 0 |

For GRPO's four-way canonical inventory, `knowledge` maps to the
customer-service/knowledge-style query source used by the final-eval generator,
not directly to the SFT label `deep_search`.

### 4.2 GRPO / Eval Four-Way Pools

| subset | recommend | compare | knowledge | sales | total |
|---|---:|---:|---:|---:|---:|
| reward_visible | 5 | 5 | 5 | 5 | 20 |
| reward_train_16 | 4 | 4 | 4 | 4 | 16 |
| reward_dev_4 | 1 | 1 | 1 | 1 | 4 |
| held_out_40 | 10 | 10 | 10 | 10 | 40 |
| grpo_final_40 | 10 | 10 | 10 | 10 | 40 |
| deterministic generator full surface | 41472 | 666 | 4900 | 4900 | 51938 |
| deterministic generator tail `>650` | 40822 | 16 | 4250 | 4250 | 49338 |
| tail after excluding final-40 | 40812 | 6 | 4240 | 4240 | 49298 |

## 5. Knowledge Availability

The reward-visible set contains 5 knowledge rows:

| id | query | expected tool | allowed model |
|---|---|---|---|
| `reward-knowledge-001` | 宋PLUS DM-i在没有家充条件下的补能与油耗表现怎样 | `retrieve_knowledge_base` | 比亚迪 宋PLUS DM-i |
| `reward-knowledge-002` | 小鹏G6的800V快充对实际长途出行有什么帮助 | `retrieve_knowledge_base` | 小鹏 G6 |
| `reward-knowledge-003` | 蔚来ES6的换电体系适合哪些用车场景 | `retrieve_knowledge_base` | 蔚来 ES6 |
| `reward-knowledge-004` | 理想L7的增程方案在高速和城市里分别有什么特点 | `retrieve_knowledge_base` | 理想 L7 |
| `reward-knowledge-005` | 大众ID.4 CROZZ的底盘与安全配置有哪些优势 | `retrieve_knowledge_base` | 大众 ID.4 CROZZ |

Signal probe evidence from `grpo_signal_probe_raw.jsonl`:

| id | split | passed completions / 8 | nonzero reward completions / 8 | gate summary |
|---|---|---:|---:|---|
| `reward-knowledge-001` | train | 0 | 0 | insufficient_anchor_bound_claims 8 |
| `reward-knowledge-002` | train | 0 | 0 | unsupported_target_value 6; insufficient_anchor_bound_claims 2 |
| `reward-knowledge-003` | dev | 0 | 0 | insufficient_anchor_bound_claims 6; unsupported_target_value 2 |
| `reward-knowledge-004` | train | 0 | 0 | insufficient_anchor_bound_claims 8 |
| `reward-knowledge-005` | train | 3 | 3 | passed 3; insufficient_anchor_bound_claims 4; unsupported_target_value 1 |

Conclusion:

- Knowledge data is not absent: there are 5 reward-visible knowledge prompts and
  4,250 conservative tail candidates from the generator.
- The base/SFT model can produce anchor-bound knowledge answers in at least one
  existing sample (`reward-knowledge-005`, 3/8 passed).
- The blind spot is therefore not pure data absence. It is a combination of
  sparse knowledge signal, hard gate strictness, and weak model behavior on most
  current knowledge prompts.

## 6. Isolation Feasibility

Existing static isolation report:

- `data/model_training/eval/grpo_final_isolation_report.json`
- Status: `passed`
- reward_visible vs held_out: ID overlap 0, query overlap 0
- reward_visible vs grpo_final: ID overlap 0, query overlap 0
- held_out vs grpo_final: ID overlap 0, query overlap 0
- grpo_final vs SFT source surface: ID overlap 0, query overlap 0

Additional read-only in-memory generator SHA check:

| pool | count | held_out query overlap | final query overlap | reward_visible query overlap |
|---|---:|---:|---:|---:|
| full generator surface | 51,938 | 0 | 40 | not used for exclusion |
| conservative generator tail `candidate_index >650` | 49,338 | 0 | 40 | 0 |

Per intent, tail overlap with final is exactly 10 each:

| intent | tail count | held_out overlap | final overlap | reward_visible overlap |
|---|---:|---:|---:|---:|
| recommend | 40,822 | 0 | 10 | 0 |
| compare | 16 | 0 | 10 | 0 |
| knowledge | 4,250 | 0 | 10 | 0 |
| sales | 4,250 | 0 | 10 | 0 |

Thus, expanding GRPO prompts to the conservative pool:

```text
tail_after_650_total = 49,338
excluded_final_40 = 40
excluded_held_out_40_overlap = 0
candidate_count_after_exclusion = 49,298
```

The held-out 40 is independent and has zero query overlap with the generator
tail, but it remains an explicit forbidden set.

## 7. Historical / Intermediate JSONL Assets

These files exist in the repo and are synthetic-generation artifacts, but they
are not the formal active SFT train/eval split unless explicitly listed above.
They are included here for asset completeness.

| path | rows | intent distribution |
|---|---:|---|
| `data/model_training/pilot_sft.jsonl` | 1 | unlabeled 1 |
| `data/model_training/teacher_decision_pilot_sft.jsonl` | 1 | unlabeled 1 |
| `data/model_training/teacher_decision_20perintent_sft.jsonl` | 100 | compare 20; customer_service 20; deep_search 20; recommend 20; sales 20 |
| `data/model_training/teacher_decision_20perintent_compare_named_lookup_rerun_v1.jsonl` | 14 | compare 14 |
| `data/model_training/teacher_decision_20perintent_compare_named_lookup_rerun_v1_failures.jsonl` | 6 | compare 6 |
| `data/model_training/teacher_decision_20perintent_compare_named_lookup_rerun_v2.jsonl` | 20 | compare 20 |
| `data/model_training/teacher_decision_20perintent_compare_named_lookup_rerun_v3.jsonl` | 20 | compare 20 |
| `data/model_training/teacher_decision_20perintent_compare_named_lookup_rerun_v4.jsonl` | 19 | compare 19 |
| `data/model_training/teacher_decision_20perintent_compare_named_lookup_rerun_v4_failures.jsonl` | 2 | compare 2 |
| `data/model_training/teacher_decision_20perintent_compare_named_lookup_rerun_v5.jsonl` | 20 | compare 20 |
| `data/model_training/teacher_decision_20perintent_customer_service_policy_rerun_v3.jsonl` | 20 | customer_service 20 |
| `data/model_training/teacher_decision_20perintent_deep_search_hardprompt_rerun_v1.jsonl` | 20 | deep_search 20 |
| `data/model_training/teacher_decision_20perintent_deep_search_hardprompt_rerun_v2.jsonl` | 20 | deep_search 20 |
| `data/model_training/teacher_decision_20perintent_failures.jsonl` | 5 | deep_search 5 |
| `data/model_training/teacher_decision_20perintent_recommend_hardprompt_rerun.jsonl` | 17 | recommend 17 |
| `data/model_training/teacher_decision_20perintent_recommend_hardprompt_rerun_failures.jsonl` | 3 | recommend 3 |
| `data/model_training/teacher_decision_20perintent_recommend_hardprompt_rerun_v2.jsonl` | 20 | recommend 20 |
| `data/model_training/teacher_decision_20perintent_recommend_hardprompt_rerun_v3.jsonl` | 20 | recommend 20 |
| `data/model_training/teacher_decision_20perintent_recommend_hardprompt_rerun_v3_failures.jsonl` | 2 | recommend 2 |
| `data/model_training/teacher_decision_20perintent_sales_rerun.jsonl` | 19 | sales 19 |
| `data/model_training/teacher_decision_20perintent_sales_rerun_failures.jsonl` | 1 | sales 1 |
| `data/model_training/teacher_decision_20perintent_sales_rerun_v2.jsonl` | 20 | sales 20 |
| `data/model_training/teacher_decision_500perintent_compare_sft.jsonl` | 500 | compare 500 |
| `data/model_training/teacher_decision_500perintent_compare_failures.jsonl` | 16 | compare 16 |
| `data/model_training/teacher_decision_500perintent_compare_named_lookup_v2_sft.jsonl` | 29 | compare 29 |
| `data/model_training/teacher_decision_500perintent_compare_named_lookup_v3_sft.jsonl` | 500 | compare 500 |
| `data/model_training/teacher_decision_500perintent_compare_named_lookup_v3_failures.jsonl` | 4 | compare 4 |
| `data/model_training/teacher_decision_500perintent_customer_service_sft.jsonl` | 80 | customer_service 80 |
| `data/model_training/teacher_decision_500perintent_customer_service_rerun_v2_sft.jsonl` | 359 | customer_service 359 |
| `data/model_training/teacher_decision_500perintent_customer_service_rerun_v2_failures.jsonl` | 1 | customer_service 1 |
| `data/model_training/teacher_decision_500perintent_customer_service_rerun_v3_sft.jsonl` | 500 | customer_service 500 |
| `data/model_training/teacher_decision_500perintent_customer_service_rerun_v3_failures.jsonl` | 1 | customer_service 1 |
| `data/model_training/teacher_decision_500perintent_deep_search_rerun_v2_sft.jsonl` | 500 | deep_search 500 |
| `data/model_training/teacher_decision_500perintent_deep_search_rerun_v2_failures.jsonl` | 2 | deep_search 2 |
| `data/model_training/teacher_decision_500perintent_recommend_sft.jsonl` | 500 | recommend 500 |
| `data/model_training/teacher_decision_500perintent_recommend_failures.jsonl` | 17 | recommend 17 |
| `data/model_training/teacher_decision_500perintent_sales_policy_rerun_v3_sft.jsonl` | 500 | sales 500 |
| `data/model_training/teacher_decision_500perintent_sales_policy_rerun_v3_failures.jsonl` | 7 | sales 7 |

## 8. Final Key Number

Under the conservative, isolation-preserving expansion policy that avoids the
historical first-650 candidate surface:

```text
GRPO-idle synthetic prompt candidates = 49,298
```

This is the number of deterministic generator-tail prompt candidates that are
not held-out/final and are not currently used by the 16/4 reward split.
