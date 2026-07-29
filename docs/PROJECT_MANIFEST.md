# Project Manifest

This manifest records the reproducibility anchors for the local model deployment
pipeline. It contains paths, SHA-256 digests, and access-state metadata only. It
does not contain API keys, raw endpoint values, or generated answer text.

## Deployment Target

| Item | Value |
|---|---|
| Served model name | `car-7b` |
| Runtime path | adapter-on-base, not merged FP16 |
| Base model | `models/Qwen2.5-7B-Instruct/` |
| Production adapter | `checkpoints/grpo/formal_v4/restart_1/sales_dense_v2/checkpoint-150/` |
| Adapter service entry | `backend/serve_adapter.py` |
| Backend override mechanism | `CHAT_*` environment variables |

## Production Checkpoint

| Artifact | Path | SHA-256 |
|---|---|---|
| GRPO/SalesDenseV2 adapter | `checkpoints/grpo/formal_v4/restart_1/sales_dense_v2/checkpoint-150/adapter_model.safetensors` | `6c0d63aad3c6d817ec80edff3b36df0323a80df16e26f0ae8f3e964388ac3c96` |
| Adapter config | `checkpoints/grpo/formal_v4/restart_1/sales_dense_v2/checkpoint-150/adapter_config.json` | `c1884f76dab70f91e7debd658aec2e5f5ec40dc090f3d900861e1f6eb345a840` |
| Reward function | `training/grpo/reward_fn.py` | `325ad44feb83ec37c35babfed4bddb928cf400788e07735eb4631fc4af6962c8` |

## Dataset And Protocol Anchors

| Artifact | Path | SHA-256 |
|---|---|---|
| SFT split manifest | `data/model_training/sft_freeze/split_manifest.json` | `404d3dc4e5d966939b02bc8ae86be62d1cd70b7ab27dd8d0550a381f015674f8` |
| GRPO reward train/dev manifest | `data/model_training/grpo/reward_train_dev_manifest.json` | `d30eedc42ae60ff462680bcf55eb4ca6ce798ebcb76b24ccd6a6314934ad09cb` |
| GRPO dev authorization manifest | `data/model_training/grpo/grpo_dev_authorization_manifest.json` | `30631cad7a8733739eafef6d285e4e0b3f007e38c1e898c064ebbf0432315bff` |
| GRPO expanded v4 train manifest | `data/model_training/grpo/reward_train_expanded_v4_manifest.json` | `a73090359bf8d23378648a40f8087e688bcd58082d9a15ab9f62daedf0d93d77` |
| Formal v4 authorization | `data/model_training/grpo/grpo_formal_v4_authorization.json` | `9d104337dacedabf1a8e91b01215439a5ff8507a44943511aca47659e9c98e75` |
| Formal v4 preregistered report | `data/model_training/grpo/grpo_formal_v4_preregistered_report.json` | `246439be7c1c8da69dd241a7c163d23d85c312c4b767322fc524c711edf9ead7` |
| Formal v4 fixed dev-4 step0 baseline | `data/model_training/grpo/grpo_formal_v4_fixed_dev4_step0_baseline.json` | `03bc8cb271fd017e61f9fee991030908205358f8f0ff1d96fda48b19f47734ed` |
| SalesDenseV2 manifest | `data/model_training/grpo/formal_v4/restart_1/sales_dense_v2_manifest.json` | `627fd6d91d8ca1619f8310ffe718a1141769d9379eba0b6e9d2ff6f9b0741ed5` |
| SalesDenseV2 train protocol | `data/model_training/grpo/formal_v4/restart_1/sales_dense_v2_train_protocol.json` | `b24f9dd32723087bf1c7d8224f01d350694c3f527d1a123d43130b2a5bbaeb14` |
| Powered dev new-anchor manifest | `data/model_training/grpo/formal_v4/restart_1/powered_dev_newanchor_128_manifest.json` | `d2411b47eca0fcd3063419e6326458c43c06536ba50bc3c1a841d68ced9e8e1e` |
| New-anchor eval protocol | `data/model_training/grpo/formal_v4/restart_1/newanchor_eval_protocol.json` | `3f53319f9f4158dd1e282cbefa9f84dcdf0e295c018023cae8d1557020a94783` |

## Held-Out And Final Evaluation Anchors

| Artifact | Path | SHA-256 |
|---|---|---|
| Held-out-40 frozen eval set | `data/model_training/grpo/formal_v4/restart_1/held_out_40_frozen_eval.jsonl` | `2b4a2e8dff52b9feee12bb451ce630dc0e03661c1e4b8935baf3a78df77013ea` |
| Held-out-40 protocol | `data/model_training/grpo/formal_v4/restart_1/held_out_40_final_protocol.json` | `eb5d6a488205666414dd7b5c3484555a52c7789de958badf572108994662531d` |
| Held-out-40 evaluations | `data/model_training/grpo/formal_v4/restart_1/held_out_40_final_evaluations.jsonl` | `9816f49933fa879c5ef6cf88ad7e3274730e04c54b81a3110fd230fc1a3d8973` |
| Held-out-40 result | `data/model_training/grpo/formal_v4/restart_1/held_out_40_final_result.json` | `822fafd3e37c0289d9a20de4b06db8b1d27c97c490efdeba46e1610a4fa9b270` |
| Held-out-40 access state | `data/model_training/grpo/formal_v4/restart_1/held_out_40_access_state.json` | `8e06ca145680fad4e4bb43ea6f71ae5feb83e83bcf0eade7f8fab80425e6c012` |
| Final-40 protocol | `data/model_training/grpo/formal_v4/restart_1/final40_confirmation_protocol.json` | `634c03ba6c2f7f309a513ed73e6da6bbdba4e8a744eac41de23c91eb65058d22` |
| Final-40 evaluations | `data/model_training/grpo/formal_v4/restart_1/final40_confirmation_evaluations.jsonl` | `f63ffa41f3891fb6082f60e2a961861223da6939192f007486429173cf2925a9` |
| Final-40 result | `data/model_training/grpo/formal_v4/restart_1/final40_confirmation_result.json` | `559cb11ce6e394229861b162d6569d189aa1cbefd192865e86e6d0af25d06c5f` |
| GRPO final-40 source | `data/model_training/eval/grpo_final_held_out.jsonl` | `0fd611ffdfed27adb50615e76a1fd8b0f43a5330676e22f19a27b764d2c4678c` |

## Access State

| Dataset | State | Notes |
|---|---|---|
| held-out-40 | `accessed=true` | Burned exactly once for terminal evaluation. Do not rerun as a selection or tuning set. |
| final-40 | `accessed=true` | Used for one-time checkpoint confirmation before held-out-40. |

## Key Reported Metrics

Held-out-40 terminal comparison:

| Object | recommend | sales | composite |
|---|---:|---:|---:|
| `sales_dense_v2_checkpoint_150` | `0.8458035714` | `0.65625` | `0.7510267857` |
| `checkpoint-300` | `0.6582202381` | `0.575625` | `0.6169226190` |
| `cloud_seedpro_ark_ep_masked` | `0.5475` | `0.579375` | `0.5634375` |

Delta for `sales_dense_v2_checkpoint_150` versus cloud:

| Metric | Delta |
|---|---:|
| recommend | `+0.2983035714` |
| sales | `+0.076875` |
| composite | `+0.1875892857` |

Service parity:

| Test | recommend | sales | composite |
|---|---:|---:|---:|
| Greedy adapter harness | `0.9` | `0.755` | `0.8275` |
| Greedy adapter endpoint | `0.9` | `0.755` | `0.8275` |
| Difference | `0.0` | `0.0` | `0.0` |

## Red Lines

- `reward_fn.py` must remain byte-stable at the SHA listed above.
- Frozen held-out and final evaluation files are read-only.
- Do not commit `.env`, real `CHAT_*`, real `ARK_*`, API keys, tokens, model
  weights, checkpoints, virtual environments, or raw generated answer text.
- `merged_model/` is retained as an audit artifact for the merge-degradation
  finding but is not the serving path and must not be committed to ordinary Git.
