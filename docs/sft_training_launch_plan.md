# SFT Training Launch Plan

## Status

- Phase 1 native base-model held-out baseline: completed.
- Training configuration and transient capacity profile: completed.
- Assistant-supervision truncation audit: completed; toxic rows excluded.
- Frozen held-out harness and failure taxonomy: completed.
- Formal SFT: completed for 3 epochs.
- GRPO: not started.
- LoRA adapter/checkpoints: saved under `checkpoints/sft/`.
- Frozen held-out evaluation: completed exactly once; strict score 0/40.

## Held-Out Baseline

Evaluation source: `data/model_training/eval/held_out.jsonl` only. The
20-row reward-visible set was not read by this baseline.

The local base model ran the existing multi-turn agent contract: it selected
tool calls, the runner executed valid calls against production tool
implementations, returned observations, and the existing evaluator scored the
saved trajectories. A case passes only when terminal protocol, mandatory tool
order, tool arguments, allowed-catalog model constraint, and grounding audit
all pass.

| Intent | Passed | Total | Score |
|---|---:|---:|---:|
| recommend | 0 | 10 | 0.0% |
| compare | 0 | 10 | 0.0% |
| knowledge | 1 | 10 | 10.0% |
| sales | 5 | 10 | 50.0% |
| **Total** | **6** | **40** | **15.0%** |

Raw protocol outputs: `data/model_training/baseline_qwen_heldout_outputs.jsonl`.
Auditable summary: `data/model_training/baseline_qwen_heldout.json`.

Failure taxonomy shows that the zero scores do not have one cause. Among the
34 failed cases, overlapping labels are: format/protocol parse failure 28,
wrong tool 8, argument error 10, and substantive grounding failure 3. The exclusive
primary causes are 28 format/protocol failures and 6 argument failures.
`recommend` is 10/10 format/protocol-primary; `compare` is 7
format/protocol-primary and 3 argument-primary. Full evidence is in
`data/model_training/baseline_failure_taxonomy.json`.

## Supervision Truncation Gate

The original 2,250-row training split contained 440 rows over 5,632 tokens.
Token-offset auditing with the same tokenizer contract as `_masked_batch`
found that all 440 lost assistant supervision tokens:

| Intent | Original train | Excluded | Active train |
|---|---:|---:|---:|
| compare | 450 | 58 | 392 |
| customer_service | 450 | 0 | 450 |
| deep_search | 450 | 139 | 311 |
| recommend | 450 | 243 | 207 |
| sales | 450 | 0 | 450 |
| **Total** | **2,250** | **440** | **1,810** |

The same audit excluded 47 toxic validation rows, leaving 203 active
validation rows. This additional exclusion is required because incomplete
validation labels would also corrupt checkpoint selection. Active train and
validation now contain zero rows with truncated assistant supervision. The
excluded rows remain auditable in `data/model_training/truncated_excluded.jsonl`;
the per-row evidence is in
`data/model_training/truncation_supervision_audit.json`.

The active train p50/p95/p99/max lengths are now
3,864/5,544/5,617/5,632 tokens. The active validation distribution is
4,043/5,502/5,572/5,626 tokens. The exclusion is materially imbalanced:
`recommend` retains only 207/450 original train rows, so this first SFT must
report per-intent results rather than only an aggregate.

## Locked Configuration

| Setting | Locked value | Basis |
|---|---|---|
| Quantization | NF4 4-bit, double quant | Verified on V100 dry-run |
| Compute/autocast | float16 | V100 Volta; `bf16=false` |
| Max sequence length | 5632 | Active splits have zero supervision truncation; 6144 leaves insufficient memory headroom |
| Micro batch size | 1 | Longest-sequence V100 capacity probe |
| Gradient accumulation | 16 | Effective batch size 16 without increasing activation memory |
| Epochs | 3 | Bounded first SFT pass over 1,810 active train trajectories |
| Learning rate | 2e-4 | Conservative QLoRA starting point |
| Warmup ratio | 0.03 | 11 of 342 actual optimizer updates |
| Gradient checkpointing | true | Required for sequence-length memory control |
| Checkpoint cadence | Every epoch | Three bounded recovery/model-selection points |
| Selection metric | Lowest active SFT validation loss | Held-out remains untouched during selection |

## Capacity And Time

Longest-sample probe at 5632 tokens:

- Loss finite: yes, `0.7850887775421143`.
- Peak allocated/reserved: `24398.0 MiB` / `28336.0 MiB`.
- No optimizer step, checkpoint, or adapter save.

The final `train_qlora_sft.py --dry-run` entrypoint also passed with the locked
configuration and explicit `autocast_dtype=float16`: loss
`0.8078994154930115`, peak reserved `27540.0 MiB`. See
`data/model_training/sft_dry_run_gpu_final_report.json`.

P95 representative profile:

- One warmup plus four measured micro-steps.
- Mean forward/backward micro-step: `6.2521s`.
- All four losses finite.
- 1,810 active train rows x 3 epochs = 5,430 micro-steps.
- Gradient accumulation 16 = 340 optimizer updates.
- Estimated model compute: `9.43h`.
- Estimated wall-clock with 15% operational margin: `10.84h`.

This reuses the measured 5,632-token micro-step as a conservative upper-bound
estimate. It excludes held-out evaluation, checkpoint serialization, and
retry time because those are not yet authorized.

## Checkpoint Selection And Held-Out Policy

- Save one checkpoint at the end of each epoch.
- Evaluate active `sft_val.jsonl` loss at each epoch and record the
  train/validation loss curves.
- Select the checkpoint with the lowest validation loss; do not blindly use
  epoch 3.
- After checkpoint selection is final, run the frozen 40-case held-out harness
  exactly once and compare it with the 15.0% base-model score.
- Do not use the 40 held-out cases for checkpoint selection, early stopping,
  prompt changes, or hyperparameter changes.

Evaluating all three checkpoints on the 40 held-out cases and choosing the
best would turn held-out into a development set and invalidate the claimed
15% to post-SFT scientific comparison. A per-epoch agent-contract curve
requires a separate development contract set; the reserved reward-visible
20 cases remain unavailable because they are held for future GRPO.

The frozen harness is
`data/model_training/eval/frozen_qwen_heldout_harness.json`. It locks the
40-case SHA-256, protocol version, eight-turn cap, greedy decoding,
512-token generation cap, `</tool_call>` boundary, strict XML parser, real
tool execution, observation feedback, and contract-plus-grounding score.

## Actual Outcome

- Epoch 1 eval_loss: `0.5016344750456034`.
- Epoch 2 eval_loss: `0.4639921804587242`.
- Epoch 3 eval_loss: `0.4557686771078063`.
- Selected checkpoint: `checkpoint-epoch-3`, using validation loss only.
- Training: 5,430 micro-steps, 342 optimizer updates, 7.07h.
- Peak allocated/reserved: `24895.9/31152.0 MiB`.
- OOM and NaN/inf: none.
- Held-out: `0/40=0.0%`; recommend/compare/knowledge/sales are all `0/10`.
- Relative to native baseline: `-15.0` percentage points.

All 40 failures have format/protocol as the primary cause. The active SFT
train and validation sets contain zero strict JSON terminal answers, while the
held-out harness requires strict `answer/mentioned_models` JSON. This contract
mismatch is consistent with the observed schema regression. No model
improvement claim is supported by this run.
