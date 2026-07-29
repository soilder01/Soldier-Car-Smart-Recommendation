# GRPO Experiment Redesign Proposal

Status: design proposal only. This document does not authorize training, does not
generate data, does not freeze a new manifest, does not change `reward_fn.py`,
does not change scorer/reporting metrics, and does not access held-out/final
evaluation outputs.

## 0. Summary

The three formal runs were stopped by experiment-design and health-monitoring
defects before GRPO had a fair optimization horizon:

- `formal_v1`: absolute intent-response abort threshold was calibrated below the
  measured pre-training baseline, causing self-abort at the first window.
- `formal_v2`: `fake_reward_rise_without_factual_gain` used cross-window
  `min()` semantics, which made one bad factual window sufficient to satisfy the
  factual side of a two-window condition.
- `formal_v3`: fixed probe and per-window AND corrected the two issues above,
  but `kl_p95_two_windows` was still configured as an abort trigger. That
  stopped the run on normal heavy-tail RL behavior, while the KL penalty already
  existed in the loss through nonzero `beta`.

The conclusion is not that GRPO failed. The current experiment is too small,
too short, and over-aborted by health checks that confuse normal RL dynamics
with catastrophic failure.

## 1. Root Causes

### 1.1 Data Scale Is the Primary Bottleneck

Current reward training data has only 16 train prompts plus 4 dev prompts. This
is too small for GRPO:

- With four intents, each intent has only a handful of training prompts.
- Group-relative advantage is estimated from 8 completions for a single prompt,
  but the prompt distribution itself is nearly unobservable at this scale.
- Health windows over 5 optimizer steps mix prompt composition noise with model
  behavior.
- Sparse gates such as intent-response and grounding can make one or two prompts
  dominate a window.

Target redesign:

- Build a reward-visible pool at the scale of hundreds of prompts, not tens.
- Minimum target: `>=512` train prompts plus `64-128` reward-dev prompts.
- Preferred target if synthesis quality passes audit: `800-1200` train prompts
  plus `100-160` reward-dev prompts.
- Keep a small fixed sentinel probe, including the existing dev-4
  `reward-recommend-002`, `reward-compare-004`, `reward-knowledge-003`,
  `reward-sales-004`, for continuity only. It must not be the sole statistical
  basis for training-health decisions in a longer run.

### 1.2 Step Budget and Schedule Were Smoke-Level

`48` optimizer steps is a smoke horizon. It can detect wiring failures, OOM,
NaN, obvious reward collapse, or severe protocol regression. It cannot evaluate
whether GRPO improves grounding over a prompt distribution.

Redesign target:

- For `512` train prompts, run at least `512-1024` optimizer steps.
- For `800-1200` train prompts, run `1000-2000` optimizer steps.
- Use checkpoints/evaluations every `50-100` steps, not every `5` steps.
- Treat the first `50-100` steps as burn-in for noisy RL dynamics; hard aborts
  remain active for NaN/Inf/OOM/protocol collapse, but not for ordinary KL tail
  movement.

Learning-rate schedule candidates:

- Candidate A: `lr=3e-6`, warmup `3%`, cosine decay to `3e-7`.
- Candidate B: `lr=2e-6`, warmup `5%`, linear decay to `2e-7`.
- Candidate C: `lr=1e-6`, warmup `5%`, cosine decay; conservative baseline if
  dry-run memory or KL behavior is unstable.

Rationale:

- `1e-6 constant` is appropriate for smoke safety but likely too weak for a
  longer GRPO run with small LoRA updates and sparse rewards.
- Warmup prevents early policy jumps before the reward distribution stabilizes.
- Decay reduces late-stage KL pressure without setting `beta=0`.

### 1.3 KL p95 Was Misclassified as an Abort Signal

KL monitoring is necessary, but `kl_p95_two_windows` is too sensitive as an
abort trigger. Token-level KL p95 is naturally heavy-tailed in autoregressive
RL, especially when a few completions shift strongly while most tokens remain
near reference.

Corrected principle:

- KL control belongs primarily in the loss through nonzero `beta`.
- `beta=0` remains forbidden.
- KL p95 remains a monitoring signal and reportable diagnostic.
- Abort only on explosive KL divergence, not ordinary heavy-tail movement.

### 1.4 Evaluation Cadence Was Too Frequent for the Intended Horizon

A fixed dev probe is useful, but evaluating every 5 steps and treating short-run
KL p95 movement as fatal is not compatible with a multi-hundred-step RL run.

Redesign:

- Keep fixed dev-4 as a sentinel probe for continuity and regression detection.
- Add a larger reward-dev set, generated and isolated with the same process as
  train data, for statistically meaningful health evaluation.
- Use health evaluations every `50` steps for `<=1024` step runs.
- Use health evaluations every `100` steps for `>1024` step runs.
- Keep per-step raw logs for loss/reward/KL/memory, but window-level abort
  decisions should use fixed evaluation sets, not rolling train prompt slices.

## 2. Data Expansion Plan

### 2.1 Sources

Generate candidate prompts programmatically from:

- `vehicle_database.csv` structured fields.
- The frozen `allowed_models` set.
- Intent templates for `recommend`, `compare`, `knowledge`, and `sales`.
- Attribute templates over price, energy type, vehicle type, range, battery,
  charging, space, assisted driving, safety, comfort, sales/service scenarios,
  and model-specific constraints.

No LLM judge is introduced. Templates must emit both prompt text and deterministic
reward specs/evidence references.

### 2.2 Intent Balance

Initial target distribution for `800` train prompts:

| intent | train prompts | reward-dev prompts |
|---|---:|---:|
| recommend | 200 | 25-40 |
| compare | 200 | 25-40 |
| knowledge | 200 | 25-40 |
| sales | 200 | 25-40 |

For a `512` prompt minimum run:

| intent | train prompts | reward-dev prompts |
|---|---:|---:|
| recommend | 128 | 16-32 |
| compare | 128 | 16-32 |
| knowledge | 128 | 16-32 |
| sales | 128 | 16-32 |

### 2.3 Deduplication

Every generated prompt must carry:

- `prompt_id`
- raw query
- normalized query
- intent
- target model IDs
- evidence source IDs
- deterministic reward spec SHA
- source data snapshot SHA

Dedup gates:

- Exact normalized query SHA-256 dedup.
- `(intent, sorted target models, normalized attribute anchors)` dedup.
- Near-template dedup by canonical template ID plus filled slots.
- Reject prompts whose normalized query SHA overlaps SFT train, SFT held-out,
  GRPO reward-visible legacy 20, held-out-40, or final-40.

### 2.4 Isolation

Before any training:

- Compute SHA-256 over normalized `prompt_id`.
- Compute SHA-256 over normalized query text.
- Compute SHA-256 over reward spec/evidence references.
- Fail closed on any overlap with:
  - SFT training data
  - SFT held-out-40
  - legacy GRPO reward-visible 20
  - GRPO final-40
  - any previous formal run diagnostics used for selection

The final-40 and held-out-40 remain physically isolated. They are not used for
generation, reward design, threshold tuning, or checkpoint selection.

## 3. Training Schedule Redesign

### 3.1 Minimal Formal Run

Use only after data-generation audit passes:

- Train prompts: `512`
- Reward-dev prompts: `64-128`
- `num_generations=8`
- Optimizer steps: `512-1024`
- Warmup: `3-5%`
- LR schedule: cosine or linear decay
- Initial LR candidates: `2e-6` or `3e-6`
- `beta`: nonzero, initial candidate `0.01`; beta sweep may be proposed only
  before a run and must be preregistered.
- Health evaluation interval: every `50` optimizer steps.

### 3.2 Preferred Formal Run

Use after prompt synthesis quality is manually audited:

- Train prompts: `800-1200`
- Reward-dev prompts: `100-160`
- Optimizer steps: `1000-2000`
- Warmup: `5%`
- LR schedule: cosine decay to `10-20%` of peak LR
- Checkpoint save/eval interval: every `100` steps
- Fixed sentinel dev-4: logged every evaluation interval, not used alone for
  statistical conclusions.

### 3.3 Required Dry Runs

Before a long run:

1. Static isolation check over generated data.
2. CPU/static reward-spec validation.
3. GPU dry-run with a small subset to verify memory and dtype.
4. Signal probe on a representative sample of the expanded train set:
   - group reward variance by intent
   - full-zero and full-one rates
   - reward component variance
   - rollout peak memory

## 4. KL Handling Redesign

### 4.1 Abort List After Redesign

Abort remains fail-closed for true safety failures:

| class | abort condition |
|---|---|
| NaN/Inf | any non-finite loss, reward, gradient, or KL |
| OOM | any CUDA OOM |
| protocol collapse | protocol pass rate `<90%` on fixed health eval, or `<95%` for two fixed evals |
| fake reward rise | per-window AND rule: two fixed evals both have reward gain `>=0.08` and factual gain `<0.01`; thresholds unchanged |
| KL explosive divergence | KL mean `>=10x KL0_mean` and absolute KL mean `>=5.0` for two fixed evals, or KL max `>=100` with simultaneous length explosion/refusal/copying |
| data/evidence drift | any prompt/spec/evidence/model/reward SHA drift |
| OOM/NaN quarantine | stop, write diagnostics, `resume_allowed=false` |

The fake-reward gate remains a scientific anti-hacking gate. It should not be
weakened. The Boolean semantics must remain per-window AND.

### 4.2 Downgraded to Monitoring

These remain logged and reported but no longer abort by themselves:

| metric | status |
|---|---|
| KL p95 two-window elevation | monitor only |
| KL max spike without non-finite/length/protocol symptoms | monitor only |
| mild KL mean increase below explosive threshold | monitor only |
| short-run reward oscillation during burn-in | monitor only |
| per-intent dev-4 noise over one or two small windows | monitor only |

### 4.3 Why This Is Not Loosening Scientific Discipline

This change does not remove KL control. It moves control back to the mechanism
designed for it: the KL penalty in the GRPO loss with nonzero `beta`.

The previous `kl_p95_two_windows` abort conflated:

- normal heavy-tailed token-level policy movement, and
- catastrophic divergence requiring shutdown.

A p95 tail moving above `KL0_p95 + 0.10` for two windows is not by itself
evidence of hacking, protocol collapse, NaN, OOM, or loss of factuality. It is
an expected RL diagnostic. Catastrophic KL still aborts under the explosive
divergence rule.

## 5. Evaluation Cadence

### 5.1 During Training

Per-step logs remain:

- reward mean/std
- component means
- loss
- grad norm
- KL mean/p50/p95/max
- length
- memory
- non-finite counters

Fixed health evaluations:

- Every `50` steps for `<=1024` total steps.
- Every `100` steps for `>1024` total steps.
- Use a larger reward-dev set for health decisions.
- Keep dev-4 sentinel as a continuity canary.

### 5.2 After Training

- Select checkpoint only from preregistered reward-dev criteria.
- Do not access held-out-40 or final-40 during training.
- final-40 remains one-shot after lock.
- Any failed samples go to `*_failures.jsonl`; do not relabel failures as
  successes.

## 6. Before/After Table

| area | current formal runs | redesigned proposal |
|---|---|---|
| reward train size | 16 prompts | `512-1200` prompts |
| reward-dev | 4 prompts | `64-160` plus legacy dev-4 sentinel |
| health windows | every 5 train steps | fixed eval every 50-100 steps |
| train prompt windows | used for abort in v1/v2 | diagnostic only |
| fake-reward gate | v2 had cross-window min bug; v3 fixed | keep per-window AND, thresholds unchanged |
| knowledge blind spot | exposed in small probe | retain explicit reporting; do not hide failures |
| KL p95 | abort after two elevated windows | monitoring only |
| KL control | beta penalty plus over-eager p95 abort | beta penalty plus explosive-divergence abort |
| step budget | 48 steps | `512-2000` steps |
| LR | `1e-6 constant` | preregistered warmup + decay candidates |
| final eval | isolated | unchanged isolated one-shot |

## 7. Methodology Corrections vs Red Lines

### Methodology Corrections

- Expand reward-visible training data to a statistically meaningful scale.
- Use fixed evaluation sets for health decisions rather than rolling train
  prompt slices.
- Increase optimizer steps to match data scale.
- Add warmup and decay schedule candidates.
- Downgrade KL p95 from abort to monitoring.
- Define catastrophic KL divergence separately.

### Red Lines That Do Not Move

- `reward_fn.py` bytes remain unchanged.
- Programmatic reward and anti-cheating logic remain unchanged.
- No LLM judge is introduced.
- `beta=0` remains forbidden.
- held-out-40 and final-40 isolation remains unchanged.
- final-40 remains one-shot only after model/checkpoint lock.
- No threshold may be tuned using held-out/final results.
- No failure may be reported as success.

## 8. Review Gate

This proposal is not an authorization. Before any implementation:

1. Approve or reject the data expansion plan.
2. Approve target dataset size and intent distribution.
3. Approve LR/schedule candidate.
4. Approve revised abort list.
5. Run static data isolation only after approval.
6. Freeze manifests and SHAs before any training.

Until those approvals exist, no data generation, no new authorization, and no
training should occur.
