"""Authorized, health-armed formal GRPO training implementation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from training.grpo.reward_fn import score_grounded_answer
from training.grpo.run_signal_probe import build_prompt, spec_and_claims


ROOT = Path(__file__).resolve().parents[2]
GRPO_DIR = ROOT / "data" / "model_training" / "grpo"
RUN_ID = "grpo-formal-v4"
AUTH_PATH = GRPO_DIR / "grpo_formal_v4_authorization.json"
PREREG_PATH = GRPO_DIR / "grpo_formal_v4_preregistered_report.json"
INPUT_PATH = GRPO_DIR / "grpo_expanded_v4_input_manifest.json"
SIGNAL_RAW_PATH = GRPO_DIR / "grpo_signal_probe_raw.jsonl"
FIXED_DEV4_BASELINE_PATH = (
    GRPO_DIR / "grpo_formal_v4_fixed_dev4_step0_baseline.json"
)
MODEL_PATH = ROOT / "models" / "Qwen2.5-7B-Instruct"
ADAPTER_PATH = ROOT / "checkpoints" / "sft" / "best_adapter"
RUN_ATTEMPT = os.environ.get("GRPO_FORMAL_RUN_ATTEMPT", "")
if RUN_ATTEMPT and (
    not RUN_ATTEMPT.startswith("restart_")
    or not RUN_ATTEMPT.removeprefix("restart_").isdigit()
):
    raise ValueError("invalid GRPO_FORMAL_RUN_ATTEMPT")
OUTPUT_DIR = ROOT / "checkpoints" / "grpo" / "formal_v4"
RUN_DIR = GRPO_DIR / "formal_v4"
if RUN_ATTEMPT:
    OUTPUT_DIR = OUTPUT_DIR / RUN_ATTEMPT
    RUN_DIR = RUN_DIR / RUN_ATTEMPT
STEP_LOG_PATH = RUN_DIR / "step_metrics.jsonl"
WINDOW_LOG_PATH = RUN_DIR / "health_windows.jsonl"
BASELINE_PATH = RUN_DIR / "step0_baseline.json"
FIXED_PROBE_WINDOW_LOG_PATH = RUN_DIR / "fixed_dev4_probe_windows.jsonl"
ABORT_PATH = RUN_DIR / "abort_report.json"
RESUME_PATH = RUN_DIR / "resume_allowed.json"
QUARANTINE_PATH = RUN_DIR / "checkpoint_quarantine.json"
DEV_ABORT_PATH = RUN_DIR / "dev_abort_diagnostics.json"
RUN_REPORT_PATH = RUN_DIR / "run_report.json"
FIXED_DEV4_PROBE_IDS = (
    "reward-recommend-002",
    "reward-compare-004",
    "reward-knowledge-003",
    "reward-sales-004",
)
FAKE_REWARD_GATE_EXCLUDED_PROMPT_IDS = ("reward-knowledge-003",)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def normalized_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if not character.isspace())


def ngram_distinct(texts: list[str], n: int) -> float:
    grams: list[str] = []
    for text in texts:
        normalized = normalized_text(text)
        grams.extend(
            normalized[index : index + n]
            for index in range(max(0, len(normalized) - n + 1))
        )
    return len(set(grams)) / len(grams) if grams else 0.0


def copy_ratio(text: str, prompt_evidence: str, n: int = 5) -> float:
    normalized = normalized_text(text)
    evidence = normalized_text(prompt_evidence)
    grams = [
        normalized[index : index + n]
        for index in range(max(0, len(normalized) - n + 1))
    ]
    return (
        sum(gram in evidence for gram in grams) / len(grams)
        if grams
        else 0.0
    )


def diagnostic_penalties(
    text: str,
    case: dict[str, Any],
    completion_tokens: int,
) -> dict[str, float]:
    normalized = normalized_text(text)
    claim_values = [
        normalized_text(item["canonical_value"])
        for item in case["evidence_claims"]
    ]
    mentions = [normalized.count(value) for value in claim_values if value]
    repeated = sum(max(0, count - 1) for count in mentions)
    total_mentions = sum(mentions)
    duplicate = repeated / total_mentions if total_mentions else 0.0
    evidence_text = " ".join(
        (
            f"{item['canonical_entity']} "
            f"{item['canonical_attribute']} "
            f"{item['canonical_value']}"
        )
        for item in case["evidence_claims"]
    )
    nonfact_copy = copy_ratio(text, evidence_text)
    verbosity = max(0.0, min(1.0, (completion_tokens - 384) / 128))
    return {
        "duplicate_claim_ratio": duplicate,
        "nonfact_copy_ratio": nonfact_copy,
        "verbosity_over_budget_ratio": verbosity,
    }


def load_signal_baseline() -> dict[str, Any]:
    groups = [
        json.loads(line)
        for line in SIGNAL_RAW_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train = [group for group in groups if group["split"] == "train"]
    completions = [
        completion for group in train for completion in group["completions"]
    ]
    lengths = [completion["completion_tokens"] for completion in completions]
    per_intent: dict[str, list[dict[str, Any]]] = {}
    for group in train:
        per_intent.setdefault(group["intent"], []).extend(
            group["completions"]
        )
    baseline = {
        "source": str(SIGNAL_RAW_PATH.relative_to(ROOT)),
        "source_sha256": sha256_file(SIGNAL_RAW_PATH),
        "groups": len(train),
        "reward_mean": statistics.fmean(
            completion["total"] for completion in completions
        ),
        "factual_precision_mean": statistics.fmean(
            completion["factual_precision"] for completion in completions
        ),
        "required_coverage_mean": statistics.fmean(
            completion["required_coverage"] for completion in completions
        ),
        "source_integrity_mean": statistics.fmean(
            completion["source_integrity"] for completion in completions
        ),
        "concision_mean": statistics.fmean(
            completion["concision"] for completion in completions
        ),
        "intent_response_fail_rate": statistics.fmean(
            completion["gate"] != "passed" for completion in completions
        ),
        "zero_variance_group_ratio": statistics.fmean(
            group["reward_distribution"]["sample_std"] < 1e-4
            for group in train
        ),
        "completion_length_p50": quantile(lengths, 0.50),
        "per_intent_fail_rate": {
            intent: statistics.fmean(
                completion["gate"] != "passed"
                for completion in intent_completions
            )
            for intent, intent_completions in sorted(per_intent.items())
        },
        "kl0_mean": None,
        "kl0_p95": None,
    }
    return baseline


def fixed_dev4_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {case["id"]: case for case in cases}
    missing = [
        identifier
        for identifier in FIXED_DEV4_PROBE_IDS
        if identifier not in by_id
    ]
    if missing:
        raise ValueError(f"fixed dev-4 probe case drift: {missing}")
    return [by_id[identifier] for identifier in FIXED_DEV4_PROBE_IDS]


def fake_reward_gate_case_ids(case_ids: list[str]) -> list[str]:
    return [
        identifier
        for identifier in case_ids
        if identifier not in FAKE_REWARD_GATE_EXCLUDED_PROMPT_IDS
    ]


def fake_reward_gate_means(
    case_results: list[dict[str, Any]],
) -> dict[str, Any]:
    included = [
        result
        for result in case_results
        if result["id"] not in FAKE_REWARD_GATE_EXCLUDED_PROMPT_IDS
    ]
    excluded = [
        result["id"]
        for result in case_results
        if result["id"] in FAKE_REWARD_GATE_EXCLUDED_PROMPT_IDS
    ]
    if not included:
        raise ValueError("fake reward gate population is empty")
    return {
        "scientific_limitation": (
            "fake_reward_rise_without_factual_gain excludes knowledge-003; "
            "this gate's factual criterion does not cover knowledge intent."
        ),
        "included_prompt_ids": [result["id"] for result in included],
        "excluded_prompt_ids": excluded,
        "reward_mean": statistics.fmean(
            result["reward_mean"] for result in included
        ),
        "factual_precision_mean": statistics.fmean(
            result["factual_precision_mean"] for result in included
        ),
    }


def model_use_cache_configs(
    model: Any,
    generation_config: Any | None = None,
) -> list[Any]:
    configs: list[Any] = []
    seen: set[int] = set()
    candidates = [
        model,
        getattr(model, "base_model", None),
        getattr(getattr(model, "base_model", None), "model", None),
        getattr(model, "pretrained_model", None),
        generation_config,
    ]
    for candidate in candidates:
        config = (
            candidate
            if hasattr(candidate, "use_cache")
            else getattr(candidate, "config", None)
        )
        if config is None or not hasattr(config, "use_cache"):
            continue
        identifier = id(config)
        if identifier in seen:
            continue
        seen.add(identifier)
        configs.append(config)
    return configs


def gradient_checkpointing_is_enabled(model: Any) -> bool:
    for candidate in (
        model,
        getattr(model, "base_model", None),
        getattr(getattr(model, "base_model", None), "model", None),
        getattr(model, "pretrained_model", None),
    ):
        if candidate is None:
            continue
        value = getattr(candidate, "is_gradient_checkpointing", None)
        if isinstance(value, bool):
            return value
    return False


def call_gradient_checkpointing_enable(model: Any) -> None:
    method = getattr(model, "gradient_checkpointing_enable", None)
    if method is None:
        raise RuntimeError("model lacks gradient_checkpointing_enable")
    try:
        method(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        method()


def call_gradient_checkpointing_disable(model: Any) -> None:
    method = getattr(model, "gradient_checkpointing_disable", None)
    if method is None:
        raise RuntimeError("model lacks gradient_checkpointing_disable")
    method()


@contextmanager
def rollout_generation_cache_context(
    model: Any,
    generation_config: Any | None = None,
) -> Iterator[dict[str, Any]]:
    configs = model_use_cache_configs(model, generation_config)
    if not configs:
        raise RuntimeError("no use_cache config found for rollout generation")
    previous_use_cache = [(config, config.use_cache) for config in configs]
    previous_gradient_checkpointing = gradient_checkpointing_is_enabled(model)
    call_gradient_checkpointing_disable(model)
    for config in configs:
        config.use_cache = True
    state = {
        "rollout_use_cache": all(config.use_cache is True for config in configs),
        "training_use_cache_before": [value for _, value in previous_use_cache],
        "gradient_checkpointing_before": previous_gradient_checkpointing,
    }
    if not state["rollout_use_cache"]:
        raise RuntimeError("failed to enable rollout KV cache")
    try:
        yield state
    finally:
        for config, _ in previous_use_cache:
            config.use_cache = False
        if previous_gradient_checkpointing:
            call_gradient_checkpointing_enable(model)
        restored = [config.use_cache for config, _ in previous_use_cache]
        expected = [False for _ in previous_use_cache]
        if restored != expected:
            raise RuntimeError("rollout use_cache state was not restored")
        if previous_gradient_checkpointing and not gradient_checkpointing_is_enabled(
            model
        ):
            raise RuntimeError(
                "gradient checkpointing was not restored after rollout"
            )


def load_fixed_dev4_baseline(
    authorization: dict[str, Any],
) -> dict[str, Any]:
    baseline_spec = authorization["fixed_dev4_probe_step0_baseline"]
    path = ROOT / baseline_spec["path"]
    if path != FIXED_DEV4_BASELINE_PATH:
        raise ValueError("fixed dev-4 baseline path drift")
    if sha256_file(path) != baseline_spec["sha256"]:
        raise ValueError("fixed dev-4 baseline SHA drift")
    baseline = json.loads(path.read_text(encoding="utf-8"))
    if baseline["prompt_ids"] != list(FIXED_DEV4_PROBE_IDS):
        raise ValueError("fixed dev-4 baseline prompt order drift")
    if baseline["fake_reward_gate"]["excluded_prompt_ids"] != list(
        FAKE_REWARD_GATE_EXCLUDED_PROMPT_IDS
    ):
        raise ValueError("fake reward gate exclusion drift")
    return baseline


class TrainingAudit:
    def __init__(
        self,
        *,
        cases: dict[str, dict[str, Any]],
        tokenizer: Any,
        authorization: dict[str, Any],
        baseline: dict[str, Any],
    ) -> None:
        self.cases = cases
        self.tokenizer = tokenizer
        self.authorization = authorization
        self.baseline = baseline
        self.pending_reward: dict[str, Any] | None = None
        self.pending_compute: dict[str, Any] | None = None
        self.steps: list[dict[str, Any]] = []
        self.windows: list[dict[str, Any]] = []
        self.abort_triggered = False
        self.abort_reasons: list[str] = []

    def reward_function(
        self,
        prompts: list[Any],
        completions: list[Any],
        case_id: list[str],
        **_: Any,
    ) -> list[float]:
        identifiers = list(dict.fromkeys(case_id))
        if len(identifiers) != 1 or len(completions) != 8:
            raise RuntimeError("reward callback expected one 8-generation group")
        identifier = identifiers[0]
        case = self.cases[identifier]
        spec, claims = spec_and_claims(case)
        texts = [
            completion[-1]["content"]
            if isinstance(completion, list)
            else str(completion)
            for completion in completions
        ]
        results: list[dict[str, Any]] = []
        for text in texts:
            score = score_grounded_answer(
                answer=text,
                spec=spec,
                evidence_claims=claims,
            )
            token_length = len(
                self.tokenizer(text, add_special_tokens=False)["input_ids"]
            )
            penalties = diagnostic_penalties(text, case, token_length)
            result = {
                "text_sha256": hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest(),
                "completion_tokens": token_length,
                "empty": not bool(text.strip()),
                "refusal": any(
                    phrase in text
                    for phrase in ("无法回答", "不能回答", "请以官方为准")
                ),
                "protocol_pass": bool(text.strip())
                and not text.lstrip().startswith(("{", "[")),
                "intent_response_pass": score.gate == "passed",
                "critical_grounding_pass": (
                    score.gate != "unsupported_target_value"
                ),
                "non_answer_pass": bool(text.strip()),
                "gate": score.gate,
                "total": float(score.total),
                "factual_precision": float(score.factual_precision),
                "required_coverage": float(score.required_coverage),
                "source_integrity": float(score.source_integrity),
                "concision": float(score.concision),
                "penalties": penalties,
            }
            if not all(
                math.isfinite(value)
                for key, value in result.items()
                if key
                in {
                    "total",
                    "factual_precision",
                    "required_coverage",
                    "source_integrity",
                    "concision",
                }
            ):
                raise RuntimeError("non-finite reward component")
            results.append(result)
        rewards = [item["total"] for item in results]
        unique_ratio = len(set(texts)) / len(texts)
        self.pending_reward = {
            "case_id": identifier,
            "intent": case["intent"],
            "completions": results,
            "reward_mean": statistics.fmean(rewards),
            "reward_std": statistics.stdev(rewards),
            "zero_variance": statistics.stdev(rewards) < 1e-4,
            "unique_ratio": unique_ratio,
            "distinct_2": ngram_distinct(texts, 2),
            "distinct_4": ngram_distinct(texts, 4),
            "completely_duplicate": unique_ratio == 1 / len(texts),
        }
        return rewards

    def record_compute(self, value: dict[str, Any]) -> None:
        self.pending_compute = value

    def consume_step(
        self,
        *,
        step: int,
        logs: dict[str, Any],
    ) -> dict[str, Any]:
        if self.pending_reward is None or self.pending_compute is None:
            raise RuntimeError("missing reward or KL audit for logged step")
        reward = self.pending_reward
        compute = self.pending_compute
        completions = reward["completions"]
        step_record = {
            "step": step,
            "case_id": reward["case_id"],
            "intent": reward["intent"],
            "reward_mean": reward["reward_mean"],
            "reward_std": reward["reward_std"],
            "zero_variance": reward["zero_variance"],
            "factual_precision_mean": statistics.fmean(
                item["factual_precision"] for item in completions
            ),
            "required_coverage_mean": statistics.fmean(
                item["required_coverage"] for item in completions
            ),
            "source_integrity_mean": statistics.fmean(
                item["source_integrity"] for item in completions
            ),
            "concision_mean": statistics.fmean(
                item["concision"] for item in completions
            ),
            "duplicate_claim_ratio_mean": statistics.fmean(
                item["penalties"]["duplicate_claim_ratio"]
                for item in completions
            ),
            "nonfact_copy_ratio_mean": statistics.fmean(
                item["penalties"]["nonfact_copy_ratio"]
                for item in completions
            ),
            "verbosity_over_budget_ratio_mean": statistics.fmean(
                item["penalties"]["verbosity_over_budget_ratio"]
                for item in completions
            ),
            "protocol_pass_rate": statistics.fmean(
                item["protocol_pass"] for item in completions
            ),
            "intent_response_fail_rate": statistics.fmean(
                not item["intent_response_pass"] for item in completions
            ),
            "critical_grounding_pass_rate": statistics.fmean(
                item["critical_grounding_pass"] for item in completions
            ),
            "non_answer_pass_rate": statistics.fmean(
                item["non_answer_pass"] for item in completions
            ),
            "empty_rate": statistics.fmean(
                item["empty"] for item in completions
            ),
            "refusal_rate": statistics.fmean(
                item["refusal"] for item in completions
            ),
            "completion_lengths": [
                item["completion_tokens"] for item in completions
            ],
            "unique_ratio": reward["unique_ratio"],
            "distinct_2": reward["distinct_2"],
            "distinct_4": reward["distinct_4"],
            "completely_duplicate": reward["completely_duplicate"],
            **compute,
            "logged_loss": float(logs.get("loss", 0.0)),
            "grad_norm": float(logs.get("grad_norm", 0.0)),
            "learning_rate": float(logs.get("learning_rate", 0.0)),
            "cuda_allocated_mib": compute["cuda_allocated_mib"],
            "cuda_reserved_mib": compute["cuda_reserved_mib"],
        }
        numeric_values = [
            value
            for value in step_record.values()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        step_record["all_finite"] = all(
            math.isfinite(float(value)) for value in numeric_values
        )
        append_jsonl(STEP_LOG_PATH, step_record)
        self.steps.append(step_record)
        self.pending_reward = None
        self.pending_compute = None
        return step_record


def window_summary(steps: list[dict[str, Any]]) -> dict[str, Any]:
    completion_lengths = [
        length for step in steps for length in step["completion_lengths"]
    ]
    intents = sorted({step["intent"] for step in steps})
    return {
        "start_step": steps[0]["step"],
        "end_step": steps[-1]["step"],
        "steps": len(steps),
        "reward_mean": statistics.fmean(
            step["reward_mean"] for step in steps
        ),
        "factual_precision_mean": statistics.fmean(
            step["factual_precision_mean"] for step in steps
        ),
        "required_coverage_mean": statistics.fmean(
            step["required_coverage_mean"] for step in steps
        ),
        "source_integrity_mean": statistics.fmean(
            step["source_integrity_mean"] for step in steps
        ),
        "concision_mean": statistics.fmean(
            step["concision_mean"] for step in steps
        ),
        "duplicate_claim_ratio_mean": statistics.fmean(
            step["duplicate_claim_ratio_mean"] for step in steps
        ),
        "nonfact_copy_ratio_mean": statistics.fmean(
            step["nonfact_copy_ratio_mean"] for step in steps
        ),
        "verbosity_over_budget_ratio_mean": statistics.fmean(
            step["verbosity_over_budget_ratio_mean"] for step in steps
        ),
        "protocol_pass_rate": statistics.fmean(
            step["protocol_pass_rate"] for step in steps
        ),
        "intent_response_fail_rate": statistics.fmean(
            step["intent_response_fail_rate"] for step in steps
        ),
        "intent_response_fail_rate_by_intent": {
            intent: statistics.fmean(
                step["intent_response_fail_rate"]
                for step in steps
                if step["intent"] == intent
            )
            for intent in intents
        },
        "critical_grounding_pass_rate": statistics.fmean(
            step["critical_grounding_pass_rate"] for step in steps
        ),
        "non_answer_pass_rate": statistics.fmean(
            step["non_answer_pass_rate"] for step in steps
        ),
        "empty_rate": statistics.fmean(step["empty_rate"] for step in steps),
        "refusal_rate": statistics.fmean(
            step["refusal_rate"] for step in steps
        ),
        "length_min": min(completion_lengths),
        "length_p10": quantile(completion_lengths, 0.10),
        "length_p50": quantile(completion_lengths, 0.50),
        "length_p90": quantile(completion_lengths, 0.90),
        "length_p95": quantile(completion_lengths, 0.95),
        "length_max": max(completion_lengths),
        "max_length_hit_rate": statistics.fmean(
            length >= 512 for length in completion_lengths
        ),
        "unique_ratio_mean": statistics.fmean(
            step["unique_ratio"] for step in steps
        ),
        "distinct_2": statistics.fmean(
            step["distinct_2"] for step in steps
        ),
        "distinct_4": statistics.fmean(
            step["distinct_4"] for step in steps
        ),
        "low_unique_group_share": statistics.fmean(
            step["unique_ratio"] < 0.25 for step in steps
        ),
        "completely_duplicate_group_share": statistics.fmean(
            step["completely_duplicate"] for step in steps
        ),
        "zero_variance_group_share": statistics.fmean(
            step["zero_variance"] for step in steps
        ),
        "advantage_mean": statistics.fmean(
            step["advantage_mean"] for step in steps
        ),
        "advantage_std_mean": statistics.fmean(
            step["advantage_std"] for step in steps
        ),
        "kl_signed_mean": statistics.fmean(
            step["kl_signed_mean"] for step in steps
        ),
        "kl_mean": statistics.fmean(step["kl_mean"] for step in steps),
        "kl_p50": statistics.fmean(step["kl_p50"] for step in steps),
        "kl_p95": max(step["kl_p95"] for step in steps),
        "kl_max": max(step["kl_max"] for step in steps),
        "loss_mean": statistics.fmean(
            step["logged_loss"] for step in steps
        ),
        "grad_norm_mean": statistics.fmean(
            step["grad_norm"] for step in steps
        ),
        "learning_rate": steps[-1]["learning_rate"],
        "step_time_mean_sec": statistics.fmean(
            step["step_time_sec"] for step in steps
        ),
        "cuda_peak_reserved_mib": max(
            step["cuda_reserved_mib"] for step in steps
        ),
        "non_finite_count": sum(not step["all_finite"] for step in steps),
    }


def abort_reasons(
    window: dict[str, Any],
    previous: dict[str, Any] | None,
    baseline: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if window["non_finite_count"]:
        reasons.append("non_finite")
    if window["protocol_pass_rate"] < 0.90:
        reasons.append("protocol_pass_rate_below_90_percent")
    if (
        previous is not None
        and previous["protocol_pass_rate"] < 0.95
        and window["protocol_pass_rate"] < 0.95
    ):
        reasons.append("protocol_below_95_percent_two_windows")
    if previous is not None:
        if baseline["kl0_mean"] is not None and (
            previous["kl_mean"] >= max(5.0, 10.0 * baseline["kl0_mean"])
            and window["kl_mean"] >= max(5.0, 10.0 * baseline["kl0_mean"])
        ):
            reasons.append("kl_explosive_divergence_two_windows")
        if (
            previous.get("kl_max", 0.0) >= 100.0
            and window.get("kl_max", 0.0) >= 100.0
            and (
                previous["length_p95"] > 486
                or window["length_p95"] > 486
                or previous["nonfact_copy_ratio_mean"] > 0.60
                or window["nonfact_copy_ratio_mean"] > 0.60
                or previous["refusal_rate"] > 0.10
                or window["refusal_rate"] > 0.10
            )
        ):
            reasons.append("kl_max_explosive_with_behavioral_symptom")
        previous_reward_gain = (
            previous["fake_reward_gate"]["reward_mean"]
            - baseline["fake_reward_gate"]["reward_mean"]
        )
        window_reward_gain = (
            window["fake_reward_gate"]["reward_mean"]
            - baseline["fake_reward_gate"]["reward_mean"]
        )
        previous_factual_gain = (
            previous["fake_reward_gate"]["factual_precision_mean"]
            - baseline["fake_reward_gate"]["factual_precision_mean"]
        )
        window_factual_gain = (
            window["fake_reward_gate"]["factual_precision_mean"]
            - baseline["fake_reward_gate"]["factual_precision_mean"]
        )
        window["fake_reward_gate"]["reward_gain"] = window_reward_gain
        window["fake_reward_gate"][
            "factual_precision_gain"
        ] = window_factual_gain
        if (
            previous_reward_gain >= 0.08
            and window_reward_gain >= 0.08
            and previous_factual_gain < 0.01
            and window_factual_gain < 0.01
        ):
            reasons.append("fake_reward_rise_without_factual_gain")
    return sorted(set(reasons))


def preflight_hashes(authorization: dict[str, Any]) -> dict[str, str]:
    gate = authorization["authorization_gate"]
    paths = {
        gate["reward_fn"]["path"]: gate["reward_fn"]["sha256"],
        gate["split"]["train_path"]: gate["split"]["train_sha256"],
        gate["split"]["dev_path"]: gate["split"]["dev_sha256"],
        gate["split"]["manifest_path"]: gate["split"]["manifest_sha256"],
        gate["dev_authorization"]["path"]: gate["dev_authorization"][
            "sha256"
        ],
    }
    observed: dict[str, str] = {}
    for relative, expected in paths.items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise ValueError(f"authorization SHA mismatch: {relative}")
        observed[relative] = actual
    return observed


def venv_train_hashes() -> dict[str, str]:
    freeze = subprocess.check_output(
        [str(ROOT / ".venv-train" / "bin" / "python"), "-m", "pip", "freeze"],
        text=True,
    )
    metadata = hashlib.sha256()
    for path in sorted(
        item for item in (ROOT / ".venv-train").rglob("*") if item.is_file()
    ):
        stat = path.stat()
        metadata.update(
            (
                f"{path.relative_to(ROOT)} {stat.st_size} "
                f"{stat.st_mtime_ns}\n"
            ).encode("utf-8")
        )
    return {
        "pip_freeze_sha256": hashlib.sha256(
            freeze.encode("utf-8")
        ).hexdigest(),
        "file_metadata_sha256": metadata.hexdigest(),
    }


def evaluate_dev_after_abort(
    *,
    model: Any,
    tokenizer: Any,
    cases: list[dict[str, Any]],
    torch: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    model.eval()
    results: list[dict[str, Any]] = []
    with torch.inference_mode():
        for case in cases:
            spec, claims = spec_and_claims(case)
            rendered = tokenizer.apply_chat_template(
                build_prompt(case),
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(
                rendered,
                return_tensors="pt",
                add_special_tokens=False,
            )
            inputs = {
                key: value.to(model.device) for key, value in inputs.items()
            }
            with rollout_generation_cache_context(
                model,
                model.generation_config,
            ):
                generated = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=config["temperature"],
                    top_p=config["top_p"],
                    num_return_sequences=8,
                    max_new_tokens=config["max_completion_length"],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=model.generation_config.eos_token_id,
                )
            prompt_length = inputs["input_ids"].shape[1]
            texts = tokenizer.batch_decode(
                generated[:, prompt_length:],
                skip_special_tokens=True,
            )
            scores = [
                score_grounded_answer(
                    answer=text,
                    spec=spec,
                    evidence_claims=claims,
                )
                for text in texts
            ]
            results.append(
                {
                    "id": case["id"],
                    "intent": case["intent"],
                    "reward_mean": statistics.fmean(
                        score.total for score in scores
                    ),
                    "factual_precision_mean": statistics.fmean(
                        score.factual_precision for score in scores
                    ),
                    "required_coverage_mean": statistics.fmean(
                        score.required_coverage for score in scores
                    ),
                    "source_integrity_mean": statistics.fmean(
                        score.source_integrity for score in scores
                    ),
                    "concision_mean": statistics.fmean(
                        score.concision for score in scores
                    ),
                    "gate_counts": dict(
                        Counter(score.gate for score in scores)
                    ),
                }
            )
    model.train()
    return {
        "status": "read_only_after_abort_no_gradient",
        "cases": results,
        "grounding_core_mean": statistics.fmean(
            0.6 * result["factual_precision_mean"]
            + 0.4 * result["required_coverage_mean"]
            for result in results
        ),
    }


def selected_logps_for_completion(
    *,
    model: Any,
    input_ids: Any,
    keep: int,
    torch: Any,
) -> Any:
    logits = model(input_ids, num_logits_to_keep=keep + 1).logits[:, :-1, :]
    rows = []
    for logits_row, ids_row in zip(logits, input_ids[:, -keep:]):
        rows.append(
            torch.gather(
                logits_row.log_softmax(dim=-1),
                dim=1,
                index=ids_row.unsqueeze(1),
            ).squeeze(1)
        )
    return torch.stack(rows)


def summarize_fixed_dev4_probe(
    *,
    case_results: list[dict[str, Any]],
    completion_lengths: list[int],
    kl_values: list[float],
    signed_kl_values: list[float],
    optimizer_step: int,
    train_window: dict[str, Any] | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    intents = sorted({result["intent"] for result in case_results})
    non_finite_count = sum(not result["all_finite"] for result in case_results)
    fake_gate = fake_reward_gate_means(case_results)
    window = {
        "health_source": "fixed_dev4_no_grad_probe",
        "optimizer_start_step": (
            optimizer_step - config["health_window_steps"] + 1
            if optimizer_step
            else 0
        ),
        "optimizer_end_step": optimizer_step,
        "optimizer_steps": (
            config["health_window_steps"] if optimizer_step else 0
        ),
        "probe_case_count": len(case_results),
        "probe_prompt_ids": [result["id"] for result in case_results],
        "prompt_id_set_is_fixed_dev4": (
            [result["id"] for result in case_results]
            == list(FIXED_DEV4_PROBE_IDS)
        ),
        "reward_mean": statistics.fmean(
            result["reward_mean"] for result in case_results
        ),
        "factual_precision_mean": statistics.fmean(
            result["factual_precision_mean"] for result in case_results
        ),
        "required_coverage_mean": statistics.fmean(
            result["required_coverage_mean"] for result in case_results
        ),
        "source_integrity_mean": statistics.fmean(
            result["source_integrity_mean"] for result in case_results
        ),
        "concision_mean": statistics.fmean(
            result["concision_mean"] for result in case_results
        ),
        "reward_mean_by_intent": {
            intent: statistics.fmean(
                result["reward_mean"]
                for result in case_results
                if result["intent"] == intent
            )
            for intent in intents
        },
        "factual_precision_mean_by_intent": {
            intent: statistics.fmean(
                result["factual_precision_mean"]
                for result in case_results
                if result["intent"] == intent
            )
            for intent in intents
        },
        "intent_response_fail_rate_by_intent": {
            intent: statistics.fmean(
                result["intent_response_fail_rate"]
                for result in case_results
                if result["intent"] == intent
            )
            for intent in intents
        },
        "duplicate_claim_ratio_mean": statistics.fmean(
            result["duplicate_claim_ratio_mean"] for result in case_results
        ),
        "nonfact_copy_ratio_mean": statistics.fmean(
            result["nonfact_copy_ratio_mean"] for result in case_results
        ),
        "verbosity_over_budget_ratio_mean": statistics.fmean(
            result["verbosity_over_budget_ratio_mean"]
            for result in case_results
        ),
        "protocol_pass_rate": statistics.fmean(
            result["protocol_pass_rate"] for result in case_results
        ),
        "intent_response_fail_rate": statistics.fmean(
            result["intent_response_fail_rate"] for result in case_results
        ),
        "critical_grounding_pass_rate": statistics.fmean(
            result["critical_grounding_pass_rate"]
            for result in case_results
        ),
        "non_answer_pass_rate": statistics.fmean(
            result["non_answer_pass_rate"] for result in case_results
        ),
        "empty_rate": statistics.fmean(
            result["empty_rate"] for result in case_results
        ),
        "refusal_rate": statistics.fmean(
            result["refusal_rate"] for result in case_results
        ),
        "length_min": min(completion_lengths),
        "length_p10": quantile(completion_lengths, 0.10),
        "length_p50": quantile(completion_lengths, 0.50),
        "length_p90": quantile(completion_lengths, 0.90),
        "length_p95": quantile(completion_lengths, 0.95),
        "length_max": max(completion_lengths),
        "max_length_hit_rate": statistics.fmean(
            length >= config["max_completion_length"]
            for length in completion_lengths
        ),
        "unique_ratio_mean": statistics.fmean(
            result["unique_ratio"] for result in case_results
        ),
        "distinct_2": statistics.fmean(
            result["distinct_2"] for result in case_results
        ),
        "distinct_4": statistics.fmean(
            result["distinct_4"] for result in case_results
        ),
        "low_unique_group_share": statistics.fmean(
            result["unique_ratio"] < 0.25 for result in case_results
        ),
        "completely_duplicate_group_share": statistics.fmean(
            result["completely_duplicate"] for result in case_results
        ),
        "zero_variance_group_share": statistics.fmean(
            result["zero_variance"] for result in case_results
        ),
        "kl_signed_mean": statistics.fmean(signed_kl_values),
        "kl_mean": statistics.fmean(kl_values),
        "kl_p50": quantile(kl_values, 0.50),
        "kl_p95": quantile(kl_values, 0.95),
        "kl_max": max(kl_values),
        "non_finite_count": non_finite_count,
        "case_results": case_results,
        "fake_reward_gate": fake_gate,
    }
    if train_window is not None:
        window["train_window"] = train_window
        window["loss_mean"] = train_window["loss_mean"]
        window["grad_norm_mean"] = train_window["grad_norm_mean"]
        window["learning_rate"] = train_window["learning_rate"]
        window["step_time_mean_sec"] = train_window["step_time_mean_sec"]
        window["cuda_peak_reserved_mib"] = train_window[
            "cuda_peak_reserved_mib"
        ]
        window["train_kl_p95"] = train_window["kl_p95"]
    else:
        window["loss_mean"] = 0.0
        window["grad_norm_mean"] = 0.0
        window["learning_rate"] = 0.0
        window["step_time_mean_sec"] = 0.0
        window["cuda_peak_reserved_mib"] = 0.0
        window["train_kl_p95"] = None
    return window


def evaluate_fixed_dev4_probe_no_grad(
    *,
    model: Any,
    tokenizer: Any,
    cases: list[dict[str, Any]],
    torch: Any,
    config: dict[str, Any],
    optimizer_step: int,
    train_window: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ordered_cases = fixed_dev4_cases(cases)
    was_training = model.training
    model.eval()
    case_results: list[dict[str, Any]] = []
    completion_lengths: list[int] = []
    kl_values: list[float] = []
    signed_kl_values: list[float] = []
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config["seed"])
        with torch.inference_mode():
            for case in ordered_cases:
                spec, claims = spec_and_claims(case)
                rendered = tokenizer.apply_chat_template(
                    build_prompt(case),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = tokenizer(
                    rendered,
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                inputs = {
                    key: value.to(model.device)
                    for key, value in inputs.items()
                }
                with rollout_generation_cache_context(
                    model,
                    model.generation_config,
                ):
                    generated = model.generate(
                        **inputs,
                        do_sample=True,
                        temperature=config["temperature"],
                        top_p=config["top_p"],
                        num_return_sequences=config["num_generations"],
                        max_new_tokens=config["max_completion_length"],
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=model.generation_config.eos_token_id,
                    )
                if generated.shape[1] > config["max_total_sequence_length"]:
                    raise RuntimeError(
                        "fixed dev-4 probe total sequence exceeds limit"
                    )
                prompt_length = inputs["input_ids"].shape[1]
                completion_ids = generated[:, prompt_length:]
                texts = tokenizer.batch_decode(
                    completion_ids,
                    skip_special_tokens=True,
                )
                scores = [
                    score_grounded_answer(
                        answer=text,
                        spec=spec,
                        evidence_claims=claims,
                    )
                    for text in texts
                ]
                keep = completion_ids.size(1)
                policy_logps = selected_logps_for_completion(
                    model=model,
                    input_ids=generated,
                    keep=keep,
                    torch=torch,
                )
                with model.disable_adapter():
                    reference_logps = selected_logps_for_completion(
                        model=model,
                        input_ids=generated,
                        keep=keep,
                        torch=torch,
                    )
                signed_kl = policy_logps - reference_logps
                per_token_kl = (
                    torch.exp(reference_logps - policy_logps)
                    - (reference_logps - policy_logps)
                    - 1
                )
                is_eos = completion_ids == tokenizer.eos_token_id
                eos_index = torch.full(
                    (is_eos.size(0),),
                    is_eos.size(1),
                    dtype=torch.long,
                    device=completion_ids.device,
                )
                has_eos = is_eos.any(dim=1)
                eos_index[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
                positions = torch.arange(
                    is_eos.size(1),
                    device=completion_ids.device,
                ).expand(is_eos.size(0), -1)
                completion_mask = (
                    positions <= eos_index.unsqueeze(1)
                ).bool()
                masked_kl = per_token_kl[completion_mask].detach().float()
                masked_signed = signed_kl[completion_mask].detach().float()
                kl_values.extend(float(value.cpu()) for value in masked_kl)
                signed_kl_values.extend(
                    float(value.cpu()) for value in masked_signed
                )
                completion_infos: list[dict[str, Any]] = []
                for text, score, token_count in zip(
                    texts,
                    scores,
                    completion_mask.sum(1).tolist(),
                ):
                    token_length = int(token_count)
                    completion_lengths.append(token_length)
                    penalties = diagnostic_penalties(
                        text,
                        case,
                        token_length,
                    )
                    completion_infos.append(
                        {
                            "text_sha256": hashlib.sha256(
                                text.encode("utf-8")
                            ).hexdigest(),
                            "completion_tokens": token_length,
                            "empty": not bool(text.strip()),
                            "refusal": any(
                                phrase in text
                                for phrase in (
                                    "无法回答",
                                    "不能回答",
                                    "请以官方为准",
                                )
                            ),
                            "protocol_pass": bool(text.strip())
                            and not text.lstrip().startswith(("{", "[")),
                            "intent_response_pass": score.gate == "passed",
                            "critical_grounding_pass": (
                                score.gate != "unsupported_target_value"
                            ),
                            "non_answer_pass": bool(text.strip()),
                            "gate": score.gate,
                            "total": float(score.total),
                            "factual_precision": float(
                                score.factual_precision
                            ),
                            "required_coverage": float(
                                score.required_coverage
                            ),
                            "source_integrity": float(
                                score.source_integrity
                            ),
                            "concision": float(score.concision),
                            "penalties": penalties,
                        }
                    )
                rewards = [item["total"] for item in completion_infos]
                case_lengths = [
                    item["completion_tokens"] for item in completion_infos
                ]
                unique_ratio = len(set(texts)) / len(texts)
                case_result = {
                    "id": case["id"],
                    "intent": case["intent"],
                    "reward_mean": statistics.fmean(rewards),
                    "reward_std": statistics.stdev(rewards),
                    "zero_variance": statistics.stdev(rewards) < 1e-4,
                    "factual_precision_mean": statistics.fmean(
                        item["factual_precision"]
                        for item in completion_infos
                    ),
                    "required_coverage_mean": statistics.fmean(
                        item["required_coverage"]
                        for item in completion_infos
                    ),
                    "source_integrity_mean": statistics.fmean(
                        item["source_integrity"]
                        for item in completion_infos
                    ),
                    "concision_mean": statistics.fmean(
                        item["concision"] for item in completion_infos
                    ),
                    "duplicate_claim_ratio_mean": statistics.fmean(
                        item["penalties"]["duplicate_claim_ratio"]
                        for item in completion_infos
                    ),
                    "nonfact_copy_ratio_mean": statistics.fmean(
                        item["penalties"]["nonfact_copy_ratio"]
                        for item in completion_infos
                    ),
                    "verbosity_over_budget_ratio_mean": statistics.fmean(
                        item["penalties"]["verbosity_over_budget_ratio"]
                        for item in completion_infos
                    ),
                    "protocol_pass_rate": statistics.fmean(
                        item["protocol_pass"] for item in completion_infos
                    ),
                    "intent_response_fail_rate": statistics.fmean(
                        not item["intent_response_pass"]
                        for item in completion_infos
                    ),
                    "critical_grounding_pass_rate": statistics.fmean(
                        item["critical_grounding_pass"]
                        for item in completion_infos
                    ),
                    "non_answer_pass_rate": statistics.fmean(
                        item["non_answer_pass"] for item in completion_infos
                    ),
                    "empty_rate": statistics.fmean(
                        item["empty"] for item in completion_infos
                    ),
                    "refusal_rate": statistics.fmean(
                        item["refusal"] for item in completion_infos
                    ),
                    "completion_lengths": case_lengths,
                    "unique_ratio": unique_ratio,
                    "distinct_2": ngram_distinct(texts, 2),
                    "distinct_4": ngram_distinct(texts, 4),
                    "completely_duplicate": unique_ratio == 1 / len(texts),
                    "gate_counts": dict(
                        Counter(item["gate"] for item in completion_infos)
                    ),
                    "all_finite": all(
                        math.isfinite(float(value))
                        for info in completion_infos
                        for value in (
                            info["total"],
                            info["factual_precision"],
                            info["required_coverage"],
                            info["source_integrity"],
                            info["concision"],
                        )
                    ),
                }
                case_results.append(case_result)
    if was_training:
        model.train()
    return summarize_fixed_dev4_probe(
        case_results=case_results,
        completion_lengths=completion_lengths,
        kl_values=kl_values,
        signed_kl_values=signed_kl_values,
        optimizer_step=optimizer_step,
        train_window=train_window,
        config=config,
    )


def make_audited_trainer_class(
    *,
    torch: Any,
    GRPOTrainer: Any,
    Trainer: Any,
    maybe_apply_chat_template: Any,
    unwrap_model_for_generation: Any,
    is_conversational: Any,
):
    class AuditedGRPOTrainer(GRPOTrainer):
        def __init__(self, *args: Any, audit: TrainingAudit, **kwargs: Any):
            self.training_audit = audit
            super().__init__(*args, **kwargs)

        def compute_loss(
            self,
            model: Any,
            inputs: list[dict[str, Any]],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ):
            if return_outputs:
                raise ValueError("Audited GRPO does not return model outputs")
            started = time.perf_counter()
            device = self.accelerator.device
            prompts = [item["prompt"] for item in inputs]
            prompt_texts = [
                maybe_apply_chat_template(
                    item,
                    self.processing_class,
                )["prompt"]
                for item in inputs
            ]
            prompt_inputs = self.processing_class(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
            prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
            if prompt_inputs["input_ids"].shape[1] > self.max_prompt_length:
                raise RuntimeError("formal prompt exceeds max_prompt_length")

            with unwrap_model_for_generation(
                model,
                self.accelerator,
            ) as unwrapped_model:
                with rollout_generation_cache_context(
                    unwrapped_model,
                    self.generation_config,
                ):
                    prompt_completion_ids = unwrapped_model.generate(
                        **prompt_inputs,
                        generation_config=self.generation_config,
                    )
            prompt_length = prompt_inputs["input_ids"].size(1)
            if prompt_completion_ids.shape[1] > 3072:
                raise RuntimeError("formal total sequence exceeds 3072")
            completion_ids = prompt_completion_ids[:, prompt_length:]

            def selected_logps(
                target_model: Any,
                input_ids: Any,
                keep: int,
            ):
                logits = target_model(
                    input_ids,
                    num_logits_to_keep=keep + 1,
                ).logits[:, :-1, :]
                rows = []
                for logits_row, ids_row in zip(
                    logits,
                    input_ids[:, -keep:],
                ):
                    rows.append(
                        torch.gather(
                            logits_row.log_softmax(dim=-1),
                            dim=1,
                            index=ids_row.unsqueeze(1),
                        ).squeeze(1)
                    )
                return torch.stack(rows)

            keep = completion_ids.size(1)
            policy_logps = selected_logps(
                model,
                prompt_completion_ids,
                keep,
            )
            with torch.inference_mode():
                if self.ref_model is not None:
                    raise RuntimeError("second reference model is forbidden")
                with self.accelerator.unwrap_model(model).disable_adapter():
                    reference_logps = selected_logps(
                        model,
                        prompt_completion_ids,
                        keep,
                    )
            signed_kl = policy_logps - reference_logps
            per_token_kl = (
                torch.exp(reference_logps - policy_logps)
                - (reference_logps - policy_logps)
                - 1
            )

            is_eos = completion_ids == self.processing_class.eos_token_id
            eos_index = torch.full(
                (is_eos.size(0),),
                is_eos.size(1),
                dtype=torch.long,
                device=device,
            )
            has_eos = is_eos.any(dim=1)
            eos_index[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
            positions = torch.arange(
                is_eos.size(1),
                device=device,
            ).expand(is_eos.size(0), -1)
            completion_mask = (
                positions <= eos_index.unsqueeze(1)
            ).int()
            completion_texts = self.processing_class.batch_decode(
                completion_ids,
                skip_special_tokens=True,
            )
            completions: Any = completion_texts
            if is_conversational(inputs[0]):
                completions = [
                    [{"role": "assistant", "content": text}]
                    for text in completion_texts
                ]

            repeated_prompts = [
                prompt
                for prompt in prompts
                for _ in range(self.num_generations)
            ]
            reward_kwargs = {
                key: []
                for key in inputs[0]
                if key not in {"prompt", "completion"}
            }
            for key in reward_kwargs:
                for item in inputs:
                    reward_kwargs[key].extend(
                        [item[key]] * self.num_generations
                    )
            reward_values = self.reward_funcs[0](
                prompts=repeated_prompts,
                completions=completions,
                **reward_kwargs,
            )
            rewards = torch.tensor(
                reward_values,
                dtype=torch.float32,
                device=device,
            )
            grouped_mean = rewards.view(
                -1,
                self.num_generations,
            ).mean(dim=1)
            grouped_std = rewards.view(
                -1,
                self.num_generations,
            ).std(dim=1)
            repeated_mean = grouped_mean.repeat_interleave(
                self.num_generations
            )
            repeated_std = grouped_std.repeat_interleave(
                self.num_generations
            )
            advantages = (
                rewards - repeated_mean
            ) / (repeated_std + 1e-4)

            zero_variance = bool((grouped_std < 1e-4).all().item())
            if zero_variance:
                loss = policy_logps.sum() * 0.0
            else:
                policy_term = torch.exp(
                    policy_logps - policy_logps.detach()
                ) * advantages.unsqueeze(1)
                token_loss = -(policy_term - self.beta * per_token_kl)
                loss = (
                    (token_loss * completion_mask).sum(dim=1)
                    / completion_mask.sum(dim=1)
                ).mean()

            mask = completion_mask.bool()
            masked_kl = per_token_kl[mask].detach().float()
            masked_signed = signed_kl[mask].detach().float()
            torch.cuda.synchronize()
            self.training_audit.record_compute(
                {
                    "kl_signed_mean": float(masked_signed.mean().cpu()),
                    "kl_mean": float(masked_kl.mean().cpu()),
                    "kl_p50": float(
                        torch.quantile(masked_kl, 0.50).cpu()
                    ),
                    "kl_p95": float(
                        torch.quantile(masked_kl, 0.95).cpu()
                    ),
                    "kl_max": float(masked_kl.max().cpu()),
                    "advantage_mean": float(
                        advantages.mean().detach().cpu()
                    ),
                    "advantage_std": float(
                        advantages.std().detach().cpu()
                    ),
                    "advantage_non_finite": int(
                        (~torch.isfinite(advantages)).sum().detach().cpu()
                    ),
                    "zero_variance_update_skipped": zero_variance,
                    "step_time_sec": time.perf_counter() - started,
                    "cuda_allocated_mib": (
                        torch.cuda.memory_allocated() / 1024**2
                    ),
                    "cuda_reserved_mib": (
                        torch.cuda.max_memory_reserved() / 1024**2
                    ),
                }
            )
            self._metrics["completion_length"].append(
                completion_mask.sum(1).float().mean().item()
            )
            self._metrics["rewards/grounding_reward"].append(
                rewards.mean().item()
            )
            self._metrics["reward"].append(rewards.mean().item())
            self._metrics["reward_std"].append(grouped_std.mean().item())
            self._metrics["kl"].append(masked_kl.mean().item())
            return loss

    return AuditedGRPOTrainer


def make_health_callback_class(TrainerCallback: Any):
    class HealthCallback(TrainerCallback):
        def __init__(
            self,
            *,
            audit: TrainingAudit,
            expected_hashes: dict[str, str],
            dev_cases: list[dict[str, Any]],
            tokenizer: Any,
            torch: Any,
            training_config: dict[str, Any],
        ) -> None:
            self.audit = audit
            self.expected_hashes = expected_hashes
            self.dev_cases = dev_cases
            self.tokenizer = tokenizer
            self.torch = torch
            self.training_config = training_config
            self.dev_history: list[dict[str, Any]] = []
            self.dev_bad_evaluations = 0
            self.dev_best = None

        def _trigger_abort(
            self,
            *,
            step: int,
            reasons: list[str],
            control: Any,
        ) -> Any:
            if self.audit.abort_triggered:
                return control
            self.audit.abort_triggered = True
            self.audit.abort_reasons = reasons
            checkpoints = sorted(
                str(path.relative_to(ROOT))
                for path in OUTPUT_DIR.glob("checkpoint-*")
                if path.is_dir()
            )
            write_json_exclusive(
                ABORT_PATH,
                {
                    "status": "safety_abort",
                    "run_id": RUN_ID,
                    "first_violating_step": step,
                    "reasons": reasons,
                    "recent_health_windows": self.audit.windows[-2:],
                    "last_good_checkpoints": checkpoints,
                    "current_checkpoint": None,
                    "resume_allowed": False,
                    "automatic_restart": False,
                    "automatic_hyperparameter_change": False,
                    "authorization_sha256": sha256_file(AUTH_PATH),
                    "reward_fn_sha256": sha256_file(
                        ROOT / "training" / "grpo" / "reward_fn.py"
                    ),
                    "cuda_reserved_mib": (
                        self.torch.cuda.max_memory_reserved() / 1024**2
                    ),
                },
            )
            write_json_exclusive(
                RESUME_PATH,
                {
                    "resume_allowed": False,
                    "reason": "safety_abort_requires_user_decision",
                },
            )
            write_json_exclusive(
                QUARANTINE_PATH,
                {
                    "quarantined_not_selectable": True,
                    "checkpoint": None,
                    "reason": (
                        "Abort occurred before a current-step checkpoint was "
                        "saved; no checkpoint may be promoted automatically."
                    ),
                },
            )
            control.should_training_stop = True
            control.should_save = False
            print(
                "AUTO_ABORT "
                + json.dumps(
                    {"step": step, "reasons": reasons},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return control

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> Any:
            if not logs or "loss" not in logs:
                return control
            step = int(state.global_step)
            record = self.audit.consume_step(step=step, logs=logs)
            health_window_steps = self.training_config["health_window_steps"]
            if step % health_window_steps != 0:
                return control
            train_window = window_summary(
                self.audit.steps[-health_window_steps:]
            )
            model = kwargs.get("model")
            if model is None:
                raise RuntimeError("fixed dev-4 health probe missing model")
            window = evaluate_fixed_dev4_probe_no_grad(
                model=model,
                tokenizer=self.tokenizer,
                cases=self.dev_cases,
                torch=self.torch,
                config=self.training_config,
                optimizer_step=step,
                train_window=train_window,
            )
            reasons = abort_reasons(
                window,
                self.audit.windows[-1] if self.audit.windows else None,
                self.audit.baseline,
            )
            for relative, expected in self.expected_hashes.items():
                if sha256_file(ROOT / relative) != expected:
                    reasons.append(f"data_or_evidence_sha_drift:{relative}")
            window["abort_reasons"] = sorted(set(reasons))
            append_jsonl(WINDOW_LOG_PATH, window)
            append_jsonl(FIXED_PROBE_WINDOW_LOG_PATH, window)
            self.audit.windows.append(window)
            print(
                "HEALTH_WINDOW "
                + json.dumps(window, ensure_ascii=False),
                flush=True,
            )
            if reasons:
                return self._trigger_abort(
                    step=step,
                    reasons=sorted(set(reasons)),
                    control=control,
                )
            return control

        def on_step_end(
            self,
            args: Any,
            state: Any,
            control: Any,
            model: Any = None,
            **kwargs: Any,
        ) -> Any:
            step = int(state.global_step)
            if (
                step == 0
                or step % self.training_config["dev_every_steps"] != 0
                or self.audit.abort_triggered
            ):
                return control
            result = evaluate_dev_after_abort(
                model=model,
                tokenizer=self.tokenizer,
                cases=self.dev_cases,
                torch=self.torch,
                config=self.training_config,
            )
            result["step"] = step
            append_jsonl(RUN_DIR / "dev_evaluations.jsonl", result)
            self.dev_history.append(result)
            score = result["grounding_core_mean"]
            if self.dev_best is None or score >= self.dev_best + 0.01:
                self.dev_best = score
                self.dev_bad_evaluations = 0
            else:
                self.dev_bad_evaluations += 1
            if self.dev_bad_evaluations >= 2:
                control.should_training_stop = True
                print(
                    "EARLY_STOP "
                    + json.dumps(
                        {"step": step, "dev_grounding_core": score},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            return control

    return HealthCallback


def run_authorized_training() -> int:
    authorization = json.loads(AUTH_PATH.read_text(encoding="utf-8"))
    preregistration = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    if authorization["run_id"] != RUN_ID:
        raise ValueError("unexpected formal run ID")
    if (
        preregistration["authorization_manifest_sha256"]
        != sha256_file(AUTH_PATH)
    ):
        raise ValueError("preregistration authorization SHA drift")
    if RUN_DIR.exists() or OUTPUT_DIR.exists():
        raise FileExistsError("formal GRPO run outputs already exist")
    if Path(sys.prefix).resolve() != Path(
        authorization["authorization_gate"]["training_environment"]["path"]
    ).resolve():
        raise RuntimeError("formal training must use frozen .venv-grpo")
    if authorization["authorization_gate"]["model_loading"] != {
        "local_files_only": True,
        "trust_remote_code": False,
    }:
        raise ValueError("local-only model loading gate drift")

    hashes_before = preflight_hashes(authorization)
    venv_before = venv_train_hashes()
    fixed_dev4_baseline = load_fixed_dev4_baseline(authorization)
    input_manifest = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    if (
        sha256_file(INPUT_PATH)
        != authorization["frozen_inputs"]["expanded_v4_input_manifest_sha256"]
    ):
        raise ValueError("formal evidence/spec manifest SHA drift")
    if (
        sha256_file(GRPO_DIR / "reward_train_expanded_v4.jsonl")
        != authorization["frozen_inputs"]["expanded_v4_train_sha256"]
    ):
        raise ValueError("expanded formal train SHA drift")
    if (
        sha256_file(GRPO_DIR / "reward_train_expanded_v4_manifest.json")
        != authorization["frozen_inputs"]["expanded_v4_manifest_sha256"]
    ):
        raise ValueError("expanded formal train manifest SHA drift")

    RUN_DIR.mkdir(parents=True, exist_ok=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(
        RUN_DIR / "start_receipt.json",
        {
            "status": "authorized_start",
            "run_id": authorization["run_id"],
            "run_attempt": RUN_ATTEMPT or "initial",
            "run_directory": str(RUN_DIR.relative_to(ROOT)),
            "checkpoint_directory": str(OUTPUT_DIR.relative_to(ROOT)),
            "authorization_sha256": sha256_file(AUTH_PATH),
            "preregistration_sha256": sha256_file(PREREG_PATH),
            "five_element_gate": {
                "reward_fn_sha_match": True,
                "split_three_file_sha_match": True,
                "dev_authorization_sha_match": True,
                "venv_grpo_path_match": True,
                "local_files_only": True,
            },
            "hashes_before": hashes_before,
            "venv_train_before": venv_before,
            "fixed_dev4_step0_baseline_sha256": sha256_file(
                FIXED_DEV4_BASELINE_PATH
            ),
            "final_40_accessed": False,
            "held_out_40_accessed": False,
        },
    )
    write_json_exclusive(BASELINE_PATH, fixed_dev4_baseline)

    try:
        import torch
        from datasets import Dataset
        from peft import (
            LoraConfig,
            prepare_model_for_kbit_training,
            set_peft_model_state_dict,
        )
        from safetensors.torch import load_file
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainerCallback,
        )
        from trl import GRPOConfig
        from trl.trainer.grpo_trainer import (
            GRPOTrainer,
            is_conversational,
            maybe_apply_chat_template,
            unwrap_model_for_generation,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable for formal GRPO")
        if (
            torch.cuda.get_device_name(0) != "Tesla V100-SXM2-32GB"
            or torch.cuda.get_device_capability(0) != (7, 0)
        ):
            raise RuntimeError("formal GRPO requires V100 compute 7.0")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        config = authorization["training_config"]
        tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_PATH),
            local_files_only=True,
            trust_remote_code=False,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        train_cases = [
            case
            for case in input_manifest["cases"]
            if case["split"] == "train"
        ]
        dev_cases = [
            case
            for case in input_manifest["cases"]
            if case["split"] == "dev"
        ]
        if len(train_cases) < 800 or len(dev_cases) != 4:
            raise ValueError("formal expanded train/dev case count drift")
        dev_cases = fixed_dev4_cases(dev_cases)
        dataset_rows: list[dict[str, Any]] = []
        for case in train_cases:
            prompt = build_prompt(case)
            rendered = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_length = len(
                tokenizer(
                    rendered,
                    add_special_tokens=False,
                )["input_ids"]
            )
            if prompt_length > config["max_prompt_length"]:
                raise RuntimeError(
                    f"{case['id']}: prompt exceeds max length"
                )
            dataset_rows.append(
                {
                    "prompt": prompt,
                    "case_id": case["id"],
                }
            )
        train_dataset = Dataset.from_list(dataset_rows)
        cases_by_id = {
            case["id"]: case for case in input_manifest["cases"]
        }

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            quantization_config=quantization,
            device_map={"": 0},
            torch_dtype=torch.float16,
            local_files_only=True,
            trust_remote_code=False,
        )
        base_model.config.use_cache = False
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=True,
        )
        adapter_config = json.loads(
            (ADAPTER_PATH / "adapter_config.json").read_text(
                encoding="utf-8"
            )
        )
        peft_config = LoraConfig(
            r=adapter_config["r"],
            lora_alpha=adapter_config["lora_alpha"],
            lora_dropout=adapter_config["lora_dropout"],
            target_modules=adapter_config["target_modules"],
            bias=adapter_config["bias"],
            task_type=adapter_config["task_type"],
            inference_mode=False,
        )
        baseline = fixed_dev4_baseline
        audit = TrainingAudit(
            cases=cases_by_id,
            tokenizer=tokenizer,
            authorization=authorization,
            baseline=baseline,
        )
        audit.reward_function.__func__.__name__ = "grounding_reward"
        expected_hashes = {
            **{
                item["path"]: item["sha256"]
                for item in (
                    authorization["authorization_gate"]["reward_fn"],
                    authorization["authorization_gate"][
                        "dev_authorization"
                    ],
                )
            },
            authorization["authorization_gate"]["split"]["train_path"]: (
                authorization["authorization_gate"]["split"][
                    "train_sha256"
                ]
            ),
            authorization["authorization_gate"]["split"]["dev_path"]: (
                authorization["authorization_gate"]["split"]["dev_sha256"]
            ),
            authorization["authorization_gate"]["split"]["manifest_path"]: (
                authorization["authorization_gate"]["split"][
                    "manifest_sha256"
                ]
            ),
            str(INPUT_PATH.relative_to(ROOT)): sha256_file(INPUT_PATH),
            authorization["fixed_dev4_probe_step0_baseline"]["path"]: (
                authorization["fixed_dev4_probe_step0_baseline"]["sha256"]
            ),
        }

        HealthCallback = make_health_callback_class(TrainerCallback)
        callback = HealthCallback(
            audit=audit,
            expected_hashes=expected_hashes,
            dev_cases=dev_cases,
            tokenizer=tokenizer,
            torch=torch,
            training_config=config,
        )
        args = GRPOConfig(
            output_dir=str(OUTPUT_DIR),
            overwrite_output_dir=False,
            do_train=True,
            per_device_train_batch_size=config[
                "per_device_train_batch_size"
            ],
            gradient_accumulation_steps=config[
                "gradient_accumulation_steps"
            ],
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            max_grad_norm=config["max_grad_norm"],
            max_steps=config["max_optimizer_steps"],
            lr_scheduler_type=config["lr_scheduler"],
            warmup_steps=config["warmup_steps"],
            logging_strategy="steps",
            logging_steps=1,
            logging_first_step=True,
            save_strategy="steps",
            save_steps=config["save_every_steps"],
            save_total_limit=3,
            report_to=[],
            disable_tqdm=True,
            remove_unused_columns=False,
            fp16=True,
            bf16=False,
            tf32=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_prompt_length=config["max_prompt_length"],
            max_completion_length=config["max_completion_length"],
            num_generations=config["num_generations"],
            temperature=config["temperature"],
            beta=config["beta"],
            use_vllm=False,
            optim=config["optimizer"],
            seed=config["seed"],
            dataloader_num_workers=0,
        )
        AuditedTrainer = make_audited_trainer_class(
            torch=torch,
            GRPOTrainer=GRPOTrainer,
            Trainer=Trainer,
            maybe_apply_chat_template=maybe_apply_chat_template,
            unwrap_model_for_generation=unwrap_model_for_generation,
            is_conversational=is_conversational,
        )
        trainer = AuditedTrainer(
            model=base_model,
            reward_funcs=audit.reward_function,
            args=args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            callbacks=[callback],
            peft_config=peft_config,
            audit=audit,
        )
        if trainer.ref_model is not None:
            raise RuntimeError("separate reference model is forbidden")
        trainer.generation_config.top_p = config["top_p"]
        adapter_state = load_file(
            str(ADAPTER_PATH / "adapter_model.safetensors"),
            device="cpu",
        )
        load_result = set_peft_model_state_dict(
            trainer.model,
            adapter_state,
            adapter_name="default",
        )
        if list(getattr(load_result, "unexpected_keys", ())):
            raise RuntimeError("unexpected SFT adapter keys")

        started = time.time()
        train_output = trainer.train()
        completed_step = int(trainer.state.global_step)
        dev_abort = None
        if audit.abort_triggered:
            dev_abort = evaluate_dev_after_abort(
                model=trainer.model,
                tokenizer=tokenizer,
                cases=dev_cases,
                torch=torch,
                config=config,
            )
            write_json_exclusive(DEV_ABORT_PATH, dev_abort)

        write_json_exclusive(
            RUN_DIR / "hyperparameters_locked.json",
            {
                "locked": True,
                "authorization_sha256": sha256_file(AUTH_PATH),
                "training_config": config,
                "dev_metrics_changed_hyperparameters": False,
            },
        )
        selection = {
            "performed": False,
            "selected_checkpoint": None,
            "reason": (
                "safety_abort_no_checkpoint_selection"
                if audit.abort_triggered
                else "pending_post_training_selection"
            ),
        }
        if not audit.abort_triggered:
            checkpoints = sorted(
                OUTPUT_DIR.glob("checkpoint-*"),
                key=lambda path: int(path.name.rsplit("-", 1)[-1]),
            )
            eligible: list[tuple[float, int, Path]] = []
            for dev in callback.dev_history:
                step = int(dev["step"])
                checkpoint = OUTPUT_DIR / f"checkpoint-{step}"
                all_gates_pass = all(
                    result["gate_counts"] == {"passed": 8}
                    for result in dev["cases"]
                )
                if checkpoint.is_dir() and all_gates_pass:
                    eligible.append(
                        (
                            dev["grounding_core_mean"],
                            -step,
                            checkpoint,
                        )
                    )
            if eligible:
                _, _, selected = max(eligible)
                final_adapter = OUTPUT_DIR / "final_adapter"
                shutil.copytree(selected, final_adapter)
                selection = {
                    "performed": True,
                    "selected_checkpoint": str(selected.relative_to(ROOT)),
                    "final_adapter": str(final_adapter.relative_to(ROOT)),
                    "final_40_accessed": False,
                }
            else:
                selection = {
                    "performed": True,
                    "selected_checkpoint": None,
                    "reason": "no_checkpoint_passed_all_dev_gates",
                    "candidate_checkpoints": [
                        str(path.relative_to(ROOT)) for path in checkpoints
                    ],
                }

        hashes_after = preflight_hashes(authorization)
        venv_after = venv_train_hashes()
        run_report = {
            "status": (
                "safety_abort"
                if audit.abort_triggered
                else "completed_or_early_stopped"
            ),
            "run_id": authorization["run_id"],
            "started_unix_sec": started,
            "completed_unix_sec": time.time(),
            "completed_optimizer_steps": completed_step,
            "planned_optimizer_steps": config["max_optimizer_steps"],
            "abort_reasons": audit.abort_reasons,
            "train_loss": float(train_output.training_loss),
            "health_windows": audit.windows,
            "dev_history": callback.dev_history,
            "dev_abort_diagnostics": dev_abort,
            "checkpoint_selection": selection,
            "resume_allowed": not audit.abort_triggered,
            "final_40_accessed": False,
            "held_out_40_accessed": False,
            "hashes_before": hashes_before,
            "hashes_after": hashes_after,
            "frozen_hashes_zero_change": hashes_before == hashes_after,
            "venv_train_before": venv_before,
            "venv_train_after": venv_after,
            "venv_train_zero_change": venv_before == venv_after,
            "cuda_peak_reserved_mib": (
                torch.cuda.max_memory_reserved() / 1024**2
            ),
        }
        write_json_exclusive(RUN_REPORT_PATH, run_report)
        return 2 if audit.abort_triggered else 0
    except Exception as error:
        if not RUN_REPORT_PATH.exists():
            if not RESUME_PATH.exists():
                write_json_exclusive(
                    RESUME_PATH,
                    {
                        "resume_allowed": False,
                        "reason": "formal_training_exception",
                    },
                )
            write_json_exclusive(
                RUN_REPORT_PATH,
                {
                    "status": "failed_exception",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "resume_allowed": False,
                    "final_40_accessed": False,
                    "held_out_40_accessed": False,
                    "hashes_before": hashes_before,
                    "venv_train_before": venv_before,
                },
            )
        return 1
