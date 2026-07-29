"""Fail-closed reward helpers for GRPO/RLVR experiments."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from data_synth.validate_tool_data import validate_record


def _normalize_id(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.strip().split()).casefold()


def _normalize_id_set(values: Iterable[str], label: str) -> set[str]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be an iterable container of strings")
    normalized: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise ValueError(f"{label} must contain only strings")
        normalized_value = _normalize_id(value)
        if not normalized_value:
            raise ValueError(f"{label} must not contain empty IDs")
        normalized.add(normalized_value)
    return normalized


def _normalize_match_token(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(
        character
        for character in normalized
        if character.isalnum() or character in {"+", "-", "."}
    )


def _validated_tokens(values: Iterable[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be an iterable container of strings")
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must contain only non-empty strings")
        normalized = _normalize_match_token(value)
        if not normalized:
            raise ValueError(f"{label} contains an empty normalized token")
        if normalized not in seen:
            seen.add(normalized)
            output.append(value.strip())
    return tuple(output)


@dataclass(frozen=True)
class EvidenceClaim:
    """One immutable entity/attribute/value proposition from tool evidence."""

    canonical_entity: str
    canonical_attribute: str
    canonical_value: str
    source_tool: str
    source_locator: str
    entity_aliases: tuple[str, ...] = ()
    attribute_aliases: tuple[str, ...] = ()
    anchor_tokens: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "canonical_entity",
            "canonical_attribute",
            "canonical_value",
            "source_tool",
            "source_locator",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"EvidenceClaim.{field_name} must be non-empty")
            object.__setattr__(self, field_name, value.strip())
        object.__setattr__(
            self,
            "entity_aliases",
            _validated_tokens(self.entity_aliases, "entity_aliases"),
        )
        object.__setattr__(
            self,
            "attribute_aliases",
            _validated_tokens(self.attribute_aliases, "attribute_aliases"),
        )
        object.__setattr__(
            self,
            "anchor_tokens",
            _validated_tokens(self.anchor_tokens, "anchor_tokens"),
        )


@dataclass(frozen=True)
class IntentResponseSpec:
    """Frozen query anchors and obligations for one reward-visible prompt."""

    prompt_id: str
    intent: str
    target_entities: tuple[str, ...]
    query_anchor_tokens: tuple[str, ...] = ()
    query_attribute_anchors: tuple[str, ...] = ()
    minimum_supported_claims: int = 1
    decision_tokens: tuple[str, ...] = ()
    communication_action_tokens: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.prompt_id, str) or not self.prompt_id.strip():
            raise ValueError("IntentResponseSpec.prompt_id must be non-empty")
        if self.intent not in {"compare", "recommend", "knowledge", "sales"}:
            raise ValueError("IntentResponseSpec.intent is unsupported")
        if (
            isinstance(self.minimum_supported_claims, bool)
            or not isinstance(self.minimum_supported_claims, int)
            or self.minimum_supported_claims <= 0
        ):
            raise ValueError("minimum_supported_claims must be a positive integer")
        object.__setattr__(
            self,
            "target_entities",
            _validated_tokens(self.target_entities, "target_entities"),
        )
        object.__setattr__(
            self,
            "query_anchor_tokens",
            _validated_tokens(
                self.query_anchor_tokens,
                "query_anchor_tokens",
            ),
        )
        object.__setattr__(
            self,
            "query_attribute_anchors",
            _validated_tokens(
                self.query_attribute_anchors,
                "query_attribute_anchors",
            ),
        )
        object.__setattr__(
            self,
            "decision_tokens",
            _validated_tokens(self.decision_tokens, "decision_tokens"),
        )
        object.__setattr__(
            self,
            "communication_action_tokens",
            _validated_tokens(
                self.communication_action_tokens,
                "communication_action_tokens",
            ),
        )


@dataclass(frozen=True)
class IntentResponseResult:
    passed: bool
    reason: str
    matched_claim_count: int
    matched_entities: tuple[str, ...]
    matched_attributes: tuple[str, ...]


@dataclass(frozen=True)
class GroundingRewardResult:
    total: float
    factual_precision: float
    required_coverage: float
    source_integrity: float
    concision: float
    gate: str
    matched_claim_count: int


_HARD_VALUE_PATTERN = re.compile(
    r"""
    \d+(?:\.\d+)?
    (?:
        \s*(?:-|~|至)\s*\d+(?:\.\d+)?
    )?
    \s*
    (?:
        万元?|万|km|公里|kwh|千瓦时|分钟|mm|毫米|l|升|座|辆/月|分|%|v
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


def _atomic_answer_segments(answer: str) -> tuple[str, ...]:
    return tuple(
        segment.strip()
        for segment in re.split(r"[\n。！？!?；;，,]+", answer)
        if segment.strip()
    )


def _table_claim_segments(answer: str) -> tuple[str, ...]:
    rows: list[list[str]] = []
    for line in answer.splitlines():
        if "|" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2:
            continue
        if all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells):
            continue
        rows.append(cells)
    if len(rows) < 2:
        return ()

    header = rows[0]
    segments: list[str] = []
    for row in rows[1:]:
        if len(row) != len(header):
            continue
        for index in range(1, len(row)):
            segments.append(f"{row[0]} {header[index]} {row[index]}")
            segments.append(f"{header[index]} {header[0]} {row[0]}")
    return tuple(segments)


def _claim_segments(answer: str) -> tuple[str, ...]:
    broad = tuple(
        segment.strip()
        for segment in re.split(r"[\n。！？!?；;]+", answer)
        if segment.strip()
    )
    return tuple(
        dict.fromkeys(
            broad
            + _atomic_answer_segments(answer)
            + _table_claim_segments(answer)
        )
    )


def _claim_entity_tokens(claim: EvidenceClaim) -> tuple[str, ...]:
    return (claim.canonical_entity,) + claim.entity_aliases


def _claim_attribute_tokens(claim: EvidenceClaim) -> tuple[str, ...]:
    leaf = claim.canonical_attribute.rsplit(".", 1)[-1]
    return (claim.canonical_attribute, leaf) + claim.attribute_aliases


def _segment_contains_any(segment: str, tokens: Iterable[str]) -> bool:
    normalized_segment = _normalize_match_token(segment)
    return any(
        _normalize_match_token(token) in normalized_segment
        for token in tokens
    )


def _claim_is_query_bound(
    claim: EvidenceClaim,
    segment: str,
    spec: IntentResponseSpec,
) -> bool:
    target_entities = {
        _normalize_match_token(entity) for entity in spec.target_entities
    }
    if target_entities and _normalize_match_token(
        claim.canonical_entity
    ) not in target_entities:
        return False

    query_attributes = {
        _normalize_match_token(attribute)
        for attribute in spec.query_attribute_anchors
    }
    attribute_bound = (
        bool(query_attributes)
        and _normalize_match_token(claim.canonical_attribute)
        in query_attributes
    )

    query_tokens = {
        _normalize_match_token(token) for token in spec.query_anchor_tokens
    }
    evidence_tokens = {
        _normalize_match_token(token) for token in claim.anchor_tokens
    }
    shared_tokens = query_tokens & evidence_tokens
    token_bound = bool(shared_tokens) and _segment_contains_any(
        segment,
        shared_tokens,
    )
    return attribute_bound or token_bound


def _matching_claims(
    answer: str,
    spec: IntentResponseSpec,
    evidence_claims: tuple[EvidenceClaim, ...],
) -> tuple[EvidenceClaim, ...]:
    matched: list[EvidenceClaim] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for claim in evidence_claims:
        for segment in _claim_segments(answer):
            if not _segment_contains_any(
                segment,
                _claim_entity_tokens(claim),
            ):
                continue
            if not _segment_contains_any(
                segment,
                _claim_attribute_tokens(claim),
            ):
                continue
            if not _segment_contains_any(
                segment,
                (claim.canonical_value,),
            ):
                continue
            if not _claim_is_query_bound(claim, segment, spec):
                continue
            key = (
                claim.canonical_entity,
                claim.canonical_attribute,
                claim.canonical_value,
                claim.source_tool,
                claim.source_locator,
            )
            if key not in seen:
                seen.add(key)
                matched.append(claim)
            break
    return tuple(matched)


def _unsupported_target_value_count(
    answer: str,
    spec: IntentResponseSpec,
    evidence_claims: tuple[EvidenceClaim, ...],
) -> int:
    query_attributes = {
        _normalize_match_token(attribute)
        for attribute in spec.query_attribute_anchors
    }
    if not query_attributes:
        return 0

    claims_by_entity_attribute: dict[
        tuple[str, str],
        list[EvidenceClaim],
    ] = {}
    for claim in evidence_claims:
        entity = _normalize_match_token(claim.canonical_entity)
        attribute = _normalize_match_token(claim.canonical_attribute)
        if attribute not in query_attributes:
            continue
        claims_by_entity_attribute.setdefault(
            (entity, attribute),
            [],
        ).append(claim)

    unsupported = 0
    for segment in _claim_segments(answer):
        if not _HARD_VALUE_PATTERN.search(
            unicodedata.normalize("NFKC", segment)
        ):
            continue
        for claims in claims_by_entity_attribute.values():
            representative = claims[0]
            if not _segment_contains_any(
                segment,
                _claim_entity_tokens(representative),
            ):
                continue
            attribute_tokens = tuple(
                token
                for claim in claims
                for token in _claim_attribute_tokens(claim)
            )
            if not _segment_contains_any(segment, attribute_tokens):
                continue
            if not any(
                _segment_contains_any(
                    segment,
                    (claim.canonical_value,),
                )
                for claim in claims
            ):
                unsupported += 1
    return unsupported


def _ordered_matches(
    requested: tuple[str, ...],
    observed: set[str],
) -> tuple[str, ...]:
    return tuple(
        item
        for item in requested
        if _normalize_match_token(item) in observed
    )


def deterministic_intent_response_check(
    *,
    answer: str,
    spec: IntentResponseSpec,
    evidence_claims: Iterable[EvidenceClaim],
) -> IntentResponseResult:
    """Fail closed unless the answer uses truthful query-bound evidence."""
    if not isinstance(answer, str) or not answer.strip():
        return IntentResponseResult(False, "empty_answer", 0, (), ())
    if not isinstance(spec, IntentResponseSpec):
        raise TypeError("spec must be an IntentResponseSpec")
    claims = tuple(evidence_claims)
    if not claims or not all(
        isinstance(claim, EvidenceClaim) for claim in claims
    ):
        return IntentResponseResult(False, "missing_evidence_claims", 0, (), ())

    matched = _matching_claims(answer, spec, claims)
    matched_entity_keys = {
        _normalize_match_token(claim.canonical_entity) for claim in matched
    }
    matched_entities = _ordered_matches(
        spec.target_entities,
        matched_entity_keys,
    )
    matched_attributes = tuple(
        sorted({claim.canonical_attribute for claim in matched})
    )
    base = {
        "matched_claim_count": len(matched),
        "matched_entities": matched_entities,
        "matched_attributes": matched_attributes,
    }

    if _unsupported_target_value_count(answer, spec, claims):
        return IntentResponseResult(
            False,
            "unsupported_target_value",
            **base,
        )

    if spec.intent == "compare":
        if len(spec.target_entities) != 2:
            return IntentResponseResult(
                False,
                "compare_requires_two_frozen_targets",
                **base,
            )
        if len(matched_entities) != 2:
            return IntentResponseResult(
                False,
                "missing_query_bound_claim_for_target",
                **base,
            )
    elif spec.intent == "recommend":
        if not spec.decision_tokens:
            return IntentResponseResult(
                False,
                "missing_frozen_decision_tokens",
                **base,
            )
        decision_bound = any(
            _segment_contains_any(segment, spec.decision_tokens)
            and any(
                _segment_contains_any(segment, (entity,))
                for entity in matched_entities
            )
            for segment in _atomic_answer_segments(answer)
        )
        if not decision_bound:
            return IntentResponseResult(
                False,
                "decision_not_bound_to_evidence_candidate",
                **base,
            )
    elif spec.intent == "knowledge":
        if not spec.query_anchor_tokens:
            return IntentResponseResult(
                False,
                "missing_frozen_query_anchor",
                **base,
            )
    elif spec.intent == "sales":
        if not spec.communication_action_tokens:
            return IntentResponseResult(
                False,
                "missing_frozen_communication_action_tokens",
                **base,
            )
        if not _segment_contains_any(
            answer,
            spec.communication_action_tokens,
        ):
            return IntentResponseResult(
                False,
                "missing_communication_action",
                **base,
            )

    if len(matched) < spec.minimum_supported_claims:
        reason = (
            "insufficient_anchor_bound_claims"
            if spec.intent == "knowledge"
            else "insufficient_query_bound_claims"
        )
        return IntentResponseResult(False, reason, **base)
    return IntentResponseResult(True, "passed", **base)


def score_grounded_answer(
    *,
    answer: str,
    spec: IntentResponseSpec,
    evidence_claims: Iterable[EvidenceClaim],
) -> GroundingRewardResult:
    """Return deterministic four-part grounding reward diagnostics."""
    claims = tuple(evidence_claims)
    intent_result = deterministic_intent_response_check(
        answer=answer,
        spec=spec,
        evidence_claims=claims,
    )
    matched = _matching_claims(answer, spec, claims) if claims else ()
    unsupported = _unsupported_target_value_count(answer, spec, claims)

    denominator = len(matched) + unsupported
    factual_precision = (
        len(matched) / denominator if denominator else 0.0
    )
    if spec.intent == "compare":
        required_coverage = (
            len(intent_result.matched_entities) / len(spec.target_entities)
            if spec.target_entities
            else 0.0
        )
    else:
        required_coverage = min(
            len(matched) / spec.minimum_supported_claims,
            1.0,
        )
    source_integrity = (
        sum(
            bool(claim.source_tool and claim.source_locator)
            for claim in matched
        )
        / len(matched)
        if matched
        else 0.0
    )
    normalized_length = len(_normalize_match_token(answer))
    concision = max(0.0, min(1.0, 1.0 - max(0, normalized_length - 800) / 800))
    total = 0.0
    gate = intent_result.reason
    if intent_result.passed:
        total = max(
            0.0,
            min(
                1.0,
                0.45 * factual_precision
                + 0.30 * required_coverage
                + 0.15 * source_integrity
                + 0.10 * concision,
            ),
        )
        gate = "passed"
    return GroundingRewardResult(
        total=total,
        factual_precision=factual_precision,
        required_coverage=required_coverage,
        source_integrity=source_integrity,
        concision=concision,
        gate=gate,
        matched_claim_count=len(matched),
    )


@dataclass(frozen=True)
class RewardContext:
    """Immutable reward split metadata with held-out leakage protection."""

    reward_visible_ids: Iterable[str]
    held_out_ids: Iterable[str]
    _reward_visible: set[str] = field(init=False, repr=False)
    _held_out: set[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        reward_visible = _normalize_id_set(
            self.reward_visible_ids,
            "reward_visible_ids",
        )
        held_out = _normalize_id_set(self.held_out_ids, "held_out_ids")
        overlap = reward_visible & held_out
        if overlap:
            raise ValueError(
                f"reward-visible and held-out overlap: {sorted(overlap)[0]}"
            )
        object.__setattr__(self, "_reward_visible", reward_visible)
        object.__setattr__(self, "_held_out", held_out)

    def is_reward_visible(self, prompt_id: str) -> bool:
        return _normalize_id(prompt_id) in self._reward_visible

    def is_held_out(self, prompt_id: str) -> bool:
        return _normalize_id(prompt_id) in self._held_out


@dataclass
class RewardCache:
    """In-memory deterministic reward cache."""

    values: dict[str, dict[str, Any]] = field(default_factory=dict)

    def key(self, prompt: str, completion: str) -> str:
        payload = json.dumps(
            {"completion": completion, "prompt": prompt},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, prompt: str, completion: str) -> dict[str, Any] | None:
        value = self.values.get(self.key(prompt, completion))
        return copy.deepcopy(value) if value is not None else None

    def set(self, prompt: str, completion: str, value: dict[str, Any]) -> None:
        self.values[self.key(prompt, completion)] = copy.deepcopy(value)


def _strict_tool_calls(tool_calls: Any) -> bool:
    if not isinstance(tool_calls, list) or not tool_calls:
        return False
    seen_ids: set[str] = set()
    for call in tool_calls:
        if not isinstance(call, dict):
            return False
        if set(call) != {"id", "type", "function"}:
            return False
        call_id = call["id"]
        if not isinstance(call_id, str) or not call_id.strip():
            return False
        if call_id in seen_ids:
            return False
        seen_ids.add(call_id)
        if call["type"] != "function":
            return False
        function = call["function"]
        if not isinstance(function, dict):
            return False
        if set(function) != {"name", "arguments"}:
            return False
    return True


def compute_format_reward(
    completion: dict[str, Any],
    *,
    format_reward_cap: float = 1.0,
) -> float:
    """Score OpenAI function-call format only; never execute tools."""
    if (
        isinstance(format_reward_cap, bool)
        or not isinstance(format_reward_cap, (int, float))
        or format_reward_cap <= 0
    ):
        return 0.0
    if not isinstance(completion, dict):
        return 0.0
    tool_calls = completion.get("tool_calls")
    if not _strict_tool_calls(tool_calls):
        return 0.0
    record = {
        "id": "reward-format-check",
        "messages": [{"role": "assistant", "tool_calls": tool_calls}],
    }
    if validate_record(record, held_out_ids=set()):
        return 0.0
    return min(float(format_reward_cap), 1.0)


def _result_has_error(content: Any) -> bool:
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            return False
    return isinstance(content, dict) and "error" in content


def compute_tool_execution_reward(completion: dict[str, Any]) -> float:
    """Require one successful tool result for every valid tool call."""
    if compute_format_reward(completion) == 0.0:
        return 0.0
    tool_calls = completion["tool_calls"]
    tool_results = completion.get("tool_results")
    if not isinstance(tool_results, list) or len(tool_results) != len(tool_calls):
        return 0.0

    expected_ids = [call["id"] for call in tool_calls]
    seen_ids: set[str] = set()
    for result in tool_results:
        if not isinstance(result, dict):
            return 0.0
        result_id = result.get("tool_call_id")
        if result_id not in expected_ids or result_id in seen_ids:
            return 0.0
        seen_ids.add(result_id)
        if "content" not in result or _result_has_error(result["content"]):
            return 0.0
    return 1.0 if seen_ids == set(expected_ids) else 0.0


def compute_answer_reward(
    prompt_id: str,
    completion: dict[str, Any],
    context: RewardContext,
    *,
    spec: IntentResponseSpec | None = None,
    evidence_claims: Iterable[EvidenceClaim] = (),
) -> float:
    """Score a frozen grounded answer while excluding all non-visible prompts."""
    if context.is_held_out(prompt_id) or not context.is_reward_visible(prompt_id):
        return 0.0
    if spec is None or spec.prompt_id != prompt_id:
        return 0.0
    answer = completion.get("answer") if isinstance(completion, dict) else None
    if not isinstance(answer, str):
        return 0.0
    return score_grounded_answer(
        answer=answer,
        spec=spec,
        evidence_claims=evidence_claims,
    ).total
