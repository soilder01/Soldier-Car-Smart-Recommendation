#!/usr/bin/env python3
"""Deterministically validate and evaluate saved model-layer outputs."""

import argparse
import csv
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

from data_synth.tool_schemas import build_tool_schemas


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VEHICLE_DATABASE = ROOT / "data" / "vehicles" / "vehicle_database.csv"
PROTOCOL_VERSION = "task14.v3"
ZERO_DENOMINATOR_RULE = "ratio=0.0 and percentage=0.0"
HALLUCINATION_SCOPE = (
    "terminal_declared_model_entities_only; "
    "unknown_free_text_omissions_are_not_deterministically_detected"
)
INTENTS = {"recommend", "compare", "knowledge", "sales"}
CASE_FIELDS = {
    "id",
    "query",
    "intent",
    "expected_tools",
    "optional_tools",
    "forbidden_tools",
    "allowed_models",
}
OUTPUT_FIELDS = {
    "id",
    "protocol_version",
    "model",
    "base_url",
    "case_digest",
    "schema_valid",
    "terminal_parse_error",
    "runner_errors",
    "raw_assistant_content",
    "tool_calls",
    "latency_ms",
    "recommended_models",
    "trajectory",
}
MANDATORY_BY_INTENT = {
    "recommend": [
        "extract_user_profile",
        "search_and_rank_vehicles",
        "generate_sales_talk",
    ],
    "compare": [
        "extract_user_profile",
        "search_and_rank_vehicles",
        "retrieve_knowledge_base",
    ],
    "knowledge": ["retrieve_knowledge_base"],
    "sales": [
        "extract_user_profile",
        "retrieve_knowledge_base",
        "generate_sales_talk",
    ],
}
OPTIONAL_BY_INTENT = {
    "recommend": ["retrieve_knowledge_base", "search_web_info"],
    "compare": ["search_web_info"],
    "knowledge": ["search_web_info"],
    "sales": [],
}
FORBIDDEN_BY_INTENT = {
    "recommend": [],
    "compare": ["generate_sales_talk"],
    "knowledge": [
        "extract_user_profile",
        "search_and_rank_vehicles",
        "generate_sales_talk",
    ],
    "sales": ["search_and_rank_vehicles", "search_web_info"],
}


def normalize_query(query: str) -> str:
    """Normalize text for exact leakage checks."""
    normalized = unicodedata.normalize("NFKC", query)
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def canonical_json(value: Any) -> str:
    """Return the canonical JSON representation used by case digests."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def case_digest(case: dict[str, Any]) -> str:
    """Hash every case field so resume cannot cross a changed case contract."""
    return hashlib.sha256(canonical_json(case).encode("utf-8")).hexdigest()


def load_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    """Load strict object-per-line JSONL with line diagnostics."""
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                raise ValueError(f"{label} line {line_number}: blank JSONL line")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{label} line {line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"{label} line {line_number}: expected JSON object"
                )
            records.append(record)
    return records


def load_vehicle_catalog(path: Path = DEFAULT_VEHICLE_DATABASE) -> list[str]:
    """Load canonical ``brand model`` names from the structured vehicle CSV."""
    models: list[str] = []
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not {"brand", "model"} <= set(
            reader.fieldnames
        ):
            raise ValueError("vehicle database must contain brand and model columns")
        for row_number, row in enumerate(reader, start=2):
            brand = (row.get("brand") or "").strip()
            model = (row.get("model") or "").strip()
            if not brand or not model:
                raise ValueError(
                    f"vehicle database row {row_number}: empty brand or model"
                )
            models.append(f"{brand} {model}")
    if len(models) != len(set(models)):
        raise ValueError("vehicle database contains duplicate canonical models")
    return models


def _index_by_id(
    records: Iterable[dict[str, Any]], *, label: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for position, record in enumerate(records):
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"{label} record {position}: ID must be a non-empty string")
        if record_id in indexed:
            raise ValueError(f"duplicate {label} ID: {record_id}")
        indexed[record_id] = record
    return indexed


def _validate_exact_fields(
    record: dict[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    missing = sorted(expected - set(record))
    extras = sorted(set(record) - expected)
    if missing:
        raise ValueError(f"{label}: missing field {missing[0]}")
    if extras:
        raise ValueError(f"{label}: unexpected field {extras[0]}")


def _validate_string_list(
    value: Any,
    *,
    label: str,
    allow_empty: bool = True,
) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    if not allow_empty and not value:
        raise ValueError(f"{label} must not be empty")
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{label}[{index}] must be a non-empty string")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} must not contain duplicates")
    return value


def _real_tools_by_intent() -> dict[str, set[str]]:
    from app.services import agent_graph

    return {
        intent: {tool.name for tool in agent_graph.TOOLS_BY_INTENT[intent]}
        for intent in INTENTS
    }


def validate_case_records(
    cases: list[dict[str, Any]],
    *,
    vehicle_catalog: set[str],
) -> None:
    """Reject every structural or intent-contract error before scoring."""
    real_tools = _real_tools_by_intent()
    approved_tools = set(_schema_map())
    for position, case in enumerate(cases):
        case_id = case.get("id", position)
        label = f"case {case_id}"
        _validate_exact_fields(case, CASE_FIELDS, label=label)
        for field in ("id", "query"):
            if not isinstance(case[field], str) or not case[field].strip():
                raise ValueError(f"{label}: {field} must be a non-empty string")

        intent = case["intent"]
        if intent not in INTENTS:
            raise ValueError(f"{label}: invalid intent {intent!r}")

        expected = _validate_string_list(
            case["expected_tools"],
            label=f"{label}: expected_tools",
            allow_empty=False,
        )
        optional = _validate_string_list(
            case["optional_tools"],
            label=f"{label}: optional_tools",
        )
        forbidden = _validate_string_list(
            case["forbidden_tools"],
            label=f"{label}: forbidden_tools",
        )
        tool_sets = [set(expected), set(optional), set(forbidden)]
        if any(
            tool_sets[left] & tool_sets[right]
            for left in range(len(tool_sets))
            for right in range(left + 1, len(tool_sets))
        ):
            raise ValueError(f"{label}: tool sets must be pairwise disjoint")
        unknown = sorted(set().union(*tool_sets) - approved_tools)
        if unknown:
            raise ValueError(f"{label}: unknown tool {unknown[0]}")

        if expected != MANDATORY_BY_INTENT[intent]:
            raise ValueError(
                f"{label}: expected_tools do not match mandatory intent contract"
            )
        if optional != OPTIONAL_BY_INTENT[intent]:
            raise ValueError(
                f"{label}: optional_tools do not match intent contract"
            )
        if forbidden != FORBIDDEN_BY_INTENT[intent]:
            raise ValueError(
                f"{label}: forbidden_tools do not match intent contract"
            )
        if set(expected + optional) != real_tools[intent]:
            raise ValueError(
                f"{label}: allowed tools differ from agent_graph.TOOLS_BY_INTENT"
            )
        if real_tools[intent] & set(forbidden):
            raise ValueError(
                f"{label}: forbidden tools overlap agent_graph.TOOLS_BY_INTENT"
            )

        allowed_models = _validate_string_list(
            case["allowed_models"],
            label=f"{label}: allowed_models",
        )
        outside_catalog = sorted(set(allowed_models) - vehicle_catalog)
        if outside_catalog:
            raise ValueError(
                f"{label}: allowed_models contains model outside catalog: "
                f"{outside_catalog[0]}"
            )


def _validate_url_metadata(base_url: Any, *, label: str) -> None:
    if not isinstance(base_url, str) or not base_url:
        raise ValueError(f"{label}: base_url must be a non-empty string")
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{label}: base_url must be an absolute HTTP URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{label}: base_url must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError(f"{label}: base_url must not contain query or fragment")


def _validate_tool_call_structure(call: Any, *, label: str) -> None:
    if not isinstance(call, dict):
        raise ValueError(f"{label} must be an object")
    _validate_exact_fields(call, {"id", "type", "function"}, label=label)
    if not isinstance(call["id"], str) or not call["id"]:
        raise ValueError(f"{label}.id must be a non-empty string")
    if call["type"] != "function":
        raise ValueError(f"{label}.type must equal function")
    function = call["function"]
    if not isinstance(function, dict):
        raise ValueError(f"{label}.function must be an object")
    _validate_exact_fields(
        function,
        {"name", "arguments"},
        label=f"{label}.function",
    )
    if not isinstance(function["name"], str) or not function["name"]:
        raise ValueError(f"{label}.function.name must be a non-empty string")
    if not isinstance(function["arguments"], (str, dict)):
        raise ValueError(f"{label}.function.arguments must be JSON text or object")


def _validate_terminal_contract(
    content: Any,
    recommended_models: list[str],
    *,
    label: str,
) -> None:
    if not isinstance(content, str):
        raise ValueError(f"{label}: valid terminal content must be a string")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label}: valid terminal content is not JSON") from error
    if not isinstance(parsed, dict):
        raise ValueError(f"{label}: valid terminal content must be an object")
    if set(parsed) != {"answer", "mentioned_models"}:
        raise ValueError(f"{label}: valid terminal content has invalid fields")
    if not isinstance(parsed["answer"], str):
        raise ValueError(f"{label}: terminal answer must be a string")
    mentioned = _validate_string_list(
        parsed["mentioned_models"],
        label=f"{label}: terminal mentioned_models",
    )
    if mentioned != recommended_models:
        raise ValueError(
            f"{label}: recommended_models must equal terminal mentioned_models"
        )
    normalized_answer = _normalize_model_text(parsed["answer"])
    absent_models = [
        model
        for model in mentioned
        if _normalize_model_text(model) not in normalized_answer
    ]
    if absent_models:
        raise ValueError(
            f"{label}: terminal mentioned model is absent from answer: "
            f"{absent_models[0]}"
        )


def _validate_trajectory(
    value: Any,
    *,
    label: str,
    schema_valid: bool,
    runner_errors: list[str],
    raw_assistant_content: str | None,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{label}: trajectory must be a list")
    if len(value) < 2:
        raise ValueError(f"{label}: trajectory must include system and user messages")
    flattened_calls: list[dict[str, Any]] = []
    seen_call_ids: set[str] = set()
    pending_call_ids: set[str] = set()
    terminal_seen = False
    terminal_content: str | None = None
    for index, message in enumerate(value):
        message_label = f"{label}: trajectory[{index}]"
        if not isinstance(message, dict):
            raise ValueError(f"{message_label} must be an object")
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"{message_label}.role is invalid")
        allowed_fields = {"role", "content"}
        if role == "assistant":
            allowed_fields.add("tool_calls")
        elif role == "tool":
            allowed_fields.add("tool_call_id")
        missing = {"role", "content"} - set(message)
        extras = set(message) - allowed_fields
        if role == "tool" and "tool_call_id" not in message:
            missing.add("tool_call_id")
        if missing:
            raise ValueError(
                f"{message_label}: missing field {sorted(missing)[0]}"
            )
        if extras:
            raise ValueError(
                f"{message_label}: unexpected field {sorted(extras)[0]}"
            )
        content = message["content"]
        if role == "assistant":
            if content is not None and not isinstance(content, str):
                raise ValueError(
                    f"{message_label}.content must be a string or null"
                )
            trajectory_calls = message.get("tool_calls", [])
            if not isinstance(trajectory_calls, list):
                raise ValueError(f"{message_label}.tool_calls must be a list")
            for call_index, call in enumerate(trajectory_calls):
                _validate_tool_call_structure(
                    call,
                    label=f"{message_label}.tool_calls[{call_index}]",
                )
        elif not isinstance(content, str):
            raise ValueError(f"{message_label}.content must be a string")
        if role == "tool" and (
            not isinstance(message["tool_call_id"], str)
            or not message["tool_call_id"]
        ):
            raise ValueError(f"{message_label}.tool_call_id must be non-empty")
        if index < 2:
            continue
        if role in {"system", "user"}:
            raise ValueError(
                f"{message_label}.role is not allowed after system and user"
            )
        if terminal_seen:
            raise ValueError(f"{message_label} appears after terminal assistant")
        if role == "assistant":
            if pending_call_ids:
                raise ValueError(
                    f"{message_label}: every tool call must have a result "
                    "before the next assistant"
                )
            trajectory_calls = message.get("tool_calls", [])
            if not trajectory_calls:
                terminal_seen = True
                terminal_content = message["content"]
                continue
            for call in trajectory_calls:
                call_id = call["id"]
                if call_id in seen_call_ids:
                    raise ValueError(
                        f"{message_label}: duplicate assistant tool call id "
                        f"{call_id}"
                    )
                seen_call_ids.add(call_id)
                pending_call_ids.add(call_id)
                flattened_calls.append(call)
            continue
        tool_call_id = message["tool_call_id"]
        if tool_call_id not in pending_call_ids:
            raise ValueError(
                f"{message_label}.tool_call_id does not reference an "
                "unconsumed current call"
            )
        pending_call_ids.remove(tool_call_id)
    if value[0]["role"] != "system" or value[1]["role"] != "user":
        raise ValueError(f"{label}: trajectory must start with system and user")
    if pending_call_ids:
        raise ValueError(f"{label}: trajectory ends with pending tool calls")
    if terminal_seen and terminal_content != raw_assistant_content:
        raise ValueError(
            f"{label}: terminal trajectory content differs from "
            "raw_assistant_content"
        )
    if not terminal_seen and raw_assistant_content is not None:
        raise ValueError(
            f"{label}: raw_assistant_content requires a terminal assistant message"
        )
    if schema_valid and (
        value[-1]["role"] != "assistant"
        or bool(value[-1].get("tool_calls"))
    ):
        raise ValueError(f"{label}: valid trajectory must end with assistant")
    if not schema_valid and value[-1]["role"] != "assistant" and not runner_errors:
        raise ValueError(
            f"{label}: non-terminal failed trajectory requires runner_errors"
        )
    return flattened_calls


def validate_output_record(
    output: dict[str, Any],
    case: dict[str, Any],
    *,
    expected_protocol: str = PROTOCOL_VERSION,
) -> None:
    """Validate a persisted output, including immutable resume metadata."""
    case_id = case["id"]
    label = f"output {case_id}"
    _validate_exact_fields(output, OUTPUT_FIELDS, label=label)
    if output["id"] != case_id:
        raise ValueError(f"{label}: ID mismatch")
    if output["protocol_version"] != expected_protocol:
        raise ValueError(
            f"{label}: protocol_version must equal {expected_protocol}"
        )
    if not isinstance(output["model"], str) or not output["model"]:
        raise ValueError(f"{label}: model must be a non-empty string")
    _validate_url_metadata(output["base_url"], label=label)
    expected_digest = case_digest(case)
    if output["case_digest"] != expected_digest:
        raise ValueError(
            f"{label}: case_digest mismatch: "
            f"expected={expected_digest} actual={output['case_digest']}"
        )
    if not isinstance(output["schema_valid"], bool):
        raise ValueError(f"{label}: schema_valid must be boolean")
    runner_errors = output["runner_errors"]
    if not isinstance(runner_errors, list):
        raise ValueError(f"{label}: runner_errors must be a list")
    for index, error in enumerate(runner_errors):
        if not isinstance(error, str) or not error:
            raise ValueError(
                f"{label}: runner_errors[{index}] must be a non-empty string"
            )
    parse_error = output["terminal_parse_error"]
    if output["schema_valid"]:
        if parse_error is not None:
            raise ValueError(
                f"{label}: terminal_parse_error must be null when schema_valid"
            )
    elif parse_error is not None and (
        not isinstance(parse_error, str) or not parse_error
    ):
        raise ValueError(
            f"{label}: terminal_parse_error must be null or a non-empty string"
        )
    elif parse_error is None and not runner_errors:
        raise ValueError(
            f"{label}: terminal_parse_error or runner_errors must explain "
            "failed record"
        )

    content = output["raw_assistant_content"]
    if content is not None and not isinstance(content, str):
        raise ValueError(
            f"{label}: raw_assistant_content must be a string or null"
        )
    tool_calls = output["tool_calls"]
    if not isinstance(tool_calls, list):
        raise ValueError(f"{label}: tool_calls must be a list")
    call_ids: set[str] = set()
    for index, call in enumerate(tool_calls):
        _validate_tool_call_structure(call, label=f"{label}: tool_calls[{index}]")
        if call["id"] in call_ids:
            raise ValueError(f"{label}: duplicate tool call id {call['id']}")
        call_ids.add(call["id"])

    latency = output["latency_ms"]
    if (
        isinstance(latency, bool)
        or not isinstance(latency, (int, float))
        or not math.isfinite(latency)
        or latency < 0
    ):
        raise ValueError(f"{label}: latency_ms must be a non-negative number")
    recommended = _validate_string_list(
        output["recommended_models"],
        label=f"{label}: recommended_models",
    )
    if not output["schema_valid"] and recommended:
        raise ValueError(
            f"{label}: invalid terminal schema cannot persist recommended_models"
        )
    if output["schema_valid"]:
        _validate_terminal_contract(content, recommended, label=label)
    flattened_calls = _validate_trajectory(
        output["trajectory"],
        label=label,
        schema_valid=output["schema_valid"],
        runner_errors=runner_errors,
        raw_assistant_content=content,
    )
    if flattened_calls != tool_calls:
        raise ValueError(
            f"{label}: flattened trajectory tool_calls differ from output.tool_calls"
        )


def _metric(
    numerator: int,
    denominator: int,
    *,
    scope: str | None = None,
) -> dict[str, int | float | str]:
    ratio = numerator / denominator if denominator else 0.0
    metric: dict[str, int | float | str] = {
        "numerator": numerator,
        "denominator": denominator,
        "ratio": ratio,
        "percentage": ratio * 100.0,
        "zero_denominator_rule": ZERO_DENOMINATOR_RULE,
    }
    if scope is not None:
        metric["scope"] = scope
    return metric


def _schema_map() -> dict[str, dict[str, Any]]:
    return {
        tool["function"]["name"]: tool["function"]["parameters"]
        for tool in build_tool_schemas()
    }


def _openai_call_name(call: Any) -> str | None:
    if not isinstance(call, dict) or call.get("type") != "function":
        return None
    function = call.get("function")
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    return name if isinstance(name, str) and name else None


def _parse_call_arguments(arguments: Any) -> tuple[dict[str, Any] | None, str | None]:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None, "arguments must be valid JSON"
    if not isinstance(arguments, dict):
        return None, "arguments must decode to an object"
    return arguments, None


def _json_type_matches(value: Any, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    return False


def _validate_tool_call(
    call: Any, schemas: dict[str, dict[str, Any]]
) -> tuple[bool, str | None]:
    name = _openai_call_name(call)
    if name is None:
        return False, "expected OpenAI function tool-call object"
    if name not in schemas:
        return False, f"unknown tool: {name}"
    function = call["function"]
    arguments, parse_error = _parse_call_arguments(function["arguments"])
    if parse_error is not None:
        return False, parse_error
    assert arguments is not None

    schema = schemas[name]
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    for field in required:
        if field not in arguments:
            return False, f"missing required argument: {field}"
    if schema.get("additionalProperties") is False:
        extras = sorted(set(arguments) - set(properties))
        if extras:
            return False, f"unexpected argument: {extras[0]}"
    for field, value in arguments.items():
        property_schema = properties.get(field)
        if property_schema is None:
            continue
        expected_type = property_schema.get("type")
        if expected_type and not _json_type_matches(value, expected_type):
            return False, f"argument {field} must be {expected_type}"
    return True, None


def _ordered_mandatory_matches(
    actual_tools: list[str],
    mandatory_tools: list[str],
) -> int:
    matched = 0
    for name in actual_tools:
        if matched < len(mandatory_tools) and name == mandatory_tools[matched]:
            matched += 1
    return matched


def _normalize_model_text(value: str) -> str:
    return re.sub(
        r"\s+",
        "",
        unicodedata.normalize("NFKC", value).casefold(),
    )


def _vehicle_alias_index(
    vehicle_catalog: set[str],
) -> dict[str, tuple[int, str] | None]:
    """Build deterministic full brand-model aliases only."""
    candidates: dict[str, list[tuple[int, str]]] = {}
    for canonical in sorted(vehicle_catalog):
        full_alias = _normalize_model_text(canonical)
        candidates.setdefault(full_alias, []).append((0, canonical))

    aliases: dict[str, tuple[int, str] | None] = {}
    for alias, matches in candidates.items():
        unique_models = {model for _priority, model in matches}
        if len(unique_models) != 1:
            aliases[alias] = None
            continue
        priority = min(priority for priority, _model in matches)
        aliases[alias] = (priority, next(iter(unique_models)))
    return aliases


def _known_models_in_answer(
    answer: str,
    vehicle_catalog: set[str],
) -> list[str]:
    """Match known catalog aliases, preferring full brand-model entities."""
    normalized_answer = _normalize_model_text(answer)
    candidates: list[tuple[int, int, int, str]] = []
    for alias, match in _vehicle_alias_index(vehicle_catalog).items():
        if match is None:
            continue
        priority, canonical = match
        start = 0
        while alias and (
            position := normalized_answer.find(alias, start)
        ) >= 0:
            candidates.append(
                (priority, position, position + len(alias), canonical)
            )
            start = position + 1

    selected: list[tuple[int, int, str]] = []
    for _priority, start, end, model in sorted(
        candidates,
        key=lambda item: (item[0], -(item[2] - item[1]), item[1], item[3]),
    ):
        if any(
            start < used_end and end > used_start
            for used_start, used_end, _model in selected
        ):
            continue
        selected.append((start, end, model))
    return [model for _start, _end, model in sorted(selected)]


def _resolve_declared_model(
    declared_model: str,
    vehicle_catalog: set[str],
) -> str | None:
    match = _vehicle_alias_index(vehicle_catalog).get(
        _normalize_model_text(declared_model)
    )
    return match[1] if match is not None else None


def _terminal_answer(content: Any) -> str | None:
    if not isinstance(content, str):
        return None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or not isinstance(parsed.get("answer"), str):
        return None
    return parsed["answer"]


def evaluate_records(
    cases: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    *,
    vehicle_catalog: set[str],
) -> dict[str, Any]:
    """Validate, exact-join, and compute deterministic auditable metrics."""
    case_by_id = _index_by_id(cases, label="cases")
    output_by_id = _index_by_id(outputs, label="outputs")
    missing = sorted(case_by_id.keys() - output_by_id.keys())
    extra = sorted(output_by_id.keys() - case_by_id.keys())
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing output IDs: {missing}")
        if extra:
            parts.append(f"extra output IDs: {extra}")
        raise ValueError("; ".join(parts))

    validate_case_records(cases, vehicle_catalog=vehicle_catalog)
    for case in cases:
        validate_output_record(output_by_id[case["id"]], case)

    schemas = _schema_map()
    tool_selection_numerator = 0
    argument_validity_numerator = 0
    mandatory_coverage_numerator = 0
    mandatory_coverage_denominator = 0
    recommendation_hit_numerator = 0
    recommendation_hit_denominator = 0
    declared_model_numerator = 0
    declared_model_denominator = 0
    response_contract_numerator = 0
    hallucination_numerator = 0
    hallucination_denominator = 0
    schema_valid_numerator = 0
    protocol_failed_ids: list[str] = []
    case_results: list[dict[str, Any]] = []

    for case in cases:
        case_id = case["id"]
        output = output_by_id[case_id]
        failures: list[str] = []
        runner_errors = output["runner_errors"]
        failures.extend(f"runner_error: {error}" for error in runner_errors)

        answer = _terminal_answer(output["raw_assistant_content"])
        known_models = (
            _known_models_in_answer(answer, vehicle_catalog)
            if answer is not None
            else []
        )
        recommended_models = output["recommended_models"]
        resolved_recommended_models = [
            _resolve_declared_model(model, vehicle_catalog)
            for model in recommended_models
        ]
        declared_known_models = {
            model for model in resolved_recommended_models if model is not None
        }
        undeclared_known_models = [
            model for model in known_models if model not in declared_known_models
        ]
        declared_model_denominator += len(known_models)
        declared_model_numerator += (
            len(known_models) - len(undeclared_known_models)
        )
        contract_consistent = not undeclared_known_models

        terminal_schema_valid = (
            output["schema_valid"] and contract_consistent
        )
        case_success = terminal_schema_valid and not runner_errors
        if terminal_schema_valid:
            schema_valid_numerator += 1
            response_contract_numerator += 1
        if not case_success:
            protocol_failed_ids.append(case_id)
            reason = output["terminal_parse_error"]
            if reason:
                failures.append(f"response_schema_invalid: {reason}")
            elif undeclared_known_models:
                failures.append(
                    "response_schema_invalid: terminal declaration mismatch"
                )
            elif runner_errors:
                failures.append("execution_failed: runner failure")
        if undeclared_known_models:
            failures.append(
                "terminal_protocol_failure: known catalog models in answer "
                f"are undeclared: {undeclared_known_models!r}"
            )

        tool_calls = output["tool_calls"]
        actual_tools = [_openai_call_name(call) or "" for call in tool_calls]
        mandatory = case["expected_tools"]
        allowed = set(mandatory + case["optional_tools"])
        disallowed = sorted({name for name in actual_tools if name not in allowed})
        matched_mandatory = _ordered_mandatory_matches(actual_tools, mandatory)
        mandatory_complete = matched_mandatory == len(mandatory)
        if disallowed:
            failures.append(f"tool_selection: disallowed tools {disallowed}")
        if not mandatory_complete:
            failures.append(
                "tool_selection: mandatory tools are missing or out of order; "
                f"expected={mandatory!r} actual={actual_tools!r}"
            )
        selection_valid = not disallowed and mandatory_complete
        if case_success and selection_valid:
            tool_selection_numerator += 1
        mandatory_coverage_denominator += len(mandatory)
        if case_success:
            mandatory_coverage_numerator += matched_mandatory

        valid_argument_calls = 0
        all_arguments_valid = True
        for index, call in enumerate(tool_calls):
            valid, failure = _validate_tool_call(call, schemas)
            if valid:
                valid_argument_calls += 1
            else:
                all_arguments_valid = False
                failures.append(f"tool_call[{index}]: {failure}")
        if case_success and mandatory_complete and all_arguments_valid:
            argument_validity_numerator += 1

        allowed_models = case["allowed_models"]
        recommendation_hit: bool | None = None
        if allowed_models:
            recommendation_hit_denominator += 1
            recommendation_hit = bool(
                case_success
                and declared_known_models.intersection(allowed_models)
            )
            if recommendation_hit:
                recommendation_hit_numerator += 1
            else:
                failures.append("recommendation_miss")

        hallucinated_models = [
            declared
            for declared, resolved in zip(
                recommended_models, resolved_recommended_models
            )
            if resolved is None
        ]
        hallucination_denominator += len(recommended_models)
        hallucination_numerator += len(hallucinated_models)
        failures.extend(
            f"hallucinated_model: {model}" for model in hallucinated_models
        )

        case_results.append(
            {
                "id": case_id,
                "failures": failures,
                "schema_valid": terminal_schema_valid,
                "case_success": case_success,
                "runner_errors": runner_errors,
                "actual_tools": actual_tools,
                "mandatory_tools_matched": (
                    matched_mandatory if case_success else 0
                ),
                "mandatory_tools_total": len(mandatory),
                "valid_argument_calls": valid_argument_calls,
                "total_tool_calls": len(tool_calls),
                "recommendation_hit": recommendation_hit,
                "hallucinated_models": hallucinated_models,
                "known_models_in_answer": known_models,
                "undeclared_known_models": undeclared_known_models,
                "resolved_declared_models": [
                    model for model in resolved_recommended_models
                    if model is not None
                ],
            }
        )

    total_cases = len(cases)
    return {
        "total_cases": total_cases,
        "protocol_gate": {
            "passed": not protocol_failed_ids,
            "required_protocol_version": PROTOCOL_VERSION,
            "failed_case_ids": protocol_failed_ids,
        },
        "metrics": {
            "tool_selection_accuracy": _metric(
                tool_selection_numerator, total_cases
            ),
            "argument_validity": _metric(
                argument_validity_numerator, total_cases
            ),
            "recommendation_hit_rate": _metric(
                recommendation_hit_numerator, recommendation_hit_denominator
            ),
            "hallucination_rate": _metric(
                hallucination_numerator,
                hallucination_denominator,
                scope=HALLUCINATION_SCOPE,
            ),
            "response_schema_validity": _metric(
                schema_valid_numerator, total_cases
            ),
            "declared_model_coverage": _metric(
                declared_model_numerator,
                declared_model_denominator,
            ),
            "response_contract_consistency": _metric(
                response_contract_numerator,
                total_cases,
            ),
            "mandatory_tool_coverage": _metric(
                mandatory_coverage_numerator,
                mandatory_coverage_denominator,
            ),
        },
        "case_results": case_results,
    }


def evaluate_files(
    cases_path: Path,
    outputs_path: Path,
    *,
    vehicle_database: Path = DEFAULT_VEHICLE_DATABASE,
) -> dict[str, Any]:
    return evaluate_records(
        load_jsonl(cases_path, label="cases"),
        load_jsonl(outputs_path, label="outputs"),
        vehicle_catalog=set(load_vehicle_catalog(vehicle_database)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and evaluate saved Task 14 model outputs."
    )
    parser.add_argument("--cases", type=Path, required=True, help="Evaluation JSONL")
    parser.add_argument(
        "--outputs", type=Path, required=True, help="Saved model output JSONL"
    )
    parser.add_argument(
        "--vehicle-database",
        type=Path,
        default=DEFAULT_VEHICLE_DATABASE,
        help="Structured vehicle catalog CSV",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional JSON report path; stdout is always emitted",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_files(
        args.cases,
        args.outputs,
        vehicle_database=args.vehicle_database,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report is not None:
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
