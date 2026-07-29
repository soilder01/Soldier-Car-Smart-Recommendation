"""Fail-closed validation for synthesized tool-calling training records."""

import json
import unicodedata
from collections.abc import Iterable
from typing import Any

from data_synth.tool_schemas import TOOL_SCHEMAS


_ALLOWED_ROLES = frozenset({"assistant", "system", "tool", "user"})
_ALLOWED_TOOL_CALL_FIELDS = frozenset({"function", "id", "type"})
_ALLOWED_TOOL_FUNCTION_FIELDS = frozenset({"arguments", "name"})
_TOOL_SCHEMAS_BY_NAME = {
    item["function"]["name"]: item["function"]["parameters"]
    for item in TOOL_SCHEMAS
}


def _normalize_for_comparison(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.strip().split()).casefold()


def _normalize_string_iterable(
    values: Any,
    path: str,
    item_label: str,
    errors: list[str],
) -> set[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        errors.append(
            f"{path}: must be an iterable container of strings, "
            "not str or bytes"
        )
        return set()

    items = list(values)
    if any(not isinstance(item, str) for item in items):
        errors.append(f"{path}: all {item_label} must be strings")
        return set()

    normalized_items = {
        _normalize_for_comparison(item) for item in items
    }
    if "" in normalized_items:
        errors.append(
            f"{path}: {item_label} must normalize to non-empty strings"
        )
        return set()
    return normalized_items


def _json_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if type(value) is int:
        return "integer"
    if type(value) is float:
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _matches_json_type(value: Any, expected_type: Any) -> bool:
    if isinstance(expected_type, list):
        return any(_matches_json_type(value, item) for item in expected_type)
    if expected_type == "null":
        return value is None
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return type(value) is int
    if expected_type == "number":
        return type(value) in {int, float}
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, dict)
    return False


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _parse_arguments(
    raw_arguments: Any,
    path: str,
    errors: list[str],
) -> dict[str, Any] | None:
    if isinstance(raw_arguments, dict):
        arguments = raw_arguments
    elif isinstance(raw_arguments, str):
        try:
            arguments = json.loads(
                raw_arguments,
                parse_constant=_reject_nonstandard_json_constant,
            )
        except (json.JSONDecodeError, ValueError):
            errors.append(f"{path}: invalid JSON")
            return None
        if not isinstance(arguments, dict):
            errors.append(f"{path}: JSON value must be an object")
            return None
    else:
        errors.append(f"{path}: must be a JSON object string or object")
        return None

    if any(not isinstance(key, str) for key in arguments):
        errors.append(f"{path}: object keys must be strings")
        return None
    return arguments


def _validate_arguments(
    arguments: dict[str, Any],
    schema: dict[str, Any],
    path: str,
    errors: list[str],
) -> None:
    properties = schema.get("properties", {})

    for property_name in schema.get("required", []):
        if property_name not in arguments:
            errors.append(
                f"{path}.{property_name}: required property is missing"
            )

    if schema.get("additionalProperties") is False:
        for property_name in sorted(arguments.keys() - properties.keys()):
            errors.append(
                f"{path}.{property_name}: additional property is not allowed"
            )

    for property_name in sorted(arguments.keys() & properties.keys()):
        expected_type = properties[property_name].get("type")
        value = arguments[property_name]
        if expected_type is not None and not _matches_json_type(
            value,
            expected_type,
        ):
            errors.append(
                f"{path}.{property_name}: expected {expected_type}, "
                f"got {_json_type_name(value)}"
            )


def _validate_tool_calls(
    tool_calls: Any,
    path: str,
    errors: list[str],
) -> None:
    if not isinstance(tool_calls, list):
        errors.append(f"{path}: must be a list")
        return
    if not tool_calls:
        errors.append(f"{path}: must contain at least one tool call")
        return

    for call_index, call in enumerate(tool_calls):
        call_path = f"{path}[{call_index}]"
        if not isinstance(call, dict):
            errors.append(f"{call_path}: must be an object")
            continue

        for field_name in sorted(
            call.keys() - _ALLOWED_TOOL_CALL_FIELDS,
            key=lambda item: (type(item).__name__, repr(item)),
        ):
            errors.append(
                f"{call_path}.{field_name}: "
                "additional property is not allowed"
            )

        if "id" in call:
            call_id = call["id"]
            if not isinstance(call_id, str) or not call_id.strip():
                errors.append(
                    f"{call_path}.id: must be a non-empty string"
                )

        if "type" in call and call["type"] != "function":
            errors.append(f"{call_path}.type: must be 'function'")

        function = call.get("function")
        function_path = f"{call_path}.function"
        if not isinstance(function, dict):
            errors.append(f"{function_path}: must be an object")
            continue

        for field_name in sorted(
            function.keys() - _ALLOWED_TOOL_FUNCTION_FIELDS,
            key=lambda item: (type(item).__name__, repr(item)),
        ):
            errors.append(
                f"{function_path}.{field_name}: "
                "additional property is not allowed"
            )

        name = function.get("name")
        name_path = f"{function_path}.name"
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{name_path}: must be a non-empty string")
            continue

        schema = _TOOL_SCHEMAS_BY_NAME.get(name)
        if schema is None:
            errors.append(f"{name_path}: unknown tool {name!r}")

        arguments_path = f"{function_path}.arguments"
        arguments = _parse_arguments(
            function.get("arguments"),
            arguments_path,
            errors,
        )
        if arguments is not None and schema is not None:
            _validate_arguments(arguments, schema, arguments_path, errors)


def validate_record(
    record: dict,
    held_out_ids: set[str],
    held_out_queries: Iterable[str] | None = None,
) -> list[str]:
    """Return deterministic validation errors without executing tool calls."""
    errors: list[str] = []
    if not isinstance(record, dict):
        return ["record: must be an object"]

    record_id = record.get("id")
    normalized_record_id: str | None = None
    if not isinstance(record_id, str):
        errors.append("record.id: must be a non-empty string")
    else:
        candidate_record_id = _normalize_for_comparison(record_id)
        if not candidate_record_id:
            errors.append("record.id: must be a non-empty string")
        else:
            normalized_record_id = candidate_record_id

    normalized_held_out_ids = _normalize_string_iterable(
        held_out_ids,
        "held_out_ids",
        "IDs",
        errors,
    )

    if (
        normalized_record_id is not None
        and normalized_record_id in normalized_held_out_ids
    ):
        errors.append(
            f"record.id: held-out ID leakage detected ({record_id!r})"
        )

    normalized_held_out_queries: set[str] = set()
    if held_out_queries is not None:
        normalized_held_out_queries = _normalize_string_iterable(
            held_out_queries,
            "held_out_queries",
            "queries",
            errors,
        )

    messages = record.get("messages")
    if not isinstance(messages, list):
        errors.append("record.messages: must be a list")
        return errors
    if not messages:
        errors.append("record.messages: must contain at least one message")
        return errors

    for message_index, message in enumerate(messages):
        message_path = f"record.messages[{message_index}]"
        if not isinstance(message, dict):
            errors.append(f"{message_path}: must be an object")
            continue

        role = message.get("role")
        if not isinstance(role, str) or role not in _ALLOWED_ROLES:
            errors.append(
                f"{message_path}.role: "
                "must be one of assistant, system, tool, user"
            )

        if role == "user":
            content = message.get("content")
            if not isinstance(content, str):
                errors.append(
                    f"{message_path}.content: user query must be a string"
                )
            elif not content.strip():
                errors.append(
                    f"{message_path}.content: "
                    "user query must be a non-empty string"
                )
            elif (
                normalized_held_out_queries
                and _normalize_for_comparison(content)
                in normalized_held_out_queries
            ):
                errors.append(
                    f"{message_path}.content: "
                    "held-out query leakage detected"
                )
        elif role == "system":
            content = message.get("content")
            if not isinstance(content, str):
                errors.append(
                    f"{message_path}.content: "
                    "system content must be a string"
                )
            elif not content.strip():
                errors.append(
                    f"{message_path}.content: "
                    "system content must be a non-empty string"
                )
        elif role == "tool":
            if not isinstance(message.get("content"), str):
                errors.append(
                    f"{message_path}.content: "
                    "tool content must be a string"
                )
            tool_call_id = message.get("tool_call_id")
            if (
                not isinstance(tool_call_id, str)
                or not tool_call_id.strip()
            ):
                errors.append(
                    f"{message_path}.tool_call_id: "
                    "must be a non-empty string"
                )
        elif role == "assistant":
            tool_calls = message.get("tool_calls")
            has_non_empty_tool_calls = (
                isinstance(tool_calls, list) and bool(tool_calls)
            )
            if has_non_empty_tool_calls:
                content = message.get("content")
                if (
                    "content" in message
                    and content is not None
                    and not isinstance(content, str)
                ):
                    errors.append(
                        f"{message_path}.content: "
                        "assistant content must be a string"
                    )
            elif "tool_calls" not in message:
                if "content" not in message:
                    errors.append(
                        f"{message_path}: assistant message must have "
                        "string content or non-empty tool_calls"
                    )
                elif not isinstance(message["content"], str):
                    errors.append(
                        f"{message_path}.content: "
                        "assistant content must be a string"
                    )
                elif not message["content"].strip():
                    errors.append(
                        f"{message_path}.content: assistant content must "
                        "be a non-empty string when tool_calls are absent"
                    )
            elif "content" in message and not isinstance(
                message["content"],
                str,
            ):
                errors.append(
                    f"{message_path}.content: "
                    "assistant content must be a string"
                )

        if "tool_calls" not in message:
            continue
        tool_calls_path = f"{message_path}.tool_calls"
        if role != "assistant":
            errors.append(
                f"{tool_calls_path}: only allowed on assistant messages"
            )
            continue
        _validate_tool_calls(message["tool_calls"], tool_calls_path, errors)

    return errors
