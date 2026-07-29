#!/usr/bin/env python3
"""Run auditable, multi-turn Task 14 model-layer evaluation."""

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit, urlunsplit

from data_synth.tool_schemas import build_tool_schemas
from scripts.evaluate_model_outputs import (
    DEFAULT_VEHICLE_DATABASE,
    PROTOCOL_VERSION,
    _known_models_in_answer,
    _normalize_model_text,
    _parse_call_arguments,
    _validate_tool_call,
    case_digest,
    load_jsonl,
    load_vehicle_catalog,
    validate_case_records,
    validate_output_record,
)


EVALUATION_TERMINAL_INSTRUCTION = """

评测终态协议（必须遵守）：
- 完成所有工具调用后，最终 assistant 消息必须且只能是一个严格 JSON object，
  不得包含 Markdown 代码块、解释、前后缀或其他文本。
- object 必须精确包含两个字段：
  {"answer": "<最终回答字符串>", "mentioned_models": ["<车型名>", "..."]}
- answer 必须是 string；mentioned_models 必须是 list[string]。
- mentioned_models 必须按 answer 中的出现顺序列出 answer 提及的每一个车型，
  包括不在本地车型库中的未知车型；不要遗漏、过滤、规范化或补造车型。
""".rstrip()


def create_openai_client(*, base_url: str, api_key: str) -> Any:
    """Construct the client lazily so ``--help`` does not import the SDK."""
    from openai import OpenAI

    return OpenAI(base_url=base_url, api_key=api_key)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _sanitize_base_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("base_url must be an absolute HTTP URL")
    hostname = parsed.hostname
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _index_ids(
    records: list[dict[str, Any]],
    *,
    label: str,
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


def _load_agent_contract(
    intent: str,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Load the production prompt, schemas, and StructuredTools lazily."""
    from app.services import agent_graph

    if intent not in agent_graph.TOOLS_BY_INTENT:
        raise ValueError(f"unsupported intent: {intent}")
    approved_schemas = build_tool_schemas()
    approved_names = [
        schema["function"]["name"] for schema in approved_schemas
    ]
    real_names = [tool.name for tool in agent_graph.TOOLS]
    if (
        len(approved_names) != len(set(approved_names))
        or set(approved_names) != set(real_names)
    ):
        raise AssertionError(
            "approved tool names must equal the production tool registry"
        )
    approved_by_name = {
        schema["function"]["name"]: schema for schema in approved_schemas
    }
    tools_for_intent = agent_graph.TOOLS_BY_INTENT[intent]
    intent_names = [tool.name for tool in tools_for_intent]
    if not set(intent_names) <= set(approved_names):
        raise AssertionError(
            f"approved schemas do not cover production intent {intent}"
        )
    prompt = agent_graph._get_prompt_for_intent(intent)
    schemas = [copy.deepcopy(approved_by_name[name]) for name in intent_names]
    if any(
        schema["function"]["parameters"].get("additionalProperties") is not False
        for schema in schemas
    ):
        raise AssertionError("approved tool schemas must forbid additional properties")
    tools = {tool.name: tool for tool in tools_for_intent}
    return prompt, schemas, tools


def _normalize_tool_calls(
    raw_tool_calls: Any,
    *,
    seen_call_ids: set[str],
) -> list[dict[str, Any]]:
    if raw_tool_calls is None:
        return []
    if not isinstance(raw_tool_calls, (list, tuple)):
        raise ValueError("assistant tool_calls must be a list")

    normalized: list[dict[str, Any]] = []
    staged_call_ids: set[str] = set()
    for index, call in enumerate(raw_tool_calls):
        call_id = _field(call, "id")
        function = _field(call, "function")
        name = _field(function, "name")
        arguments = _field(function, "arguments")
        call_type = _field(call, "type")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError(
                f"assistant tool_calls[{index}] requires a non-empty tool call id"
            )
        if call_id in seen_call_ids or call_id in staged_call_ids:
            raise ValueError(f"duplicate assistant tool call id: {call_id}")
        if call_type != "function":
            raise ValueError(
                f"assistant tool_calls[{index}] type must equal function"
            )
        if function is None:
            raise ValueError(
                f"assistant tool_calls[{index}] requires a function object"
            )
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"assistant tool_calls[{index}] requires a non-empty function name"
            )
        if not isinstance(arguments, (str, dict)):
            raise ValueError(
                f"assistant tool_calls[{index}] arguments must be JSON text or object"
            )
        staged_call_ids.add(call_id)
        normalized.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }
        )
    seen_call_ids.update(staged_call_ids)
    return normalized


def _response_message(response: Any) -> Any:
    choices = _field(response, "choices")
    if not isinstance(choices, (list, tuple)) or not choices:
        raise ValueError("OpenAI-compatible response has no choices")
    message = _field(choices[0], "message")
    if message is None:
        raise ValueError("OpenAI-compatible response choice has no message")
    return message


def _serialize_tool_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _invoke_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    registry: Mapping[str, Any],
    tool_executor: Callable[[str, dict[str, Any]], Any] | None,
) -> Any:
    if tool_executor is not None:
        return tool_executor(name, arguments)
    tool = registry.get(name)
    if tool is None:
        raise ValueError(f"tool registry has no implementation for {name}")
    invoke = getattr(tool, "invoke", None)
    if callable(invoke):
        return invoke(arguments)
    if callable(tool):
        return tool(arguments)
    raise ValueError(f"tool registry entry {name} is not invokable")


def _parse_terminal_content(
    content: str | None,
    *,
    vehicle_catalog: set[str],
) -> tuple[bool, str | None, list[str]]:
    prefix = "protocol failure:"
    if not isinstance(content, str):
        return False, f"{prefix} terminal assistant content must be a string", []
    try:
        terminal = json.loads(content)
    except json.JSONDecodeError as error:
        return (
            False,
            f"{prefix} terminal assistant content is not strict JSON: {error.msg}",
            [],
        )
    if not isinstance(terminal, dict):
        return False, f"{prefix} terminal response must be a JSON object", []
    if set(terminal) != {"answer", "mentioned_models"}:
        return (
            False,
            f"{prefix} terminal object fields must be answer and mentioned_models",
            [],
        )
    if not isinstance(terminal["answer"], str):
        return False, f"{prefix} answer must be a string", []
    mentioned = terminal["mentioned_models"]
    if not isinstance(mentioned, list):
        return False, f"{prefix} mentioned_models must be a list", []
    for index, model in enumerate(mentioned):
        if not isinstance(model, str) or not model.strip():
            return (
                False,
                f"{prefix} mentioned_models[{index}] must be a non-empty string",
                [],
            )
    if len(mentioned) != len(set(mentioned)):
        return (
            False,
            f"{prefix} mentioned_models must not contain duplicates",
            [],
        )
    normalized_answer = _normalize_model_text(terminal["answer"])
    absent_models = [
        model
        for model in mentioned
        if _normalize_model_text(model) not in normalized_answer
    ]
    if absent_models:
        return (
            False,
            f"{prefix} declared models are absent from answer: "
            f"{absent_models!r}",
            [],
        )
    known_models = _known_models_in_answer(
        terminal["answer"],
        vehicle_catalog,
    )
    normalized_mentions = {
        _normalize_model_text(model) for model in mentioned
    }
    undeclared = [
        model
        for model in known_models
        if _normalize_model_text(model) not in normalized_mentions
    ]
    if undeclared:
        return (
            False,
            f"{prefix} known catalog models in answer are undeclared: "
            f"{undeclared!r}",
            [],
        )
    return True, None, mentioned


def _run_case(
    case: dict[str, Any],
    *,
    client: Any,
    model: str,
    persisted_base_url: str,
    clock: Callable[[], float],
    max_steps: int,
    vehicle_catalog: set[str],
    tool_registry: Mapping[str, Any] | None,
    tool_executor: Callable[[str, dict[str, Any]], Any] | None,
    initial_runner_error: str | None = None,
    system_prompt: str | None = None,
    append_evaluation_terminal_instruction: bool = True,
) -> dict[str, Any]:
    intent = case["intent"]
    real_prompt, schemas, real_registry = _load_agent_contract(intent)
    allowed_names = {
        schema["function"]["name"] for schema in schemas
    }
    registry = tool_registry if tool_registry is not None else real_registry
    selected_prompt = system_prompt if system_prompt is not None else real_prompt
    if append_evaluation_terminal_instruction:
        selected_prompt = (
            f"{selected_prompt}\n{EVALUATION_TERMINAL_INSTRUCTION}"
        )
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": selected_prompt,
        },
        {"role": "user", "content": case["query"]},
    ]
    trajectory = copy.deepcopy(messages)
    aggregate_tool_calls: list[dict[str, Any]] = []
    seen_call_ids: set[str] = set()
    terminal_content: str | None = None
    schema_valid = False
    terminal_parse_error: str | None = (
        f"protocol failure: max_steps={max_steps} reached before terminal response"
    )
    recommended_models: list[str] = []
    runner_errors: list[str] = []
    terminal_reached = False
    ended_early = False
    approved_schema_map = {
        schema["function"]["name"]: schema["function"]["parameters"]
        for schema in build_tool_schemas()
    }

    try:
        started = clock()
    except Exception as error:
        started = 0.0
        runner_errors.append(f"clock_start_error: {type(error).__name__}")
        ended_early = True
    if initial_runner_error is not None:
        runner_errors.append(initial_runner_error)
        ended_early = True

    for _step in range(max_steps):
        if ended_early:
            break
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=schemas,
                tool_choice="auto",
                temperature=0,
            )
        except Exception as error:
            runner_errors.append(f"api_error: {type(error).__name__}")
            terminal_parse_error = None
            ended_early = True
            break
        try:
            message = _response_message(response)
            content = _field(message, "content")
            if content is not None and not isinstance(content, str):
                raise ValueError("assistant content must be a string or null")
            tool_calls = _normalize_tool_calls(
                _field(message, "tool_calls", []),
                seen_call_ids=seen_call_ids,
            )
        except Exception as error:
            runner_errors.append(
                f"tool_call_structure_error: {error}"
                if isinstance(error, ValueError)
                else f"tool_call_structure_error: {type(error).__name__}"
            )
            terminal_parse_error = None
            ended_early = True
            break
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": content,
        }
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
        messages.append(assistant_message)
        trajectory.append(copy.deepcopy(assistant_message))

        if not tool_calls:
            terminal_content = content
            (
                schema_valid,
                terminal_parse_error,
                recommended_models,
            ) = _parse_terminal_content(
                content,
                vehicle_catalog=vehicle_catalog,
            )
            terminal_reached = True
            break

        aggregate_tool_calls.extend(copy.deepcopy(tool_calls))
        for call in tool_calls:
            name = call["function"]["name"]
            call_id = call["id"]
            failure: str | None = None
            if name not in allowed_names:
                failure = f"tool {name} is not allowed for intent {intent}"
            else:
                valid, failure = _validate_tool_call(
                    call,
                    approved_schema_map,
                )
                if valid:
                    failure = None
            if failure is not None:
                runner_errors.append(f"tool_call_error[{call_id}]: {failure}")
                result = {
                    "error": {
                        "type": "invalid_tool_call",
                        "message": failure,
                    }
                }
            else:
                arguments, parse_error = _parse_call_arguments(
                    call["function"]["arguments"]
                )
                if parse_error is not None or arguments is None:
                    raise AssertionError(
                        "approved argument validation and parsing diverged"
                    )
                try:
                    result = _invoke_tool(
                        name,
                        arguments,
                        registry=registry,
                        tool_executor=tool_executor,
                    )
                except Exception as error:
                    runner_errors.append(
                        f"tool_execution_error[{call_id}]: "
                        f"{type(error).__name__}"
                    )
                    result = {
                        "error": {
                            "type": "tool_execution_failed",
                            "message": (
                                "tool execution failed; result unavailable"
                            ),
                        }
                    }
            try:
                serialized_result = _serialize_tool_result(result)
            except Exception as error:
                runner_errors.append(
                    f"tool_result_serialization_error[{call_id}]: "
                    f"{type(error).__name__}"
                )
                serialized_result = _serialize_tool_result(
                    {
                        "error": {
                            "type": "tool_result_serialization_failed",
                            "message": (
                                "tool result could not be serialized"
                            ),
                        }
                    }
                )
            tool_message = {
                "role": "tool",
                "content": serialized_result,
                "tool_call_id": call_id,
            }
            messages.append(tool_message)
            trajectory.append(copy.deepcopy(tool_message))

    if not terminal_reached and not ended_early:
        runner_errors.append(f"max_steps_exceeded: max_steps={max_steps}")

    try:
        elapsed = clock() - started
    except Exception as error:
        runner_errors.append(f"clock_end_error: {type(error).__name__}")
        elapsed = 0.0
    latency_ms = max(0.0, elapsed * 1000.0)
    if not math.isfinite(latency_ms):
        runner_errors.append("clock_end_error: non-finite latency")
        latency_ms = 0.0
    return {
        "id": case["id"],
        "protocol_version": PROTOCOL_VERSION,
        "model": model,
        "base_url": persisted_base_url,
        "case_digest": case_digest(case),
        "schema_valid": schema_valid,
        "terminal_parse_error": terminal_parse_error,
        "runner_errors": runner_errors,
        "raw_assistant_content": terminal_content,
        "tool_calls": aggregate_tool_calls,
        "latency_ms": latency_ms,
        "recommended_models": recommended_models if schema_valid else [],
        "trajectory": trajectory,
    }


def _atomic_write_jsonl(
    output_path: Path,
    records: list[dict[str, Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output_file:
            for record in records:
                output_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, output_path)
        try:
            directory_fd = os.open(output_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def run_evaluation(
    cases_path: Path,
    output_path: Path,
    *,
    model: str,
    base_url: str = "http://127.0.0.1:8000/v1",
    api_key: str = "dummy",
    client: Any = None,
    client_factory: Callable[..., Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    vehicle_database: Path = DEFAULT_VEHICLE_DATABASE,
    tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
    tool_registry: Mapping[str, Any] | None = None,
    max_steps: int = 8,
    system_prompts_by_intent: Mapping[str, str] | None = None,
    append_evaluation_terminal_instruction: bool = True,
) -> dict[str, int]:
    """Run missing cases and atomically replace an auditable JSONL snapshot."""
    cases_path = Path(cases_path)
    output_path = Path(output_path)
    if cases_path.resolve() == output_path.resolve():
        raise ValueError("cases and output paths must differ")
    if not isinstance(model, str) or not model:
        raise ValueError("model must be a non-empty string")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    if not isinstance(append_evaluation_terminal_instruction, bool):
        raise ValueError(
            "append_evaluation_terminal_instruction must be boolean"
        )
    if (
        not append_evaluation_terminal_instruction
        and system_prompts_by_intent is None
    ):
        raise ValueError(
            "an explicit frozen prompt mapping is required when the evaluator "
            "terminal instruction is disabled"
        )

    persisted_base_url = _sanitize_base_url(base_url)
    source_hash_before = _source_hash(cases_path)
    cases = load_jsonl(cases_path, label="cases")
    catalog = set(load_vehicle_catalog(vehicle_database))
    case_by_id = _index_ids(cases, label="cases")
    validate_case_records(cases, vehicle_catalog=catalog)
    case_intents = sorted({case["intent"] for case in cases})
    for intent in case_intents:
        _load_agent_contract(intent)
    if system_prompts_by_intent is not None:
        missing_prompts = [
            intent for intent in case_intents
            if intent not in system_prompts_by_intent
        ]
        if missing_prompts:
            raise ValueError(
                f"frozen prompt mapping is missing intents: {missing_prompts}"
            )
        for intent in case_intents:
            prompt = system_prompts_by_intent[intent]
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"frozen prompt for {intent} must be a non-empty string"
                )

    if output_path.exists():
        existing_records = load_jsonl(output_path, label="outputs")
    else:
        existing_records = []
    existing_by_id = _index_ids(existing_records, label="outputs")
    extra_ids = sorted(existing_by_id.keys() - case_by_id.keys())
    if extra_ids:
        raise ValueError(f"existing output has extra IDs: {extra_ids}")

    for record in existing_records:
        case = case_by_id[record["id"]]
        validate_output_record(record, case)
        if record["model"] != model:
            raise ValueError(
                f"existing output {record['id']}: model mismatch: "
                f"expected={model} actual={record['model']}"
            )
        if record["base_url"] != persisted_base_url:
            raise ValueError(
                f"existing output {record['id']}: base_url mismatch: "
                f"expected={persisted_base_url} actual={record['base_url']}"
            )

    pending = [case for case in cases if case["id"] not in existing_by_id]
    client_initialization_error: str | None = None
    if pending and client is None:
        factory = client_factory or create_openai_client
        try:
            client = factory(base_url=base_url, api_key=api_key)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as error:
            client_initialization_error = (
                f"api_client_error: {type(error).__name__}"
            )

    snapshot = list(existing_records)
    written = 0
    for case in pending:
        record = _run_case(
            case,
            client=client,
            model=model,
            persisted_base_url=persisted_base_url,
            clock=clock,
            max_steps=max_steps,
            vehicle_catalog=catalog,
            tool_registry=tool_registry,
            tool_executor=tool_executor,
            initial_runner_error=client_initialization_error,
            system_prompt=(
                system_prompts_by_intent[case["intent"]]
                if system_prompts_by_intent is not None
                else None
            ),
            append_evaluation_terminal_instruction=(
                append_evaluation_terminal_instruction
            ),
        )
        validate_output_record(record, case)
        if _source_hash(cases_path) != source_hash_before:
            raise ValueError("source cases changed during evaluation")
        snapshot.append(record)
        _atomic_write_jsonl(output_path, snapshot)
        if _source_hash(cases_path) != source_hash_before:
            raise ValueError("source cases changed during evaluation")
        written += 1

    if _source_hash(cases_path) != source_hash_before:
        raise ValueError("source cases changed during evaluation")
    return {
        "total_cases": len(cases),
        "existing": len(existing_records),
        "written": written,
        "failed_cases": sum(
            not record["schema_valid"] or bool(record["runner_errors"])
            for record in snapshot
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Task 14 evaluation set through the production intent prompts "
            "and multi-turn tool protocol."
        )
    )
    parser.add_argument("--cases", type=Path, required=True, help="Source case JSONL")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Atomic output snapshot JSONL",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("CHAT_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("CHAT_API_KEY", "dummy"),
        help="Endpoint API key (never persisted)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("CHAT_MODEL", "qwen7b-nev"),
        help="Model name exposed by the endpoint",
    )
    parser.add_argument(
        "--vehicle-database",
        type=Path,
        default=DEFAULT_VEHICLE_DATABASE,
        help="Structured finite vehicle catalog CSV",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=8,
        help="Maximum assistant turns per case",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_evaluation(
        args.cases,
        args.output,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        vehicle_database=args.vehicle_database,
        max_steps=args.max_steps,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
