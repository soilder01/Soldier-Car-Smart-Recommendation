"""Teacher configuration gate for future SFT data synthesis."""

import json
import os
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Mapping

from data_synth.tool_schemas import build_tool_schemas
from data_synth.validate_tool_data import validate_record


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "model_training"
DEFAULT_ARK_BASE_URL = os.environ.get("ARK_BASE_URL", "<ARK_BASE_URL_PLACEHOLDER>")
PROVIDERS = (
    (
        "chat",
        ("CHAT_BASE_URL", "CHAT_MODEL", "CHAT_API_KEY"),
    ),
    (
        "ark",
        ("ARK_BASE_URL", "ARK_CHAT_MODEL", "ARK_API_KEY"),
    ),
    (
        "openai",
        ("OPENAI_BASE_URL", "OPENAI_CHAT_MODEL", "OPENAI_API_KEY"),
    ),
)

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
    "customer_service": ["retrieve_knowledge_base"],
    "deep_search": [
        "extract_user_profile",
        "search_and_rank_vehicles",
        "retrieve_knowledge_base",
        "search_web_info",
    ],
}
OPTIONAL_BY_INTENT = {
    "recommend": ["retrieve_knowledge_base", "search_web_info"],
    "compare": ["search_web_info"],
    "knowledge": ["search_web_info"],
    "sales": [],
    "customer_service": ["search_web_info"],
    "deep_search": ["generate_sales_talk"],
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
    "customer_service": [
        "extract_user_profile",
        "search_and_rank_vehicles",
        "generate_sales_talk",
    ],
    "deep_search": [],
}
GROUNDING_INSTRUCTION = """

接地与结构约束（必须遵守）：
- 最终回答只能陈述用户输入或真实工具结果中已经出现的事实。
- 优先使用 search_and_rank_vehicles 返回的 specs 字段给出具体建议。
- 答案中的价格、续航、电池、快充、轴距、后备箱、座位数、评分等带单位规格，
  必须能在工具结果中逐一反查到同一数值；工具未返回的规格不得补写。
- 不得对工具返回的数值做四舍五入、约数化或区间粗略改写，例如不要把
  “12.98-16.98万”写成“13-17万”。
- 工具没有返回油耗/电耗/保修政策时，不要编造；如用户关注该维度，只说明
  “该项工具未返回，需以官方实时信息核验”。
- 如工具返回 energy_evidence，可复述其中原文；不要把“省油/低油耗”等原文
  扩写成“远低于燃油车”“能耗成本极低”等未返回的比较结论。
- 严禁编造任何硬指标；严禁给编造内容挂 [1][2] 等引用。
- 引用只能用于支持工具结果中确实出现的内容；如果使用 [1][2] 来源列表，
  来源行必须逐字来自工具结果里的 source/title/url/content，不得自行概括成新标题。
- 不得编造《来源标题》；如果工具结果没有 source 原文，就不要写引用标题或来源列表。
- 如果工具结果包含与用户硬约束冲突的候选（例如用户明确要插混却返回纯电），
  不要把冲突候选列入推荐表，只能说明其因不匹配被过滤。
- 能源口径：增程算作广义插电范畴，可作为插混需求的备选并明确说明；
  如用户严格要求 PHEV，则只保留插混车型。
- 如果 search_and_rank_vehicles 返回不少于 3 款合规候选，最终回答必须保留至少
  3 款候选做对比，不能塌缩成单车推荐。
""".rstrip()
_HARD_CLAIM_PATTERN = re.compile(
    r"(?P<numbers>\d+(?:\.\d+)?\s*(?:[-~—–至到]\s*\d+(?:\.\d+)?)?)\s*"
    r"(?P<unit>L/100km|l/100km|升/百公里|kWh|KWh|度|万元|万|公里|km|"
    r"毫米|mm|分钟|min|座|升|L|%|分)"
)
_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")
_QUOTED_SOURCE_PATTERN = re.compile(r"《([^》]+)》")
_BRACKET_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_BRACKET_SOURCE_LINE_PATTERN = re.compile(r"(?m)^\s*\[(\d+)\]\s*([^\n]+)")
_POLICY_CLAIM_PATTERN = re.compile(
    r"(?=[^。；;\n]*(?:保修|质保|权益|政策))"
    r"(?=[^。；;\n]*(?:终身|免费|赠送|承诺|包含|提供|可获得|可享|首任|"
    r"三电|不限|\d+\s*(?:年|万公里|公里)))"
    r"[^。；;\n]*"
)
_UNIT_ALIASES = {
    "万元": "万",
    "万": "万",
    "公里": "km",
    "km": "km",
    "毫米": "mm",
    "mm": "mm",
    "kwh": "kWh",
    "度": "kWh",
    "分钟": "分钟",
    "min": "分钟",
    "座": "座",
    "升": "L",
    "l": "L",
    "l/100km": "L/100km",
    "升/百公里": "L/100km",
    "%": "%",
    "分": "分",
}


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _compact_text(value: str) -> str:
    normalized = value.casefold()
    return re.sub(r"[\s,，。；;：:\[\]（）()、+]+", "", normalized)


def _numbers(value: str) -> set[str]:
    return set(_NUMBER_PATTERN.findall(value))


def _normalize_number(value: str) -> str:
    try:
        decimal = Decimal(value)
    except InvalidOperation:
        return value
    normalized = format(decimal.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def _canonical_unit(value: str) -> str:
    return _UNIT_ALIASES.get(value.strip().casefold(), value.strip())


def _claim_key(match: re.Match[str]) -> tuple[tuple[str, ...], str]:
    return (
        tuple(_normalize_number(number) for number in _numbers(match.group("numbers"))),
        _canonical_unit(match.group("unit")),
    )


def _numeric_claim_keys(value: str) -> set[tuple[tuple[str, ...], str]]:
    return {_claim_key(match) for match in _HARD_CLAIM_PATTERN.finditer(value)}


def _is_deferred_policy_claim(value: str) -> bool:
    compact = _compact_text(value)
    return (
        ("官方" in compact or "品牌" in compact)
        and any(marker in compact for marker in ("为准", "核验", "确认", "咨询", "公布"))
    )


def _read_dotenv(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _environment(dotenv_path: Path | None) -> dict[str, str]:
    values = _read_dotenv(dotenv_path)
    for key, value in os.environ.items():
        if value.strip():
            values[key] = value
    if (
        values.get("ARK_API_KEY", "").strip()
        and values.get("SEEDPRO_EP", "").strip()
    ):
        values.setdefault("ARK_BASE_URL", DEFAULT_ARK_BASE_URL)
        values.setdefault("ARK_CHAT_MODEL", values["SEEDPRO_EP"])
    return values


def _configured(values: dict[str, str], variable: str) -> bool:
    return bool(values.get(variable, "").strip())


def _teacher_status(
    *,
    available: bool,
    provider: str | None,
    configured_fields: tuple[bool, bool, bool],
) -> dict[str, Any]:
    return {
        "available": available,
        "status": "config_ready" if available else "blocked",
        "provider": provider,
        "base_url_configured": configured_fields[0],
        "model_configured": configured_fields[1],
        "api_key_configured": configured_fields[2],
        "endpoint_verified": False,
        "note": (
            "Teacher configuration is complete; endpoint connectivity has "
            "not been verified."
            if available
            else (
                "A complete CHAT, ARK, or legacy OPENAI teacher "
                "configuration is required."
            )
        ),
    }


def check_teacher_available(
    dotenv_path: Path | None = None,
) -> dict[str, Any]:
    """Return a secret-free, same-provider teacher configuration status."""
    values = _environment(dotenv_path)
    partial_fields = (False, False, False)
    for provider, variables in PROVIDERS:
        configured_fields = tuple(
            _configured(values, name) for name in variables
        )
        if all(configured_fields):
            return _teacher_status(
                available=True,
                provider=provider,
                configured_fields=configured_fields,
            )
        if not any(partial_fields) and any(configured_fields):
            partial_fields = configured_fields
    return _teacher_status(
        available=False,
        provider=None,
        configured_fields=partial_fields,
    )


def _approved_schema_map() -> dict[str, dict[str, Any]]:
    return {
        schema["function"]["name"]: schema["function"]["parameters"]
        for schema in build_tool_schemas()
    }


def _schemas_for_intent(intent: str) -> list[dict[str, Any]]:
    allowed = set(MANDATORY_BY_INTENT[intent] + OPTIONAL_BY_INTENT[intent])
    return [
        schema for schema in build_tool_schemas()
        if schema["function"]["name"] in allowed
    ]


def _tool_names_from_record(record: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for message in record.get("messages", []):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            function = call.get("function") if isinstance(call, dict) else None
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                names.append(function["name"])
    return names


def _tool_calls_with_arguments(record: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for message in record.get("messages", []):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            function = call.get("function") if isinstance(call, dict) else None
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            calls.append(
                {
                    "id": call.get("id") if isinstance(call, dict) else None,
                    "name": function.get("name"),
                    "arguments": arguments if isinstance(arguments, dict) else {},
                }
            )
    return calls


def _final_answer_from_record(record: dict[str, Any]) -> str:
    final_answer = ""
    for message in record.get("messages", []):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip() and not message.get("tool_calls"):
            final_answer = content
    return final_answer


def _compact_model_name(value: str) -> str:
    return re.sub(r"[\s,，。；;：:（）()、]+", "", value).casefold()


def _audit_compare_named_lookup(record: dict[str, Any]) -> list[str]:
    search_calls = [
        call for call in _tool_calls_with_arguments(record)
        if call["name"] == "search_and_rank_vehicles"
    ]
    if not search_calls:
        return ["compare named lookup is missing search_and_rank_vehicles"]
    lookup_call = search_calls[0]
    model_names = lookup_call["arguments"].get("model_names")
    if (
        not isinstance(model_names, list)
        or len(model_names) < 2
        or not all(isinstance(name, str) and name.strip() for name in model_names)
    ):
        return [
            "compare first search_and_rank_vehicles call must include "
            "two non-empty model_names"
        ]

    tool_contents = {
        message.get("tool_call_id"): message.get("content")
        for message in record.get("messages", [])
        if isinstance(message, dict) and message.get("role") == "tool"
    }
    content = tool_contents.get(lookup_call["id"])
    if not isinstance(content, str):
        return ["compare named lookup tool result is missing"]
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        return ["compare named lookup tool result is not JSON"]
    if not isinstance(result, dict):
        return ["compare named lookup tool result must be an object"]
    lookup = result.get("named_vehicle_lookup")
    if not isinstance(lookup, dict):
        return ["compare named_vehicle_lookup metadata is missing"]

    named_vehicle_missing = lookup.get("named_vehicle_missing")
    missing_names = lookup.get("missing_model_names")
    named_vehicles = result.get("named_vehicles")
    supplemental_vehicles = result.get("supplemental_vehicles")
    if not isinstance(named_vehicle_missing, bool):
        return ["compare named_vehicle_missing must be boolean"]
    if not isinstance(missing_names, list):
        return ["compare missing_model_names must be a list"]
    if not isinstance(named_vehicles, list):
        return ["compare named_vehicles must be a list"]
    if not isinstance(supplemental_vehicles, list):
        return ["compare supplemental_vehicles must be a list"]

    final_answer = _final_answer_from_record(record)
    if named_vehicle_missing:
        if not missing_names:
            return ["compare named_vehicle_missing=true requires missing_model_names"]
        if "库中无此车规格" not in final_answer:
            return ["compare missing vehicle answer must state 库中无此车规格"]
        if not supplemental_vehicles:
            return ["compare missing vehicle lookup requires supplemental_vehicles"]
        return []

    if missing_names:
        return ["compare named_vehicle_missing=false cannot include missing_model_names"]
    if len(named_vehicles) != len(model_names):
        return ["compare named lookup did not return every requested vehicle"]
    resolved_names = [
        item.get("full_name")
        for item in named_vehicles
        if isinstance(item, dict) and isinstance(item.get("full_name"), str)
    ]
    if len(resolved_names) != len(model_names):
        return ["compare named vehicles must include full_name"]
    if any(
        not isinstance(item, dict) or not isinstance(item.get("specs"), dict)
        for item in named_vehicles
    ):
        return ["compare named vehicles must include specs"]
    compact_answer = _compact_model_name(final_answer)
    if any(_compact_model_name(name) not in compact_answer for name in resolved_names):
        return ["compare final answer must include every resolved named vehicle"]
    if "|" not in final_answer:
        return ["compare final answer must contain a parallel comparison table"]
    return []


def _ordered_matches(actual: list[str], mandatory: list[str]) -> int:
    matched = 0
    for name in actual:
        if matched < len(mandatory) and name == mandatory[matched]:
            matched += 1
    return matched


def audit_teacher_decision_record(intent: str, record: dict[str, Any]) -> list[str]:
    """Return quality errors for teacher-decided tool trajectories."""
    if intent not in MANDATORY_BY_INTENT:
        return [f"unsupported intent: {intent}"]
    actual = _tool_names_from_record(record)
    allowed = set(MANDATORY_BY_INTENT[intent] + OPTIONAL_BY_INTENT[intent])
    unknown_or_forbidden = sorted(
        name for name in actual
        if name not in allowed or name in FORBIDDEN_BY_INTENT[intent]
    )
    errors: list[str] = []
    if unknown_or_forbidden:
        errors.append(f"disallowed tools: {unknown_or_forbidden}")
    mandatory = MANDATORY_BY_INTENT[intent]
    if _ordered_matches(actual, mandatory) != len(mandatory):
        errors.append(
            "mandatory tools missing or out of order: "
            f"expected {mandatory!r} actual {actual!r}"
        )
    if intent == "compare":
        errors.extend(_audit_compare_named_lookup(record))
    return errors


def audit_answer_grounding(record: dict[str, Any]) -> list[str]:
    """Reject final hard metrics that cannot be found in user/tool evidence."""
    messages = record.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return ["record.messages: must be a non-empty list"]
    final_answer = None
    evidence_parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role in {"user", "tool"} and isinstance(content, str):
            evidence_parts.append(content)
        if role == "assistant" and isinstance(content, str) and content.strip():
            final_answer = content
    if final_answer is None:
        return ["final assistant answer is missing"]

    evidence_text = "\n".join(evidence_parts)
    compact_evidence = _compact_text(evidence_text)
    evidence_claim_keys = _numeric_claim_keys(evidence_text)
    errors: list[str] = []

    seen_claims: set[str] = set()
    for match in _HARD_CLAIM_PATTERN.finditer(final_answer):
        claim = match.group(0).strip()
        compact_claim = _compact_text(claim)
        if compact_claim in seen_claims:
            continue
        seen_claims.add(compact_claim)
        if (
            compact_claim not in compact_evidence
            and _claim_key(match) not in evidence_claim_keys
        ):
            errors.append(f"unsupported hard claim: {claim}")

    seen_sources: set[str] = set()
    for match in _QUOTED_SOURCE_PATTERN.finditer(final_answer):
        source = match.group(1).strip()
        compact_source = _compact_text(source)
        if compact_source in seen_sources:
            continue
        seen_sources.add(compact_source)
        if compact_source not in compact_evidence:
            errors.append(f"unsupported quoted source: {source}")

    bracket_source_numbers: set[str] = set()
    for match in _BRACKET_SOURCE_LINE_PATTERN.finditer(final_answer):
        number = match.group(1)
        source_line = match.group(2).strip()
        bracket_source_numbers.add(number)
        compact_source_line = _compact_text(source_line)
        if compact_source_line and compact_source_line not in compact_evidence:
            errors.append(
                f"unsupported bracket source: [{number}] {source_line}"
            )

    for number in sorted(set(_BRACKET_CITATION_PATTERN.findall(final_answer))):
        if number not in bracket_source_numbers:
            errors.append(f"unsupported bracket citation: [{number}]")

    seen_policy_claims: set[str] = set()
    for match in _POLICY_CLAIM_PATTERN.finditer(final_answer):
        claim = match.group(0).strip()
        compact_claim = _compact_text(claim)
        if compact_claim in seen_policy_claims:
            continue
        seen_policy_claims.add(compact_claim)
        if (
            compact_claim not in compact_evidence
            and not _is_deferred_policy_claim(claim)
        ):
            errors.append(f"unsupported policy claim: {claim}")
    return errors


def _normalize_teacher_tool_calls(raw_tool_calls: Any) -> list[dict[str, Any]]:
    if raw_tool_calls is None:
        return []
    if not isinstance(raw_tool_calls, (list, tuple)):
        raise ValueError("teacher tool_calls must be a list")
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, call in enumerate(raw_tool_calls):
        call_id = _field(call, "id")
        call_type = _field(call, "type")
        function = _field(call, "function")
        name = _field(function, "name")
        arguments = _field(function, "arguments")
        if not isinstance(call_id, str) or not call_id.strip():
            raise ValueError(
                f"teacher tool_calls[{index}] requires non-empty id"
            )
        if call_id in seen_ids:
            raise ValueError(f"duplicate teacher tool call id: {call_id}")
        if call_type != "function":
            raise ValueError(
                f"teacher tool_calls[{index}] type must equal function"
            )
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"teacher tool_calls[{index}] requires function name"
            )
        if not isinstance(arguments, (str, dict)):
            raise ValueError(
                f"teacher tool_calls[{index}] arguments must be JSON text or object"
            )
        seen_ids.add(call_id)
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
    return normalized


def _parse_arguments(arguments: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    parsed = json.loads(arguments)
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must decode to object")
    return parsed


def _default_tool_registry() -> Mapping[str, Any]:
    from app.services import agent_graph

    return {tool.name: tool for tool in agent_graph.TOOLS}


def _invoke_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    tool_registry: Mapping[str, Any],
    tool_executor: Callable[[str, dict[str, Any]], Any] | None,
) -> Any:
    if tool_executor is not None:
        return tool_executor(name, arguments)
    tool = tool_registry.get(name)
    if tool is None:
        raise ValueError(f"tool registry has no implementation for {name}")
    invoke = getattr(tool, "invoke", None)
    if callable(invoke):
        return invoke(arguments)
    if callable(tool):
        return tool(arguments)
    raise ValueError(f"tool registry entry {name} is not invokable")


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


def _grounding_retry_prompt(errors: list[str]) -> str:
    return (
        "上一版最终回答未通过接地校验，不能落盘。"
        f"校验错误：{errors!r}。\n"
        "请只重写最终回答，不要再调用工具。要求：\n"
        "1. 只能使用上方用户输入和真实工具结果中已经出现的事实。\n"
        "2. 优先使用 search_and_rank_vehicles 返回的 specs 字段给出具体建议。\n"
        "3. 答案中的价格、续航、电池、快充、轴距、后备箱、座位数、评分等"
        "带单位规格，必须能在工具结果中反查到同一数值和单位。\n"
        "4. 不得对工具数值做四舍五入、约数化或区间粗略改写。\n"
        "5. 如果工具未返回油耗/电耗/保修政策，只说明该项工具未返回，需以官方"
        "实时信息核验；不要补写具体数值或政策。\n"
        "6. 不要使用 [1][2] 来源列表，除非来源行能逐字对应工具结果中的 "
        "source/title/url/content。\n"
        "7. 若 search_and_rank_vehicles 返回至少 3 款合规候选，请保留至少 "
        "3 款车型对比；增程可作为广义插电备选，并注明严格 PHEV 可进一步筛选。"
    )


def generate_teacher_decision_record(
    *,
    record_id: str,
    query: str,
    intent: str,
    client: Any,
    model: str,
    max_steps: int = 6,
    tool_registry: Mapping[str, Any] | None = None,
    tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
    system_prompt: str | None = None,
    max_grounding_retries: int = 1,
    max_tokens: int | None = None,
    include_audit_metadata: bool = False,
) -> dict[str, Any]:
    """Generate one teacher-decided tool trajectory using real tool results."""
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if max_grounding_retries < 0:
        raise ValueError("max_grounding_retries must be non-negative")
    if max_tokens is not None and max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if intent not in MANDATORY_BY_INTENT:
        raise ValueError(f"unsupported intent: {intent}")
    schemas = _schemas_for_intent(intent)
    schema_map = _approved_schema_map()
    registry = tool_registry or _default_tool_registry()
    messages: list[dict[str, Any]] = []
    if system_prompt is not None:
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must be a non-empty string")
        messages.append(
            {
                "role": "system",
                "content": f"{system_prompt}\n\n{GROUNDING_INSTRUCTION}",
            }
        )
    messages.append({"role": "user", "content": query})
    teacher_messages = list(messages)
    tool_call_count = 0
    tool_call_rounds = 0
    grounding_retries = 0
    rewrite_triggered = False

    for _step in range(max_steps + max_grounding_retries):
        rewrite_mode = (
            grounding_retries > 0
            and teacher_messages
            and teacher_messages[-1].get("role") == "user"
            and "接地校验" in str(teacher_messages[-1].get("content", ""))
        )
        request: dict[str, Any] = {
            "model": model,
            "messages": teacher_messages,
            "temperature": 0 if rewrite_mode else 0.2,
        }
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        if not rewrite_mode:
            request.update({"tools": schemas, "tool_choice": "auto"})
        response = client.chat.completions.create(**request)
        choice = response.choices[0]
        finish_reason = _field(choice, "finish_reason")
        if finish_reason == "length":
            raise ValueError("teacher response was truncated by max_tokens")
        teacher_message = choice.message
        content = _field(teacher_message, "content")
        if content is not None and not isinstance(content, str):
            raise ValueError("teacher assistant content must be string or null")
        tool_calls = _normalize_teacher_tool_calls(
            _field(teacher_message, "tool_calls", [])
        )
        if not tool_calls:
            if not isinstance(content, str) or not content.strip():
                raise ValueError("teacher final answer must be non-empty")
            if tool_call_count == 0:
                raise ValueError("at least one teacher tool call is required")
            messages.append({"role": "assistant", "content": content})
            record = {"id": record_id, "messages": messages}
            errors = validate_record(record, held_out_ids=set())
            if errors:
                raise ValueError(f"generated record is invalid: {errors}")
            grounding_errors = audit_answer_grounding(record)
            if grounding_errors:
                if grounding_retries < max_grounding_retries:
                    rewrite_triggered = True
                    teacher_messages.append(
                        {"role": "assistant", "content": content}
                    )
                    teacher_messages.append(
                        {
                            "role": "user",
                            "content": _grounding_retry_prompt(grounding_errors),
                        }
                    )
                    grounding_retries += 1
                    continue
                raise ValueError(
                    f"answer grounding failed: {grounding_errors}"
                )
            if include_audit_metadata:
                record.update(
                    {
                        "intent": intent,
                        "finish_reason": finish_reason or "unknown",
                        "bounded_rewrite_triggered": rewrite_triggered,
                        "tool_call_rounds": tool_call_rounds,
                        "tool_call_count": tool_call_count,
                    }
                )
            return record

        if rewrite_mode:
            raise ValueError("grounding rewrite must not call tools")
        assistant_message = {
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        }
        trial_record = {"id": record_id, "messages": [assistant_message]}
        errors = validate_record(trial_record, held_out_ids=set())
        if errors:
            raise ValueError(f"teacher tool call validation failed: {errors}")
        messages.append(assistant_message)
        teacher_messages.append(assistant_message)
        tool_call_rounds += 1
        tool_call_count += len(tool_calls)

        for call in tool_calls:
            name = call["function"]["name"]
            if name not in schema_map:
                raise ValueError(f"unknown tool: {name}")
            arguments = _parse_arguments(call["function"]["arguments"])
            result = _invoke_tool(
                name,
                arguments,
                tool_registry=registry,
                tool_executor=tool_executor,
            )
            tool_message = {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": _serialize_tool_result(result),
            }
            messages.append(tool_message)
            teacher_messages.append(tool_message)
    raise ValueError(f"teacher did not finish within max_steps={max_steps}")


def write_pilot_report(dotenv_path: Path | None = None) -> Path:
    """Write a status-only report without probing or exposing the endpoint."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    teacher = check_teacher_available(dotenv_path=dotenv_path)
    path = OUT_DIR / "data_synth_report.md"
    fixed_smoke_path = OUT_DIR / "pilot_sft.jsonl"
    decision_pilot_path = OUT_DIR / "teacher_decision_pilot_sft.jsonl"
    fixed_smoke_rows = 0
    decision_pilot_rows = 0
    if fixed_smoke_path.exists():
        fixed_smoke_rows = sum(
            1 for line in fixed_smoke_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if decision_pilot_path.exists():
        decision_pilot_rows = sum(
            1
            for line in decision_pilot_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    data_status = (
        "teacher_decision_pilot_generated"
        if decision_pilot_rows
        else "fixed_smoke_only"
        if fixed_smoke_rows
        else "not_started"
    )
    lines = [
        "# 数据合成状态报告",
        "",
        f"- 教师配置状态：{teacher['status']}",
        f"- 教师 provider：{teacher['provider'] or 'none'}",
        f"- base_url 配置：{teacher['base_url_configured']}",
        f"- model 配置：{teacher['model_configured']}",
        f"- api_key 配置：{teacher['api_key_configured']}",
        f"- 端点验证：{teacher['endpoint_verified']}",
        f"- 工具 schema 数：{len(build_tool_schemas())}",
        f"- 数据生成状态：{data_status}",
        f"- 固定调用 smoke 记录数：{fixed_smoke_rows}",
        f"- 教师决策 pilot 记录数：{decision_pilot_rows}",
        "- 接地校验：enabled（带单位数值 / 规格逐项反查、来源行逐字反查、"
        "政策类具体承诺需有证据、截断回答拒绝落盘）",
        "- 扩量门禁：幻觉 / 假引用样本一票否决",
        "",
        (
            "当前报告只记录同一 provider 的环境变量三元组完整性；报告生成过程"
            "不执行端点验证。固定调用 smoke 只证明管道连通，不作为训练数据生成"
            "方式；教师决策 pilot 不等于全量训练集，不得宣称全量数据完成。"
            if teacher["available"]
            else (
                "当前没有完整的同源教师配置；本阶段只完成 schema、validator "
                "和配置门禁，不得启动或声明数据合成。"
            )
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    dotenv_path = ROOT / ".env"
    path = write_pilot_report(dotenv_path=dotenv_path)
    print(
        json.dumps(
            {
                "report": str(path),
                "teacher": check_teacher_available(dotenv_path=dotenv_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
