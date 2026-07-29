import json

import pytest

from data_synth.validate_tool_data import validate_record


def _record(
    *,
    record_id="train-1",
    tool_name="extract_user_profile",
    arguments='{"query": "推荐一辆家用新能源车"}',
):
    return {
        "id": record_id,
        "messages": [
            {"role": "user", "content": "推荐一辆家用新能源车"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": tool_name,
                            "arguments": arguments,
                        }
                    }
                ],
            },
        ],
    }


def test_validate_record_rejects_unknown_tool():
    errors = validate_record(
        _record(tool_name="unknown_tool", arguments="{}"),
        held_out_ids=set(),
    )

    assert errors == [
        "record.messages[1].tool_calls[0].function.name: "
        "unknown tool 'unknown_tool'"
    ]


def test_validate_record_rejects_held_out_id_leakage():
    errors = validate_record(_record(record_id="heldout-1"), {"heldout-1"})

    assert errors == [
        "record.id: held-out ID leakage detected ('heldout-1')",
    ]


@pytest.mark.parametrize(
    ("record_id", "held_out_id"),
    [
        ("ＴＲＡＩＮ－１", "train-1"),
        ("Train-Case", "train-case"),
        ("  train\t id  ", "train id"),
    ],
)
def test_validate_record_rejects_normalized_held_out_id_leakage(
    record_id,
    held_out_id,
):
    errors = validate_record(_record(record_id=record_id), {held_out_id})

    assert errors == [
        f"record.id: held-out ID leakage detected ({record_id!r})",
    ]


@pytest.mark.parametrize(
    ("held_out_ids", "expected_error"),
    [
        ({123}, "held_out_ids: all IDs must be strings"),
        ({"\u3000\t"}, "held_out_ids: IDs must normalize to non-empty strings"),
    ],
)
def test_validate_record_rejects_invalid_held_out_ids(
    held_out_ids,
    expected_error,
):
    assert validate_record(_record(), held_out_ids) == [expected_error]


@pytest.mark.parametrize("held_out_ids", ["train-1", b"train-1", 123])
def test_validate_record_rejects_invalid_held_out_id_containers(held_out_ids):
    assert validate_record(_record(), held_out_ids) == [
        "held_out_ids: must be an iterable container of strings, "
        "not str or bytes"
    ]


def test_validate_record_rejects_normalized_held_out_query_leakage():
    record = _record()
    record["messages"][0]["content"] = "  ＴＥＳＴ－ＣＡＳＥ\u3000 Query\t"

    errors = validate_record(
        record,
        held_out_ids=set(),
        held_out_queries={"test-case query"},
    )

    assert errors == [
        "record.messages[0].content: held-out query leakage detected",
    ]


@pytest.mark.parametrize(
    "held_out_queries",
    ["test query", b"test query", 123],
)
def test_validate_record_rejects_string_like_held_out_query_containers(
    held_out_queries,
):
    errors = validate_record(
        _record(),
        held_out_ids=set(),
        held_out_queries=held_out_queries,
    )

    assert errors == [
        "held_out_queries: must be an iterable container of strings, "
        "not str or bytes"
    ]


def test_validate_record_materializes_other_held_out_query_iterables():
    record = _record()
    record["messages"][0]["content"] = "  ＴＥＳＴ\u3000 Query\t"
    held_out_queries = (query for query in ["test query"])

    errors = validate_record(
        record,
        held_out_ids=set(),
        held_out_queries=held_out_queries,
    )

    assert errors == [
        "record.messages[0].content: held-out query leakage detected",
    ]


def test_validate_record_rejects_non_string_held_out_queries():
    errors = validate_record(
        _record(),
        held_out_ids=set(),
        held_out_queries=["valid query", 123],
    )

    assert errors == ["held_out_queries: all queries must be strings"]


def test_validate_record_rejects_empty_normalized_held_out_queries():
    assert validate_record(
        _record(),
        held_out_ids=set(),
        held_out_queries=["\u3000\t"],
    ) == [
        "held_out_queries: queries must normalize to non-empty strings",
    ]


def test_validate_record_rejects_invalid_arguments_json():
    errors = validate_record(_record(arguments="{broken"), set())

    assert errors == [
        "record.messages[1].tool_calls[0].function.arguments: invalid JSON",
    ]


@pytest.mark.parametrize("arguments", ["[]", '"query"', "null"])
def test_validate_record_rejects_non_object_arguments_json(arguments):
    errors = validate_record(_record(arguments=arguments), set())

    assert errors == [
        "record.messages[1].tool_calls[0].function.arguments: "
        "JSON value must be an object"
    ]


def test_validate_record_rejects_non_string_non_dict_arguments():
    errors = validate_record(_record(arguments=["not", "an", "object"]), set())

    assert errors == [
        "record.messages[1].tool_calls[0].function.arguments: "
        "must be a JSON object string or object"
    ]


def test_validate_record_rejects_missing_required_argument():
    errors = validate_record(
        _record(tool_name="retrieve_knowledge_base", arguments={}),
        set(),
    )

    assert errors == [
        "record.messages[1].tool_calls[0].function.arguments.query: "
        "required property is missing"
    ]


def test_validate_record_rejects_additional_argument():
    errors = validate_record(
        _record(
            tool_name="retrieve_knowledge_base",
            arguments={"query": "续航", "unexpected": True},
        ),
        set(),
    )

    assert errors == [
        "record.messages[1].tool_calls[0].function.arguments.unexpected: "
        "additional property is not allowed"
    ]


@pytest.mark.parametrize(
    ("tool_name", "arguments", "property_name", "expected_type", "actual_type"),
    [
        (
            "extract_user_profile",
            {"query": "家用车", "budget_max": True},
            "budget_max",
            "integer",
            "boolean",
        ),
        (
            "retrieve_knowledge_base",
            {"query": 42},
            "query",
            "string",
            "integer",
        ),
    ],
)
def test_validate_record_rejects_argument_type_errors(
    tool_name,
    arguments,
    property_name,
    expected_type,
    actual_type,
):
    errors = validate_record(
        _record(tool_name=tool_name, arguments=arguments),
        set(),
    )

    assert errors == [
        "record.messages[1].tool_calls[0].function.arguments."
        f"{property_name}: expected {expected_type}, got {actual_type}"
    ]


@pytest.mark.parametrize(
    "arguments",
    [
        '{"budget_max": 300000, "top_k": 5}',
        {"budget_max": 300000, "top_k": 5},
    ],
)
def test_validate_record_accepts_valid_json_string_and_dict_arguments(arguments):
    errors = validate_record(
        _record(tool_name="search_and_rank_vehicles", arguments=arguments),
        set(),
    )

    assert errors == []


@pytest.mark.parametrize(
    ("record", "expected_error"),
    [
        (
            {"id": "train-1"},
            "record.messages: must be a list",
        ),
        (
            {"id": "train-1", "messages": []},
            "record.messages: must contain at least one message",
        ),
        (
            {"id": "train-1", "messages": "not-a-list"},
            "record.messages: must be a list",
        ),
    ],
)
def test_validate_record_rejects_missing_or_invalid_messages(
    record,
    expected_error,
):
    assert validate_record(record, set()) == [expected_error]


@pytest.mark.parametrize(
    ("message", "expected_error"),
    [
        (
            None,
            "record.messages[0]: must be an object",
        ),
        (
            {},
            "record.messages[0].role: "
            "must be one of assistant, system, tool, user",
        ),
        (
            {"role": "user", "content": 123},
            "record.messages[0].content: user query must be a string",
        ),
        (
            {"role": "user"},
            "record.messages[0].content: user query must be a string",
        ),
        (
            {"role": "developer", "content": "query"},
            "record.messages[0].role: "
            "must be one of assistant, system, tool, user",
        ),
        (
            {"role": "user", "content": "query", "tool_calls": []},
            "record.messages[0].tool_calls: "
            "only allowed on assistant messages",
        ),
    ],
)
def test_validate_record_rejects_malformed_messages(message, expected_error):
    record = {"id": "train-1", "messages": [message]}

    assert validate_record(record, set()) == [expected_error]


@pytest.mark.parametrize("role", [[], {}, 123, None])
def test_validate_record_rejects_non_string_roles_without_raising(role):
    record = {
        "id": "train-1",
        "messages": [{"role": role, "content": "query"}],
    }

    assert validate_record(record, set()) == [
        "record.messages[0].role: "
        "must be one of assistant, system, tool, user"
    ]


@pytest.mark.parametrize(
    ("role", "content", "expected_error"),
    [
        (
            "user",
            "",
            "record.messages[0].content: "
            "user query must be a non-empty string",
        ),
        (
            "user",
            "  \t",
            "record.messages[0].content: "
            "user query must be a non-empty string",
        ),
        (
            "system",
            "",
            "record.messages[0].content: "
            "system content must be a non-empty string",
        ),
        (
            "system",
            "  \t",
            "record.messages[0].content: "
            "system content must be a non-empty string",
        ),
    ],
)
def test_validate_record_rejects_blank_user_and_system_content(
    role,
    content,
    expected_error,
):
    record = {
        "id": "train-1",
        "messages": [{"role": role, "content": content}],
    }

    assert validate_record(record, set()) == [expected_error]


@pytest.mark.parametrize("content", [None, 123])
def test_validate_record_rejects_missing_or_invalid_system_content(content):
    message = {"role": "system"}
    if content is not None:
        message["content"] = content
    record = {"id": "train-1", "messages": [message]}

    assert validate_record(record, set()) == [
        "record.messages[0].content: system content must be a string"
    ]


@pytest.mark.parametrize("content", [None, 123])
def test_validate_record_rejects_missing_or_non_string_tool_content(content):
    message = {"role": "tool", "tool_call_id": "call-1"}
    if content is not None:
        message["content"] = content
    record = {"id": "train-1", "messages": [message]}

    assert validate_record(record, set()) == [
        "record.messages[0].content: tool content must be a string"
    ]


@pytest.mark.parametrize("tool_call_id", [None, "", "  \t", 123])
def test_validate_record_rejects_missing_or_invalid_tool_call_id(tool_call_id):
    message = {"role": "tool", "content": ""}
    if tool_call_id is not None:
        message["tool_call_id"] = tool_call_id
    record = {"id": "train-1", "messages": [message]}

    assert validate_record(record, set()) == [
        "record.messages[0].tool_call_id: must be a non-empty string"
    ]


@pytest.mark.parametrize(
    "message",
    [
        {"role": "assistant"},
    ],
)
def test_validate_record_rejects_assistant_without_content_or_tool_calls(
    message,
):
    record = {"id": "train-1", "messages": [message]}

    assert validate_record(record, set()) == [
        "record.messages[0]: assistant message must have string content "
        "or non-empty tool_calls"
    ]


@pytest.mark.parametrize("content", [None, 123, []])
def test_validate_record_rejects_non_string_assistant_content(content):
    record = {
        "id": "train-1",
        "messages": [{"role": "assistant", "content": content}],
    }

    assert validate_record(record, set()) == [
        "record.messages[0].content: assistant content must be a string"
    ]


@pytest.mark.parametrize("content", ["", "  \t"])
def test_validate_record_rejects_blank_assistant_content_without_tool_calls(
    content,
):
    record = {
        "id": "train-1",
        "messages": [{"role": "assistant", "content": content}],
    }

    assert validate_record(record, set()) == [
        "record.messages[0].content: assistant content must be a "
        "non-empty string when tool_calls are absent"
    ]


@pytest.mark.parametrize(
    ("include_content", "content"),
    [
        (False, None),
        (True, None),
        (True, ""),
    ],
)
def test_validate_record_accepts_empty_assistant_content_with_valid_tool_calls(
    include_content,
    content,
):
    record = _record()
    if include_content:
        record["messages"][1]["content"] = content

    assert validate_record(record, set()) == []


@pytest.mark.parametrize("content", [123, [], {}])
def test_validate_record_rejects_non_string_assistant_content_with_tool_calls(
    content,
):
    record = _record()
    record["messages"][1]["content"] = content

    assert validate_record(record, set()) == [
        "record.messages[1].content: assistant content must be a string"
    ]


@pytest.mark.parametrize(
    "message",
    [
        {"role": "user", "content": "query"},
        {"role": "system", "content": "prompt"},
        {"role": "tool", "content": "", "tool_call_id": "call-1"},
        {"role": "assistant", "content": "answer"},
    ],
)
def test_validate_record_accepts_valid_role_specific_content(message):
    record = {"id": "train-1", "messages": [message]}

    assert validate_record(record, set()) == []

@pytest.mark.parametrize(
    "message",
    [
        {"role": "system", "content": "prompt", "tool_calls": []},
        {
            "role": "tool",
            "content": "result",
            "tool_call_id": "call-1",
            "tool_calls": [],
        },
    ],
)
def test_validate_record_rejects_tool_calls_on_other_non_assistant_roles(
    message,
):
    record = {"id": "train-1", "messages": [message]}

    assert validate_record(record, set()) == [
        "record.messages[0].tool_calls: "
        "only allowed on assistant messages"
    ]


def test_validate_record_reports_tool_message_fields_in_stable_order():
    record = {
        "id": "train-1",
        "messages": [
            {"role": "tool", "content": 123, "tool_call_id": ""},
        ],
    }

    assert validate_record(record, set()) == [
        "record.messages[0].content: tool content must be a string",
        "record.messages[0].tool_call_id: must be a non-empty string",
    ]


@pytest.mark.parametrize(
    ("tool_calls", "expected_error"),
    [
        (
            None,
            "record.messages[0].tool_calls: must be a list",
        ),
        (
            [],
            "record.messages[0].tool_calls: "
            "must contain at least one tool call",
        ),
        (
            ["bad-call"],
            "record.messages[0].tool_calls[0]: must be an object",
        ),
        (
            [{"function": []}],
            "record.messages[0].tool_calls[0].function: must be an object",
        ),
        (
            [{"function": {"name": "", "arguments": {}}}],
            "record.messages[0].tool_calls[0].function.name: "
            "must be a non-empty string",
        ),
    ],
)
def test_validate_record_rejects_malformed_assistant_tool_calls(
    tool_calls,
    expected_error,
):
    record = {
        "id": "train-1",
        "messages": [{"role": "assistant", "tool_calls": tool_calls}],
    }

    assert validate_record(record, set()) == [expected_error]


@pytest.mark.parametrize(
    ("extra_field", "expected_path"),
    [
        ("unexpected", "record.messages[1].tool_calls[0].unexpected"),
        (
            "function.unexpected",
            "record.messages[1].tool_calls[0].function.unexpected",
        ),
    ],
)
def test_validate_record_rejects_unknown_tool_call_fields(
    extra_field,
    expected_path,
):
    record = _record()
    call = record["messages"][1]["tool_calls"][0]
    if extra_field.startswith("function."):
        call["function"][extra_field.removeprefix("function.")] = True
    else:
        call[extra_field] = True

    assert validate_record(record, set()) == [
        f"{expected_path}: additional property is not allowed"
    ]


@pytest.mark.parametrize("call_type", ["computer", "", 123])
def test_validate_record_rejects_invalid_tool_call_type(call_type):
    record = _record()
    record["messages"][1]["tool_calls"][0]["type"] = call_type

    assert validate_record(record, set()) == [
        "record.messages[1].tool_calls[0].type: must be 'function'"
    ]


@pytest.mark.parametrize("call_id", ["", "  \t", 123, None])
def test_validate_record_rejects_invalid_optional_tool_call_id(call_id):
    record = _record()
    record["messages"][1]["tool_calls"][0]["id"] = call_id

    assert validate_record(record, set()) == [
        "record.messages[1].tool_calls[0].id: must be a non-empty string"
    ]


def test_validate_record_accepts_openai_tool_call_fields():
    record = _record()
    record["messages"][1]["tool_calls"][0].update(
        {"id": "call-1", "type": "function"}
    )

    assert validate_record(record, set()) == []


def test_validate_record_still_validates_tool_calls_with_assistant_content():
    record = {
        "id": "train-1",
        "messages": [
            {"role": "assistant", "content": "answer", "tool_calls": []},
        ],
    }

    assert validate_record(record, set()) == [
        "record.messages[0].tool_calls: must contain at least one tool call",
    ]


@pytest.mark.parametrize("record_id", [None, "", "   ", 123])
def test_validate_record_rejects_invalid_record_id(record_id):
    record = _record(record_id=record_id)

    assert validate_record(record, set()) == [
        "record.id: must be a non-empty string",
    ]


_TEACHER_ENV_NAMES = (
    "CHAT_BASE_URL",
    "CHAT_MODEL",
    "CHAT_API_KEY",
    "ARK_BASE_URL",
    "ARK_CHAT_MODEL",
    "ARK_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_CHAT_MODEL",
    "OPENAI_API_KEY",
)


def _clear_teacher_env(monkeypatch):
    for name in _TEACHER_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_teacher_gate_is_blocked_without_complete_config(monkeypatch):
    from data_synth.generate_sft_data import check_teacher_available

    _clear_teacher_env(monkeypatch)

    result = check_teacher_available()

    assert result == {
        "available": False,
        "status": "blocked",
        "provider": None,
        "base_url_configured": False,
        "model_configured": False,
        "api_key_configured": False,
        "endpoint_verified": False,
        "note": (
            "A complete CHAT, ARK, or legacy OPENAI teacher "
            "configuration is required."
        ),
    }


@pytest.mark.parametrize(
    ("provider", "variables"),
    [
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
    ],
)
def test_teacher_gate_accepts_complete_atomic_provider(
    monkeypatch,
    provider,
    variables,
):
    from data_synth.generate_sft_data import check_teacher_available

    _clear_teacher_env(monkeypatch)
    for name, value in zip(
        variables,
        ("https://teacher.example/v1", "teacher-model", "secret"),
    ):
        monkeypatch.setenv(name, value)

    result = check_teacher_available()

    assert result["available"] is True
    assert result["status"] == "config_ready"
    assert result["provider"] == provider
    assert result["base_url_configured"] is True
    assert result["model_configured"] is True
    assert result["api_key_configured"] is True
    assert result["endpoint_verified"] is False


def test_teacher_gate_rejects_mixed_and_whitespace_config(monkeypatch):
    from data_synth.generate_sft_data import check_teacher_available

    _clear_teacher_env(monkeypatch)
    monkeypatch.setenv("CHAT_BASE_URL", "https://teacher.example/v1")
    monkeypatch.setenv("ARK_CHAT_MODEL", "teacher-model")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    assert check_teacher_available()["available"] is False

    _clear_teacher_env(monkeypatch)
    monkeypatch.setenv("CHAT_BASE_URL", " \t")
    monkeypatch.setenv("CHAT_MODEL", "teacher-model")
    monkeypatch.setenv("CHAT_API_KEY", "secret")

    assert check_teacher_available()["available"] is False


def test_teacher_gate_uses_chat_then_ark_then_openai_priority(monkeypatch):
    from data_synth.generate_sft_data import check_teacher_available

    _clear_teacher_env(monkeypatch)
    for prefix, model_name in (
        ("OPENAI", "OPENAI_CHAT_MODEL"),
        ("ARK", "ARK_CHAT_MODEL"),
        ("CHAT", "CHAT_MODEL"),
    ):
        monkeypatch.setenv(
            f"{prefix}_BASE_URL",
            f"https://{prefix.lower()}.example/v1",
        )
        monkeypatch.setenv(model_name, f"{prefix.lower()}-model")
        monkeypatch.setenv(f"{prefix}_API_KEY", f"{prefix.lower()}-secret")

    assert check_teacher_available()["provider"] == "chat"

    monkeypatch.delenv("CHAT_API_KEY")
    assert check_teacher_available()["provider"] == "ark"

    monkeypatch.delenv("ARK_API_KEY")
    assert check_teacher_available()["provider"] == "openai"


def test_teacher_report_and_cli_do_not_leak_config_or_claim_generation(
    monkeypatch,
    tmp_path,
    capsys,
):
    from data_synth import generate_sft_data

    _clear_teacher_env(monkeypatch)
    secrets = {
        "CHAT_BASE_URL": "https://teacher-secret.example/v1",
        "CHAT_MODEL": "secret-model-name",
        "CHAT_API_KEY": "secret-api-key",
    }
    for name, value in secrets.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(generate_sft_data, "OUT_DIR", tmp_path)

    path = generate_sft_data.write_pilot_report()
    report = path.read_text(encoding="utf-8")
    generate_sft_data.main()
    cli_output = capsys.readouterr().out
    payload = json.loads(cli_output)

    for secret in secrets.values():
        assert secret not in report
        assert secret not in cli_output
    assert payload["teacher"]["status"] == "config_ready"
    assert payload["teacher"]["endpoint_verified"] is False
    assert "工具 schema 数：5" in report
    assert "端点验证：False" in report
    assert "数据生成状态：not_started" in report
    assert "数据已生成" not in report


def test_teacher_gate_accepts_seedpro_alias_from_dotenv(tmp_path, monkeypatch):
    from data_synth.generate_sft_data import check_teacher_available

    _clear_teacher_env(monkeypatch)
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "ARK_API_KEY=secret-api-key\n"
        "SEEDPRO_EP=seedpro-endpoint-id\n",
        encoding="utf-8",
    )

    result = check_teacher_available(dotenv_path=dotenv)

    assert result["available"] is True
    assert result["status"] == "config_ready"
    assert result["provider"] == "ark"
    assert result["base_url_configured"] is True
    assert result["model_configured"] is True
    assert result["api_key_configured"] is True
    assert result["endpoint_verified"] is False
    assert "secret-api-key" not in json.dumps(result, ensure_ascii=False)
    assert "seedpro-endpoint-id" not in json.dumps(result, ensure_ascii=False)
