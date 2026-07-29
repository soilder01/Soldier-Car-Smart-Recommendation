import json
from types import SimpleNamespace

import pytest

from data_synth.generate_sft_data import audit_answer_grounding
from data_synth.generate_sft_data import audit_teacher_decision_record
from data_synth.generate_sft_data import generate_teacher_decision_record
from data_synth.validate_tool_data import validate_record


def _fake_message(content=None, tool_calls=None, finish_reason="stop"):
    return SimpleNamespace(
        content=content,
        tool_calls=list(tool_calls or []),
        finish_reason=finish_reason,
    )


def _fake_call(name, arguments=None, call_id="call-1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(
                {"query": "补能策略"} if arguments is None else arguments,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        ),
    )


class _FakeCompletions:
    def __init__(self, messages):
        self._messages = iter(messages)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        message = next(self._messages)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=message,
                    finish_reason=getattr(message, "finish_reason", "stop"),
                )
            ]
        )


class _FakeClient:
    def __init__(self, messages):
        self.chat = SimpleNamespace(
            completions=_FakeCompletions(messages)
        )


def test_teacher_decision_record_preserves_teacher_tool_sequence():
    events = []

    def execute(name, arguments):
        events.append((name, arguments))
        return f"{name} result"

    client = _FakeClient(
        [
            _fake_message(
                tool_calls=[
                    _fake_call(
                        "retrieve_knowledge_base",
                        {"query": "补能策略"},
                        call_id="teacher-call-1",
                    )
                ]
            ),
            _fake_message(content="最终答案"),
        ]
    )

    record = generate_teacher_decision_record(
        record_id="pilot-1",
        query="解释插混 SUV 的补能策略",
        intent="knowledge",
        client=client,
        model="teacher-model",
        tool_executor=execute,
        max_tokens=512,
    )

    assert [message["role"] for message in record["messages"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert events == [("retrieve_knowledge_base", {"query": "补能策略"})]
    assert record["messages"][1]["tool_calls"][0]["id"] == "teacher-call-1"
    assert record["messages"][2]["tool_call_id"] == "teacher-call-1"
    assert validate_record(record, held_out_ids=set()) == []
    assert client.chat.completions.requests[0]["tools"]
    assert client.chat.completions.requests[0]["tool_choice"] == "auto"
    assert client.chat.completions.requests[0]["max_tokens"] == 512


def test_teacher_decision_record_supports_multiple_tool_rounds():
    client = _FakeClient(
        [
            _fake_message(
                tool_calls=[
                    _fake_call(
                        "extract_user_profile",
                        {"query": "预算25万三口之家", "budget_max": 250000},
                        call_id="profile",
                    )
                ]
            ),
            _fake_message(
                tool_calls=[
                    _fake_call(
                        "search_and_rank_vehicles",
                        {"budget_max": 250000, "top_k": 3},
                        call_id="search",
                    )
                ]
            ),
            _fake_message(content="最终推荐"),
        ]
    )

    record = generate_teacher_decision_record(
        record_id="pilot-2",
        query="预算25万三口之家推荐新能源车",
        intent="recommend",
        client=client,
        model="teacher-model",
        tool_executor=lambda name, arguments: {"tool": name, "ok": True},
    )

    assert [message["role"] for message in record["messages"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    assert validate_record(record, held_out_ids=set()) == []


def test_teacher_decision_record_can_persist_system_prompt():
    client = _FakeClient(
        [
            _fake_message(
                tool_calls=[
                    _fake_call(
                        "retrieve_knowledge_base",
                        {"query": "补能策略"},
                        call_id="tool-1",
                    )
                ]
            ),
            _fake_message(content="最终答案"),
        ]
    )

    record = generate_teacher_decision_record(
        record_id="pilot-system",
        query="解释补能策略",
        intent="knowledge",
        client=client,
        model="teacher-model",
        tool_executor=lambda name, arguments: "result",
        system_prompt="你是新能源汽车知识问答助手。",
    )

    assert [message["role"] for message in record["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert validate_record(record, held_out_ids=set()) == []


def test_audit_teacher_decision_record_enforces_intent_tool_contract():
    record = {
        "id": "audit-1",
        "messages": [
            {"role": "user", "content": "推荐新能源车"},
            {
                "role": "assistant",
                "tool_calls": [
                    _fake_call(
                        "search_and_rank_vehicles",
                        {"budget_max": 250000},
                    ).__dict__,
                    _fake_call("generate_sales_talk", {}, call_id="call-2").__dict__,
                ],
            },
            {"role": "assistant", "content": "最终答案"},
        ],
    }
    for call in record["messages"][1]["tool_calls"]:
        call["function"] = call["function"].__dict__

    assert audit_teacher_decision_record("recommend", record) == [
        "mandatory tools missing or out of order: expected "
        "['extract_user_profile', 'search_and_rank_vehicles', "
        "'generate_sales_talk'] actual ['search_and_rank_vehicles', "
        "'generate_sales_talk']"
    ]


def test_compare_audit_requires_named_lookup_and_parallel_answer():
    calls = [
        _fake_call(
            "extract_user_profile",
            {"query": "宋PLUS和银河L7怎么选"},
            call_id="profile",
        ).__dict__,
        _fake_call(
            "search_and_rank_vehicles",
            {
                "model_names": ["比亚迪宋PLUS DM-i", "吉利银河L7"],
                "concerns": "空间,能耗",
            },
            call_id="lookup",
        ).__dict__,
        _fake_call(
            "retrieve_knowledge_base",
            {"query": "宋PLUS 银河L7 对比"},
            call_id="knowledge",
        ).__dict__,
    ]
    for call in calls:
        call["function"] = call["function"].__dict__
    record = {
        "id": "compare-named",
        "messages": [
            {"role": "user", "content": "宋PLUS和银河L7怎么选"},
            {"role": "assistant", "tool_calls": [calls[0]]},
            {"role": "tool", "tool_call_id": "profile", "content": "{}"},
            {"role": "assistant", "tool_calls": [calls[1]]},
            {
                "role": "tool",
                "tool_call_id": "lookup",
                "content": json.dumps(
                    {
                        "named_vehicle_lookup": {
                            "requested_model_names": [
                                "比亚迪宋PLUS DM-i",
                                "吉利银河L7",
                            ],
                            "resolved_model_names": [
                                "比亚迪宋PLUS DM-i",
                                "吉利银河L7",
                            ],
                            "missing_model_names": [],
                            "named_vehicle_missing": False,
                        },
                        "named_vehicles": [
                            {
                                "full_name": "比亚迪宋PLUS DM-i",
                                "specs": {"price_range": "12.98-16.98万"},
                            },
                            {
                                "full_name": "吉利银河L7",
                                "specs": {"price_range": "12.57-16.97万"},
                            },
                        ],
                        "supplemental_vehicles": [],
                    },
                    ensure_ascii=False,
                ),
            },
            {"role": "assistant", "tool_calls": [calls[2]]},
            {"role": "tool", "tool_call_id": "knowledge", "content": "[]"},
            {
                "role": "assistant",
                "content": "| 车型 | 对比 |\n|---|---|\n| 比亚迪宋PLUS DM-i | A |\n| 吉利银河L7 | B |",
            },
        ],
    }

    assert audit_teacher_decision_record("compare", record) == []


def test_compare_audit_rejects_missing_model_names():
    calls = [
        _fake_call(
            "extract_user_profile",
            {"query": "宋PLUS和银河L7怎么选"},
            call_id="profile",
        ).__dict__,
        _fake_call(
            "search_and_rank_vehicles",
            {"concerns": "空间,能耗"},
            call_id="lookup",
        ).__dict__,
        _fake_call(
            "retrieve_knowledge_base",
            {"query": "宋PLUS 银河L7 对比"},
            call_id="knowledge",
        ).__dict__,
    ]
    for call in calls:
        call["function"] = call["function"].__dict__
    record = {
        "id": "compare-missing-model-names",
        "messages": [
            {"role": "user", "content": "宋PLUS和银河L7怎么选"},
            {"role": "assistant", "tool_calls": [calls[0]]},
            {"role": "tool", "tool_call_id": "profile", "content": "{}"},
            {"role": "assistant", "tool_calls": [calls[1]]},
            {"role": "tool", "tool_call_id": "lookup", "content": "[]"},
            {"role": "assistant", "tool_calls": [calls[2]]},
            {"role": "tool", "tool_call_id": "knowledge", "content": "[]"},
            {"role": "assistant", "content": "最终答案"},
        ],
    }

    assert audit_teacher_decision_record("compare", record) == [
        "compare first search_and_rank_vehicles call must include "
        "two non-empty model_names"
    ]


def test_answer_grounding_rejects_unsupported_hard_metrics_and_warranty():
    record = {
        "id": "grounding-1",
        "messages": [
            {"role": "user", "content": "预算28万，关注插混SUV"},
            {
                "role": "tool",
                "tool_call_id": "tool-1",
                "content": json.dumps(
                    {
                        "model": "比亚迪 宋PLUS DM-i",
                        "price": "12-16万",
                        "range_km": 1100,
                    },
                    ensure_ascii=False,
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "宋PLUS DM-i 价格12.98-16.98万元[1]，"
                    "亏电油耗4.4L/100km，并提供三电终身保修。"
                ),
            },
        ],
    }

    errors = audit_answer_grounding(record)

    assert "unsupported hard claim: 12.98-16.98万元" in errors
    assert "unsupported hard claim: 4.4L/100km" in errors
    assert any("三电终身保修" in error for error in errors)


def test_answer_grounding_rejects_fabricated_quoted_sources():
    record = {
        "id": "grounding-source",
        "messages": [
            {"role": "user", "content": "推荐插混SUV"},
            {
                "role": "tool",
                "tool_call_id": "tool-1",
                "content": json.dumps(
                    {
                        "source": "06_家庭用户购车指南.md",
                        "content": "家庭用户关注空间、安全、补能便利。",
                    },
                    ensure_ascii=False,
                ),
            },
            {
                "role": "assistant",
                "content": "[1] 本地车型库推荐案例：《推荐案例-比亚迪 宋PLUS DM-i》",
            },
        ],
    }

    errors = audit_answer_grounding(record)

    assert "unsupported quoted source: 推荐案例-比亚迪 宋PLUS DM-i" in errors


def test_answer_grounding_rejects_fabricated_bracket_sources():
    record = {
        "id": "grounding-bracket-source",
        "messages": [
            {"role": "user", "content": "推荐插混SUV"},
            {
                "role": "tool",
                "tool_call_id": "tool-1",
                "content": json.dumps(
                    {
                        "source": "08-推荐案例/推荐案例-20260622-182226-比亚迪-宋PLUS-DM-i.md",
                        "content": "title: 推荐案例-20260622-182226-比亚迪 宋PLUS DM-i",
                    },
                    ensure_ascii=False,
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "推荐比亚迪宋PLUS DM-i[1]。\n\n"
                    "[1] 历史推荐案例参考：比亚迪宋PLUS DM-i家庭用户推荐记录"
                ),
            },
        ],
    }

    assert audit_answer_grounding(record) == [
        "unsupported bracket source: [1] 历史推荐案例参考：比亚迪宋PLUS DM-i家庭用户推荐记录"
    ]


def test_answer_grounding_rejects_unresolved_bracket_citation():
    record = {
        "id": "grounding-bracket-citation",
        "messages": [
            {"role": "user", "content": "推荐插混SUV"},
            {
                "role": "tool",
                "tool_call_id": "tool-1",
                "content": json.dumps({"model": "宋PLUS DM-i"}, ensure_ascii=False),
            },
            {
                "role": "assistant",
                "content": "推荐宋PLUS DM-i[1]。",
            },
        ],
    }

    assert audit_answer_grounding(record) == [
        "unsupported bracket citation: [1]"
    ]


def test_answer_grounding_rejects_unsupported_spec_number_with_same_unit():
    record = {
        "id": "grounding-spec-number",
        "messages": [
            {"role": "user", "content": "关注空间和能耗"},
            {
                "role": "tool",
                "tool_call_id": "tool-1",
                "content": json.dumps(
                    {
                        "model": "比亚迪 宋PLUS DM-i",
                        "price": "12-16万",
                        "specs": {
                            "wheelbase": "2765mm",
                            "trunk_volume": "574L",
                        },
                    },
                    ensure_ascii=False,
                ),
            },
            {
                "role": "assistant",
                "content": "该车轴距2920毫米，后备箱容积574L。",
            },
        ],
    }

    assert audit_answer_grounding(record) == [
        "unsupported hard claim: 2920毫米"
    ]


def test_answer_grounding_accepts_unit_aliases_from_tool_specs():
    record = {
        "id": "grounding-unit-alias",
        "messages": [
            {"role": "user", "content": "关注空间和能耗"},
            {
                "role": "tool",
                "tool_call_id": "tool-1",
                "content": json.dumps(
                    {
                        "model": "比亚迪 宋PLUS DM-i",
                        "specs": {
                            "price_range": "12.98-16.98万",
                            "cltc_range": "1100km",
                            "battery": "18.3kWh",
                            "fast_charge": "35分钟",
                            "seats": "5座",
                            "wheelbase": "2765mm",
                        },
                    },
                    ensure_ascii=False,
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "宋PLUS DM-i 价格为12.98-16.98万元，CLTC续航1100公里，"
                    "电池18.3度，快充35min，5座，轴距2765毫米。"
                ),
            },
        ],
    }

    assert audit_answer_grounding(record) == []


def test_answer_grounding_accepts_policy_deferral_without_specific_claim():
    record = {
        "id": "grounding-policy-deferral",
        "messages": [
            {"role": "user", "content": "关心保修政策"},
            {
                "role": "tool",
                "tool_call_id": "tool-1",
                "content": json.dumps(
                    {
                        "model": "比亚迪 宋PLUS DM-i",
                        "known_missing_specs": ["warranty_policy"],
                    },
                    ensure_ascii=False,
                ),
            },
            {
                "role": "assistant",
                "content": "工具未返回保修政策，具体政策需以品牌官方实时公布信息核验。",
            },
        ],
    }

    assert audit_answer_grounding(record) == []


def test_answer_grounding_accepts_generic_warranty_topic_without_commitment():
    record = {
        "id": "grounding-generic-warranty-topic",
        "messages": [
            {"role": "user", "content": "客户担心保修"},
            {
                "role": "tool",
                "tool_call_id": "tool-1",
                "content": json.dumps({"topic": "电池安全沟通"}, ensure_ascii=False),
            },
            {
                "role": "assistant",
                "content": "销售可以围绕电池保修顾虑做解释，但具体政策需以官方实时信息核验。",
            },
        ],
    }

    assert audit_answer_grounding(record) == []


def test_answer_grounding_rejects_specific_warranty_commitment_without_evidence():
    record = {
        "id": "grounding-specific-warranty-commitment",
        "messages": [
            {"role": "user", "content": "客户担心保修"},
            {
                "role": "tool",
                "tool_call_id": "tool-1",
                "content": json.dumps({"topic": "电池安全沟通"}, ensure_ascii=False),
            },
            {
                "role": "assistant",
                "content": "该车型提供三电终身保修。",
            },
        ],
    }

    assert audit_answer_grounding(record) == [
        "unsupported policy claim: 该车型提供三电终身保修"
    ]


def test_answer_grounding_accepts_metrics_present_in_tool_results():
    record = {
        "id": "grounding-2",
        "messages": [
            {"role": "user", "content": "预算28万，关注插混SUV"},
            {
                "role": "tool",
                "tool_call_id": "tool-1",
                "content": json.dumps(
                    {
                        "model": "比亚迪 宋PLUS DM-i",
                        "price": "12-16万",
                        "specs": {"cltc_range": "1100km"},
                        "note": "以官方实时信息为准",
                    },
                    ensure_ascii=False,
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "工具结果显示，宋PLUS DM-i 价格为12-16万，"
                    "综合续航1100km；其他油耗和保修信息以官方实时信息为准。"
                ),
            },
        ],
    }

    assert audit_answer_grounding(record) == []


def test_teacher_decision_record_rejects_ungrounded_final_answer():
    client = _FakeClient(
        [
            _fake_message(
                tool_calls=[
                    _fake_call(
                        "search_and_rank_vehicles",
                        {"budget_max": 250000},
                        call_id="tool-1",
                    )
                ]
            ),
            _fake_message(content="价格12.98-16.98万元，三电终身保修。"),
        ]
    )

    with pytest.raises(ValueError, match="answer grounding failed"):
        generate_teacher_decision_record(
            record_id="grounding-3",
            query="推荐新能源 SUV",
            intent="recommend",
            client=client,
            model="teacher-model",
            tool_executor=lambda name, arguments: {"price": "12-16万"},
            max_grounding_retries=0,
        )


def test_teacher_decision_record_rejects_truncated_teacher_response():
    client = _FakeClient(
        [
            _fake_message(
                tool_calls=[
                    _fake_call(
                        "search_and_rank_vehicles",
                        {"budget_max": 250000},
                        call_id="tool-1",
                    )
                ]
            ),
            _fake_message(content="被截断的最终答案", finish_reason="length"),
        ]
    )

    with pytest.raises(ValueError, match="truncated"):
        generate_teacher_decision_record(
            record_id="truncated",
            query="推荐新能源 SUV",
            intent="recommend",
            client=client,
            model="teacher-model",
            tool_executor=lambda name, arguments: {"price": "12-16万"},
        )


def test_teacher_decision_record_rewrites_ungrounded_final_answer_once():
    client = _FakeClient(
        [
            _fake_message(
                tool_calls=[
                    _fake_call(
                        "search_and_rank_vehicles",
                        {"budget_max": 250000},
                        call_id="tool-1",
                    )
                ]
            ),
            _fake_message(content="价格12.98-16.98万元，亏电油耗4.4L/100km。"),
            _fake_message(content="工具结果显示价格为12-16万，相关硬指标以官方实时信息为准。"),
        ]
    )

    record = generate_teacher_decision_record(
        record_id="grounding-retry",
        query="推荐新能源 SUV",
        intent="recommend",
        client=client,
        model="teacher-model",
        tool_executor=lambda name, arguments: {"price": "12-16万"},
        max_grounding_retries=1,
    )

    assert record["messages"][-1]["content"] == (
        "工具结果显示价格为12-16万，相关硬指标以官方实时信息为准。"
    )
    assert len(client.chat.completions.requests) == 3
    assert client.chat.completions.requests[0]["tool_choice"] == "auto"
    assert "tools" not in client.chat.completions.requests[2]
    assert validate_record(record, held_out_ids=set()) == []


def test_teacher_decision_record_can_return_audit_metadata():
    client = _FakeClient(
        [
            _fake_message(
                tool_calls=[
                    _fake_call(
                        "retrieve_knowledge_base",
                        {"query": "补能策略"},
                        call_id="tool-1",
                    )
                ]
            ),
            _fake_message(content="最终答案"),
        ]
    )

    record = generate_teacher_decision_record(
        record_id="metadata",
        query="解释补能策略",
        intent="customer_service",
        client=client,
        model="teacher-model",
        tool_executor=lambda name, arguments: "result",
        include_audit_metadata=True,
    )

    assert record["intent"] == "customer_service"
    assert record["finish_reason"] == "stop"
    assert record["bounded_rewrite_triggered"] is False
    assert record["tool_call_rounds"] == 1
    assert record["tool_call_count"] == 1


def test_audit_teacher_decision_record_rejects_customer_service_recommendation_tool():
    record = {
        "id": "bad-customer-service",
        "messages": [
            {"role": "user", "content": "保修政策怎么查"},
            {
                "role": "assistant",
                "tool_calls": [
                    _fake_call(
                        "search_and_rank_vehicles",
                        {"budget_max": 250000},
                    ).__dict__,
                ],
            },
            {"role": "assistant", "content": "最终答案"},
        ],
    }
    for call in record["messages"][1]["tool_calls"]:
        call["function"] = call["function"].__dict__

    errors = audit_teacher_decision_record("customer_service", record)

    assert "disallowed tools: ['search_and_rank_vehicles']" in errors


def test_teacher_decision_record_rejects_final_without_tool_call():
    client = _FakeClient([_fake_message(content="直接回答")])

    with pytest.raises(ValueError, match="at least one teacher tool call"):
        generate_teacher_decision_record(
            record_id="pilot-3",
            query="推荐新能源 SUV",
            intent="recommend",
            client=client,
            model="teacher-model",
            tool_executor=lambda name, arguments: "unused",
        )


def test_teacher_decision_record_rejects_invalid_teacher_tool_call():
    client = _FakeClient(
        [
            _fake_message(
                tool_calls=[
                    _fake_call(
                        "unknown_tool",
                        {},
                        call_id="bad",
                    )
                ]
            )
        ]
    )

    with pytest.raises(ValueError, match="unknown tool"):
        generate_teacher_decision_record(
            record_id="pilot-4",
            query="推荐新能源 SUV",
            intent="recommend",
            client=client,
            model="teacher-model",
            tool_executor=lambda name, arguments: "unused",
        )
