import copy
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services import agent_graph
from data_synth.tool_schemas import build_tool_schemas
from scripts import evaluate_model_outputs as evaluator
from scripts import run_model_layer_eval as runner


ROOT = Path(__file__).resolve().parents[2]
VEHICLE_DATABASE = ROOT / "data" / "vehicles" / "vehicle_database.csv"
REWARD_VISIBLE = ROOT / "data" / "model_training" / "eval" / "reward_visible.jsonl"
HELD_OUT = ROOT / "data" / "model_training" / "eval" / "held_out.jsonl"
EVAL_README = ROOT / "data" / "model_training" / "eval" / "README.md"
PROTOCOL_VERSION = "task14.v3"
HALLUCINATION_SCOPE = (
    "terminal_declared_model_entities_only; "
    "unknown_free_text_omissions_are_not_deterministically_detected"
)
_UNSET = object()

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


def _case(
    case_id="case-1",
    *,
    intent="recommend",
    allowed_models=None,
):
    return {
        "id": case_id,
        "query": "预算25万，无家充，三口之家，需要SUV",
        "intent": intent,
        "expected_tools": list(MANDATORY_BY_INTENT[intent]),
        "optional_tools": list(OPTIONAL_BY_INTENT[intent]),
        "forbidden_tools": list(FORBIDDEN_BY_INTENT[intent]),
        "allowed_models": (
            allowed_models
            if allowed_models is not None
            else ["比亚迪 宋PLUS DM-i"]
        ),
    }


def _tool_arguments(name):
    return {
        "extract_user_profile": {
            "query": "预算25万，无家充，三口之家，需要SUV"
        },
        "search_and_rank_vehicles": {
            "budget_max": 250000,
            "preferred_type": "SUV",
            "top_k": 5,
        },
        "retrieve_knowledge_base": {"query": "新能源SUV补能与家庭使用"},
        "search_web_info": {"query": "新能源SUV最新权益"},
        "generate_sales_talk": {
            "budget_max": 250000,
            "top_model": "比亚迪 宋PLUS DM-i",
        },
    }[name]


def _tool_call(name, arguments=None, call_id=None):
    return {
        "id": call_id or f"call-{name}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(
                arguments if arguments is not None else _tool_arguments(name),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    }


def _calls(names):
    return [_tool_call(name, call_id=f"call-{index}") for index, name in enumerate(names)]


def _case_digest(case):
    canonical = json.dumps(
        case,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _output(
    case=None,
    *,
    tool_calls=None,
    recommended_models=None,
    schema_valid=True,
    terminal_parse_error=None,
    runner_errors=None,
    raw_assistant_content=_UNSET,
    model="test-model",
    base_url="http://local.invalid/v1",
):
    case = case or _case()
    models = (
        list(recommended_models)
        if recommended_models is not None
        else ["比亚迪 宋PLUS DM-i"]
    )
    if raw_assistant_content is _UNSET:
        raw_assistant_content = json.dumps(
            {
                "answer": "建议优先试驾比亚迪 宋PLUS DM-i。",
                "mentioned_models": models,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    calls = list(tool_calls) if tool_calls is not None else _calls(
        MANDATORY_BY_INTENT[case["intent"]]
    )
    trajectory = [
        {"role": "system", "content": "真实意图提示词和评测终态协议"},
        {"role": "user", "content": case["query"]},
    ]
    if calls:
        trajectory.append(
            {"role": "assistant", "content": None, "tool_calls": copy.deepcopy(calls)}
        )
        trajectory.extend(
            {
                "role": "tool",
                "content": '{"ok":true}',
                "tool_call_id": call["id"],
            }
            for call in calls
        )
    if raw_assistant_content is not None:
        trajectory.append(
            {"role": "assistant", "content": raw_assistant_content}
        )
    return {
        "id": case["id"],
        "protocol_version": PROTOCOL_VERSION,
        "model": model,
        "base_url": base_url,
        "case_digest": _case_digest(case),
        "schema_valid": schema_valid,
        "terminal_parse_error": terminal_parse_error,
        "runner_errors": list(runner_errors or []),
        "raw_assistant_content": raw_assistant_content,
        "tool_calls": calls,
        "latency_ms": 12.5,
        "recommended_models": models,
        "trajectory": trajectory,
    }


def _metric(report, name):
    return report["metrics"][name]


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _tool_names(tools):
    return [tool.name for tool in tools]


def test_eval_datasets_match_real_intent_truth_and_are_nfkc_disjoint():
    reward = evaluator.load_jsonl(REWARD_VISIBLE, label="cases")
    held_out = evaluator.load_jsonl(HELD_OUT, label="cases")
    catalog = set(evaluator.load_vehicle_catalog(VEHICLE_DATABASE))
    fields = {
        "id",
        "query",
        "intent",
        "expected_tools",
        "optional_tools",
        "forbidden_tools",
        "allowed_models",
    }

    assert len(reward) == 20
    assert len(held_out) == 40
    assert Counter(row["intent"] for row in reward) == {
        "recommend": 5,
        "compare": 5,
        "knowledge": 5,
        "sales": 5,
    }
    assert Counter(row["intent"] for row in held_out) == {
        "recommend": 10,
        "compare": 10,
        "knowledge": 10,
        "sales": 10,
    }

    for row in reward + held_out:
        intent = row["intent"]
        assert set(row) == fields
        assert row["expected_tools"] == MANDATORY_BY_INTENT[intent]
        assert row["optional_tools"] == OPTIONAL_BY_INTENT[intent]
        assert row["forbidden_tools"] == FORBIDDEN_BY_INTENT[intent]
        assert set(row["allowed_models"]) <= catalog
        real_tools = _tool_names(agent_graph.TOOLS_BY_INTENT[intent])
        assert set(real_tools) == set(
            row["expected_tools"] + row["optional_tools"]
        )
        assert set(real_tools).isdisjoint(row["forbidden_tools"])
        assert agent_graph.PROMPTS_BY_INTENT[intent]

    normalize = lambda value: unicodedata.normalize("NFKC", value).casefold().strip()
    reward_ids = {normalize(row["id"]) for row in reward}
    held_out_ids = {normalize(row["id"]) for row in held_out}
    reward_queries = {evaluator.normalize_query(row["query"]) for row in reward}
    held_out_queries = {evaluator.normalize_query(row["query"]) for row in held_out}
    assert len(reward_ids) == len(reward)
    assert len(held_out_ids) == len(held_out)
    assert len(reward_queries) == len(reward)
    assert len(held_out_queries) == len(held_out)
    assert reward_ids.isdisjoint(held_out_ids)
    assert reward_queries.isdisjoint(held_out_queries)


def test_eval_dataset_provenance_marks_manual_contract_cases_not_teacher_trajectories():
    readme = EVAL_README.read_text(encoding="utf-8")

    assert "人工编写的结构化模型评测用例" in readme
    assert "不是\nSeedPro 教师生成的多轮工具调用轨迹" in readme
    assert "而非占位符" in readme
    assert "不得进入 SFT、GRPO" in readme
    assert "reward、调参或 early stopping" in readme


@pytest.mark.parametrize("intent", ["recommend", "compare", "knowledge", "sales"])
def test_runner_sends_approved_schemas_in_real_registry_order(intent):
    approved = {
        schema["function"]["name"]: schema for schema in build_tool_schemas()
    }
    real_names = _tool_names(agent_graph.TOOLS_BY_INTENT[intent])
    prompt, schemas, registry = runner._load_agent_contract(intent)

    assert prompt == agent_graph._get_prompt_for_intent(intent)
    assert schemas == [approved[name] for name in real_names]
    assert list(registry) == real_names
    assert all(
        schema["function"]["parameters"]["additionalProperties"] is False
        for schema in schemas
    )


def test_evaluator_accepts_mandatory_ordered_subsequence_with_optional_calls():
    case = _case()
    actual = [
        "extract_user_profile",
        "retrieve_knowledge_base",
        "search_and_rank_vehicles",
        "search_web_info",
        "generate_sales_talk",
    ]
    report = evaluator.evaluate_records(
        [case],
        [_output(case, tool_calls=_calls(actual))],
        vehicle_catalog=set(evaluator.load_vehicle_catalog(VEHICLE_DATABASE)),
    )

    for name in (
        "tool_selection_accuracy",
        "argument_validity",
        "recommendation_hit_rate",
        "response_schema_validity",
    ):
        assert _metric(report, name)["numerator"] == 1
        assert _metric(report, name)["denominator"] == 1
        assert _metric(report, name)["ratio"] == 1.0
        assert _metric(report, name)["percentage"] == 100.0
    assert _metric(report, "hallucination_rate")["ratio"] == 0.0
    assert _metric(report, "hallucination_rate")["scope"] == HALLUCINATION_SCOPE
    assert _metric(report, "declared_model_coverage")["ratio"] == 1.0
    assert _metric(report, "response_contract_consistency")["ratio"] == 1.0
    assert _metric(report, "mandatory_tool_coverage")["numerator"] == 3
    assert _metric(report, "mandatory_tool_coverage")["denominator"] == 3
    assert report["case_results"][0]["failures"] == []


@pytest.mark.parametrize(
    ("actual", "failure"),
    [
        (
            [
                "search_and_rank_vehicles",
                "extract_user_profile",
                "generate_sales_talk",
            ],
            "mandatory tools are missing or out of order",
        ),
        (
            ["extract_user_profile", "search_and_rank_vehicles"],
            "mandatory tools are missing or out of order",
        ),
        (
            [
                "extract_user_profile",
                "search_and_rank_vehicles",
                "unknown_tool",
                "generate_sales_talk",
            ],
            "disallowed tools",
        ),
    ],
)
def test_tool_selection_enforces_ordered_mandatory_and_allowed_union(actual, failure):
    case = _case()
    calls = []
    for index, name in enumerate(actual):
        if name == "unknown_tool":
            calls.append(_tool_call(name, {}, f"call-{index}"))
        else:
            calls.append(_tool_call(name, call_id=f"call-{index}"))
    report = evaluator.evaluate_records(
        [case],
        [_output(case, tool_calls=calls)],
        vehicle_catalog={"比亚迪 宋PLUS DM-i"},
    )

    assert _metric(report, "tool_selection_accuracy")["numerator"] == 0
    assert any(
        failure in item for item in report["case_results"][0]["failures"]
    )


def test_argument_validity_is_case_level_and_requires_mandatory_completion():
    case = _case()
    report = evaluator.evaluate_records(
        [case],
        [_output(case, tool_calls=[_tool_call("retrieve_knowledge_base")])],
        vehicle_catalog={"比亚迪 宋PLUS DM-i"},
    )

    assert _metric(report, "argument_validity")["numerator"] == 0
    assert _metric(report, "argument_validity")["denominator"] == 1
    assert _metric(report, "mandatory_tool_coverage")["numerator"] == 0
    assert _metric(report, "mandatory_tool_coverage")["denominator"] == 3


def test_argument_validity_fails_case_when_any_actual_call_has_invalid_schema():
    case = _case()
    calls = _calls(MANDATORY_BY_INTENT["recommend"])
    calls[1]["function"]["arguments"] = '{"budget_max":true}'
    report = evaluator.evaluate_records(
        [case],
        [_output(case, tool_calls=calls)],
        vehicle_catalog={"比亚迪 宋PLUS DM-i"},
    )

    assert _metric(report, "argument_validity")["numerator"] == 0
    assert _metric(report, "argument_validity")["denominator"] == 1
    assert any(
        "argument budget_max must be integer" in failure
        for failure in report["case_results"][0]["failures"]
    )


def test_invalid_terminal_schema_cannot_contribute_to_success_metrics():
    case = _case()
    output = _output(
        case,
        recommended_models=[],
        schema_valid=False,
        terminal_parse_error="protocol failure: mentioned_models must be a list",
        raw_assistant_content='{"answer":"建议比亚迪 宋PLUS DM-i"}',
    )
    report = evaluator.evaluate_records(
        [case],
        [output],
        vehicle_catalog={"比亚迪 宋PLUS DM-i"},
    )

    assert _metric(report, "response_schema_validity")["numerator"] == 0
    assert _metric(report, "response_schema_validity")["denominator"] == 1
    assert _metric(report, "tool_selection_accuracy")["numerator"] == 0
    assert _metric(report, "argument_validity")["numerator"] == 0
    assert _metric(report, "recommendation_hit_rate")["numerator"] == 0
    assert report["protocol_gate"]["passed"] is False
    assert report["protocol_gate"]["failed_case_ids"] == ["case-1"]
    assert "response_schema_invalid" in report["case_results"][0]["failures"][0]


def test_declared_unknown_model_is_preserved_and_counted_as_hallucination():
    case = _case()
    models = ["比亚迪 宋PLUS DM-i", "幻影 007"]
    output = _output(
        case,
        recommended_models=models,
        raw_assistant_content=json.dumps(
            {
                "answer": "可考虑比亚迪 宋PLUS DM-i和幻影 007。",
                "mentioned_models": models,
            },
            ensure_ascii=False,
        ),
    )
    report = evaluator.evaluate_records(
        [case],
        [output],
        vehicle_catalog={"比亚迪 宋PLUS DM-i"},
    )

    metric = _metric(report, "hallucination_rate")
    assert metric["numerator"] == 1
    assert metric["denominator"] == 2
    assert metric["ratio"] == 0.5
    assert metric["scope"] == HALLUCINATION_SCOPE
    assert report["case_results"][0]["hallucinated_models"] == ["幻影 007"]


def test_known_answer_models_must_be_declared_and_gate_success_metrics():
    case = _case(allowed_models=["比亚迪 宋PLUS DM-i", "特斯拉 Model Y"])
    models = ["比亚迪 宋PLUS DM-i"]
    output = _output(
        case,
        recommended_models=models,
        raw_assistant_content=json.dumps(
            {
                "answer": "比亚迪宋PLUS DM-i更省油，特斯拉Model Y补能更成熟。",
                "mentioned_models": models,
            },
            ensure_ascii=False,
        ),
    )
    catalog = {"比亚迪 宋PLUS DM-i", "特斯拉 Model Y"}
    report = evaluator.evaluate_records([case], [output], vehicle_catalog=catalog)

    assert _metric(report, "declared_model_coverage")["numerator"] == 1
    assert _metric(report, "declared_model_coverage")["denominator"] == 2
    assert _metric(report, "response_contract_consistency")["numerator"] == 0
    assert _metric(report, "response_schema_validity")["numerator"] == 0
    assert report["protocol_gate"]["failed_case_ids"] == ["case-1"]
    assert report["case_results"][0]["undeclared_known_models"] == [
        "特斯拉 Model Y"
    ]
    for name in (
        "tool_selection_accuracy",
        "argument_validity",
        "recommendation_hit_rate",
        "mandatory_tool_coverage",
    ):
        assert _metric(report, name)["numerator"] == 0


def test_evaluator_rejects_declared_model_absent_from_terminal_answer():
    case = _case(allowed_models=["特斯拉 Model Y"])
    output = _output(
        case,
        recommended_models=["特斯拉 Model Y"],
        raw_assistant_content=json.dumps(
            {
                "answer": "暂无具体车型建议。",
                "mentioned_models": ["特斯拉 Model Y"],
            },
            ensure_ascii=False,
        ),
    )

    with pytest.raises(
        ValueError,
        match="terminal mentioned model is absent from answer",
    ):
        evaluator.evaluate_records(
            [case],
            [output],
            vehicle_catalog={"比亚迪 宋PLUS DM-i", "特斯拉 Model Y"},
        )


def test_catalog_matching_does_not_restore_short_or_numeric_bare_aliases():
    case = _case(allowed_models=[])
    answer = "预算7万元不代表车型，编号007和12也不是本次车型声明。"
    output = _output(
        case,
        recommended_models=[],
        raw_assistant_content=json.dumps(
            {"answer": answer, "mentioned_models": []},
            ensure_ascii=False,
        ),
    )
    report = evaluator.evaluate_records(
        [case],
        [output],
        vehicle_catalog={"极氪 007", "阿维塔 12", "哪吒 L"},
    )

    assert _metric(report, "declared_model_coverage")["denominator"] == 0
    assert _metric(report, "response_contract_consistency")["numerator"] == 1
    assert report["case_results"][0]["known_models_in_answer"] == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update({"extra": True}), "unexpected field extra"),
        (lambda row: row.pop("optional_tools"), "missing field optional_tools"),
        (lambda row: row.update({"intent": "sales_talk"}), "invalid intent"),
        (lambda row: row.update({"expected_tools": "bad"}), "expected_tools must be a list"),
        (
            lambda row: row["optional_tools"].append("generate_sales_talk"),
            "tool sets must be pairwise disjoint",
        ),
        (
            lambda row: row.update({"allowed_models": ["虚构牌 飞车"]}),
            "allowed_models contains model outside catalog",
        ),
    ],
)
def test_evaluator_rejects_strict_case_schema_before_scoring(mutation, message):
    case = _case()
    mutation(case)

    with pytest.raises(ValueError, match=message):
        evaluator.evaluate_records(
            [case],
            [_output(_case())],
            vehicle_catalog={"比亚迪 宋PLUS DM-i"},
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update({"extra": True}), "unexpected field extra"),
        (lambda row: row.pop("trajectory"), "missing field trajectory"),
        (lambda row: row.pop("runner_errors"), "missing field runner_errors"),
        (lambda row: row.update({"protocol_version": 2}), "protocol_version"),
        (lambda row: row.update({"model": ""}), "model must be a non-empty string"),
        (
            lambda row: row.update({"base_url": "http://user:key@local/v1"}),
            "base_url must not contain credentials",
        ),
        (lambda row: row.update({"case_digest": "wrong"}), "case_digest mismatch"),
        (lambda row: row.update({"schema_valid": "yes"}), "schema_valid must be boolean"),
        (
            lambda row: row.update({"runner_errors": "bad"}),
            "runner_errors must be a list",
        ),
        (
            lambda row: row.update({"trajectory": {}}),
            "trajectory must be a list",
        ),
        (
            lambda row: row.update({"recommended_models": [3]}),
            r"recommended_models\[0\]",
        ),
    ],
)
def test_evaluator_rejects_strict_output_schema_and_metadata(mutation, message):
    case = _case()
    output = _output(case)
    mutation(output)

    with pytest.raises(ValueError, match=message):
        evaluator.evaluate_records(
            [case],
            [output],
            vehicle_catalog={"比亚迪 宋PLUS DM-i"},
        )


def test_evaluator_accepts_typed_runner_failure_record_without_terminal():
    case = _case()
    output = _output(
        case,
        tool_calls=[],
        recommended_models=[],
        schema_valid=False,
        terminal_parse_error=None,
        runner_errors=["api_error: RuntimeError"],
        raw_assistant_content=None,
    )
    output["trajectory"] = output["trajectory"][:2]

    evaluator.validate_output_record(output, case)
    report = evaluator.evaluate_records(
        [case],
        [output],
        vehicle_catalog={"比亚迪 宋PLUS DM-i"},
    )

    assert _metric(report, "response_schema_validity")["numerator"] == 0
    assert _metric(report, "tool_selection_accuracy")["numerator"] == 0
    assert _metric(report, "argument_validity")["numerator"] == 0
    assert report["case_results"][0]["runner_errors"] == [
        "api_error: RuntimeError"
    ]


def test_evaluator_keeps_declared_hallucinations_from_failed_execution():
    case = _case(allowed_models=[])
    output = _output(
        case,
        recommended_models=["幻影 007"],
        schema_valid=True,
        terminal_parse_error=None,
        runner_errors=["tool_execution_error[call-1]: RuntimeError"],
        raw_assistant_content=json.dumps(
            {
                "answer": "可以考虑幻影 007。",
                "mentioned_models": ["幻影 007"],
            },
            ensure_ascii=False,
        ),
    )

    report = evaluator.evaluate_records(
        [case],
        [output],
        vehicle_catalog={"比亚迪 宋PLUS DM-i"},
    )

    assert report["protocol_gate"]["passed"] is False
    assert _metric(report, "response_schema_validity")["numerator"] == 1
    assert _metric(report, "tool_selection_accuracy")["numerator"] == 0
    assert _metric(report, "hallucination_rate")["numerator"] == 1
    assert _metric(report, "hallucination_rate")["denominator"] == 1


def test_evaluator_rejects_failed_record_without_any_failure_reason():
    case = _case()
    output = _output(
        case,
        tool_calls=[],
        recommended_models=[],
        schema_valid=False,
        terminal_parse_error=None,
        runner_errors=[],
        raw_assistant_content=None,
    )
    output["trajectory"] = output["trajectory"][:2]

    with pytest.raises(ValueError, match="must explain failed record"):
        evaluator.validate_output_record(output, case)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda output: output["trajectory"][2]["tool_calls"][1].update(
                {"id": output["trajectory"][2]["tool_calls"][0]["id"]}
            ),
            "duplicate assistant tool call id",
        ),
        (
            lambda output: output["trajectory"].pop(-2),
            "before the next assistant",
        ),
        (
            lambda output: output["trajectory"][3].update(
                {"tool_call_id": "not-current"}
            ),
            "does not reference an unconsumed current call",
        ),
        (
            lambda output: output["trajectory"].insert(
                -1, copy.deepcopy(output["trajectory"][3])
            ),
            "does not reference an unconsumed current call",
        ),
        (
            lambda output: output["trajectory"][2]["tool_calls"][0]["function"].update(
                {"arguments": '{"query":"tampered"}'}
            ),
            "flattened trajectory tool_calls differ",
        ),
    ],
)
def test_evaluator_enforces_strong_trajectory_state_machine(mutation, message):
    case = _case()
    output = _output(case)
    mutation(output)

    with pytest.raises(ValueError, match=message):
        evaluator.validate_output_record(output, case)


@pytest.mark.parametrize("side", ["cases", "outputs"])
def test_evaluator_rejects_duplicate_ids_on_both_sides(side):
    case = _case()
    cases = [case]
    outputs = [_output(case)]
    if side == "cases":
        cases.append(copy.deepcopy(case))
    else:
        outputs.append(copy.deepcopy(outputs[0]))

    with pytest.raises(ValueError, match=rf"duplicate {side} ID: case-1"):
        evaluator.evaluate_records(
            cases,
            outputs,
            vehicle_catalog={"比亚迪 宋PLUS DM-i"},
        )


@pytest.mark.parametrize(
    ("outputs", "message"),
    [
        ([], r"missing output IDs: \['case-1'\]"),
        (
            [_output(_case()), _output(_case("extra-1"))],
            r"extra output IDs: \['extra-1'\]",
        ),
        (
            [_output(_case("other-1"))],
            "missing output IDs.*extra output IDs",
        ),
    ],
)
def test_evaluator_rejects_missing_and_extra_ids(outputs, message):
    with pytest.raises(ValueError, match=message):
        evaluator.evaluate_records(
            [_case()],
            outputs,
            vehicle_catalog={"比亚迪 宋PLUS DM-i"},
        )


def test_all_metrics_have_auditable_counts_and_percentages():
    case = _case(allowed_models=[])
    output = _output(case, recommended_models=[])
    report = evaluator.evaluate_records([case], [output], vehicle_catalog=set())

    for metric in report["metrics"].values():
        assert set(metric) >= {
            "numerator",
            "denominator",
            "ratio",
            "percentage",
        }
        assert metric["percentage"] == metric["ratio"] * 100.0
    assert _metric(report, "recommendation_hit_rate")["denominator"] == 0
    assert _metric(report, "hallucination_rate")["denominator"] == 0


class _FakeCompletions:
    def __init__(self, messages):
        self._messages = iter(messages)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        message = next(self._messages)
        if isinstance(message, BaseException):
            raise message
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeClient:
    def __init__(self, messages):
        self.chat = SimpleNamespace(completions=_FakeCompletions(messages))


def _fake_call(name, *, call_id, arguments=None):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(
            name=name,
            arguments=json.dumps(
                arguments if arguments is not None else _tool_arguments(name),
                ensure_ascii=False,
            ),
        ),
    )


def _fake_message(content=None, calls=None):
    return SimpleNamespace(content=content, tool_calls=list(calls or []))


class _FakeTool:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def invoke(self, arguments):
        self.events.append((self.name, arguments))
        return {"tool": self.name, "ok": True}


def _fake_registry(events):
    names = {tool.name for tool in agent_graph.TOOLS}
    return {name: _FakeTool(name, events) for name in names}


def _terminal(answer="建议优先考虑比亚迪 宋PLUS DM-i。", models=None):
    return json.dumps(
        {
            "answer": answer,
            "mentioned_models": (
                ["比亚迪 宋PLUS DM-i"] if models is None else models
            ),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def test_runner_executes_real_multiturn_protocol_with_fake_tools(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "outputs.jsonl"
    case = _case()
    _write_jsonl(cases_path, [case])
    messages = [
        _fake_message(
            calls=[
                _fake_call("extract_user_profile", call_id="call-profile"),
                _fake_call("search_and_rank_vehicles", call_id="call-search"),
            ]
        ),
        _fake_message(
            calls=[
                _fake_call("retrieve_knowledge_base", call_id="call-rag"),
                _fake_call("generate_sales_talk", call_id="call-sales"),
            ]
        ),
        _fake_message(content=_terminal()),
    ]
    client = _FakeClient(messages)
    events = []
    ticks = iter([10.0, 10.025])

    summary = runner.run_evaluation(
        cases_path,
        output_path,
        model="test-model",
        base_url="http://local.invalid/v1",
        api_key="never-save-this-key",
        client=client,
        tool_registry=_fake_registry(events),
        clock=lambda: next(ticks),
        vehicle_database=VEHICLE_DATABASE,
    )

    assert summary == {
        "total_cases": 1,
        "existing": 0,
        "written": 1,
        "failed_cases": 0,
    }
    assert [name for name, _args in events] == [
        "extract_user_profile",
        "search_and_rank_vehicles",
        "retrieve_knowledge_base",
        "generate_sales_talk",
    ]
    requests = client.chat.completions.requests
    assert len(requests) == 3
    approved = {
        schema["function"]["name"]: schema for schema in build_tool_schemas()
    }
    assert requests[0]["tools"] == [
        approved[tool.name] for tool in agent_graph.TOOLS_BY_INTENT["recommend"]
    ]
    assert requests[0]["messages"][0]["content"].startswith(
        agent_graph._get_prompt_for_intent("recommend")
    )
    assert "mentioned_models" in requests[0]["messages"][0]["content"]
    assert requests[0]["messages"][1] == {
        "role": "user",
        "content": case["query"],
    }
    assert [message["role"] for message in requests[1]["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
    ]
    assert requests[1]["messages"][-2]["tool_call_id"] == "call-profile"
    assert requests[1]["messages"][-1]["tool_call_id"] == "call-search"

    record = evaluator.load_jsonl(output_path, label="outputs")[0]
    assert record["protocol_version"] == PROTOCOL_VERSION
    assert record["model"] == "test-model"
    assert record["base_url"] == "http://local.invalid/v1"
    assert record["case_digest"] == _case_digest(case)
    assert record["schema_valid"] is True
    assert record["terminal_parse_error"] is None
    assert record["runner_errors"] == []
    assert record["recommended_models"] == ["比亚迪 宋PLUS DM-i"]
    assert [call["function"]["name"] for call in record["tool_calls"]] == [
        "extract_user_profile",
        "search_and_rank_vehicles",
        "retrieve_knowledge_base",
        "generate_sales_talk",
    ]
    assert [message["role"] for message in record["trajectory"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]
    assert "never-save-this-key" not in output_path.read_text(encoding="utf-8")


def test_runner_uses_injected_tool_executor_instead_of_real_tools(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "outputs.jsonl"
    case = _case(intent="knowledge")
    _write_jsonl(cases_path, [case])
    client = _FakeClient(
        [
            _fake_message(
                calls=[_fake_call("retrieve_knowledge_base", call_id="call-rag")]
            ),
            _fake_message(content=_terminal()),
        ]
    )
    events = []

    def execute(name, arguments):
        events.append((name, arguments))
        return "fake result"

    runner.run_evaluation(
        cases_path,
        output_path,
        model="test-model",
        client=client,
        tool_executor=execute,
        clock=lambda: 1.0,
        vehicle_database=VEHICLE_DATABASE,
    )

    assert events == [
        ("retrieve_knowledge_base", _tool_arguments("retrieve_knowledge_base"))
    ]


def test_runner_can_use_exact_frozen_production_prompt_without_eval_suffix(
    tmp_path,
):
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "outputs.jsonl"
    case = _case(intent="knowledge", allowed_models=[])
    _write_jsonl(cases_path, [case])
    production_prompt = agent_graph._get_prompt_for_intent("knowledge")
    client = _FakeClient([_fake_message(content="这是自然语言知识终答。")])

    runner.run_evaluation(
        cases_path,
        output_path,
        model="test-model",
        client=client,
        tool_registry=_fake_registry([]),
        clock=lambda: 1.0,
        vehicle_database=VEHICLE_DATABASE,
        system_prompts_by_intent={"knowledge": production_prompt},
        append_evaluation_terminal_instruction=False,
    )

    request_prompt = client.chat.completions.requests[0]["messages"][0][
        "content"
    ]
    assert request_prompt == production_prompt
    assert "评测终态协议" not in request_prompt
    assert "mentioned_models" not in request_prompt
    record = evaluator.load_jsonl(output_path, label="outputs")[0]
    assert record["raw_assistant_content"] == "这是自然语言知识终答。"
    assert record["schema_valid"] is False


def test_runner_preserves_declared_unknown_models_without_catalog_extraction(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "outputs.jsonl"
    case = _case()
    _write_jsonl(cases_path, [case])
    client = _FakeClient(
        [
            _fake_message(
                content=_terminal(
                    answer="幻影 007适合你；预算里的数字7不是另一款车。",
                    models=["幻影 007"],
                )
            )
        ]
    )

    runner.run_evaluation(
        cases_path,
        output_path,
        model="test-model",
        client=client,
        tool_registry=_fake_registry([]),
        clock=lambda: 1.0,
        vehicle_database=VEHICLE_DATABASE,
    )

    record = evaluator.load_jsonl(output_path, label="outputs")[0]
    assert record["schema_valid"] is True
    assert record["recommended_models"] == ["幻影 007"]


def test_runner_rejects_declared_model_absent_from_terminal_answer(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "outputs.jsonl"
    case = _case(allowed_models=["特斯拉 Model Y"])
    _write_jsonl(cases_path, [case])
    client = _FakeClient(
        [
            _fake_message(
                content=_terminal(
                    answer="暂无具体车型建议。",
                    models=["特斯拉 Model Y"],
                )
            )
        ]
    )

    summary = runner.run_evaluation(
        cases_path,
        output_path,
        model="test-model",
        client=client,
        tool_registry=_fake_registry([]),
        clock=lambda: 1.0,
        vehicle_database=VEHICLE_DATABASE,
    )

    assert summary["failed_cases"] == 1
    record = evaluator.load_jsonl(output_path, label="outputs")[0]
    assert record["schema_valid"] is False
    assert record["recommended_models"] == []
    assert "declared models are absent from answer" in record["terminal_parse_error"]


def test_runner_records_terminal_protocol_failure_without_inventing_models(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "outputs.jsonl"
    case = _case()
    _write_jsonl(cases_path, [case])
    client = _FakeClient(
        [
            _fake_message(
                content='{"answer":"建议幻影 007","mentioned_models":"幻影 007"}'
            )
        ]
    )

    runner.run_evaluation(
        cases_path,
        output_path,
        model="test-model",
        client=client,
        tool_registry=_fake_registry([]),
        clock=lambda: 1.0,
        vehicle_database=VEHICLE_DATABASE,
    )

    record = evaluator.load_jsonl(output_path, label="outputs")[0]
    assert record["schema_valid"] is False
    assert "protocol failure" in record["terminal_parse_error"]
    assert "mentioned_models" in record["terminal_parse_error"]
    assert record["recommended_models"] == []


@pytest.mark.parametrize(
    ("intent", "call", "stored", "message"),
    [
        (
            "compare",
            _fake_call("generate_sales_talk", call_id="bad-tool"),
            True,
            "tool generate_sales_talk is not allowed for intent compare",
        ),
        (
            "knowledge",
            _fake_call("retrieve_knowledge_base", call_id=None),
            False,
            "non-empty tool call id",
        ),
        (
            "knowledge",
            SimpleNamespace(
                id="bad-args",
                type="function",
                function=SimpleNamespace(
                    name="retrieve_knowledge_base",
                    arguments="[]",
                ),
            ),
            True,
            "arguments must decode to an object",
        ),
        (
            "knowledge",
            SimpleNamespace(
                id="bad-json",
                type="function",
                function=SimpleNamespace(
                    name="retrieve_knowledge_base",
                    arguments="{bad",
                ),
            ),
            True,
            "arguments must be valid JSON",
        ),
        (
            "knowledge",
            _fake_call("retrieve_knowledge_base", call_id="missing", arguments={}),
            True,
            "missing required argument: query",
        ),
        (
            "knowledge",
            _fake_call(
                "retrieve_knowledge_base",
                call_id="extra",
                arguments={"query": "补能", "extra": True},
            ),
            True,
            "unexpected argument: extra",
        ),
        (
            "knowledge",
            _fake_call(
                "retrieve_knowledge_base",
                call_id="wrong-type",
                arguments={"query": 7},
            ),
            True,
            "argument query must be string",
        ),
    ],
)
def test_runner_persists_invalid_tool_calls_without_executing_them(
    tmp_path, intent, call, stored, message
):
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "outputs.jsonl"
    _write_jsonl(cases_path, [_case(intent=intent)])
    client = _FakeClient(
        [_fake_message(calls=[call]), _fake_message(content=_terminal())]
    )
    events = []

    summary = runner.run_evaluation(
        cases_path,
        output_path,
        model="test-model",
        client=client,
        tool_registry=_fake_registry(events),
        clock=lambda: 1.0,
        vehicle_database=VEHICLE_DATABASE,
    )

    assert summary["failed_cases"] == 1
    assert events == []
    record = evaluator.load_jsonl(output_path, label="outputs")[0]
    assert record["schema_valid"] is stored
    assert any(message in error for error in record["runner_errors"])
    assert len(record["tool_calls"]) == int(stored)
    if stored:
        assert [item["role"] for item in record["trajectory"]][-2:] == [
            "tool",
            "assistant",
        ]
        assert "invalid_tool_call" in record["trajectory"][-2]["content"]
    else:
        assert [item["role"] for item in record["trajectory"]] == [
            "system",
            "user",
        ]


def test_runner_persists_api_failure_keeps_prior_trace_and_continues(
    tmp_path, monkeypatch
):
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "outputs.jsonl"
    cases = [_case("case-1", intent="knowledge"), _case("case-2")]
    _write_jsonl(cases_path, cases)
    client = _FakeClient(
        [
            _fake_message(
                calls=[_fake_call("retrieve_knowledge_base", call_id="prior-call")]
            ),
            RuntimeError("never-persist-api-key-secret"),
            _fake_message(content=_terminal()),
        ]
    )
    replace_calls = []
    real_replace = runner.os.replace

    def spy_replace(source, destination):
        replace_calls.append((Path(source), Path(destination)))
        return real_replace(source, destination)

    monkeypatch.setattr(runner.os, "replace", spy_replace)
    summary = runner.run_evaluation(
        cases_path,
        output_path,
        model="test-model",
        client=client,
        tool_registry=_fake_registry([]),
        clock=lambda: 1.0,
        vehicle_database=VEHICLE_DATABASE,
    )

    assert summary == {
        "total_cases": 2,
        "existing": 0,
        "written": 2,
        "failed_cases": 1,
    }
    assert len(replace_calls) == 2
    records = evaluator.load_jsonl(output_path, label="outputs")
    assert records[0]["runner_errors"] == ["api_error: RuntimeError"]
    assert [item["role"] for item in records[0]["trajectory"]] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert [call["id"] for call in records[0]["tool_calls"]] == ["prior-call"]
    assert records[1]["schema_valid"] is True
    assert "never-persist-api-key-secret" not in output_path.read_text(
        encoding="utf-8"
    )


def test_runner_persists_broken_tool_call_and_continues_next_case(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "outputs.jsonl"
    cases = [_case("case-1", intent="knowledge"), _case("case-2")]
    _write_jsonl(cases_path, cases)
    broken = SimpleNamespace(
        id="broken-call",
        type="function",
        function=SimpleNamespace(arguments='{"query":"missing name"}'),
    )
    client = _FakeClient(
        [
            _fake_message(calls=[broken]),
            _fake_message(content=_terminal()),
        ]
    )

    summary = runner.run_evaluation(
        cases_path,
        output_path,
        model="test-model",
        client=client,
        tool_registry=_fake_registry([]),
        clock=lambda: 1.0,
        vehicle_database=VEHICLE_DATABASE,
    )

    assert summary["written"] == 2
    assert summary["failed_cases"] == 1
    records = evaluator.load_jsonl(output_path, label="outputs")
    assert records[0]["tool_calls"] == []
    assert records[0]["runner_errors"] == [
        "tool_call_structure_error: assistant tool_calls[0] requires a "
        "non-empty function name"
    ]
    assert [item["role"] for item in records[0]["trajectory"]] == [
        "system",
        "user",
    ]
    assert records[1]["schema_valid"] is True


def test_runner_persists_tool_exception_and_returns_deterministic_error(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "outputs.jsonl"
    case = _case(intent="knowledge")
    _write_jsonl(cases_path, [case])
    client = _FakeClient(
        [
            _fake_message(
                calls=[_fake_call("retrieve_knowledge_base", call_id="tool-fails")]
            ),
            _fake_message(
                content=_terminal(
                    answer="可以考虑幻影 007。",
                    models=["幻影 007"],
                )
            ),
        ]
    )

    def execute(_name, _arguments):
        raise RuntimeError("never-persist-tool-secret")

    summary = runner.run_evaluation(
        cases_path,
        output_path,
        model="test-model",
        client=client,
        tool_executor=execute,
        clock=lambda: 1.0,
        vehicle_database=VEHICLE_DATABASE,
    )

    assert summary["failed_cases"] == 1
    record = evaluator.load_jsonl(output_path, label="outputs")[0]
    assert record["runner_errors"] == [
        "tool_execution_error[tool-fails]: RuntimeError"
    ]
    assert record["schema_valid"] is True
    assert record["recommended_models"] == ["幻影 007"]
    assert "tool_execution_failed" in record["trajectory"][-2]["content"]
    assert "never-persist-tool-secret" not in output_path.read_text(
        encoding="utf-8"
    )
    report = evaluator.evaluate_records(
        [case],
        [record],
        vehicle_catalog=set(evaluator.load_vehicle_catalog(VEHICLE_DATABASE)),
    )
    assert report["protocol_gate"]["passed"] is False
    assert _metric(report, "hallucination_rate")["numerator"] == 1
    assert _metric(report, "hallucination_rate")["denominator"] == 1


def test_runner_persists_tool_result_serialization_error_and_continues(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "outputs.jsonl"
    cases = [_case("case-1", intent="knowledge"), _case("case-2")]
    _write_jsonl(cases_path, cases)
    client = _FakeClient(
        [
            _fake_message(
                calls=[
                    _fake_call(
                        "retrieve_knowledge_base",
                        call_id="circular-result",
                    )
                ]
            ),
            _fake_message(content=_terminal()),
            _fake_message(content=_terminal()),
        ]
    )

    def execute(_name, _arguments):
        result = {}
        result["self"] = result
        return result

    summary = runner.run_evaluation(
        cases_path,
        output_path,
        model="test-model",
        client=client,
        tool_executor=execute,
        clock=lambda: 1.0,
        vehicle_database=VEHICLE_DATABASE,
    )

    assert summary == {
        "total_cases": 2,
        "existing": 0,
        "written": 2,
        "failed_cases": 1,
    }
    records = evaluator.load_jsonl(output_path, label="outputs")
    assert records[0]["runner_errors"] == [
        "tool_result_serialization_error[circular-result]: ValueError"
    ]
    assert records[0]["schema_valid"] is True
    assert "tool_result_serialization_failed" in records[0]["trajectory"][-2]["content"]
    assert records[1]["schema_valid"] is True


def test_runner_persists_max_steps_with_closed_tool_batch(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "outputs.jsonl"
    case = _case(intent="knowledge")
    _write_jsonl(cases_path, [case])

    summary = runner.run_evaluation(
        cases_path,
        output_path,
        model="test-model",
        client=_FakeClient(
            [
                _fake_message(
                    calls=[
                        _fake_call(
                            "retrieve_knowledge_base", call_id="last-tool-call"
                        )
                    ]
                )
            ]
        ),
        tool_registry=_fake_registry([]),
        clock=lambda: 1.0,
        vehicle_database=VEHICLE_DATABASE,
        max_steps=1,
    )

    assert summary["failed_cases"] == 1
    record = evaluator.load_jsonl(output_path, label="outputs")[0]
    assert record["runner_errors"] == ["max_steps_exceeded: max_steps=1"]
    assert record["terminal_parse_error"] == (
        "protocol failure: max_steps=1 reached before terminal response"
    )
    assert [item["role"] for item in record["trajectory"]][-2:] == [
        "assistant",
        "tool",
    ]
    evaluator.validate_output_record(record, case)


def test_runner_does_not_swallow_keyboard_interrupt(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "outputs.jsonl"
    _write_jsonl(cases_path, [_case()])

    with pytest.raises(KeyboardInterrupt):
        runner.run_evaluation(
            cases_path,
            output_path,
            model="test-model",
            client=_FakeClient([KeyboardInterrupt()]),
            tool_registry=_fake_registry([]),
            clock=lambda: 1.0,
            vehicle_database=VEHICLE_DATABASE,
        )
    assert not output_path.exists()


def test_runner_rejects_cross_model_resume_and_case_digest_change(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "outputs.jsonl"
    case = _case()
    _write_jsonl(cases_path, [case])
    runner.run_evaluation(
        cases_path,
        output_path,
        model="model-a",
        client=_FakeClient([_fake_message(content=_terminal())]),
        tool_registry=_fake_registry([]),
        clock=lambda: 1.0,
        vehicle_database=VEHICLE_DATABASE,
    )

    with pytest.raises(ValueError, match="model mismatch"):
        runner.run_evaluation(
            cases_path,
            output_path,
            model="model-b",
            client=_FakeClient([]),
            tool_registry=_fake_registry([]),
            vehicle_database=VEHICLE_DATABASE,
        )

    changed = copy.deepcopy(case)
    changed["query"] += "，并优先考虑后备箱"
    _write_jsonl(cases_path, [changed])
    with pytest.raises(ValueError, match="case_digest mismatch"):
        runner.run_evaluation(
            cases_path,
            output_path,
            model="model-a",
            client=_FakeClient([]),
            tool_registry=_fake_registry([]),
            vehicle_database=VEHICLE_DATABASE,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda row: row.update({"protocol_version": "task14.tampered"}),
            "protocol_version",
        ),
        (
            lambda row: row.update({"base_url": "http://other.invalid/v1"}),
            "base_url mismatch",
        ),
        (
            lambda row: row["trajectory"].pop(3),
            "before the next assistant",
        ),
        (
            lambda row: row["tool_calls"][0]["function"].update(
                {"arguments": '{"query":"tampered"}'}
            ),
            "flattened trajectory tool_calls differ",
        ),
    ],
)
def test_runner_resume_rejects_protocol_base_url_and_trajectory_tampering(
    tmp_path, mutation, message
):
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "outputs.jsonl"
    case = _case()
    _write_jsonl(cases_path, [case])
    record = _output(case)
    mutation(record)
    _write_jsonl(output_path, [record])

    with pytest.raises(ValueError, match=message):
        runner.run_evaluation(
            cases_path,
            output_path,
            model="test-model",
            base_url="http://local.invalid/v1",
            client=_FakeClient([]),
            tool_registry=_fake_registry([]),
            vehicle_database=VEHICLE_DATABASE,
        )


def test_runner_rejects_malformed_existing_output_schema(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "outputs.jsonl"
    case = _case()
    _write_jsonl(cases_path, [case])
    malformed = _output(case)
    malformed.pop("trajectory")
    _write_jsonl(output_path, [malformed])

    with pytest.raises(ValueError, match="missing field trajectory"):
        runner.run_evaluation(
            cases_path,
            output_path,
            model="test-model",
            client=_FakeClient([]),
            tool_registry=_fake_registry([]),
            vehicle_database=VEHICLE_DATABASE,
        )


def test_runner_replaces_atomic_snapshot_after_each_case_and_preserves_source(
    tmp_path, monkeypatch
):
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "outputs.jsonl"
    cases = [_case("case-1"), _case("case-2")]
    _write_jsonl(cases_path, cases)
    source_before = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    client = _FakeClient(
        [
            _fake_message(content=_terminal()),
            _fake_message(content=_terminal()),
        ]
    )
    replace_calls = []
    real_replace = runner.os.replace

    def spy_replace(source, destination):
        replace_calls.append((Path(source), Path(destination)))
        return real_replace(source, destination)

    monkeypatch.setattr(runner.os, "replace", spy_replace)
    summary = runner.run_evaluation(
        cases_path,
        output_path,
        model="test-model",
        client=client,
        tool_registry=_fake_registry([]),
        clock=lambda: 1.0,
        vehicle_database=VEHICLE_DATABASE,
    )

    assert summary == {
        "total_cases": 2,
        "existing": 0,
        "written": 2,
        "failed_cases": 0,
    }
    assert len(replace_calls) == 2
    assert all(source.parent == output_path.parent for source, _ in replace_calls)
    assert all(destination == output_path for _, destination in replace_calls)
    assert hashlib.sha256(cases_path.read_bytes()).hexdigest() == source_before
    assert [row["id"] for row in evaluator.load_jsonl(output_path, label="outputs")] == [
        "case-1",
        "case-2",
    ]
    assert not list(tmp_path.glob("*.tmp"))


def test_runner_sanitizes_persisted_base_url_and_client_factory_is_injectable(
    tmp_path,
):
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "outputs.jsonl"
    _write_jsonl(cases_path, [_case()])
    client = _FakeClient([_fake_message(content=_terminal())])
    received = {}

    def factory(**kwargs):
        received.update(kwargs)
        return client

    runner.run_evaluation(
        cases_path,
        output_path,
        model="test-model",
        base_url="http://user:secret@local.invalid/v1?api_key=secret",
        api_key="another-secret",
        client_factory=factory,
        tool_registry=_fake_registry([]),
        clock=lambda: 1.0,
        vehicle_database=VEHICLE_DATABASE,
    )

    assert received == {
        "base_url": "http://user:secret@local.invalid/v1?api_key=secret",
        "api_key": "another-secret",
    }
    rendered = output_path.read_text(encoding="utf-8")
    assert "secret" not in rendered
    assert "another-secret" not in rendered
    assert evaluator.load_jsonl(output_path, label="outputs")[0]["base_url"] == (
        "http://local.invalid/v1"
    )
