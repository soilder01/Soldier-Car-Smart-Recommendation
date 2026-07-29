import copy
from pathlib import Path

from scripts import evaluate_model_outputs as legacy_evaluator
from scripts import rescore_product_faithful_v2 as v2


def _knowledge_case() -> dict:
    return {
        "id": "heldout-knowledge-test",
        "query": "解释动力电池质保边界",
        "intent": "knowledge",
        "expected_tools": ["retrieve_knowledge_base"],
        "optional_tools": ["search_web_info"],
        "forbidden_tools": [
            "extract_user_profile",
            "search_and_rank_vehicles",
            "generate_sales_talk",
        ],
        "allowed_models": [],
    }


def _call(
    name: str = "retrieve_knowledge_base",
    arguments: str = '{"query":"动力电池质保"}',
) -> dict:
    return {
        "id": "call-1",
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def _terminal_output(case: dict, content: str) -> dict:
    call = _call()
    return {
        "id": case["id"],
        "protocol_version": legacy_evaluator.PROTOCOL_VERSION,
        "model": "fixture-model",
        "base_url": "https://local-transformers.invalid/v1",
        "case_digest": legacy_evaluator.case_digest(case),
        "schema_valid": False,
        "terminal_parse_error": "legacy strict JSON protocol failure",
        "runner_errors": [],
        "raw_assistant_content": content,
        "tool_calls": [call],
        "latency_ms": 1.0,
        "recommended_models": [],
        "trajectory": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": case["query"]},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [call],
            },
            {
                "role": "tool",
                "content": "evidence",
                "tool_call_id": "call-1",
            },
            {"role": "assistant", "content": content},
        ],
    }


def test_frozen_v2_manifest_hash_is_recorded_before_scoring():
    manifest, manifest_sha = v2.load_frozen_manifest()

    assert manifest["status"] == "frozen_before_scoring"
    assert manifest_sha == v2.EXPECTED_MANIFEST_SHA256
    assert manifest["scoring_rules"]["v2_composite"]["rule"].startswith(
        "Pass if and only if"
    )
    assert "model-generated mentioned_models" in manifest[
        "explicitly_not_scored"
    ]


def test_answer_resolution_accepts_direct_text_and_legacy_json_wrapper():
    assert v2.resolve_terminal_answer("自然语言终答") == (
        "自然语言终答",
        "raw_assistant_content",
    )
    assert v2.resolve_terminal_answer(
        '{"answer":"旧包装中的终答","mentioned_models":[]}'
    ) == ("旧包装中的终答", "legacy_json_answer")
    assert v2.resolve_terminal_answer(None) == (None, "missing")


def test_case_passes_only_when_all_product_execution_gates_pass():
    case = _knowledge_case()
    output = _terminal_output(case, "这是自然语言终答。")

    result = v2.score_output_set(
        label="fixture",
        cases=[case],
        outputs=[output],
        vehicle_catalog=set(),
        schemas=v2.schema_map(),
        max_steps=8,
    )

    row = result["cases"][0]
    assert row["gates"] == {
        "normal_stop": True,
        "mandatory_tool_order": True,
        "tool_allowlist": True,
        "argument_schema": True,
    }
    assert row["v2_passed"] is True
    assert result["metrics"]["v2_composite_pass_rate"]["numerator"] == 1

    forbidden = copy.deepcopy(output)
    forbidden_call = _call(name="generate_sales_talk", arguments="{}")
    forbidden["tool_calls"].append(forbidden_call)
    forbidden["trajectory"].insert(
        -1,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [forbidden_call],
        },
    )
    forbidden["trajectory"].insert(
        -1,
        {
            "role": "tool",
            "content": "talk",
            "tool_call_id": "call-1",
        },
    )
    forbidden["tool_calls"][1]["id"] = "call-2"
    forbidden["trajectory"][-3]["tool_calls"][0]["id"] = "call-2"
    forbidden["trajectory"][-2]["tool_call_id"] = "call-2"

    failed = v2.score_output_set(
        label="fixture",
        cases=[case],
        outputs=[forbidden],
        vehicle_catalog=set(),
        schemas=v2.schema_map(),
        max_steps=8,
    )["cases"][0]

    assert failed["gates"]["normal_stop"] is True
    assert failed["gates"]["tool_allowlist"] is False
    assert failed["v2_passed"] is False


def test_argument_schema_validates_array_items_not_only_top_level_type():
    schemas = v2.schema_map()
    call = _call(
        name="search_and_rank_vehicles",
        arguments='{"model_names":["小鹏G6",7]}',
    )

    valid, failure = v2.validate_tool_call_schema(call, schemas)

    assert valid is False
    assert failure == "arguments.model_names[1] must be string"


def test_report_writer_refuses_to_overwrite_existing_artifact(tmp_path: Path):
    report_path = tmp_path / "report.json"
    report_path.write_text("{}\n", encoding="utf-8")

    try:
        v2.write_new_json(report_path, {"status": "completed"})
    except FileExistsError as error:
        assert str(report_path) in str(error)
    else:
        raise AssertionError("existing report must not be overwritten")


def test_frozen_outputs_rescore_to_expected_symmetric_metrics():
    manifest, _manifest_sha = v2.load_frozen_manifest()
    inputs = manifest["frozen_inputs"]
    cases = legacy_evaluator.load_jsonl(
        v2.ROOT / inputs["cases"]["path"],
        label="cases",
    )
    catalog = set(legacy_evaluator.load_vehicle_catalog())
    summaries = {}
    for label, input_name in (
        ("baseline", "baseline_outputs"),
        ("sft", "sft_outputs"),
    ):
        outputs = legacy_evaluator.load_jsonl(
            v2.ROOT / inputs[input_name]["path"],
            label=label,
        )
        summaries[label] = v2.score_output_set(
            label=label,
            cases=cases,
            outputs=outputs,
            vehicle_catalog=catalog,
            schemas=v2.schema_map(),
            max_steps=manifest["runner_context"]["max_steps"],
        )

    assert summaries["baseline"]["metrics"]["normal_stop_rate"][
        "numerator"
    ] == 34
    assert summaries["sft"]["metrics"]["normal_stop_rate"]["numerator"] == 29
    assert summaries["baseline"]["metrics"]["v2_composite_pass_rate"][
        "numerator"
    ] == 18
    assert summaries["sft"]["metrics"]["v2_composite_pass_rate"][
        "numerator"
    ] == 29
    assert summaries["baseline"]["metrics"]["argument_schema_call_accuracy"][
        "numerator"
    ] == 133
    assert summaries["baseline"]["metrics"]["argument_schema_call_accuracy"][
        "denominator"
    ] == 147
    assert summaries["sft"]["metrics"]["argument_schema_call_accuracy"][
        "numerator"
    ] == 197
    assert summaries["sft"]["metrics"]["argument_schema_call_accuracy"][
        "denominator"
    ] == 197
    assert summaries["baseline"]["per_intent"]["knowledge"][
        "v2_composite_pass_rate"
    ]["numerator"] == 9
    assert summaries["sft"]["per_intent"]["knowledge"][
        "v2_composite_pass_rate"
    ]["numerator"] == 2
