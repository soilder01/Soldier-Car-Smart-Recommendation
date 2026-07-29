import hashlib
import json

from app.services import agent_graph
from scripts import run_production_prompt_v3_eval as v3
from scripts import score_production_prompt_v3 as scorer


def test_v3_prompt_and_harness_were_frozen_before_inference():
    harness, prompts = v3.validate_frozen_v3(verify_model_files=False)

    assert v3.sha256_file(v3.PROMPTS_PATH) == v3.EXPECTED_PROMPTS_SHA256
    assert v3.sha256_file(v3.HARNESS_PATH) == v3.EXPECTED_HARNESS_SHA256
    assert harness["status"] == "frozen_before_inference"
    assert prompts["status"] == "frozen_before_inference"
    assert harness["prompt"] == {
        "artifact_path": "data/model_training/eval/production_prompts_v3.json",
        "artifact_sha256": v3.EXPECTED_PROMPTS_SHA256,
        "mode": "exact_frozen_production_prompt",
        "append_evaluator_terminal_instruction": False,
    }


def test_v3_uses_exact_backend_production_prompts_without_legacy_suffix():
    _harness, prompts = v3.validate_frozen_v3(verify_model_files=False)

    for intent in ("recommend", "compare", "knowledge", "sales"):
        frozen = prompts["intents"][intent]
        assert frozen["text"] == agent_graph._get_prompt_for_intent(intent)
        assert frozen["utf8_sha256"] == hashlib.sha256(
            frozen["text"].encode("utf-8")
        ).hexdigest()
        assert "评测终态协议" not in frozen["text"]
        assert "mentioned_models" not in frozen["text"]


def test_v3_copies_all_non_prompt_inference_settings_from_parent():
    harness, _prompts = v3.validate_frozen_v3(verify_model_files=False)
    parent = v3.load_json(v3.PARENT_HARNESS_PATH)

    for field in ("evaluation", "runner", "generation", "tool_protocol"):
        assert harness[field] == parent[field]
    assert harness["scoring"]["rules_manifest_sha256"] == (
        "49d3a2b1da23490b236f839f73d55654743a596899909fa80b4fcc090d721113"
    )


def test_v3_model_modes_have_new_non_overlapping_output_paths():
    harness, _prompts = v3.validate_frozen_v3(verify_model_files=False)

    baseline = v3.campaign_paths(harness, "baseline")
    sft = v3.campaign_paths(harness, "sft_epoch_3")

    assert baseline.output_path != sft.output_path
    assert not baseline.output_path.name.startswith("baseline_qwen_heldout")
    assert not sft.output_path.name.startswith("sft_heldout_outputs")
    assert baseline.receipt_path != sft.receipt_path


def test_v3_scorer_marks_overall_and_per_intent_regressions():
    good = {
        name: {"numerator": 10, "denominator": 10, "percentage": 100.0}
        for name in v3_score_metric_names()
    }
    lower = {
        name: dict(metric)
        for name, metric in good.items()
    }
    lower["normal_stop_rate"] = {
        "numerator": 8,
        "denominator": 10,
        "percentage": 80.0,
    }
    baseline = {
        "metrics": good,
        "per_intent": {"knowledge": good},
    }
    sft = {
        "metrics": lower,
        "per_intent": {"knowledge": lower},
    }

    _overall, _per_intent, regressions = scorer._compare_all(baseline, sft)

    assert regressions == [
        {
            "scope": "overall",
            "metric": "normal_stop_rate",
            "absolute_percentage_point_change": -20.0,
            "severity": "RED_REGRESSION",
        },
        {
            "scope": "intent:knowledge",
            "metric": "normal_stop_rate",
            "absolute_percentage_point_change": -20.0,
            "severity": "RED_REGRESSION",
        },
    ]


def v3_score_metric_names():
    return (
        "normal_stop_rate",
        "mandatory_tool_order_accuracy",
        "tool_allowlist_accuracy",
        "argument_schema_accuracy",
        "argument_schema_call_accuracy",
        "v2_composite_pass_rate",
    )


def test_completed_v3_report_uses_frozen_v2_and_preserves_regressions():
    report_path = (
        v3.ROOT
        / "data"
        / "model_training"
        / "production_prompt_v3_product_faithful_v2_report.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["v3_harness"]["sha256"] == v3.EXPECTED_HARNESS_SHA256
    assert report["production_prompts"]["sha256"] == (
        v3.EXPECTED_PROMPTS_SHA256
    )
    assert report["scoring"]["manifest_sha256"] == (
        v3.EXPECTED_V2_MANIFEST_SHA256
    )
    assert report["baseline"]["metrics"]["v2_composite_pass_rate"][
        "numerator"
    ] == 17
    assert report["sft"]["metrics"]["v2_composite_pass_rate"]["numerator"] == 40
    assert report["knowledge_stop_focus"]["baseline"]["numerator"] == 10
    assert report["knowledge_stop_focus"]["sft"]["numerator"] == 10
    assert report["honesty_self_check"]["sft_regressions"] == []
