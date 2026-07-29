import json
from pathlib import Path

import pytest

from scripts import run_local_qwen_baseline as baseline


def _case(case_id: str, intent: str, query: str) -> dict:
    return {
        "id": case_id,
        "intent": intent,
        "query": query,
        "expected_tools": ["retrieve_knowledge_base"],
        "optional_tools": [],
        "forbidden_tools": [],
        "allowed_models": ["问界 M5"],
    }


def test_parse_qwen_tool_calls_converts_xml_to_openai_calls():
    text = (
        "<tool_call>\n"
        '{"name":"retrieve_knowledge_base","arguments":{"query":"问界M5"}}'
        "\n</tool_call>"
    )

    content, calls = baseline.parse_qwen_tool_calls(
        text,
        call_id_factory=lambda index: f"local-call-{index}",
    )

    assert content == ""
    assert calls == [
        {
            "id": "local-call-0",
            "type": "function",
            "function": {
                "name": "retrieve_knowledge_base",
                "arguments": '{"query":"问界M5"}',
            },
        }
    ]


def test_parse_qwen_tool_calls_fails_closed_for_invalid_payload():
    text = "<tool_call>{bad-json}</tool_call>"

    content, calls = baseline.parse_qwen_tool_calls(
        text,
        call_id_factory=lambda index: f"local-call-{index}",
    )

    assert content == text
    assert calls == []


def test_tool_call_boundary_stopper_stops_only_after_closed_xml():
    class FakeTokenizer:
        def decode(self, values, **kwargs):
            kwargs.clear()
            return "".join(values)

    stopper = baseline.ToolCallBoundaryStopper(
        tokenizer=FakeTokenizer(),
        prompt_token_count=2,
    )

    assert stopper([["a", "b", "<tool_call>", "x"]], None) is False
    assert stopper([["a", "b", "<tool_call>", "x", "</tool_call>"]], None) is True


def test_build_baseline_summary_requires_contract_catalog_and_grounding():
    cases = [
        _case("heldout-recommend-001", "recommend", "问题一"),
        _case("heldout-sales-001", "sales", "问题二"),
    ]
    evaluation = {
        "metrics": {
            "tool_selection_accuracy": {"numerator": 1, "denominator": 2},
        },
        "case_results": [
            {
                "id": "heldout-recommend-001",
                "case_success": True,
                "failures": [],
                "recommendation_hit": True,
                "hallucinated_models": [],
                "actual_tools": ["retrieve_knowledge_base"],
                "valid_argument_calls": 1,
                "total_tool_calls": 1,
            },
            {
                "id": "heldout-sales-001",
                "case_success": True,
                "failures": [],
                "recommendation_hit": True,
                "hallucinated_models": [],
                "actual_tools": ["retrieve_knowledge_base"],
                "valid_argument_calls": 1,
                "total_tool_calls": 1,
            },
        ],
    }

    summary = baseline.build_baseline_summary(
        cases=cases,
        evaluation=evaluation,
        grounding_by_id={
            "heldout-recommend-001": [],
            "heldout-sales-001": ["unsupported hard claim: 900km"],
        },
        raw_outputs_path=Path("outputs.jsonl"),
    )

    assert summary["total_score"] == {
        "name": "contract_grounding_pass_rate",
        "numerator": 1,
        "denominator": 2,
        "percentage": 50.0,
    }
    assert summary["per_intent"]["recommend"]["passed"] == 1
    assert summary["per_intent"]["sales"]["passed"] == 0
    assert summary["cases"][1]["passed"] is False
    assert "grounding: unsupported hard claim: 900km" in summary["cases"][1][
        "failure_reasons"
    ]


def test_validate_heldout_only_rejects_reward_visible_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    held_out = tmp_path / "held_out.jsonl"
    reward_visible = tmp_path / "reward_visible.jsonl"
    held_out.write_text(
        json.dumps(_case("heldout-recommend-001", "recommend", "相同问题"))
        + "\n",
        encoding="utf-8",
    )
    reward_visible.write_text(
        json.dumps(_case("reward-recommend-001", "recommend", "相同问题"))
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(baseline, "HELD_OUT_PATH", held_out)
    monkeypatch.setattr(baseline, "REWARD_VISIBLE_PATH", reward_visible)

    with pytest.raises(ValueError, match="held-out/reward-visible query overlap"):
        baseline.validate_heldout_only(held_out)


def test_frozen_harness_manifest_locks_cases_decode_and_scoring_contract():
    contract = baseline.build_frozen_harness_contract()

    assert contract["harness_version"] == "qwen-heldout-contract-v1"
    assert contract["evaluation"]["cases"] == (
        "data/model_training/eval/held_out.jsonl"
    )
    assert contract["evaluation"]["case_count"] == 40
    assert contract["evaluation"]["protocol_version"] == "task14.v3"
    assert contract["runner"] == {"max_steps": 8}
    assert contract["generation"] == {
        "max_new_tokens": 512,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "tool_call_stop_sequence": "</tool_call>",
    }
    assert contract["scoring"]["name"] == "contract_grounding_pass_rate"
    assert contract["scoring"]["required_checks"] == [
        "terminal_protocol",
        "mandatory_tool_order",
        "tool_argument_schema",
        "allowed_catalog_models",
        "no_declared_hallucinated_models",
        "grounding_audit",
    ]
    baseline.validate_frozen_harness_manifest()


def test_frozen_harness_rejects_decode_or_runner_drift():
    baseline.validate_frozen_runtime_settings(
        max_steps=8,
        max_new_tokens=512,
    )

    with pytest.raises(ValueError, match="max_steps"):
        baseline.validate_frozen_runtime_settings(
            max_steps=7,
            max_new_tokens=512,
        )
    with pytest.raises(ValueError, match="max_new_tokens"):
        baseline.validate_frozen_runtime_settings(
            max_steps=8,
            max_new_tokens=256,
        )


def test_local_client_generation_is_greedy_and_uses_frozen_boundary():
    class FakeTensor:
        shape = (1, 2)

        def to(self, _device):
            return self

        def __getitem__(self, _key):
            return []

    class FakeTokenizer:
        eos_token_id = 9

        def apply_chat_template(self, *args, **kwargs):
            args = ()
            kwargs.clear()
            return FakeTensor()

        def decode(self, *args, **kwargs):
            args = ()
            kwargs.clear()
            return "{}"

    class FakeModel:
        device = "cuda"

        def __init__(self):
            self.generate_kwargs = None

        def generate(self, **kwargs):
            self.generate_kwargs = kwargs
            return FakeTensor()

    class Context:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class FakeCuda:
        @staticmethod
        def empty_cache():
            return None

    class FakeTorch:
        float16 = "float16"
        cuda = FakeCuda()

        @staticmethod
        def ones_like(value):
            return value

        @staticmethod
        def inference_mode():
            return Context()

        @staticmethod
        def autocast(**kwargs):
            assert kwargs == {"device_type": "cuda", "dtype": "float16"}
            return Context()

    model = FakeModel()
    client = baseline.LocalQwenToolClient(
        model=model,
        tokenizer=FakeTokenizer(),
        torch=FakeTorch(),
        max_new_tokens=512,
        stopping_criteria_list=lambda values: values,
    )

    client.create(messages=[], tools=[])

    assert model.generate_kwargs["do_sample"] is False
    assert model.generate_kwargs["max_new_tokens"] == 512
    assert model.generate_kwargs["use_cache"] is True
    assert len(model.generate_kwargs["stopping_criteria"]) == 1
