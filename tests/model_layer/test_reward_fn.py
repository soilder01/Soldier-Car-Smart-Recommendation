import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from training.grpo.reward_fn import (
    RewardCache,
    RewardContext,
    compute_answer_reward,
    compute_format_reward,
    compute_tool_execution_reward,
)


ROOT = Path(__file__).resolve().parents[2]
GRPO_CONFIG = ROOT / "training" / "grpo" / "grpo_config.yaml"
GRPO_README = ROOT / "training" / "grpo" / "README.md"
GRPO_REPORT = ROOT / "docs" / "model_layer_phase4_grpo_report.md"
TRAIN_GRPO = ROOT / "training" / "grpo" / "train_grpo.py"


def _call(name="retrieve_knowledge_base", arguments=None, call_id="call-1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(
                {"query": "补能策略"} if arguments is None else arguments,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    }


def test_reward_context_rejects_held_out_overlap_with_normalization():
    with pytest.raises(ValueError, match="held-out overlap"):
        RewardContext(
            reward_visible_ids={" ＣＡＳＥ-1 "},
            held_out_ids={"case-1"},
        )


@pytest.mark.parametrize(
    "bad_ids",
    [{123}, {""}, {" \t"}],
)
def test_reward_context_rejects_invalid_id_sets(bad_ids):
    with pytest.raises(ValueError):
        RewardContext(reward_visible_ids=bad_ids, held_out_ids=set())


def test_answer_reward_never_scores_held_out_prompts():
    context = RewardContext(
        reward_visible_ids={"reward-1"},
        held_out_ids={"held-1"},
    )

    assert compute_answer_reward("held-1", {}, context) == 0.0
    assert compute_answer_reward("unknown", {}, context) == 0.0
    assert compute_answer_reward("reward-1", {}, context) == 0.0


@pytest.mark.parametrize(
    "completion",
    [
        {"tool_calls": []},
        {"tool_calls": [{"function": {"name": "unknown_tool", "arguments": "{}"}}]},
        {"tool_calls": [{"id": "x", "type": "function"}]},
        {"tool_calls": [_call(arguments={})]},
        {"tool_calls": [_call(arguments={"query": "补能", "extra": True})]},
        {"tool_calls": [_call(arguments={"query": 7})]},
        {
            "tool_calls": [
                _call(
                    name="search_and_rank_vehicles",
                    arguments={"budget_max": True},
                )
            ]
        },
    ],
)
def test_format_reward_rejects_invalid_tool_calls(completion):
    assert compute_format_reward(completion) == 0.0


def test_format_reward_accepts_approved_schema_without_executing_tools():
    completion = {"tool_calls": [_call()]}

    assert compute_format_reward(completion) == 1.0


def test_tool_execution_reward_requires_matching_successful_results():
    call = _call(call_id="call-rag")

    assert compute_tool_execution_reward({"tool_calls": [call]}) == 0.0
    assert (
        compute_tool_execution_reward(
            {
                "tool_calls": [call],
                "tool_results": [
                    {
                        "tool_call_id": "other-call",
                        "content": "ok",
                    }
                ],
            }
        )
        == 0.0
    )
    assert (
        compute_tool_execution_reward(
            {
                "tool_calls": [call],
                "tool_results": [
                    {
                        "tool_call_id": "call-rag",
                        "content": {"error": {"type": "tool_failed"}},
                    }
                ],
            }
        )
        == 0.0
    )
    assert (
        compute_tool_execution_reward(
            {
                "tool_calls": [call],
                "tool_results": [
                    {
                        "tool_call_id": "call-rag",
                        "content": "grounded result",
                    }
                ],
            }
        )
        == 1.0
    )


def test_reward_cache_uses_stable_key_and_defensive_copies():
    cache = RewardCache()
    value = {"score": 0.5, "parts": {"format": 1.0}}

    cache.set("prompt", "completion", value)
    value["parts"]["format"] = 0.0
    cached = cache.get("prompt", "completion")
    assert cached == {"score": 0.5, "parts": {"format": 1.0}}
    cached["parts"]["format"] = 0.0
    assert cache.get("prompt", "completion") == {
        "score": 0.5,
        "parts": {"format": 1.0},
    }
    assert cache.key("prompt", "completion") == cache.key(
        "prompt",
        "completion",
    )


def test_grpo_config_locks_rollout_reward_and_memory_contracts():
    config = yaml.safe_load(GRPO_CONFIG.read_text(encoding="utf-8"))

    assert config["model"]["sft_adapter_path"] == "checkpoints/sft"
    assert config["gates"]["requires_sft_gate_passed"] is True
    assert config["rollout"]["backend"] == "vllm"
    assert config["rollout"]["num_generations"] >= 8
    assert 0.7 <= config["rollout"]["temperature"] <= 1.0
    assert config["kl"]["beta"] == 0.01
    assert (
        config["kl"]["strategy"]
        == "shared_frozen_base_disable_adapter_reference"
    )
    assert config["kl"]["variance_control"] == "clip-higher"
    assert config["memory"]["single_gpu_vram_gb"] == 32
    assert config["reward"]["format_reward_cap"] <= 1.0
    assert config["reward"]["answer_reward_primary"] is True
    assert config["reward"]["cache_enabled"] is True
    assert config["data"]["held_out_file"] == "data/model_training/eval/held_out.jsonl"


def test_grpo_docs_and_entrypoint_are_fail_closed():
    readme = GRPO_README.read_text(encoding="utf-8")
    report = GRPO_REPORT.read_text(encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(TRAIN_GRPO)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "gated" in (completed.stderr + completed.stdout)
    assert "held-out cases 不得进入 reward、调参或 early stopping" in readme
    assert "当前状态：blocked" in report
    assert "未完成 held-out 对比前，不得声明最终模型提升" in report
