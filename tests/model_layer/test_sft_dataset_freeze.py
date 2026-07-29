import json
from pathlib import Path

import pytest

from data_synth import freeze_sft_dataset as freezer


def _record(record_id: str, intent: str, query: str) -> dict:
    return {
        "id": record_id,
        "intent": intent,
        "messages": [
            {"role": "system", "content": "Follow the tool contract."},
            {"role": "user", "content": query},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call-{record_id}",
                        "type": "function",
                        "function": {
                            "name": "extract_user_profile",
                            "arguments": '{"query":"%s"}' % query,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": f"call-{record_id}",
                "content": '{"budget_max":25}',
            },
            {"role": "assistant", "content": "已基于工具结果完成回答。"},
        ],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def test_render_qwen_chatml_preserves_multiturn_tool_trajectory():
    record = _record("recommend-001", "recommend", "预算25万推荐SUV")

    rendered = freezer.render_qwen_chatml(record["messages"])

    assert "<|im_start|>system" in rendered
    assert "<tools>" in rendered
    assert '{"type": "function", "function": {"name": "extract_user_profile"' in rendered
    assert '<tool_call>\n{"name": "extract_user_profile"' in rendered
    assert (
        '"arguments": "{\\"query\\":\\"预算25万推荐SUV\\"}"'
        in rendered
    )
    assert '<tool_response>\n{"budget_max":25}\n</tool_response>' in rendered
    assert rendered.index("<tool_call>") < rendered.index("<tool_response>")
    assert rendered.rstrip().endswith("<|im_end|>")


def test_freeze_stratifies_deterministically_and_preserves_source_messages(
    tmp_path: Path,
):
    source_a = tmp_path / "recommend.jsonl"
    source_b = tmp_path / "sales.jsonl"
    _write_jsonl(
        source_a,
        [
            _record(f"recommend-{index:03d}", "recommend", f"推荐问题{index}")
            for index in range(10)
        ],
    )
    _write_jsonl(
        source_b,
        [
            _record(f"sales-{index:03d}", "sales", f"销售问题{index}")
            for index in range(10)
        ],
    )
    sources = {
        "recommend": source_a,
        "sales": source_b,
    }

    first = freezer.freeze_dataset(
        sources=sources,
        output_dir=tmp_path / "first",
        eval_fraction=0.2,
        split_seed=20260724,
        reward_rows=[],
        held_out_rows=[],
        teacher_endpoint_alias="configured_teacher_alias",
    )
    second = freezer.freeze_dataset(
        sources=sources,
        output_dir=tmp_path / "second",
        eval_fraction=0.2,
        split_seed=20260724,
        reward_rows=[],
        held_out_rows=[],
        teacher_endpoint_alias="configured_teacher_alias",
    )

    assert first["counts"] == {
        "source_total": 20,
        "train_total": 16,
        "eval_total": 4,
        "per_intent": {
            "recommend": {"source": 10, "train": 8, "eval": 2},
            "sales": {"source": 10, "train": 8, "eval": 2},
        },
    }
    assert first["split_manifest"] == second["split_manifest"]

    train_rows = freezer.load_jsonl(first["train_path"])
    assert all("messages" in row and "qwen_chatml" in row for row in train_rows)
    assert any(message["role"] == "tool" for message in train_rows[0]["messages"])
    assert all(
        row["assistant_char_spans"]
        and all(0 <= start < end <= len(row["qwen_chatml"]) for start, end in row["assistant_char_spans"])
        for row in train_rows
    )


def test_freeze_rejects_reward_or_held_out_id_and_query_overlap(tmp_path: Path):
    source = tmp_path / "recommend.jsonl"
    _write_jsonl(
        source,
        [_record("recommend-001", "recommend", "预算25万推荐SUV")],
    )

    with pytest.raises(ValueError, match="SFT/reward isolation violation"):
        freezer.freeze_dataset(
            sources={"recommend": source},
            output_dir=tmp_path / "out",
            eval_fraction=0.1,
            split_seed=42,
            reward_rows=[
                {
                    "id": " RECOMMEND-001 ",
                    "query": "无关问题",
                }
            ],
            held_out_rows=[],
            teacher_endpoint_alias="configured_teacher_alias",
        )

    with pytest.raises(ValueError, match="SFT/reward isolation violation"):
        freezer.freeze_dataset(
            sources={"recommend": source},
            output_dir=tmp_path / "out",
            eval_fraction=0.1,
            split_seed=42,
            reward_rows=[],
            held_out_rows=[
                {
                    "id": "heldout-001",
                    "query": " 预算25万推荐SUV ",
                }
            ],
            teacher_endpoint_alias="configured_teacher_alias",
        )
