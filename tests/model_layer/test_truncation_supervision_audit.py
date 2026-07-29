import json
from pathlib import Path

from training.sft import audit_truncation_supervision as audit


class CharacterTokenizer:
    """Return one token per character so span/cut behavior is explicit."""

    def __call__(self, text, **kwargs):
        kwargs.clear()
        return {
            "input_ids": list(range(len(text))),
            "offset_mapping": [
                (index, index + 1)
                for index in range(len(text))
            ],
        }


def _row(
    record_id: str,
    intent: str,
    text: str,
    spans: list[list[int]],
) -> dict:
    return {
        "id": record_id,
        "intent": intent,
        "qwen_chatml": text,
        "assistant_char_spans": spans,
    }


def test_audit_row_distinguishes_complete_partial_and_fully_cut_supervision():
    tokenizer = CharacterTokenizer()

    complete = audit.audit_row_supervision(
        tokenizer,
        _row("complete", "sales", "abcdefgh", [[1, 4]]),
        max_seq_len=6,
    )
    partial = audit.audit_row_supervision(
        tokenizer,
        _row("partial", "recommend", "abcdefgh", [[4, 8]]),
        max_seq_len=6,
    )
    fully_cut = audit.audit_row_supervision(
        tokenizer,
        _row("fully-cut", "deep_search", "abcdefgh", [[6, 8]]),
        max_seq_len=6,
    )

    assert complete["status"] == audit.SUPERVISION_COMPLETE
    assert complete["supervised_tokens_truncated"] == 0
    assert partial["status"] == audit.SUPERVISION_PARTIALLY_TRUNCATED
    assert partial["supervised_tokens_retained"] == 2
    assert partial["supervised_tokens_truncated"] == 2
    assert fully_cut["status"] == audit.SUPERVISION_FULLY_TRUNCATED
    assert fully_cut["supervised_tokens_retained"] == 0
    assert fully_cut["supervised_tokens_truncated"] == 2


def test_partition_excludes_every_row_with_truncated_supervision():
    tokenizer = CharacterTokenizer()
    rows = [
        _row("safe-short", "sales", "abcd", [[1, 4]]),
        _row("safe-long", "compare", "abcdefgh", [[1, 4]]),
        _row("toxic-partial", "recommend", "abcdefgh", [[4, 8]]),
        _row("toxic-full", "deep_search", "abcdefgh", [[6, 8]]),
    ]

    active, excluded, results = audit.partition_rows_by_supervision(
        tokenizer,
        rows,
        max_seq_len=6,
        split="train",
    )

    assert [row["id"] for row in active] == ["safe-short", "safe-long"]
    assert [row["id"] for row in excluded] == [
        "toxic-partial",
        "toxic-full",
    ]
    assert {
        result["id"]: result["status"]
        for result in results
    } == {
        "safe-long": audit.SUPERVISION_COMPLETE,
        "toxic-partial": audit.SUPERVISION_PARTIALLY_TRUNCATED,
        "toxic-full": audit.SUPERVISION_FULLY_TRUNCATED,
    }


def test_repository_active_splits_match_persisted_supervision_audit():
    root = Path(__file__).resolve().parents[2]
    report_path = (
        root
        / "data"
        / "model_training"
        / "truncation_supervision_audit.json"
    )
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    token_report = json.loads(
        (
            root
            / "data"
            / "model_training"
            / "sft_token_length_report.json"
        ).read_text(encoding="utf-8")
    )
    train_ids = {
        json.loads(line)["id"]
        for line in (
            root / "data" / "model_training" / "sft_train.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    val_ids = {
        json.loads(line)["id"]
        for line in (
            root / "data" / "model_training" / "sft_val.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    excluded_train_ids = set(report["splits"]["train"]["excluded_ids"])
    excluded_val_ids = set(report["splits"]["validation"]["excluded_ids"])

    assert not train_ids & excluded_train_ids
    assert not val_ids & excluded_val_ids
    assert len(train_ids) == report["splits"]["train"]["active_rows"]
    assert len(val_ids) == report["splits"]["validation"]["active_rows"]
    assert report["training_invariant"] == {
        "active_train_rows_with_truncated_supervision": 0,
        "active_validation_rows_with_truncated_supervision": 0,
        "status": "passed",
    }
    assert token_report["rows_exceeding_selected_max_seq_len"] == 0
    assert (
        token_report["token_length_distribution"]["max"]
        <= token_report["selected_max_seq_len"]
    )
    assert (
        token_report["validation_token_length_distribution"]["max"]
        <= token_report["selected_max_seq_len"]
    )
