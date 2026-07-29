from training.sft import verify_qwen_template as verifier


def _row(text: str) -> dict:
    return {
        "id": "sft-001",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "query"},
            {"role": "assistant", "content": "answer"},
        ],
        "tools": [],
        "qwen_chatml": text,
    }


def test_compare_template_text_reports_byte_exact_match():
    frozen = "<|im_start|>system\nsystem<|im_end|>\n"

    result = verifier.compare_template_text(
        _row(frozen),
        rendered_by_tokenizer=frozen,
    )

    assert result["status"] == "matched"
    assert result["byte_exact"] is True
    assert result["first_difference_offset"] is None
    assert result["frozen_sha256"] == result["tokenizer_sha256"]


def test_compare_template_text_reports_first_difference_without_text_leakage():
    result = verifier.compare_template_text(
        _row("abc"),
        rendered_by_tokenizer="axc",
    )

    assert result["status"] == "mismatched"
    assert result["byte_exact"] is False
    assert result["first_difference_offset"] == 1
    assert "abc" not in str(result)
    assert "axc" not in str(result)
