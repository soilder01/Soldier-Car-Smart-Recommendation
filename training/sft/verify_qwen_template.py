"""Byte-compare frozen Qwen ChatML with the local tokenizer template."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

from training.sft import train_qlora_sft as sft


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "training" / "sft" / "qlora_sft_config.yaml"
DEFAULT_REPORT_PATH = (
    ROOT / "data" / "model_training" / "qwen_template_verification.json"
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _first_difference_byte_offset(left: bytes, right: bytes) -> int | None:
    for offset, (left_byte, right_byte) in enumerate(zip(left, right)):
        if left_byte != right_byte:
            return offset
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def compare_template_text(
    row: dict[str, Any],
    *,
    rendered_by_tokenizer: str,
) -> dict[str, Any]:
    """Return a content-free byte-level comparison result for one frozen row."""
    frozen = row.get("qwen_chatml")
    record_id = row.get("id")
    if not isinstance(record_id, str) or not record_id.strip():
        raise ValueError("frozen row id must be a non-empty string")
    if not isinstance(frozen, str):
        raise ValueError("frozen row qwen_chatml must be a string")
    if not isinstance(rendered_by_tokenizer, str):
        raise ValueError("tokenizer rendering must be a string")

    frozen_bytes = frozen.encode("utf-8")
    tokenizer_bytes = rendered_by_tokenizer.encode("utf-8")
    first_difference = _first_difference_byte_offset(frozen_bytes, tokenizer_bytes)
    matched = first_difference is None
    return {
        "status": "matched" if matched else "mismatched",
        "record_id": record_id,
        "byte_exact": matched,
        "first_difference_offset": first_difference,
        "frozen_byte_length": len(frozen_bytes),
        "tokenizer_byte_length": len(tokenizer_bytes),
        "frozen_sha256": _sha256(frozen),
        "tokenizer_sha256": _sha256(rendered_by_tokenizer),
    }


def _write_report(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_template_verification(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    """Use the local tokenizer's official template and write a byte comparison."""
    try:
        config = sft.load_training_config(config_path)
        model_path = sft._model_path(config, repo_root)
        row = sft.load_one_frozen_batch(config, repo_root=repo_root)
        transformers = importlib.import_module("transformers")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=False,
        )
        rendered = tokenizer.apply_chat_template(
            row["messages"],
            tools=row["tools"],
            tokenize=False,
            add_generation_prompt=False,
        )
        result = compare_template_text(row, rendered_by_tokenizer=rendered)
    except Exception as error:
        result = {
            "status": "blocked",
            "byte_exact": False,
            "blocker": "template_verification_failed",
            "detail": f"{type(error).__name__}: {error}",
        }
    _write_report(report_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()
    result = run_template_verification(
        config_path=args.config,
        report_path=args.report_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "matched" else 1)


if __name__ == "__main__":
    main()
