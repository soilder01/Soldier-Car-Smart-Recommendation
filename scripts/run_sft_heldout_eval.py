#!/usr/bin/env python3
"""Evaluate the selected SFT adapter once with the frozen held-out harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts import run_local_qwen_baseline as baseline
from training.sft import train_qlora_sft as sft


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_REPORT_PATH = (
    ROOT / "data" / "model_training" / "sft_training_report.json"
)
DEFAULT_OUTPUT_PATH = (
    ROOT / "data" / "model_training" / "sft_heldout_outputs.jsonl"
)
DEFAULT_REPORT_PATH = (
    ROOT / "data" / "model_training" / "sft_heldout_report.json"
)
DEFAULT_FAILURES_PATH = (
    ROOT / "data" / "model_training" / "sft_heldout_failures.jsonl"
)
SFT_MODEL_ALIAS = "local_sft_best_adapter"


def failure_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only failed held-out cases for the dedicated failure artifact."""
    return [
        {
            key: value
            for key, value in row.items()
            if key
            in {
                "id",
                "intent",
                "failure_reasons",
                "actual_tools",
                "recommendation_hit",
                "grounding_errors",
            }
        }
        for row in summary["cases"]
        if row["passed"] is False
    ]


def baseline_comparison(summary: dict[str, Any]) -> dict[str, Any]:
    """Compare only against the frozen 6/40 native-model anchor."""
    score = summary["total_score"]
    return {
        "baseline": {
            "numerator": 6,
            "denominator": 40,
            "percentage": 15.0,
        },
        "sft": {
            "numerator": score["numerator"],
            "denominator": score["denominator"],
            "percentage": score["percentage"],
        },
        "absolute_percentage_point_change": round(
            score["percentage"] - 15.0,
            1,
        ),
        "claim_scope": (
            "tool-call contract and structured-protocol compliance; "
            "not a standalone decision-quality claim"
        ),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _load_selected_adapter(training_report_path: Path) -> tuple[dict[str, Any], Path]:
    report = json.loads(training_report_path.read_text(encoding="utf-8"))
    if report.get("status") != "completed":
        raise ValueError("formal SFT training is not completed")
    if report.get("held_out_evaluated") is not False:
        raise ValueError("held-out evaluation has already been recorded")
    adapter_value = report.get("best_adapter")
    if not isinstance(adapter_value, str) or not adapter_value:
        raise ValueError("training report has no selected best adapter")
    adapter_path = Path(adapter_value)
    if not adapter_path.is_absolute():
        adapter_path = ROOT / adapter_path
    if not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError("selected adapter_config.json was not found")
    if not (adapter_path / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("selected adapter weights were not found")
    return report, adapter_path


def _load_adapter_client(
    *,
    adapter_path: Path,
    config_path: Path,
) -> baseline.LocalQwenToolClient:
    config = sft.load_training_config(config_path)
    model_path = sft._model_path(config, ROOT)
    dependencies = sft._load_training_dependencies()
    torch = dependencies["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for held-out evaluation")
    quantization = dependencies["BitsAndBytesConfig"](
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = dependencies["AutoTokenizer"].from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = dependencies["AutoModelForCausalLM"].from_pretrained(
        str(model_path),
        quantization_config=quantization,
        device_map={"": 0},
        local_files_only=True,
        trust_remote_code=False,
    )
    model = dependencies["PeftModel"].from_pretrained(
        base_model,
        str(adapter_path),
        is_trainable=False,
        local_files_only=True,
    )
    model.config.use_cache = True
    model.eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    return baseline.LocalQwenToolClient(
        model=model,
        tokenizer=tokenizer,
        torch=torch,
        max_new_tokens=baseline.FROZEN_MAX_NEW_TOKENS,
        stopping_criteria_list=dependencies["StoppingCriteriaList"],
    )


def run_once(
    *,
    config_path: Path = sft.DEFAULT_CONFIG_PATH,
    authorization_path: Path = sft.DEFAULT_AUTHORIZATION_PATH,
    training_report_path: Path = DEFAULT_TRAINING_REPORT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    failures_path: Path = DEFAULT_FAILURES_PATH,
) -> dict[str, Any]:
    """Run one post-selection held-out campaign and refuse every rerun."""
    for path in (output_path, report_path, failures_path):
        if path.exists():
            raise FileExistsError(
                f"held-out evaluation is single-use; output exists: {path}"
            )
    sft.validate_training_authorization(
        config_path=config_path,
        authorization_path=authorization_path,
        repo_root=ROOT,
    )
    baseline.validate_frozen_harness_manifest()
    training_report, adapter_path = _load_selected_adapter(
        training_report_path
    )
    client = _load_adapter_client(
        adapter_path=adapter_path,
        config_path=config_path,
    )
    try:
        summary = baseline.run_frozen_heldout_harness(
            client=client,
            output_path=output_path,
            report_path=report_path,
            model_alias=SFT_MODEL_ALIAS,
        )
    finally:
        client.close()

    summary["adapter_path"] = str(adapter_path)
    summary["harness_version"] = baseline.FROZEN_HARNESS_VERSION
    summary["evaluation_count"] = 1
    summary["checkpoint_selection_used_held_out"] = False
    summary["baseline_comparison"] = baseline_comparison(summary)
    baseline._write_json(report_path, summary)
    _write_jsonl(failures_path, failure_rows(summary))

    training_report["held_out_evaluated"] = True
    training_report["held_out_evaluation_count"] = 1
    training_report["held_out_report"] = str(report_path)
    training_report["held_out_score"] = summary["total_score"]
    training_report_path.write_text(
        json.dumps(training_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=sft.DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--authorization",
        type=Path,
        default=sft.DEFAULT_AUTHORIZATION_PATH,
    )
    parser.add_argument(
        "--training-report",
        type=Path,
        default=DEFAULT_TRAINING_REPORT_PATH,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--failures", type=Path, default=DEFAULT_FAILURES_PATH)
    args = parser.parse_args()
    result = run_once(
        config_path=args.config,
        authorization_path=args.authorization,
        training_report_path=args.training_report,
        output_path=args.output,
        report_path=args.report,
        failures_path=args.failures,
    )
    print(
        json.dumps(
            {
                "total_score": result["total_score"],
                "per_intent": result["per_intent"],
                "baseline_comparison": result["baseline_comparison"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
