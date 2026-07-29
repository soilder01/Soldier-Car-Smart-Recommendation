#!/usr/bin/env python3
"""Audit persisted knowledge trajectories without loading either model."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = (
    ROOT / "data" / "model_training" / "baseline_qwen_heldout_outputs.jsonl"
)
SFT_PATH = ROOT / "data" / "model_training" / "sft_heldout_outputs.jsonl"
TRAIN_PATH = ROOT / "data" / "model_training" / "sft_train.jsonl"
DEFAULT_REPORT_PATH = (
    ROOT / "data" / "model_training" / "knowledge_stop_regression_audit.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _tool_result_shape(content: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return {"json_type": "invalid", "items": None, "empty": False}
    return {
        "json_type": type(value).__name__,
        "items": len(value) if hasattr(value, "__len__") else None,
        "empty": value in ([], {}),
    }


def _trajectory_steps(output: dict[str, Any]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for message in output["trajectory"][2:]:
        if message["role"] == "assistant":
            calls = message.get("tool_calls", [])
            if not calls:
                steps.append(
                    {
                        "step": len(steps) + 1,
                        "terminal": True,
                        "answer_characters": len(message.get("content") or ""),
                    }
                )
                current = None
                continue
            current = {
                "step": len(steps) + 1,
                "terminal": False,
                "calls": [
                    {
                        "name": call["function"]["name"],
                        "arguments": call["function"]["arguments"],
                    }
                    for call in calls
                ],
                "results": [],
            }
            steps.append(current)
        elif message["role"] == "tool":
            if current is None:
                raise ValueError(
                    f"{output['id']}: tool result has no current assistant call"
                )
            current["results"].append(_tool_result_shape(message["content"]))
    return steps


def _training_distribution() -> dict[str, Any]:
    rows = _load_jsonl(TRAIN_PATH)
    counts = Counter(row["intent"] for row in rows)
    terminal_counts: Counter[str] = Counter()
    for row in rows:
        finals = [
            message.get("content")
            for message in row["messages"]
            if message.get("role") == "assistant"
            and not message.get("tool_calls")
        ]
        if finals and isinstance(finals[-1], str) and finals[-1].strip():
            terminal_counts[row["intent"]] += 1
    return {
        "path": str(TRAIN_PATH.relative_to(ROOT)),
        "sha256": _sha256(TRAIN_PATH),
        "rows_by_intent": dict(sorted(counts.items())),
        "nonempty_terminal_rows_by_intent": dict(
            sorted(terminal_counts.items())
        ),
        "knowledge_rows": counts["knowledge"],
    }


def build_report() -> dict[str, Any]:
    baseline = {row["id"]: row for row in _load_jsonl(BASELINE_PATH)}
    sft = {row["id"]: row for row in _load_jsonl(SFT_PATH)}
    failed_ids = sorted(
        row_id
        for row_id, row in sft.items()
        if row_id.startswith("heldout-knowledge-")
        and row["raw_assistant_content"] is None
    )
    if len(failed_ids) != 8:
        raise ValueError(f"expected 8 non-stopping SFT knowledge cases: {failed_ids}")

    cases: list[dict[str, Any]] = []
    total_sft_calls = 0
    total_baseline_calls = 0
    empty_sft_web_results = 0
    exact_alternating = 0
    prompt_equal = True
    for case_id in failed_ids:
        baseline_row = baseline[case_id]
        sft_row = sft[case_id]
        baseline_steps = _trajectory_steps(baseline_row)
        sft_steps = _trajectory_steps(sft_row)
        baseline_tools = [
            call["name"]
            for step in baseline_steps
            for call in step.get("calls", [])
        ]
        sft_tools = [
            call["name"]
            for step in sft_steps
            for call in step.get("calls", [])
        ]
        total_baseline_calls += len(baseline_tools)
        total_sft_calls += len(sft_tools)
        exact_alternating += sft_tools == [
            "retrieve_knowledge_base",
            "search_web_info",
        ] * 4
        for step in sft_steps:
            for call, result in zip(
                step.get("calls", []),
                step.get("results", []),
            ):
                if call["name"] == "search_web_info" and result["empty"]:
                    empty_sft_web_results += 1
        same_prompt = (
            baseline_row["trajectory"][0] == sft_row["trajectory"][0]
        )
        prompt_equal = prompt_equal and same_prompt
        cases.append(
            {
                "id": case_id,
                "same_legacy_system_prompt": same_prompt,
                "baseline": {
                    "steps": baseline_steps,
                    "tool_sequence": baseline_tools,
                    "terminal_reached": (
                        baseline_row["raw_assistant_content"] is not None
                    ),
                    "runner_errors": baseline_row["runner_errors"],
                },
                "sft": {
                    "steps": sft_steps,
                    "tool_sequence": sft_tools,
                    "terminal_reached": False,
                    "runner_errors": sft_row["runner_errors"],
                },
            }
        )

    return {
        "status": "completed",
        "mode": "offline_persisted_outputs_only",
        "model_inference": False,
        "inputs": {
            "baseline": {
                "path": str(BASELINE_PATH.relative_to(ROOT)),
                "sha256": _sha256(BASELINE_PATH),
            },
            "sft": {
                "path": str(SFT_PATH.relative_to(ROOT)),
                "sha256": _sha256(SFT_PATH),
            },
        },
        "aggregate_evidence": {
            "sft_non_stopping_knowledge_cases": len(failed_ids),
            "sft_tool_calls_in_failed_cases": total_sft_calls,
            "baseline_tool_calls_in_matched_cases": total_baseline_calls,
            "sft_failed_cases_with_nonempty_first_retrieval": sum(
                bool(
                    case["sft"]["steps"]
                    and case["sft"]["steps"][0]["results"]
                    and not case["sft"]["steps"][0]["results"][0]["empty"]
                )
                for case in cases
            ),
            "sft_empty_web_results_followed_by_more_calls": (
                empty_sft_web_results
            ),
            "sft_exact_retrieve_web_alternation_cases": exact_alternating,
            "matched_cases_with_identical_legacy_system_prompt": sum(
                case["same_legacy_system_prompt"] for case in cases
            ),
            "all_matched_legacy_system_prompts_identical": prompt_equal,
            "baseline_terminal_step_range": [
                min(len(case["baseline"]["steps"]) for case in cases),
                max(len(case["baseline"]["steps"]) for case in cases),
            ],
        },
        "training_context": _training_distribution(),
        "diagnosis": {
            "evidence_starvation": "rejected",
            "pure_prompt_only_failure": "rejected",
            "supported_interpretation": (
                "The failure is SFT-specific post-tool stopping behavior under "
                "the legacy prompt mismatch. The shared JSON instruction may "
                "interact with an SFT model trained only on natural-language "
                "terminals, but offline trajectories alone cannot establish "
                "that it is the sole cause."
            ),
            "additional_risk_factor": (
                "The active SFT split contains no intent=knowledge rows, so "
                "knowledge stopping is out-of-intent-distribution despite "
                "customer_service examples using similar tools."
            ),
            "causal_test": (
                "Re-run both models with the frozen production prompt and no "
                "JSON terminal suffix while holding all other settings fixed."
            ),
        },
        "cases": cases,
    }


def main() -> None:
    report = build_report()
    DEFAULT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DEFAULT_REPORT_PATH.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps(report["aggregate_evidence"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
