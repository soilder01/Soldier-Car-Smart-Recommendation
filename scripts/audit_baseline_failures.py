#!/usr/bin/env python3
"""Classify frozen base-model held-out failures without rerunning inference."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts import evaluate_model_outputs as evaluator
from scripts import run_local_qwen_baseline as baseline


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_PATH = (
    ROOT / "data" / "model_training" / "baseline_failure_taxonomy.json"
)
CATEGORIES = [
    "format_parse_failure",
    "wrong_tool",
    "argument_error",
    "grounding_failure",
    "other_contract_failure",
]
PRIMARY_PRECEDENCE = list(CATEGORIES)
PROTOCOL_DERIVED_GROUNDING_ERRORS = {
    "final assistant answer is missing",
}


def _format_subtype(output: dict[str, Any]) -> str | None:
    for message in output.get("trajectory", []):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if (
            isinstance(content, str)
            and "<tool_call>" in content
            and not message.get("tool_calls")
        ):
            return "invalid_tool_call_xml"

    runner_errors = output.get("runner_errors", [])
    if any(
        isinstance(error, str)
        and error.startswith("tool_call_structure_error:")
        for error in runner_errors
    ):
        return "invalid_tool_call_structure"
    terminal_error = output.get("terminal_parse_error")
    if not isinstance(terminal_error, str) or not terminal_error:
        return None
    if "max_steps=" in terminal_error:
        return "no_terminal_response"
    return "invalid_terminal_json"


def classify_failure_case(
    output: dict[str, Any],
    result: dict[str, Any],
    *,
    grounding_errors: list[str],
) -> dict[str, Any]:
    """Return independent failure labels plus one explicit primary reason."""
    categories: list[str] = []
    format_subtype = _format_subtype(output)
    if format_subtype is not None:
        categories.append("format_parse_failure")
    failures = result.get("failures", [])
    if any(
        isinstance(failure, str)
        and failure.startswith("tool_selection:")
        for failure in failures
    ):
        categories.append("wrong_tool")
    if result.get("valid_argument_calls") != result.get("total_tool_calls"):
        categories.append("argument_error")
    substantive_grounding_errors = [
        error
        for error in grounding_errors
        if error not in PROTOCOL_DERIVED_GROUNDING_ERRORS
    ]
    if substantive_grounding_errors:
        categories.append("grounding_failure")
    if not categories:
        categories.append("other_contract_failure")
    primary = next(
        category
        for category in PRIMARY_PRECEDENCE
        if category in categories
    )
    return {
        "categories": categories,
        "primary_category": primary,
        "format_subtype": format_subtype,
    }


def build_failure_taxonomy(
    *,
    cases: list[dict[str, Any]],
    outputs_by_id: dict[str, dict[str, Any]],
    results_by_id: dict[str, dict[str, Any]],
    grounding_by_id: dict[str, list[str]],
    passed_ids: set[str],
) -> dict[str, Any]:
    """Build mutually exclusive primary and overlapping multi-label counts."""
    multi_label = Counter({category: 0 for category in CATEGORIES})
    primary = Counter({category: 0 for category in CATEGORIES})
    per_intent: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "total_cases": 0,
            "passed_cases": 0,
            "failed_cases": 0,
            "multi_label_counts": Counter(
                {category: 0 for category in CATEGORIES}
            ),
            "primary_failure_counts": Counter(
                {category: 0 for category in CATEGORIES}
            ),
            "format_subtypes": Counter(),
        }
    )
    format_subtypes: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for case in cases:
        case_id = case["id"]
        intent = case["intent"]
        intent_counts = per_intent[intent]
        intent_counts["total_cases"] += 1
        if case_id in passed_ids:
            intent_counts["passed_cases"] += 1
            rows.append(
                {
                    "id": case_id,
                    "intent": intent,
                    "passed": True,
                    "categories": [],
                    "primary_category": None,
                    "format_subtype": None,
                }
            )
            continue

        intent_counts["failed_cases"] += 1
        classification = classify_failure_case(
            outputs_by_id[case_id],
            results_by_id[case_id],
            grounding_errors=grounding_by_id[case_id],
        )
        for category in classification["categories"]:
            multi_label[category] += 1
            intent_counts["multi_label_counts"][category] += 1
        primary_category = classification["primary_category"]
        primary[primary_category] += 1
        intent_counts["primary_failure_counts"][primary_category] += 1
        subtype = classification["format_subtype"]
        if subtype is not None:
            format_subtypes[subtype] += 1
            intent_counts["format_subtypes"][subtype] += 1
        rows.append(
            {
                "id": case_id,
                "intent": intent,
                "passed": False,
                **classification,
                "evaluator_failures": results_by_id[case_id]["failures"],
                "grounding_errors": grounding_by_id[case_id],
            }
        )

    formatted_intents = {}
    for intent, counts in sorted(per_intent.items()):
        formatted_intents[intent] = {
            "total_cases": counts["total_cases"],
            "passed_cases": counts["passed_cases"],
            "failed_cases": counts["failed_cases"],
            "multi_label_counts": {
                category: counts["multi_label_counts"][category]
                for category in CATEGORIES
            },
            "primary_failure_counts": {
                category: counts["primary_failure_counts"][category]
                for category in CATEGORIES
            },
            "format_subtypes": dict(sorted(counts["format_subtypes"].items())),
        }
    return {
        "status": "completed",
        "baseline_score": "6/40=15.0%",
        "taxonomy_scope": "failed held-out cases only",
        "counting_note": (
            "multi_label_counts overlap; primary_failure_counts are exclusive "
            f"with precedence {PRIMARY_PRECEDENCE}"
        ),
        "total_cases": len(cases),
        "passed_cases": len(passed_ids),
        "failed_cases": len(cases) - len(passed_ids),
        "multi_label_counts": {
            category: multi_label[category]
            for category in CATEGORIES
        },
        "primary_failure_counts": {
            category: primary[category]
            for category in CATEGORIES
        },
        "format_subtypes": dict(sorted(format_subtypes.items())),
        "per_intent": formatted_intents,
        "cases": rows,
    }


def generate_report(
    *,
    output_path: Path = baseline.DEFAULT_OUTPUT_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    """Re-evaluate persisted trajectories and write deterministic taxonomy."""
    baseline.validate_frozen_harness_manifest()
    cases = baseline.validate_heldout_only(baseline.HELD_OUT_PATH)
    outputs = evaluator.load_jsonl(output_path, label="local baseline outputs")
    evaluation = evaluator.evaluate_records(
        cases,
        outputs,
        vehicle_catalog=set(evaluator.load_vehicle_catalog()),
    )
    grounding = baseline._grounding_errors(outputs)
    summary = baseline.build_baseline_summary(
        cases=cases,
        evaluation=evaluation,
        grounding_by_id=grounding,
        raw_outputs_path=output_path,
    )
    passed_ids = {
        row["id"]
        for row in summary["cases"]
        if row["passed"]
    }
    report = build_failure_taxonomy(
        cases=cases,
        outputs_by_id={row["id"]: row for row in outputs},
        results_by_id={
            row["id"]: row for row in evaluation["case_results"]
        },
        grounding_by_id=grounding,
        passed_ids=passed_ids,
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify persisted local-Qwen held-out failures.",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=baseline.DEFAULT_OUTPUT_PATH,
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()
    report = generate_report(
        output_path=args.outputs,
        report_path=args.report,
    )
    print(
        json.dumps(
            {
                "failed_cases": report["failed_cases"],
                "multi_label_counts": report["multi_label_counts"],
                "primary_failure_counts": report[
                    "primary_failure_counts"
                ],
                "report": str(args.report),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
