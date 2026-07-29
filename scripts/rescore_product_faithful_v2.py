#!/usr/bin/env python3
"""Symmetrically rescore persisted held-out outputs against the product contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from data_synth.tool_schemas import build_tool_schemas
from scripts import evaluate_model_outputs as legacy_evaluator


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    ROOT
    / "data"
    / "model_training"
    / "eval"
    / "product_faithful_v2_scoring_manifest.json"
)
MANIFEST_SHA_PATH = MANIFEST_PATH.with_suffix(".sha256")
DEFAULT_REPORT_PATH = (
    ROOT / "data" / "model_training" / "product_faithful_v2_report.json"
)
EXPECTED_MANIFEST_SHA256 = (
    "49d3a2b1da23490b236f839f73d55654743a596899909fa80b4fcc090d721113"
)
COMPARISON_METRICS = [
    "normal_stop_rate",
    "mandatory_tool_order_accuracy",
    "tool_allowlist_accuracy",
    "argument_schema_accuracy",
    "argument_schema_call_accuracy",
    "v2_composite_pass_rate",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _repo_path(value: str) -> Path:
    path = ROOT / value
    if not path.is_file():
        raise FileNotFoundError(f"frozen file not found: {path}")
    return path


def load_frozen_manifest(
    manifest_path: Path = MANIFEST_PATH,
    sha_path: Path = MANIFEST_SHA_PATH,
) -> tuple[dict[str, Any], str]:
    """Load the pre-scoring manifest and fail closed on any SHA drift."""
    manifest_path = Path(manifest_path)
    sha_path = Path(sha_path)
    observed_sha = sha256_file(manifest_path)
    if observed_sha != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            "v2 scoring manifest SHA mismatch: "
            f"expected={EXPECTED_MANIFEST_SHA256} actual={observed_sha}"
        )
    recorded_parts = sha_path.read_text(encoding="utf-8").strip().split()
    if not recorded_parts or recorded_parts[0] != observed_sha:
        raise ValueError("recorded v2 scoring manifest SHA does not match")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "frozen_before_scoring":
        raise ValueError("v2 scoring manifest was not frozen before scoring")
    if manifest.get("manifest_version") != "product-faithful-heldout-v2.0":
        raise ValueError("unsupported v2 scoring manifest version")
    return manifest, observed_sha


def _validate_frozen_files(manifest: dict[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    sections = [
        *manifest["product_contract"]["evidence"],
        *manifest["frozen_inputs"].values(),
        *manifest["historical_artifacts_preserved"].values(),
    ]
    for entry in sections:
        path_value = entry["path"]
        path = _repo_path(path_value)
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise ValueError(
                f"frozen file SHA mismatch for {path_value}: "
                f"expected={entry['sha256']} actual={digest}"
            )
        observed[path_value] = digest

    for key in ("cases", "baseline_outputs", "sft_outputs"):
        entry = manifest["frozen_inputs"][key]
        records = legacy_evaluator.load_jsonl(
            _repo_path(entry["path"]),
            label=f"frozen {key}",
        )
        if len(records) != entry["records"]:
            raise ValueError(
                f"frozen {key} record count mismatch: "
                f"expected={entry['records']} actual={len(records)}"
            )
    return observed


def resolve_terminal_answer(content: Any) -> tuple[str | None, str]:
    """Resolve legacy JSON-wrapped and direct production answers symmetrically."""
    if not isinstance(content, str):
        return None, "missing"
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return content, "raw_assistant_content"
    if isinstance(parsed, dict) and isinstance(parsed.get("answer"), str):
        return parsed["answer"], "legacy_json_answer"
    return content, "raw_assistant_content"


def schema_map() -> dict[str, dict[str, Any]]:
    return {
        schema["function"]["name"]: schema["function"]["parameters"]
        for schema in build_tool_schemas()
    }


def _type_matches(value: Any, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    return False


def _validate_json_value(
    value: Any,
    schema: dict[str, Any],
    *,
    path: str,
) -> str | None:
    expected_type = schema.get("type")
    if expected_type and not _type_matches(value, expected_type):
        return f"{path} must be {expected_type}"
    if expected_type == "object":
        properties = schema.get("properties", {})
        for field in schema.get("required", []):
            if field not in value:
                return f"{path} missing required argument: {field}"
        if schema.get("additionalProperties") is False:
            extras = sorted(set(value) - set(properties))
            if extras:
                return f"{path} has unexpected argument: {extras[0]}"
        for field, child in value.items():
            if field not in properties:
                continue
            failure = _validate_json_value(
                child,
                properties[field],
                path=f"{path}.{field}",
            )
            if failure is not None:
                return failure
    if expected_type == "array" and "items" in schema:
        for index, child in enumerate(value):
            failure = _validate_json_value(
                child,
                schema["items"],
                path=f"{path}[{index}]",
            )
            if failure is not None:
                return failure
    return None


def validate_tool_call_schema(
    call: Any,
    schemas: dict[str, dict[str, Any]],
) -> tuple[bool, str | None]:
    name = legacy_evaluator._openai_call_name(call)
    if name is None:
        return False, "expected OpenAI function tool-call object"
    if name not in schemas:
        return False, f"unknown tool: {name}"
    arguments, parse_error = legacy_evaluator._parse_call_arguments(
        call["function"]["arguments"]
    )
    if parse_error is not None:
        return False, parse_error
    assert arguments is not None
    failure = _validate_json_value(
        arguments,
        schemas[name],
        path="arguments",
    )
    return failure is None, failure


def _metric(numerator: int, denominator: int) -> dict[str, int | float]:
    ratio = numerator / denominator if denominator else 0.0
    return {
        "numerator": numerator,
        "denominator": denominator,
        "ratio": ratio,
        "percentage": round(ratio * 100.0, 1),
    }


def _ordered_subsequence_complete(
    actual: list[str],
    mandatory: list[str],
) -> bool:
    matched = 0
    for name in actual:
        if matched < len(mandatory) and name == mandatory[matched]:
            matched += 1
    return matched == len(mandatory)


def _normal_stop(
    output: dict[str, Any],
    *,
    max_steps: int,
) -> tuple[bool, str | None, str, int]:
    trajectory = output["trajectory"]
    assistant_turns = [
        message
        for message in trajectory[2:]
        if message["role"] == "assistant"
    ]
    terminal_turns = [
        message
        for message in assistant_turns
        if not message.get("tool_calls")
    ]
    answer, source = resolve_terminal_answer(output["raw_assistant_content"])
    passed = bool(
        len(terminal_turns) == 1
        and len(assistant_turns) <= max_steps
        and terminal_turns[0]["content"] == output["raw_assistant_content"]
        and isinstance(answer, str)
        and answer.strip()
    )
    return passed, answer, source, len(assistant_turns)


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    valid_calls = sum(row["valid_argument_calls"] for row in rows)
    total_calls = sum(row["total_tool_calls"] for row in rows)
    return {
        "cases": total,
        "normal_stop_rate": _metric(
            sum(row["gates"]["normal_stop"] for row in rows),
            total,
        ),
        "mandatory_tool_order_accuracy": _metric(
            sum(row["gates"]["mandatory_tool_order"] for row in rows),
            total,
        ),
        "tool_allowlist_accuracy": _metric(
            sum(row["gates"]["tool_allowlist"] for row in rows),
            total,
        ),
        "argument_schema_accuracy": _metric(
            sum(row["gates"]["argument_schema"] for row in rows),
            total,
        ),
        "argument_schema_call_accuracy": _metric(
            valid_calls,
            total_calls,
        ),
        "v2_composite_pass_rate": _metric(
            sum(row["v2_passed"] for row in rows),
            total,
        ),
    }


def score_output_set(
    *,
    label: str,
    cases: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    vehicle_catalog: set[str],
    schemas: dict[str, dict[str, Any]],
    max_steps: int,
) -> dict[str, Any]:
    """Apply the same frozen v2 case gates to one persisted output set."""
    legacy_evaluator.validate_case_records(
        cases,
        vehicle_catalog=vehicle_catalog,
    )
    case_by_id = legacy_evaluator._index_by_id(cases, label="cases")
    output_by_id = legacy_evaluator._index_by_id(outputs, label=label)
    if set(case_by_id) != set(output_by_id):
        missing = sorted(set(case_by_id) - set(output_by_id))
        extra = sorted(set(output_by_id) - set(case_by_id))
        raise ValueError(
            f"{label} IDs differ from cases: missing={missing} extra={extra}"
        )

    rows: list[dict[str, Any]] = []
    for case in cases:
        output = output_by_id[case["id"]]
        legacy_evaluator.validate_output_record(output, case)
        actual_tools = [
            legacy_evaluator._openai_call_name(call) or ""
            for call in output["tool_calls"]
        ]
        normal_stop, answer, answer_source, assistant_turns = _normal_stop(
            output,
            max_steps=max_steps,
        )
        mandatory_order = _ordered_subsequence_complete(
            actual_tools,
            case["expected_tools"],
        )
        allowed = set(case["expected_tools"] + case["optional_tools"])
        disallowed_tools = sorted(
            {name for name in actual_tools if name not in allowed}
        )
        call_results = [
            validate_tool_call_schema(call, schemas)
            for call in output["tool_calls"]
        ]
        argument_schema = all(valid for valid, _failure in call_results)
        gates = {
            "normal_stop": normal_stop,
            "mandatory_tool_order": mandatory_order,
            "tool_allowlist": not disallowed_tools,
            "argument_schema": argument_schema,
        }
        failures = [name for name, passed in gates.items() if not passed]
        rows.append(
            {
                "id": case["id"],
                "intent": case["intent"],
                "gates": gates,
                "v2_passed": all(gates.values()),
                "failure_gates": failures,
                "assistant_turns": assistant_turns,
                "answer_source": answer_source,
                "answer_characters": len(answer) if answer is not None else 0,
                "actual_tools": actual_tools,
                "disallowed_tools": disallowed_tools,
                "valid_argument_calls": sum(
                    valid for valid, _failure in call_results
                ),
                "total_tool_calls": len(call_results),
                "argument_schema_failures": [
                    {
                        "call_index": index,
                        "error": failure,
                    }
                    for index, (valid, failure) in enumerate(call_results)
                    if not valid
                ],
                "vehicle_entities_metadata": (
                    legacy_evaluator._known_models_in_answer(
                        answer,
                        vehicle_catalog,
                    )
                    if answer is not None
                    else []
                ),
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["intent"]].append(row)
    return {
        "label": label,
        "model": outputs[0]["model"] if outputs else None,
        "metrics": _summarize_rows(rows),
        "per_intent": {
            intent: _summarize_rows(intent_rows)
            for intent, intent_rows in sorted(grouped.items())
        },
        "cases": rows,
    }


def _compare_metrics(
    baseline_metrics: dict[str, Any],
    sft_metrics: dict[str, Any],
    *,
    scope: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    comparisons: dict[str, Any] = {}
    regressions: list[dict[str, Any]] = []
    for name in COMPARISON_METRICS:
        baseline_metric = baseline_metrics[name]
        sft_metric = sft_metrics[name]
        delta = round(
            sft_metric["percentage"] - baseline_metric["percentage"],
            1,
        )
        outcome = (
            "improved"
            if delta > 0
            else "regressed"
            if delta < 0
            else "tied"
        )
        row = {
            "baseline": baseline_metric,
            "sft": sft_metric,
            "absolute_percentage_point_change": delta,
            "outcome": outcome,
        }
        comparisons[name] = row
        if outcome == "regressed":
            regressions.append(
                {
                    "scope": scope,
                    "metric": name,
                    "absolute_percentage_point_change": delta,
                    "severity": "RED_REGRESSION",
                }
            )
    return comparisons, regressions


def write_new_json(path: Path, payload: dict[str, Any]) -> None:
    """Create a new report and refuse to replace any existing artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def run_rescore(
    *,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    manifest, manifest_sha = load_frozen_manifest()
    before_hashes = _validate_frozen_files(manifest)
    inputs = manifest["frozen_inputs"]
    cases = legacy_evaluator.load_jsonl(
        _repo_path(inputs["cases"]["path"]),
        label="v2 held-out cases",
    )
    baseline_outputs = legacy_evaluator.load_jsonl(
        _repo_path(inputs["baseline_outputs"]["path"]),
        label="v2 baseline outputs",
    )
    sft_outputs = legacy_evaluator.load_jsonl(
        _repo_path(inputs["sft_outputs"]["path"]),
        label="v2 SFT outputs",
    )
    expected_models = {
        "baseline": inputs["baseline_outputs"]["model"],
        "sft": inputs["sft_outputs"]["model"],
    }
    observed_models = {
        "baseline": sorted({row["model"] for row in baseline_outputs}),
        "sft": sorted({row["model"] for row in sft_outputs}),
    }
    for label, expected in expected_models.items():
        if observed_models[label] != [expected]:
            raise ValueError(
                f"{label} model mismatch: expected={expected!r} "
                f"actual={observed_models[label]!r}"
            )

    catalog = set(legacy_evaluator.load_vehicle_catalog())
    schemas = schema_map()
    max_steps = manifest["runner_context"]["max_steps"]
    baseline = score_output_set(
        label="baseline",
        cases=cases,
        outputs=baseline_outputs,
        vehicle_catalog=catalog,
        schemas=schemas,
        max_steps=max_steps,
    )
    sft = score_output_set(
        label="sft_epoch_3",
        cases=cases,
        outputs=sft_outputs,
        vehicle_catalog=catalog,
        schemas=schemas,
        max_steps=max_steps,
    )
    comparison, regressions = _compare_metrics(
        baseline["metrics"],
        sft["metrics"],
        scope="overall",
    )
    per_intent_comparison: dict[str, Any] = {}
    for intent in sorted(baseline["per_intent"]):
        intent_comparison, intent_regressions = _compare_metrics(
            baseline["per_intent"][intent],
            sft["per_intent"][intent],
            scope=f"intent:{intent}",
        )
        per_intent_comparison[intent] = intent_comparison
        regressions.extend(intent_regressions)
    after_hashes = _validate_frozen_files(manifest)
    if before_hashes != after_hashes:
        raise ValueError("a frozen input or historical artifact changed during scoring")

    composite_outcome = comparison["v2_composite_pass_rate"]["outcome"]
    report = {
        "status": "completed",
        "scoring_contract": manifest["manifest_version"],
        "manifest": {
            "path": str(MANIFEST_PATH.relative_to(ROOT)),
            "sha256": manifest_sha,
            "frozen_before_scoring": True,
        },
        "execution": {
            "offline_only": True,
            "model_inference": False,
            "tool_execution": False,
            "same_rules_for_both_models": True,
            "scorer_path": str(Path(__file__).resolve().relative_to(ROOT)),
            "scorer_sha256": sha256_file(Path(__file__).resolve()),
        },
        "historical_integrity": {
            "verified_before_and_after": True,
            "observed_sha256": after_hashes,
            "strict_scores_preserved": {
                "baseline": "6/40",
                "sft": "0/40",
            },
        },
        "scope_limit": (
            "v2 measures the requested production execution contract only; "
            "it does not score answer grounding or semantic quality."
        ),
        "observability_limits": {
            "generation_prompt": (
                "Both persisted output sets were originally generated with the "
                "legacy evaluator's strict-JSON terminal instruction appended. "
                "This task changes only offline scoring and does not claim the "
                "outputs came from an unmodified production prompt."
            ),
            "normal_stop": (
                "Persisted outputs do not contain model finish_reason. Therefore "
                "normal_stop means the runner observed a non-empty terminal "
                "assistant turn within max_steps; it cannot distinguish EOS from "
                "the per-turn token limit."
            ),
        },
        "baseline": baseline,
        "sft": sft,
        "comparison": comparison,
        "per_intent_comparison": per_intent_comparison,
        "honesty_self_check": {
            "rules_derived_before_scoring": True,
            "manifest_sha256": manifest_sha,
            "model_specific_exceptions": [],
            "sft_regressions": regressions,
            "sft_regression_count": len(regressions),
            "all_regressions_explicit": True,
        },
        "assessment": {
            "v2_composite_outcome": composite_outcome,
            "v2_composite_classification": {
                "improved": "提升",
                "tied": "持平",
                "regressed": "回退",
            }[composite_outcome],
            "mixed_dimension_result": bool(regressions),
            "classification": (
                "综合提升，但存在显式回退维度"
                if composite_outcome == "improved" and regressions
                else {
                    "improved": "提升",
                    "tied": "持平",
                    "regressed": "回退",
                }[composite_outcome]
            ),
        },
    }
    write_new_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline product-faithful v2 symmetric held-out rescoring.",
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()
    report = run_rescore(report_path=args.report)
    print(
        json.dumps(
            {
                "manifest": report["manifest"],
                "baseline": report["baseline"]["metrics"],
                "sft": report["sft"]["metrics"],
                "comparison": report["comparison"],
                "assessment": report["assessment"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
