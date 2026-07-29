#!/usr/bin/env python3
"""Score both production-prompt v3 campaigns with the frozen v2 rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts import evaluate_model_outputs as evaluator
from scripts import rescore_product_faithful_v2 as v2
from scripts import run_production_prompt_v3_eval as v3


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_PATH = (
    ROOT
    / "data"
    / "model_training"
    / "production_prompt_v3_product_faithful_v2_report.json"
)


def _validate_receipt(
    *,
    mode: str,
    paths: v3.CampaignPaths,
) -> dict[str, Any]:
    receipt = v3.load_json(paths.receipt_path)
    if receipt.get("status") != "completed" or receipt.get("mode") != mode:
        raise ValueError(f"incomplete or mismatched v3 receipt: {mode}")
    if receipt["harness"]["sha256"] != v3.EXPECTED_HARNESS_SHA256:
        raise ValueError(f"v3 receipt harness SHA mismatch: {mode}")
    if receipt["prompts"]["sha256"] != v3.EXPECTED_PROMPTS_SHA256:
        raise ValueError(f"v3 receipt prompt SHA mismatch: {mode}")
    if (
        receipt["v2_scoring_manifest_sha256"]
        != v2.EXPECTED_MANIFEST_SHA256
    ):
        raise ValueError(f"v3 receipt v2 manifest SHA mismatch: {mode}")
    if v3.sha256_file(paths.output_path) != receipt["output"]["sha256"]:
        raise ValueError(f"v3 output SHA differs from receipt: {mode}")
    if receipt["output"]["records"] != 40:
        raise ValueError(f"v3 receipt does not contain 40 records: {mode}")
    return receipt


def _compare_all(
    baseline: dict[str, Any],
    sft: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    overall, regressions = v2._compare_metrics(
        baseline["metrics"],
        sft["metrics"],
        scope="overall",
    )
    per_intent: dict[str, Any] = {}
    for intent in sorted(baseline["per_intent"]):
        comparison, intent_regressions = v2._compare_metrics(
            baseline["per_intent"][intent],
            sft["per_intent"][intent],
            scope=f"intent:{intent}",
        )
        per_intent[intent] = comparison
        regressions.extend(intent_regressions)
    return overall, per_intent, regressions


def score_campaigns(
    *,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, Any]:
    harness, prompts = v3.validate_frozen_v3(verify_model_files=False)
    manifest, manifest_sha = v2.load_frozen_manifest()
    if manifest_sha != v3.EXPECTED_V2_MANIFEST_SHA256:
        raise ValueError("v3 and v2 scorer disagree on manifest SHA")
    old_artifact_hashes_before = v2._validate_frozen_files(manifest)
    cases_path = ROOT / harness["evaluation"]["cases"]
    cases = evaluator.load_jsonl(cases_path, label="v3 cases")
    catalog = set(evaluator.load_vehicle_catalog())
    schemas = v2.schema_map()
    scores: dict[str, Any] = {}
    receipts: dict[str, Any] = {}
    output_provenance: dict[str, Any] = {}
    for mode, score_label in (
        ("baseline", "baseline_production_prompt_v3"),
        ("sft_epoch_3", "sft_epoch_3_production_prompt_v3"),
    ):
        paths = v3.campaign_paths(harness, mode)
        receipts[mode] = _validate_receipt(mode=mode, paths=paths)
        outputs = v3._validate_completed_output(
            output_path=paths.output_path,
            cases=cases,
            model_alias=harness["models"][mode]["alias"],
            prompts=prompts,
        )
        scores[mode] = v2.score_output_set(
            label=score_label,
            cases=cases,
            outputs=outputs,
            vehicle_catalog=catalog,
            schemas=schemas,
            max_steps=harness["runner"]["max_steps"],
        )
        output_provenance[mode] = {
            "path": str(paths.output_path.relative_to(ROOT)),
            "sha256": v3.sha256_file(paths.output_path),
            "records": len(outputs),
        }

    comparison, per_intent_comparison, regressions = _compare_all(
        scores["baseline"],
        scores["sft_epoch_3"],
    )
    old_artifact_hashes_after = v2._validate_frozen_files(manifest)
    if old_artifact_hashes_before != old_artifact_hashes_after:
        raise ValueError("a historical frozen artifact changed during v3 scoring")
    composite_outcome = comparison["v2_composite_pass_rate"]["outcome"]
    knowledge_stop = per_intent_comparison["knowledge"]["normal_stop_rate"]
    report = {
        "status": "completed",
        "campaign": "production-prompt-heldout-v3.0",
        "v3_harness": {
            "path": str(v3.HARNESS_PATH.relative_to(ROOT)),
            "sha256": v3.EXPECTED_HARNESS_SHA256,
        },
        "production_prompts": {
            "path": str(v3.PROMPTS_PATH.relative_to(ROOT)),
            "sha256": v3.EXPECTED_PROMPTS_SHA256,
        },
        "scoring": {
            "manifest_path": str(v2.MANIFEST_PATH.relative_to(ROOT)),
            "manifest_sha256": manifest_sha,
            "manifest_modified": False,
            "same_score_output_set_for_both_models": True,
            "note": (
                "The frozen v2 rule definitions are applied to new v3 outputs. "
                "The old-output hashes inside that manifest remain unchanged "
                "historical provenance and are not rewritten."
            ),
        },
        "execution_invariant": {
            "only_prompt_changed_from_parent_inference_harness": True,
            "model_inference": True,
            "training": False,
            "data_mutation": False,
            "weight_mutation": False,
            "output_provenance": output_provenance,
            "receipts": receipts,
        },
        "historical_integrity": {
            "verified_before_and_after_scoring": True,
            "observed_sha256": old_artifact_hashes_after,
        },
        "baseline": scores["baseline"],
        "sft": scores["sft_epoch_3"],
        "comparison": comparison,
        "per_intent_comparison": per_intent_comparison,
        "knowledge_stop_focus": {
            "baseline": knowledge_stop["baseline"],
            "sft": knowledge_stop["sft"],
            "absolute_percentage_point_change": knowledge_stop[
                "absolute_percentage_point_change"
            ],
            "outcome": knowledge_stop["outcome"],
        },
        "honesty_self_check": {
            "model_specific_scoring_exceptions": [],
            "sft_regressions": regressions,
            "sft_regression_count": len(regressions),
            "all_regressions_explicit": True,
        },
        "scope_limit": (
            "This v2 execution-contract score does not measure grounding or "
            "semantic answer quality."
        ),
        "assessment": {
            "v2_composite_outcome": composite_outcome,
            "v2_composite_classification": {
                "improved": "提升",
                "tied": "持平",
                "regressed": "回退",
            }[composite_outcome],
            "mixed_dimension_result": bool(regressions),
        },
    }
    v2.write_new_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score frozen production-prompt v3 outputs with v2 rules.",
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()
    report = score_campaigns(report_path=args.report)
    print(
        json.dumps(
            {
                "baseline": report["baseline"]["metrics"],
                "sft": report["sft"]["metrics"],
                "comparison": report["comparison"],
                "knowledge_stop_focus": report["knowledge_stop_focus"],
                "honesty_self_check": report["honesty_self_check"],
                "assessment": report["assessment"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
