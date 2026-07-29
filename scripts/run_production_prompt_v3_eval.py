#!/usr/bin/env python3
"""Run one frozen production-prompt v3 held-out campaign per model."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services import agent_graph
from scripts import evaluate_model_outputs as evaluator
from scripts import run_local_qwen_baseline as baseline_runner
from scripts import run_model_layer_eval as runner
from scripts import run_sft_heldout_eval as sft_runner
from training.sft import train_qlora_sft as sft


ROOT = Path(__file__).resolve().parents[1]
PROMPTS_PATH = (
    ROOT / "data" / "model_training" / "eval" / "production_prompts_v3.json"
)
HARNESS_PATH = (
    ROOT
    / "data"
    / "model_training"
    / "eval"
    / "frozen_production_prompt_harness_v3.json"
)
PARENT_HARNESS_PATH = (
    ROOT / "data" / "model_training" / "eval" / "frozen_qwen_heldout_harness.json"
)
EXPECTED_PROMPTS_SHA256 = (
    "22ace6242e57b2bee32dde880487970dbc195de257a4b4a1264ebf6e75da8c2b"
)
EXPECTED_HARNESS_SHA256 = (
    "55453e5047e12a636cdaf9c73003cc3768e8e8417cac6bb5f74ce74f68c1fe3c"
)
EXPECTED_V2_MANIFEST_SHA256 = (
    "49d3a2b1da23490b236f839f73d55654743a596899909fa80b4fcc090d721113"
)
MODES = ("baseline", "sft_epoch_3")


@dataclass(frozen=True)
class CampaignPaths:
    output_path: Path
    receipt_path: Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_sha_record(path: Path, expected: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"frozen SHA mismatch for {path}: expected={expected} actual={actual}"
        )
    sha_path = path.with_suffix(".sha256")
    recorded = sha_path.read_text(encoding="ascii").strip().split()
    if not recorded or recorded[0] != expected:
        raise ValueError(f"companion SHA record mismatch: {sha_path}")


def _validate_file_entries(entries: list[dict[str, str]]) -> None:
    for entry in entries:
        path = ROOT / entry["path"]
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            raise ValueError(
                f"model file SHA mismatch for {entry['path']}: "
                f"expected={entry['sha256']} actual={actual}"
            )


def validate_frozen_v3(
    *,
    verify_model_files: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fail closed unless every frozen v3 input and invariant still matches."""
    _validate_sha_record(PROMPTS_PATH, EXPECTED_PROMPTS_SHA256)
    _validate_sha_record(HARNESS_PATH, EXPECTED_HARNESS_SHA256)
    prompts = load_json(PROMPTS_PATH)
    harness = load_json(HARNESS_PATH)
    if prompts.get("status") != "frozen_before_inference":
        raise ValueError("production prompts were not frozen before inference")
    if harness.get("status") != "frozen_before_inference":
        raise ValueError("v3 harness was not frozen before inference")
    if harness["prompt"]["artifact_sha256"] != EXPECTED_PROMPTS_SHA256:
        raise ValueError("v3 harness references a different prompt artifact")
    if harness["prompt"]["append_evaluator_terminal_instruction"] is not False:
        raise ValueError("v3 harness must disable the evaluator terminal suffix")

    parent_path = ROOT / harness["parent_harness"]["path"]
    if sha256_file(parent_path) != harness["parent_harness"]["sha256"]:
        raise ValueError("parent frozen harness SHA drift")
    parent = load_json(parent_path)
    for field in ("evaluation", "runner", "generation", "tool_protocol"):
        if harness[field] != parent[field]:
            raise ValueError(f"v3 non-prompt setting drift: {field}")

    cases_path = ROOT / harness["evaluation"]["cases"]
    if sha256_file(cases_path) != harness["evaluation"]["cases_sha256"]:
        raise ValueError("held-out cases SHA drift")
    cases = evaluator.load_jsonl(cases_path, label="v3 held-out cases")
    if len(cases) != harness["evaluation"]["case_count"]:
        raise ValueError("held-out case count drift")

    v2_path = ROOT / harness["scoring"]["rules_manifest_path"]
    if (
        harness["scoring"]["rules_manifest_sha256"]
        != EXPECTED_V2_MANIFEST_SHA256
        or sha256_file(v2_path) != EXPECTED_V2_MANIFEST_SHA256
    ):
        raise ValueError("frozen v2 scoring manifest SHA drift")

    source_path = ROOT / prompts["source"]["path"]
    if sha256_file(source_path) != prompts["source"]["sha256"]:
        raise ValueError("production prompt source SHA drift")
    case_intents = {case["intent"] for case in cases}
    if case_intents != set(prompts["intents"]):
        raise ValueError("frozen prompt intents differ from held-out intents")
    for intent in sorted(case_intents):
        frozen = prompts["intents"][intent]
        text = frozen["text"]
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != frozen[
            "utf8_sha256"
        ]:
            raise ValueError(f"frozen prompt text SHA drift for {intent}")
        if text != agent_graph._get_prompt_for_intent(intent):
            raise ValueError(f"frozen prompt differs from production for {intent}")
        if "评测终态协议" in text or "mentioned_models" in text:
            raise ValueError(f"frozen prompt contains evaluator protocol for {intent}")

    if verify_model_files:
        _validate_file_entries(harness["models"]["shared_base_files"])
        _validate_file_entries(
            harness["models"]["sft_epoch_3"]["adapter_files"]
        )
    return harness, prompts


def campaign_paths(
    harness: dict[str, Any],
    mode: str,
) -> CampaignPaths:
    if mode not in MODES:
        raise ValueError(f"unsupported v3 model mode: {mode}")
    output_path = ROOT / harness["outputs"][mode]
    receipt_name = output_path.name.replace(
        "_outputs.jsonl",
        "_run_receipt.json",
    )
    if receipt_name == output_path.name:
        raise ValueError("v3 output path does not end in _outputs.jsonl")
    return CampaignPaths(
        output_path=output_path,
        receipt_path=output_path.with_name(receipt_name),
    )


def _prompt_mapping(prompts: dict[str, Any]) -> dict[str, str]:
    return {
        intent: payload["text"]
        for intent, payload in prompts["intents"].items()
    }


def _validate_completed_output(
    *,
    output_path: Path,
    cases: list[dict[str, Any]],
    model_alias: str,
    prompts: dict[str, Any],
) -> list[dict[str, Any]]:
    outputs = evaluator.load_jsonl(output_path, label="v3 outputs")
    if len(outputs) != len(cases):
        raise ValueError(
            f"v3 output is incomplete: expected={len(cases)} actual={len(outputs)}"
        )
    case_by_id = {case["id"]: case for case in cases}
    if set(case_by_id) != {output["id"] for output in outputs}:
        raise ValueError("v3 output IDs differ from held-out cases")
    frozen_prompts = _prompt_mapping(prompts)
    for output in outputs:
        case = case_by_id[output["id"]]
        evaluator.validate_output_record(output, case)
        if output["model"] != model_alias:
            raise ValueError(f"v3 output model alias drift: {output['id']}")
        if output["trajectory"][0] != {
            "role": "system",
            "content": frozen_prompts[case["intent"]],
        }:
            raise ValueError(f"v3 output prompt mismatch: {output['id']}")
    return outputs


def _write_receipt(
    *,
    paths: CampaignPaths,
    mode: str,
    model_alias: str,
    output_records: list[dict[str, Any]],
    runner_summary: dict[str, int],
) -> dict[str, Any]:
    receipt = {
        "status": "completed",
        "campaign": "production-prompt-heldout-v3.0",
        "mode": mode,
        "model_alias": model_alias,
        "harness": {
            "path": str(HARNESS_PATH.relative_to(ROOT)),
            "sha256": EXPECTED_HARNESS_SHA256,
        },
        "prompts": {
            "path": str(PROMPTS_PATH.relative_to(ROOT)),
            "sha256": EXPECTED_PROMPTS_SHA256,
        },
        "v2_scoring_manifest_sha256": EXPECTED_V2_MANIFEST_SHA256,
        "output": {
            "path": str(paths.output_path.relative_to(ROOT)),
            "sha256": sha256_file(paths.output_path),
            "records": len(output_records),
        },
        "runner_summary": runner_summary,
        "prompt_verified_per_case": True,
        "legacy_terminal_parser_is_not_the_v3_scorer": True,
        "training": False,
        "weight_mutation": False,
    }
    with paths.receipt_path.open("x", encoding="utf-8") as file:
        json.dump(receipt, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return receipt


def run_campaign(mode: str) -> dict[str, Any]:
    harness, prompts = validate_frozen_v3(verify_model_files=True)
    paths = campaign_paths(harness, mode)
    if paths.receipt_path.exists():
        raise FileExistsError(
            f"v3 campaign is single-use and already completed: "
            f"{paths.receipt_path}"
        )
    cases_path = ROOT / harness["evaluation"]["cases"]
    cases = baseline_runner.validate_heldout_only(cases_path)
    model_alias = harness["models"][mode]["alias"]

    existing_count = 0
    if paths.output_path.exists():
        existing = evaluator.load_jsonl(paths.output_path, label="partial v3 outputs")
        existing_count = len(existing)
        if existing_count > len(cases):
            raise ValueError("v3 output has more rows than held-out cases")
    if existing_count < len(cases):
        if mode == "baseline":
            client = baseline_runner._load_local_client(
                config_path=sft.DEFAULT_CONFIG_PATH,
                max_new_tokens=harness["generation"]["max_new_tokens"],
            )
        else:
            client = sft_runner._load_adapter_client(
                adapter_path=ROOT / "checkpoints" / "sft" / "best_adapter",
                config_path=sft.DEFAULT_CONFIG_PATH,
            )
        try:
            runner_summary = runner.run_evaluation(
                cases_path=cases_path,
                output_path=paths.output_path,
                model=model_alias,
                base_url=baseline_runner.LOCAL_ENDPOINT_ALIAS,
                client=client,
                max_steps=harness["runner"]["max_steps"],
                system_prompts_by_intent=_prompt_mapping(prompts),
                append_evaluation_terminal_instruction=False,
            )
        finally:
            client.close()
    else:
        runner_summary = {
            "total_cases": len(cases),
            "existing": existing_count,
            "written": 0,
            "failed_cases": sum(
                not row["schema_valid"] or bool(row["runner_errors"])
                for row in evaluator.load_jsonl(
                    paths.output_path,
                    label="complete v3 outputs",
                )
            ),
        }

    outputs = _validate_completed_output(
        output_path=paths.output_path,
        cases=cases,
        model_alias=model_alias,
        prompts=prompts,
    )
    return _write_receipt(
        paths=paths,
        mode=mode,
        model_alias=model_alias,
        output_records=outputs,
        runner_summary=runner_summary,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one frozen production-prompt v3 held-out campaign.",
    )
    parser.add_argument("--mode", choices=MODES, required=True)
    args = parser.parse_args()
    receipt = run_campaign(args.mode)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
