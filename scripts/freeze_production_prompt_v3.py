#!/usr/bin/env python3
"""Freeze exact production prompts and the prompt-only held-out v3 harness."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services import agent_graph


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
V2_MANIFEST_PATH = (
    ROOT
    / "data"
    / "model_training"
    / "eval"
    / "product_faithful_v2_scoring_manifest.json"
)
INTENTS = ("recommend", "compare", "knowledge", "sales")
V2_MANIFEST_SHA256 = (
    "49d3a2b1da23490b236f839f73d55654743a596899909fa80b4fcc090d721113"
)
MODEL_FILES = (
    "models/Qwen2.5-7B-Instruct/model-00001-of-00004.safetensors",
    "models/Qwen2.5-7B-Instruct/model-00002-of-00004.safetensors",
    "models/Qwen2.5-7B-Instruct/model-00003-of-00004.safetensors",
    "models/Qwen2.5-7B-Instruct/model-00004-of-00004.safetensors",
    "models/Qwen2.5-7B-Instruct/model.safetensors.index.json",
    "models/Qwen2.5-7B-Instruct/config.json",
    "models/Qwen2.5-7B-Instruct/tokenizer.json",
)
ADAPTER_FILES = (
    "checkpoints/sft/best_adapter/adapter_model.safetensors",
    "checkpoints/sft/best_adapter/adapter_config.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _write_sha_file(path: Path, digest: str) -> None:
    sha_path = path.with_suffix(".sha256")
    with sha_path.open("x", encoding="ascii") as file:
        file.write(f"{digest}  {path.name}\n")


def _file_manifest(paths: tuple[str, ...]) -> list[dict[str, str]]:
    return [
        {"path": value, "sha256": _sha256(ROOT / value)}
        for value in paths
    ]


def build_prompts_payload(frozen_at: str) -> dict[str, Any]:
    source_path = ROOT / "backend" / "app" / "services" / "agent_graph.py"
    prompts: dict[str, Any] = {}
    for intent in INTENTS:
        prompt = agent_graph._get_prompt_for_intent(intent)
        forbidden_fragments = (
            "评测终态协议",
            "严格 JSON object",
            '"mentioned_models"',
        )
        present = [
            fragment for fragment in forbidden_fragments if fragment in prompt
        ]
        if present:
            raise ValueError(
                f"production prompt {intent} contains evaluator suffix: {present}"
            )
        prompts[intent] = {
            "utf8_sha256": _text_sha256(prompt),
            "characters": len(prompt),
            "text": prompt,
        }
    return {
        "artifact_version": "production-prompts-v3.0",
        "status": "frozen_before_inference",
        "frozen_at_utc": frozen_at,
        "source": {
            "path": str(source_path.relative_to(ROOT)),
            "sha256": _sha256(source_path),
            "symbol": "_get_prompt_for_intent",
        },
        "prompt_construction": (
            "Exact return value of agent_graph._get_prompt_for_intent(intent); "
            "no evaluator terminal suffix is appended."
        ),
        "intents": prompts,
    }


def build_harness_payload(
    *,
    frozen_at: str,
    prompts_sha256: str,
) -> dict[str, Any]:
    parent = json.loads(PARENT_HARNESS_PATH.read_text(encoding="utf-8"))
    if _sha256(V2_MANIFEST_PATH) != V2_MANIFEST_SHA256:
        raise ValueError("frozen v2 scoring manifest SHA drift")
    return {
        "harness_version": "production-prompt-heldout-v3.0",
        "status": "frozen_before_inference",
        "frozen_at_utc": frozen_at,
        "parent_harness": {
            "path": str(PARENT_HARNESS_PATH.relative_to(ROOT)),
            "sha256": _sha256(PARENT_HARNESS_PATH),
            "version": parent["harness_version"],
        },
        "evaluation": parent["evaluation"],
        "runner": parent["runner"],
        "generation": parent["generation"],
        "tool_protocol": parent["tool_protocol"],
        "prompt": {
            "artifact_path": str(PROMPTS_PATH.relative_to(ROOT)),
            "artifact_sha256": prompts_sha256,
            "mode": "exact_frozen_production_prompt",
            "append_evaluator_terminal_instruction": False,
        },
        "only_inference_change_from_parent": {
            "field": "system_prompt",
            "before": "production prompt plus legacy evaluator terminal suffix",
            "after": "exact frozen production prompt",
            "unchanged": [
                "held-out cases and order",
                "greedy decoding",
                "max_steps",
                "max_new_tokens",
                "tool-call stop sequence",
                "tool schemas",
                "real tool execution",
                "tool observation feedback",
                "base model weights",
                "SFT adapter weights",
            ],
        },
        "scoring": {
            "rules_manifest_path": str(V2_MANIFEST_PATH.relative_to(ROOT)),
            "rules_manifest_sha256": V2_MANIFEST_SHA256,
            "same_rules_for_both_models": True,
        },
        "models": {
            "shared_base_files": _file_manifest(MODEL_FILES),
            "baseline": {
                "alias": "local_base_nf4_production_prompt_v3",
                "base_files_ref": "shared_base_files",
            },
            "sft_epoch_3": {
                "alias": "local_sft_epoch_3_production_prompt_v3",
                "base_files_ref": "shared_base_files",
                "adapter_files": _file_manifest(ADAPTER_FILES),
            },
        },
        "outputs": {
            "baseline": (
                "data/model_training/"
                "baseline_production_prompt_v3_heldout_outputs.jsonl"
            ),
            "sft_epoch_3": (
                "data/model_training/"
                "sft_production_prompt_v3_heldout_outputs.jsonl"
            ),
        },
        "safety": {
            "training": False,
            "data_mutation": False,
            "weight_mutation": False,
            "legacy_output_overwrite": False,
        },
    }


def main() -> None:
    for path in (
        PROMPTS_PATH,
        PROMPTS_PATH.with_suffix(".sha256"),
        HARNESS_PATH,
        HARNESS_PATH.with_suffix(".sha256"),
    ):
        if path.exists():
            raise FileExistsError(f"freeze artifact already exists: {path}")
    frozen_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    prompts = build_prompts_payload(frozen_at)
    _write_new_json(PROMPTS_PATH, prompts)
    prompts_sha = _sha256(PROMPTS_PATH)
    _write_sha_file(PROMPTS_PATH, prompts_sha)
    harness = build_harness_payload(
        frozen_at=frozen_at,
        prompts_sha256=prompts_sha,
    )
    _write_new_json(HARNESS_PATH, harness)
    harness_sha = _sha256(HARNESS_PATH)
    _write_sha_file(HARNESS_PATH, harness_sha)
    print(
        json.dumps(
            {
                "prompts": {
                    "path": str(PROMPTS_PATH.relative_to(ROOT)),
                    "sha256": prompts_sha,
                },
                "harness": {
                    "path": str(HARNESS_PATH.relative_to(ROOT)),
                    "sha256": harness_sha,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
