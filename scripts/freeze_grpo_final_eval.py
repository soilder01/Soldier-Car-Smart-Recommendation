#!/usr/bin/env python3
"""Freeze a new GRPO final eval set from never-materialized candidate tails."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any

from data_synth.generate_500perintent_sft import (
    _compare_expected_model_names,
    build_queries,
)
from scripts import evaluate_model_outputs as evaluator


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = (
    ROOT / "data" / "model_training" / "eval" / "grpo_final_held_out.jsonl"
)
MANIFEST_PATH = (
    ROOT
    / "data"
    / "model_training"
    / "eval"
    / "grpo_final_held_out_manifest.json"
)
OUTPUT_SHA_PATH = OUTPUT_PATH.with_suffix(".sha256")
MANIFEST_SHA_PATH = MANIFEST_PATH.with_suffix(".sha256")
SELECTION_SEED = "grpo-final-v1"
HISTORICAL_MAX_CANDIDATE_INDEX = 650
PER_INTENT = 10
SOURCE_SPECS = {
    "recommend": {
        "source_intent": "recommend",
        "candidate_count": 41472,
        "source_manifest": (
            "data/model_training/"
            "teacher_decision_500perintent_recommend_manifest.json"
        ),
    },
    "compare": {
        "source_intent": "compare",
        "candidate_count": 666,
        "source_manifest": (
            "data/model_training/"
            "teacher_decision_500perintent_compare_named_lookup_v3_manifest.json"
        ),
    },
    "knowledge": {
        "source_intent": "customer_service",
        "candidate_count": 4900,
        "source_manifest": (
            "data/model_training/"
            "teacher_decision_500perintent_customer_service_rerun_v3_manifest.json"
        ),
    },
    "sales": {
        "source_intent": "sales",
        "candidate_count": 4900,
        "source_manifest": (
            "data/model_training/"
            "teacher_decision_500perintent_sales_policy_rerun_v3_manifest.json"
        ),
    },
}


def normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.strip().split()).casefold()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def selection_digest(
    *,
    intent: str,
    candidate_index: int,
    query: str,
) -> str:
    payload = (
        f"{SELECTION_SEED}:{intent}:{candidate_index}:{normalize(query)}"
    )
    return sha256_bytes(payload.encode("utf-8"))


def _canonical_compare_models(query: str) -> list[str]:
    requested = _compare_expected_model_names(query)
    catalog = evaluator.load_vehicle_catalog()
    by_compact = {
        normalize(model).replace(" ", ""): model
        for model in catalog
    }
    resolved = [
        by_compact.get(normalize(model).replace(" ", ""))
        for model in requested
    ]
    if len(resolved) != 2 or any(model is None for model in resolved):
        raise ValueError(f"compare query models did not resolve: {query}")
    return [model for model in resolved if model is not None]


def _build_case(
    *,
    intent: str,
    ordinal: int,
    query: str,
) -> dict[str, Any]:
    return {
        "id": f"grpo-final-{intent}-{ordinal:03d}",
        "query": query,
        "intent": intent,
        "expected_tools": evaluator.MANDATORY_BY_INTENT[intent],
        "optional_tools": evaluator.OPTIONAL_BY_INTENT[intent],
        "forbidden_tools": evaluator.FORBIDDEN_BY_INTENT[intent],
        "allowed_models": (
            _canonical_compare_models(query)
            if intent == "compare"
            else []
        ),
    }


def build_split() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    source_manifests: dict[str, dict[str, Any]] = {}
    for intent, spec in SOURCE_SPECS.items():
        source_manifest_path = ROOT / spec["source_manifest"]
        source_manifest = json.loads(
            source_manifest_path.read_text(encoding="utf-8")
        )
        if source_manifest.get("max_candidates") != (
            HISTORICAL_MAX_CANDIDATE_INDEX
        ):
            raise ValueError(
                f"historical candidate boundary drift for {intent}"
            )
        if len(source_manifest.get("queries", [])) != (
            HISTORICAL_MAX_CANDIDATE_INDEX
        ):
            raise ValueError(
                f"historical manifest query count drift for {intent}"
            )
        source_manifests[intent] = {
            "path": spec["source_manifest"],
            "sha256": sha256_file(source_manifest_path),
            "max_candidate_index": HISTORICAL_MAX_CANDIDATE_INDEX,
        }

        queries = build_queries(
            spec["source_intent"],
            spec["candidate_count"],
        )
        candidates = [
            {
                "candidate_index": index,
                "query": query,
                "selection_sha256": selection_digest(
                    intent=intent,
                    candidate_index=index,
                    query=query,
                ),
            }
            for index, query in enumerate(queries, start=1)
            if index > HISTORICAL_MAX_CANDIDATE_INDEX
        ]
        if len(candidates) < PER_INTENT:
            raise ValueError(f"not enough unused candidates for {intent}")
        selected = sorted(
            candidates,
            key=lambda item: (
                item["selection_sha256"],
                item["candidate_index"],
            ),
        )[:PER_INTENT]
        for ordinal, item in enumerate(selected, start=1):
            case = _build_case(
                intent=intent,
                ordinal=ordinal,
                query=item["query"],
            )
            rows.append(case)
            provenance.append(
                {
                    "id": case["id"],
                    "intent": intent,
                    "source_intent": spec["source_intent"],
                    "candidate_index": item["candidate_index"],
                    "selection_sha256": item["selection_sha256"],
                    "normalized_id_sha256": sha256_bytes(
                        normalize(case["id"]).encode("utf-8")
                    ),
                    "normalized_query_sha256": sha256_bytes(
                        normalize(case["query"]).encode("utf-8")
                    ),
                }
            )

    evaluator.validate_case_records(
        rows,
        vehicle_catalog=set(evaluator.load_vehicle_catalog()),
    )
    if len(rows) != PER_INTENT * len(SOURCE_SPECS):
        raise ValueError("GRPO final eval row count drift")
    if len({normalize(row["id"]) for row in rows}) != len(rows):
        raise ValueError("GRPO final eval contains duplicate normalized IDs")
    if len({normalize(row["query"]) for row in rows}) != len(rows):
        raise ValueError("GRPO final eval contains duplicate normalized queries")
    manifest = {
        "format_version": 1,
        "status": "frozen_without_model_inference",
        "purpose": "GRPO final evaluation only; forbidden for rollout, reward, tuning, checkpoint selection, or early stopping",
        "selection": {
            "seed": SELECTION_SEED,
            "algorithm": (
                "For each canonical intent, enumerate the deterministic source "
                "generator, discard candidate_index <= 650, sort remaining "
                "candidates by SHA256(seed:intent:index:normalized_query), and "
                "take the first 10."
            ),
            "per_intent": PER_INTENT,
            "historical_max_candidate_index": (
                HISTORICAL_MAX_CANDIDATE_INDEX
            ),
        },
        "source_generator": {
            "path": "data_synth/generate_500perintent_sft.py",
            "sha256": sha256_file(
                ROOT / "data_synth" / "generate_500perintent_sft.py"
            ),
        },
        "source_manifests": source_manifests,
        "counts": {
            "total": len(rows),
            "per_intent": {
                intent: sum(row["intent"] == intent for row in rows)
                for intent in SOURCE_SPECS
            },
        },
        "provenance": provenance,
    }
    return rows, manifest


def main() -> None:
    for path in (
        OUTPUT_PATH,
        OUTPUT_SHA_PATH,
        MANIFEST_PATH,
        MANIFEST_SHA_PATH,
    ):
        if path.exists():
            raise FileExistsError(f"freeze output already exists: {path}")
    rows, manifest = build_split()
    OUTPUT_PATH.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    manifest["dataset"] = {
        "path": str(OUTPUT_PATH.relative_to(ROOT)),
        "sha256": sha256_file(OUTPUT_PATH),
        "records": len(rows),
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUTPUT_SHA_PATH.write_text(
        f"{manifest['dataset']['sha256']}  {OUTPUT_PATH.name}\n",
        encoding="ascii",
    )
    MANIFEST_SHA_PATH.write_text(
        f"{sha256_file(MANIFEST_PATH)}  {MANIFEST_PATH.name}\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "dataset": manifest["dataset"],
                "manifest": {
                    "path": str(MANIFEST_PATH.relative_to(ROOT)),
                    "sha256": sha256_file(MANIFEST_PATH),
                },
                "counts": manifest["counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
