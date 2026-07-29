#!/usr/bin/env python3
"""Fail-closed static isolation checks for GRPO training and final eval data."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from scripts import evaluate_model_outputs as evaluator
from scripts import freeze_grpo_final_eval as freeze
from scripts import freeze_grpo_reward_split as reward_split


ROOT = Path(__file__).resolve().parents[1]
EVAL_PATHS = {
    "reward_visible": (
        ROOT / "data" / "model_training" / "eval" / "reward_visible.jsonl"
    ),
    "sft_held_out": (
        ROOT / "data" / "model_training" / "eval" / "held_out.jsonl"
    ),
    "grpo_final": freeze.OUTPUT_PATH,
}
SFT_SOURCE_PATHS = (
    ROOT / "data" / "model_training" / "sft_train.jsonl",
    ROOT / "data" / "model_training" / "sft_val.jsonl",
    ROOT / "data" / "model_training" / "truncated_excluded.jsonl",
)
DEFAULT_REPORT_PATH = (
    ROOT
    / "data"
    / "model_training"
    / "eval"
    / "grpo_final_isolation_report.json"
)
EXPECTED_COUNTS = {
    "reward_visible": 20,
    "sft_held_out": 40,
    "grpo_final": 40,
}
EXPECTED_PER_INTENT = {
    "reward_visible": {
        "recommend": 5,
        "compare": 5,
        "knowledge": 5,
        "sales": 5,
    },
    "sft_held_out": {
        "recommend": 10,
        "compare": 10,
        "knowledge": 10,
        "sales": 10,
    },
    "grpo_final": {
        "recommend": 10,
        "compare": 10,
        "knowledge": 10,
        "sales": 10,
    },
}


def normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.strip().split()).casefold()


def normalized_sha256(value: str) -> str:
    return hashlib.sha256(normalize(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"required regular JSONL file is missing: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            raise ValueError(f"{path}:{line_number}: blank JSONL line")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{path}:{line_number}: invalid JSON"
            ) from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: row must be an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: empty JSONL file")
    return rows


def _row_id_query(row: dict[str, Any], *, label: str) -> tuple[str, str]:
    record_id = row.get("id")
    query = row.get("query")
    if not isinstance(query, str) or not query.strip():
        for message in row.get("messages", []):
            if (
                isinstance(message, dict)
                and message.get("role") == "user"
                and isinstance(message.get("content"), str)
                and message["content"].strip()
            ):
                query = message["content"]
                break
    if not isinstance(record_id, str) or not normalize(record_id):
        raise ValueError(f"{label}: row has invalid ID")
    if not isinstance(query, str) or not normalize(query):
        raise ValueError(f"{label}: row {record_id} has no user query")
    return record_id, query


@dataclass(frozen=True)
class Fingerprint:
    label: str
    rows: int
    id_hashes: frozenset[str]
    query_hashes: frozenset[str]


def fingerprint_rows(
    rows: Iterable[dict[str, Any]],
    *,
    label: str,
) -> Fingerprint:
    id_hashes: set[str] = set()
    query_hashes: set[str] = set()
    count = 0
    for row in rows:
        count += 1
        record_id, query = _row_id_query(row, label=label)
        id_hash = normalized_sha256(record_id)
        query_hash = normalized_sha256(query)
        if id_hash in id_hashes:
            raise ValueError(f"{label}: duplicate normalized ID")
        if query_hash in query_hashes:
            raise ValueError(f"{label}: duplicate normalized query")
        id_hashes.add(id_hash)
        query_hashes.add(query_hash)
    if count == 0:
        raise ValueError(f"{label}: no rows")
    return Fingerprint(
        label=label,
        rows=count,
        id_hashes=frozenset(id_hashes),
        query_hashes=frozenset(query_hashes),
    )


def assert_disjoint(left: Fingerprint, right: Fingerprint) -> dict[str, int]:
    id_overlap = left.id_hashes & right.id_hashes
    query_overlap = left.query_hashes & right.query_hashes
    if id_overlap:
        raise ValueError(
            f"{left.label}/{right.label}: normalized ID SHA overlap"
        )
    if query_overlap:
        raise ValueError(
            f"{left.label}/{right.label}: normalized query SHA overlap"
        )
    return {
        "normalized_id_sha256_overlap": 0,
        "normalized_query_sha256_overlap": 0,
    }


def _collect_query_strings(value: Any, result: set[str]) -> None:
    if isinstance(value, dict):
        query = value.get("query")
        if isinstance(query, str) and normalize(query):
            result.add(query)
        if (
            value.get("role") == "user"
            and isinstance(value.get("content"), str)
            and normalize(value["content"])
        ):
            result.add(value["content"])
        for child in value.values():
            _collect_query_strings(child, result)
    elif isinstance(value, list):
        for child in value:
            _collect_query_strings(child, result)


def _historical_query_corpus() -> tuple[set[str], list[str]]:
    excluded = {
        freeze.OUTPUT_PATH.resolve(),
        freeze.MANIFEST_PATH.resolve(),
        DEFAULT_REPORT_PATH.resolve(),
    }
    queries: set[str] = set()
    scanned: list[str] = []
    for path in sorted(
        (ROOT / "data" / "model_training").rglob("*.json*")
    ):
        if (
            path.resolve() in excluded
            or not path.is_file()
            or path.name.endswith(".sha256")
        ):
            continue
        scanned.append(str(path.relative_to(ROOT)))
        if path.suffix == ".jsonl":
            values: Any = load_jsonl(path)
        else:
            try:
                values = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid historical JSON: {path}") from error
        _collect_query_strings(values, queries)
    return {normalized_sha256(query) for query in queries}, scanned


def _validate_manifest(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest = json.loads(
        freeze.MANIFEST_PATH.read_text(encoding="utf-8")
    )
    if manifest.get("status") != "frozen_without_model_inference":
        raise ValueError("GRPO final manifest status is not frozen")
    dataset = manifest.get("dataset", {})
    if dataset.get("sha256") != sha256_file(freeze.OUTPUT_PATH):
        raise ValueError("GRPO final dataset SHA differs from manifest")
    if dataset.get("records") != len(rows):
        raise ValueError("GRPO final record count differs from manifest")
    for artifact_path, sha_path, expected in (
        (
            freeze.OUTPUT_PATH,
            freeze.OUTPUT_SHA_PATH,
            dataset.get("sha256"),
        ),
        (
            freeze.MANIFEST_PATH,
            freeze.MANIFEST_SHA_PATH,
            sha256_file(freeze.MANIFEST_PATH),
        ),
    ):
        recorded = sha_path.read_text(encoding="ascii").strip().split()
        if (
            not recorded
            or recorded[0] != expected
            or recorded[-1] != artifact_path.name
        ):
            raise ValueError(f"companion SHA record drift: {sha_path}")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, list) or len(provenance) != len(rows):
        raise ValueError("GRPO final provenance is incomplete")
    rows_by_id = {row["id"]: row for row in rows}
    for item in provenance:
        row = rows_by_id.get(item.get("id"))
        if row is None:
            raise ValueError("GRPO final provenance contains unknown ID")
        candidate_index = item.get("candidate_index")
        if (
            isinstance(candidate_index, bool)
            or not isinstance(candidate_index, int)
            or candidate_index <= (
                freeze.HISTORICAL_MAX_CANDIDATE_INDEX
            )
        ):
            raise ValueError("GRPO final candidate is not above index 650")
        if item.get("normalized_id_sha256") != normalized_sha256(
            row["id"]
        ):
            raise ValueError("GRPO final normalized ID SHA drift")
        if item.get("normalized_query_sha256") != normalized_sha256(
            row["query"]
        ):
            raise ValueError("GRPO final normalized query SHA drift")
        if item.get("selection_sha256") != freeze.selection_digest(
            intent=row["intent"],
            candidate_index=candidate_index,
            query=row["query"],
        ):
            raise ValueError("GRPO final selection SHA drift")
    for source in manifest["source_manifests"].values():
        source_path = ROOT / source["path"]
        if sha256_file(source_path) != source["sha256"]:
            raise ValueError("historical source manifest SHA drift")
        source_payload = json.loads(
            source_path.read_text(encoding="utf-8")
        )
        if source_payload.get("max_candidates") != 650:
            raise ValueError("historical source boundary is not 650")
    return {
        "path": str(freeze.MANIFEST_PATH.relative_to(ROOT)),
        "sha256": sha256_file(freeze.MANIFEST_PATH),
        "dataset_sha256": dataset["sha256"],
        "all_candidate_indices_above_650": True,
    }


def _validate_reward_train_dev(
    eval_fingerprints: dict[str, Fingerprint],
) -> dict[str, Any]:
    train_rows = load_jsonl(reward_split.TRAIN_PATH)
    dev_rows = load_jsonl(reward_split.DEV_PATH)
    if len(train_rows) != 16 or len(dev_rows) != 4:
        raise ValueError("reward train/dev counts must be 16/4")
    for label, rows, expected_per_intent in (
        (
            "reward_train_16",
            train_rows,
            {intent: 4 for intent in reward_split.INTENTS},
        ),
        (
            "reward_dev_4",
            dev_rows,
            {intent: 1 for intent in reward_split.INTENTS},
        ),
    ):
        if dict(Counter(row["intent"] for row in rows)) != (
            expected_per_intent
        ):
            raise ValueError(f"{label}: per-intent counts drift")
        evaluator.validate_case_records(
            rows,
            vehicle_catalog=set(evaluator.load_vehicle_catalog()),
        )
    train_fingerprint = fingerprint_rows(
        train_rows,
        label="reward_train_16",
    )
    dev_fingerprint = fingerprint_rows(
        dev_rows,
        label="reward_dev_4",
    )
    train_dev_isolation = assert_disjoint(
        train_fingerprint,
        dev_fingerprint,
    )
    reward_visible = eval_fingerprints["reward_visible"]
    if (
        train_fingerprint.id_hashes | dev_fingerprint.id_hashes
        != reward_visible.id_hashes
        or train_fingerprint.query_hashes | dev_fingerprint.query_hashes
        != reward_visible.query_hashes
    ):
        raise ValueError(
            "reward train/dev union does not reconstruct reward-visible"
        )

    against_read_only: list[dict[str, Any]] = []
    for split_label, fingerprint in (
        ("reward_train_16", train_fingerprint),
        ("reward_dev_4", dev_fingerprint),
    ):
        for eval_label in ("sft_held_out", "grpo_final"):
            result = assert_disjoint(
                fingerprint,
                eval_fingerprints[eval_label],
            )
            against_read_only.append(
                {
                    "left": split_label,
                    "right": eval_label,
                    **result,
                }
            )

    manifest = json.loads(
        reward_split.MANIFEST_PATH.read_text(encoding="utf-8")
    )
    if manifest.get("status") != (
        "frozen_without_training_or_inference"
    ):
        raise ValueError("reward train/dev manifest status drift")
    if manifest.get("gradient_policy") != {
        "train_16": "eligible_for_gradient_updates",
        "dev_4": "evaluation_and_early_stopping_only_no_gradient",
    }:
        raise ValueError("reward train/dev gradient policy drift")
    if manifest["source"]["sha256"] != sha256_file(
        reward_split.SOURCE_PATH
    ):
        raise ValueError("reward train/dev source SHA drift")
    for label, path, sha_path, rows in (
        (
            "train",
            reward_split.TRAIN_PATH,
            reward_split.TRAIN_SHA_PATH,
            train_rows,
        ),
        (
            "dev",
            reward_split.DEV_PATH,
            reward_split.DEV_SHA_PATH,
            dev_rows,
        ),
    ):
        artifact = manifest["artifacts"][label]
        observed_sha = sha256_file(path)
        if (
            artifact["sha256"] != observed_sha
            or artifact["records"] != len(rows)
        ):
            raise ValueError(f"reward {label} artifact drift")
        recorded = sha_path.read_text(encoding="ascii").strip().split()
        if (
            not recorded
            or recorded[0] != observed_sha
            or recorded[-1] != path.name
        ):
            raise ValueError(f"reward {label} companion SHA drift")
    manifest_sha = sha256_file(reward_split.MANIFEST_PATH)
    recorded_manifest = (
        reward_split.MANIFEST_SHA_PATH.read_text(encoding="ascii")
        .strip()
        .split()
    )
    if (
        not recorded_manifest
        or recorded_manifest[0] != manifest_sha
        or recorded_manifest[-1] != reward_split.MANIFEST_PATH.name
    ):
        raise ValueError("reward train/dev manifest companion SHA drift")
    return {
        "counts": {
            "train": len(train_rows),
            "dev": len(dev_rows),
        },
        "sha256": {
            "train": sha256_file(reward_split.TRAIN_PATH),
            "dev": sha256_file(reward_split.DEV_PATH),
            "manifest": manifest_sha,
        },
        "train_dev_isolation": train_dev_isolation,
        "union_exactly_matches_reward_visible": True,
        "against_read_only_eval": against_read_only,
        "dev_ids": [row["id"] for row in dev_rows],
        "gradient_policy": manifest["gradient_policy"],
    }


def validate_isolation() -> dict[str, Any]:
    resolved_eval_paths = [path.resolve() for path in EVAL_PATHS.values()]
    if len(set(resolved_eval_paths)) != len(resolved_eval_paths):
        raise ValueError("evaluation sets are not physically separate files")

    rows_by_split = {
        label: load_jsonl(path)
        for label, path in EVAL_PATHS.items()
    }
    for label, rows in rows_by_split.items():
        if len(rows) != EXPECTED_COUNTS[label]:
            raise ValueError(
                f"{label}: expected {EXPECTED_COUNTS[label]} rows, "
                f"got {len(rows)}"
            )
        intent_counts = Counter(row.get("intent") for row in rows)
        if dict(intent_counts) != EXPECTED_PER_INTENT[label]:
            raise ValueError(f"{label}: per-intent counts drift")
        evaluator.validate_case_records(
            rows,
            vehicle_catalog=set(evaluator.load_vehicle_catalog()),
        )

    fingerprints = {
        label: fingerprint_rows(rows, label=label)
        for label, rows in rows_by_split.items()
    }
    pairwise: list[dict[str, Any]] = []
    for left_label, right_label in itertools.combinations(EVAL_PATHS, 2):
        result = assert_disjoint(
            fingerprints[left_label],
            fingerprints[right_label],
        )
        pairwise.append(
            {
                "left": left_label,
                "right": right_label,
                **result,
            }
        )

    sft_rows = [
        row
        for path in SFT_SOURCE_PATHS
        for row in load_jsonl(path)
    ]
    if len(sft_rows) != 2500:
        raise ValueError(
            f"complete SFT source surface must contain 2500 rows, got "
            f"{len(sft_rows)}"
        )
    sft_fingerprint = fingerprint_rows(
        sft_rows,
        label="complete_sft_source_surface",
    )
    grpo_vs_sft = assert_disjoint(
        fingerprints["grpo_final"],
        sft_fingerprint,
    )

    historical_hashes, scanned_files = _historical_query_corpus()
    historical_overlap = (
        fingerprints["grpo_final"].query_hashes & historical_hashes
    )
    if historical_overlap:
        raise ValueError(
            "grpo_final/historical corpus: normalized query SHA overlap"
        )

    manifest_summary = _validate_manifest(
        rows_by_split["grpo_final"]
    )
    reward_train_dev = _validate_reward_train_dev(fingerprints)
    return {
        "status": "passed",
        "mode": "static_only_no_model_access",
        "split_counts": {
            label: len(rows)
            for label, rows in rows_by_split.items()
        },
        "split_sha256": {
            label: sha256_file(path)
            for label, path in EVAL_PATHS.items()
        },
        "pairwise_eval_isolation": pairwise,
        "complete_sft_source_surface": {
            "rows": len(sft_rows),
            "paths": [
                str(path.relative_to(ROOT))
                for path in SFT_SOURCE_PATHS
            ],
        },
        "grpo_final_vs_sft_source": grpo_vs_sft,
        "grpo_final_vs_historical_query_corpus": {
            "normalized_query_sha256_overlap": 0,
            "historical_unique_query_hashes": len(historical_hashes),
            "files_scanned": len(scanned_files),
        },
        "manifest": manifest_summary,
        "reward_train_dev": reward_train_dev,
        "training_boundary": {
            "gradient_updates_only": (
                "data/model_training/grpo/reward_train_16.jsonl"
            ),
            "early_stopping_only_no_gradient": (
                "data/model_training/grpo/reward_dev_4.jsonl"
            ),
            "never_train": [
                "data/model_training/eval/held_out.jsonl",
                "data/model_training/eval/grpo_final_held_out.jsonl",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate GRPO eval isolation without model access.",
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()
    report = validate_isolation()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
