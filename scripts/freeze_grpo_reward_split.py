#!/usr/bin/env python3
"""Freeze a deterministic 16-train/4-dev split of reward-visible cases."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from scripts import evaluate_model_outputs as evaluator


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = (
    ROOT / "data" / "model_training" / "eval" / "reward_visible.jsonl"
)
OUTPUT_DIR = ROOT / "data" / "model_training" / "grpo"
TRAIN_PATH = OUTPUT_DIR / "reward_train_16.jsonl"
DEV_PATH = OUTPUT_DIR / "reward_dev_4.jsonl"
MANIFEST_PATH = OUTPUT_DIR / "reward_train_dev_manifest.json"
TRAIN_SHA_PATH = TRAIN_PATH.with_suffix(".sha256")
DEV_SHA_PATH = DEV_PATH.with_suffix(".sha256")
MANIFEST_SHA_PATH = MANIFEST_PATH.with_suffix(".sha256")
SOURCE_SHA256 = (
    "98117983647ab5f3618f96831612bca7984af09db85ef434485f42901b391c5e"
)
SELECTION_SEED = "grpo-reward-split-v1"
INTENTS = ("recommend", "compare", "knowledge", "sales")


def normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.strip().split()).casefold()


def normalized_sha256(value: str) -> str:
    return hashlib.sha256(normalize(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            raise ValueError(f"{path}:{line_number}: blank line")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected object")
        rows.append(row)
    return rows


def selection_sha256(row: dict[str, Any]) -> str:
    payload = (
        f"{SELECTION_SEED}:{row['intent']}:"
        f"{normalize(row['id'])}:{normalize(row['query'])}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_split() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    if sha256_file(SOURCE_PATH) != SOURCE_SHA256:
        raise ValueError("frozen reward-visible SHA drift")
    source = load_jsonl(SOURCE_PATH)
    if len(source) != 20:
        raise ValueError("reward-visible must contain exactly 20 rows")
    evaluator.validate_case_records(
        source,
        vehicle_catalog=set(evaluator.load_vehicle_catalog()),
    )
    if Counter(row["intent"] for row in source) != Counter(
        {intent: 5 for intent in INTENTS}
    ):
        raise ValueError("reward-visible must contain 5 rows per intent")

    dev_ids: set[str] = set()
    selection: list[dict[str, Any]] = []
    for intent in INTENTS:
        rows = [row for row in source if row["intent"] == intent]
        selected = min(
            rows,
            key=lambda row: (
                selection_sha256(row),
                normalize(row["id"]),
            ),
        )
        dev_ids.add(selected["id"])
        selection.append(
            {
                "intent": intent,
                "dev_id": selected["id"],
                "selection_sha256": selection_sha256(selected),
                "normalized_id_sha256": normalized_sha256(
                    selected["id"]
                ),
                "normalized_query_sha256": normalized_sha256(
                    selected["query"]
                ),
            }
        )
    train = [row for row in source if row["id"] not in dev_ids]
    dev = [row for row in source if row["id"] in dev_ids]
    if len(train) != 16 or len(dev) != 4:
        raise ValueError("reward split count drift")
    train_ids = {normalized_sha256(row["id"]) for row in train}
    dev_ids_hashed = {normalized_sha256(row["id"]) for row in dev}
    train_queries = {
        normalized_sha256(row["query"]) for row in train
    }
    dev_queries = {normalized_sha256(row["query"]) for row in dev}
    if train_ids & dev_ids_hashed:
        raise ValueError("reward train/dev normalized ID SHA overlap")
    if train_queries & dev_queries:
        raise ValueError("reward train/dev normalized query SHA overlap")
    manifest = {
        "format_version": 1,
        "status": "frozen_without_training_or_inference",
        "source": {
            "path": str(SOURCE_PATH.relative_to(ROOT)),
            "sha256": SOURCE_SHA256,
            "records": len(source),
        },
        "selection": {
            "seed": SELECTION_SEED,
            "algorithm": (
                "Within each intent, select the row with the lowest "
                "SHA256(seed:intent:normalized_id:normalized_query) as dev; "
                "all other rows are train."
            ),
            "records": selection,
        },
        "counts": {
            "train": len(train),
            "dev": len(dev),
            "train_per_intent": dict(
                sorted(Counter(row["intent"] for row in train).items())
            ),
            "dev_per_intent": dict(
                sorted(Counter(row["intent"] for row in dev).items())
            ),
        },
        "gradient_policy": {
            "train_16": "eligible_for_gradient_updates",
            "dev_4": "evaluation_and_early_stopping_only_no_gradient",
        },
        "union_contract": (
            "train and dev are disjoint by normalized ID/query SHA and their "
            "union exactly reconstructs frozen reward-visible 20"
        ),
    }
    return train, dev, manifest


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def main() -> None:
    outputs = (
        TRAIN_PATH,
        DEV_PATH,
        MANIFEST_PATH,
        TRAIN_SHA_PATH,
        DEV_SHA_PATH,
        MANIFEST_SHA_PATH,
    )
    for path in outputs:
        if path.exists():
            raise FileExistsError(f"reward split output exists: {path}")
    train, dev, manifest = build_split()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_jsonl(TRAIN_PATH, train)
    _write_jsonl(DEV_PATH, dev)
    manifest["artifacts"] = {
        "train": {
            "path": str(TRAIN_PATH.relative_to(ROOT)),
            "sha256": sha256_file(TRAIN_PATH),
            "records": len(train),
        },
        "dev": {
            "path": str(DEV_PATH.relative_to(ROOT)),
            "sha256": sha256_file(DEV_PATH),
            "records": len(dev),
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    TRAIN_SHA_PATH.write_text(
        f"{manifest['artifacts']['train']['sha256']}  {TRAIN_PATH.name}\n",
        encoding="ascii",
    )
    DEV_SHA_PATH.write_text(
        f"{manifest['artifacts']['dev']['sha256']}  {DEV_PATH.name}\n",
        encoding="ascii",
    )
    MANIFEST_SHA_PATH.write_text(
        f"{sha256_file(MANIFEST_PATH)}  {MANIFEST_PATH.name}\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "train": manifest["artifacts"]["train"],
                "dev": manifest["artifacts"]["dev"],
                "manifest": {
                    "path": str(MANIFEST_PATH.relative_to(ROOT)),
                    "sha256": sha256_file(MANIFEST_PATH),
                },
                "dev_ids": [row["id"] for row in dev],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
