"""Audit and exclude SFT rows whose assistant labels cross a token cutoff."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from data_synth import freeze_sft_dataset as freezer
from training.sft import train_qlora_sft as sft


ROOT = Path(__file__).resolve().parents[2]
MODEL_TRAINING_DIR = ROOT / "data" / "model_training"
DEFAULT_TRAIN_PATH = MODEL_TRAINING_DIR / "sft_train.jsonl"
DEFAULT_VAL_PATH = MODEL_TRAINING_DIR / "sft_val.jsonl"
DEFAULT_EXCLUDED_PATH = MODEL_TRAINING_DIR / "truncated_excluded.jsonl"
DEFAULT_REPORT_PATH = MODEL_TRAINING_DIR / "truncation_supervision_audit.json"
DEFAULT_MANIFEST_PATH = (
    MODEL_TRAINING_DIR / "sft_freeze" / "split_manifest.json"
)
DEFAULT_DATASET_CARD_PATH = MODEL_TRAINING_DIR / "sft_dataset_card.md"
DEFAULT_TOKEN_REPORT_PATH = (
    MODEL_TRAINING_DIR / "sft_token_length_report.json"
)
DEFAULT_PROFILE_REPORT_PATH = (
    MODEL_TRAINING_DIR / "sft_step_profile_report.json"
)

SUPERVISION_COMPLETE = "supervision_complete"
SUPERVISION_PARTIALLY_TRUNCATED = "supervision_partially_truncated"
SUPERVISION_FULLY_TRUNCATED = "supervision_fully_truncated"
EXCLUSION_FIELD = "_truncation_exclusion"


def _validate_row(row: dict[str, Any]) -> None:
    record_id = row.get("id")
    intent = row.get("intent")
    text = row.get("qwen_chatml")
    spans = row.get("assistant_char_spans")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("SFT row id must be a non-empty string")
    if not isinstance(intent, str) or not intent:
        raise ValueError(f"{record_id}: intent must be a non-empty string")
    if not isinstance(text, str) or not text:
        raise ValueError(f"{record_id}: qwen_chatml must be non-empty")
    if (
        not isinstance(spans, list)
        or not spans
        or any(
            not isinstance(span, list)
            or len(span) != 2
            or not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in span
            )
            or span[0] < 0
            or span[0] >= span[1]
            or span[1] > len(text)
            for span in spans
        )
    ):
        raise ValueError(f"{record_id}: assistant_char_spans are invalid")


def _token_offsets(tokenizer: Any, text: str) -> tuple[list[Any], list[list[int]]]:
    encoded = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=False,
    )
    input_ids = encoded.get("input_ids")
    offsets = encoded.get("offset_mapping")
    if not isinstance(input_ids, list) or not isinstance(offsets, list):
        raise ValueError("tokenizer must return list input_ids and offset_mapping")
    if len(input_ids) != len(offsets):
        raise ValueError("token IDs and offsets must have equal lengths")
    normalized_offsets: list[list[int]] = []
    for offset in offsets:
        if (
            not isinstance(offset, (list, tuple))
            or len(offset) != 2
            or not all(isinstance(value, int) for value in offset)
        ):
            raise ValueError("tokenizer returned an invalid offset mapping")
        normalized_offsets.append([offset[0], offset[1]])
    return input_ids, normalized_offsets


def _span_token_indices(
    offsets: list[list[int]],
    span_start: int,
    span_end: int,
) -> list[int]:
    return [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_start != token_end
        and token_start < span_end
        and token_end > span_start
    ]


def audit_row_supervision(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    max_seq_len: int,
) -> dict[str, Any]:
    """Map assistant character spans to tokens and classify prefix truncation."""
    if (
        not isinstance(max_seq_len, int)
        or isinstance(max_seq_len, bool)
        or max_seq_len < 1
    ):
        raise ValueError("max_seq_len must be a positive integer")
    _validate_row(row)
    input_ids, offsets = _token_offsets(tokenizer, row["qwen_chatml"])

    retained_total = 0
    truncated_total = 0
    span_results: list[dict[str, Any]] = []
    for span_index, (span_start, span_end) in enumerate(
        row["assistant_char_spans"]
    ):
        token_indices = _span_token_indices(offsets, span_start, span_end)
        retained = sum(index < max_seq_len for index in token_indices)
        truncated = len(token_indices) - retained
        if not token_indices:
            raise ValueError(
                f"{row['id']}: assistant span {span_index} maps to no tokens"
            )
        if truncated == 0:
            span_status = "complete"
        elif retained == 0:
            span_status = "fully_truncated"
        else:
            span_status = "partially_truncated"
        retained_total += retained
        truncated_total += truncated
        span_results.append(
            {
                "span_index": span_index,
                "char_start": span_start,
                "char_end": span_end,
                "supervised_tokens": len(token_indices),
                "retained_tokens": retained,
                "truncated_tokens": truncated,
                "status": span_status,
            }
        )

    if truncated_total == 0:
        status = SUPERVISION_COMPLETE
    elif retained_total == 0:
        status = SUPERVISION_FULLY_TRUNCATED
    else:
        status = SUPERVISION_PARTIALLY_TRUNCATED
    return {
        "id": row["id"],
        "intent": row["intent"],
        "raw_token_length": len(input_ids),
        "max_seq_len": max_seq_len,
        "status": status,
        "supervised_tokens_total": retained_total + truncated_total,
        "supervised_tokens_retained": retained_total,
        "supervised_tokens_truncated": truncated_total,
        "assistant_spans": span_results,
    }


def partition_rows_by_supervision(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    max_seq_len: int,
    split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return active rows, toxic rows, and audits for over-length rows."""
    if split not in {"train", "validation"}:
        raise ValueError("split must be train or validation")
    active: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in rows:
        _validate_row(row)
        if row["id"] in seen_ids:
            raise ValueError(f"duplicate {split} row ID: {row['id']}")
        seen_ids.add(row["id"])
        result = audit_row_supervision(
            tokenizer,
            row,
            max_seq_len=max_seq_len,
        )
        if result["raw_token_length"] <= max_seq_len:
            active.append(row)
            continue
        result["split"] = split
        results.append(result)
        if result["status"] == SUPERVISION_COMPLETE:
            active.append(row)
        else:
            excluded.append(row)
    return active, excluded, results


def _load_jsonl(path: Path, *, optional: bool = False) -> list[dict[str, Any]]:
    if optional and not path.is_file():
        return []
    return freezer.load_jsonl(path)


def _atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            for row in rows:
                output.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _restore_original_splits(
    *,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    excluded_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    restored = {"train": list(train_rows), "validation": list(val_rows)}
    for archived in excluded_rows:
        metadata = archived.get(EXCLUSION_FIELD)
        if not isinstance(metadata, dict):
            raise ValueError("excluded row is missing truncation split metadata")
        split = metadata.get("split")
        if split not in restored:
            raise ValueError("excluded row has an invalid original split")
        row = dict(archived)
        row.pop(EXCLUSION_FIELD)
        restored[split].append(row)

    for split, rows in restored.items():
        ids = [row.get("id") for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"restored {split} split contains duplicate IDs")
        rows.sort(key=lambda row: row["id"])
    return restored["train"], restored["validation"]


def _split_summary(
    *,
    original_rows: list[dict[str, Any]],
    active_rows: list[dict[str, Any]],
    excluded_rows: list[dict[str, Any]],
    audits: list[dict[str, Any]],
) -> dict[str, Any]:
    status_counts = Counter(audit["status"] for audit in audits)
    original_by_intent = Counter(row["intent"] for row in original_rows)
    active_by_intent = Counter(row["intent"] for row in active_rows)
    excluded_by_intent = Counter(row["intent"] for row in excluded_rows)
    audited_by_intent: dict[str, Counter[str]] = defaultdict(Counter)
    for item in audits:
        audited_by_intent[item["intent"]]["truncated_rows"] += 1
        audited_by_intent[item["intent"]][item["status"]] += 1
    intents = sorted(original_by_intent)
    return {
        "original_rows": len(original_rows),
        "rows_exceeding_max_seq_len": len(audits),
        "supervision_complete": status_counts[SUPERVISION_COMPLETE],
        "supervision_partially_truncated": status_counts[
            SUPERVISION_PARTIALLY_TRUNCATED
        ],
        "supervision_fully_truncated": status_counts[
            SUPERVISION_FULLY_TRUNCATED
        ],
        "excluded_rows": len(excluded_rows),
        "active_rows": len(active_rows),
        "excluded_ids": [row["id"] for row in excluded_rows],
        "per_intent": {
            intent: {
                "original_rows": original_by_intent[intent],
                "rows_exceeding_max_seq_len": audited_by_intent[intent][
                    "truncated_rows"
                ],
                "supervision_complete": audited_by_intent[intent][
                    SUPERVISION_COMPLETE
                ],
                "supervision_partially_truncated": audited_by_intent[intent][
                    SUPERVISION_PARTIALLY_TRUNCATED
                ],
                "supervision_fully_truncated": audited_by_intent[intent][
                    SUPERVISION_FULLY_TRUNCATED
                ],
                "excluded_rows": excluded_by_intent[intent],
                "active_rows": active_by_intent[intent],
            }
            for intent in intents
        },
    }


def _nearest_rank(values: list[int], percentile: int) -> int:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile / 100 * len(ordered)) - 1)
    return ordered[index]


def _length_distribution(tokenizer: Any, rows: list[dict[str, Any]]) -> dict[str, int]:
    lengths = [
        len(
            tokenizer(
                row["qwen_chatml"],
                add_special_tokens=False,
            )["input_ids"]
        )
        for row in rows
    ]
    if not lengths:
        raise ValueError("cannot summarize an empty active split")
    return {
        "count": len(lengths),
        "p50": _nearest_rank(lengths, 50),
        "p95": _nearest_rank(lengths, 95),
        "p99": _nearest_rank(lengths, 99),
        "max": max(lengths),
    }


def _updated_token_report(
    *,
    path: Path,
    train_distribution: dict[str, int],
    validation_distribution: dict[str, int],
    excluded_train: int,
    excluded_validation: int,
    max_seq_len: int,
) -> dict[str, Any]:
    previous = json.loads(path.read_text(encoding="utf-8"))
    pre_audit_distribution = previous.get(
        "pre_audit_token_length_distribution",
        previous["token_length_distribution"],
    )
    return {
        "status": "completed_after_supervision_filter",
        "dataset_rows": train_distribution["count"],
        "token_length_distribution": train_distribution,
        "validation_rows": validation_distribution["count"],
        "validation_token_length_distribution": validation_distribution,
        "selected_max_seq_len": max_seq_len,
        "rows_exceeding_selected_max_seq_len": 0,
        "truncation_rate_at_selected_max_seq_len": 0.0,
        "pre_audit_token_length_distribution": pre_audit_distribution,
        "supervision_filter": {
            "excluded_train_rows": excluded_train,
            "excluded_validation_rows": excluded_validation,
            "report": str(DEFAULT_REPORT_PATH.relative_to(ROOT)),
        },
    }


def _update_profile_estimate(
    *,
    path: Path,
    train_examples: int,
) -> dict[str, Any]:
    profile = json.loads(path.read_text(encoding="utf-8"))
    old_estimate = profile.get(
        "pre_audit_wall_clock_estimate",
        profile["wall_clock_estimate"],
    )
    plan = profile["plan"]
    mean_step = profile["steady_p95_profile"]["mean_micro_step_sec"]
    total_micro_steps = (
        math.ceil(train_examples / plan["micro_batch_size"])
        * plan["epochs"]
    )
    optimizer_steps = math.ceil(
        total_micro_steps / plan["gradient_accumulation_steps"]
    )
    compute_seconds = round(total_micro_steps * mean_step, 2)
    overhead_ratio = old_estimate["overhead_ratio"]
    wall_seconds = round(compute_seconds * (1 + overhead_ratio), 2)
    profile["pre_audit_wall_clock_estimate"] = old_estimate
    profile["wall_clock_estimate"] = {
        "train_examples": train_examples,
        "epochs": plan["epochs"],
        "micro_batch_size": plan["micro_batch_size"],
        "gradient_accumulation_steps": plan[
            "gradient_accumulation_steps"
        ],
        "total_micro_steps": total_micro_steps,
        "optimizer_steps": optimizer_steps,
        "measured_micro_step_sec": mean_step,
        "estimated_compute_sec": compute_seconds,
        "overhead_ratio": overhead_ratio,
        "estimated_wall_clock_sec": wall_seconds,
        "estimated_wall_clock_hours": round(wall_seconds / 3600, 2),
    }
    profile["estimate_note"] = (
        "Recomputed for the active post-audit train split while conservatively "
        "reusing the measured 5632-token micro-step latency."
    )
    return profile


def _updated_manifest(
    *,
    previous: dict[str, Any],
    original_train: list[dict[str, Any]],
    active_train: list[dict[str, Any]],
    excluded_train: list[dict[str, Any]],
    original_val: list[dict[str, Any]],
    active_val: list[dict[str, Any]],
    excluded_val: list[dict[str, Any]],
    report_path: Path,
    max_seq_len: int,
) -> dict[str, Any]:
    intents = sorted(
        {
            row["intent"]
            for row in [*original_train, *original_val]
        }
    )
    counters = {
        "original_train": Counter(row["intent"] for row in original_train),
        "active_train": Counter(row["intent"] for row in active_train),
        "excluded_train": Counter(row["intent"] for row in excluded_train),
        "original_val": Counter(row["intent"] for row in original_val),
        "active_val": Counter(row["intent"] for row in active_val),
        "excluded_val": Counter(row["intent"] for row in excluded_val),
    }
    per_intent = {
        intent: {
            "source": counters["original_train"][intent]
            + counters["original_val"][intent],
            "pre_audit_train": counters["original_train"][intent],
            "train": counters["active_train"][intent],
            "excluded_train": counters["excluded_train"][intent],
            "pre_audit_eval": counters["original_val"][intent],
            "eval": counters["active_val"][intent],
            "excluded_eval": counters["excluded_val"][intent],
        }
        for intent in intents
    }
    counts = {
        "source_total": len(original_train) + len(original_val),
        "pre_audit_train_total": len(original_train),
        "train_total": len(active_train),
        "excluded_train_total": len(excluded_train),
        "pre_audit_eval_total": len(original_val),
        "eval_total": len(active_val),
        "excluded_eval_total": len(excluded_val),
        "per_intent": per_intent,
    }
    return {
        **previous,
        "format_version": 2,
        "counts": counts,
        "train_ids": [row["id"] for row in active_train],
        "eval_ids": [row["id"] for row in active_val],
        "excluded_train_ids": [row["id"] for row in excluded_train],
        "excluded_eval_ids": [row["id"] for row in excluded_val],
        "truncation_supervision_audit": {
            "status": "passed_after_exclusion",
            "max_seq_len": max_seq_len,
            "report": str(report_path.relative_to(ROOT)),
            "excluded_file": str(DEFAULT_EXCLUDED_PATH.relative_to(ROOT)),
        },
    }


def _write_dataset_card(
    *,
    path: Path,
    manifest: dict[str, Any],
    report: dict[str, Any],
) -> None:
    reservation = json.loads(
        (DEFAULT_MANIFEST_PATH.parent / "reward_reservation_manifest.json")
        .read_text(encoding="utf-8")
    )
    card = freezer._dataset_card(
        counts=manifest["counts"]["per_intent"],
        metrics=freezer._summary_metrics(),
        split_seed=manifest["split_seed"],
        eval_fraction=manifest["eval_fraction"],
        reservation_entries=reservation["entries"],
        teacher_endpoint_alias=freezer.TEACHER_ENDPOINT_ALIAS,
        truncation_audit=report,
    )
    path.write_text(card, encoding="utf-8")


def run_audit(
    *,
    config_path: Path = sft.DEFAULT_CONFIG_PATH,
    train_path: Path = DEFAULT_TRAIN_PATH,
    val_path: Path = DEFAULT_VAL_PATH,
    excluded_path: Path = DEFAULT_EXCLUDED_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    dataset_card_path: Path = DEFAULT_DATASET_CARD_PATH,
    token_report_path: Path = DEFAULT_TOKEN_REPORT_PATH,
    profile_report_path: Path = DEFAULT_PROFILE_REPORT_PATH,
) -> dict[str, Any]:
    """Audit both active splits, exclude toxic rows, and update freeze metadata."""
    config = sft.load_training_config(config_path)
    max_seq_len = int(config["data"]["max_seq_len"])
    model_path = sft._model_path(config, ROOT)
    transformers = importlib.import_module("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
    )

    active_train_before = _load_jsonl(train_path)
    active_val_before = _load_jsonl(val_path)
    archived_before = _load_jsonl(excluded_path, optional=True)
    original_train, original_val = _restore_original_splits(
        train_rows=active_train_before,
        val_rows=active_val_before,
        excluded_rows=archived_before,
    )
    active_train, excluded_train, train_audits = partition_rows_by_supervision(
        tokenizer,
        original_train,
        max_seq_len=max_seq_len,
        split="train",
    )
    active_val, excluded_val, val_audits = partition_rows_by_supervision(
        tokenizer,
        original_val,
        max_seq_len=max_seq_len,
        split="validation",
    )

    train_summary = _split_summary(
        original_rows=original_train,
        active_rows=active_train,
        excluded_rows=excluded_train,
        audits=train_audits,
    )
    val_summary = _split_summary(
        original_rows=original_val,
        active_rows=active_val,
        excluded_rows=excluded_val,
        audits=val_audits,
    )
    train_distribution = _length_distribution(tokenizer, active_train)
    validation_distribution = _length_distribution(tokenizer, active_val)
    report = {
        "status": "passed_after_exclusion",
        "purpose": (
            "prevent prefix truncation from teaching incomplete assistant "
            "tool calls or final answers"
        ),
        "tokenization_contract": {
            "model_path": config["model"]["base_model_path"],
            "local_files_only": True,
            "max_seq_len": max_seq_len,
            "truncation_side": getattr(tokenizer, "truncation_side", "right"),
            "label_source": "assistant_char_spans",
            "matches_training_function": "train_qlora_sft._masked_batch",
        },
        "policy": (
            "exclude every row where any supervised assistant token would be "
            "removed by max_seq_len prefix truncation"
        ),
        "splits": {
            "train": train_summary,
            "validation": val_summary,
        },
        "active_token_length_distribution": {
            "train": train_distribution,
            "validation": validation_distribution,
        },
        "excluded_total": len(excluded_train) + len(excluded_val),
        "excluded_path": str(excluded_path.relative_to(ROOT)),
        "training_invariant": {
            "active_train_rows_with_truncated_supervision": 0,
            "active_validation_rows_with_truncated_supervision": 0,
            "status": "passed",
        },
        "cases": sorted(
            [*train_audits, *val_audits],
            key=lambda item: (item["split"], item["id"]),
        ),
    }

    previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = _updated_manifest(
        previous=previous_manifest,
        original_train=original_train,
        active_train=active_train,
        excluded_train=excluded_train,
        original_val=original_val,
        active_val=active_val,
        excluded_val=excluded_val,
        report_path=report_path,
        max_seq_len=max_seq_len,
    )
    archived_rows = [
        {
            **row,
            EXCLUSION_FIELD: {
                "split": split,
                "reason": "supervised_assistant_tokens_cross_max_seq_len",
                "max_seq_len": max_seq_len,
            },
        }
        for split, rows in (
            ("train", excluded_train),
            ("validation", excluded_val),
        )
        for row in rows
    ]
    archived_rows.sort(
        key=lambda row: (row[EXCLUSION_FIELD]["split"], row["id"])
    )

    _atomic_write_jsonl(train_path, active_train)
    _atomic_write_jsonl(val_path, active_val)
    _atomic_write_jsonl(excluded_path, archived_rows)
    _atomic_write_json(report_path, report)
    _atomic_write_json(manifest_path, manifest)
    _atomic_write_json(
        token_report_path,
        _updated_token_report(
            path=token_report_path,
            train_distribution=train_distribution,
            validation_distribution=validation_distribution,
            excluded_train=len(excluded_train),
            excluded_validation=len(excluded_val),
            max_seq_len=max_seq_len,
        ),
    )
    _atomic_write_json(
        profile_report_path,
        _update_profile_estimate(
            path=profile_report_path,
            train_examples=len(active_train),
        ),
    )
    _write_dataset_card(
        path=dataset_card_path,
        manifest=manifest,
        report=report,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exclude frozen SFT rows with truncated assistant labels.",
    )
    parser.add_argument("--config", type=Path, default=sft.DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    report = run_audit(config_path=args.config)
    train = report["splits"]["train"]
    validation = report["splits"]["validation"]
    print(
        json.dumps(
            {
                "status": report["status"],
                "train": {
                    "original_rows": train["original_rows"],
                    "rows_exceeding_max_seq_len": train[
                        "rows_exceeding_max_seq_len"
                    ],
                    "excluded_rows": train["excluded_rows"],
                    "active_rows": train["active_rows"],
                    "per_intent": train["per_intent"],
                },
                "validation": {
                    "original_rows": validation["original_rows"],
                    "rows_exceeding_max_seq_len": validation[
                        "rows_exceeding_max_seq_len"
                    ],
                    "excluded_rows": validation["excluded_rows"],
                    "active_rows": validation["active_rows"],
                    "per_intent": validation["per_intent"],
                },
                "report": str(DEFAULT_REPORT_PATH),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
