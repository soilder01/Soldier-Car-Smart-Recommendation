#!/usr/bin/env python3
"""Derive formal_v2 authorization by changing abort calibration only."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GRPO_DIR = ROOT / "data" / "model_training" / "grpo"
V1_AUTH = GRPO_DIR / "grpo_formal_training_authorization.json"
V1_PREREG = GRPO_DIR / "grpo_formal_training_preregistered_report.json"
V2_AUTH = GRPO_DIR / "grpo_formal_v2_authorization.json"
V2_AUTH_SHA = V2_AUTH.with_suffix(".sha256")
V2_PREREG = GRPO_DIR / "grpo_formal_v2_preregistered_report.json"
V2_PREREG_SHA = V2_PREREG.with_suffix(".sha256")
V1_AUTH_SHA256 = (
    "5672d83bdd2a0c71f73b7513841b8d29aa1d2e499ecdab27e3cf9a5b02f402a9"
)
V1_PREREG_SHA256 = (
    "ee7080fa23279db6c5b660d8197c8789e345ceba6d8bf283d678cec750688e13"
)
SIGNAL_RAW_SHA256 = (
    "1a0dc606c0800c0531e4c4c375375aaebd3141b0542b176bd71c3366e72da07a"
)
BASELINE = {
    "source": "data/model_training/grpo/grpo_signal_probe_raw.jsonl",
    "source_sha256": SIGNAL_RAW_SHA256,
    "population": "train_16_only_128_completions",
    "intent_response_fail_rate": 0.6953125,
    "zero_variance_group_ratio": 0.25,
    "completion_length_p50": 73.5,
    "per_intent_fail_rate": {
        "compare": 0.8125,
        "knowledge": 0.90625,
        "recommend": 0.34375,
        "sales": 0.71875,
    },
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_v2() -> tuple[dict, dict]:
    if sha256_file(V1_AUTH) != V1_AUTH_SHA256:
        raise ValueError("formal_v1 authorization SHA drift")
    if sha256_file(V1_PREREG) != V1_PREREG_SHA256:
        raise ValueError("formal_v1 preregistration SHA drift")
    auth = copy.deepcopy(json.loads(V1_AUTH.read_text(encoding="utf-8")))
    auth["status"] = "authorized_once_before_formal_grpo_v2_training"
    auth["run_id"] = "grpo-formal-v2"
    auth["derived_from"] = {
        "formal_v1_authorization_sha256": V1_AUTH_SHA256,
        "allowed_change": "abort_threshold_calibration_only",
        "reward_or_scoring_change": False,
    }
    auth["training_config"]["output_dir"] = "checkpoints/grpo/formal_v2"
    thresholds = auth["health_and_abort"]["thresholds"]
    thresholds.pop("intent_response_fail_rate_immediate_above")
    thresholds.pop("intent_response_fail_rate_two_windows_above")
    thresholds.pop("length_collapse_p50_floor")
    thresholds.pop("zero_variance_groups_share_two_windows_at_least")
    thresholds.pop("zero_variance_full_sweep_immediate")
    thresholds.update(
        {
            "relative_baseline": BASELINE,
            "intent_response_fail_rate_abort": {
                "condition": "window_fail_rate > baseline + 0.10",
                "resolved_threshold": 0.7953125,
                "consecutive_windows": 2,
                "absolute_10_percent_trigger_removed": True,
            },
            "zero_variance_group_ratio_abort": {
                "two_window_condition": (
                    "ratio >= baseline + 0.20 for two consecutive windows"
                ),
                "two_window_resolved_threshold": 0.45,
                "full_sweep_immediate_condition": (
                    "ratio >= baseline + 0.50 for one complete train sweep"
                ),
                "full_sweep_resolved_threshold": 0.75,
                "absolute_50_percent_trigger_removed": True,
            },
            "length_collapse_abort": {
                "condition": "p50 < 0.50 * baseline_p50",
                "resolved_floor": 36.75,
                "consecutive_windows": 2,
                "absolute_64_token_floor_removed": True,
            },
        }
    )
    auth["health_and_abort"]["unchanged_abort_classes"] = [
        "reward_rises_without_factual_precision",
        "factual_precision_regression",
        "kl_relative_to_kl0",
        "protocol_pass_rate",
        "non_finite_or_oom",
        "length_explosion",
        "diversity_collapse",
        "data_or_evidence_sha_drift",
        "dev_reverse_divergence",
    ]

    prereg = copy.deepcopy(
        json.loads(V1_PREREG.read_text(encoding="utf-8"))
    )
    prereg["status"] = "preregistered_before_formal_v2_training"
    prereg["run_id"] = "grpo-formal-v2"
    prereg["derived_from"] = {
        "formal_v1_preregistration_sha256": V1_PREREG_SHA256,
        "reason": (
            "formal_v1 self-aborted because an absolute intent fail-rate "
            "threshold was below the frozen pre-training baseline"
        ),
        "reward_or_report_metric_changed": False,
    }
    prereg["abort_recalibration"] = {
        "baseline": BASELINE,
        "intent_response": {
            "healthy_direction": "decrease",
            "abort": "above baseline + 0.10 for two consecutive windows",
            "resolved_threshold": 0.7953125,
        },
        "zero_variance": {
            "abort": "at least baseline + 0.20 for two consecutive windows",
            "resolved_threshold": 0.45,
        },
        "length": {
            "abort": "below 0.50 * baseline p50 for two consecutive windows",
            "resolved_floor": 36.75,
        },
        "absolute_intent_fail_10_percent_trigger_removed": True,
        "absolute_zero_variance_50_percent_trigger_removed": True,
        "absolute_length_64_token_floor_removed": True,
    }
    prereg["expected_abort_risk"] = {
        "intent_response": (
            "Abort only on relative degradation above 0.7953125 for two "
            "consecutive windows; baseline-level failures are training signal."
        ),
        "kl_p95": (
            "formal_v1 KL0 p95 was heavy-tailed. Existing KL0-relative "
            "mean/p95 abort rules remain unchanged and must not be relaxed."
        ),
    }
    return auth, prereg


def main() -> None:
    for path in (V2_AUTH, V2_AUTH_SHA, V2_PREREG, V2_PREREG_SHA):
        if path.exists():
            raise FileExistsError(f"refusing overwrite: {path}")
    auth, prereg = build_v2()
    write_json(V2_AUTH, auth)
    auth_sha = sha256_file(V2_AUTH)
    V2_AUTH_SHA.write_text(
        f"{auth_sha}  {V2_AUTH.name}\n",
        encoding="ascii",
    )
    prereg["authorization_manifest_sha256"] = auth_sha
    write_json(V2_PREREG, prereg)
    prereg_sha = sha256_file(V2_PREREG)
    V2_PREREG_SHA.write_text(
        f"{prereg_sha}  {V2_PREREG.name}\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "authorization_sha256": auth_sha,
                "preregistration_sha256": prereg_sha,
                "baseline": BASELINE,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
