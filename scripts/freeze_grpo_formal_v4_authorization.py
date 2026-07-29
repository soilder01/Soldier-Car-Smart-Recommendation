#!/usr/bin/env python3
"""Freeze formal_v4 authorization after expanded data and baseline exist."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GRPO_DIR = ROOT / "data" / "model_training" / "grpo"
V3_AUTH = GRPO_DIR / "grpo_formal_v3_authorization.json"
V3_PREREG = GRPO_DIR / "grpo_formal_v3_preregistered_report.json"
EXPANDED_TRAIN = GRPO_DIR / "reward_train_expanded_v4.jsonl"
EXPANDED_MANIFEST = GRPO_DIR / "reward_train_expanded_v4_manifest.json"
INPUT_MANIFEST = GRPO_DIR / "grpo_expanded_v4_input_manifest.json"
BASELINE = GRPO_DIR / "grpo_formal_v4_fixed_dev4_step0_baseline.json"
V4_AUTH = GRPO_DIR / "grpo_formal_v4_authorization.json"
V4_PREREG = GRPO_DIR / "grpo_formal_v4_preregistered_report.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_sidecar(path: Path) -> str:
    digest, filename = path.with_suffix(".sha256").read_text(
        encoding="ascii"
    ).strip().split(maxsplit=1)
    if filename != path.name or digest != sha256_file(path):
        raise ValueError(f"SHA sidecar drift: {path}")
    return digest


def write_json_exclusive(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    for path in (
        V4_AUTH,
        V4_AUTH.with_suffix(".sha256"),
        V4_PREREG,
        V4_PREREG.with_suffix(".sha256"),
    ):
        if path.exists():
            raise FileExistsError(f"refusing overwrite: {path}")
    v3_auth_sha = read_sidecar(V3_AUTH)
    v3_prereg_sha = read_sidecar(V3_PREREG)
    expanded_train_sha = read_sidecar(EXPANDED_TRAIN)
    expanded_manifest_sha = read_sidecar(EXPANDED_MANIFEST)
    input_manifest_sha = read_sidecar(INPUT_MANIFEST)
    baseline_sha = read_sidecar(BASELINE)
    expanded_manifest = json.loads(EXPANDED_MANIFEST.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))

    auth = copy.deepcopy(json.loads(V3_AUTH.read_text(encoding="utf-8")))
    auth["status"] = "authorized_once_before_formal_grpo_v4_training"
    auth["run_id"] = "grpo-formal-v4"
    config = auth["training_config"]
    config.update(
        {
            "train_prompts": expanded_manifest["counts"]["train"],
            "dev_prompts": expanded_manifest["counts"]["dev"],
            "planned_sweeps": None,
            "max_optimizer_steps": 800,
            "learning_rate": 2e-6,
            "lr_scheduler": "cosine",
            "warmup_steps": 40,
            "save_every_steps": 100,
            "dev_every_steps": 50,
            "health_window_steps": 50,
            "output_dir": "checkpoints/grpo/formal_v4",
        }
    )
    auth["authorization_gate"]["split"] = {
        "train_path": str(EXPANDED_TRAIN.relative_to(ROOT)),
        "train_sha256": expanded_train_sha,
        "dev_path": "data/model_training/grpo/reward_dev_4.jsonl",
        "dev_sha256": auth["authorization_gate"]["split"]["dev_sha256"],
        "manifest_path": str(EXPANDED_MANIFEST.relative_to(ROOT)),
        "manifest_sha256": expanded_manifest_sha,
    }
    auth["data_policy"]["gradient_updates_only"] = (
        "data/model_training/grpo/reward_train_expanded_v4.jsonl"
    )
    auth["frozen_inputs"].update(
        {
            "expanded_v4_train_sha256": expanded_train_sha,
            "expanded_v4_manifest_sha256": expanded_manifest_sha,
            "expanded_v4_input_manifest_sha256": input_manifest_sha,
            "formal_v4_step0_baseline_sha256": baseline_sha,
        }
    )
    auth["fixed_dev4_probe_step0_baseline"] = {
        "path": str(BASELINE.relative_to(ROOT)),
        "sha256": baseline_sha,
        "computed_before_optimizer_step": True,
        "optimizer_step_performed": False,
        "prompt_ids": baseline["prompt_ids"],
        "generation_parameters": baseline["generation_parameters"],
        "fake_reward_gate": baseline["fake_reward_gate"],
    }
    thresholds = auth["health_and_abort"]["thresholds"]
    thresholds["reward_gain_fake_min"] = 0.08
    thresholds["factual_precision_gain_required"] = 0.01
    thresholds["kl_p95_two_windows"] = "monitor_only"
    thresholds["kl_explosive_abort"] = {
        "condition": "kl_mean >= max(5.0, 10*KL0_mean) for two fixed eval windows, or kl_max >=100 with length/refusal/copying symptom",
        "beta_nonzero": True,
    }
    auth["health_and_abort"]["abort_classes_v4"] = [
        "non_finite_or_oom",
        "protocol_pass_rate",
        "fake_reward_rise_without_factual_precision",
        "kl_explosive_divergence",
        "data_or_evidence_sha_drift",
    ]
    auth["health_and_abort"]["monitor_only_v4"] = [
        "kl_p95_two_windows",
        "kl_max_spike_without_behavioral_symptom",
        "zero_variance_group_share",
        "length_distribution",
        "intent_response_fail_rate",
    ]
    auth["derived_from"] = {
        "formal_v3_authorization_sha256": v3_auth_sha,
        "formal_v3_preregistration_sha256": v3_prereg_sha,
        "allowed_change": "expanded train set, longer schedule, KL p95 monitor-only, explosive KL abort only",
        "reward_fn_changed": False,
        "fake_reward_gate_changed": False,
        "held_out_final_isolation_changed": False,
        "thresholds_reward_fake_changed": False,
    }

    prereg = copy.deepcopy(json.loads(V3_PREREG.read_text(encoding="utf-8")))
    prereg["status"] = "preregistered_before_formal_v4_training"
    prereg["run_id"] = "grpo-formal-v4"
    prereg["derived_from"] = auth["derived_from"]
    prereg["formal_v4"] = {
        "expanded_train": expanded_manifest["counts"],
        "expanded_train_sha256": expanded_train_sha,
        "expanded_input_manifest_sha256": input_manifest_sha,
        "fixed_dev4_step0_baseline_sha256": baseline_sha,
        "schedule": {
            "max_optimizer_steps": config["max_optimizer_steps"],
            "learning_rate": config["learning_rate"],
            "lr_scheduler": config["lr_scheduler"],
            "warmup_steps": config["warmup_steps"],
            "health_window_steps": config["health_window_steps"],
            "dev_every_steps": config["dev_every_steps"],
        },
        "kl_policy": {
            "beta": config["beta"],
            "beta_zero_forbidden": True,
            "kl_p95": "monitor_only",
            "abort": thresholds["kl_explosive_abort"],
        },
    }

    write_json_exclusive(V4_AUTH, auth)
    auth_sha = sha256_file(V4_AUTH)
    V4_AUTH.with_suffix(".sha256").write_text(
        f"{auth_sha}  {V4_AUTH.name}\n",
        encoding="ascii",
    )
    prereg["authorization_manifest_sha256"] = auth_sha
    write_json_exclusive(V4_PREREG, prereg)
    prereg_sha = sha256_file(V4_PREREG)
    V4_PREREG.with_suffix(".sha256").write_text(
        f"{prereg_sha}  {V4_PREREG.name}\n",
        encoding="ascii",
    )
    print(json.dumps({"authorization_sha256": auth_sha, "preregistration_sha256": prereg_sha, "baseline_sha256": baseline_sha, "expanded_train_sha256": expanded_train_sha}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
