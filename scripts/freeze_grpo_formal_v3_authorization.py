#!/usr/bin/env python3
"""Freeze formal_v3 authorization after fixed dev-4 step0 baseline exists."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GRPO_DIR = ROOT / "data" / "model_training" / "grpo"
V2_AUTH = GRPO_DIR / "grpo_formal_v2_authorization.json"
V2_AUTH_SHA = V2_AUTH.with_suffix(".sha256")
V2_PREREG = GRPO_DIR / "grpo_formal_v2_preregistered_report.json"
V2_PREREG_SHA = V2_PREREG.with_suffix(".sha256")
V3_BASELINE = GRPO_DIR / "grpo_formal_v3_fixed_dev4_step0_baseline.json"
V3_BASELINE_SHA = V3_BASELINE.with_suffix(".sha256")
V3_AUTH = GRPO_DIR / "grpo_formal_v3_authorization.json"
V3_AUTH_SHA = V3_AUTH.with_suffix(".sha256")
V3_PREREG = GRPO_DIR / "grpo_formal_v3_preregistered_report.json"
V3_PREREG_SHA = V3_PREREG.with_suffix(".sha256")
DIAG6_FINAL = (
    GRPO_DIR
    / "formal_v2"
    / "diagnostics_readonly"
    / "diag6_fix_proposal_final.json"
)
DIAG6_FINAL_SHA = DIAG6_FINAL.with_name(DIAG6_FINAL.name + ".sha256")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_sidecar(path: Path) -> str:
    digest, filename = path.with_suffix(".sha256").read_text(
        encoding="ascii"
    ).strip().split(maxsplit=1)
    if filename != path.name:
        raise ValueError(f"SHA sidecar filename drift: {path}")
    if digest != sha256_file(path):
        raise ValueError(f"SHA sidecar digest drift: {path}")
    return digest


def write_json_exclusive(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    for path in (V3_AUTH, V3_AUTH_SHA, V3_PREREG, V3_PREREG_SHA):
        if path.exists():
            raise FileExistsError(f"refusing overwrite: {path}")
    v2_auth_sha = read_sidecar(V2_AUTH)
    v2_prereg_sha = read_sidecar(V2_PREREG)
    baseline_sha = read_sidecar(V3_BASELINE)
    diag6_sha = DIAG6_FINAL_SHA.read_text(
        encoding="ascii"
    ).strip().split(maxsplit=1)[0]
    if diag6_sha != sha256_file(DIAG6_FINAL):
        raise ValueError("diag6 final SHA drift")

    baseline = json.loads(V3_BASELINE.read_text(encoding="utf-8"))
    if baseline["run_id"] != "grpo-formal-v3":
        raise ValueError("baseline run_id drift")
    if baseline["fake_reward_gate"]["excluded_prompt_ids"] != [
        "reward-knowledge-003"
    ]:
        raise ValueError("knowledge exclusion option drift")

    auth = copy.deepcopy(json.loads(V2_AUTH.read_text(encoding="utf-8")))
    auth["status"] = "authorized_once_before_formal_grpo_v3_training"
    auth["run_id"] = "grpo-formal-v3"
    auth["training_config"]["output_dir"] = "checkpoints/grpo/formal_v3"
    auth["fixed_dev4_probe_step0_baseline"] = {
        "path": str(V3_BASELINE.relative_to(ROOT)),
        "sha256": baseline_sha,
        "computed_before_optimizer_step": True,
        "optimizer_step_performed": False,
        "prompt_ids": baseline["prompt_ids"],
        "generation_parameters": baseline["generation_parameters"],
        "fake_reward_gate": {
            "knowledge_handling_option": "A_exclude_knowledge_003",
            "included_prompt_ids": baseline["fake_reward_gate"][
                "included_prompt_ids"
            ],
            "excluded_prompt_ids": baseline["fake_reward_gate"][
                "excluded_prompt_ids"
            ],
            "scientific_limitation": baseline["fake_reward_gate"][
                "scientific_limitation"
            ],
        },
    }
    thresholds = auth["health_and_abort"]["thresholds"]
    thresholds["reward_gain_fake_min"] = 0.08
    thresholds["factual_precision_gain_required"] = 0.01
    thresholds["fake_reward_rise_without_factual_precision"] = {
        "condition": (
            "previous_reward_gain >= 0.08 and window_reward_gain >= 0.08 "
            "and previous_factual_gain < 0.01 and "
            "window_factual_gain < 0.01"
        ),
        "implementation": "per_window_AND_no_cross_window_min",
        "reward_gain_fake_min": 0.08,
        "factual_precision_gain_required": 0.01,
        "knowledge_handling_option": "A_exclude_knowledge_003",
        "gain_population": baseline["fake_reward_gate"][
            "included_prompt_ids"
        ],
        "excluded_from_gain_population": baseline["fake_reward_gate"][
            "excluded_prompt_ids"
        ],
        "thresholds_relaxed": False,
    }
    thresholds["fixed_dev4_health_probe"] = {
        "prompt_ids": baseline["prompt_ids"],
        "every_optimizer_steps": auth["training_config"][
            "health_window_steps"
        ],
        "no_grad": True,
        "baseline_path": str(V3_BASELINE.relative_to(ROOT)),
        "baseline_sha256": baseline_sha,
        "rolling_train_windows_drive_abort": False,
    }
    auth["frozen_inputs"]["formal_v3_step0_baseline_sha256"] = baseline_sha
    auth["frozen_inputs"]["diag6_final_sha256"] = diag6_sha
    auth["derived_from"] = {
        "formal_v2_authorization_sha256": v2_auth_sha,
        "formal_v2_preregistration_sha256": v2_prereg_sha,
        "diag6_final_sha256": diag6_sha,
        "allowed_change": (
            "per-window AND fake reward gate plus fixed dev-4 no_grad "
            "health probe; option A excludes knowledge-003 from this gate "
            "gain population only"
        ),
        "reward_fn_changed": False,
        "reward_or_scoring_or_reporting_metric_changed": False,
        "thresholds_relaxed": False,
        "terminal_clause": (
            "If clean formal_v3 rerun aborts, stop and deliver SFT-only; "
            "no further structural changes."
        ),
    }

    prereg = copy.deepcopy(json.loads(V2_PREREG.read_text(encoding="utf-8")))
    prereg["status"] = "preregistered_before_formal_v3_training"
    prereg["run_id"] = "grpo-formal-v3"
    prereg["derived_from"] = auth["derived_from"]
    prereg["formal_v3_fix"] = {
        "fake_reward_gate": thresholds[
            "fake_reward_rise_without_factual_precision"
        ],
        "fixed_dev4_health_probe": thresholds["fixed_dev4_health_probe"],
        "fixed_dev4_step0_baseline": {
            "path": str(V3_BASELINE.relative_to(ROOT)),
            "sha256": baseline_sha,
            "reward_mean": baseline["reward_mean"],
            "factual_precision_mean": baseline[
                "factual_precision_mean"
            ],
            "fake_reward_gate": baseline["fake_reward_gate"],
            "kl0_mean": baseline["kl0_mean"],
            "kl0_p95": baseline["kl0_p95"],
        },
        "knowledge_scientific_limitation": (
            "fake_reward_rise_without_factual_gain does not cover "
            "knowledge-003 factual gain under option A; knowledge remains "
            "reported in normal health metrics and checkpoint/dev diagnostics."
        ),
    }

    write_json_exclusive(V3_AUTH, auth)
    auth_sha = sha256_file(V3_AUTH)
    V3_AUTH_SHA.write_text(f"{auth_sha}  {V3_AUTH.name}\n", encoding="ascii")
    prereg["authorization_manifest_sha256"] = auth_sha
    write_json_exclusive(V3_PREREG, prereg)
    prereg_sha = sha256_file(V3_PREREG)
    V3_PREREG_SHA.write_text(
        f"{prereg_sha}  {V3_PREREG.name}\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "authorization_sha256": auth_sha,
                "preregistration_sha256": prereg_sha,
                "fixed_dev4_step0_baseline_sha256": baseline_sha,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
