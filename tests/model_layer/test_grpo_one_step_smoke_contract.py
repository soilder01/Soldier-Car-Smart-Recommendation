import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "training" / "grpo" / "run_one_step_smoke.py"
CASE = (
    ROOT
    / "data"
    / "model_training"
    / "grpo"
    / "grpo_one_step_smoke_case.json"
)
CASE_SHA = CASE.with_suffix(".sha256")
REPORT = (
    ROOT
    / "data"
    / "model_training"
    / "grpo"
    / "grpo_one_step_smoke_report.json"
)
REPORT_SHA = REPORT.with_suffix(".sha256")


def test_smoke_case_is_frozen_to_one_train_group_and_no_online_tools():
    case = json.loads(CASE.read_text(encoding="utf-8"))
    recorded_sha, recorded_name = CASE_SHA.read_text(
        encoding="ascii"
    ).strip().split(maxsplit=1)

    assert recorded_sha == hashlib.sha256(CASE.read_bytes()).hexdigest()
    assert recorded_name == CASE.name
    assert case["prompt_case"]["id"] == "reward-compare-001"
    assert case["scope"] == {
        "train_prompt_group_count": 1,
        "num_generations": 8,
        "optimizer_steps": 1,
        "checkpoint_saving_allowed": False,
        "multi_turn_tool_execution_inside_rollout": False,
        "validated_path": (
            "frozen_evidence_terminal_answer_rollout_to_reward_to_kl_to_update"
        ),
        "limitation": (
            "TRL 0.14.0 GRPOTrainer generates terminal completions and does "
            "not execute the project's multi-turn tool environment. This "
            "smoke supplies a deterministic precomputed observation from "
            "the local catalog; it does not claim to validate online tool "
            "orchestration."
        ),
    }


def test_smoke_script_locks_one_step_fp16_kl_and_no_checkpoint_save_calls():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "max_steps=1" in source
    assert "num_generations=8" in source
    assert "beta=0.01" in source
    assert "fp16=True" in source
    assert "bf16=False" in source
    assert 'save_strategy="no"' in source
    assert "use_vllm=False" in source
    assert "disable_adapter_calls_during_train" in source
    assert "data_seed=" not in source
    assert ".save_model(" not in source
    assert ".save_pretrained(" not in source


def test_smoke_report_preserves_recovery_qualification_and_required_checks():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    recorded_sha, recorded_name = REPORT_SHA.read_text(
        encoding="ascii"
    ).strip().split(maxsplit=1)

    assert recorded_sha == hashlib.sha256(REPORT.read_bytes()).hexdigest()
    assert recorded_name == REPORT.name
    assert report["status"] == "passed"
    assert report["qualified_by_post_step_reporting_recovery"] is True
    assert report["post_step_reporting_recovery"]["process_exit_code"] == 1
    assert (
        report["post_step_reporting_recovery"][
            "reporting_code_fixed_without_rerunning_optimizer_step"
        ]
        is True
    )
    assert all(report["required_checks"].values())
    assert report["memory"]["rollout_measured_peak_gib"] < 30
    assert report["memory"]["rollout_headroom_gib"] >= 2
    assert report["kl"]["finite"] is True
    assert report["reward"]["all_values_finite"] is True
