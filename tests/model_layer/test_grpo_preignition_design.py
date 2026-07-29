import json
import hashlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / "docs" / "grpo_rlvr_grounding_offline_design.md"
LOCK = ROOT / "training" / "grpo" / "requirements-cu118-design.lock.txt"
COMPATIBILITY = (
    ROOT
    / "data"
    / "model_training"
    / "grpo"
    / "trl_compatibility_static_report.json"
)
TRAIN_ENTRY = ROOT / "training" / "grpo" / "train_grpo.py"
DEV_AUTHORIZATION = (
    ROOT
    / "data"
    / "model_training"
    / "grpo"
    / "grpo_dev_authorization_manifest.json"
)
DEV_AUTHORIZATION_SHA = DEV_AUTHORIZATION.with_suffix(".sha256")
SPLIT_MANIFEST = (
    ROOT
    / "data"
    / "model_training"
    / "grpo"
    / "reward_train_dev_manifest.json"
)


def test_design_locks_health_windows_abort_and_intent_response_gate():
    text = DESIGN.read_text(encoding="utf-8")

    assert "每 `N=5` 个 optimizer step" in text
    assert "伪 reward 上升" in text
    assert "长度塌缩" in text
    assert "reward 零方差" in text
    assert "resume_allowed=false" in text
    assert "G_intent_response = deterministic_intent_response_check" in text
    assert "失败直接 `reward=0`" in text


def test_design_requires_nonzero_adapter_off_reference_and_static_memory_budget():
    text = DESIGN.read_text(encoding="utf-8")

    assert "`beta=0` 被禁止" in text
    assert "初值锁定为 `beta=0.01`" in text
    assert "`disable_adapter()`" in text
    assert "ref_model=None + disable_adapter()" in text
    assert "25.9-27.9" in text
    assert "静态余量约 3.8-5.8 GiB" in text


def test_trl_compatibility_is_pinned_but_not_claimed_installed():
    lock = LOCK.read_text(encoding="utf-8")
    report = json.loads(COMPATIBILITY.read_text(encoding="utf-8"))

    for requirement in (
        "torch==2.3.1+cu118",
        "transformers==4.46.3",
        "accelerate==0.34.2",
        "trl==0.14.0",
        "datasets==2.21.0",
        "bitsandbytes==0.43.3",
    ):
        assert requirement in lock
    assert report["status"] == "resolver_passed_no_install"
    assert report["installed_before_and_after"]["trl"] == "MISSING"
    assert report["installed_before_and_after"]["datasets"] == "MISSING"
    assert report["pip_dry_run"]["environment_modified"] is False
    assert report["trl_metadata"]["contains_grpo_trainer"] is True


def test_training_entry_remains_fail_closed():
    completed = subprocess.run(
        [sys.executable, str(TRAIN_ENTRY)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "gated" in completed.stderr + completed.stdout


def test_dev_authorization_is_frozen_and_forbids_all_hyperparameter_feedback():
    manifest = json.loads(DEV_AUTHORIZATION.read_text(encoding="utf-8"))
    actual_sha = hashlib.sha256(DEV_AUTHORIZATION.read_bytes()).hexdigest()
    recorded_sha, recorded_name = DEV_AUTHORIZATION_SHA.read_text(
        encoding="ascii"
    ).strip().split(maxsplit=1)

    assert recorded_sha == actual_sha
    assert recorded_name == DEV_AUTHORIZATION.name
    assert manifest["bound_split_manifest"] == {
        "path": "data/model_training/grpo/reward_train_dev_manifest.json",
        "sha256": hashlib.sha256(SPLIT_MANIFEST.read_bytes()).hexdigest(),
    }
    policy = manifest["dev_4_authorization"]
    assert policy["read_only"] is True
    assert policy["gradient_updates_allowed"] is False
    assert set(policy["only_allowed_uses"]) == {
        "one_way_early_stop",
        "abort_trigger",
        "post_training_locked_hyperparameter_checkpoint_selection",
    }
    assert policy["early_stop"]["retroactive_checkpoint_rescue_allowed"] is False
    assert policy["checkpoint_selection"]["requires_training_complete"] is True
    assert policy["checkpoint_selection"]["requires_hyperparameters_locked"] is True

    prohibition = manifest["hyperparameter_feedback_prohibition"]
    assert prohibition["mode"] == "forbidden_fail_closed"
    assert prohibition["dev_metrics_may_change_any_hyperparameter"] is False
    assert prohibition["scope"] == "all_hyperparameters_without_exception"
    assert {
        "beta",
        "learning_rate",
        "planned_optimizer_steps",
        "max_steps",
        "reward_weights",
        "reward_thresholds",
    }.issubset(prohibition["explicitly_forbidden_targets"])
    assert prohibition["violation_action"] == (
        "invalidate_run_forbid_checkpoint_promotion_and_require_new_authorization"
    )
