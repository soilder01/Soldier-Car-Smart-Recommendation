import hashlib
import json
import subprocess
import sys
from pathlib import Path

from training.grpo.formal_training import (
    FAKE_REWARD_GATE_EXCLUDED_PROMPT_IDS,
    FIXED_DEV4_PROBE_IDS,
    abort_reasons,
    fake_reward_gate_case_ids,
    fake_reward_gate_means,
    gradient_checkpointing_is_enabled,
    rollout_generation_cache_context,
)
from training.grpo import train_grpo


ROOT = Path(__file__).resolve().parents[2]
AUTH = (
    ROOT
    / "data"
    / "model_training"
    / "grpo"
    / "grpo_formal_v4_authorization.json"
)
AUTH_SHA = AUTH.with_suffix(".sha256")
PREREG = (
    ROOT
    / "data"
    / "model_training"
    / "grpo"
    / "grpo_formal_v4_preregistered_report.json"
)
PREREG_SHA = PREREG.with_suffix(".sha256")
FORMAL = ROOT / "training" / "grpo" / "formal_training.py"


def test_formal_authorization_and_preregistration_are_sha_frozen():
    auth_digest, auth_name = AUTH_SHA.read_text(
        encoding="ascii"
    ).strip().split(maxsplit=1)
    prereg_digest, prereg_name = PREREG_SHA.read_text(
        encoding="ascii"
    ).strip().split(maxsplit=1)

    assert auth_digest == hashlib.sha256(AUTH.read_bytes()).hexdigest()
    assert auth_name == AUTH.name
    assert prereg_digest == hashlib.sha256(PREREG.read_bytes()).hexdigest()
    assert prereg_name == PREREG.name


def test_five_element_gate_passes_only_with_frozen_grpo_venv(monkeypatch):
    monkeypatch.setattr(
        train_grpo.sys,
        "prefix",
        str(ROOT / ".venv-grpo"),
    )

    receipt = train_grpo.validate_authorization(AUTH)

    assert all(receipt["checks"].values())
    assert set(receipt["checks"]) == {
        "reward_fn_sha_match",
        "split_three_file_sha_match",
        "dev_authorization_sha_match",
        "venv_grpo_path_match",
        "local_files_only",
    }


def test_formal_config_locks_training_health_and_data_boundaries():
    auth = json.loads(AUTH.read_text(encoding="utf-8"))
    config = auth["training_config"]
    policy = auth["data_policy"]
    health = auth["health_and_abort"]

    assert config["max_optimizer_steps"] == 800
    assert config["learning_rate"] == 2e-6
    assert config["lr_scheduler"] == "cosine"
    assert config["warmup_steps"] == 40
    assert config["beta"] == 0.01
    assert config["num_generations"] == 8
    assert config["fp16"] is True
    assert config["bf16"] is False
    assert config["max_completion_length"] == 512
    assert config["max_total_sequence_length"] == 3072
    assert config["health_window_steps"] == 50
    assert config["dev_every_steps"] == 50
    assert (
        policy["gradient_updates_only"]
        == "data/model_training/grpo/reward_train_expanded_v4.jsonl"
    )
    assert policy["dev_gradient_allowed"] is False
    assert policy["held_out_40_allowed"] is False
    assert policy["final_40_allowed_during_training_or_selection"] is False
    assert health["abort_action"]["resume_allowed"] is False
    assert health["abort_action"]["automatic_restart"] is False
    thresholds = health["thresholds"]
    assert "intent_response_fail_rate_immediate_above" not in thresholds
    assert "intent_response_fail_rate_two_windows_above" not in thresholds
    assert "length_collapse_p50_floor" not in thresholds
    assert thresholds["reward_gain_fake_min"] == 0.08
    assert thresholds["factual_precision_gain_required"] == 0.01
    assert thresholds["kl_p95_two_windows"] == "monitor_only"
    assert thresholds["kl_explosive_abort"]["beta_nonzero"] is True
    assert health["abort_classes_v4"] == [
        "non_finite_or_oom",
        "protocol_pass_rate",
        "fake_reward_rise_without_factual_precision",
        "kl_explosive_divergence",
        "data_or_evidence_sha_drift",
    ]


def test_scientific_expectations_are_preregistered_before_training():
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    expected = prereg["scientific_expectations"]

    assert expected["knowledge"]["expected"] == "likely_no_improvement"
    assert expected["knowledge"]["gate_may_be_changed_to_force_gain"] is False
    assert expected["concision"]["may_claim_grpo_improved_concision"] is False
    assert expected["three_vehicle_compare"]["expected"] == "fail_closed"
    assert prereg["terminal_evaluation_plan"]["automatic_final_40"] is False


def test_formal_source_arms_window_abort_and_never_names_eval_sets():
    source = FORMAL.read_text(encoding="utf-8")

    for required in (
        "HEALTH_WINDOW",
        "AUTO_ABORT",
        "fake_reward_rise_without_factual_gain",
        "fixed_dev4_no_grad_probe",
        "FAKE_REWARD_GATE_EXCLUDED_PROMPT_IDS",
        "rollout_generation_cache_context",
        "kl_explosive_divergence_two_windows",
        '"resume_allowed": False',
        "disable_adapter()",
    ):
        assert required in source
    assert "intent_response_fail_rate_above_10_percent" not in source
    assert "intent_response_fail_rate_two_windows_above" not in source
    assert "held_out.jsonl" not in source
    assert "grpo_final_held_out.jsonl" not in source


def _window(**overrides):
    value = {
        "non_finite_count": 0,
        "protocol_pass_rate": 1.0,
        "intent_response_fail_rate": 0.60,
        "empty_rate": 0.0,
        "refusal_rate": 0.0,
        "length_p95": 100.0,
        "nonfact_copy_ratio_mean": 0.0,
        "kl_mean": 1.0,
        "reward_mean": 0.4,
        "factual_precision_mean": 0.5,
        "length_p50": 61.0,
        "low_unique_group_share": 0.0,
        "distinct_2": 0.4,
        "zero_variance_group_share": 0.2,
        "kl_p95": 6.0,
        "kl_max": 10.0,
    }
    value.update(overrides)
    value.setdefault(
        "fake_reward_gate",
        {
            "reward_mean": value["reward_mean"],
            "factual_precision_mean": value["factual_precision_mean"],
        },
    )
    return value


def _baseline():
    return {
        "reward_mean": 0.3,
        "factual_precision_mean": 0.4,
        "intent_response_fail_rate": 0.6953125,
        "zero_variance_group_ratio": 0.25,
        "completion_length_p50": 73.5,
        "kl0_mean": 1.0,
        "kl0_p95": 6.0,
        "fake_reward_gate": {
            "reward_mean": 0.3,
            "factual_precision_mean": 0.4,
        },
    }


def test_intent_fail_rate_is_monitor_only_not_abort():
    baseline = _baseline()

    reasons = abort_reasons(
        _window(intent_response_fail_rate=1.0),
        _window(intent_response_fail_rate=1.0),
        baseline,
    )

    assert "intent_response_relative_degradation_two_windows" not in reasons


def test_kl_p95_zero_variance_and_length_are_monitor_only():
    baseline = _baseline()

    reasons = abort_reasons(
        _window(
            zero_variance_group_share=1.0,
            length_p50=1.0,
            kl_p95=100.0,
            kl_max=50.0,
        ),
        _window(
            zero_variance_group_share=1.0,
            length_p50=1.0,
            kl_p95=100.0,
            kl_max=50.0,
        ),
        baseline,
    )

    assert "reward_zero_variance_relative_two_windows" not in reasons
    assert "length_collapse_two_windows" not in reasons
    assert "kl_p95_two_windows" not in reasons


def test_kl_abort_requires_explosive_divergence():
    baseline = _baseline()

    ordinary = abort_reasons(
        _window(kl_mean=4.9, kl_max=99.0),
        _window(kl_mean=4.9, kl_max=99.0),
        baseline,
    )
    explosive = abort_reasons(
        _window(kl_mean=10.0, kl_max=10.0),
        _window(kl_mean=10.0, kl_max=10.0),
        baseline,
    )

    assert "kl_explosive_divergence_two_windows" not in ordinary
    assert "kl_explosive_divergence_two_windows" in explosive


def test_fake_reward_gate_uses_per_window_and_not_cross_window_min():
    baseline = _baseline()

    both_flat = abort_reasons(
        _window(
            reward_mean=0.39,
            factual_precision_mean=0.405,
            fake_reward_gate={
                "reward_mean": 0.39,
                "factual_precision_mean": 0.405,
            },
        ),
        _window(
            reward_mean=0.40,
            factual_precision_mean=0.400,
            fake_reward_gate={
                "reward_mean": 0.40,
                "factual_precision_mean": 0.400,
            },
        ),
        baseline,
    )
    assert "fake_reward_rise_without_factual_gain" in both_flat

    one_window_jumps = abort_reasons(
        _window(
            reward_mean=0.40,
            factual_precision_mean=0.45,
            fake_reward_gate={
                "reward_mean": 0.40,
                "factual_precision_mean": 0.45,
            },
        ),
        _window(
            reward_mean=0.55,
            factual_precision_mean=0.7408333333333333,
            fake_reward_gate={
                "reward_mean": 0.55,
                "factual_precision_mean": 0.7408333333333333,
            },
        ),
        {
            **baseline,
            "reward_mean": 0.3046875,
            "factual_precision_mean": 0.4973958333333333,
            "fake_reward_gate": {
                "reward_mean": 0.3046875,
                "factual_precision_mean": 0.4973958333333333,
            },
        },
    )
    assert "fake_reward_rise_without_factual_gain" not in one_window_jumps


def test_fixed_dev4_prompt_ids_are_frozen_for_probe_windows():
    assert FIXED_DEV4_PROBE_IDS == (
        "reward-recommend-002",
        "reward-compare-004",
        "reward-knowledge-003",
        "reward-sales-004",
    )


def test_fake_reward_gate_excludes_knowledge_symmetrically():
    assert FAKE_REWARD_GATE_EXCLUDED_PROMPT_IDS == ("reward-knowledge-003",)
    assert fake_reward_gate_case_ids(list(FIXED_DEV4_PROBE_IDS)) == [
        "reward-recommend-002",
        "reward-compare-004",
        "reward-sales-004",
    ]

    means = fake_reward_gate_means(
        [
            {
                "id": "reward-recommend-002",
                "reward_mean": 0.6,
                "factual_precision_mean": 0.5,
            },
            {
                "id": "reward-compare-004",
                "reward_mean": 0.3,
                "factual_precision_mean": 0.25,
            },
            {
                "id": "reward-knowledge-003",
                "reward_mean": 1.0,
                "factual_precision_mean": 1.0,
            },
            {
                "id": "reward-sales-004",
                "reward_mean": 0.9,
                "factual_precision_mean": 0.75,
            },
        ]
    )

    assert means["included_prompt_ids"] == [
        "reward-recommend-002",
        "reward-compare-004",
        "reward-sales-004",
    ]
    assert means["excluded_prompt_ids"] == ["reward-knowledge-003"]
    assert means["reward_mean"] == 0.6
    assert means["factual_precision_mean"] == 0.5


def test_fake_reward_thresholds_remain_unrelaxed():
    source = FORMAL.read_text(encoding="utf-8")

    assert "previous_reward_gain >= 0.08" in source
    assert "window_reward_gain >= 0.08" in source
    assert "previous_factual_gain < 0.01" in source
    assert "window_factual_gain < 0.01" in source
    assert "reward_gain >= 0.08 and factual_gain < 0.01" not in source


class _DummyConfig:
    def __init__(self, use_cache=False):
        self.use_cache = use_cache


class _DummyModel:
    def __init__(self):
        self.config = _DummyConfig(use_cache=False)
        self.is_gradient_checkpointing = True
        self.disable_calls = 0
        self.enable_calls = 0

    def gradient_checkpointing_disable(self):
        self.disable_calls += 1
        self.is_gradient_checkpointing = False

    def gradient_checkpointing_enable(self, **_kwargs):
        self.enable_calls += 1
        self.is_gradient_checkpointing = True


def test_rollout_generation_cache_context_restores_training_state():
    model = _DummyModel()
    generation_config = _DummyConfig(use_cache=True)

    assert gradient_checkpointing_is_enabled(model) is True
    assert model.config.use_cache is False
    assert generation_config.use_cache is True

    with rollout_generation_cache_context(model, generation_config) as state:
        assert state["rollout_use_cache"] is True
        assert model.config.use_cache is True
        assert generation_config.use_cache is True
        assert gradient_checkpointing_is_enabled(model) is False

    assert model.config.use_cache is False
    assert generation_config.use_cache is False
    assert gradient_checkpointing_is_enabled(model) is True
    assert model.disable_calls == 1
    assert model.enable_calls == 1


def test_entry_without_explicit_authorization_remains_gated():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "training" / "grpo" / "train_grpo.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "gated" in completed.stdout + completed.stderr
