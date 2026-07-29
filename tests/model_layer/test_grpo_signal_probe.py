import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts import freeze_grpo_signal_probe_inputs as freeze


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    ROOT
    / "data"
    / "model_training"
    / "grpo"
    / "grpo_signal_probe_input_manifest.json"
)
MANIFEST_SHA = MANIFEST.with_suffix(".sha256")
REPORT = (
    ROOT
    / "data"
    / "model_training"
    / "grpo"
    / "grpo_signal_probe_report.json"
)
REPORT_SHA = REPORT.with_suffix(".sha256")
PROBE_SCRIPT = ROOT / "training" / "grpo" / "run_signal_probe.py"


def test_probe_input_builder_covers_frozen_16_train_4_dev():
    manifest = freeze.build_manifest()
    cases = manifest["cases"]

    assert manifest["counts"] == {
        "cases": 20,
        "train": 16,
        "dev": 4,
        "per_intent": {
            "compare": 5,
            "knowledge": 5,
            "recommend": 5,
            "sales": 5,
        },
        "expected_completions": 160,
    }
    assert Counter(case["split"] for case in cases) == {
        "train": 16,
        "dev": 4,
    }
    assert all(case["evidence_claims"] for case in cases)
    assert all(
        case["intent_response_spec"]["query_attribute_anchors"]
        for case in cases
    )
    three_target = next(
        case for case in cases if case["id"] == "reward-compare-005"
    )
    assert len(three_target["target_entities"]) == 3
    assert three_target["known_reward_contract_risks"]


def test_probe_contract_has_pre_registered_fire_gate_and_no_tuning_feedback():
    manifest = freeze.build_manifest()

    assert manifest["sampling_config"] == {
        "policy": "Qwen2.5-7B-Instruct plus frozen SFT best adapter",
        "num_generations": 8,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.95,
        "max_prompt_length": 2560,
        "max_completion_length": 512,
        "max_total_sequence_length": 3072,
        "beta_recorded_not_used_without_kl": 0.01,
        "fp16": True,
        "bf16": False,
        "seed": 20260725,
        "gpu_processes": 1,
    }
    assert manifest["fire_gate"][
        "nonzero_variance_group_ratio_lower_bound"
    ] == 0.30
    assert manifest["fire_gate"][
        "nonzero_variance_group_ratio_clear_pass"
    ] == 0.40
    assert "No dev metric may change any hyperparameter" in (
        manifest["authorization"]["dev_use"]
    )
    assert manifest["execution_contract"]["loss"] is False
    assert manifest["execution_contract"]["backward"] is False
    assert manifest["execution_contract"]["optimizer_step"] is False
    assert manifest["execution_contract"]["checkpoint"] is False


def test_frozen_probe_manifest_and_report_companion_hashes_when_present():
    if MANIFEST.exists():
        digest, filename = MANIFEST_SHA.read_text(
            encoding="ascii"
        ).strip().split(maxsplit=1)
        assert digest == hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
        assert filename == MANIFEST.name
        assert json.loads(MANIFEST.read_text(encoding="utf-8")) == (
            freeze.build_manifest()
        )
    if REPORT.exists():
        digest, filename = REPORT_SHA.read_text(
            encoding="ascii"
        ).strip().split(maxsplit=1)
        assert digest == hashlib.sha256(REPORT.read_bytes()).hexdigest()
        assert filename == REPORT.name


def test_probe_source_forbids_training_and_checkpoint_calls_when_present():
    if not PROBE_SCRIPT.exists():
        return
    source = PROBE_SCRIPT.read_text(encoding="utf-8")

    assert ".backward(" not in source
    assert "GRPOTrainer" not in source
    assert "optimizer.step(" not in source
    assert ".save_pretrained(" not in source
    assert ".save_model(" not in source
