import pytest

from training.sft import profile_training_plan as profile


def test_summarize_token_lengths_uses_nearest_rank_percentiles():
    summary = profile.summarize_token_lengths([10, 20, 30, 40, 50])

    assert summary == {
        "count": 5,
        "p50": 30,
        "p95": 50,
        "p99": 50,
        "max": 50,
    }


def test_validate_v100_precision_rejects_non_fp16_autocast():
    config = {
        "data": {"max_seq_len": 4096},
        "train": {
            "fp16": True,
            "bf16": False,
            "autocast_dtype": "float16",
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "epochs": 3,
            "learning_rate": 0.0002,
            "warmup_ratio": 0.03,
        },
    }

    assert profile.validate_v100_training_plan(config) == {
        "max_seq_len": 4096,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "effective_batch_size": 16,
        "epochs": 3,
        "learning_rate": 0.0002,
        "warmup_ratio": 0.03,
        "autocast_dtype": "float16",
    }

    config["train"]["autocast_dtype"] = "bfloat16"
    with pytest.raises(ValueError, match="autocast_dtype=float16"):
        profile.validate_v100_training_plan(config)


def test_estimate_wall_clock_uses_real_micro_step_and_accumulation():
    estimate = profile.estimate_wall_clock(
        train_examples=2250,
        epochs=3,
        micro_batch_size=1,
        gradient_accumulation_steps=16,
        mean_micro_step_sec=2.0,
        overhead_ratio=0.15,
    )

    assert estimate == {
        "train_examples": 2250,
        "epochs": 3,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "total_micro_steps": 6750,
        "optimizer_steps": 422,
        "measured_micro_step_sec": 2.0,
        "estimated_compute_sec": 13500.0,
        "overhead_ratio": 0.15,
        "estimated_wall_clock_sec": 15525.0,
        "estimated_wall_clock_hours": 4.31,
    }
