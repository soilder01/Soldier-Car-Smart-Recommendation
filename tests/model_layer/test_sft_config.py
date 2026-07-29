from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
TRAIN_REQUIREMENTS = ROOT / "requirements-train.txt"
SFT_CONFIG = ROOT / "training" / "sft" / "qlora_sft_config.yaml"
SFT_README = ROOT / "training" / "sft" / "README.md"
SFT_REPORT = ROOT / "docs" / "model_layer_phase3_sft_report.md"


def _requirement_names() -> set[str]:
    names = set()
    for raw_line in TRAIN_REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        names.add(
            line.split(";", 1)[0]
            .split("[", 1)[0]
            .split("=", 1)[0]
            .split("<", 1)[0]
            .split(">", 1)[0]
            .strip()
            .lower()
        )
    return names


def test_training_requirements_are_isolated_and_complete():
    names = _requirement_names()

    assert {
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "trl",
        "bitsandbytes",
        "sentencepiece",
        "protobuf",
        "pyyaml",
    } <= names
    assert "vllm" not in names


def test_sft_config_locks_v100_fp16_qlora_contract():
    config = yaml.safe_load(SFT_CONFIG.read_text(encoding="utf-8"))

    assert config["model"] == {
        "base_model_path": "models/Qwen2.5-7B-Instruct",
        "output_dir": "checkpoints/sft",
    }
    assert config["data"] == {
        "train_file": "data/model_training/sft_train.jsonl",
        "val_file": "data/model_training/sft_val.jsonl",
        "max_seq_len": 5632,
    }
    assert config["qlora"] == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_use_double_quant": True,
    }
    assert config["lora"]["r"] == 16
    assert config["lora"]["alpha"] == 32
    assert config["lora"]["dropout"] == 0.05
    assert config["lora"]["target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    assert config["train"]["fp16"] is True
    assert config["train"]["bf16"] is False
    assert config["train"]["autocast_dtype"] == "float16"
    assert config["train"]["gradient_checkpointing"] is True
    assert config["train"]["per_device_train_batch_size"] == 1
    assert config["train"]["gradient_accumulation_steps"] == 16
    assert config["train"]["epochs"] == 3
    assert config["train"]["learning_rate"] == 0.0002
    assert config["train"]["warmup_ratio"] == 0.03
    assert config["train"]["evaluation_strategy"] == "epoch"
    assert config["train"]["save_strategy"] == "epoch"
    assert config["train"]["save_total_limit"] == 3
    assert config["train"]["load_best_model_at_end"] is True
    assert config["train"]["metric_for_best_model"] == "eval_loss"
    assert config["train"]["greater_is_better"] is False
    assert config["held_out_evaluation"] == {
        "harness_manifest": (
            "data/model_training/eval/frozen_qwen_heldout_harness.json"
        ),
        "evaluate_once_after_checkpoint_selection": True,
        "use_for_checkpoint_selection": False,
    }
    assert config["runtime"]["training_environment"] == ".venv-train"


def test_sft_docs_record_completed_training_without_false_improvement_claim():
    readme = SFT_README.read_text(encoding="utf-8")
    report = SFT_REPORT.read_text(encoding="utf-8")

    assert "服务 `.venv`" in readme
    assert "独立训练环境" in readme
    assert "float16" in readme
    assert "bfloat16" in readme
    assert "当前状态：completed" in report
    assert "adapter reload：已验证" in report
    assert "0/40=0.0%" in report
    assert "不得声称模型能力提升" in report
