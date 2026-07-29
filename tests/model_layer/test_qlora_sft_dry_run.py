import json
import hashlib
from pathlib import Path

import pytest

from training.sft import train_qlora_sft as sft


def _write_config(path: Path, *, model_path: str, train_file: str) -> None:
    path.write_text(
        "\n".join(
            [
                "model:",
                f'  base_model_path: "{model_path}"',
                '  output_dir: "checkpoints/sft"',
                "data:",
                f'  train_file: "{train_file}"',
                '  val_file: "data/model_training/sft_val.jsonl"',
                "  max_seq_len: 128",
                "qlora:",
                "  load_in_4bit: true",
                '  bnb_4bit_quant_type: "nf4"',
                '  bnb_4bit_compute_dtype: "float16"',
                "  bnb_4bit_use_double_quant: true",
                "lora:",
                "  r: 16",
                "  alpha: 32",
                "  dropout: 0.05",
                "  target_modules: [q_proj, k_proj, v_proj]",
                "train:",
                "  fp16: true",
                "  bf16: false",
                '  autocast_dtype: "float16"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_dataset(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "id": "sft-001",
                "intent": "recommend",
                "qwen_chatml": "<|im_start|>system\nx<|im_end|>\n",
                "assistant_char_spans": [[0, 1]],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def test_model_loading_contract_forces_local_nf4(tmp_path: Path):
    model_path = tmp_path / "local-model"
    model_path.mkdir()
    dataset_path = tmp_path / "train.jsonl"
    _write_dataset(dataset_path)
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        model_path=str(model_path),
        train_file=str(dataset_path),
    )

    config = sft.load_training_config(config_path)
    kwargs = sft.build_model_load_kwargs(config, model_path)

    assert kwargs["local_files_only"] is True
    assert kwargs["quantization"]["load_in_4bit"] is True
    assert kwargs["quantization"]["bnb_4bit_quant_type"] == "nf4"
    assert kwargs["quantization"]["bnb_4bit_compute_dtype"] == "float16"
    assert kwargs["quantization"]["bnb_4bit_use_double_quant"] is True


def test_dry_run_reads_one_frozen_qwen_trajectory_without_checkpoint_output(
    tmp_path: Path,
):
    model_path = tmp_path / "local-model"
    model_path.mkdir()
    dataset_path = tmp_path / "train.jsonl"
    _write_dataset(dataset_path)
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        model_path=str(model_path),
        train_file=str(dataset_path),
    )

    config = sft.load_training_config(config_path)
    row = sft.load_one_frozen_batch(config, repo_root=tmp_path)

    assert row["id"] == "sft-001"
    assert row["qwen_chatml"].startswith("<|im_start|>system")
    assert not (tmp_path / "checkpoints" / "sft").exists()


def test_dry_run_records_dependency_failure_without_claiming_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    model_path = tmp_path / "local-model"
    model_path.mkdir()
    dataset_path = tmp_path / "train.jsonl"
    _write_dataset(dataset_path)
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        model_path=str(model_path),
        train_file=str(dataset_path),
    )
    log_path = tmp_path / "dry_run.json"

    def missing_dependencies():
        raise ModuleNotFoundError("No module named 'torch'")

    monkeypatch.setattr(sft, "_load_training_dependencies", missing_dependencies)

    result = sft.run_dry_run(
        config_path=config_path,
        log_path=log_path,
        repo_root=tmp_path,
    )

    assert result["status"] == "blocked"
    assert result["blocker"] == "training_dependency_missing"
    assert "torch" in result["detail"]
    assert json.loads(log_path.read_text(encoding="utf-8")) == result
    assert not (tmp_path / "checkpoints" / "sft").exists()


def test_cli_refuses_full_training_without_dry_run(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("sys.argv", ["train_qlora_sft.py"])

    with pytest.raises(SystemExit, match="select exactly one mode"):
        sft.main()


def test_cli_refuses_planned_train_until_explicitly_enabled(
    monkeypatch: pytest.MonkeyPatch,
):
    expected = {"status": "completed"}
    monkeypatch.setattr(
        sft,
        "run_training",
        lambda **_kwargs: expected,
        raising=False,
    )
    monkeypatch.setattr("sys.argv", ["train_qlora_sft.py", "--train"])

    with pytest.raises(SystemExit) as exit_info:
        sft.main()

    assert exit_info.value.code == 0


def test_training_authorization_locks_hashes_local_only_and_train_venv(
    tmp_path: Path,
):
    model_path = tmp_path / "local-model"
    model_path.mkdir()
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    _write_dataset(train_path)
    _write_dataset(val_path)
    lock_path = tmp_path / "requirements.lock.txt"
    lock_path.write_text("torch==2.3.1+cu118\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        model_path=str(model_path),
        train_file=str(train_path),
    )
    config = sft.load_training_config(config_path)
    config["data"]["val_file"] = str(val_path)
    config_path.write_text(
        __import__("yaml").safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    train_python = tmp_path / ".venv-train" / "bin" / "python"
    train_python.parent.mkdir(parents=True)
    train_python.touch()
    authorization_path = tmp_path / "authorization.json"

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    authorization_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "config": {
                    "path": str(config_path.relative_to(tmp_path)),
                    "sha256": digest(config_path),
                },
                "data": {
                    "train": {
                        "path": str(train_path.relative_to(tmp_path)),
                        "rows": 1,
                        "sha256": digest(train_path),
                    },
                    "validation": {
                        "path": str(val_path.relative_to(tmp_path)),
                        "rows": 1,
                        "sha256": digest(val_path),
                    },
                },
                "environment": {
                    "python": ".venv-train/bin/python",
                    "lock_path": str(lock_path.relative_to(tmp_path)),
                    "lock_sha256": digest(lock_path),
                },
                "model_loading": {
                    "local_files_only": True,
                },
            }
        ),
        encoding="utf-8",
    )

    result = sft.validate_training_authorization(
        config_path=config_path,
        authorization_path=authorization_path,
        repo_root=tmp_path,
        python_executable=train_python,
    )

    assert result["config_sha256"] == digest(config_path)
    assert result["train_sha256"] == digest(train_path)
    assert result["validation_sha256"] == digest(val_path)
    assert result["local_files_only"] is True
    assert result["python_executable"] == str(train_python.resolve())

    train_path.write_text(
        train_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="train SHA-256 mismatch"):
        sft.validate_training_authorization(
            config_path=config_path,
            authorization_path=authorization_path,
            repo_root=tmp_path,
            python_executable=train_python,
        )

    _write_dataset(train_path)
    with pytest.raises(ValueError, match=r"\.venv-train"):
        sft.validate_training_authorization(
            config_path=config_path,
            authorization_path=authorization_path,
            repo_root=tmp_path,
            python_executable=tmp_path / ".venv" / "bin" / "python",
        )


def test_select_best_checkpoint_uses_lowest_validation_loss_only():
    checkpoints = [
        {"epoch": 1, "eval_loss": 0.7, "checkpoint": "epoch-1"},
        {"epoch": 2, "eval_loss": 0.5, "checkpoint": "epoch-2"},
        {"epoch": 3, "eval_loss": 0.6, "checkpoint": "epoch-3"},
    ]

    assert sft.select_best_checkpoint(checkpoints) == checkpoints[1]


def test_require_finite_loss_rejects_nan_and_infinity():
    assert sft.require_finite_loss(0.25) == 0.25
    with pytest.raises(sft.NonFiniteLossError):
        sft.require_finite_loss(float("nan"))
    with pytest.raises(sft.NonFiniteLossError):
        sft.require_finite_loss(float("inf"))


def test_cuda_memory_report_uses_peak_metrics():
    class FakeCuda:
        @staticmethod
        def memory_allocated() -> int:
            return 10 * 1024**2

        @staticmethod
        def memory_reserved() -> int:
            return 20 * 1024**2

        @staticmethod
        def max_memory_allocated() -> int:
            return 30 * 1024**2

        @staticmethod
        def max_memory_reserved() -> int:
            return 40 * 1024**2

    class FakeTorch:
        cuda = FakeCuda()

    assert sft.cuda_memory_report(FakeTorch()) == {
        "memory_allocated_mib": 10.0,
        "memory_reserved_mib": 20.0,
        "peak_memory_allocated_mib": 30.0,
        "peak_memory_reserved_mib": 40.0,
    }
