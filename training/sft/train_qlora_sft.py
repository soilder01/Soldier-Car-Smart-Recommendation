"""Hash-gated local QLoRA SFT training and dry-run entrypoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "training" / "sft" / "qlora_sft_config.yaml"
DEFAULT_LOG_PATH = ROOT / "data" / "model_training" / "sft_dry_run_report.json"
DEFAULT_AUTHORIZATION_PATH = (
    ROOT / "data" / "model_training" / "sft_training_authorization.json"
)
DEFAULT_TRAINING_REPORT_PATH = (
    ROOT / "data" / "model_training" / "sft_training_report.json"
)
DEFAULT_STEP_LOG_PATH = (
    ROOT / "data" / "model_training" / "sft_training_steps.jsonl"
)
DEFAULT_FAILURE_LOG_PATH = (
    ROOT / "data" / "model_training" / "sft_training_failures.jsonl"
)


class NonFiniteLossError(RuntimeError):
    """Raised when loss or gradients become non-finite."""


def load_training_config(config_path: Path) -> dict[str, Any]:
    """Load and minimally validate the local-only QLoRA configuration."""
    if not config_path.is_file():
        raise FileNotFoundError(f"SFT config not found: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("SFT config must be a mapping")
    for section in ("model", "data", "qlora", "lora"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"SFT config.{section} must be a mapping")
    for section, field in (
        ("model", "base_model_path"),
        ("model", "output_dir"),
        ("data", "train_file"),
        ("data", "max_seq_len"),
    ):
        if field not in config[section]:
            raise ValueError(f"SFT config.{section}.{field} is required")
    return config


def _resolve_repo_path(value: str, repo_root: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("configured path must be a non-empty string")
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root / path


def _model_path(config: dict[str, Any], repo_root: Path) -> Path:
    path = _resolve_repo_path(config["model"]["base_model_path"], repo_root)
    if not path.is_dir():
        raise FileNotFoundError(
            "local base model directory is required and was not found"
        )
    return path


def build_model_load_kwargs(
    config: dict[str, Any],
    model_path: Path,
) -> dict[str, Any]:
    """Return a serializable local-only NF4 load contract for tests and logs."""
    qlora = config["qlora"]
    if qlora.get("load_in_4bit") is not True:
        raise ValueError("QLoRA dry-run requires load_in_4bit=true")
    if qlora.get("bnb_4bit_quant_type") != "nf4":
        raise ValueError("QLoRA dry-run requires NF4 quantization")
    if qlora.get("bnb_4bit_compute_dtype") != "float16":
        raise ValueError("QLoRA dry-run requires float16 compute dtype")
    if qlora.get("bnb_4bit_use_double_quant") is not True:
        raise ValueError("QLoRA dry-run requires double quantization")
    return {
        "local_files_only": True,
        "trust_remote_code": False,
        "model_path_exists": model_path.is_dir(),
        "quantization": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_use_double_quant": True,
        },
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl_row_count(path: Path) -> int:
    with path.open(encoding="utf-8") as input_file:
        return sum(1 for line in input_file if line.strip())


def _canonical_venv_executable(path: str | Path) -> Path:
    executable = Path(path).absolute()
    return executable.parent.resolve() / executable.name


def _authorization_path(
    entry: dict[str, Any],
    *,
    repo_root: Path,
    label: str,
) -> Path:
    value = entry.get("path")
    if not isinstance(value, str) or not value:
        raise ValueError(f"authorization {label}.path must be non-empty")
    return _resolve_repo_path(value, repo_root)


def _validate_authorized_file(
    entry: dict[str, Any],
    *,
    repo_root: Path,
    label: str,
    expected_rows: int | None = None,
) -> tuple[Path, str]:
    path = _authorization_path(entry, repo_root=repo_root, label=label)
    if not path.is_file():
        raise FileNotFoundError(f"authorized {label} file not found: {path}")
    actual_hash = _sha256(path)
    expected_hash = entry.get("sha256")
    if actual_hash != expected_hash:
        raise ValueError(f"{label} SHA-256 mismatch")
    if expected_rows is not None:
        actual_rows = _jsonl_row_count(path)
        if actual_rows != expected_rows:
            raise ValueError(
                f"{label} row count mismatch: "
                f"expected={expected_rows} actual={actual_rows}"
            )
    return path, actual_hash


def _locked_plan(config: dict[str, Any]) -> dict[str, Any]:
    qlora = config["qlora"]
    train = config["train"]
    return {
        "max_seq_len": config["data"]["max_seq_len"],
        "load_in_4bit": qlora["load_in_4bit"],
        "quant_type": qlora["bnb_4bit_quant_type"],
        "compute_dtype": qlora["bnb_4bit_compute_dtype"],
        "double_quant": qlora["bnb_4bit_use_double_quant"],
        "fp16": train["fp16"],
        "bf16": train["bf16"],
        "epochs": train["epochs"],
        "learning_rate": train["learning_rate"],
        "warmup_ratio": train["warmup_ratio"],
        "micro_batch_size": train["per_device_train_batch_size"],
        "gradient_accumulation_steps": train[
            "gradient_accumulation_steps"
        ],
        "gradient_checkpointing": train["gradient_checkpointing"],
        "checkpoint_strategy": train["save_strategy"],
        "selection_metric": train["metric_for_best_model"],
        "greater_is_better": train["greater_is_better"],
    }


def validate_training_authorization(
    *,
    config_path: Path,
    authorization_path: Path,
    repo_root: Path = ROOT,
    python_executable: str | Path = sys.executable,
) -> dict[str, Any]:
    """Validate immutable hashes and local runtime before loading the model."""
    authorization = json.loads(
        authorization_path.read_text(encoding="utf-8")
    )
    if authorization.get("format_version") != 1:
        raise ValueError("unsupported training authorization format")
    config_entry = authorization.get("config")
    data_entries = authorization.get("data")
    environment = authorization.get("environment")
    model_loading = authorization.get("model_loading")
    if not all(
        isinstance(item, dict)
        for item in (
            config_entry,
            data_entries,
            environment,
            model_loading,
        )
    ):
        raise ValueError("training authorization is incomplete")

    authorized_config, config_hash = _validate_authorized_file(
        config_entry,
        repo_root=repo_root,
        label="config",
    )
    if Path(config_path).resolve() != authorized_config.resolve():
        raise ValueError("CLI config path differs from authorized config")
    config = load_training_config(config_path)
    model_path = _model_path(config, repo_root)
    load_contract = build_model_load_kwargs(config, model_path)
    if (
        model_loading.get("local_files_only") is not True
        or load_contract["local_files_only"] is not True
    ):
        raise ValueError("formal SFT requires local_files_only=True")
    if model_loading.get("trust_remote_code") not in {None, False}:
        raise ValueError("formal SFT requires trust_remote_code=False")
    configured_model = model_loading.get("base_model_path")
    if configured_model is not None and (
        _resolve_repo_path(configured_model, repo_root).resolve()
        != model_path.resolve()
    ):
        raise ValueError("authorized base model path mismatch")

    validated_data: dict[str, dict[str, Any]] = {}
    for name, config_field in (
        ("train", "train_file"),
        ("validation", "val_file"),
    ):
        entry = data_entries.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"authorization data.{name} is missing")
        expected_rows = entry.get("rows")
        if not isinstance(expected_rows, int) or expected_rows < 1:
            raise ValueError(f"authorization data.{name}.rows is invalid")
        path, digest = _validate_authorized_file(
            entry,
            repo_root=repo_root,
            label=name,
            expected_rows=expected_rows,
        )
        configured_path = _resolve_repo_path(
            config["data"][config_field],
            repo_root,
        )
        if path.resolve() != configured_path.resolve():
            raise ValueError(f"authorized {name} path mismatch")
        validated_data[name] = {
            "path": path,
            "rows": expected_rows,
            "sha256": digest,
        }

    for name in ("held_out", "reward_visible"):
        entry = data_entries.get(name)
        if entry is None:
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"authorization data.{name} is invalid")
        expected_rows = entry.get("rows")
        path, digest = _validate_authorized_file(
            entry,
            repo_root=repo_root,
            label=name,
            expected_rows=expected_rows,
        )
        validated_data[name] = {
            "path": path,
            "rows": expected_rows,
            "sha256": digest,
        }

    expected_python = _resolve_repo_path(environment["python"], repo_root)
    actual_python = _canonical_venv_executable(python_executable)
    authorized_python = _canonical_venv_executable(expected_python)
    if actual_python != authorized_python:
        raise ValueError(
            "formal SFT must run with the authorized .venv-train interpreter"
        )
    lock_entry = {
        "path": environment.get("lock_path"),
        "sha256": environment.get("lock_sha256"),
    }
    lock_path, lock_hash = _validate_authorized_file(
        lock_entry,
        repo_root=repo_root,
        label="environment lock",
    )
    expected_plan = authorization.get("locked_plan")
    if expected_plan is not None and _locked_plan(config) != expected_plan:
        raise ValueError("locked training plan differs from authorized config")

    harness = authorization.get("held_out_harness")
    harness_result = None
    if harness is not None:
        if not isinstance(harness, dict):
            raise ValueError("authorized held-out harness is invalid")
        harness_path, harness_hash = _validate_authorized_file(
            harness,
            repo_root=repo_root,
            label="held-out harness",
        )
        if harness.get("use_for_checkpoint_selection") is not False:
            raise ValueError("held-out cannot be used for checkpoint selection")
        if harness.get("evaluate_once_after_selection") is not True:
            raise ValueError("held-out must be evaluated once after selection")
        harness_result = {
            "path": str(harness_path),
            "sha256": harness_hash,
        }

    return {
        "config": config,
        "config_sha256": config_hash,
        "train_path": validated_data["train"]["path"],
        "train_rows": validated_data["train"]["rows"],
        "train_sha256": validated_data["train"]["sha256"],
        "validation_path": validated_data["validation"]["path"],
        "validation_rows": validated_data["validation"]["rows"],
        "validation_sha256": validated_data["validation"]["sha256"],
        "reserved_data": {
            name: {
                "path": str(item["path"]),
                "rows": item["rows"],
                "sha256": item["sha256"],
            }
            for name, item in validated_data.items()
            if name not in {"train", "validation"}
        },
        "python_executable": str(actual_python),
        "environment_lock_path": str(lock_path),
        "environment_lock_sha256": lock_hash,
        "local_files_only": True,
        "held_out_harness": harness_result,
    }


def validate_v100_autocast(config: dict[str, Any]) -> str:
    """Require the explicit Volta-safe autocast contract before probing."""
    train = config.get("train")
    if not isinstance(train, dict):
        raise ValueError("SFT config.train must be a mapping")
    if train.get("fp16") is not True:
        raise ValueError("V100 dry-run requires fp16=true")
    if train.get("bf16") is not False:
        raise ValueError("V100 dry-run requires bf16=false")
    if train.get("autocast_dtype") != "float16":
        raise ValueError("V100 dry-run requires autocast_dtype=float16")
    return "float16"


def _load_training_dependencies() -> dict[str, Any]:
    """Lazily import training dependencies so the service environment stays clean."""
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    peft = importlib.import_module("peft")
    importlib.import_module("bitsandbytes")
    return {
        "torch": torch,
        "AutoModelForCausalLM": transformers.AutoModelForCausalLM,
        "AutoTokenizer": transformers.AutoTokenizer,
        "BitsAndBytesConfig": transformers.BitsAndBytesConfig,
        "StoppingCriteriaList": transformers.StoppingCriteriaList,
        "get_linear_schedule_with_warmup": (
            transformers.get_linear_schedule_with_warmup
        ),
        "LoraConfig": peft.LoraConfig,
        "PeftModel": peft.PeftModel,
        "get_peft_model": peft.get_peft_model,
        "prepare_model_for_kbit_training": peft.prepare_model_for_kbit_training,
    }


def load_one_frozen_batch(
    config: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Read exactly one frozen trajectory without creating training outputs."""
    path = _resolve_repo_path(config["data"]["train_file"], repo_root)
    if not path.is_file():
        raise FileNotFoundError("frozen SFT train dataset was not found")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("frozen SFT row must be an object")
        if not isinstance(row.get("id"), str) or not row["id"].strip():
            raise ValueError("frozen SFT row id must be a non-empty string")
        text = row.get("qwen_chatml")
        spans = row.get("assistant_char_spans")
        if not isinstance(text, str) or not text:
            raise ValueError("frozen SFT row must contain qwen_chatml")
        if (
            not isinstance(spans, list)
            or not spans
            or any(
                not isinstance(span, list)
                or len(span) != 2
                or not all(isinstance(value, int) for value in span)
                or span[0] < 0
                or span[0] >= span[1]
                or span[1] > len(text)
                for span in spans
            )
        ):
            raise ValueError(
                "frozen SFT row must contain valid assistant_char_spans"
            )
        return row
    raise ValueError("frozen SFT train dataset is empty")


def _masked_batch(
    tokenizer: Any,
    row: dict[str, Any],
    *,
    max_seq_len: int,
) -> dict[str, Any]:
    """Tokenize one trajectory and supervise assistant messages only."""
    encoded = tokenizer(
        row["qwen_chatml"],
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_len,
        return_offsets_mapping=True,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    labels = encoded["input_ids"].clone()
    labels.fill_(-100)
    spans = row["assistant_char_spans"]
    supervised_tokens = 0
    for token_index, (start, end) in enumerate(offsets):
        if start == end:
            continue
        if any(start < span_end and end > span_start for span_start, span_end in spans):
            labels[0, token_index] = encoded["input_ids"][0, token_index]
            supervised_tokens += 1
    if supervised_tokens == 0:
        raise ValueError("tokenization produced no supervised assistant tokens")
    encoded["labels"] = labels
    return encoded


def _runtime_versions(dependencies: dict[str, Any]) -> dict[str, str]:
    torch = dependencies["torch"]
    return {
        "torch": str(getattr(torch, "__version__", "unknown")),
        "cuda": str(getattr(torch.version, "cuda", None)),
        "transformers": str(
            getattr(importlib.import_module("transformers"), "__version__", "unknown")
        ),
        "peft": str(getattr(importlib.import_module("peft"), "__version__", "unknown")),
        "bitsandbytes": str(
            getattr(importlib.import_module("bitsandbytes"), "__version__", "unknown")
        ),
    }


def cuda_memory_report(torch: Any) -> dict[str, float]:
    """Report current and process-peak CUDA memory in MiB."""
    return {
        "memory_allocated_mib": round(torch.cuda.memory_allocated() / 1024**2, 1),
        "memory_reserved_mib": round(torch.cuda.memory_reserved() / 1024**2, 1),
        "peak_memory_allocated_mib": round(
            torch.cuda.max_memory_allocated() / 1024**2,
            1,
        ),
        "peak_memory_reserved_mib": round(
            torch.cuda.max_memory_reserved() / 1024**2,
            1,
        ),
    }


def _write_log(log_path: Path, result: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _blocked(
    *,
    log_path: Path,
    blocker: str,
    detail: str,
) -> dict[str, Any]:
    result = {
        "status": "blocked",
        "mode": "dry_run",
        "checkpoint_saved": False,
        "blocker": blocker,
        "detail": detail,
    }
    _write_log(log_path, result)
    return result


def run_dry_run(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    log_path: Path = DEFAULT_LOG_PATH,
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    """Load local NF4 QLoRA, run one backward step, and never save a checkpoint."""
    try:
        config = load_training_config(config_path)
        model_path = _model_path(config, repo_root)
        load_contract = build_model_load_kwargs(config, model_path)
        autocast_dtype = validate_v100_autocast(config)
        row = load_one_frozen_batch(config, repo_root=repo_root)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        return _blocked(
            log_path=log_path,
            blocker="local_config_or_dataset_invalid",
            detail=f"{type(error).__name__}: {error}",
        )

    try:
        dependencies = _load_training_dependencies()
    except (ImportError, ModuleNotFoundError) as error:
        return _blocked(
            log_path=log_path,
            blocker="training_dependency_missing",
            detail=f"{type(error).__name__}: {error}",
        )

    torch = dependencies["torch"]
    if not torch.cuda.is_available():
        return _blocked(
            log_path=log_path,
            blocker="cuda_unavailable",
            detail="torch.cuda.is_available() returned false",
        )

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        quantization = dependencies["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = dependencies["AutoTokenizer"].from_pretrained(
            str(model_path),
            local_files_only=True,
            trust_remote_code=False,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = dependencies["AutoModelForCausalLM"].from_pretrained(
            str(model_path),
            quantization_config=quantization,
            device_map={"": 0},
            local_files_only=True,
            trust_remote_code=False,
        )
        model.config.use_cache = False
        model = dependencies["prepare_model_for_kbit_training"](
            model,
            use_gradient_checkpointing=True,
        )
        lora = config["lora"]
        model = dependencies["get_peft_model"](
            model,
            dependencies["LoraConfig"](
                r=lora["r"],
                lora_alpha=lora["alpha"],
                lora_dropout=lora["dropout"],
                target_modules=lora["target_modules"],
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        batch = _masked_batch(
            tokenizer,
            row,
            max_seq_len=int(config["data"]["max_seq_len"]),
        )
        batch = {name: value.to(model.device) for name, value in batch.items()}
        model.train()
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(**batch)
        loss = outputs.loss
        if loss is None or not bool(torch.isfinite(loss).item()):
            raise RuntimeError("dry-run loss is not finite")
        loss_value = float(loss.detach().cpu().item())
        loss.backward()
        torch.cuda.synchronize()
        step_elapsed_sec = round(time.perf_counter() - step_started, 4)
        result = {
            "status": "ready",
            "mode": "dry_run",
            "checkpoint_saved": False,
            "loss_finite": math.isfinite(loss_value),
            "loss": loss_value,
            "forward_backward_elapsed_sec": step_elapsed_sec,
            "autocast_dtype": autocast_dtype,
            **cuda_memory_report(torch),
            "runtime_versions": _runtime_versions(dependencies),
            "local_files_only": load_contract["local_files_only"],
            "quantization": load_contract["quantization"],
            "batch_record_id": row["id"],
        }
        _write_log(log_path, result)
        return result
    except Exception as error:
        return _blocked(
            log_path=log_path,
            blocker="qlora_dry_run_failed",
            detail=f"{type(error).__name__}: {error}",
        )
    finally:
        if "model" in locals():
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def require_finite_loss(value: float) -> float:
    """Reject NaN or infinity before backward."""
    if not math.isfinite(value):
        raise NonFiniteLossError(f"non-finite loss detected: {value}")
    return value


def select_best_checkpoint(
    checkpoints: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select only by the lowest finite active-validation loss."""
    if not checkpoints:
        raise ValueError("no epoch checkpoints were produced")
    for checkpoint in checkpoints:
        if set(("epoch", "eval_loss", "checkpoint")) - set(checkpoint):
            raise ValueError("checkpoint result is incomplete")
        require_finite_loss(float(checkpoint["eval_loss"]))
    return min(
        checkpoints,
        key=lambda item: (float(item["eval_loss"]), int(item["epoch"])),
    )


def _load_frozen_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            raise ValueError(f"{path}:{line_number}: blank line")
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: row must be an object")
        text = row.get("qwen_chatml")
        spans = row.get("assistant_char_spans")
        if (
            not isinstance(row.get("id"), str)
            or not isinstance(text, str)
            or not isinstance(spans, list)
            or not spans
        ):
            raise ValueError(f"{path}:{line_number}: invalid frozen row")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no frozen rows")
    return rows


def _pretokenize_rows(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    max_seq_len: int,
) -> list[dict[str, Any]]:
    tokenized: list[dict[str, Any]] = []
    for row in rows:
        batch = _masked_batch(
            tokenizer,
            row,
            max_seq_len=max_seq_len,
        )
        token_count = int(batch["input_ids"].shape[1])
        if token_count > max_seq_len:
            raise ValueError(f"{row['id']}: token count exceeds max_seq_len")
        tokenized.append(
            {
                "id": row["id"],
                "intent": row.get("intent"),
                "token_count": token_count,
                "batch": batch,
            }
        )
    return tokenized


def _append_jsonl(file_handle: Any, payload: dict[str, Any]) -> None:
    file_handle.write(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "\n"
    )
    file_handle.flush()


def _public_preflight(preflight: dict[str, Any]) -> dict[str, Any]:
    return {
        "config_sha256": preflight["config_sha256"],
        "train_rows": preflight["train_rows"],
        "train_sha256": preflight["train_sha256"],
        "validation_rows": preflight["validation_rows"],
        "validation_sha256": preflight["validation_sha256"],
        "reserved_data": preflight["reserved_data"],
        "python_executable": preflight["python_executable"],
        "environment_lock_path": preflight["environment_lock_path"],
        "environment_lock_sha256": preflight[
            "environment_lock_sha256"
        ],
        "local_files_only": preflight["local_files_only"],
        "held_out_harness": preflight["held_out_harness"],
    }


def _build_qlora_model(
    *,
    config: dict[str, Any],
    dependencies: dict[str, Any],
    model_path: Path,
) -> tuple[Any, Any]:
    torch = dependencies["torch"]
    quantization = dependencies["BitsAndBytesConfig"](
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = dependencies["AutoTokenizer"].from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = dependencies["AutoModelForCausalLM"].from_pretrained(
        str(model_path),
        quantization_config=quantization,
        device_map={"": 0},
        local_files_only=True,
        trust_remote_code=False,
    )
    model.config.use_cache = False
    model = dependencies["prepare_model_for_kbit_training"](
        model,
        use_gradient_checkpointing=True,
    )
    lora = config["lora"]
    model = dependencies["get_peft_model"](
        model,
        dependencies["LoraConfig"](
            r=lora["r"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    return tokenizer, model


def _trainable_parameter_counts(model: Any) -> dict[str, int | float]:
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "percentage": round(trainable / total * 100, 6),
    }


def _gradient_norm(model: Any, torch: Any) -> float:
    squared_norm = None
    for parameter in model.parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        norm = parameter.grad.detach().float().norm(2)
        term = norm * norm
        squared_norm = term if squared_norm is None else squared_norm + term
    if squared_norm is None:
        raise NonFiniteLossError("no trainable gradients were produced")
    value = float(torch.sqrt(squared_norm).detach().cpu().item())
    if not math.isfinite(value):
        raise NonFiniteLossError(f"non-finite gradient norm detected: {value}")
    return value


def _evaluate_validation(
    *,
    model: Any,
    rows: list[dict[str, Any]],
    torch: Any,
    epoch: int,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for row in rows:
            batch = {
                name: value.to(model.device)
                for name, value in row["batch"].items()
            }
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(**batch)
            if outputs.loss is None:
                raise NonFiniteLossError(
                    f"validation loss missing at epoch {epoch}"
                )
            loss_value = require_finite_loss(
                float(outputs.loss.detach().float().cpu().item())
            )
            losses.append(loss_value)
            del batch, outputs
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    model.train()
    return {
        "epoch": epoch,
        "eval_loss": require_finite_loss(sum(losses) / len(losses)),
        "examples": len(losses),
        "elapsed_sec": round(elapsed, 4),
        **cuda_memory_report(torch),
    }


def _save_epoch_checkpoint(
    *,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    output_dir: Path,
    epoch: int,
    eval_result: dict[str, Any],
    optimizer_step: int,
    torch: Any,
) -> Path:
    checkpoint = output_dir / f"checkpoint-epoch-{epoch}"
    checkpoint.mkdir(parents=False, exist_ok=False)
    model.save_pretrained(checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    torch.save(
        {
            "epoch": epoch,
            "optimizer_step": optimizer_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        },
        checkpoint / "training_state.pt",
    )
    (checkpoint / "checkpoint_metadata.json").write_text(
        json.dumps(
            {
                "epoch": epoch,
                "optimizer_step": optimizer_step,
                "selection_metric": "eval_loss",
                "eval_loss": eval_result["eval_loss"],
                "held_out_used_for_selection": False,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return checkpoint


def _copy_best_adapter(best_checkpoint: Path, output_dir: Path) -> Path:
    best_adapter = output_dir / "best_adapter"
    shutil.copytree(
        best_checkpoint,
        best_adapter,
        ignore=shutil.ignore_patterns("training_state.pt"),
    )
    return best_adapter


def run_training(
    *,
    config_path: Path = DEFAULT_CONFIG_PATH,
    authorization_path: Path = DEFAULT_AUTHORIZATION_PATH,
    report_path: Path = DEFAULT_TRAINING_REPORT_PATH,
    step_log_path: Path = DEFAULT_STEP_LOG_PATH,
    failure_log_path: Path = DEFAULT_FAILURE_LOG_PATH,
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    """Run the one authorized three-epoch QLoRA SFT campaign."""
    preflight = validate_training_authorization(
        config_path=config_path,
        authorization_path=authorization_path,
        repo_root=repo_root,
    )
    config = preflight["config"]
    validate_v100_autocast(config)
    output_dir = _resolve_repo_path(config["model"]["output_dir"], repo_root)
    for path in (output_dir, report_path, step_log_path, failure_log_path):
        if path.exists():
            raise FileExistsError(
                f"formal training refuses to overwrite existing output: {path}"
            )

    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    dependencies = _load_training_dependencies()
    torch = dependencies["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for formal SFT")
    if "V100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("authorized formal SFT requires the V100 target")

    train_config = config["train"]
    epochs = int(train_config["epochs"])
    accumulation = int(train_config["gradient_accumulation_steps"])
    if epochs != 3 or accumulation != 16:
        raise ValueError("runtime training plan differs from authorization")
    if train_config["per_device_train_batch_size"] != 1:
        raise ValueError("formal SFT requires micro batch size 1")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    step_log_path.parent.mkdir(parents=True, exist_ok=True)
    failure_log_path.parent.mkdir(parents=True, exist_ok=True)
    started_wall = time.time()
    started_perf = time.perf_counter()
    report: dict[str, Any] = {
        "status": "running",
        "mode": "formal_qlora_sft",
        "authorization": _public_preflight(preflight),
        "locked_plan": _locked_plan(config),
        "epochs": [],
        "best_checkpoint": None,
        "best_adapter": None,
        "oom_encountered": False,
        "non_finite_encountered": False,
        "started_unix": started_wall,
    }
    _write_log(report_path, report)

    model = None
    tokenizer = None
    global_micro_step = 0
    optimizer_step = 0
    peak_allocated = 0.0
    peak_reserved = 0.0
    failure_handle = None
    try:
        random.seed(int(train_config["seed"]))
        torch.manual_seed(int(train_config["seed"]))
        torch.cuda.manual_seed_all(int(train_config["seed"]))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model_path = _model_path(config, repo_root)
        tokenizer, model = _build_qlora_model(
            config=config,
            dependencies=dependencies,
            model_path=model_path,
        )
        model.train()
        train_rows = _load_frozen_rows(preflight["train_path"])
        validation_rows = _load_frozen_rows(
            preflight["validation_path"]
        )
        tokenization_started = time.perf_counter()
        tokenized_train = _pretokenize_rows(
            tokenizer,
            train_rows,
            max_seq_len=int(config["data"]["max_seq_len"]),
        )
        tokenized_validation = _pretokenize_rows(
            tokenizer,
            validation_rows,
            max_seq_len=int(config["data"]["max_seq_len"]),
        )
        report["tokenization_elapsed_sec"] = round(
            time.perf_counter() - tokenization_started,
            4,
        )
        report["parameter_counts"] = _trainable_parameter_counts(model)
        report["runtime_versions"] = _runtime_versions(dependencies)
        report["gpu"] = torch.cuda.get_device_name(0)
        report["train_rows"] = len(tokenized_train)
        report["validation_rows"] = len(tokenized_validation)

        trainable_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=float(train_config["learning_rate"]),
            weight_decay=0.0,
        )
        updates_per_epoch = math.ceil(len(tokenized_train) / accumulation)
        total_optimizer_steps = updates_per_epoch * epochs
        warmup_steps = math.ceil(
            total_optimizer_steps * float(train_config["warmup_ratio"])
        )
        scheduler = dependencies["get_linear_schedule_with_warmup"](
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_optimizer_steps,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        report["planned_optimizer_steps"] = total_optimizer_steps
        report["warmup_steps"] = warmup_steps
        _write_log(report_path, report)

        with step_log_path.open("x", encoding="utf-8", buffering=1) as step_log:
            optimizer.zero_grad(set_to_none=True)
            for epoch in range(1, epochs + 1):
                epoch_started = time.perf_counter()
                epoch_losses: list[float] = []
                generator = torch.Generator()
                generator.manual_seed(int(train_config["seed"]) + epoch)
                order = torch.randperm(
                    len(tokenized_train),
                    generator=generator,
                ).tolist()
                for position, row_index in enumerate(order):
                    row = tokenized_train[row_index]
                    group_start = (position // accumulation) * accumulation
                    group_size = min(
                        accumulation,
                        len(order) - group_start,
                    )
                    end_of_group = (
                        position - group_start + 1 == group_size
                    )
                    global_micro_step += 1
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    micro_started = time.perf_counter()
                    device_batch = {
                        name: value.to(model.device)
                        for name, value in row["batch"].items()
                    }
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                    ):
                        outputs = model(**device_batch)
                    if outputs.loss is None:
                        raise NonFiniteLossError(
                            f"loss missing at micro-step {global_micro_step}"
                        )
                    loss_value = require_finite_loss(
                        float(outputs.loss.detach().float().cpu().item())
                    )
                    scaler.scale(outputs.loss / group_size).backward()
                    gradient_norm = None
                    if end_of_group:
                        scaler.unscale_(optimizer)
                        gradient_norm = _gradient_norm(model, torch)
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        optimizer_step += 1
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - micro_started
                    memory = cuda_memory_report(torch)
                    peak_allocated = max(
                        peak_allocated,
                        memory["peak_memory_allocated_mib"],
                    )
                    peak_reserved = max(
                        peak_reserved,
                        memory["peak_memory_reserved_mib"],
                    )
                    epoch_losses.append(loss_value)
                    _append_jsonl(
                        step_log,
                        {
                            "status": "ok",
                            "epoch": epoch,
                            "epoch_micro_step": position + 1,
                            "global_micro_step": global_micro_step,
                            "optimizer_step": optimizer_step,
                            "optimizer_step_executed": end_of_group,
                            "gradient_accumulation_group_size": group_size,
                            "record_id": row["id"],
                            "intent": row["intent"],
                            "token_count": row["token_count"],
                            "loss": loss_value,
                            "loss_finite": True,
                            "gradient_norm": gradient_norm,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "elapsed_sec": round(elapsed, 4),
                            "oom": False,
                            **memory,
                        },
                    )
                    del device_batch, outputs

                validate_training_authorization(
                    config_path=config_path,
                    authorization_path=authorization_path,
                    repo_root=repo_root,
                )
                eval_result = _evaluate_validation(
                    model=model,
                    rows=tokenized_validation,
                    torch=torch,
                    epoch=epoch,
                )
                checkpoint = _save_epoch_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    output_dir=output_dir,
                    epoch=epoch,
                    eval_result=eval_result,
                    optimizer_step=optimizer_step,
                    torch=torch,
                )
                epoch_result = {
                    "epoch": epoch,
                    "train_loss_mean": require_finite_loss(
                        sum(epoch_losses) / len(epoch_losses)
                    ),
                    "micro_steps": len(epoch_losses),
                    "optimizer_step_end": optimizer_step,
                    "elapsed_sec": round(
                        time.perf_counter() - epoch_started,
                        4,
                    ),
                    "eval_loss": eval_result["eval_loss"],
                    "validation": eval_result,
                    "checkpoint": str(checkpoint),
                }
                report["epochs"].append(epoch_result)
                report["peak_memory_allocated_mib"] = peak_allocated
                report["peak_memory_reserved_mib"] = peak_reserved
                _write_log(report_path, report)

        if optimizer_step != total_optimizer_steps:
            raise RuntimeError(
                "optimizer step count drift: "
                f"expected={total_optimizer_steps} actual={optimizer_step}"
            )
        best = select_best_checkpoint(report["epochs"])
        best_checkpoint = Path(best["checkpoint"])
        best_adapter = _copy_best_adapter(best_checkpoint, output_dir)
        (output_dir / "best_checkpoint.json").write_text(
            json.dumps(
                {
                    "selection_metric": "active_validation_eval_loss",
                    "greater_is_better": False,
                    "held_out_used_for_selection": False,
                    "epoch": best["epoch"],
                    "eval_loss": best["eval_loss"],
                    "checkpoint": str(best_checkpoint),
                    "best_adapter": str(best_adapter),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        validate_training_authorization(
            config_path=config_path,
            authorization_path=authorization_path,
            repo_root=repo_root,
        )
        report.update(
            {
                "status": "completed",
                "best_checkpoint": str(best_checkpoint),
                "best_adapter": str(best_adapter),
                "best_epoch": best["epoch"],
                "best_eval_loss": best["eval_loss"],
                "global_micro_steps": global_micro_step,
                "optimizer_steps": optimizer_step,
                "peak_memory_allocated_mib": peak_allocated,
                "peak_memory_reserved_mib": peak_reserved,
                "elapsed_sec": round(
                    time.perf_counter() - started_perf,
                    4,
                ),
                "completed_unix": time.time(),
                "oom_encountered": False,
                "non_finite_encountered": False,
                "held_out_evaluated": False,
            }
        )
        _write_log(report_path, report)
        return report
    except BaseException as error:
        is_oom = isinstance(error, torch.cuda.OutOfMemoryError)
        is_non_finite = isinstance(error, NonFiniteLossError)
        failure = {
            "status": "aborted",
            "error_type": type(error).__name__,
            "detail": str(error),
            "global_micro_step": global_micro_step,
            "optimizer_step": optimizer_step,
            "oom": is_oom,
            "non_finite": is_non_finite,
            "elapsed_sec": round(time.perf_counter() - started_perf, 4),
            **cuda_memory_report(torch),
        }
        with failure_log_path.open("x", encoding="utf-8") as failure_handle:
            _append_jsonl(failure_handle, failure)
        report.update(
            {
                "status": "aborted",
                "oom_encountered": is_oom,
                "non_finite_encountered": is_non_finite,
                "failure": failure,
                "elapsed_sec": failure["elapsed_sec"],
                "completed_unix": time.time(),
            }
        )
        _write_log(report_path, report)
        return report
    finally:
        del failure_handle
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--train", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument(
        "--authorization",
        type=Path,
        default=DEFAULT_AUTHORIZATION_PATH,
    )
    args = parser.parse_args()
    if args.train:
        result = run_training(
            config_path=args.config,
            authorization_path=args.authorization,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result["status"] == "completed" else 1)
    if not args.dry_run:
        raise SystemExit(
            "select exactly one mode: --dry-run or --train"
        )
    result = run_dry_run(config_path=args.config, log_path=args.log_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "ready" else 1)


if __name__ == "__main__":
    main()
