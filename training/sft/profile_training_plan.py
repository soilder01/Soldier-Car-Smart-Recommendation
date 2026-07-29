"""Profile frozen SFT token lengths and transient V100 QLoRA micro-steps."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from training.sft import train_qlora_sft as sft


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOKEN_REPORT_PATH = (
    ROOT / "data" / "model_training" / "sft_token_length_report.json"
)
DEFAULT_PROFILE_REPORT_PATH = (
    ROOT / "data" / "model_training" / "sft_step_profile_report.json"
)
OVERHEAD_RATIO = 0.15


def _nearest_rank(values: list[int], percentile: int) -> int:
    if not values:
        raise ValueError("cannot calculate a percentile of no values")
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be between 1 and 100")
    ordered = sorted(values)
    index = max(0, math.ceil(percentile / 100 * len(ordered)) - 1)
    return ordered[index]


def summarize_token_lengths(lengths: list[int]) -> dict[str, int]:
    """Return deterministic nearest-rank length statistics."""
    if not lengths or any(
        not isinstance(length, int) or isinstance(length, bool) or length < 1
        for length in lengths
    ):
        raise ValueError("token lengths must be non-empty positive integers")
    return {
        "count": len(lengths),
        "p50": _nearest_rank(lengths, 50),
        "p95": _nearest_rank(lengths, 95),
        "p99": _nearest_rank(lengths, 99),
        "max": max(lengths),
    }


def validate_v100_training_plan(config: dict[str, Any]) -> dict[str, Any]:
    """Reject precision or batch settings incompatible with the approved plan."""
    data = config.get("data")
    train = config.get("train")
    if not isinstance(data, dict) or not isinstance(train, dict):
        raise ValueError("SFT config requires data and train mappings")
    if train.get("fp16") is not True:
        raise ValueError("V100 plan requires fp16=true")
    if train.get("bf16") is not False:
        raise ValueError("V100 plan requires bf16=false")
    if train.get("autocast_dtype") != "float16":
        raise ValueError("V100 plan requires autocast_dtype=float16")

    max_seq_len = data.get("max_seq_len")
    micro_batch = train.get("per_device_train_batch_size")
    accumulation = train.get("gradient_accumulation_steps")
    epochs = train.get("epochs")
    learning_rate = train.get("learning_rate")
    warmup_ratio = train.get("warmup_ratio")
    if (
        not isinstance(max_seq_len, int)
        or isinstance(max_seq_len, bool)
        or max_seq_len < 1
    ):
        raise ValueError("data.max_seq_len must be a positive integer")
    if (
        not isinstance(micro_batch, int)
        or isinstance(micro_batch, bool)
        or micro_batch < 1
    ):
        raise ValueError("train.per_device_train_batch_size must be positive")
    if (
        not isinstance(accumulation, int)
        or isinstance(accumulation, bool)
        or accumulation < 1
    ):
        raise ValueError("train.gradient_accumulation_steps must be positive")
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1:
        raise ValueError("train.epochs must be a positive integer")
    if (
        not isinstance(learning_rate, (int, float))
        or isinstance(learning_rate, bool)
        or learning_rate <= 0
    ):
        raise ValueError("train.learning_rate must be positive")
    if (
        not isinstance(warmup_ratio, (int, float))
        or isinstance(warmup_ratio, bool)
        or not 0 <= warmup_ratio < 1
    ):
        raise ValueError("train.warmup_ratio must be in [0, 1)")
    return {
        "max_seq_len": max_seq_len,
        "micro_batch_size": micro_batch,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": micro_batch * accumulation,
        "epochs": epochs,
        "learning_rate": float(learning_rate),
        "warmup_ratio": float(warmup_ratio),
        "autocast_dtype": "float16",
    }


def estimate_wall_clock(
    *,
    train_examples: int,
    epochs: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    mean_micro_step_sec: float,
    overhead_ratio: float,
) -> dict[str, Any]:
    """Estimate training wall-clock from measured micro-step latency."""
    if min(train_examples, epochs, micro_batch_size, gradient_accumulation_steps) < 1:
        raise ValueError("training counts must be positive")
    if not math.isfinite(mean_micro_step_sec) or mean_micro_step_sec <= 0:
        raise ValueError("mean_micro_step_sec must be finite and positive")
    if not 0 <= overhead_ratio < 1:
        raise ValueError("overhead_ratio must be in [0, 1)")
    total_micro_steps = math.ceil(train_examples / micro_batch_size) * epochs
    optimizer_steps = math.ceil(total_micro_steps / gradient_accumulation_steps)
    estimated_compute_sec = round(total_micro_steps * mean_micro_step_sec, 2)
    estimated_wall_clock_sec = round(
        estimated_compute_sec * (1 + overhead_ratio),
        2,
    )
    return {
        "train_examples": train_examples,
        "epochs": epochs,
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "total_micro_steps": total_micro_steps,
        "optimizer_steps": optimizer_steps,
        "measured_micro_step_sec": round(mean_micro_step_sec, 4),
        "estimated_compute_sec": estimated_compute_sec,
        "overhead_ratio": overhead_ratio,
        "estimated_wall_clock_sec": estimated_wall_clock_sec,
        "estimated_wall_clock_hours": round(estimated_wall_clock_sec / 3600, 2),
    }


def _load_train_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    path = sft._resolve_repo_path(config["data"]["train_file"], ROOT)
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise ValueError("frozen SFT training data is empty")
    return rows


def _token_length_rows(tokenizer: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        token_ids = tokenizer(
            row["qwen_chatml"],
            add_special_tokens=False,
        )["input_ids"]
        result.append(
            {
                "id": row["id"],
                "intent": row["intent"],
                "token_length": len(token_ids),
                "row": row,
            }
        )
    return result


def _select_closest(
    rows: list[dict[str, Any]],
    target_length: int,
) -> dict[str, Any]:
    return min(
        rows,
        key=lambda row: (
            abs(row["token_length"] - target_length),
            row["token_length"],
            row["id"],
        ),
    )


def _load_probe_model(
    config: dict[str, Any],
    *,
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
    model.train()
    return tokenizer, model


def _run_micro_step(
    *,
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    max_seq_len: int,
    torch: Any,
) -> tuple[float, float, int]:
    batch = sft._masked_batch(
        tokenizer,
        row,
        max_seq_len=max_seq_len,
    )
    batch = {name: value.to(model.device) for name, value in batch.items()}
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(**batch)
    loss = outputs.loss
    if loss is None or not bool(torch.isfinite(loss).item()):
        raise RuntimeError("profile micro-step loss is not finite")
    loss_value = float(loss.detach().cpu().item())
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return elapsed, loss_value, int(batch["input_ids"].shape[1])


def run_profile(
    *,
    config_path: Path = sft.DEFAULT_CONFIG_PATH,
    token_report_path: Path = DEFAULT_TOKEN_REPORT_PATH,
    profile_report_path: Path = DEFAULT_PROFILE_REPORT_PATH,
    measured_steps: int = 4,
    warmup_steps: int = 1,
) -> dict[str, Any]:
    """Measure token distribution then transient capacity and steady micro-steps."""
    if measured_steps < 3 or measured_steps > 5:
        raise ValueError("measured_steps must be between 3 and 5")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    config = sft.load_training_config(config_path)
    plan = validate_v100_training_plan(config)
    dependencies = sft._load_training_dependencies()
    torch = dependencies["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for training plan profiling")
    model_path = sft._model_path(config, ROOT)
    tokenizer = dependencies["AutoTokenizer"].from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    rows = _load_train_rows(config)
    length_rows = _token_length_rows(tokenizer, rows)
    length_stats = summarize_token_lengths(
        [row["token_length"] for row in length_rows],
    )
    max_seq_len = plan["max_seq_len"]
    longest = max(
        length_rows,
        key=lambda row: (row["token_length"], row["id"]),
    )
    representative = _select_closest(length_rows, length_stats["p95"])
    token_report = {
        "status": "completed",
        "dataset_rows": len(rows),
        "token_length_distribution": length_stats,
        "selected_max_seq_len": max_seq_len,
        "rows_exceeding_selected_max_seq_len": sum(
            row["token_length"] > max_seq_len for row in length_rows
        ),
        "truncation_rate_at_selected_max_seq_len": round(
            sum(row["token_length"] > max_seq_len for row in length_rows)
            / len(length_rows)
            * 100,
            1,
        ),
        "longest_record": {
            "id": longest["id"],
            "intent": longest["intent"],
            "raw_token_length": longest["token_length"],
        },
        "p95_representative_record": {
            "id": representative["id"],
            "intent": representative["intent"],
            "raw_token_length": representative["token_length"],
        },
    }
    sft._write_log(token_report_path, token_report)

    del tokenizer
    torch.cuda.empty_cache()
    tokenizer, model = _load_probe_model(
        config,
        dependencies=dependencies,
        model_path=model_path,
    )
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        capacity_elapsed, capacity_loss, capacity_tokens = _run_micro_step(
            model=model,
            tokenizer=tokenizer,
            row=longest["row"],
            max_seq_len=max_seq_len,
            torch=torch,
        )
        capacity_memory = sft.cuda_memory_report(torch)
        model.zero_grad(set_to_none=True)

        for _ in range(warmup_steps):
            _run_micro_step(
                model=model,
                tokenizer=tokenizer,
                row=representative["row"],
                max_seq_len=max_seq_len,
                torch=torch,
            )
        torch.cuda.reset_peak_memory_stats()
        elapsed_steps: list[float] = []
        losses: list[float] = []
        effective_tokens: list[int] = []
        for _ in range(measured_steps):
            elapsed, loss, token_count = _run_micro_step(
                model=model,
                tokenizer=tokenizer,
                row=representative["row"],
                max_seq_len=max_seq_len,
                torch=torch,
            )
            elapsed_steps.append(elapsed)
            losses.append(loss)
            effective_tokens.append(token_count)
        steady_memory = sft.cuda_memory_report(torch)
        mean_micro_step_sec = sum(elapsed_steps) / len(elapsed_steps)
        estimate = estimate_wall_clock(
            train_examples=len(rows),
            epochs=plan["epochs"],
            micro_batch_size=plan["micro_batch_size"],
            gradient_accumulation_steps=plan["gradient_accumulation_steps"],
            mean_micro_step_sec=mean_micro_step_sec,
            overhead_ratio=OVERHEAD_RATIO,
        )
        profile_report = {
            "status": "completed",
            "mode": "transient_qlora_profile_only",
            "checkpoint_saved": False,
            "optimizer_step_executed": False,
            "plan": plan,
            "capacity_check_longest_record": {
                "id": longest["id"],
                "raw_token_length": longest["token_length"],
                "effective_token_length": capacity_tokens,
                "loss": capacity_loss,
                "loss_finite": math.isfinite(capacity_loss),
                "forward_backward_elapsed_sec": round(capacity_elapsed, 4),
                **capacity_memory,
            },
            "steady_p95_profile": {
                "record_id": representative["id"],
                "raw_token_length": representative["token_length"],
                "effective_token_lengths": effective_tokens,
                "warmup_steps": warmup_steps,
                "measured_steps": measured_steps,
                "micro_step_seconds": [
                    round(value, 4) for value in elapsed_steps
                ],
                "mean_micro_step_sec": round(mean_micro_step_sec, 4),
                "losses": losses,
                "all_losses_finite": all(math.isfinite(loss) for loss in losses),
                **steady_memory,
            },
            "wall_clock_estimate": estimate,
        }
        sft._write_log(profile_report_path, profile_report)
        return {
            "token_report": token_report,
            "profile_report": profile_report,
        }
    finally:
        del model
        del tokenizer
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile frozen SFT sequence lengths and transient QLoRA steps.",
    )
    parser.add_argument("--config", type=Path, default=sft.DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--token-report",
        type=Path,
        default=DEFAULT_TOKEN_REPORT_PATH,
    )
    parser.add_argument(
        "--profile-report",
        type=Path,
        default=DEFAULT_PROFILE_REPORT_PATH,
    )
    parser.add_argument("--measured-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=1)
    args = parser.parse_args()
    result = run_profile(
        config_path=args.config,
        token_report_path=args.token_report,
        profile_report_path=args.profile_report,
        measured_steps=args.measured_steps,
        warmup_steps=args.warmup_steps,
    )
    print(
        json.dumps(
            {
                "token_length_distribution": result["token_report"][
                    "token_length_distribution"
                ],
                "selected_max_seq_len": result["token_report"][
                    "selected_max_seq_len"
                ],
                "wall_clock_estimate": result["profile_report"][
                    "wall_clock_estimate"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
