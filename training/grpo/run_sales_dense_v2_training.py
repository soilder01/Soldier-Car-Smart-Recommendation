#!/usr/bin/env python3
"""Run the preregistered sales_dense_v2 continuation training protocol."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from training.grpo import formal_training as formal
from training.grpo.run_signal_probe import build_prompt


ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "data" / "model_training" / "grpo" / "formal_v4" / "restart_1"
RUN_DIR = BASE_DIR / "sales_dense_v2"
OUTPUT_DIR = ROOT / "checkpoints" / "grpo" / "formal_v4" / "restart_1" / "sales_dense_v2"
PROTOCOL_PATH = BASE_DIR / "sales_dense_v2_train_protocol.json"
TRAIN_MIX_PATH = BASE_DIR / "sales_dense_v2_train_mix.jsonl"
REWARD_FN_PATH = ROOT / "training" / "grpo" / "reward_fn.py"
MODEL_PATH = ROOT / "models" / "Qwen2.5-7B-Instruct"
CKPT100_PATH = (
    ROOT
    / "checkpoints"
    / "grpo"
    / "formal_v4"
    / "restart_1"
    / "sales_targeted"
    / "checkpoint-100"
)

STEP_LOG_PATH = RUN_DIR / "step_metrics.jsonl"
WINDOW_LOG_PATH = RUN_DIR / "health_windows.jsonl"
REPORT_PATH = RUN_DIR / "training_report.json"
START_RECEIPT_PATH = RUN_DIR / "start_receipt.json"
ABORT_PATH = RUN_DIR / "abort_report.json"

EXPECTED_REWARD_SHA = "325ad44feb83ec37c35babfed4bddb928cf400788e07735eb4631fc4af6962c8"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_sha_sidecar(artifact: Path) -> None:
    with Path(str(artifact) + ".sha256").open("x", encoding="ascii") as handle:
        handle.write(f"{sha256_file(artifact)}  {artifact.name}\n")


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def assert_frozen_hashes(protocol: dict[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    expected = protocol["expected_hashes"]
    for relative, digest in expected.items():
        actual = sha256_file(ROOT / relative)
        if actual != digest:
            raise RuntimeError(f"SHA mismatch: {relative}")
        observed[relative] = actual
    reward_sha = sha256_file(REWARD_FN_PATH)
    if reward_sha != EXPECTED_REWARD_SHA:
        raise RuntimeError("reward_fn.py SHA mismatch")
    return observed


def assert_preflight(protocol: dict[str, Any]) -> dict[str, str]:
    hashes = assert_frozen_hashes(protocol)
    if RUN_DIR.exists() or OUTPUT_DIR.exists():
        raise FileExistsError("sales_dense_v2 run/checkpoint directory already exists")
    if not (CKPT100_PATH / "adapter_model.safetensors").exists():
        raise RuntimeError("sales-targeted checkpoint-100 adapter is missing")
    if sha256_file(CKPT100_PATH / "adapter_model.safetensors") != protocol["initialization"]["adapter_sha256"]:
        raise RuntimeError("checkpoint-100 adapter SHA mismatch")
    return hashes


def build_training_cases(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    cases = load_jsonl(TRAIN_MIX_PATH)
    counts = Counter(case["intent"] for case in cases)
    expected = protocol["training_data_mixture"]
    if sha256_file(TRAIN_MIX_PATH) != expected["sha256"]:
        raise RuntimeError("train mix SHA drift")
    if len(cases) != expected["total_prompts_per_logical_epoch"]:
        raise RuntimeError("train mix total count drift")
    if counts != {"sales": expected["sales_dense_count"], "recommend": expected["recommend_rehearsal_count"]}:
        raise RuntimeError(f"train mix intent count drift: {dict(counts)}")
    return cases


def install_formal_path_overrides() -> None:
    formal.STEP_LOG_PATH = STEP_LOG_PATH
    formal.WINDOW_LOG_PATH = WINDOW_LOG_PATH
    formal.RUN_DIR = RUN_DIR
    formal.OUTPUT_DIR = OUTPUT_DIR


def window_summary_for_steps(steps: list[dict[str, Any]]) -> dict[str, Any]:
    summary = formal.window_summary(steps)
    by_intent: dict[str, dict[str, float]] = {}
    for intent in sorted({step["intent"] for step in steps}):
        intent_steps = [step for step in steps if step["intent"] == intent]
        by_intent[intent] = {
            "steps": len(intent_steps),
            "loss_mean": mean([step["logged_loss"] for step in intent_steps]),
            "reward_mean": mean([step["reward_mean"] for step in intent_steps]),
            "factual_precision_mean": mean([step["factual_precision_mean"] for step in intent_steps]),
            "required_coverage_mean": mean([step["required_coverage_mean"] for step in intent_steps]),
            "kl_mean": mean([step["kl_mean"] for step in intent_steps]),
            "intent_response_fail_rate": mean([step["intent_response_fail_rate"] for step in intent_steps]),
        }
    summary["by_intent"] = by_intent
    return summary


def make_callback_class(TrainerCallback: Any):
    class SalesDenseV2Callback(TrainerCallback):
        def __init__(self, audit: formal.TrainingAudit, config: dict[str, Any]) -> None:
            self.audit = audit
            self.config = config
            self.last_consumed_step = 0
            self.windows: list[dict[str, Any]] = []
            self.kl0_mean: float | None = None
            self.abort_reasons: list[str] = []

        def _abort(self, reason: str, control: Any, step: int) -> None:
            self.abort_reasons.append(reason)
            self.audit.abort_triggered = True
            self.audit.abort_reasons = list(self.abort_reasons)
            control.should_training_stop = True
            control.should_save = False
            if not ABORT_PATH.exists():
                write_json_exclusive(
                    ABORT_PATH,
                    {
                        "status": "aborted",
                        "step": step,
                        "reason": reason,
                        "resume_allowed": False,
                        "recent_windows": self.windows[-2:],
                    },
                )

        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **_: Any) -> Any:
            logs = logs or {}
            step = int(state.global_step)
            if step <= self.last_consumed_step:
                return control
            if self.audit.pending_reward is None or self.audit.pending_compute is None:
                return control
            record = self.audit.consume_step(step=step, logs=logs)
            self.last_consumed_step = step
            print(
                "train_progress "
                f"step={step} loss={record['logged_loss']:.6f} "
                f"reward={record['reward_mean']:.6f} kl={record['kl_mean']:.6f} "
                f"intent={record['intent']} step_time={record['step_time_sec']:.1f}s",
                flush=True,
            )
            if not record["all_finite"]:
                self._abort("non_finite", control, step)
                return control

            health_window = int(self.config["health_window_steps"])
            if step % health_window == 0:
                window = window_summary_for_steps(self.audit.steps[-health_window:])
                if self.kl0_mean is None:
                    self.kl0_mean = float(window["kl_mean"])
                    window["kl0_source"] = "first_health_window"
                    window["kl0_mean"] = self.kl0_mean
                else:
                    window["kl0_mean"] = self.kl0_mean
                    if self.kl0_mean > 0 and float(window["kl_mean"]) >= 10.0 * self.kl0_mean:
                        window["abort_candidate"] = "kl_explosive_divergence"
                        append_jsonl(WINDOW_LOG_PATH, window)
                        self.windows.append(window)
                        self._abort("kl_explosive_divergence", control, step)
                        return control
                append_jsonl(WINDOW_LOG_PATH, window)
                self.windows.append(window)
                print(
                    "health_window "
                    f"steps={window['start_step']}-{window['end_step']} "
                    f"loss={window['loss_mean']:.6f} reward={window['reward_mean']:.6f} "
                    f"kl={window['kl_mean']:.6f} kl_p95={window['kl_p95']:.6f} "
                    f"sales_reward={window['by_intent'].get('sales', {}).get('reward_mean', 0.0):.6f} "
                    f"recommend_reward={window['by_intent'].get('recommend', {}).get('reward_mean', 0.0):.6f}",
                    flush=True,
                )
            return control

    return SalesDenseV2Callback


def main() -> int:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    hashes_before = assert_preflight(protocol)
    config = protocol["training_hyperparameters"]
    if config["max_optimizer_steps"] != 200 or config["save_every_steps"] != 50:
        raise RuntimeError("sales_dense_v2 protocol step/save drift")
    install_formal_path_overrides()
    RUN_DIR.mkdir(parents=True, exist_ok=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(
        START_RECEIPT_PATH,
        {
            "status": "started",
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "hashes_before": hashes_before,
            "output_dir": str(OUTPUT_DIR.relative_to(ROOT)),
            "run_dir": str(RUN_DIR.relative_to(ROOT)),
            "held_out_40_body_read": False,
            "final_40_body_read": False,
        },
    )

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, prepare_model_for_kbit_training, set_peft_model_state_dict
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainerCallback
        from trl import GRPOConfig
        from trl.trainer.grpo_trainer import GRPOTrainer, is_conversational, maybe_apply_chat_template, unwrap_model_for_generation

        if Path(sys.prefix).resolve() != (ROOT / ".venv-grpo").resolve():
            raise RuntimeError("sales_dense_v2 training must run in .venv-grpo")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True, trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        cases = build_training_cases(protocol)
        rows: list[dict[str, Any]] = []
        for case in cases:
            rendered = tokenizer.apply_chat_template(build_prompt(case), tokenize=False, add_generation_prompt=True)
            prompt_length = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
            if prompt_length > config["max_prompt_length"]:
                raise RuntimeError(f"{case['id']}: prompt exceeds max_prompt_length")
            rows.append({"prompt": build_prompt(case), "case_id": case["id"]})
        dataset = Dataset.from_list(rows)
        cases_by_id = {case["id"]: case for case in cases}

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            quantization_config=quantization,
            device_map={"": 0},
            torch_dtype=torch.float16,
            local_files_only=True,
            trust_remote_code=False,
        )
        base_model.config.use_cache = False
        base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
        adapter_config = json.loads((CKPT100_PATH / "adapter_config.json").read_text(encoding="utf-8"))
        peft_config = LoraConfig(
            r=adapter_config["r"],
            lora_alpha=adapter_config["lora_alpha"],
            lora_dropout=adapter_config["lora_dropout"],
            target_modules=adapter_config["target_modules"],
            bias=adapter_config["bias"],
            task_type=adapter_config["task_type"],
            inference_mode=False,
        )
        baseline = {"kl0_mean": None, "fake_reward_gate": {"reward_mean": 0.0, "factual_precision_mean": 0.0}}
        audit = formal.TrainingAudit(cases=cases_by_id, tokenizer=tokenizer, authorization=protocol, baseline=baseline)
        audit.reward_function.__func__.__name__ = "grounding_reward"
        args = GRPOConfig(
            output_dir=str(OUTPUT_DIR),
            overwrite_output_dir=False,
            do_train=True,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            max_grad_norm=config["max_grad_norm"],
            max_steps=config["max_optimizer_steps"],
            lr_scheduler_type=config["lr_scheduler"],
            warmup_steps=config["warmup_steps"],
            logging_strategy="steps",
            logging_steps=1,
            logging_first_step=True,
            save_strategy="steps",
            save_steps=config["save_every_steps"],
            save_total_limit=4,
            report_to=[],
            disable_tqdm=True,
            remove_unused_columns=False,
            fp16=True,
            bf16=False,
            tf32=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_prompt_length=config["max_prompt_length"],
            max_completion_length=config["max_completion_length"],
            num_generations=config["num_generations"],
            temperature=config["temperature"],
            beta=config["beta"],
            use_vllm=False,
            optim=config["optimizer"],
            seed=config["seed"],
            dataloader_num_workers=0,
        )
        AuditedTrainer = formal.make_audited_trainer_class(
            torch=torch,
            GRPOTrainer=GRPOTrainer,
            Trainer=Trainer,
            maybe_apply_chat_template=maybe_apply_chat_template,
            unwrap_model_for_generation=unwrap_model_for_generation,
            is_conversational=is_conversational,
        )
        Callback = make_callback_class(TrainerCallback)
        callback = Callback(audit=audit, config=config)
        trainer = AuditedTrainer(
            model=base_model,
            reward_funcs=audit.reward_function,
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=[callback],
            peft_config=peft_config,
            audit=audit,
        )
        if trainer.ref_model is not None:
            raise RuntimeError("separate reference model is forbidden")
        trainer.generation_config.top_p = config["top_p"]
        adapter_state = load_file(str(CKPT100_PATH / "adapter_model.safetensors"), device="cpu")
        load_result = set_peft_model_state_dict(trainer.model, adapter_state, adapter_name="default")
        if list(getattr(load_result, "unexpected_keys", ())):
            raise RuntimeError("unexpected checkpoint-100 adapter keys")

        started = time.time()
        train_output = trainer.train()
        completed_step = int(trainer.state.global_step)
        checkpoint_dirs = sorted(OUTPUT_DIR.glob("checkpoint-*"), key=lambda path: int(path.name.rsplit("-", 1)[-1]))
        expected_checkpoints = [OUTPUT_DIR / f"checkpoint-{step}" for step in protocol["pre_registered_gate"]["candidate_checkpoint_steps"]]
        missing = [str(path.relative_to(ROOT)) for path in expected_checkpoints if not path.is_dir()]
        hashes_after = assert_frozen_hashes(protocol)
        report = {
            "status": "aborted" if audit.abort_triggered else "completed",
            "started_unix_sec": started,
            "completed_unix_sec": time.time(),
            "completed_optimizer_steps": completed_step,
            "planned_optimizer_steps": config["max_optimizer_steps"],
            "train_loss": float(train_output.training_loss),
            "abort_reasons": audit.abort_reasons,
            "resume_allowed": not audit.abort_triggered,
            "checkpoint_dirs": [str(path.relative_to(ROOT)) for path in checkpoint_dirs],
            "missing_expected_checkpoints": missing,
            "health_windows": callback.windows,
            "hashes_before": hashes_before,
            "hashes_after": hashes_after,
            "frozen_hashes_zero_change": hashes_before == hashes_after,
            "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "reward_fn_sha256": sha256_file(REWARD_FN_PATH),
            "held_out_40_body_read": False,
            "final_40_body_read": False,
            "postflight_directory_exclusive_check_reused": False,
        }
        write_json_exclusive(REPORT_PATH, report)
        write_sha_sidecar(REPORT_PATH)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "completed_optimizer_steps": completed_step,
                    "train_loss": report["train_loss"],
                    "checkpoint_dirs": report["checkpoint_dirs"],
                    "missing_expected_checkpoints": missing,
                    "training_report_sha256": sha256_file(REPORT_PATH),
                    "cuda_peak_reserved_mib": report["cuda_peak_reserved_mib"],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
        return 0 if not audit.abort_triggered else 2
    except BaseException as exc:
        if not ABORT_PATH.exists():
            write_json_exclusive(
                ABORT_PATH,
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "resume_allowed": False,
                },
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
