#!/usr/bin/env python3
"""Run one no-checkpoint GRPO optimizer step in the isolated GRPO venv."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Any
from unittest.mock import patch

from training.grpo.reward_fn import (
    EvidenceClaim,
    IntentResponseSpec,
    score_grounded_answer,
)


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "Qwen2.5-7B-Instruct"
ADAPTER_PATH = ROOT / "checkpoints" / "sft" / "best_adapter"
TRAIN_PATH = (
    ROOT / "data" / "model_training" / "grpo" / "reward_train_16.jsonl"
)
CASE_PATH = (
    ROOT
    / "data"
    / "model_training"
    / "grpo"
    / "grpo_one_step_smoke_case.json"
)
CASE_SHA_PATH = CASE_PATH.with_suffix(".sha256")
REPORT_PATH = (
    ROOT
    / "data"
    / "model_training"
    / "grpo"
    / "grpo_one_step_smoke_report.json"
)
LOCK_PATH = ROOT / "training" / "grpo" / "requirements-cu118-design.lock.txt"
VEHICLE_DATABASE_PATH = ROOT / "data" / "vehicles" / "vehicle_database.csv"
DEV_AUTHORIZATION_PATH = (
    ROOT
    / "data"
    / "model_training"
    / "grpo"
    / "grpo_dev_authorization_manifest.json"
)

EXPECTED_SHA256 = {
    TRAIN_PATH: "0390c0ee32156c02c84b08d0bc96191b0a7040a57fcaab969112f998b4539cc7",
    CASE_PATH: "341141379dccf0a43b88afae3ba0fe86983d250ca2dee8a084758ddbdec62351",
    LOCK_PATH: "e09e0dac47a8d15a88716fe2fc58e985a776b06832ea66d8051a2440545ce774",
    VEHICLE_DATABASE_PATH: "f2b2b8070571e9a02513ec942b44f79bd7a1c87b083b94c434a2e2a3be6af5f7",
    ADAPTER_PATH
    / "adapter_model.safetensors": "c55ec155f08e21125d9cf0383c9a6ee066f61d31d3a0e9285eb72a8fc4d17800",
    ADAPTER_PATH
    / "adapter_config.json": "ec3f1c5de4da14fa695ccc65ba1f27083c34df32e0c71722c0c78166441bb8b9",
    DEV_AUTHORIZATION_PATH: "30631cad7a8733739eafef6d285e4e0b3f007e38c1e898c064ebbf0432315bff",
}
PINNED_VERSIONS = {
    "torch": "2.3.1+cu118",
    "transformers": "4.46.3",
    "tokenizers": "0.20.3",
    "accelerate": "0.34.2",
    "peft": "0.12.0",
    "trl": "0.14.0",
    "datasets": "2.21.0",
    "bitsandbytes": "0.43.3",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_frozen_inputs() -> dict[str, str]:
    observed: dict[str, str] = {}
    for path, expected in EXPECTED_SHA256.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen SHA drift: {path.relative_to(ROOT)}")
        observed[str(path.relative_to(ROOT))] = actual
    recorded_sha, recorded_name = CASE_SHA_PATH.read_text(
        encoding="ascii"
    ).strip().split(maxsplit=1)
    if recorded_sha != EXPECTED_SHA256[CASE_PATH]:
        raise ValueError("smoke case companion SHA drift")
    if recorded_name != CASE_PATH.name:
        raise ValueError("smoke case companion filename drift")
    return observed


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_case() -> tuple[
    dict[str, Any],
    IntentResponseSpec,
    tuple[EvidenceClaim, ...],
]:
    case = json.loads(CASE_PATH.read_text(encoding="utf-8"))
    if case["scope"] != {
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
    }:
        raise ValueError("smoke scope drift")

    prompt_case = case["prompt_case"]
    train_rows = _load_jsonl(TRAIN_PATH)
    matching_rows = [
        row for row in train_rows if row["id"] == prompt_case["id"]
    ]
    if len(matching_rows) != 1:
        raise ValueError("smoke prompt is not exactly one train_16 row")
    for field in ("id", "query", "intent"):
        if matching_rows[0][field] != prompt_case[field]:
            raise ValueError(f"smoke train prompt {field} drift")

    rows_by_name: dict[str, dict[str, str]] = {}
    with VEHICLE_DATABASE_PATH.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows_by_name[f"{row['brand']} {row['model']}"] = row
    expected_values = {
        "小米 SU7": "19",
        "特斯拉 Model 3": "27",
    }
    for entity, expected in expected_values.items():
        if rows_by_name[entity]["fast_charge_minutes"] != expected:
            raise ValueError(f"catalog fast-charge drift for {entity}")

    spec_data = case["intent_response_spec"]
    spec = IntentResponseSpec(
        prompt_id=prompt_case["id"],
        intent=prompt_case["intent"],
        target_entities=tuple(prompt_case["target_entities"]),
        query_anchor_tokens=tuple(spec_data["query_anchor_tokens"]),
        query_attribute_anchors=tuple(
            spec_data["query_attribute_anchors"]
        ),
        minimum_supported_claims=spec_data["minimum_supported_claims"],
    )
    claims = tuple(
        EvidenceClaim(
            canonical_entity=item["canonical_entity"],
            canonical_attribute=item["canonical_attribute"],
            canonical_value=item["canonical_value"],
            source_tool=item["source_tool"],
            source_locator=item["source_locator"],
            entity_aliases=tuple(item["entity_aliases"]),
            attribute_aliases=tuple(item["attribute_aliases"]),
            anchor_tokens=tuple(item["anchor_tokens"]),
        )
        for item in case["evidence_claims"]
    )
    return case, spec, claims


def _write_report(report: dict[str, Any]) -> None:
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


class GpuMemoryMonitor:
    """Sample global GPU memory; the GPU must be idle before the smoke."""

    def __init__(self) -> None:
        self.phase = "preflight"
        self.samples: list[dict[str, Any]] = []
        self.peaks: dict[str, int] = defaultdict(int)
        self.total_mib = 0
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self.phase = phase

    def _run(self) -> None:
        while not self._stop.is_set():
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode == 0 and completed.stdout.strip():
                total, used = (
                    int(value.strip())
                    for value in completed.stdout.splitlines()[0].split(",")
                )
                with self._lock:
                    phase = self.phase
                    self.total_mib = total
                    self.peaks[phase] = max(self.peaks[phase], used)
                    self.samples.append(
                        {
                            "phase": phase,
                            "used_mib": used,
                            "monotonic_sec": round(time.monotonic(), 3),
                        }
                    )
            self._stop.wait(0.05)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_mib": self.total_mib,
                "sample_count": len(self.samples),
                "phase_peak_used_mib": dict(sorted(self.peaks.items())),
                "global_peak_used_mib": max(self.peaks.values(), default=0),
            }


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if (
        isinstance(completion, list)
        and completion
        and isinstance(completion[-1], dict)
        and isinstance(completion[-1].get("content"), str)
    ):
        return completion[-1]["content"]
    return ""


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _adapter_probe(model: Any, tokenizer: Any, torch: Any) -> float:
    inputs = tokenizer(
        "小米SU7与特斯拉Model 3补能对比",
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = inputs["input_ids"].to(model.device)
    model.eval()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
    ):
        adapter_logits = model(
            input_ids,
            num_logits_to_keep=1,
        ).logits[:, -1, :].float()
        with model.disable_adapter():
            reference_logits = model(
                input_ids,
                num_logits_to_keep=1,
            ).logits[:, -1, :].float()
    difference = float(
        (adapter_logits - reference_logits).abs().max().cpu().item()
    )
    model.train()
    if not math.isfinite(difference) or difference <= 0:
        raise RuntimeError("SFT adapter-on logits equal adapter-off reference")
    return difference


def main() -> int:
    if REPORT_PATH.exists():
        raise FileExistsError(
            f"refusing to rerun completed/attempted smoke: {REPORT_PATH}"
        )
    if Path(sys.prefix).resolve() != (ROOT / ".venv-grpo").resolve():
        raise RuntimeError("smoke must run from the isolated .venv-grpo")

    report: dict[str, Any] = {
        "status": "running",
        "mode": "one_step_smoke_not_formal_training",
        "checkpoint_saved": False,
        "formal_training_entry_enabled": False,
        "started_unix_sec": time.time(),
    }
    monitor = GpuMemoryMonitor()
    monitor.start()
    temp_output: tempfile.TemporaryDirectory[str] | None = None
    torch = None
    try:
        report["frozen_inputs"] = _require_frozen_inputs()
        case, spec, evidence_claims = _load_case()
        actual_versions = {
            package: version(package) for package in PINNED_VERSIONS
        }
        if actual_versions != PINNED_VERSIONS:
            raise RuntimeError(f"runtime version drift: {actual_versions}")
        freeze = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
        )
        report["runtime"] = {
            "versions": actual_versions,
            "pip_freeze_sha256": hashlib.sha256(
                freeze.encode("utf-8")
            ).hexdigest(),
            "python": sys.version.split()[0],
        }

        import torch as torch_module
        from datasets import Dataset
        from peft import (
            LoraConfig,
            prepare_model_for_kbit_training,
            set_peft_model_state_dict,
        )
        from safetensors.torch import load_file
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainerCallback,
        )
        from trl import GRPOConfig, GRPOTrainer

        torch = torch_module
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        device_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        if device_name != "Tesla V100-SXM2-32GB" or capability != (7, 0):
            raise RuntimeError(
                f"unexpected target GPU: {device_name} {capability}"
            )
        monitor.set_phase("model_load")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_PATH),
            local_files_only=True,
            trust_remote_code=False,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            quantization_config=quantization,
            device_map={"": 0},
            torch_dtype=torch.float16,
            local_files_only=True,
            trust_remote_code=False,
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

        adapter_config = json.loads(
            (ADAPTER_PATH / "adapter_config.json").read_text(
                encoding="utf-8"
            )
        )
        peft_config = LoraConfig(
            r=adapter_config["r"],
            lora_alpha=adapter_config["lora_alpha"],
            lora_dropout=adapter_config["lora_dropout"],
            target_modules=adapter_config["target_modules"],
            bias=adapter_config["bias"],
            task_type=adapter_config["task_type"],
            inference_mode=False,
        )

        evidence_lines = "\n".join(
            (
                f"- {claim.canonical_entity} | "
                f"{claim.canonical_attribute} | "
                f"{claim.canonical_value} | "
                f"source={claim.source_locator}"
            )
            for claim in evidence_claims
        )
        prompt = [
            {
                "role": "system",
                "content": (
                    "你是只基于已执行工具 observation 作答的车型对比助手。"
                    "不得使用模型记忆补充数字。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户问题：{case['prompt_case']['query']}\n"
                    "冻结工具证据：\n"
                    f"{evidence_lines}\n"
                    f"{case['generation_instruction']}"
                ),
            },
        ]
        rendered_prompt = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_tokens = len(
            tokenizer(
                rendered_prompt,
                add_special_tokens=False,
            )["input_ids"]
        )
        if prompt_tokens > 512:
            raise RuntimeError("smoke prompt exceeds frozen max_prompt_length")

        reward_details: list[dict[str, Any]] = []

        def grounding_reward(
            prompts: list[Any],
            completions: list[Any],
            **_: Any,
        ) -> list[float]:
            if len(prompts) != 8 or len(completions) != 8:
                raise RuntimeError("smoke reward did not receive one group of 8")
            output: list[float] = []
            for completion in completions:
                text = _completion_text(completion)
                score = score_grounded_answer(
                    answer=text,
                    spec=spec,
                    evidence_claims=evidence_claims,
                )
                detail = {
                    "completion_sha256": hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest(),
                    "completion_chars": len(text),
                    "gate": score.gate,
                    "total": score.total,
                    "factual_precision": score.factual_precision,
                    "required_coverage": score.required_coverage,
                    "source_integrity": score.source_integrity,
                    "concision": score.concision,
                    "matched_claim_count": score.matched_claim_count,
                }
                if not all(
                    _finite(detail[name])
                    for name in (
                        "total",
                        "factual_precision",
                        "required_coverage",
                        "source_integrity",
                        "concision",
                    )
                ):
                    raise RuntimeError("non-finite reward component")
                reward_details.append(detail)
                output.append(float(score.total))
            return output

        grounding_reward.__name__ = "grounding_reward"

        class SmokeCallback(TrainerCallback):
            def __init__(self) -> None:
                self.optimizer_steps = 0
                self.logs: list[dict[str, Any]] = []

            def on_optimizer_step(
                self,
                args: Any,
                state: Any,
                control: Any,
                **kwargs: Any,
            ) -> None:
                self.optimizer_steps += 1

            def on_log(
                self,
                args: Any,
                state: Any,
                control: Any,
                logs: dict[str, Any] | None = None,
                **kwargs: Any,
            ) -> None:
                if logs:
                    self.logs.append(dict(logs))

        callback = SmokeCallback()
        temp_output = tempfile.TemporaryDirectory(
            prefix="agentic-grpo-one-step-smoke-"
        )
        args = GRPOConfig(
            output_dir=temp_output.name,
            overwrite_output_dir=False,
            do_train=True,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-6,
            max_steps=1,
            lr_scheduler_type="constant",
            warmup_steps=0,
            logging_strategy="steps",
            logging_steps=1,
            logging_first_step=True,
            save_strategy="no",
            eval_strategy="no",
            report_to=[],
            disable_tqdm=True,
            remove_unused_columns=False,
            fp16=True,
            bf16=False,
            tf32=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_prompt_length=512,
            max_completion_length=64,
            num_generations=8,
            temperature=0.8,
            beta=0.01,
            use_vllm=False,
            optim="adamw_torch",
            seed=20260725,
            dataloader_num_workers=0,
        )
        dataset = Dataset.from_list(
            [{"prompt": prompt, "prompt_id": spec.prompt_id}]
        )
        monitor.set_phase("trainer_init")
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=grounding_reward,
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=[callback],
            peft_config=peft_config,
        )
        if trainer.ref_model is not None:
            raise RuntimeError("PEFT smoke unexpectedly created a second ref model")

        adapter_state = load_file(
            str(ADAPTER_PATH / "adapter_model.safetensors"),
            device="cpu",
        )
        load_result = set_peft_model_state_dict(
            trainer.model,
            adapter_state,
            adapter_name="default",
        )
        unexpected_keys = list(getattr(load_result, "unexpected_keys", ()))
        if unexpected_keys:
            raise RuntimeError(
                f"unexpected SFT adapter keys: {unexpected_keys[:3]}"
            )
        trainable_parameters = sum(
            parameter.numel()
            for parameter in trainer.model.parameters()
            if parameter.requires_grad
        )
        if trainable_parameters <= 0:
            raise RuntimeError("no trainable LoRA parameters")

        monitor.set_phase("adapter_probe")
        adapter_logit_max_abs_diff = _adapter_probe(
            trainer.model,
            tokenizer,
            torch,
        )

        unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
        original_generate = unwrapped_model.generate
        original_disable_adapter = unwrapped_model.disable_adapter
        telemetry: dict[str, Any] = {
            "disable_adapter_calls_during_train": 0,
        }

        def measured_generate(*generate_args: Any, **generate_kwargs: Any):
            monitor.set_phase("rollout")
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            generated = original_generate(*generate_args, **generate_kwargs)
            torch.cuda.synchronize()
            telemetry["rollout_torch_peak_allocated_mib"] = round(
                torch.cuda.max_memory_allocated() / 1024**2,
                3,
            )
            telemetry["rollout_torch_peak_reserved_mib"] = round(
                torch.cuda.max_memory_reserved() / 1024**2,
                3,
            )
            monitor.set_phase("policy_reference_update")
            torch.cuda.reset_peak_memory_stats()
            return generated

        @contextmanager
        def measured_disable_adapter():
            telemetry["disable_adapter_calls_during_train"] += 1
            with original_disable_adapter():
                yield

        monitor.set_phase("train_start")
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        with patch.object(
            unwrapped_model,
            "generate",
            new=measured_generate,
        ), patch.object(
            unwrapped_model,
            "disable_adapter",
            new=measured_disable_adapter,
        ):
            train_output = trainer.train()
        torch.cuda.synchronize()
        step_elapsed_sec = time.perf_counter() - step_started
        telemetry["post_rollout_torch_peak_allocated_mib"] = round(
            torch.cuda.max_memory_allocated() / 1024**2,
            3,
        )
        telemetry["post_rollout_torch_peak_reserved_mib"] = round(
            torch.cuda.max_memory_reserved() / 1024**2,
            3,
        )
        monitor.set_phase("post_train")
        time.sleep(0.2)
        monitor.stop()
        memory = monitor.snapshot()

        output_root = Path(temp_output.name)
        output_files = sorted(
            str(path.relative_to(output_root))
            for path in output_root.rglob("*")
            if path.is_file()
        )
        checkpoint_dirs = sorted(
            str(path.relative_to(output_root))
            for path in output_root.rglob("checkpoint-*")
            if path.is_dir()
        )
        if checkpoint_dirs:
            raise RuntimeError("smoke unexpectedly saved a checkpoint")

        kl_values = [
            float(log["kl"])
            for log in callback.logs
            if "kl" in log
        ]
        if len(kl_values) != 1 or not _finite(kl_values[0]):
            raise RuntimeError(f"invalid KL metrics: {kl_values}")
        if len(reward_details) != 8:
            raise RuntimeError("reward function did not score 8 completions")

        component_names = (
            "factual_precision",
            "required_coverage",
            "source_integrity",
            "concision",
        )
        component_summary = {
            name: {
                "mean": _mean(
                    [float(detail[name]) for detail in reward_details]
                ),
                "min": min(float(detail[name]) for detail in reward_details),
                "max": max(float(detail[name]) for detail in reward_details),
                "all_finite": all(
                    _finite(detail[name]) for detail in reward_details
                ),
            }
            for name in component_names
        }
        phase_peaks = memory["phase_peak_used_mib"]
        rollout_peak_mib = max(
            float(telemetry["rollout_torch_peak_reserved_mib"]),
            float(phase_peaks.get("rollout", 0)),
        )
        total_mib = int(
            memory["total_mib"]
            or torch.cuda.get_device_properties(0).total_memory / 1024**2
        )
        rollout_headroom_mib = total_mib - rollout_peak_mib
        memory_passed = (
            rollout_peak_mib < 30 * 1024
            and rollout_headroom_mib >= 2 * 1024
        )

        bnb_compute_dtypes = sorted(
            {
                str(compute_dtype)
                for module in trainer.model.modules()
                if module.__class__.__name__ == "Linear4bit"
                for compute_dtype in (getattr(module, "compute_dtype", None),)
                if compute_dtype is not None
            }
        )
        if not bnb_compute_dtypes:
            bnb_compute_dtypes = [
                str(quantization.bnb_4bit_compute_dtype)
            ]
        report.update(
            {
                "status": "passed",
                "completed_unix_sec": time.time(),
                "checkpoint_saved": False,
                "case": {
                    "path": str(CASE_PATH.relative_to(ROOT)),
                    "sha256": EXPECTED_SHA256[CASE_PATH],
                    "prompt_id": spec.prompt_id,
                    "prompt_tokens": prompt_tokens,
                    "group_size": 8,
                    "scope": case["scope"],
                },
                "model": {
                    "base_path": str(MODEL_PATH.relative_to(ROOT)),
                    "quantization": "NF4 4-bit double-quant",
                    "bnb_4bit_compute_dtypes": bnb_compute_dtypes,
                    "sft_adapter_path": str(
                        ADAPTER_PATH.relative_to(ROOT)
                    ),
                    "sft_adapter_sha256": EXPECTED_SHA256[
                        ADAPTER_PATH / "adapter_model.safetensors"
                    ],
                    "trainable_parameters": trainable_parameters,
                    "adapter_on_vs_off_logit_max_abs_diff": (
                        adapter_logit_max_abs_diff
                    ),
                    "reference_model_object_is_none": trainer.ref_model is None,
                    "reference_method": "shared_base_disable_adapter",
                },
                "precision": {
                    "fp16": args.fp16,
                    "bf16": args.bf16,
                    "accelerator_mixed_precision": (
                        trainer.accelerator.mixed_precision
                    ),
                    "device_name": device_name,
                    "compute_capability": list(capability),
                    "hardware_native_bf16": False,
                    "torch_is_bf16_supported_reported": (
                        torch.cuda.is_bf16_supported()
                    ),
                    "dtype_or_autocast_error": False,
                },
                "optimizer_step": {
                    "global_step": trainer.state.global_step,
                    "optimizer_step_callback_count": callback.optimizer_steps,
                    "max_steps": args.max_steps,
                    "learning_rate": args.learning_rate,
                    "beta": args.beta,
                    "elapsed_sec": round(step_elapsed_sec, 3),
                    "training_loss": float(train_output.training_loss),
                    "pipeline": [
                        "rollout",
                        "programmatic_reward",
                        "adapter_off_reference_kl",
                        "backward",
                        "optimizer_update",
                    ],
                },
                "reward": {
                    "completion_count": len(reward_details),
                    "components": component_summary,
                    "total_mean": _mean(
                        [
                            float(detail["total"])
                            for detail in reward_details
                        ]
                    ),
                    "all_values_finite": all(
                        _finite(detail[name])
                        for detail in reward_details
                        for name in ("total",) + component_names
                    ),
                    "per_completion": reward_details,
                },
                "kl": {
                    "mean": kl_values[0],
                    "finite": _finite(kl_values[0]),
                    "beta": args.beta,
                    "disable_adapter_calls_during_train": telemetry[
                        "disable_adapter_calls_during_train"
                    ],
                },
                "memory": {
                    **memory,
                    **telemetry,
                    "rollout_measured_peak_mib": round(
                        rollout_peak_mib,
                        3,
                    ),
                    "rollout_measured_peak_gib": round(
                        rollout_peak_mib / 1024,
                        3,
                    ),
                    "rollout_headroom_mib": round(
                        rollout_headroom_mib,
                        3,
                    ),
                    "rollout_headroom_gib": round(
                        rollout_headroom_mib / 1024,
                        3,
                    ),
                    "criterion_peak_below_30_gib": (
                        rollout_peak_mib < 30 * 1024
                    ),
                    "criterion_headroom_at_least_2_gib": (
                        rollout_headroom_mib >= 2 * 1024
                    ),
                    "passed": memory_passed,
                },
                "output_contract": {
                    "save_strategy": str(args.save_strategy),
                    "checkpoint_directories": checkpoint_dirs,
                    "temporary_output_files": output_files,
                    "temporary_output_removed_after_report": True,
                    "persistent_output": str(REPORT_PATH.relative_to(ROOT)),
                },
            }
        )

        required_checks = {
            "one_optimizer_step": (
                trainer.state.global_step == 1
                and callback.optimizer_steps == 1
            ),
            "adapter_off_reference_used": (
                trainer.ref_model is None
                and telemetry["disable_adapter_calls_during_train"] >= 1
            ),
            "fp16_without_bf16": (
                args.fp16
                and not args.bf16
                and trainer.accelerator.mixed_precision == "fp16"
                and bnb_compute_dtypes == ["torch.float16"]
            ),
            "memory_within_budget": memory_passed,
            "four_reward_components_finite": all(
                item["all_finite"] for item in component_summary.values()
            ),
            "kl_finite": _finite(kl_values[0]),
            "no_checkpoint": not checkpoint_dirs,
            "training_loss_finite": _finite(train_output.training_loss),
        }
        report["required_checks"] = required_checks
        if not all(required_checks.values()):
            report["status"] = "failed"
            report["failure"] = {
                "type": "SmokeCriterionFailure",
                "message": (
                    "one or more required smoke criteria did not pass"
                ),
            }
        _write_report(report)
        return 0 if report["status"] == "passed" else 1
    except Exception as error:
        if monitor._thread.is_alive():
            monitor.stop()
        report.update(
            {
                "status": "failed",
                "completed_unix_sec": time.time(),
                "checkpoint_saved": False,
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
                "memory": monitor.snapshot(),
            }
        )
        _write_report(report)
        return 1
    finally:
        if temp_output is not None:
            temp_output.cleanup()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
