#!/usr/bin/env python3
"""Freeze formal_v4 fixed dev-4 no_grad step0 baseline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GRPO_DIR = ROOT / "data" / "model_training" / "grpo"
SOURCE_AUTH = GRPO_DIR / "grpo_formal_v3_authorization.json"
INPUT_PATH = GRPO_DIR / "grpo_expanded_v4_input_manifest.json"
OUT = GRPO_DIR / "grpo_formal_v4_fixed_dev4_step0_baseline.json"
OUT_SHA = OUT.with_suffix(".sha256")
MODEL_PATH = ROOT / "models" / "Qwen2.5-7B-Instruct"
ADAPTER_PATH = ROOT / "checkpoints" / "sft" / "best_adapter"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_exclusive(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    for path in (OUT, OUT_SHA):
        if path.exists():
            raise FileExistsError(f"refusing overwrite: {path}")
    authorization = json.loads(SOURCE_AUTH.read_text(encoding="utf-8"))
    config = authorization["training_config"]
    input_manifest = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    import torch
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        set_peft_model_state_dict,
    )
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from training.grpo.formal_training import (
        FIXED_DEV4_PROBE_IDS,
        evaluate_fixed_dev4_probe_no_grad,
        fixed_dev4_cases,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dev_cases = fixed_dev4_cases(
        [case for case in input_manifest["cases"] if case["split"] == "dev"]
    )
    if [case["id"] for case in dev_cases] != list(FIXED_DEV4_PROBE_IDS):
        raise RuntimeError("fixed dev-4 prompt order drift")

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
    base_model = prepare_model_for_kbit_training(
        base_model, use_gradient_checkpointing=True
    )
    adapter_config = json.loads(
        (ADAPTER_PATH / "adapter_config.json").read_text(encoding="utf-8")
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
    model = get_peft_model(base_model, peft_config)
    load_result = set_peft_model_state_dict(
        model,
        load_file(str(ADAPTER_PATH / "adapter_model.safetensors"), device="cpu"),
        adapter_name="default",
    )
    if list(getattr(load_result, "unexpected_keys", ())):
        raise RuntimeError("unexpected SFT adapter keys")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    probe = evaluate_fixed_dev4_probe_no_grad(
        model=model,
        tokenizer=tokenizer,
        cases=dev_cases,
        torch=torch,
        config=config,
        optimizer_step=0,
        train_window=None,
    )
    baseline = {
        "format_version": 1,
        "status": "frozen_before_formal_v4_training_no_optimizer_step",
        "run_id": "grpo-formal-v4",
        "source": "fixed_dev4_no_grad_probe_step0",
        "prompt_ids": probe["probe_prompt_ids"],
        "prompt_id_set_is_fixed_dev4": probe["prompt_id_set_is_fixed_dev4"],
        "generation_parameters": {
            key: config[key]
            for key in (
                "num_generations",
                "temperature",
                "top_p",
                "max_prompt_length",
                "max_completion_length",
                "max_total_sequence_length",
                "fp16",
                "bf16",
                "seed",
                "beta",
                "reference",
            )
        },
        "reward_mean": probe["reward_mean"],
        "factual_precision_mean": probe["factual_precision_mean"],
        "required_coverage_mean": probe["required_coverage_mean"],
        "source_integrity_mean": probe["source_integrity_mean"],
        "concision_mean": probe["concision_mean"],
        "reward_mean_by_intent": probe["reward_mean_by_intent"],
        "factual_precision_mean_by_intent": probe[
            "factual_precision_mean_by_intent"
        ],
        "intent_response_fail_rate": probe["intent_response_fail_rate"],
        "per_intent_fail_rate": probe["intent_response_fail_rate_by_intent"],
        "zero_variance_group_ratio": probe["zero_variance_group_share"],
        "completion_length_p50": probe["length_p50"],
        "kl0_mean": probe["kl_mean"],
        "kl0_p95": probe["kl_p95"],
        "fake_reward_gate": probe["fake_reward_gate"],
        "case_results": probe["case_results"],
        "frozen_hashes": {
            "source_authorization_sha256": sha256_file(SOURCE_AUTH),
            "expanded_v4_input_manifest_sha256": sha256_file(INPUT_PATH),
            "sft_adapter_model_sha256": sha256_file(
                ADAPTER_PATH / "adapter_model.safetensors"
            ),
            "sft_adapter_config_sha256": sha256_file(
                ADAPTER_PATH / "adapter_config.json"
            ),
            "reward_fn_sha256": sha256_file(
                ROOT / "training" / "grpo" / "reward_fn.py"
            ),
        },
        "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "optimizer_step_performed": False,
        "loss_backward_performed": False,
        "checkpoint_created": False,
    }
    write_json_exclusive(OUT, baseline)
    digest = sha256_file(OUT)
    OUT_SHA.write_text(f"{digest}  {OUT.name}\n", encoding="ascii")
    print(json.dumps({"baseline_sha256": digest, "fake_reward_gate": baseline["fake_reward_gate"], "kl0_p95": baseline["kl0_p95"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
