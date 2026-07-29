#!/usr/bin/env python3
"""Evaluate locked local adapters on frozen held-out-40 reward cases."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from training.grpo.run_newanchor_local_eval import evaluate_adapter, load_jsonl


ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "data" / "model_training" / "grpo" / "formal_v4" / "restart_1"
DATASET = BASE_DIR / "held_out_40_frozen_eval.jsonl"
PROTOCOL = BASE_DIR / "held_out_40_final_protocol.json"
REWARD_FN = ROOT / "training" / "grpo" / "reward_fn.py"
MODEL_PATH = ROOT / "models" / "Qwen2.5-7B-Instruct"
SFT_ADAPTER = ROOT / "checkpoints" / "sft" / "best_adapter"
DENSE_CKPT150 = ROOT / "checkpoints" / "grpo" / "formal_v4" / "restart_1" / "sales_dense_v2" / "checkpoint-150"
CKPT300 = ROOT / "checkpoints" / "grpo" / "formal_v4" / "restart_1" / "checkpoint-300"
GREEDY = os.getenv("HELDOUT40_GREEDY", "") == "1"
DEFAULT_CACHE_OUT = (
    BASE_DIR / f"heldout40_greedy_adapter_{int(time.time())}.json"
    if GREEDY
    else Path("/tmp/heldout40_local_eval_cache.json")
)
CACHE_OUT = Path(os.environ.get("HELDOUT40_LOCAL_EVAL_CACHE", str(DEFAULT_CACHE_OUT)))

EXPECTED_REWARD_SHA = "325ad44feb83ec37c35babfed4bddb928cf400788e07735eb4631fc4af6962c8"
EXPECTED_DATASET_SHA = "2b4a2e8dff52b9feee12bb451ce630dc0e03661c1e4b8935baf3a78df77013ea"
EXPECTED_PROTOCOL_SHA = "eb5d6a488205666414dd7b5c3484555a52c7789de958badf572108994662531d"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_sha_sidecar(path: Path) -> None:
    with Path(str(path) + ".sha256").open("x", encoding="ascii") as handle:
        handle.write(f"{sha256_file(path)}  {path.name}\n")


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def summarize(records: list[dict[str, Any]], prompt_count: int) -> dict[str, Any]:
    return {
        "prompt_count": prompt_count,
        "completion_count": sum(record["completion_count"] for record in records),
        "mean_core": mean([record["per_prompt_core_mean"] for record in records]),
        "factual_precision_mean": mean([record["factual_precision_mean"] for record in records]),
        "required_coverage_mean": mean([record["required_coverage_mean"] for record in records]),
        "reward_mean": mean([record["reward_mean"] for record in records]),
        "source_integrity_mean": mean([record["source_integrity_mean"] for record in records]),
        "concision_mean": mean([record["concision_mean"] for record in records]),
        "gate_counts_total": dict(sorted(sum((Counter(record["gate_counts"]) for record in records), Counter()).items())),
    }


def enrich_all_intents(row: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    records = row["case_results"]
    row["by_intent_all"] = {
        intent: summarize(
            [record for record in records if record["intent"] == intent],
            sum(1 for case in cases if case["intent"] == intent),
        )
        for intent in ("recommend", "sales", "compare", "knowledge")
    }
    row["by_intent"] = {
        "recommend": row["by_intent_all"]["recommend"],
        "sales": row["by_intent_all"]["sales"],
    }
    row["composite"] = 0.5 * row["by_intent"]["recommend"]["mean_core"] + 0.5 * row["by_intent"]["sales"]["mean_core"]
    row["composite_recommend_sales_core"] = row["composite"]
    return row


def assert_preflight() -> None:
    expected = {
        REWARD_FN: EXPECTED_REWARD_SHA,
        DATASET: EXPECTED_DATASET_SHA,
        PROTOCOL: EXPECTED_PROTOCOL_SHA,
    }
    for path, digest in expected.items():
        actual = sha256_file(path)
        if actual != digest:
            raise RuntimeError(f"SHA mismatch: {path.relative_to(ROOT)}")
    for path in (DENSE_CKPT150, CKPT300):
        if not (path / "adapter_model.safetensors").exists():
            raise RuntimeError(f"missing adapter: {path.relative_to(ROOT)}")
    if CACHE_OUT.exists():
        raise FileExistsError(f"cache already exists: {CACHE_OUT}")


def main() -> int:
    assert_preflight()
    import sys
    import torch
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if Path(sys.prefix).resolve() != (ROOT / ".venv-grpo").resolve():
        raise RuntimeError("held-out local eval must run in .venv-grpo")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    sampling = protocol["sampling"]
    cases = load_jsonl(DATASET)
    if len(cases) != 40 or Counter(case["intent"] for case in cases) != {"recommend": 10, "sales": 10, "compare": 10, "knowledge": 10}:
        raise RuntimeError("held-out frozen dataset count drift")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
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
    adapter_config = json.loads((SFT_ADAPTER / "adapter_config.json").read_text(encoding="utf-8"))
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
    rows: list[dict[str, Any]] = []
    eval_targets = (("sales_dense_v2_checkpoint_150", DENSE_CKPT150),) if GREEDY else (
        ("sales_dense_v2_checkpoint_150", DENSE_CKPT150),
        ("checkpoint-300", CKPT300),
    )
    for label, adapter_path in eval_targets:
        row = evaluate_adapter(
            label=label,
            adapter_path=adapter_path,
            model=model,
            tokenizer=tokenizer,
            set_peft_model_state_dict=set_peft_model_state_dict,
            load_file=load_file,
            torch=torch,
            cases=cases,
            sampling=sampling,
            greedy=GREEDY,
        )
        row["raw_answers_persisted"] = False
        row["held_out_40_body_read"] = True
        rows.append(enrich_all_intents(row, cases))
        torch.cuda.empty_cache()
    payload = {
        "status": "heldout40_greedy_adapter_eval_completed" if GREEDY else "heldout40_local_eval_completed",
        "dataset_sha256": sha256_file(DATASET),
        "protocol_sha256": sha256_file(PROTOCOL),
        "reward_fn_sha256": sha256_file(REWARD_FN),
        "rows": rows,
        "raw_answers_persisted": False,
        "greedy": GREEDY,
        "terminal_artifacts_overwritten": False,
        "access_state_updated": False,
        "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
    }
    with CACHE_OUT.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    write_sha_sidecar(CACHE_OUT)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "cache": str(CACHE_OUT),
                "labels": [row["label"] for row in rows],
                "scores": {
                    row["label"]: {
                        "recommend": row["by_intent"]["recommend"]["mean_core"],
                        "sales": row["by_intent"]["sales"]["mean_core"],
                        "composite": row["composite"],
                    }
                    for row in rows
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
