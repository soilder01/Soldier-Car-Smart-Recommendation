#!/usr/bin/env python3
"""Evaluate local checkpoints on the frozen powered-dev-newanchor-128 exam."""

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

from training.grpo.formal_training import rollout_generation_cache_context
from training.grpo.reward_fn import score_grounded_answer
from training.grpo.run_signal_probe import build_prompt, spec_and_claims


ROOT = Path(__file__).resolve().parents[2]
DATASET = (
    ROOT
    / "data"
    / "model_training"
    / "grpo"
    / "formal_v4"
    / "restart_1"
    / "powered_dev_newanchor_128.jsonl"
)
PROTOCOL = (
    ROOT
    / "data"
    / "model_training"
    / "grpo"
    / "formal_v4"
    / "restart_1"
    / "newanchor_eval_protocol.json"
)
REWARD_FN = ROOT / "training" / "grpo" / "reward_fn.py"
MODEL_PATH = ROOT / "models" / "Qwen2.5-7B-Instruct"
SFT_ADAPTER = ROOT / "checkpoints" / "sft" / "best_adapter"
CKPT300 = ROOT / "checkpoints" / "grpo" / "formal_v4" / "restart_1" / "checkpoint-300"
TARGET_DIR = ROOT / "checkpoints" / "grpo" / "formal_v4" / "restart_1" / "sales_targeted"
CACHE_OUT = Path(os.environ.get("NEWANCHOR_LOCAL_EVAL_CACHE", "/tmp/newanchor_local_eval_cache.json"))

EXPECTED_REWARD_SHA = "325ad44feb83ec37c35babfed4bddb928cf400788e07735eb4631fc4af6962c8"
EXPECTED_DATASET_SHA = "e74de770e8dbc95dbbf813d87fa9b38e631941ec464f790410b2c86be41c0c2b"
EXPECTED_PROTOCOL_SHA = "3f53319f9f4158dd1e282cbefa9f84dcdf0e295c018023cae8d1557020a94783"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


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
    for path in (SFT_ADAPTER, CKPT300, TARGET_DIR):
        if not path.is_dir():
            raise RuntimeError(f"missing adapter/checkpoint directory: {path}")
    for step in (50, 100, 150, 200):
        if not (TARGET_DIR / f"checkpoint-{step}" / "adapter_model.safetensors").exists():
            raise RuntimeError(f"missing sales targeted checkpoint-{step}")
    if CACHE_OUT.exists():
        raise FileExistsError(f"cache already exists: {CACHE_OUT}")


def summarize_intent(records: list[dict[str, Any]], prompt_count: int) -> dict[str, Any]:
    return {
        "prompt_count": prompt_count,
        "completion_count": sum(record["completion_count"] for record in records),
        "mean_core": mean([record["per_prompt_core_mean"] for record in records]),
        "factual_precision_mean": mean([record["factual_precision_mean"] for record in records]),
        "required_coverage_mean": mean([record["required_coverage_mean"] for record in records]),
        "reward_mean": mean([record["reward_mean"] for record in records]),
        "source_integrity_mean": mean([record["source_integrity_mean"] for record in records]),
        "concision_mean": mean([record["concision_mean"] for record in records]),
        "gate_counts_total": dict(
            sorted(
                sum((Counter(record["gate_counts"]) for record in records), Counter()).items()
            )
        ),
    }


def case_result_from_scores(case: dict[str, Any], scores: list[Any], lengths: list[int], generation_time: float) -> dict[str, Any]:
    fp = [float(score.factual_precision) for score in scores]
    cov = [float(score.required_coverage) for score in scores]
    src = [float(score.source_integrity) for score in scores]
    con = [float(score.concision) for score in scores]
    reward = [float(score.total) for score in scores]
    core = [0.6 * f + 0.4 * c for f, c in zip(fp, cov)]
    query = case["query"]
    return {
        "prompt_id": case["id"],
        "query_summary": query[:80],
        "intent": case["intent"],
        "candidate_index": case["candidate_index"],
        "target_entities": case["target_entities"],
        "query_anchor_tokens": case["intent_response_spec"]["query_anchor_tokens"],
        "query_attribute_anchors": case["intent_response_spec"]["query_attribute_anchors"],
        "completion_count": len(scores),
        "per_prompt_core_mean": mean(core),
        "factual_precision_mean": mean(fp),
        "required_coverage_mean": mean(cov),
        "reward_mean": mean(reward),
        "source_integrity_mean": mean(src),
        "concision_mean": mean(con),
        "gate_counts": dict(sorted(Counter(score.gate for score in scores).items())),
        "completion_length_mean": mean(lengths),
        "completion_length_max": max(lengths) if lengths else 0,
        "generation_time_sec": generation_time,
    }


def evaluate_adapter(
    *,
    label: str,
    adapter_path: Path,
    model: Any,
    tokenizer: Any,
    set_peft_model_state_dict: Any,
    load_file: Any,
    torch: Any,
    cases: list[dict[str, Any]],
    sampling: dict[str, Any],
    greedy: bool = False,
) -> dict[str, Any]:
    state = load_file(str(adapter_path / "adapter_model.safetensors"), device="cpu")
    result = set_peft_model_state_dict(model, state, adapter_name="default")
    if list(getattr(result, "unexpected_keys", ())):
        raise RuntimeError(f"{label}: unexpected adapter keys")
    model.eval()
    torch.manual_seed(int(sampling["seed"]))
    torch.cuda.manual_seed_all(int(sampling["seed"]))
    per_prompt: list[dict[str, Any]] = []
    started = time.time()
    with torch.inference_mode():
        for index, case in enumerate(cases, start=1):
            spec, claims = spec_and_claims(case)
            rendered = tokenizer.apply_chat_template(
                build_prompt(case),
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(
                rendered,
                return_tensors="pt",
                add_special_tokens=False,
            )
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            prompt_length = int(inputs["input_ids"].shape[1])
            if prompt_length > 2560:
                raise RuntimeError(f"{case['id']}: prompt exceeds max_prompt_length")
            gen_started = time.time()
            generation_kwargs = {
                "do_sample": not greedy,
                "num_return_sequences": 1 if greedy else int(sampling["num_generations"]),
                "max_new_tokens": int(sampling["max_completion_length"]),
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": model.generation_config.eos_token_id,
            }
            if not greedy:
                generation_kwargs["temperature"] = float(sampling["temperature"])
                generation_kwargs["top_p"] = float(sampling["top_p"])
            with rollout_generation_cache_context(model, model.generation_config):
                generated = model.generate(
                    **inputs,
                    **generation_kwargs,
                )
            generation_time = time.time() - gen_started
            completion_ids = generated[:, prompt_length:]
            texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
            scores = [
                score_grounded_answer(answer=text, spec=spec, evidence_claims=claims)
                for text in texts
            ]
            lengths = [
                len(tokenizer(text, add_special_tokens=False)["input_ids"])
                for text in texts
            ]
            per_prompt.append(case_result_from_scores(case, scores, lengths, generation_time))
            if index % 16 == 0 or index == len(cases):
                print(
                    f"local_eval_progress label={label} prompts={index}/{len(cases)} "
                    f"elapsed_sec={time.time() - started:.1f}",
                    flush=True,
                )
    by_intent = {
        intent: summarize_intent(
            [record for record in per_prompt if record["intent"] == intent],
            sum(1 for case in cases if case["intent"] == intent),
        )
        for intent in ("recommend", "sales")
    }
    composite = 0.5 * by_intent["recommend"]["mean_core"] + 0.5 * by_intent["sales"]["mean_core"]
    return {
        "label": label,
        "object_type": "local_adapter",
        "adapter_path": str(adapter_path.relative_to(ROOT)),
        "adapter_sha256": sha256_file(adapter_path / "adapter_model.safetensors"),
        "sampling_config": sampling | {"greedy": greedy, "do_sample": not greedy},
        "raw_answers_persisted": False,
        "by_intent": by_intent,
        "composite": composite,
        "case_results": per_prompt,
        "completed_unix_sec": time.time(),
    }


def main() -> int:
    assert_preflight()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    sampling = protocol["sampling"]
    cases = load_jsonl(DATASET)
    if len(cases) != 128 or Counter(case["intent"] for case in cases) != {"recommend": 64, "sales": 64}:
        raise RuntimeError("newanchor dataset count drift")

    import torch
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if Path(sys.prefix).resolve() != (ROOT / ".venv-grpo").resolve():
        raise RuntimeError("local eval must run in .venv-grpo")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH), local_files_only=True, trust_remote_code=False
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
    labels = [
        ("step0", SFT_ADAPTER),
        ("checkpoint-300", CKPT300),
        ("sales-targeted-checkpoint-50", TARGET_DIR / "checkpoint-50"),
        ("sales-targeted-checkpoint-100", TARGET_DIR / "checkpoint-100"),
        ("sales-targeted-checkpoint-150", TARGET_DIR / "checkpoint-150"),
        ("sales-targeted-checkpoint-200", TARGET_DIR / "checkpoint-200"),
    ]
    rows = []
    for label, adapter_path in labels:
        rows.append(
            evaluate_adapter(
                label=label,
                adapter_path=adapter_path,
                model=model,
                tokenizer=tokenizer,
                set_peft_model_state_dict=set_peft_model_state_dict,
                load_file=load_file,
                torch=torch,
                cases=cases,
                sampling=sampling,
            )
        )
        torch.cuda.empty_cache()
    payload = {
        "status": "local_newanchor_eval_completed",
        "dataset_sha256": sha256_file(DATASET),
        "protocol_sha256": sha256_file(PROTOCOL),
        "reward_fn_sha256": sha256_file(REWARD_FN),
        "rows": rows,
        "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "held_out_40_body_read": False,
        "final_40_body_read": False,
    }
    with CACHE_OUT.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "cache": str(CACHE_OUT),
                "labels": [row["label"] for row in rows],
                "cuda_peak_reserved_mib": payload["cuda_peak_reserved_mib"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
