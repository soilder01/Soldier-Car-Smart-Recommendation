#!/usr/bin/env python3
"""Evaluate sales_dense_v2 checkpoints on the frozen newanchor exam."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from training.grpo.run_newanchor_local_eval import evaluate_adapter, load_jsonl


ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "data" / "model_training" / "grpo" / "formal_v4" / "restart_1"
DATASET = BASE_DIR / "powered_dev_newanchor_128.jsonl"
NEWANCHOR_PROTOCOL = BASE_DIR / "newanchor_eval_protocol.json"
DENSE_PROTOCOL = BASE_DIR / "sales_dense_v2_train_protocol.json"
SOURCE_FOURWAY = BASE_DIR / "newanchor_fourway_evaluations.jsonl"
OUT_EVAL = BASE_DIR / "sales_dense_v2_evaluations.jsonl"
OUT_SELECTION = BASE_DIR / "sales_dense_v2_selection_result.json"
REWARD_FN = ROOT / "training" / "grpo" / "reward_fn.py"
MODEL_PATH = ROOT / "models" / "Qwen2.5-7B-Instruct"
SFT_ADAPTER = ROOT / "checkpoints" / "sft" / "best_adapter"
TARGET_DIR = ROOT / "checkpoints" / "grpo" / "formal_v4" / "restart_1" / "sales_dense_v2"

EXPECTED_REWARD_SHA = "325ad44feb83ec37c35babfed4bddb928cf400788e07735eb4631fc4af6962c8"
EXPECTED_DATASET_SHA = "e74de770e8dbc95dbbf813d87fa9b38e631941ec464f790410b2c86be41c0c2b"
EXPECTED_NEWANCHOR_PROTOCOL_SHA = "3f53319f9f4158dd1e282cbefa9f84dcdf0e295c018023cae8d1557020a94783"
EXPECTED_SOURCE_FOURWAY_SHA = "beeb1d41c212e3d65428ae7a380b135889786d1838cbc6656976a16e4a9a79a5"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_sha_sidecar(artifact: Path) -> None:
    with Path(str(artifact) + ".sha256").open("x", encoding="ascii") as handle:
        handle.write(f"{sha256_file(artifact)}  {artifact.name}\n")


def write_jsonl_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n")


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def assert_preflight() -> tuple[dict[str, str], dict[str, Any]]:
    if OUT_EVAL.exists() or Path(str(OUT_EVAL) + ".sha256").exists():
        raise FileExistsError(f"refusing overwrite: {OUT_EVAL}")
    if OUT_SELECTION.exists() or Path(str(OUT_SELECTION) + ".sha256").exists():
        raise FileExistsError(f"refusing overwrite: {OUT_SELECTION}")
    expected = {
        REWARD_FN: EXPECTED_REWARD_SHA,
        DATASET: EXPECTED_DATASET_SHA,
        NEWANCHOR_PROTOCOL: EXPECTED_NEWANCHOR_PROTOCOL_SHA,
        SOURCE_FOURWAY: EXPECTED_SOURCE_FOURWAY_SHA,
    }
    observed: dict[str, str] = {}
    for path, digest in expected.items():
        actual = sha256_file(path)
        if actual != digest:
            raise RuntimeError(f"SHA mismatch: {path.relative_to(ROOT)}")
        observed[str(path.relative_to(ROOT))] = actual
    protocol = json.loads(DENSE_PROTOCOL.read_text(encoding="utf-8"))
    for relative, digest in protocol["expected_hashes"].items():
        actual = sha256_file(ROOT / relative)
        if actual != digest:
            raise RuntimeError(f"sales_dense_v2 expected hash drift: {relative}")
        observed[relative] = actual
    for step in protocol["pre_registered_gate"]["candidate_checkpoint_steps"]:
        adapter = TARGET_DIR / f"checkpoint-{step}" / "adapter_model.safetensors"
        if not adapter.exists():
            raise RuntimeError(f"missing dense checkpoint adapter: checkpoint-{step}")
    return observed, protocol


def by_intent_score(row: dict[str, Any], intent: str) -> float:
    return float(row["by_intent"][intent]["mean_core"])


def main() -> int:
    observed_hashes, dense_protocol = assert_preflight()
    newanchor_protocol = json.loads(NEWANCHOR_PROTOCOL.read_text(encoding="utf-8"))
    sampling = newanchor_protocol["sampling"]
    cases = load_jsonl(DATASET)
    if len(cases) != 128:
        raise RuntimeError("newanchor dataset count drift")
    source_rows = [json.loads(line) for line in SOURCE_FOURWAY.read_text(encoding="utf-8").splitlines() if line.strip()]
    source_by_label = {row["label"]: row for row in source_rows}
    reused_labels = ("step0", "checkpoint-300", "cloud_seedpro_ark_ep_masked")
    for label in reused_labels:
        if label not in source_by_label:
            raise RuntimeError(f"missing reusable source row: {label}")

    import sys
    import torch
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if Path(sys.prefix).resolve() != (ROOT / ".venv-grpo").resolve():
        raise RuntimeError("sales_dense_v2 newanchor eval must run in .venv-grpo")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
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
    dense_rows: list[dict[str, Any]] = []
    for step in dense_protocol["pre_registered_gate"]["candidate_checkpoint_steps"]:
        label = f"sales-dense-v2-checkpoint-{step}"
        dense_rows.append(
            evaluate_adapter(
                label=label,
                adapter_path=TARGET_DIR / f"checkpoint-{step}",
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

    rows = [
        source_by_label["step0"],
        source_by_label["checkpoint-300"],
        *dense_rows,
        source_by_label["cloud_seedpro_ark_ep_masked"],
    ]
    for row in dense_rows:
        row["evaluation_added_by"] = "sales_dense_v2_newanchor_eval"
    for row in (source_by_label["step0"], source_by_label["checkpoint-300"], source_by_label["cloud_seedpro_ark_ep_masked"]):
        row["reused_from"] = str(SOURCE_FOURWAY.relative_to(ROOT))
        row["reused_from_sha256"] = EXPECTED_SOURCE_FOURWAY_SHA
    write_jsonl_exclusive(OUT_EVAL, rows)
    write_sha_sidecar(OUT_EVAL)

    gate = dense_protocol["pre_registered_gate"]
    recommend_threshold = float(gate["recommend_mean_core_must_be_at_least"])
    sales_threshold = float(gate["sales_mean_core_must_be_strictly_greater_than"])
    candidates = []
    eligible = []
    for row in dense_rows:
        record = {
            "label": row["label"],
            "recommend": by_intent_score(row, "recommend"),
            "sales": by_intent_score(row, "sales"),
            "composite": float(row["composite"]),
            "passes_recommend_gate": by_intent_score(row, "recommend") >= recommend_threshold,
            "passes_sales_gate": by_intent_score(row, "sales") > sales_threshold,
        }
        record["passes_all_pre_registered_gates"] = record["passes_recommend_gate"] and record["passes_sales_gate"]
        candidates.append(record)
        if record["passes_all_pre_registered_gates"]:
            eligible.append(record)
    selected = None
    if eligible:
        selected = sorted(
            eligible,
            key=lambda row: (-row["sales"], int(row["label"].rsplit("-", 1)[-1])),
        )[0]
    result = {
        "status": "passed" if selected else "invalid_no_checkpoint_satisfied_pre_registered_gate",
        "selection_rule": gate,
        "selected": selected,
        "eligible": eligible,
        "candidates": candidates,
        "checkpoint_300": {
            "label": "checkpoint-300",
            "recommend": by_intent_score(source_by_label["checkpoint-300"], "recommend"),
            "sales": by_intent_score(source_by_label["checkpoint-300"], "sales"),
            "composite": float(source_by_label["checkpoint-300"]["composite"]),
        },
        "cloud": {
            "label": "cloud_seedpro_ark_ep_masked",
            "recommend": by_intent_score(source_by_label["cloud_seedpro_ark_ep_masked"], "recommend"),
            "sales": by_intent_score(source_by_label["cloud_seedpro_ark_ep_masked"], "sales"),
            "composite": float(source_by_label["cloud_seedpro_ark_ep_masked"]["composite"]),
        },
        "artifacts": {
            "evaluations_jsonl": str(OUT_EVAL.relative_to(ROOT)),
            "evaluations_sha256": sha256_file(OUT_EVAL),
        },
        "observed_hashes": observed_hashes,
        "reward_fn_sha256": sha256_file(REWARD_FN),
        "held_out_40_body_read": False,
        "final_40_body_read": False,
        "cloud_called": False,
        "raw_answers_persisted": False,
    }
    write_json_exclusive(OUT_SELECTION, result)
    write_sha_sidecar(OUT_SELECTION)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected": selected,
                "candidates": candidates,
                "evaluations_sha256": sha256_file(OUT_EVAL),
                "selection_sha256": sha256_file(OUT_SELECTION),
                "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
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
