#!/usr/bin/env python3
"""Run the read-only 20x8 GRPO reward-variance signal probe."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

from training.grpo.reward_fn import (
    EvidenceClaim,
    IntentResponseSpec,
    score_grounded_answer,
)


ROOT = Path(__file__).resolve().parents[2]
GRPO_DIR = ROOT / "data" / "model_training" / "grpo"
INPUT_PATH = GRPO_DIR / "grpo_signal_probe_input_manifest.json"
INPUT_SHA_PATH = INPUT_PATH.with_suffix(".sha256")
RAW_PATH = GRPO_DIR / "grpo_signal_probe_raw.jsonl"
RAW_SHA_PATH = RAW_PATH.with_suffix(".sha256")
REPORT_PATH = GRPO_DIR / "grpo_signal_probe_report.json"
REPORT_SHA_PATH = REPORT_PATH.with_suffix(".sha256")
FAILURE_PATH = GRPO_DIR / "grpo_signal_probe_failure.json"
MODEL_PATH = ROOT / "models" / "Qwen2.5-7B-Instruct"
ADAPTER_PATH = ROOT / "checkpoints" / "sft" / "best_adapter"
FORMAL_ENTRY = ROOT / "training" / "grpo" / "train_grpo.py"
EXPECTED_INPUT_SHA256 = (
    "bf1f894cc90c3eb503e08dcd06b661315388262f11a413ea2e00fd0604d161ef"
)
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
EXTRA_FROZEN_SHA256 = {
    ROOT
    / "data"
    / "model_training"
    / "eval"
    / "product_faithful_v2_scoring_manifest.json": (
        "49d3a2b1da23490b236f839f73d55654743a596899909fa80b4fcc090d721113"
    ),
    ROOT / "data" / "model_training" / "eval" / "production_prompts_v3.json": (
        "22ace6242e57b2bee32dde880487970dbc195de257a4b4a1264ebf6e75da8c2b"
    ),
    ROOT
    / "data"
    / "model_training"
    / "eval"
    / "frozen_production_prompt_harness_v3.json": (
        "55453e5047e12a636cdaf9c73003cc3768e8e8417cac6bb5f74ce74f68c1fe3c"
    ),
    ROOT / "data" / "model_training" / "eval" / "held_out.jsonl": (
        "964fc352d1c83fa2738042d377c8070d6e355c51a5ddbb36c1fc9a9b99771a79"
    ),
    ROOT
    / "data"
    / "model_training"
    / "eval"
    / "grpo_final_held_out.jsonl": (
        "0fd611ffdfed27adb50615e76a1fd8b0f43a5330676e22f19a27b764d2c4678c"
    ),
    ADAPTER_PATH / "adapter_model.safetensors": (
        "c55ec155f08e21125d9cf0383c9a6ee066f61d31d3a0e9285eb72a8fc4d17800"
    ),
    ADAPTER_PATH / "adapter_config.json": (
        "ec3f1c5de4da14fa695ccc65ba1f27083c34df32e0c71722c0c78166441bb8b9"
    ),
    MODEL_PATH / "config.json": (
        "7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c"
    ),
    MODEL_PATH / "tokenizer.json": (
        "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
    ),
    MODEL_PATH / "model.safetensors.index.json": (
        "624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028"
    ),
}
COMPONENTS = (
    "factual_precision",
    "required_coverage",
    "source_integrity",
    "concision",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exclusive_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_sha(path: Path, artifact: Path) -> None:
    with path.open("x", encoding="ascii") as handle:
        handle.write(f"{sha256_file(artifact)}  {artifact.name}\n")


def venv_train_snapshot() -> dict[str, str]:
    freeze = subprocess.check_output(
        [str(ROOT / ".venv-train" / "bin" / "python"), "-m", "pip", "freeze"],
        text=True,
    )
    metadata = hashlib.sha256()
    for path in sorted(
        item for item in (ROOT / ".venv-train").rglob("*") if item.is_file()
    ):
        stat = path.stat()
        metadata.update(
            (
                f"{path.relative_to(ROOT)} {stat.st_size} "
                f"{stat.st_mtime_ns}\n"
            ).encode("utf-8")
        )
    return {
        "pip_freeze_sha256": hashlib.sha256(
            freeze.encode("utf-8")
        ).hexdigest(),
        "file_metadata_sha256": metadata.hexdigest(),
    }


def formal_entry_status() -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, str(FORMAL_ENTRY)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    message = (result.stdout + result.stderr).strip()
    return {
        "exit_code": result.returncode,
        "fail_closed": result.returncode != 0 and "gated" in message,
        "message": message,
        "sha256": sha256_file(FORMAL_ENTRY),
    }


def frozen_snapshot(manifest: dict[str, Any]) -> dict[str, str]:
    expected = {
        ROOT / relative: digest
        for relative, digest in manifest["frozen_sources"].items()
    }
    expected.update(EXTRA_FROZEN_SHA256)
    expected[INPUT_PATH] = EXPECTED_INPUT_SHA256
    observed: dict[str, str] = {}
    for path, digest in expected.items():
        actual = sha256_file(path)
        if actual != digest:
            raise ValueError(f"frozen SHA drift: {path.relative_to(ROOT)}")
        observed[str(path.relative_to(ROOT))] = actual
    return dict(sorted(observed.items()))


def gpu_state() -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    name, total, used, free = [
        value.strip() for value in result.stdout.splitlines()[0].split(",")
    ]
    apps = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "name": name,
        "total_mib": int(total),
        "used_mib": int(used),
        "free_mib": int(free),
        "compute_apps": [
            line.strip() for line in apps.stdout.splitlines() if line.strip()
        ],
    }


class GpuMonitor:
    def __init__(self) -> None:
        self.phase = "preflight"
        self.total_mib = 0
        self.peaks: dict[str, int] = defaultdict(int)
        self.sample_count = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self.phase = phase

    def _run(self) -> None:
        while not self._stop.is_set():
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                total, used = (
                    int(value.strip())
                    for value in result.stdout.splitlines()[0].split(",")
                )
                with self._lock:
                    self.total_mib = total
                    self.peaks[self.phase] = max(
                        self.peaks[self.phase],
                        used,
                    )
                    self.sample_count += 1
            self._stop.wait(0.05)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_mib": self.total_mib,
                "sample_count": self.sample_count,
                "phase_peak_used_mib": dict(sorted(self.peaks.items())),
                "global_peak_used_mib": max(self.peaks.values(), default=0),
            }


def spec_and_claims(
    case: dict[str, Any],
) -> tuple[IntentResponseSpec, tuple[EvidenceClaim, ...]]:
    data = case["intent_response_spec"]
    spec = IntentResponseSpec(
        prompt_id=case["id"],
        intent=case["intent"],
        target_entities=tuple(case["target_entities"]),
        query_anchor_tokens=tuple(data["query_anchor_tokens"]),
        query_attribute_anchors=tuple(data["query_attribute_anchors"]),
        minimum_supported_claims=data["minimum_supported_claims"],
        decision_tokens=tuple(data["decision_tokens"]),
        communication_action_tokens=tuple(
            data["communication_action_tokens"]
        ),
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
    return spec, claims


def build_prompt(case: dict[str, Any]) -> list[dict[str, str]]:
    evidence = "\n".join(
        (
            f"- {item['canonical_entity']} | "
            f"{item['attribute_aliases'][0]} | "
            f"{item['canonical_value']} | "
            f"source={item['source_locator']}"
        )
        for item in case["evidence_claims"]
    )
    return [
        {
            "role": "system",
            "content": (
                "你是只基于已执行工具 observation 生成终答的新能源汽车助手。"
                "直接输出终答，不调用工具，不使用模型记忆补充事实。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"用户问题：{case['query']}\n"
                f"冻结证据：\n{evidence}\n"
                f"{case['generation_instruction']}"
            ),
        },
    ]


def numeric_stats(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "population_variance": statistics.pvariance(values),
        "population_std": statistics.pstdev(values),
        "sample_variance": statistics.variance(values),
        "sample_std": statistics.stdev(values),
        "distinct_values": sorted(set(values)),
    }


def aggregate_groups(groups: list[dict[str, Any]]) -> dict[str, Any]:
    nonzero = [
        group
        for group in groups
        if group["reward_distribution"]["sample_variance"] > 0.0
    ]
    all_one = [group for group in groups if group["all_reward_one"]]
    all_zero = [group for group in groups if group["all_reward_zero"]]
    component_summary: dict[str, Any] = {}
    for name in COMPONENTS:
        stds = [
            group["component_distributions"][name]["sample_std"]
            for group in groups
        ]
        variable = [value for value in stds if value > 0.0]
        all_values = [
            value
            for group in groups
            for value in group["component_distributions"][name]["values"]
        ]
        component_summary[name] = {
            "nonzero_variance_groups": len(variable),
            "nonzero_variance_group_ratio": len(variable) / len(groups),
            "group_sample_std_mean": statistics.fmean(stds),
            "group_sample_std_min": min(stds),
            "group_sample_std_max": max(stds),
            "completion_value_mean": statistics.fmean(all_values),
            "completion_value_min": min(all_values),
            "completion_value_max": max(all_values),
        }
    return {
        "groups": len(groups),
        "nonzero_variance_groups": len(nonzero),
        "nonzero_variance_group_ratio": len(nonzero) / len(groups),
        "all_one_groups": len(all_one),
        "all_one_group_ratio": len(all_one) / len(groups),
        "all_zero_groups": len(all_zero),
        "all_zero_group_ratio": len(all_zero) / len(groups),
        "group_sample_std_mean": statistics.fmean(
            group["reward_distribution"]["sample_std"] for group in groups
        ),
        "group_sample_std_min": min(
            group["reward_distribution"]["sample_std"] for group in groups
        ),
        "group_sample_std_max": max(
            group["reward_distribution"]["sample_std"] for group in groups
        ),
        "component_group_distributions": component_summary,
    }


def validate_outputs_absent() -> None:
    existing = [
        path
        for path in (
            RAW_PATH,
            RAW_SHA_PATH,
            REPORT_PATH,
            REPORT_SHA_PATH,
            FAILURE_PATH,
        )
        if path.exists()
    ]
    if existing:
        raise FileExistsError(f"refusing probe overwrite: {existing}")


def main() -> int:
    validate_outputs_absent()
    if Path(sys.prefix).resolve() != (ROOT / ".venv-grpo").resolve():
        raise RuntimeError("signal probe must run from isolated .venv-grpo")
    if sha256_file(INPUT_PATH) != EXPECTED_INPUT_SHA256:
        raise ValueError("probe input manifest SHA drift")
    recorded_sha, recorded_name = INPUT_SHA_PATH.read_text(
        encoding="ascii"
    ).strip().split(maxsplit=1)
    if (
        recorded_sha != EXPECTED_INPUT_SHA256
        or recorded_name != INPUT_PATH.name
    ):
        raise ValueError("probe input companion SHA drift")

    manifest = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    before = {
        "frozen_sha256": frozen_snapshot(manifest),
        "venv_train": venv_train_snapshot(),
        "formal_entry": formal_entry_status(),
        "gpu": gpu_state(),
    }
    if not before["formal_entry"]["fail_closed"]:
        raise RuntimeError("formal GRPO entry is not fail closed")
    if before["gpu"]["used_mib"] != 0 or before["gpu"]["compute_apps"]:
        raise RuntimeError(f"GPU is not idle: {before['gpu']}")

    monitor = GpuMonitor()
    monitor.start()
    completed_groups: list[dict[str, Any]] = []
    started = time.time()
    raw_handle = None
    torch = None
    try:
        actual_versions = {
            package: version(package) for package in PINNED_VERSIONS
        }
        if actual_versions != PINNED_VERSIONS:
            raise RuntimeError(f"runtime version drift: {actual_versions}")

        import torch as torch_module
        from peft import PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        torch = torch_module
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        if (
            torch.cuda.get_device_name(0) != "Tesla V100-SXM2-32GB"
            or torch.cuda.get_device_capability(0) != (7, 0)
        ):
            raise RuntimeError("unexpected GPU target")
        torch.manual_seed(manifest["sampling_config"]["seed"])
        torch.cuda.manual_seed_all(manifest["sampling_config"]["seed"])
        torch.set_grad_enabled(False)
        monitor.set_phase("model_load")
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_PATH),
            local_files_only=True,
            trust_remote_code=False,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        base_model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_PATH),
            quantization_config=quantization,
            device_map={"": 0},
            torch_dtype=torch.float16,
            local_files_only=True,
            trust_remote_code=False,
        )
        model = PeftModel.from_pretrained(
            base_model,
            str(ADAPTER_PATH),
            is_trainable=False,
        )
        model.eval()
        model.config.use_cache = True
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError("probe model unexpectedly has trainable parameters")

        config = manifest["sampling_config"]
        eos_token_id = model.generation_config.eos_token_id
        raw_handle = RAW_PATH.open("x", encoding="utf-8")
        for index, case in enumerate(manifest["cases"], start=1):
            spec, claims = spec_and_claims(case)
            messages = build_prompt(case)
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(
                rendered,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_tokens = int(inputs["input_ids"].shape[1])
            if prompt_tokens > config["max_prompt_length"]:
                raise RuntimeError(
                    f"{case['id']}: prompt length {prompt_tokens} exceeds limit"
                )
            inputs = {
                key: value.to(model.device) for key, value in inputs.items()
            }
            phase = f"rollout:{case['id']}"
            monitor.set_phase(phase)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            case_started = time.perf_counter()
            with torch.inference_mode(), torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                generated = model.generate(
                    **inputs,
                    do_sample=config["do_sample"],
                    temperature=config["temperature"],
                    top_p=config["top_p"],
                    num_return_sequences=config["num_generations"],
                    max_new_tokens=config["max_completion_length"],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=eos_token_id,
                    use_cache=True,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - case_started
            max_total_tokens = int(generated.shape[1])
            if max_total_tokens > config["max_total_sequence_length"]:
                raise RuntimeError(
                    f"{case['id']}: total length {max_total_tokens} exceeds limit"
                )
            completion_ids = generated[:, prompt_tokens:]
            texts = tokenizer.batch_decode(
                completion_ids,
                skip_special_tokens=True,
            )
            if len(texts) != 8:
                raise RuntimeError(f"{case['id']}: expected 8 completions")

            completion_results: list[dict[str, Any]] = []
            for completion_index, (text, token_ids) in enumerate(
                zip(texts, completion_ids),
                start=1,
            ):
                score = score_grounded_answer(
                    answer=text,
                    spec=spec,
                    evidence_claims=claims,
                )
                values = {
                    "total": float(score.total),
                    "factual_precision": float(score.factual_precision),
                    "required_coverage": float(score.required_coverage),
                    "source_integrity": float(score.source_integrity),
                    "concision": float(score.concision),
                }
                if not all(math.isfinite(value) for value in values.values()):
                    raise RuntimeError(f"{case['id']}: non-finite reward")
                completion_results.append(
                    {
                        "index": completion_index,
                        "text": text,
                        "text_sha256": hashlib.sha256(
                            text.encode("utf-8")
                        ).hexdigest(),
                        "completion_tokens": int(
                            (token_ids != tokenizer.pad_token_id).sum().item()
                        ),
                        "gate": score.gate,
                        "matched_claim_count": score.matched_claim_count,
                        **values,
                    }
                )

            rewards = [item["total"] for item in completion_results]
            component_distributions = {
                name: numeric_stats(
                    [item[name] for item in completion_results]
                )
                for name in COMPONENTS
            }
            torch_peak_reserved = (
                torch.cuda.max_memory_reserved() / 1024**2
            )
            time.sleep(0.1)
            nvidia_peak = monitor.snapshot()["phase_peak_used_mib"].get(
                phase,
                0,
            )
            group = {
                "id": case["id"],
                "split": case["split"],
                "intent": case["intent"],
                "target_count": len(case["target_entities"]),
                "known_reward_contract_risks": case[
                    "known_reward_contract_risks"
                ],
                "prompt_tokens": prompt_tokens,
                "max_total_tokens": max_total_tokens,
                "elapsed_sec": round(elapsed, 3),
                "rollout_peak_used_mib": round(
                    max(torch_peak_reserved, nvidia_peak),
                    3,
                ),
                "reward_distribution": numeric_stats(rewards),
                "all_reward_one": all(value == 1.0 for value in rewards),
                "all_reward_zero": all(value == 0.0 for value in rewards),
                "component_distributions": component_distributions,
                "gate_counts": dict(
                    sorted(Counter(item["gate"] for item in completion_results).items())
                ),
                "completions": completion_results,
            }
            raw_handle.write(
                json.dumps(group, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
            completed_groups.append(group)
            print(
                json.dumps(
                    {
                        "completed": f"{index}/20",
                        "id": case["id"],
                        "sample_std": group["reward_distribution"]["sample_std"],
                        "all_one": group["all_reward_one"],
                        "all_zero": group["all_reward_zero"],
                        "peak_mib": group["rollout_peak_used_mib"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        raw_handle.close()
        raw_handle = None
        write_sha(RAW_SHA_PATH, RAW_PATH)
        monitor.set_phase("cleanup")
        del model
        del base_model
        torch.cuda.empty_cache()
        time.sleep(0.2)
        monitor.stop()

        after = {
            "frozen_sha256": frozen_snapshot(manifest),
            "venv_train": venv_train_snapshot(),
            "formal_entry": formal_entry_status(),
            "gpu": gpu_state(),
        }
        zero_change = {
            "frozen_sha256": (
                before["frozen_sha256"] == after["frozen_sha256"]
            ),
            "venv_train": before["venv_train"] == after["venv_train"],
            "formal_entry_fail_closed": (
                before["formal_entry"]["fail_closed"]
                and after["formal_entry"]["fail_closed"]
                and before["formal_entry"]["sha256"]
                == after["formal_entry"]["sha256"]
            ),
            "gpu_released": after["gpu"]["used_mib"] == 0,
        }
        overall = aggregate_groups(completed_groups)
        by_intent = {
            intent: aggregate_groups(
                [group for group in completed_groups if group["intent"] == intent]
            )
            for intent in ("recommend", "compare", "knowledge", "sales")
        }
        ratio = overall["nonzero_variance_group_ratio"]
        lower = manifest["fire_gate"][
            "nonzero_variance_group_ratio_lower_bound"
        ]
        clear = manifest["fire_gate"][
            "nonzero_variance_group_ratio_clear_pass"
        ]
        if ratio >= clear:
            verdict = "clear_pass_signal_present"
        elif ratio >= lower:
            verdict = "lower_bound_pass_signal_present"
        else:
            verdict = "fail_insufficient_signal_do_not_ignite"
        memory = monitor.snapshot()
        cross_prompt_peak = max(
            group["rollout_peak_used_mib"] for group in completed_groups
        )
        report = {
            "status": "passed",
            "mode": "read_only_rollout_and_reward_only",
            "started_unix_sec": started,
            "completed_unix_sec": time.time(),
            "input_manifest": {
                "path": str(INPUT_PATH.relative_to(ROOT)),
                "sha256": EXPECTED_INPUT_SHA256,
            },
            "raw_results": {
                "path": str(RAW_PATH.relative_to(ROOT)),
                "sha256": sha256_file(RAW_PATH),
                "records": len(completed_groups),
                "completions": sum(
                    len(group["completions"]) for group in completed_groups
                ),
            },
            "sampling_config": manifest["sampling_config"],
            "execution_observed": {
                "rollout": True,
                "programmatic_reward": True,
                "autograd_enabled": torch.is_grad_enabled(),
                "trainable_parameters": 0,
                "backward": False,
                "optimizer_step": False,
                "checkpoint_saved": False,
                "beta_note": (
                    "beta=0.01 is recorded for formal parity but unused because "
                    "this probe computes no KL or training objective."
                ),
            },
            "learnable_signal": {
                "overall": overall,
                "by_intent": by_intent,
                "per_group": [
                    {
                        key: group[key]
                        for key in (
                            "id",
                            "split",
                            "intent",
                            "target_count",
                            "known_reward_contract_risks",
                            "prompt_tokens",
                            "max_total_tokens",
                            "elapsed_sec",
                            "rollout_peak_used_mib",
                            "reward_distribution",
                            "all_reward_one",
                            "all_reward_zero",
                            "component_distributions",
                            "gate_counts",
                        )
                    }
                    for group in completed_groups
                ],
            },
            "fire_gate": {
                **manifest["fire_gate"],
                "observed_nonzero_variance_group_ratio": ratio,
                "verdict": verdict,
                "automatic_training_started": False,
            },
            "memory": {
                **memory,
                "cross_prompt_rollout_peak_mib": cross_prompt_peak,
                "cross_prompt_rollout_peak_gib": cross_prompt_peak / 1024,
                "headroom_mib": memory["total_mib"] - cross_prompt_peak,
                "headroom_gib": (
                    memory["total_mib"] - cross_prompt_peak
                )
                / 1024,
            },
            "integrity": {
                "before": before,
                "after": after,
                "zero_change": zero_change,
                "all_zero_change_checks_passed": all(zero_change.values()),
                "reward_fn_sha256_before_and_after": manifest[
                    "frozen_sources"
                ]["training/grpo/reward_fn.py"],
            },
            "runtime_versions": actual_versions,
        }
        exclusive_json(REPORT_PATH, report)
        write_sha(REPORT_SHA_PATH, REPORT_PATH)
        return 0
    except Exception as error:
        if raw_handle is not None:
            raw_handle.close()
        if monitor._thread.is_alive():
            monitor.stop()
        exclusive_json(
            FAILURE_PATH,
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "completed_groups": len(completed_groups),
                "optimizer_step": False,
                "checkpoint_saved": False,
                "memory": monitor.snapshot(),
            },
        )
        return 1
    finally:
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
