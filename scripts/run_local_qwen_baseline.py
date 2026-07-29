#!/usr/bin/env python3
"""Run the local base Qwen model on held-out agent-contract cases.

This is a Phase 1 baseline runner, not an SFT entrypoint. It uses the existing
multi-turn evaluator, executes only model-selected production tools, and stores
raw protocol outputs plus an auditable contract-and-grounding summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from data_synth.generate_sft_data import audit_answer_grounding
from scripts import evaluate_model_outputs as evaluator
from scripts import run_model_layer_eval as runner
from training.sft import train_qlora_sft as sft


ROOT = Path(__file__).resolve().parents[1]
HELD_OUT_PATH = ROOT / "data" / "model_training" / "eval" / "held_out.jsonl"
REWARD_VISIBLE_PATH = (
    ROOT / "data" / "model_training" / "eval" / "reward_visible.jsonl"
)
DEFAULT_OUTPUT_PATH = (
    ROOT / "data" / "model_training" / "baseline_qwen_heldout_outputs.jsonl"
)
DEFAULT_REPORT_PATH = (
    ROOT / "data" / "model_training" / "baseline_qwen_heldout.json"
)
FROZEN_HARNESS_MANIFEST_PATH = (
    ROOT
    / "data"
    / "model_training"
    / "eval"
    / "frozen_qwen_heldout_harness.json"
)
LOCAL_MODEL_ALIAS = "local_base_nf4"
LOCAL_ENDPOINT_ALIAS = "https://local-transformers.invalid/v1"
FROZEN_HARNESS_VERSION = "qwen-heldout-contract-v1"
FROZEN_MAX_STEPS = 8
FROZEN_MAX_NEW_TOKENS = 512
TOOL_CALL_STOP_SEQUENCE = "</tool_call>"
TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(?P<payload>\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_frozen_harness_contract() -> dict[str, Any]:
    """Describe every setting that must match the base and adapted evaluations."""
    cases = evaluator.load_jsonl(HELD_OUT_PATH, label="held-out cases")
    return {
        "harness_version": FROZEN_HARNESS_VERSION,
        "evaluation": {
            "cases": str(HELD_OUT_PATH.relative_to(ROOT)),
            "case_count": len(cases),
            "cases_sha256": _sha256(HELD_OUT_PATH),
            "protocol_version": evaluator.PROTOCOL_VERSION,
            "reward_visible_included": False,
        },
        "runner": {
            "max_steps": FROZEN_MAX_STEPS,
        },
        "generation": {
            "max_new_tokens": FROZEN_MAX_NEW_TOKENS,
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "tool_call_stop_sequence": TOOL_CALL_STOP_SEQUENCE,
        },
        "tool_protocol": {
            "parser": "strict_qwen_xml_to_openai_function_call",
            "required_payload_keys": ["name", "arguments"],
            "real_tool_execution": True,
            "tool_observation_feedback": True,
        },
        "scoring": {
            "name": "contract_grounding_pass_rate",
            "required_checks": [
                "terminal_protocol",
                "mandatory_tool_order",
                "tool_argument_schema",
                "allowed_catalog_models",
                "no_declared_hallucinated_models",
                "grounding_audit",
            ],
        },
    }


def validate_frozen_harness_manifest(
    manifest_path: Path = FROZEN_HARNESS_MANIFEST_PATH,
) -> dict[str, Any]:
    """Fail closed if cases or any locked harness setting has drifted."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"frozen harness manifest not found: {manifest_path}")
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = build_frozen_harness_contract()
    if persisted != expected:
        raise ValueError("frozen held-out harness manifest does not match runtime")
    return persisted


def validate_frozen_runtime_settings(
    *,
    max_steps: int,
    max_new_tokens: int,
) -> None:
    """Reject caller overrides that would invalidate baseline comparability."""
    if max_steps != FROZEN_MAX_STEPS:
        raise ValueError(
            f"frozen harness requires max_steps={FROZEN_MAX_STEPS}"
        )
    if max_new_tokens != FROZEN_MAX_NEW_TOKENS:
        raise ValueError(
            "frozen harness requires "
            f"max_new_tokens={FROZEN_MAX_NEW_TOKENS}"
        )


def _normalize(value: str) -> str:
    return evaluator.normalize_query(value)


def validate_heldout_only(cases_path: Path) -> list[dict[str, Any]]:
    """Reject any evaluation source except the immutable held-out set."""
    cases_path = Path(cases_path)
    if cases_path.resolve() != HELD_OUT_PATH.resolve():
        raise ValueError("baseline evaluation must use the canonical held-out set")

    cases = evaluator.load_jsonl(cases_path, label="held-out cases")
    reward_cases = evaluator.load_jsonl(REWARD_VISIBLE_PATH, label="reward cases")
    held_out_ids = {_normalize(case["id"]) for case in cases}
    reward_ids = {_normalize(case["id"]) for case in reward_cases}
    held_out_queries = {_normalize(case["query"]) for case in cases}
    reward_queries = {_normalize(case["query"]) for case in reward_cases}
    if held_out_ids & reward_ids:
        raise ValueError("held-out/reward-visible ID overlap")
    if held_out_queries & reward_queries:
        raise ValueError("held-out/reward-visible query overlap")
    return cases


def parse_qwen_tool_calls(
    text: str,
    *,
    call_id_factory: Callable[[int], str],
) -> tuple[str, list[dict[str, Any]]]:
    """Convert only well-formed Qwen XML calls to strict OpenAI call objects."""
    if not isinstance(text, str):
        raise ValueError("generated Qwen content must be a string")
    matches = list(TOOL_CALL_PATTERN.finditer(text))
    if "<tool_call>" in text and not matches:
        return text, []
    calls: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        try:
            payload = json.loads(match.group("payload"))
        except json.JSONDecodeError:
            return text, []
        if not isinstance(payload, dict) or set(payload) != {"name", "arguments"}:
            return text, []
        name = payload["name"]
        arguments = payload["arguments"]
        if not isinstance(name, str) or not name.strip():
            return text, []
        if isinstance(arguments, str):
            try:
                decoded_arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return text, []
        else:
            decoded_arguments = arguments
        if not isinstance(decoded_arguments, dict):
            return text, []
        calls.append(
            {
                "id": call_id_factory(index),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(
                        decoded_arguments,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            }
        )
    residual = TOOL_CALL_PATTERN.sub("", text).strip()
    return residual, calls


class ToolCallBoundaryStopper:
    """Stop a local generation only after a complete Qwen tool-call turn."""

    def __init__(self, *, tokenizer: Any, prompt_token_count: int) -> None:
        self.tokenizer = tokenizer
        self.prompt_token_count = prompt_token_count

    def __call__(self, input_ids: Any, _scores: Any, **kwargs: Any) -> bool:
        del _scores
        kwargs.clear()
        generated = input_ids[0][self.prompt_token_count:]
        text = self.tokenizer.decode(generated, skip_special_tokens=False)
        return TOOL_CALL_STOP_SEQUENCE in text


class LocalQwenToolClient:
    """OpenAI-client-shaped adapter over local Qwen transformer inference."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        torch: Any,
        max_new_tokens: int,
        stopping_criteria_list: Any,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.stopping_criteria_list = stopping_criteria_list
        self._call_index = 0
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create),
        )

    def _next_call_id(self, tool_index: int) -> str:
        call_id = f"local-call-{self._call_index:04d}-{tool_index:02d}"
        return call_id

    def create(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **_ignored: Any,
    ) -> Any:
        _ignored.clear()
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)
        attention_mask = self.torch.ones_like(input_ids)
        stopper = ToolCallBoundaryStopper(
            tokenizer=self.tokenizer,
            prompt_token_count=int(input_ids.shape[1]),
        )
        with self.torch.inference_mode(), self.torch.autocast(
            device_type="cuda",
            dtype=self.torch.float16,
        ):
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                stopping_criteria=self.stopping_criteria_list([stopper]),
            )
        continuation = generated[0, input_ids.shape[1]:]
        text = self.tokenizer.decode(continuation, skip_special_tokens=True)
        generated_content, tool_calls = parse_qwen_tool_calls(
            text,
            call_id_factory=self._next_call_id,
        )
        self._call_index += 1
        message = SimpleNamespace(
            content=generated_content,
            tool_calls=tool_calls,
        )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    def close(self) -> None:
        del self.model
        self.torch.cuda.empty_cache()


def _load_local_client(
    *,
    config_path: Path,
    max_new_tokens: int,
) -> LocalQwenToolClient:
    config = sft.load_training_config(config_path)
    model_path = sft._model_path(config, ROOT)
    dependencies = sft._load_training_dependencies()
    torch = dependencies["torch"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for local baseline inference")
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
    model.config.use_cache = True
    model.eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    return LocalQwenToolClient(
        model=model,
        tokenizer=tokenizer,
        torch=torch,
        max_new_tokens=max_new_tokens,
        stopping_criteria_list=dependencies["StoppingCriteriaList"],
    )


def _grounding_errors(outputs: list[dict[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for output in outputs:
        record_id = output["id"]
        result[record_id] = audit_answer_grounding(
            {"messages": output["trajectory"]},
        )
    return result


def _case_passes_contract_and_grounding(
    result: dict[str, Any],
    grounding_errors: list[str],
) -> bool:
    has_tool_selection_error = any(
        failure.startswith("tool_selection:")
        for failure in result["failures"]
    )
    has_valid_arguments = (
        result["valid_argument_calls"] == result["total_tool_calls"]
    )
    return bool(
        result["case_success"]
        and not has_tool_selection_error
        and has_valid_arguments
        and result["recommendation_hit"] is True
        and not result["hallucinated_models"]
        and not grounding_errors
    )


def build_baseline_summary(
    *,
    cases: list[dict[str, Any]],
    evaluation: dict[str, Any],
    grounding_by_id: dict[str, list[str]],
    raw_outputs_path: Path,
) -> dict[str, Any]:
    """Combine established evaluator metrics with explicit grounding results."""
    case_by_id = {case["id"]: case for case in cases}
    per_intent: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "passed": 0},
    )
    rows: list[dict[str, Any]] = []
    passed = 0
    for result in evaluation["case_results"]:
        case_id = result["id"]
        case = case_by_id[case_id]
        grounding_errors = grounding_by_id[case_id]
        case_passed = _case_passes_contract_and_grounding(
            result,
            grounding_errors,
        )
        failure_reasons = list(result["failures"])
        failure_reasons.extend(
            f"grounding: {error}" for error in grounding_errors
        )
        per_intent[case["intent"]]["total"] += 1
        if case_passed:
            passed += 1
            per_intent[case["intent"]]["passed"] += 1
        rows.append(
            {
                "id": case_id,
                "intent": case["intent"],
                "passed": case_passed,
                "failure_reasons": failure_reasons,
                "actual_tools": result["actual_tools"],
                "recommendation_hit": result["recommendation_hit"],
                "grounding_errors": grounding_errors,
            }
        )

    total = len(cases)
    formatted_per_intent = {
        intent: {
            **counts,
            "percentage": round(counts["passed"] / counts["total"] * 100, 1),
        }
        for intent, counts in sorted(per_intent.items())
    }
    return {
        "status": "completed",
        "evaluation_set": "held_out_only",
        "reward_visible_included": False,
        "scoring_contract": (
            "A case passes only when the existing terminal protocol, mandatory "
            "tool order, tool arguments, allowed-catalog model constraint, and "
            "grounding audit all pass."
        ),
        "total_score": {
            "name": "contract_grounding_pass_rate",
            "numerator": passed,
            "denominator": total,
            "percentage": round(passed / total * 100, 1) if total else 0.0,
        },
        "per_intent": formatted_per_intent,
        "metrics": evaluation["metrics"],
        "cases": rows,
        "raw_outputs_path": str(raw_outputs_path),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_baseline(
    *,
    cases_path: Path = HELD_OUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    config_path: Path = sft.DEFAULT_CONFIG_PATH,
    max_steps: int = FROZEN_MAX_STEPS,
    max_new_tokens: int = FROZEN_MAX_NEW_TOKENS,
) -> dict[str, Any]:
    """Run the immutable held-out baseline and persist outputs plus score."""
    validate_frozen_runtime_settings(
        max_steps=max_steps,
        max_new_tokens=max_new_tokens,
    )
    validate_frozen_harness_manifest()
    if Path(cases_path).resolve() != HELD_OUT_PATH.resolve():
        raise ValueError("frozen harness requires the canonical held-out set")
    client = _load_local_client(
        config_path=config_path,
        max_new_tokens=max_new_tokens,
    )
    try:
        return run_frozen_heldout_harness(
            client=client,
            output_path=output_path,
            report_path=report_path,
            model_alias=LOCAL_MODEL_ALIAS,
        )
    finally:
        client.close()


def run_frozen_heldout_harness(
    *,
    client: Any,
    output_path: Path,
    report_path: Path,
    model_alias: str,
) -> dict[str, Any]:
    """Evaluate any local base or adapted model with the frozen held-out contract."""
    validate_frozen_harness_manifest()
    cases = validate_heldout_only(HELD_OUT_PATH)
    runner.run_evaluation(
        cases_path=HELD_OUT_PATH,
        output_path=output_path,
        model=model_alias,
        base_url=LOCAL_ENDPOINT_ALIAS,
        client=client,
        max_steps=FROZEN_MAX_STEPS,
    )
    outputs = evaluator.load_jsonl(output_path, label="frozen harness outputs")
    evaluation = evaluator.evaluate_records(
        cases,
        outputs,
        vehicle_catalog=set(evaluator.load_vehicle_catalog()),
    )
    summary = build_baseline_summary(
        cases=cases,
        evaluation=evaluation,
        grounding_by_id=_grounding_errors(outputs),
        raw_outputs_path=output_path,
    )
    _write_json(report_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run local base Qwen on immutable held-out contract cases.",
    )
    parser.add_argument("--cases", type=Path, default=HELD_OUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--config", type=Path, default=sft.DEFAULT_CONFIG_PATH)
    parser.add_argument("--max-steps", type=int, default=FROZEN_MAX_STEPS)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=FROZEN_MAX_NEW_TOKENS,
    )
    args = parser.parse_args()
    result = run_baseline(
        cases_path=args.cases,
        output_path=args.output,
        report_path=args.report,
        config_path=args.config,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
    )
    print(
        json.dumps(
            {
                "total_score": result["total_score"],
                "per_intent": result["per_intent"],
                "report": str(args.report),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
