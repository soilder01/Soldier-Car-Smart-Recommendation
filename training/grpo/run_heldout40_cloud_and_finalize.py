#!/usr/bin/env python3
"""Run held-out-40 cloud recomputation and write final artifacts."""

from __future__ import annotations

import json
import os
import ssl
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path
from typing import Any

from training.grpo.reward_fn import score_grounded_answer
from training.grpo.run_signal_probe import build_prompt, spec_and_claims


ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "data" / "model_training" / "grpo" / "formal_v4" / "restart_1"
DATASET = BASE_DIR / "held_out_40_frozen_eval.jsonl"
PROTOCOL = BASE_DIR / "held_out_40_final_protocol.json"
LOCAL_CACHE = Path(os.environ.get("HELDOUT40_LOCAL_EVAL_CACHE", "/tmp/heldout40_local_eval_cache.json"))
OUT_EVAL = BASE_DIR / "held_out_40_final_evaluations.jsonl"
OUT_RESULT = BASE_DIR / "held_out_40_final_result.json"
OUT_STATE = BASE_DIR / "held_out_40_access_state.json"
REWARD_FN = ROOT / "training" / "grpo" / "reward_fn.py"

EXPECTED_REWARD_SHA = "325ad44feb83ec37c35babfed4bddb928cf400788e07735eb4631fc4af6962c8"
EXPECTED_DATASET_SHA = "2b4a2e8dff52b9feee12bb451ce630dc0e03661c1e4b8935baf3a78df77013ea"
EXPECTED_PROTOCOL_SHA = "eb5d6a488205666414dd7b5c3484555a52c7789de958badf572108994662531d"
DEFAULT_ARK_BASE_URL = os.environ.get("ARK_BASE_URL", "<ARK_BASE_URL_PLACEHOLDER>")


def sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_sha_sidecar(artifact: Path) -> None:
    with Path(str(artifact) + ".sha256").open("x", encoding="ascii") as handle:
        handle.write(f"{sha256_file(artifact)}  {artifact.name}\n")


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def load_dotenv() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def cloud_config() -> dict[str, str]:
    load_dotenv()
    api_key = os.getenv("CHAT_API_KEY") or os.getenv("ARK_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    base_url = os.getenv("CHAT_BASE_URL") or os.getenv("ARK_BASE_URL") or os.getenv("OPENAI_BASE_URL") or ""
    model = os.getenv("CHAT_MODEL") or os.getenv("ARK_CHAT_MODEL") or os.getenv("OPENAI_CHAT_MODEL") or ""
    if not base_url and os.getenv("SEEDPRO_EP") and os.getenv("ARK_API_KEY"):
        base_url = DEFAULT_ARK_BASE_URL
    if not model and os.getenv("SEEDPRO_EP"):
        model = os.getenv("SEEDPRO_EP", "")
    if not (api_key and base_url and model):
        raise RuntimeError("missing cloud configuration triplet")
    return {"api_key": api_key, "base_url": base_url.rstrip("/"), "model": model}


def assert_preflight() -> tuple[dict[str, str], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    for path in (
        OUT_EVAL,
        OUT_RESULT,
        OUT_STATE,
        Path(str(OUT_EVAL) + ".sha256"),
        Path(str(OUT_RESULT) + ".sha256"),
        Path(str(OUT_STATE) + ".sha256"),
    ):
        if path.exists():
            raise FileExistsError(f"refusing overwrite: {path}")
    expected = {
        "training/grpo/reward_fn.py": EXPECTED_REWARD_SHA,
        "data/model_training/grpo/formal_v4/restart_1/held_out_40_frozen_eval.jsonl": EXPECTED_DATASET_SHA,
        "data/model_training/grpo/formal_v4/restart_1/held_out_40_final_protocol.json": EXPECTED_PROTOCOL_SHA,
    }
    observed = {}
    for relative, digest in expected.items():
        actual = sha256_file(ROOT / relative)
        if actual != digest:
            raise RuntimeError(f"SHA mismatch: {relative}")
        observed[relative] = actual
    if not LOCAL_CACHE.exists():
        raise FileNotFoundError(f"missing local eval cache: {LOCAL_CACHE}")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    cases = load_jsonl(DATASET)
    local_payload = json.loads(LOCAL_CACHE.read_text(encoding="utf-8"))
    if local_payload.get("status") != "heldout40_local_eval_completed":
        raise RuntimeError("local eval cache status mismatch")
    if len(local_payload.get("rows", [])) != 2:
        raise RuntimeError("local eval cache row count mismatch")
    return observed, protocol, cases, local_payload


def post_json(url: str, api_key: str, body: dict[str, Any], timeout: int = 180) -> dict[str, Any]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    context = ssl.create_default_context()
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        return json.loads(response.read().decode("utf-8"))


def call_cloud_once(
    *,
    config: dict[str, str],
    case: dict[str, Any],
    sampling: dict[str, Any],
) -> tuple[str, Any, str | None]:
    url = f"{config['base_url']}/chat/completions"
    body = {
        "model": config["model"],
        "messages": build_prompt(case),
        "temperature": float(sampling["temperature"]),
        "top_p": float(sampling["top_p"]),
        "max_tokens": int(sampling["max_completion_length"]),
        "n": 1,
        "seed": int(sampling["seed"]),
    }
    last_error = None
    for attempt in range(3):
        try:
            payload = post_json(url, config["api_key"], body)
            choices = payload.get("choices") or []
            if len(choices) != 1:
                raise RuntimeError(f"cloud returned {len(choices)} choices")
            message = choices[0].get("message") or {}
            answer = message.get("content") or ""
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("cloud returned empty content")
            finish_reason = choices[0].get("finish_reason")
            spec, claims = spec_and_claims(case)
            score = score_grounded_answer(answer=answer, spec=spec, evidence_claims=claims)
            return case["id"], score, str(finish_reason) if finish_reason is not None else None
        except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            last_error = type(exc).__name__
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"cloud call failed after retries: {case['id']} {last_error}")


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


def build_cloud_row(config: dict[str, str], protocol: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    sampling = protocol["sampling"]
    grouped: dict[str, list[Any]] = {case["id"]: [] for case in cases}
    finish_reasons: dict[str, Counter[str]] = {case["id"]: Counter() for case in cases}
    tasks = [(case, sample_index) for case in cases for sample_index in range(int(sampling["num_generations"]))]
    started = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [
            executor.submit(call_cloud_once, config=config, case=case, sampling=sampling)
            for case, _sample_index in tasks
        ]
        for future in as_completed(futures):
            case_id, score, finish_reason = future.result()
            grouped[case_id].append(score)
            if finish_reason is not None:
                finish_reasons[case_id][finish_reason] += 1
            completed += 1
            if completed % 40 == 0 or completed == len(tasks):
                complete_prompts = sum(1 for scores in grouped.values() if len(scores) == int(sampling["num_generations"]))
                print(
                    f"cloud_heldout_progress calls={completed}/{len(tasks)} complete_prompts={complete_prompts}/{len(cases)}",
                    flush=True,
                )
    per_prompt = []
    for case in cases:
        scores = grouped[case["id"]]
        if len(scores) != int(sampling["num_generations"]):
            raise RuntimeError(f"{case['id']}: completion count mismatch")
        fp = [float(score.factual_precision) for score in scores]
        cov = [float(score.required_coverage) for score in scores]
        src = [float(score.source_integrity) for score in scores]
        con = [float(score.concision) for score in scores]
        reward = [float(score.total) for score in scores]
        core = [0.6 * f + 0.4 * c for f, c in zip(fp, cov)]
        per_prompt.append(
            {
                "prompt_id": case["id"],
                "query_summary": case["query"][:80],
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
                "finish_reason_counts": dict(sorted(finish_reasons[case["id"]].items())),
            }
        )
    by_intent_all = {
        intent: summarize(
            [record for record in per_prompt if record["intent"] == intent],
            sum(1 for case in cases if case["intent"] == intent),
        )
        for intent in ("recommend", "sales", "compare", "knowledge")
    }
    by_intent = {"recommend": by_intent_all["recommend"], "sales": by_intent_all["sales"]}
    composite = 0.5 * by_intent["recommend"]["mean_core"] + 0.5 * by_intent["sales"]["mean_core"]
    return {
        "label": "cloud_seedpro_ark_ep_masked",
        "object_type": "cloud_openai_compatible",
        "provider_masked": "ark_seedpro_ep_masked",
        "client": "stdlib_urllib_openai_compatible",
        "sampling_config": {
            "temperature": sampling["temperature"],
            "top_p": sampling["top_p"],
            "max_completion_length": sampling["max_completion_length"],
            "num_generations": sampling["num_generations"],
            "seed": sampling["seed"],
        },
        "cloud_sampling_mechanism": {
            "calls_per_prompt": 8,
            "n_per_call": 1,
            "global_concurrency": 16,
            "seed_parameter_sent": int(sampling["seed"]),
            "total_cloud_calls": len(tasks),
        },
        "raw_answers_persisted": False,
        "by_intent": by_intent,
        "by_intent_all": by_intent_all,
        "composite": composite,
        "composite_recommend_sales_core": composite,
        "case_results": per_prompt,
        "completed_unix_sec": time.time(),
        "elapsed_sec": time.time() - started,
    }


def score_row(row: dict[str, Any]) -> dict[str, float]:
    return {
        "recommend": float(row["by_intent"]["recommend"]["mean_core"]),
        "sales": float(row["by_intent"]["sales"]["mean_core"]),
        "composite": float(row["composite"]),
    }


def delta(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {key: left[key] - right[key] for key in ("recommend", "sales", "composite")}


def main() -> int:
    observed, protocol, cases, local_payload = assert_preflight()
    config = cloud_config()
    cloud_row = build_cloud_row(config, protocol, cases)
    local_rows = local_payload["rows"]
    rows = [local_rows[0], local_rows[1], cloud_row]
    write_jsonl_exclusive(OUT_EVAL, rows)
    write_sha_sidecar(OUT_EVAL)

    by_label = {row["label"]: row for row in rows}
    local_score = score_row(by_label["sales_dense_v2_checkpoint_150"])
    ckpt300_score = score_row(by_label["checkpoint-300"])
    cloud_score = score_row(by_label["cloud_seedpro_ark_ep_masked"])
    result = {
        "status": "completed_held_out_40_terminal_evaluation",
        "redline": {
            "reward_fn_sha256": sha256_file(REWARD_FN),
            "held_out_40_accessed_before": False,
            "held_out_40_accessed_after": True,
            "frozen_inputs_read_only": True,
            "raw_answers_persisted": False,
            "cloud_recomputed_on_held_out_40": True,
            "cloud_secret_or_base_url_printed": False,
        },
        "scores": {
            "sales_dense_v2_checkpoint_150": local_score,
            "checkpoint-300": ckpt300_score,
            "cloud_seedpro_ark_ep_masked": cloud_score,
        },
        "deltas": {
            "sales_dense_v2_checkpoint_150_minus_cloud": delta(local_score, cloud_score),
            "sales_dense_v2_checkpoint_150_minus_checkpoint_300": delta(local_score, ckpt300_score),
        },
        "dataset": protocol["dataset"],
        "protocol_sha256": sha256_file(PROTOCOL),
        "observed_hashes": observed,
        "artifacts": {
            "evaluations_jsonl": str(OUT_EVAL.relative_to(ROOT)),
            "evaluations_sha256": sha256_file(OUT_EVAL),
            "result_json": str(OUT_RESULT.relative_to(ROOT)),
        },
    }
    write_json_exclusive(OUT_RESULT, result)
    write_sha_sidecar(OUT_RESULT)
    state = {
        "held_out_40_accessed": True,
        "access_scope": "single terminal held-out-40 evaluation",
        "evaluations_jsonl": str(OUT_EVAL.relative_to(ROOT)),
        "evaluations_sha256": sha256_file(OUT_EVAL),
        "result_json": str(OUT_RESULT.relative_to(ROOT)),
        "result_sha256": sha256_file(OUT_RESULT),
        "raw_answers_persisted": False,
        "cloud_secret_or_base_url_printed": False,
        "completed_unix_sec": time.time(),
    }
    write_json_exclusive(OUT_STATE, state)
    write_sha_sidecar(OUT_STATE)
    print(
        json.dumps(
            {
                "status": result["status"],
                "scores": result["scores"],
                "deltas": result["deltas"],
                "evaluation_sha256": sha256_file(OUT_EVAL),
                "result_sha256": sha256_file(OUT_RESULT),
                "state_sha256": sha256_file(OUT_STATE),
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
