#!/usr/bin/env python3
"""Retest held-out-40 through the local OpenAI-compatible endpoint."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from training.grpo.reward_fn import score_grounded_answer
from training.grpo.run_signal_probe import build_prompt, spec_and_claims


ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "data" / "model_training" / "grpo" / "formal_v4" / "restart_1"
DATASET = BASE_DIR / "held_out_40_frozen_eval.jsonl"
PROTOCOL = BASE_DIR / "held_out_40_final_protocol.json"
REWARD_FN = ROOT / "training" / "grpo" / "reward_fn.py"
ENDPOINT_BASE_URL = os.environ.get("HELDOUT40_ENDPOINT_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
ENDPOINT_MODEL = os.environ.get("HELDOUT40_ENDPOINT_MODEL", "car-7b")
GREEDY = os.getenv("HELDOUT40_GREEDY", "") == "1"
OUT_PATH = Path(
    os.environ.get(
        "HELDOUT40_ENDPOINT_RETEST_OUT",
        BASE_DIR
        / (
            f"heldout40_greedy_endpoint_{int(time.time())}.json"
            if GREEDY
            else f"heldout40_endpoint_retest_{int(time.time())}.json"
        ),
    )
)

EXPECTED_REWARD_SHA = "325ad44feb83ec37c35babfed4bddb928cf400788e07735eb4631fc4af6962c8"
EXPECTED_DATASET_SHA = "2b4a2e8dff52b9feee12bb451ce630dc0e03661c1e4b8935baf3a78df77013ea"
EXPECTED_PROTOCOL_SHA = "eb5d6a488205666414dd7b5c3484555a52c7789de958badf572108994662531d"

TRAINING_BASELINE = {
    "recommend": 0.8458035714285714,
    "sales": 0.65625,
    "composite": 0.7510267857142857,
}
CLOUD_BASELINE = {
    "recommend": 0.5475,
    "sales": 0.579375,
    "composite": 0.5634375,
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_sha_sidecar(path: Path) -> None:
    with Path(str(path) + ".sha256").open("x", encoding="ascii") as handle:
        handle.write(f"{sha256_file(path)}  {path.name}\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": mean(values), "min": min(values), "max": max(values)}


def post_chat(body: dict[str, Any], timeout: int = 300) -> dict[str, Any]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{ENDPOINT_BASE_URL}/chat/completions",
        data=data,
        headers={
            "Authorization": "Bearer dummy",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def summarize_scores(case: dict[str, Any], scores: list[Any], usage: dict[str, Any], elapsed: float) -> dict[str, Any]:
    fp = [float(score.factual_precision) for score in scores]
    cov = [float(score.required_coverage) for score in scores]
    src = [float(score.source_integrity) for score in scores]
    con = [float(score.concision) for score in scores]
    reward = [float(score.total) for score in scores]
    core = [0.6 * f + 0.4 * c for f, c in zip(fp, cov)]
    return {
        "prompt_id": case["id"],
        "query_summary": case["query"][:80],
        "intent": case["intent"],
        "candidate_index": case["candidate_index"],
        "target_entities": case["target_entities"],
        "completion_count": len(scores),
        "per_prompt_core_mean": mean(core),
        "factual_precision_mean": mean(fp),
        "required_coverage_mean": mean(cov),
        "reward_mean": mean(reward),
        "source_integrity_mean": mean(src),
        "concision_mean": mean(con),
        "gate_counts": dict(sorted(Counter(score.gate for score in scores).items())),
        "endpoint_usage": usage,
        "request_time_sec": elapsed,
    }


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
        "per_prompt_core_distribution": distribution([record["per_prompt_core_mean"] for record in records]),
        "gate_counts_total": dict(sorted(sum((Counter(record["gate_counts"]) for record in records), Counter()).items())),
    }


def assert_preflight() -> tuple[dict[str, str], dict[str, Any], list[dict[str, Any]]]:
    if OUT_PATH.exists() or Path(str(OUT_PATH) + ".sha256").exists():
        raise FileExistsError(f"refusing overwrite: {OUT_PATH}")
    expected = {
        "training/grpo/reward_fn.py": EXPECTED_REWARD_SHA,
        "data/model_training/grpo/formal_v4/restart_1/held_out_40_frozen_eval.jsonl": EXPECTED_DATASET_SHA,
        "data/model_training/grpo/formal_v4/restart_1/held_out_40_final_protocol.json": EXPECTED_PROTOCOL_SHA,
    }
    observed: dict[str, str] = {}
    for relative, digest in expected.items():
        actual = sha256_file(ROOT / relative)
        if actual != digest:
            raise RuntimeError(f"SHA mismatch: {relative}")
        observed[relative] = actual
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    cases = load_jsonl(DATASET)
    if len(cases) != 40 or Counter(case["intent"] for case in cases) != {
        "recommend": 10,
        "sales": 10,
        "compare": 10,
        "knowledge": 10,
    }:
        raise RuntimeError("held-out frozen dataset count drift")
    return observed, protocol, cases


def evaluate_cases(protocol: dict[str, Any], cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sampling = protocol["sampling"]
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        spec, claims = spec_and_claims(case)
        expected_n = 1 if GREEDY else int(sampling["num_generations"])
        body = {
            "model": ENDPOINT_MODEL,
            "messages": build_prompt(case),
            "temperature": 0 if GREEDY else float(sampling["temperature"]),
            "max_tokens": int(sampling["max_completion_length"]),
            "n": expected_n,
        }
        if not GREEDY:
            body["top_p"] = float(sampling["top_p"])
            body["seed"] = int(sampling["seed"])
        started = time.time()
        payload = post_chat(body)
        elapsed = time.time() - started
        choices = payload.get("choices") or []
        if len(choices) != expected_n:
            raise RuntimeError(f"{case['id']}: endpoint returned {len(choices)} choices")
        scores = []
        for choice in choices:
            message = choice.get("message") or {}
            answer = message.get("content") or ""
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError(f"{case['id']}: empty endpoint answer")
            scores.append(score_grounded_answer(answer=answer, spec=spec, evidence_claims=claims))
        rows.append(summarize_scores(case, scores, payload.get("usage") or {}, elapsed))
        print(
            f"endpoint_retest_progress prompts={index}/{len(cases)} "
            f"prompt_id={case['id']} intent={case['intent']} "
            f"mean_core={rows[-1]['per_prompt_core_mean']:.6f} elapsed_sec={elapsed:.1f}",
            flush=True,
        )
    return rows


def main() -> int:
    observed, protocol, cases = assert_preflight()
    started = time.time()
    case_results = evaluate_cases(protocol, cases)
    by_intent_all = {
        intent: summarize_intent(
            [record for record in case_results if record["intent"] == intent],
            sum(1 for case in cases if case["intent"] == intent),
        )
        for intent in ("recommend", "sales", "compare", "knowledge")
    }
    by_intent = {
        "recommend": by_intent_all["recommend"],
        "sales": by_intent_all["sales"],
    }
    scores = {
        "recommend": by_intent["recommend"]["mean_core"],
        "sales": by_intent["sales"]["mean_core"],
    }
    scores["composite"] = 0.5 * scores["recommend"] + 0.5 * scores["sales"]
    deltas_vs_training = {key: scores[key] - TRAINING_BASELINE[key] for key in TRAINING_BASELINE}
    deltas_vs_cloud = {key: scores[key] - CLOUD_BASELINE[key] for key in CLOUD_BASELINE}
    result = {
        "status": "completed_heldout40_endpoint_retest",
        "started_unix_sec": started,
        "completed_unix_sec": time.time(),
        "endpoint": {
            "base_url": ENDPOINT_BASE_URL,
            "model": ENDPOINT_MODEL,
        },
        "seed_semantics": {
            "implementation": "serve_local sets random, numpy, torch, and torch.cuda seeds once before each endpoint request generation",
            "scope": "per_prompt_request_reset",
            "limitation": "does not reproduce the original adapter-local evaluation's single RNG stream consumed sequentially across 40 prompts",
            "success_band": "per-intent absolute delta <= 0.02-0.03 is treated as sampling-noise reproduction",
        },
        "sampling_config": protocol["sampling"] | {
            "endpoint_mapping": (
                "one chat/completions request per prompt with n=1 greedy choice"
                if GREEDY
                else "one chat/completions request per prompt with n=num_generations choices"
            ),
            "greedy": GREEDY,
            "do_sample": not GREEDY,
            "endpoint_n": 1 if GREEDY else int(protocol["sampling"]["num_generations"]),
            "endpoint_temperature": 0 if GREEDY else float(protocol["sampling"]["temperature"]),
        },
        "observed_hashes": observed,
        "raw_answers_persisted": False,
        "terminal_artifacts_overwritten": False,
        "access_state_updated": False,
        "by_intent": by_intent,
        "by_intent_all": by_intent_all,
        "scores": scores,
        "training_baseline": TRAINING_BASELINE,
        "cloud_baseline": CLOUD_BASELINE,
        "delta_vs_training_baseline": deltas_vs_training,
        "delta_vs_cloud_baseline": deltas_vs_cloud,
        "case_results": case_results,
    }
    with OUT_PATH.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    write_sha_sidecar(OUT_PATH)
    print(
        json.dumps(
            {
                "status": result["status"],
                "out": str(OUT_PATH),
                "scores": scores,
                "delta_vs_training_baseline": deltas_vs_training,
                "delta_vs_cloud_baseline": deltas_vs_cloud,
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
