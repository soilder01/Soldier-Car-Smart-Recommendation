from copy import deepcopy
from datetime import datetime
from math import log1p
from typing import Any, Dict, List

from app.database import list_recommendation_feedback, recommendation_feedback_summary

MIN_MODEL_SAMPLES = 3
MIN_POOL_SAMPLES = 5
MAX_MODEL_DELTA = 10.0
MAX_POOL_DELTA = 3.0


def _model_name(item: Dict[str, Any]) -> str:
    return f"{item.get('brand', '')} {item.get('model', '')}".strip()


def _parse_time(value: str) -> datetime | None:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(value or "", fmt)
        except ValueError:
            continue
    return None


def _recency_weight(created_at: str, now: datetime) -> float:
    dt = _parse_time(created_at)
    if not dt:
        return 0.7
    age_days = max((now - dt).days, 0)
    if age_days <= 7:
        return 1.0
    if age_days <= 30:
        return 0.85
    if age_days <= 90:
        return 0.65
    return 0.45


def _confidence(total: int, weighted_total: float) -> float:
    sample_factor = min(1.0, log1p(total) / log1p(12))
    freshness_factor = min(1.0, weighted_total / max(total, 1))
    return round(max(0.25, sample_factor * freshness_factor), 2)


def _bucket_template(name_key: str, name: str) -> Dict[str, Any]:
    return {name_key: name, "total": 0, "positive": 0, "negative": 0, "neutral": 0, "weighted_positive": 0.0, "weighted_negative": 0.0, "weighted_total": 0.0, "latest_feedback_at": ""}


def _aggregate_weighted_feedback() -> Dict[str, Dict[str, Dict[str, Any]]]:
    now = datetime.now()
    model_rows: Dict[str, Dict[str, Any]] = {}
    pool_rows: Dict[str, Dict[str, Any]] = {}
    scene_rows: Dict[str, Dict[str, Any]] = {}
    for row in list_recommendation_feedback(500):
        rating = row.get("rating") if row.get("rating") in {"positive", "negative"} else "neutral"
        weight = _recency_weight(row.get("created_at", ""), now)
        model = row.get("model_name") or "未命名车型"
        pool = row.get("candidate_pool") or "unknown"
        buckets = [
            model_rows.setdefault(model, _bucket_template("model_name", model)),
            pool_rows.setdefault(pool, _bucket_template("candidate_pool", pool)),
        ]
        for scene in row.get("scenario_tags") or ["未标注场景"]:
            buckets.append(scene_rows.setdefault(scene, _bucket_template("scenario", scene)))
        for bucket in buckets:
            bucket["total"] += 1
            bucket[rating] += 1
            bucket["weighted_total"] += weight
            if rating == "positive":
                bucket["weighted_positive"] += weight
            elif rating == "negative":
                bucket["weighted_negative"] += weight
            if row.get("created_at", "") > bucket.get("latest_feedback_at", ""):
                bucket["latest_feedback_at"] = row.get("created_at", "")
    return {"models": model_rows, "pools": pool_rows, "scenes": scene_rows}


def _build_model_rule(row: Dict[str, Any]) -> Dict[str, Any] | None:
    total = row.get("total", 0)
    if total < MIN_MODEL_SAMPLES:
        return None
    weighted_positive = row.get("weighted_positive", 0.0)
    weighted_negative = row.get("weighted_negative", 0.0)
    weighted_total = max(row.get("weighted_total", 0.0), 0.1)
    net_rate = (weighted_positive - weighted_negative) / weighted_total
    if abs(net_rate) < 0.2:
        return None
    confidence = _confidence(total, weighted_total)
    delta = round(max(-MAX_MODEL_DELTA, min(MAX_MODEL_DELTA, net_rate * 8.0 * confidence)), 1)
    if not delta:
        return None
    direction = "boost" if delta > 0 else "penalty"
    return {
        "delta": delta,
        "direction": direction,
        "confidence": confidence,
        "positive": row.get("positive", 0),
        "negative": row.get("negative", 0),
        "neutral": row.get("neutral", 0),
        "total": total,
        "weighted_positive": round(weighted_positive, 2),
        "weighted_negative": round(weighted_negative, 2),
        "sample_threshold": MIN_MODEL_SAMPLES,
        "latest_feedback_at": row.get("latest_feedback_at", ""),
        "reason": f"样本{total}条，近期加权正反馈{weighted_positive:.1f}/负反馈{weighted_negative:.1f}，置信度{confidence}",
    }


def _build_pool_rule(row: Dict[str, Any]) -> Dict[str, Any] | None:
    total = row.get("total", 0)
    if total < MIN_POOL_SAMPLES:
        return None
    weighted_positive = row.get("weighted_positive", 0.0)
    weighted_negative = row.get("weighted_negative", 0.0)
    weighted_total = max(row.get("weighted_total", 0.0), 0.1)
    positive_rate = round(weighted_positive / weighted_total * 100, 1)
    confidence = _confidence(total, weighted_total)
    delta = 0.0
    if positive_rate < 55:
        delta = -MAX_POOL_DELTA * confidence
    elif positive_rate >= 80:
        delta = MAX_POOL_DELTA * 0.7 * confidence
    delta = round(delta, 1)
    if not delta:
        return None
    return {
        "delta": delta,
        "direction": "boost" if delta > 0 else "penalty",
        "confidence": confidence,
        "positive_rate": positive_rate,
        "positive": row.get("positive", 0),
        "negative": row.get("negative", 0),
        "neutral": row.get("neutral", 0),
        "total": total,
        "sample_threshold": MIN_POOL_SAMPLES,
        "latest_feedback_at": row.get("latest_feedback_at", ""),
        "reason": f"候选池样本{total}条，加权正反馈率{positive_rate}%，置信度{confidence}",
    }


def build_feedback_policy() -> Dict[str, Any]:
    summary = recommendation_feedback_summary()
    weighted = _aggregate_weighted_feedback()
    model_rules = {}
    for name, row in weighted["models"].items():
        rule = _build_model_rule(row)
        if rule:
            model_rules[name] = rule
    pool_rules = {}
    for name, row in weighted["pools"].items():
        rule = _build_pool_rule(row)
        if rule:
            pool_rules[name] = rule
    risky_scenes = []
    for name, row in weighted["scenes"].items():
        if row.get("total", 0) >= 2:
            rate = round(row.get("weighted_negative", 0) / max(row.get("weighted_total", 0.1), 0.1) * 100, 1)
            if rate >= 50:
                risky_scenes.append({"scenario": name, "negative_rate": rate, "total": row.get("total", 0), "latest_feedback_at": row.get("latest_feedback_at", "")})
    return {
        "summary": summary,
        "model_rules": model_rules,
        "pool_rules": pool_rules,
        "risky_scenes": sorted(risky_scenes, key=lambda item: (item["negative_rate"], item["total"]), reverse=True)[:5],
        "stability": {
            "min_model_samples": MIN_MODEL_SAMPLES,
            "min_pool_samples": MIN_POOL_SAMPLES,
            "uses_recency_decay": True,
            "uses_confidence": True,
            "model_delta_cap": MAX_MODEL_DELTA,
            "pool_delta_cap": MAX_POOL_DELTA,
        },
    }


def apply_feedback_policy(recommendations: List[Dict[str, Any]], selected_pool: str) -> Dict[str, Any]:
    policy = build_feedback_policy()
    adjusted = []
    applied = []
    pool_rule = policy["pool_rules"].get(selected_pool)
    for raw in recommendations:
        item = deepcopy(raw)
        name = _model_name(item)
        score_delta = 0.0
        reasons = list(item.get("reasons") or [])
        cautions = list(item.get("cautions") or [])
        item_rules = []
        model_rule = policy["model_rules"].get(name)
        if model_rule:
            score_delta += model_rule["delta"]
            if model_rule["delta"] < 0:
                cautions.append(f"反馈策略降权：{model_rule['reason']}")
            else:
                reasons.append(f"反馈策略加权：{model_rule['reason']}")
            rule_record = {"target": name, "type": "model", **model_rule}
            item_rules.append(rule_record)
            applied.append(rule_record)
        if pool_rule:
            score_delta += pool_rule["delta"]
            if pool_rule["delta"] < 0:
                cautions.append(f"候选池反馈降权：{pool_rule['reason']}")
            else:
                reasons.append(f"候选池反馈加权：{pool_rule['reason']}")
            item_rules.append({"target": selected_pool, "type": "pool", **pool_rule})
        if score_delta:
            item["score_before_feedback"] = item.get("score", 0)
            item["feedback_score_delta"] = round(score_delta, 1)
            item["feedback_policy_rules"] = item_rules
            item["score"] = round(max(0, min(100, float(item.get("score", 0)) + score_delta)), 1)
            item["reasons"] = reasons[:6]
            item["cautions"] = cautions[:6]
        adjusted.append(item)
    adjusted.sort(key=lambda item: item.get("score", 0), reverse=True)
    if pool_rule:
        applied.append({"target": selected_pool, "type": "pool", **pool_rule})
    return {"recommendations": adjusted, "policy": policy, "applied_rules": applied[:12]}
