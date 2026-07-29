import csv
from pathlib import Path
from typing import Any, Dict, List

from app.config import DATA_DIR
from app.schemas import RecommendRequest, UserProfile
from app.services.recommender import normalize_profile, score_vehicle

REAL_RECOMMENDER_CSV = DATA_DIR / "real_world" / "real_ev_specs_for_recommender.csv"
NUMERIC_FIELDS = {
    "id": int,
    "price_min": int,
    "price_max": int,
    "cltc_range": int,
    "battery_kwh": float,
    "fast_charge_minutes": int,
    "seats": int,
    "wheelbase": int,
    "trunk_volume": int,
    "safety_score": float,
    "monthly_sales": int,
    "data_quality_score": float,
}


def _cast(row: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(row)
    for field, caster in NUMERIC_FIELDS.items():
        try:
            result[field] = caster(float(result.get(field) or 0))
        except (TypeError, ValueError):
            result[field] = caster(0)
    return result


def list_real_world_candidates(limit: int = 500) -> List[Dict[str, Any]]:
    if not REAL_RECOMMENDER_CSV.exists():
        return []
    with REAL_RECOMMENDER_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        rows = [_cast(row) for row in csv.DictReader(f)]
    return rows[:limit]


def recommend_real_world(req: RecommendRequest) -> Dict[str, Any]:
    profile = normalize_profile(req.query, req.profile)
    candidates = list_real_world_candidates()
    scored = [score_vehicle(vehicle, profile) for vehicle in candidates]
    for item in scored:
        quality = item.get("data_quality_score", 100)
        if quality < 80:
            item["score"] = round(item["score"] - 6, 1)
            item.setdefault("cautions", []).insert(0, f"数据完整度 {quality} 分，推荐前需重点核验")
        item["source_type"] = "real_world_enriched"
    scored.sort(key=lambda item: item["score"], reverse=True)
    top = scored[:req.top_k]
    answer = "基于真实公开新能源车型数据集完成候选排序。系统优先考虑预算、续航、空间、补能、品牌/车型偏好，并对低完整度数据做降权。"
    if top:
        answer += "\n\nTop 推荐：" + "；".join(f"{item['brand']} {item['model']}（{item['score']}分）" for item in top[:5])
    return {
        "profile": profile,
        "recommendations": top,
        "answer": answer,
        "candidate_count": len(candidates),
        "agent_trace": [{"agent": "RealWorldRecommender", "observation": f"已在 {len(candidates)} 条真实补齐候选中完成排序"}],
        "skill_trace": [],
        "sources": [{"rank": 1, "domain": "真实数据集", "source": "data/real_world/real_ev_specs_for_recommender.csv", "score": 1, "content": "Open EV Data 与 OSkrk EV Database 清洗补齐后的真实候选集"}],
    }
