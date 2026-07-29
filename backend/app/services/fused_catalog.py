import re
from collections import Counter
from typing import Any, Dict, List

from app.database import list_vehicles
from app.schemas import RecommendRequest
from app.services.real_world_recommender import list_real_world_candidates
from app.services.recommender import normalize_profile, score_vehicle

BRAND_ZH = {
    "Audi": "奥迪",
    "BMW": "宝马",
    "Mercedes": "奔驰",
    "Merceds": "奔驰",
    "Porsche": "保时捷",
    "Tesla": "特斯拉",
    "Volkswagen": "大众",
    "VW": "大众",
    "Hyundai": "现代",
    "Kia": "起亚",
    "Ford": "福特",
    "Chevrolet": "雪佛兰",
    "Volvo": "沃尔沃",
    "Nissan": "日产",
    "Renault": "雷诺",
    "Peugeot": "标致",
    "Citroen": "雪铁龙",
    "Jaguar": "捷豹",
    "Lucid": "Lucid",
    "Smart": "smart",
}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fa5]", "", (text or "").lower())


def _local_vehicle(vehicle: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(vehicle)
    item["catalog_source"] = "local_curated"
    item["data_quality_score"] = 100
    item["source_type"] = "local_curated"
    return item


def _real_vehicle(vehicle: Dict[str, Any], next_id: int) -> Dict[str, Any]:
    item = dict(vehicle)
    brand = item.get("brand", "")
    item["original_brand"] = brand
    item["brand"] = BRAND_ZH.get(brand, brand)
    item["id"] = next_id
    item["catalog_source"] = "real_world_enriched"
    item["source_type"] = "real_world_enriched"
    item["monthly_sales"] = max(1, int(item.get("monthly_sales") or 1))
    return item


def fused_catalog(limit_real: int = 220) -> Dict[str, Any]:
    local = [_local_vehicle(vehicle) for vehicle in list_vehicles()]
    seen = {_norm(f"{item['brand']}{item['model']}") for item in local}
    fused = list(local)
    skipped = 0
    for vehicle in list_real_world_candidates(limit_real):
        item = _real_vehicle(vehicle, len(fused) + 1)
        key = _norm(f"{item['brand']}{item['model']}{item.get('raw_model_year', '')}")
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        fused.append(item)
    sources = Counter(item["catalog_source"] for item in fused)
    brands = Counter(item["brand"] for item in fused)
    types = Counter(item["vehicle_type"] for item in fused)
    return {
        "vehicles": fused,
        "summary": {
            "total": len(fused),
            "local_count": sources.get("local_curated", 0),
            "real_count": sources.get("real_world_enriched", 0),
            "dedup_skipped": skipped,
            "brand_count": len(brands),
            "brand_distribution": brands.most_common(16),
            "vehicle_type_distribution": types.most_common(),
        },
    }


def recommend_fused(req: RecommendRequest) -> Dict[str, Any]:
    profile = normalize_profile(req.query, req.profile)
    catalog = fused_catalog()
    scored = []
    for vehicle in catalog["vehicles"]:
        item = score_vehicle(vehicle, profile)
        if item.get("catalog_source") == "real_world_enriched":
            item["score"] = round(item["score"] - 2, 1)
            if item.get("data_quality_score", 100) < 85:
                item["score"] = round(item["score"] - 5, 1)
                item.setdefault("cautions", []).insert(0, "真实扩展数据完整度偏低，成交前需核验参数")
        scored.append(item)
    scored.sort(key=lambda item: item["score"], reverse=True)
    top = scored[:req.top_k]
    if top and not any(item.get("catalog_source") == "real_world_enriched" for item in top):
        real_best = next((item for item in scored if item.get("catalog_source") == "real_world_enriched"), None)
        if real_best:
            top[-1] = real_best
    return {
        "profile": profile,
        "recommendations": top,
        "answer": f"已启用融合候选池：本地精选 {catalog['summary']['local_count']} 条 + 真实扩展 {catalog['summary']['real_count']} 条，去重跳过 {catalog['summary']['dedup_skipped']} 条。",
        "catalog_summary": catalog["summary"],
        "agent_trace": [{"agent": "FusedCatalogRecommender", "observation": f"融合候选池共 {catalog['summary']['total']} 条，已完成排序"}],
        "skill_trace": [],
        "sources": [{"rank": 1, "domain": "融合候选池", "source": "local vehicle_database.csv + real_world enriched csv", "score": 1, "content": "融合本地中文精选车型和真实公开数据补齐候选"}],
    }
