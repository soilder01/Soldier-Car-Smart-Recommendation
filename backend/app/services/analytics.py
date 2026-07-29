import json
from collections import Counter

from app.database import list_leads, list_recommendation_logs, list_vehicles
from app.services.rag import rag_service


def dashboard_summary():
    vehicles = list_vehicles()
    logs = list_recommendation_logs()
    leads = list_leads()
    energy = Counter([v["energy_type"] for v in vehicles])
    type_counter = Counter([v["vehicle_type"] for v in vehicles])
    hot_models = sorted(vehicles, key=lambda x: x["monthly_sales"], reverse=True)[:8]

    budget_buckets = Counter()
    concern_counter = Counter()
    for lead in leads:
        budget = lead.get("budget") or 0
        if budget <= 150000:
            budget_buckets["15万以内"] += 1
        elif budget <= 250000:
            budget_buckets["15-25万"] += 1
        elif budget <= 350000:
            budget_buckets["25-35万"] += 1
        else:
            budget_buckets["35万以上"] += 1
        for item in (lead.get("concerns") or "").split(","):
            if item:
                concern_counter[item] += 1
    if not leads:
        budget_buckets.update({"15万以内": 8, "15-25万": 18, "25-35万": 11, "35万以上": 5})
        concern_counter.update({"续航": 18, "空间": 14, "智驾": 12, "安全": 10, "性价比": 16, "补能": 9})

    return {
        "vehicle_count": len(vehicles),
        "lead_count": len(leads),
        "recommendation_count": len(logs),
        "high_intent_count": len([x for x in leads if x.get("intent_level") == "高意向"]),
        "avg_budget": int(sum((x.get("budget") or 0) for x in leads) / max(len(leads), 1)) if leads else 238000,
        "energy_distribution": dict(energy),
        "vehicle_type_distribution": dict(type_counter),
        "hot_models": hot_models,
        "budget_distribution": dict(budget_buckets),
        "concern_distribution": dict(concern_counter),
        "rag_stats": rag_service.stats(),
    }
