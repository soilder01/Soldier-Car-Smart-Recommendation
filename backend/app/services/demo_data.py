import json
from datetime import datetime, timedelta
from typing import Any, Dict, List

from app.database import get_conn, list_vehicles


SCENARIOS = [
    {"name": "演示客户-家庭通勤", "budget": 250000, "city": "上海", "concerns": ["续航", "空间", "智驾"], "intent": "高意向", "query": "预算25万以内，三口之家，上海每天通勤50公里，有家充，推荐新能源SUV"},
    {"name": "演示客户-无家充", "budget": 220000, "city": "北京", "concerns": ["补能", "性价比", "安全"], "intent": "中意向", "query": "没有家充，主要市区通勤，偶尔长途，纯电插混增程怎么选"},
    {"name": "演示客户-智驾优先", "budget": 320000, "city": "深圳", "concerns": ["智驾", "座舱", "品牌"], "intent": "高意向", "query": "预算30万左右，想要智驾和座舱体验好一点，推荐哪几款"},
    {"name": "演示客户-二胎家庭", "budget": 350000, "city": "杭州", "concerns": ["空间", "安全", "舒适"], "intent": "高意向", "query": "二胎家庭，想买大空间SUV或MPV，关注安全和舒适"},
    {"name": "演示客户-首购用户", "budget": 160000, "city": "成都", "concerns": ["性价比", "保值", "用车成本"], "intent": "了解中", "query": "第一次买新能源，预算十五六万，想省心耐用"},
    {"name": "演示客户-长途场景", "budget": 280000, "city": "南京", "concerns": ["续航", "补能", "长途"], "intent": "中意向", "query": "经常跨城出差，担心续航和补能，应该选增程还是纯电"},
    {"name": "演示客户-豪华换购", "budget": 450000, "city": "广州", "concerns": ["品牌", "舒适", "智能化"], "intent": "高意向", "query": "从燃油豪华车换新能源，希望品牌和舒适性都好"},
    {"name": "演示客户-女性用户", "budget": 180000, "city": "苏州", "concerns": ["安全", "好开", "颜值"], "intent": "中意向", "query": "女生通勤代步，想要好开安全，预算18万以内"},
    {"name": "演示客户-性能关注", "budget": 300000, "city": "重庆", "concerns": ["性能", "操控", "智驾"], "intent": "了解中", "query": "喜欢驾驶感和加速，预算30万左右，有什么新能源车适合"},
    {"name": "演示客户-老人接送", "budget": 240000, "city": "天津", "concerns": ["舒适", "空间", "上下车便利"], "intent": "中意向", "query": "主要接送老人和孩子，想要舒适空间大，上下车方便"},
    {"name": "演示客户-营运兼家用", "budget": 200000, "city": "武汉", "concerns": ["能耗", "可靠", "空间"], "intent": "了解中", "query": "想兼顾家用和偶尔跑业务，关注能耗和可靠性"},
    {"name": "演示客户-竞品纠结", "budget": 270000, "city": "西安", "concerns": ["对比", "价格", "续航"], "intent": "高意向", "query": "Model Y、小鹏G6、宋L EV 这几款怎么选"},
]


def _pick_models(vehicles: List[Dict[str, Any]], index: int) -> List[Dict[str, Any]]:
    if not vehicles:
        return []
    start = (index * 3) % len(vehicles)
    return [vehicles[(start + offset) % len(vehicles)] for offset in range(3)]


def _score_vehicle(vehicle: Dict[str, Any], rank: int) -> Dict[str, Any]:
    base = 92 - rank * 5
    return {
        **vehicle,
        "score": base,
        "budget_score": max(base - 4, 0),
        "range_score": min(base + 2, 100),
        "space_score": max(base - 2, 0),
        "charging_score": max(base - 8, 0),
        "smart_score": max(base - 3, 0),
        "safety_score": min(float(vehicle.get("safety_score") or 90), 100),
        "scenario_score": max(base - 1, 0),
        "reasons": [
            f"匹配当前预算和{vehicle.get('vehicle_type', '车型')}偏好",
            f"{vehicle.get('cltc_range', 0)}km 续航适合通勤与周末出行",
            str(vehicle.get("highlights") or "配置均衡，适合进一步试驾确认"),
        ],
        "cautions": [str(vehicle.get("weaknesses") or "价格权益和配置以官方实时信息为准")],
        "source_type": "local",
        "source_url": "",
        "source_title": "",
    }


def seed_demo_data() -> Dict[str, Any]:
    vehicles = list_vehicles()
    now = datetime.now()
    with get_conn() as conn:
        conn.execute("DELETE FROM leads WHERE name LIKE '演示客户-%'")
        conn.execute("DELETE FROM recommendation_logs WHERE user_query LIKE '[演示]%'")
        for index, item in enumerate(SCENARIOS):
            picked = _pick_models(vehicles, index)
            profile = {
                "budget_max": item["budget"],
                "city": item["city"],
                "family_size": 3 if "家庭" in item["name"] or "二胎" in item["query"] else None,
                "commute_km": 50 if "通勤" in item["query"] else None,
                "has_home_charger": "无家充" not in item["query"],
                "preferred_type": "SUV" if "SUV" in item["query"] else "",
                "preferred_energy": "新能源",
                "concerns": item["concerns"],
                "intent_level": item["intent"],
            }
            created_at = (now - timedelta(days=index)).strftime("%Y-%m-%d %H:%M:%S")
            models = [f"{v['brand']} {v['model']}" for v in picked]
            conn.execute(
                """
                INSERT INTO leads (created_at, name, phone_masked, profile_json, budget, city, concerns, intent_level, recommended_models, next_action)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (created_at, item["name"], f"138****{index + 1:04d}", json.dumps(profile, ensure_ascii=False), item["budget"], item["city"], ",".join(item["concerns"]), item["intent"], ",".join(models), "邀约试驾并确认充电条件"),
            )
            scored = [_score_vehicle(vehicle, rank) for rank, vehicle in enumerate(picked)]
            top_model = models[0] if models else ""
            conn.execute(
                "INSERT INTO recommendation_logs (created_at, user_query, profile_json, result_json, top_model, confidence) VALUES (?, ?, ?, ?, ?, ?)",
                (created_at, f"[演示]{item['query']}", json.dumps(profile, ensure_ascii=False), json.dumps(scored, ensure_ascii=False), top_model, scored[0]["score"] if scored else 0),
            )
    return {"status": "ok", "lead_count": len(SCENARIOS), "recommendation_count": len(SCENARIOS)}
