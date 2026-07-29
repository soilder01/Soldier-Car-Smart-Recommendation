import re
from typing import Any, Dict, List

from app.database import list_vehicles, match_mentioned_vehicles
from app.schemas import UserProfile


CITY_ALIASES = [
    "北京", "上海", "广州", "深圳", "杭州", "成都", "重庆", "武汉", "南京", "苏州",
    "天津", "西安", "郑州", "长沙", "青岛", "宁波", "合肥", "佛山", "东莞", "厦门",
]


def _infer_city(text: str) -> str:
    for city in CITY_ALIASES:
        if city in text:
            return city
    match = re.search(r"(?:在|坐标|城市|常住地|上牌地)\s*([\u4e00-\u9fa5]{2,4})(?:市|通勤|用车|上牌|$)", text)
    return match.group(1) if match else ""


def _format_value(field: str, value: Any) -> str:
    if value in (None, "", []):
        return "未识别"
    if field in {"budget_min", "budget_max"}:
        return f"{round(int(value) / 10000)}万"
    if field == "commute_km":
        return f"{value}公里/天"
    if field == "family_size":
        return f"{value}人"
    if field == "has_home_charger":
        return "有家充/固定车位" if value else "无家充/不确定可安装"
    if isinstance(value, list):
        return "、".join(map(str, value))
    return str(value)

def normalize_profile(query: str, profile: UserProfile) -> Dict:
    data = profile.model_dump()
    text = query or ""

    budget_match = re.search(r"(\d+(?:\.\d+)?)\s*(万|w|W)", text)
    if budget_match and not data.get("budget_max"):
        data["budget_max"] = int(float(budget_match.group(1)) * 10000)

    km_match = re.search(r"通勤.*?(\d+(?:\.\d+)?)\s*(公里|km|KM)", text)
    if km_match and not data.get("commute_km"):
        data["commute_km"] = float(km_match.group(1))

    people_match = re.search(r"(\d)\s*(口|人|个).*?(家|家庭)", text)
    if people_match and not data.get("family_size"):
        data["family_size"] = int(people_match.group(1))
    if "三口" in text and not data.get("family_size"):
        data["family_size"] = 3
    if "二胎" in text and not data.get("family_size"):
        data["family_size"] = 4

    city = _infer_city(text)
    if city and not data.get("city"):
        data["city"] = city

    if any(x in text for x in ["没有家充", "不能装充电桩", "无家充", "没法装充电桩", "没有固定车位"]):
        data["has_home_charger"] = False
    elif any(x in text for x in ["家充", "充电桩", "固定车位"]):
        data["has_home_charger"] = True

    if not data.get("preferred_type"):
        for item in ["SUV", "轿车", "MPV", "旅行车", "猎装车", "跑车", "轿跑"]:
            if item.lower() in text.lower():
                data["preferred_type"] = item
                break
        if not data.get("preferred_type") and "豪车" in text:
            data["preferred_type"] = "豪车"

    if not data.get("preferred_energy"):
        if "纯电" in text:
            data["preferred_energy"] = "纯电"
        elif "插混" in text or "混动" in text:
            data["preferred_energy"] = "插混"
        elif "增程" in text:
            data["preferred_energy"] = "增程"

    concern_map = {
        "续航": ["续航", "长途", "高速"],
        "能耗": ["能耗", "电耗", "油耗", "馈电", "亏电", "能效"],
        "安全": ["安全", "电池", "自燃"],
        "智驾": ["智驾", "辅助驾驶", "智能驾驶", "NOA"],
        "空间": ["空间", "后排", "后备箱", "小孩", "家庭"],
        "性价比": ["性价比", "便宜", "价格敏感", "预算有限", "低价"],
        "补能": ["充电", "家充", "补能", "快充"],
        "保值": ["保值", "二手"],
    }
    concerns = list(data.get("concerns") or [])

    def add_concern(label: str) -> None:
        if label not in concerns:
            concerns.append(label)

    if any(word in text for word in ["撩妹", "泡妞", "约会", "面子", "牌面", "社交"]):
        add_concern("社交形象")
        add_concern("内饰氛围")
        add_concern("品牌")
    for label, words in concern_map.items():
        if any(word in text for word in words):
            add_concern(label)
    data["concerns"] = concerns

    for brand in ["比亚迪", "特斯拉", "小鹏", "理想", "问界", "享界", "尊界", "智界", "蔚来", "极氪", "吉利", "长安", "大众", "宝马", "奔驰", "奥迪", "小米", "阿维塔"]:
        if brand in text and brand not in data.get("preferred_brands", []):
            data.setdefault("preferred_brands", []).append(brand)

    mentioned = match_mentioned_vehicles(text)
    data["mentioned_models"] = [f"{item['brand']} {item['model']}" for item in mentioned]
    if mentioned:
        data["explicit_vehicle_compare"] = True
        data["preferred_type"] = ""
        data["preferred_energy"] = ""

    if any(word in text for word in ["试驾", "近期买", "准备买", "订车"]):
        data["intent_level"] = "高意向"
    elif any(word in text for word in ["对比", "纠结", "怎么选", "哪个更适合", "哪款更适合"]):
        data["intent_level"] = "对比中"

    return data


def profile_preview(query: str, profile: UserProfile) -> Dict:
    original = profile.model_dump()
    normalized = normalize_profile(query, profile)
    field_labels = {
        "budget_max": "预算上限",
        "city": "用车城市",
        "family_size": "家庭人数",
        "commute_km": "日常通勤",
        "has_home_charger": "家充条件",
        "preferred_type": "偏好车型",
        "preferred_energy": "偏好能源",
        "preferred_brands": "偏好品牌",
        "concerns": "核心关注点",
        "intent_level": "购车意向",
        "mentioned_models": "点名车型",
    }
    important_fields = ["budget_max", "city", "family_size", "commute_km", "has_home_charger", "concerns"]
    parsed_fields = []
    for field, label in field_labels.items():
        value = normalized.get(field)
        old_value = original.get(field)
        detected = value not in (None, "", [])
        source = "手动填写" if old_value not in (None, "", []) else "自然语言解析" if detected else "待补充"
        parsed_fields.append({
            "field": field,
            "label": label,
            "value": value,
            "display": _format_value(field, value),
            "detected": detected,
            "source": source,
        })

    missing_fields = [field_labels[field] for field in important_fields if normalized.get(field) in (None, "", [])]
    insights = []
    if normalized.get("budget_max"):
        insights.append(f"预算锁定在 {_format_value('budget_max', normalized['budget_max'])} 以内，推荐会优先过滤价格重叠车型。")
    if normalized.get("commute_km"):
        insights.append(f"通勤强度为 {_format_value('commute_km', normalized['commute_km'])}，续航与补能权重会被放大。")
    if normalized.get("family_size") and normalized["family_size"] >= 4:
        insights.append("家庭人数偏多，空间、座椅舒适度和后备箱实用性需要重点关注。")
    if normalized.get("has_home_charger") is False:
        insights.append("当前按无家充处理，插混/增程和公共快充便利性会更关键。")
    elif normalized.get("has_home_charger") is True:
        insights.append("已识别具备家充条件，纯电车型的日常使用成本优势更明显。")
    if normalized.get("concerns"):
        insights.append(f"核心关注点包括：{_format_value('concerns', normalized['concerns'])}。")
    if not insights:
        insights.append("当前信息较少，建议补充预算、城市、家庭人数、通勤和补能条件后再生成推荐。")

    detected_count = sum(1 for field in important_fields if normalized.get(field) not in (None, "", []))
    confidence = round(detected_count / len(important_fields) * 100)
    return {
        "profile": normalized,
        "fields": parsed_fields,
        "missing_fields": missing_fields,
        "insights": insights,
        "confidence": confidence,
        "summary": f"已识别 {detected_count}/{len(important_fields)} 个关键画像字段",
    }


def score_vehicle(vehicle: Dict, profile: Dict) -> Dict:
    budget_max = profile.get("budget_max") or 300000
    budget_min = profile.get("budget_min") or 0
    commute_km = profile.get("commute_km") or 40
    family_size = profile.get("family_size") or 3
    concerns = profile.get("concerns") or []
    mentioned_models = profile.get("mentioned_models") or []
    full_name = f"{vehicle['brand']} {vehicle['model']}"
    is_mentioned = full_name in mentioned_models

    mid_price = (vehicle["price_min"] + vehicle["price_max"]) / 2
    if is_mentioned:
        budget_score = 88
    elif vehicle["price_min"] <= budget_max and vehicle["price_max"] >= budget_min:
        budget_score = 100 - max((mid_price - budget_max) / max(budget_max, 1) * 40, 0)
    else:
        budget_score = max(30, 100 - abs(mid_price - budget_max) / max(budget_max, 1) * 100)

    range_need = max(commute_km * 7, 350)
    if vehicle["energy_type"] == "燃油":
        range_score = 86
    else:
        range_score = min(vehicle["cltc_range"] / range_need * 100, 100)

    space_base = 75
    if family_size >= 4:
        space_base += 10 if vehicle["seats"] >= 5 and vehicle["wheelbase"] >= 2900 else -5
    if vehicle["vehicle_type"] in {"SUV", "MPV"}:
        space_base += 10
    space_score = min(max(space_base, 40), 100)

    has_home_charger = profile.get("has_home_charger")
    if vehicle["energy_type"] == "燃油":
        charging_score = 82
    elif has_home_charger is True:
        charging_score = 92 if vehicle["energy_type"] == "纯电" else 84
    elif has_home_charger is False:
        charging_score = 94 if vehicle["energy_type"] in {"插混", "增程"} else 72
    else:
        charging_score = 82

    smart_score = 92 if "+" in vehicle["adas_level"] else 78
    if "智驾" in concerns:
        smart_score += 6
    safety_score = min(float(vehicle["safety_score"]), 100)

    scenario_score = 75
    scenario_text = f"{vehicle['suitable_scenarios']};{vehicle['highlights']};{vehicle['brand']};{vehicle['model']}"
    for concern in concerns:
        if concern in scenario_text or concern in vehicle["highlights"]:
            scenario_score += 5
    if profile.get("preferred_type") and profile["preferred_type"] in vehicle["vehicle_type"]:
        scenario_score += 8
    if profile.get("preferred_energy") and profile["preferred_energy"] == vehicle["energy_type"]:
        scenario_score += 8
    if profile.get("preferred_brands") and vehicle["brand"] in profile["preferred_brands"]:
        scenario_score += 8
    if is_mentioned:
        scenario_score += 24
    scenario_score = min(scenario_score, 100)

    total = (
        budget_score * 0.25
        + range_score * 0.20
        + space_score * 0.15
        + charging_score * 0.15
        + smart_score * 0.10
        + safety_score * 0.10
        + scenario_score * 0.05
    )
    if profile.get("preferred_type"):
        if profile["preferred_type"] in vehicle["vehicle_type"]:
            total += 10
        else:
            total -= 10
    if "社交形象" in concerns:
        luxury_text = f"{vehicle['brand']};{vehicle['suitable_scenarios']};{vehicle['highlights']}"
        if vehicle["brand"] in {"奔驰", "宝马", "奥迪", "享界", "尊界", "蔚来"}:
            total += 22
        elif any(word in luxury_text for word in ["豪华", "商务", "行政", "社交形象"]):
            total += 12
    if family_size >= 4 and vehicle["vehicle_type"] == "MPV":
        total += 10
    total = min(max(total, 0), 100)

    reasons = [
        f"预算匹配度 {budget_score:.0f} 分",
        f"续航匹配度 {range_score:.0f} 分",
        f"空间匹配度 {space_score:.0f} 分",
    ]
    if has_home_charger is False and vehicle["energy_type"] in {"插混", "增程"}:
        reasons.append("无家充场景下补能更稳妥")
    if "智驾" in concerns and "+" in vehicle["adas_level"]:
        reasons.append("智能驾驶能力更贴合关注点")
    if "空间" in concerns and vehicle["vehicle_type"] == "SUV":
        reasons.append("SUV 空间和家庭实用性较强")
    if is_mentioned:
        reasons.insert(0, "用户问题中明确点名该车型，已优先纳入对比")
    if "社交形象" in concerns:
        if vehicle["brand"] in {"奔驰", "宝马", "奥迪"}:
            reasons.append("豪华品牌在社交形象和商务场景中识别度更高")
        elif vehicle["brand"] in {"特斯拉", "小米", "阿维塔", "享界", "尊界", "智界"}:
            reasons.append("科技品牌和新能源属性更容易体现年轻化与科技感")

    cautions = []
    if vehicle["price_min"] > budget_max:
        cautions.append("入门价格高于当前预算，需要确认金融方案或调整预算")
    if vehicle["energy_type"] == "纯电" and has_home_charger is False:
        cautions.append("无家充时需要确认公共充电便利性")
    if vehicle["cltc_range"] < 550 and "续航" in concerns:
        cautions.append("续航关注度较高，建议试驾并核算实际续航")
    if vehicle["weaknesses"]:
        cautions.append(vehicle["weaknesses"])

    return {
        **vehicle,
        "score": round(total, 1),
        "budget_score": round(budget_score, 1),
        "range_score": round(range_score, 1),
        "space_score": round(space_score, 1),
        "charging_score": round(charging_score, 1),
        "smart_score": round(min(smart_score, 100), 1),
        "safety_score": round(safety_score, 1),
        "scenario_score": round(scenario_score, 1),
        "reasons": reasons,
        "cautions": cautions[:3],
    }


def recommend(query: str, profile: UserProfile, top_k: int = 5) -> Dict:
    normalized = normalize_profile(query, profile)
    vehicles = list_vehicles()
    scored = [score_vehicle(vehicle, normalized) for vehicle in vehicles]
    mentioned_names = set(normalized.get("mentioned_models") or [])
    if mentioned_names:
        mentioned = [item for item in scored if f"{item['brand']} {item['model']}" in mentioned_names]
        others = [item for item in scored if f"{item['brand']} {item['model']}" not in mentioned_names]
        mentioned.sort(key=lambda x: x["score"], reverse=True)
        others.sort(key=lambda x: x["score"], reverse=True)
        scored = mentioned + others
    else:
        scored.sort(key=lambda x: x["score"], reverse=True)
    return {"profile": normalized, "recommendations": scored[:top_k]}
