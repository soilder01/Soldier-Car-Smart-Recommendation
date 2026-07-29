import re
from typing import Dict, List

from openai import OpenAI

from app.config import CHAT_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL, TEMPERATURE
from app.database import save_recommendation_log
from app.schemas import RecommendRequest, RecommendResponse, UserProfile
from app.services.llm_client import openai_client
from app.services.skills import skills


def join_sources(sources: List[Dict]) -> str:
    lines = []
    for i, src in enumerate(sources, 1):
        if src.get("source") == "web":
            lines.append(f"[{i}] Web：{src.get('title')} {src.get('url', '')}")
        else:
            lines.append(f"[{i}] {src['source']} | {src['domain']}：{src['content'][:220]}")
    return "\n".join(lines)


def detect_recommend_scene(query: str, profile: Dict) -> str:
    text = f"{query} {profile.get('preferred_type', '')} {' '.join(profile.get('concerns') or [])}".lower()
    social_words = ["泡妞", "撩妹", "约会", "社交", "面子", "牌面", "豪车", "跑车", "轿跑", "性能车", "amg", "保时捷", "m2"]
    family_words = ["三口", "家庭", "家用", "小孩", "通勤", "空间", "suv", "露营"]
    if any(word in text for word in social_words):
        return "social_luxury"
    if any(word in text for word in family_words):
        return "family_commute"
    return "general"


def build_web_search_query(query: str, scene: str) -> str:
    if scene == "social_luxury":
        return f"{query} 年轻人 豪车 跑车 轿跑 价格 参数 推荐 官方"
    return f"{query} 车型 价格 参数 官方"


EXTERNAL_VEHICLE_CATALOG = [
    {
        "brand": "保时捷", "model": "718 Cayman", "vehicle_type": "跑车", "energy_type": "燃油",
        "price_min": 565000, "price_max": 1578000, "cltc_range": 0, "seats": 2, "score": 86.0,
        "keywords": ["保时捷", "porsche", "718", "cayman", "跑车", "年轻", "社交", "撩妹"],
        "reasons": ["跑车造型和品牌识别度强", "适合强调个性、运动感和社交第一印象"],
        "cautions": ["两座布局实用性有限，购车和维护成本较高"],
    },
    {
        "brand": "宝马", "model": "M2", "vehicle_type": "跑车", "energy_type": "燃油",
        "price_min": 599000, "price_max": 599000, "cltc_range": 0, "seats": 4, "score": 84.0,
        "keywords": ["宝马m2", "bmw m2", "m2", "跑车", "性能", "年轻"],
        "reasons": ["性能取向明确，品牌运动属性强", "比大型豪华轿车更强调驾驶乐趣"],
        "cautions": ["后排和舒适性不如中大型轿车"],
    },
    {
        "brand": "奔驰AMG", "model": "CLE 53", "vehicle_type": "轿跑", "energy_type": "燃油",
        "price_min": 720000, "price_max": 760000, "cltc_range": 0, "seats": 4, "score": 83.0,
        "keywords": ["amg", "奔驰amg", "cle", "轿跑", "豪华", "社交"],
        "reasons": ["奔驰豪华感和 AMG 运动属性兼具", "适合重视品牌、内饰氛围和个性化表达"],
        "cautions": ["价格较高，具体配置和引入版本需以官方为准"],
    },
    {
        "brand": "名爵", "model": "Cyberster", "vehicle_type": "跑车", "energy_type": "纯电",
        "price_min": 319800, "price_max": 359800, "cltc_range": 580, "seats": 2, "score": 82.0,
        "keywords": ["mg cyberster", "cyberster", "名爵", "敞篷", "跑车", "年轻", "纯电跑车"],
        "reasons": ["敞篷纯电跑车视觉冲击强", "价格相比传统豪华跑车更低"],
        "cautions": ["品牌豪华属性弱于保时捷、奔驰、宝马"],
    },
    {
        "brand": "路特斯", "model": "Emira", "vehicle_type": "跑车", "energy_type": "燃油",
        "price_min": 858000, "price_max": 1118000, "cltc_range": 0, "seats": 2, "score": 81.0,
        "keywords": ["lotus", "路特斯", "emira", "跑车", "操控", "年轻"],
        "reasons": ["小众跑车品牌辨识度高，操控标签强", "适合追求个性和稀缺感的用户"],
        "cautions": ["售后网络和日常实用性需重点确认"],
    },
    {
        "brand": "特斯拉", "model": "Model Y L", "vehicle_type": "SUV", "energy_type": "纯电",
        "price_min": 300000, "price_max": 380000, "cltc_range": 650, "seats": 6, "score": 80.0,
        "keywords": ["model y l", "modelyl", "model y-l", "modely-l", "特斯拉modelyl", "六座"],
        "reasons": ["适合关注特斯拉品牌、空间和家庭场景的用户", "可作为 Model Y 的更大空间备选"],
        "cautions": ["具体上市节奏、价格和配置以官方实时信息为准"],
    },
    {
        "brand": "特斯拉", "model": "Model 3", "vehicle_type": "轿车", "energy_type": "纯电",
        "price_min": 235500, "price_max": 339500, "cltc_range": 606, "seats": 5, "score": 84.0,
        "keywords": ["model 3", "model3", "特斯拉model3", "特斯拉 model 3"],
        "reasons": ["品牌认知度高，适合城市通勤和年轻用户", "补能网络成熟，能耗和智能化体验有优势"],
        "cautions": ["后排空间和底盘舒适性需结合试驾判断"],
    },
    {
        "brand": "特斯拉", "model": "Model Y", "vehicle_type": "SUV", "energy_type": "纯电",
        "price_min": 263500, "price_max": 363500, "cltc_range": 688, "seats": 5, "score": 85.0,
        "keywords": ["model y", "modely", "特斯拉modely", "特斯拉 model y"],
        "reasons": ["SUV 空间更适合家庭和多场景使用", "补能便利性、保值率和品牌认知度突出"],
        "cautions": ["价格和权益波动较快，需核验官方实时政策"],
    },
    {
        "brand": "奔驰", "model": "E300L", "vehicle_type": "轿车", "energy_type": "燃油",
        "price_min": 500000, "price_max": 600000, "cltc_range": 0, "seats": 5, "score": 82.0,
        "keywords": ["e300l", "e 300 l", "奔驰e300l", "奔驰 e300l", "奔驰e级", "mercedes e"],
        "reasons": ["豪华行政轿车形象稳定，内饰氛围和商务属性强", "适合重视品牌、舒适性和社交场景的用户"],
        "cautions": ["购置税、保险和保养成本高于多数新能源车型"],
    },
    {
        "brand": "宝马", "model": "5系", "vehicle_type": "轿车", "energy_type": "燃油",
        "price_min": 430000, "price_max": 560000, "cltc_range": 0, "seats": 5, "score": 81.0,
        "keywords": ["宝马5系", "宝马 5系", "bmw 5", "bmw 5 series", "525li", "530li"],
        "reasons": ["品牌运动属性和豪华行政定位兼具", "适合喜欢驾驶质感且兼顾商务形象的用户"],
        "cautions": ["终端优惠和配置差异较大，需要结合当地经销商政策核验"],
    },
]


def filter_external_candidates(candidates: List[Dict], profile: Dict) -> List[Dict]:
    budget_max = profile.get("budget_max")
    preferred_type = profile.get("preferred_type") or ""
    filtered = []
    for item in candidates:
        if budget_max and item["price_min"] > budget_max * 1.25:
            continue
        if preferred_type in {"跑车", "轿跑"} and item["vehicle_type"] not in {"跑车", "轿跑"}:
            continue
        if preferred_type == "豪车" and item["brand"] not in {"保时捷", "奔驰AMG", "奔驰", "宝马", "路特斯", "特斯拉"}:
            continue
        filtered.append(item)
    return filtered or candidates


def external_candidate_to_card(item: Dict, source: Dict) -> Dict:
    return {
        "id": -1000 - abs(hash(item["brand"] + item["model"])) % 100000,
        "brand": item["brand"],
        "model": item["model"],
        "vehicle_type": item["vehicle_type"],
        "energy_type": item["energy_type"],
        "price_min": item["price_min"],
        "price_max": item["price_max"],
        "cltc_range": item["cltc_range"],
        "seats": item["seats"],
        "score": item["score"],
        "budget_score": 0.0,
        "range_score": 0.0,
        "space_score": 0.0,
        "charging_score": 0.0,
        "smart_score": 0.0,
        "safety_score": 0.0,
        "scenario_score": item["score"],
        "reasons": ["DeepSearch 联网搜索 + 外部候选库匹配"] + item["reasons"],
        "cautions": item["cautions"] + ["联网候选参数建议以官方实时信息核验"],
        "highlights": "来自 Web Search 和外部候选库",
        "weaknesses": "非本地结构化主库车型",
        "source_type": "web",
        "source_url": source.get("url", ""),
        "source_title": source.get("title", ""),
    }


def extract_web_vehicle_candidates(web_sources: List[Dict], existing: List[Dict], query: str = "", limit: int = 4) -> List[Dict]:
    existing_names = {f"{item['brand']} {item['model']}".lower().replace(" ", "") for item in existing}
    query_text = (query + " " + " ".join([src.get("title", "") + " " + src.get("content", "") for src in web_sources])).lower()
    patterns = [
        r"(BMW|宝马)\s*([A-Za-z0-9系]+)",
        r"(Mercedes|奔驰)\s*([A-Za-z0-9级]+)",
        r"(Porsche|保时捷)\s*([A-Za-z0-9]+)",
        r"(Tesla|特斯拉)\s*(Model\s*[A-Za-z0-9]+)",
        r"(Polestar|极星)\s*([A-Za-z0-9]+)",
        r"(MG|名爵)\s*([A-Za-z0-9]+)",
        r"(蔚来|小米|阿维塔|智界|享界|尊界|问界|极氪|腾势)\s*([A-Za-z0-9]+)",
    ]
    candidates = []
    seen = set()
    generic_keywords = {"跑车", "年轻", "社交", "撩妹", "性能", "豪华", "操控", "敞篷", "纯电跑车", "六座"}
    source = web_sources[0] if web_sources else {}

    for item in EXTERNAL_VEHICLE_CATALOG:
        exact_hit = any(
            keyword not in generic_keywords and keyword.lower().replace(" ", "") in query_text.replace(" ", "")
            for keyword in item["keywords"]
        )
        if exact_hit:
            key = f"{item['brand']}{item['model']}".lower().replace(" ", "")
            if key in seen:
                continue
            seen.add(key)
            candidates.append(external_candidate_to_card(item, source))
            if len(candidates) >= limit:
                return candidates

    for item in EXTERNAL_VEHICLE_CATALOG:
        broad_scene_hit = any(word in query_text for word in ["sports car", "跑车", "豪华", "young", "年轻", "attractive", "社交"])
        is_sports_candidate = item.get("model") in {"718 Cayman", "M2", "CLE 53", "Cyberster", "Emira"}
        if broad_scene_hit and is_sports_candidate:
            key = f"{item['brand']}{item['model']}".lower().replace(" ", "")
            if key in seen or key in existing_names:
                continue
            seen.add(key)
            candidates.append(external_candidate_to_card(item, source))
            if len(candidates) >= limit:
                return candidates
    for src in web_sources:
        title = src.get("title") or src.get("content") or ""
        for pattern in patterns:
            for brand, model in re.findall(pattern, title, flags=re.IGNORECASE):
                brand = brand.strip()
                model = model.strip().replace(" ", "")
                if len(model) < 1:
                    continue
                key = f"{brand}{model}".lower().replace(" ", "")
                if key in seen or key in existing_names:
                    continue
                seen.add(key)
                candidates.append({
                    "id": -1000 - len(candidates),
                    "brand": brand,
                    "model": model,
                    "vehicle_type": "外部候选",
                    "energy_type": "待核验",
                    "price_min": 0,
                    "price_max": 0,
                    "cltc_range": 0,
                    "seats": 0,
                    "score": 78.0,
                    "budget_score": 0.0,
                    "range_score": 0.0,
                    "space_score": 0.0,
                    "charging_score": 0.0,
                    "smart_score": 0.0,
                    "safety_score": 0.0,
                    "scenario_score": 78.0,
                    "reasons": ["DeepSearch 联网搜索发现的外部候选车型", "参数、价格和配置需要进入详情页或官方渠道核验"],
                    "cautions": ["联网候选未进入本地结构化车型库，建议核验官方价格、配置和交付信息"],
                    "highlights": "来自 Web Search 的候选结果",
                    "weaknesses": "缺少本地结构化参数",
                    "source_type": "web",
                    "source_url": src.get("url", ""),
                    "source_title": title,
                })
                if len(candidates) >= limit:
                    return candidates
    if not candidates:
        for src in web_sources[:limit]:
            title = (src.get("title") or src.get("content") or "联网搜索候选").strip()
            short_title = title[:22] + ("..." if len(title) > 22 else "")
            candidates.append({
                "id": -1000 - len(candidates),
                "brand": "联网候选",
                "model": short_title,
                "vehicle_type": "Web Search",
                "energy_type": "待核验",
                "price_min": 0,
                "price_max": 0,
                "cltc_range": 0,
                "seats": 0,
                "score": 76.0,
                "budget_score": 0.0,
                "range_score": 0.0,
                "space_score": 0.0,
                "charging_score": 0.0,
                "smart_score": 0.0,
                "safety_score": 0.0,
                "scenario_score": 76.0,
                "reasons": ["DeepSearch 联网搜索发现的外部候选资料", "标题中可能包含车型、榜单或竞品建议，建议点击来源进一步核验"],
                "cautions": ["该卡片来自网页搜索结果，不等同于本地结构化车型库"],
                "highlights": "来自 Web Search 的候选资料",
                "weaknesses": "缺少本地结构化参数",
                "source_type": "web",
                "source_url": src.get("url", ""),
                "source_title": title,
            })
    return candidates


def social_luxury_candidates(web_sources: List[Dict], existing: List[Dict], profile: Dict, query: str, limit: int = 5) -> List[Dict]:
    candidates = extract_web_vehicle_candidates(web_sources, existing, query, limit=12)
    if not candidates:
        source = web_sources[0] if web_sources else {}
        for item in EXTERNAL_VEHICLE_CATALOG:
            if item["model"] in {"718 Cayman", "M2", "CLE 53", "Cyberster", "Emira", "Model 3", "E300L", "5系"}:
                candidates.append(external_candidate_to_card(item, source))
    return filter_external_candidates(candidates, profile)[:limit]


class MultiAgentOrchestrator:
    def route(self, query: str) -> Dict:
        if any(word in query for word in ["对比", "相比", "怎么选", "差异"]):
            intent = "compare"
        elif any(word in query for word in ["话术", "客户", "销售", "异议"]):
            intent = "sales"
        elif any(word in query for word in ["推荐", "买", "预算", "家用", "通勤"]):
            intent = "recommend"
        else:
            intent = "knowledge"
        return {"agent": "RouterAgent", "intent": intent, "observation": f"识别意图为 {intent}"}

    def generate_answer(self, query: str, profile: Dict, recommendations: List[Dict], sources: List[Dict], sales_script: Dict) -> str:
        top_lines = []
        for i, item in enumerate(recommendations[:5], 1):
            if item.get("source_type") == "web":
                top_lines.append(
                    f"{i}. {item['brand']} {item['model']}：联网候选，来源：{item.get('source_title', '')}。理由：{'；'.join(item['reasons'][:2])}"
                )
                continue
            top_lines.append(
                f"{i}. {item['brand']} {item['model']}：推荐分 {item['score']}，价格 {item['price_min']//10000}-{item['price_max']//10000} 万，"
                f"{item['energy_type']}，CLTC {item['cltc_range']} km。理由：{'；'.join(item['reasons'][:3])}"
            )
        social_note = ""
        if any(x in profile.get("concerns", []) for x in ["社交形象", "内饰氛围", "品牌"]):
            social_note = "你提到的“撩妹”我会按更专业的“社交形象/约会第一印象/个人气质匹配”来分析：车辆能增强外在呈现，但真正的吸引力还取决于谈吐、审美和相处体验。\n\n"
        fallback = f"""结论：根据当前画像，推荐优先关注 {recommendations[0]['brand']} {recommendations[0]['model']}。

用户画像摘要：预算上限 {profile.get('budget_max') or '未明确'}，家庭人数 {profile.get('family_size') or '未明确'}，通勤 {profile.get('commute_km') or '未明确'} km，充电条件 {profile.get('has_home_charger')}，关注点 {profile.get('concerns') or '综合体验'}。

{social_note}Hybrid 检索补充：本次推荐同时参考本地结构化车型库、本地 RAG 知识库和 Web Search 公开资料；卡片来自本地结构化车型库，联网资料用于补充价格、政策、口碑和实时信息判断，最终仍建议以官方实时信息为准。

推荐车型：
{chr(10).join(top_lines)}

销售建议：
- 开场：{sales_script.get('opening')}
- 推荐：{sales_script.get('recommendation')}
- 异议处理：{sales_script.get('objection')}
- 下一步：{sales_script.get('next_action')}

风险提示：实际价格、权益、续航和辅助驾驶可用范围以官方实时信息和试驾体验为准。"""
        if not OPENAI_API_KEY:
            return fallback

        allowed_models = [f"{item['brand']} {item['model']}" for item in recommendations[:5]]
        client = openai_client()
        prompt = f"""你是新能源汽车销售推荐专家。请基于用户画像、本地结构化车型评分、本地 RAG 知识库证据、联网 Web Search 公开资料和销售话术，生成专业、克制、可执行的购车建议。

要求：
1. 先给明确推荐结论。
2. 说明用户画像和关键约束。
3. 输出 Top 车型推荐理由。
4. 车型推荐只能围绕“允许推荐车型列表”中的车型展开；联网搜索资料只能作为补充证据，不能把网页标题或未结构化车型当作推荐卡片。
5. 如果联网资料与本地车型库存在差异，请说明“需以官方实时信息核验”，不要直接覆盖结构化参数。
6. 给出销售跟进话术。
7. 使用 [1][2] 引用证据。
8. 避免绝对化表达，不要夸大辅助驾驶或续航。
9. 如果用户使用“撩妹”等说法，请转化为“社交形象、约会第一印象、个人气质匹配”来专业回答。

用户问题：{query}
用户画像：{profile}
允许推荐车型列表：{allowed_models}
推荐结果：{recommendations[:5]}
销售话术：{sales_script}
知识库证据：
{join_sources(sources)}
"""
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    def recommend(self, req: RecommendRequest) -> RecommendResponse:
        skills.reset()
        trace = []
        route = self.route(req.query)
        trace.append(route)

        profile = skills.extract_profile(req.query, req.profile)
        scene = detect_recommend_scene(req.query, profile)
        trace.append({"agent": "ProfileAgent", "observation": "完成用户画像抽取", "profile": profile})
        trace.append({"agent": "SceneAgent", "observation": f"识别推荐场景为 {scene}"})

        skills.vehicle_recall(profile)
        ranked = skills.vehicle_rank(req.query, UserProfile(**profile), req.top_k)
        recommendations = ranked["recommendations"]
        trace.append({"agent": "RecommenderAgent", "observation": f"输出 Top {len(recommendations)} 推荐车型"})

        sources = skills.rag_retrieve(req.query, 6)
        web_sources = skills.web_search(build_web_search_query(req.query, scene), 6) if req.use_deep_search else []
        sources = web_sources + sources
        if req.use_deep_search and web_sources:
            if scene == "social_luxury":
                web_candidates = social_luxury_candidates(web_sources, recommendations, profile, req.query, max(req.top_k, 5))
                if web_candidates:
                    recommendations = (web_candidates + recommendations)[:max(req.top_k, 5)]
                    trace.append({
                        "agent": "DeepSearchAgent",
                        "observation": f"联网搜索返回 {len(web_sources)} 条公开资料，并整理出 {len(web_candidates)} 个豪华/跑车候选卡片",
                    })
                else:
                    trace.append({
                        "agent": "DeepSearchAgent",
                        "observation": f"联网搜索返回 {len(web_sources)} 条公开资料，未找到可结构化的豪华/跑车候选",
                    })
            else:
                trace.append({
                    "agent": "DeepSearchAgent",
                    "observation": f"联网搜索返回 {len(web_sources)} 条公开资料，作为模型综合推荐证据，不直接覆盖结构化车型卡片",
                })
        trace.append({"agent": "ResearchAgent", "observation": f"检索知识证据 {len(sources)} 条，其中联网结果 {len(web_sources)} 条"})

        if recommendations:
            skills.range_estimate(recommendations[0], profile)
            skills.budget_analysis(recommendations[0], profile)

        sales_script = skills.sales_script(profile, recommendations)
        trace.append({"agent": "SalesAgent", "observation": "生成销售辅助话术"})

        answer = self.generate_answer(req.query, profile, recommendations, sources, sales_script)
        answer = skills.compliance_check(answer)
        trace.append({"agent": "ReflectionAgent", "observation": "完成合规检查和风险表达修正"})

        save_recommendation_log(req.query, profile, recommendations)
        return RecommendResponse(
            profile=profile,
            recommendations=recommendations,
            answer=answer,
            agent_trace=trace,
            skill_trace=skills.trace,
            sources=sources,
        )


orchestrator = MultiAgentOrchestrator()

from typing import Any, Tuple

from app.services.feedback_policy import apply_feedback_policy
from app.services.fused_catalog import recommend_fused
from app.services.obsidian_vault import save_recommendation_case
from app.services.rag import rag_service
from app.services.real_world_recommender import recommend_real_world
from app.services.recommender import normalize_profile, recommend as recommend_local

REAL_WORLD_HINTS = ["真实数据", "公开数据", "海外", "欧洲", "美国", "Open EV", "数据库", "参数", "规格", "年份", "Tesla", "BMW", "Audi", "Volkswagen", "Porsche"]
LOCAL_HINTS = ["销售", "中国", "国内", "门店", "试驾", "家庭", "通勤", "置换", "预算", "新能源"]
COMPLEX_HINTS = ["对比", "纠结", "怎么选", "哪款", "兼顾", "综合", "不确定"]


def _has_any(text: str, words: List[str]) -> bool:
    lower = text.lower()
    return any(word.lower() in lower for word in words)


def select_candidate_pool(req: RecommendRequest, profile: Dict[str, Any]) -> Tuple[str, str]:
    strategy = (getattr(req, "candidate_pool_strategy", "auto") or "auto").strip().lower()
    if strategy in {"local", "real", "fused"}:
        return strategy, f"用户显式选择 {strategy} 候选池，Agent 保持该策略并统一编排后端工具。"
    text = req.query or ""
    missing = [key for key in ["budget_max", "family_size", "commute_km", "has_home_charger"] if profile.get(key) in (None, "", [])]
    if _has_any(text, REAL_WORLD_HINTS):
        return "real", "需求包含真实/公开规格或海外车型线索，优先使用真实扩展候选池。"
    if profile.get("preferred_type") == "MPV" or (profile.get("family_size") or 0) >= 5:
        return "local", "多人家庭或 MPV 明确场景优先使用本地精选中文车型库，避免海外商用车型干扰。"
    if len(missing) >= 2 or _has_any(text, COMPLEX_HINTS):
        return "fused", "画像信息不完整或问题较复杂，使用融合池同时覆盖本地精选与真实扩展候选。"
    if _has_any(text, LOCAL_HINTS):
        return "local", "需求更接近中国销售和门店推荐场景，优先使用本地精选候选池。"
    return "fused", "未识别到强约束候选池，默认使用融合池提升覆盖率。"


def _rank_with_pool(req: RecommendRequest, pool: str) -> Dict[str, Any]:
    if pool == "real":
        return recommend_real_world(req)
    if pool == "fused":
        return recommend_fused(req)
    local = recommend_local(req.query, req.profile, req.top_k)
    return {
        "profile": local["profile"],
        "recommendations": local["recommendations"],
        "answer": "已启用本地精选候选池，围绕中国新能源销售场景完成画像解析和车型排序。",
        "agent_trace": [{"agent": "VehicleRankTool", "observation": f"本地精选候选池返回 {len(local['recommendations'])} 条推荐"}],
        "skill_trace": [],
        "sources": [{"rank": 1, "domain": "本地精选车型库", "source": "data/vehicles/vehicle_database.csv", "score": 1, "content": "项目内置中文新能源车型库"}],
    }


def _risk_check(profile: Dict[str, Any], recommendations: List[Dict[str, Any]]) -> List[str]:
    risks = []
    if not recommendations:
        return ["当前候选池没有返回推荐结果，需要检查车型数据和画像条件。"]
    budget_max = profile.get("budget_max") or 0
    if budget_max and all(item.get("price_min", 0) > budget_max for item in recommendations[:3]):
        risks.append("Top 推荐入门价整体高于预算，建议调整预算或放宽车型条件。")
    if profile.get("has_home_charger") is False:
        pure_count = sum(1 for item in recommendations[:3] if item.get("energy_type") == "纯电")
        if pure_count >= 2:
            risks.append("无家充场景下 Top 推荐纯电占比较高，需要核验公共补能便利性。")
    low_quality = [item for item in recommendations if item.get("data_quality_score", 100) < 85]
    if low_quality:
        risks.append("部分真实扩展候选的数据完整度低于 85 分，成交前必须核验价格、续航和配置。")
    caution_count = sum(len(item.get("cautions", [])) for item in recommendations[:5])
    if caution_count:
        risks.append(f"推荐结果包含 {caution_count} 条车型级风险提示，已在结果卡片中展示。")
    if not risks:
        risks.append("未发现阻断性风险，仍需以官方价格、权益、试驾和交付周期为准。")
    return risks[:4]


def _format_orchestrator_answer(pool: str, reason: str, recommendations: List[Dict[str, Any]], risks: List[str]) -> str:
    pool_label = {"local": "本地精选", "real": "真实扩展", "fused": "融合池"}.get(pool, pool)
    lines = []
    for index, item in enumerate(recommendations[:5], 1):
        source = item.get("catalog_source") or item.get("source_type") or "local"
        reasons = "；".join(item.get("reasons", [])[:3])
        lines.append(f"{index}. {item.get('brand')} {item.get('model')}（{item.get('score')}分，{source}）：{reasons}")
    return "\n".join([
        f"Agent 已自动选择「{pool_label}」候选池。",
        f"选择原因：{reason}",
        "",
        "推荐结果：",
        "\n".join(lines) if lines else "暂无推荐结果",
        "",
        "风险核验：",
        "\n".join(f"- {item}" for item in risks),
    ])


def _build_explainability(profile: Dict[str, Any], pool: str, pool_reason: str, recommendations: List[Dict[str, Any]], risks: List[str], sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    top = recommendations[:3]
    budget_max = profile.get("budget_max") or 0
    comparisons = []
    for index, item in enumerate(top, 1):
        cautions = list(item.get("cautions") or [])
        if item.get("source_type") == "real_world_enriched" or item.get("catalog_source") == "real_world_enriched":
            cautions.append("真实扩展数据包含规则补齐字段，成交前需核验官方参数")
        if budget_max and item.get("price_min", 0) > budget_max:
            cautions.append("入门价格高于当前预算，需要确认金融方案或调整预算")
        if not cautions:
            cautions.append("价格、权益、交付周期和辅助驾驶边界需以官方实时信息核验")
        comparisons.append({
            "rank": index,
            "model": f"{item.get('brand')} {item.get('model')}",
            "score": item.get("score"),
            "source": item.get("catalog_source") or item.get("source_type") or "local",
            "why_selected": (item.get("reasons") or [])[:4],
            "cautions": cautions[:3],
            "best_for": _best_for(item, profile),
            "why_not_others": _why_not_others(item, top),
        })
    return {
        "profile_summary": {
            "budget_max": profile.get("budget_max"),
            "city": profile.get("city"),
            "family_size": profile.get("family_size"),
            "commute_km": profile.get("commute_km"),
            "has_home_charger": profile.get("has_home_charger"),
            "concerns": profile.get("concerns") or [],
        },
        "pool_decision": {"selected_pool": pool, "reason": pool_reason},
        "feedback_policy": {
            "applied_rules": [
                {"target": item.get("brand", "") + " " + item.get("model", ""), "delta": item.get("feedback_score_delta", 0), "score_before": item.get("score_before_feedback")}
                for item in recommendations if item.get("feedback_score_delta")
            ]
        },
        "top_comparisons": comparisons,
        "not_recommended": _not_recommended_reason(profile, recommendations),
        "risk_checklist": risks,
        "follow_up_actions": _follow_up_actions(profile, top),
        "evidence_sources": [
            {"domain": item.get("domain"), "source": item.get("source"), "content": item.get("content", "")[:120]}
            for item in sources[:5]
        ],
    }


def _best_for(item: Dict[str, Any], profile: Dict[str, Any]) -> str:
    concerns = profile.get("concerns") or []
    if "空间" in concerns and item.get("vehicle_type") in {"SUV", "MPV"}:
        return "更适合重视家庭空间、后排舒适度和多场景装载的用户"
    if "补能" in concerns or profile.get("has_home_charger") is False:
        return "更适合需要重点核验补能便利性和长途稳定性的用户"
    if "智驾" in concerns:
        return "更适合重视辅助驾驶体验但愿意试驾核验边界的用户"
    return "更适合综合平衡预算、续航、空间和品牌接受度的用户"


def _why_not_others(item: Dict[str, Any], top: List[Dict[str, Any]]) -> str:
    others = [x for x in top if x is not item]
    if not others:
        return "当前 Top 候选不足，建议补充预算、城市和补能条件后再扩大对比。"
    stronger = []
    if item.get("budget_score", 0) >= max(x.get("budget_score", 0) for x in others):
        stronger.append("预算匹配更稳")
    if item.get("space_score", 0) >= max(x.get("space_score", 0) for x in others):
        stronger.append("空间维度更突出")
    if item.get("charging_score", 0) >= max(x.get("charging_score", 0) for x in others):
        stronger.append("补能适配更好")
    return "、".join(stronger) + "，因此排序靠前。" if stronger else "综合分接近，建议通过试驾体验和实际报价进一步区分。"


def _not_recommended_reason(profile: Dict[str, Any], recommendations: List[Dict[str, Any]]) -> List[str]:
    reasons = []
    if profile.get("has_home_charger") is False:
        reasons.append("无家充场景下，对纯电车型需谨慎推荐，除非客户公共补能条件稳定。")
    if profile.get("budget_max"):
        over_budget = [item for item in recommendations if item.get("price_min", 0) > profile["budget_max"]]
        if over_budget:
            reasons.append("部分候选入门价高于预算，不能作为强推荐，只能作为加预算备选。")
    real_low = [item for item in recommendations if item.get("data_quality_score", 100) < 85]
    if real_low:
        reasons.append("真实扩展候选中存在低完整度数据，不应直接用于销售承诺。")
    return reasons or ["暂未发现需要排除的候选，但所有推荐仍需结合官方价格、权益和试驾反馈复核。"]


def _follow_up_actions(profile: Dict[str, Any], top: List[Dict[str, Any]]) -> List[str]:
    first = top[0] if top else {}
    model = f"{first.get('brand', '')} {first.get('model', '')}".strip() or "首推车型"
    actions = [
        f"优先邀约试驾 {model}，现场核验空间、座舱、底盘和辅助驾驶体验。",
        "确认客户可接受预算、金融方案、置换权益和交付周期。",
        "准备 Top 2/Top 3 备选车型报价，避免客户只看单一车型。",
    ]
    if profile.get("has_home_charger") is False:
        actions.insert(1, "补充核验居住地、公司和常去商圈 3km 内公共充电条件。")
    return actions[:4]


def _llm_orchestrator_answer(req: RecommendRequest, profile: Dict[str, Any], pool: str, reason: str, recommendations: List[Dict[str, Any]], risks: List[str], base_answer: str) -> str:
    if not OPENAI_API_KEY or not recommendations:
        return base_answer
    try:
        client = openai_client()
        compact_recommendations = [
            {
                "model": f"{item.get('brand')} {item.get('model')}",
                "score": item.get("score"),
                "energy_type": item.get("energy_type"),
                "vehicle_type": item.get("vehicle_type"),
                "price_min": item.get("price_min"),
                "price_max": item.get("price_max"),
                "range": item.get("cltc_range"),
                "reasons": (item.get("reasons") or [])[:4],
                "cautions": (item.get("cautions") or [])[:3],
                "source": item.get("catalog_source") or item.get("source_type") or "local",
            }
            for item in recommendations[:5]
        ]
        prompt = f"""你是新能源汽车推荐 Agent。请基于后端工具已经完成的画像解析、候选池选择、车型排序和风险检查，生成一份专业、克制、可执行的中文推荐报告。

要求：
1. 必须保留候选池选择原因，不要编造未出现的车型。
2. 只能围绕给定推荐列表说明，不要新增车型。
3. 必须说明价格、续航、配置、交付和辅助驾驶需以官方实时信息核验。
4. 输出结构包含：结论、画像理解、候选池决策、Top车型解释、风险核验、下一步建议。

用户问题：{req.query}
用户画像：{profile}
候选池：{pool}
候选池选择原因：{reason}
推荐列表：{compact_recommendations}
风险：{risks}
基础报告：{base_answer[:1200]}
"""
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
            timeout=60,
        )
        return resp.choices[0].message.content or base_answer
    except Exception:
        return base_answer


def recommend_with_orchestrator(req: RecommendRequest) -> Dict[str, Any]:
    profile = normalize_profile(req.query, req.profile)
    pool, reason = select_candidate_pool(req, profile)
    ranking = _rank_with_pool(req, pool)
    recommendations = ranking.get("recommendations", [])
    feedback_policy_result = apply_feedback_policy(recommendations, pool)
    recommendations = feedback_policy_result["recommendations"]
    evidence = rag_service.retrieve(req.query, 4 if req.use_deep_search else 2)
    risks = _risk_check(profile, recommendations)
    sources = list(ranking.get("sources", [])) + evidence
    explainability = _build_explainability(profile, pool, reason, recommendations, risks, sources)
    base_answer = _format_orchestrator_answer(pool, reason, recommendations, risks)
    answer = _llm_orchestrator_answer(req, profile, pool, reason, recommendations, risks, base_answer)
    obsidian_note = save_recommendation_case(req.query, profile, recommendations, answer, explainability, sources) if recommendations else {}
    trace = [
        {"agent": "ProfileParserTool", "observation": f"已解析画像：预算 {profile.get('budget_max') or '未明确'}，家庭 {profile.get('family_size') or '未明确'} 人，关注 {profile.get('concerns') or '未明确'}"},
        {"agent": "CandidatePoolSelectorTool", "observation": reason},
        {"agent": "RankTool", "observation": f"调用 {pool} 候选池完成排序，返回 {len(recommendations)} 条推荐"},
        {"agent": "FeedbackPolicyTool", "observation": f"应用反馈策略 {len(feedback_policy_result.get('applied_rules', []))} 条，负反馈车型降权、正反馈车型加权"},
        {"agent": "EvidenceRetrievalTool", "observation": f"检索到 {len(evidence)} 条 RAG 证据"},
        {"agent": "RiskCheckTool", "observation": "；".join(risks[:2])},
        {"agent": "LLMReportAgent", "observation": "已调用 ARK 对话 Endpoint 生成可解释推荐报告" if answer != base_answer else "LLM 报告生成不可用，保留后端规则报告"},
        {"agent": "ObsidianCaseWriterTool", "observation": obsidian_note.get("path", "未写入案例")},
    ]
    return {
        "profile": profile,
        "recommendations": recommendations,
        "answer": answer,
        "agent_trace": trace,
        "skill_trace": ranking.get("skill_trace", []),
        "sources": sources,
        "pool_decision": {"selected_pool": pool, "reason": reason, "strategy": getattr(req, "candidate_pool_strategy", "auto") or "auto"},
        "risks": risks,
        "explainability": explainability,
        "feedback_policy": feedback_policy_result,
        "obsidian_note": obsidian_note,
        "catalog_summary": ranking.get("catalog_summary"),
        "candidate_count": ranking.get("candidate_count") or (ranking.get("catalog_summary") or {}).get("total"),
    }
