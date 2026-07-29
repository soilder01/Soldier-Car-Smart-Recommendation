from typing import Dict, List

from app.database import find_vehicles_by_models, list_vehicles, match_mentioned_vehicles
from app.schemas import UserProfile
from app.services.rag import rag_service
from app.services.recommender import normalize_profile, recommend, score_vehicle


class SkillRegistry:
    def __init__(self):
        self.trace: List[Dict] = []

    def reset(self):
        self.trace = []

    def record(self, name: str, observation: str, payload=None):
        self.trace.append({"skill": name, "observation": observation, "payload": payload})

    def extract_profile(self, query: str, profile: UserProfile) -> Dict:
        result = normalize_profile(query, profile)
        self.record("extract_profile_skill", "完成自然语言购车画像抽取", result)
        return result

    def vehicle_recall(self, profile: Dict) -> List[Dict]:
        vehicles = list_vehicles()
        mentioned = profile.get("mentioned_models") or []
        if mentioned:
            rows = [v for v in vehicles if f"{v['brand']} {v['model']}" in mentioned]
            self.record("vehicle_recall_skill", f"用户点名车型优先召回 {len(rows)} 款", {"models": mentioned})
            return rows
        budget_max = profile.get("budget_max")
        preferred_type = profile.get("preferred_type")
        preferred_energy = profile.get("preferred_energy")
        rows = []
        for vehicle in vehicles:
            if budget_max and vehicle["price_min"] > budget_max * 1.25:
                continue
            if preferred_type and preferred_type not in vehicle["vehicle_type"]:
                continue
            if preferred_energy and preferred_energy != vehicle["energy_type"]:
                continue
            rows.append(vehicle)
        if not rows:
            rows = vehicles
        self.record("vehicle_recall_skill", f"召回候选车型 {len(rows)} 款", {"count": len(rows)})
        return rows

    def vehicle_rank(self, query: str, profile: UserProfile, top_k: int = 5) -> Dict:
        result = recommend(query, profile, top_k)
        self.record("vehicle_rank_skill", f"完成车型评分排序，输出 Top {len(result['recommendations'])}", result["recommendations"])
        return result

    def range_estimate(self, vehicle: Dict, profile: Dict) -> Dict:
        cltc = vehicle["cltc_range"]
        factor = 0.85
        if profile.get("long_trip_frequency") in {"经常", "高"}:
            factor = 0.72
        if "冬季" in profile.get("city", ""):
            factor = min(factor, 0.72)
        estimate = int(cltc * factor)
        result = {"model": vehicle["model"], "cltc_range": cltc, "estimated_real_range": estimate, "factor": factor}
        self.record("range_estimate_skill", f"{vehicle['model']} 估算实际续航 {estimate} km", result)
        return result

    def budget_analysis(self, vehicle: Dict, profile: Dict) -> Dict:
        price = int((vehicle["price_min"] + vehicle["price_max"]) / 2)
        down_payment = int(price * 0.3)
        loan = price - down_payment
        monthly = int(loan / 36)
        energy_cost_year = 2600 if vehicle["energy_type"] == "纯电" else 5200
        result = {
            "model": vehicle["model"],
            "estimated_price": price,
            "down_payment_30_percent": down_payment,
            "monthly_36": monthly,
            "estimated_energy_cost_year": energy_cost_year,
        }
        self.record("budget_analysis_skill", f"{vehicle['model']} 完成预算测算", result)
        return result

    def compare_vehicle(self, models: List[str], profile: UserProfile) -> Dict:
        vehicles = find_vehicles_by_models(models)
        if not vehicles:
            vehicles = match_mentioned_vehicles(" ".join(models))
        profile_dict = profile.model_dump()
        scored = [score_vehicle(vehicle, profile_dict) for vehicle in vehicles]
        self.record("compare_vehicle_skill", f"完成 {len(scored)} 款车型对比", scored)
        return {"vehicles": scored}

    def rag_retrieve(self, query: str, top_k: int = 6) -> List[Dict]:
        sources = rag_service.retrieve(query, top_k)
        self.record("rag_retrieve_skill", f"召回知识库证据 {len(sources)} 条", sources)
        return sources

    def web_search(self, query: str, top_k: int = 5) -> List[Dict]:
        try:
            from app.config import TAVILY_API_KEY
            if not TAVILY_API_KEY:
                self.record("web_search_skill", "Tavily API Key 未配置，跳过联网搜索", [])
                return []

            from langchain_tavily import TavilySearch
            searcher = TavilySearch(
                tavily_api_key=TAVILY_API_KEY,
                max_results=top_k,
                search_depth="basic",
                include_answer=False,
                include_images=False,
            )
            raw = searcher.invoke(query)
            results = raw.get("results", []) if isinstance(raw, dict) else []
            rows = []
            for i, r in enumerate(results[:top_k], 1):
                rows.append({
                    "rank": i,
                    "title": r.get("title", "")[:120],
                    "url": r.get("url", ""),
                    "source": "web",
                    "domain": "Tavily 联网搜索",
                    "content": r.get("content", "")[:300],
                    "score": round(r.get("score", 1.0), 4),
                })
            self.record("web_search_skill", f"Tavily 联网搜索返回 {len(rows)} 条结果", rows)
            return rows
        except Exception as exc:
            self.record("web_search_skill", f"Tavily 搜索失败，已回退本地知识库：{exc}", [])
            return []

    def sales_script(self, profile: Dict, recommendations: List[Dict]) -> Dict:
        top = recommendations[0] if recommendations else {}
        concerns = "、".join(profile.get("concerns", [])) or "预算和综合体验"
        model_name = f"{top.get('brand', '')} {top.get('model', '')}".strip()
        script = {
            "opening": f"我先根据您的预算、通勤和关注点做一个匹配分析，重点看{concerns}。",
            "recommendation": f"目前匹配度最高的是 {model_name}，主要原因是它在预算、续航和家庭场景上比较均衡。",
            "objection": "如果您担心续航、电池安全或保修边界，我们可以先梳理使用场景；具体质保和政策信息需以品牌官方实时公开资料核验。",
            "next_action": "建议安排一次试驾，并同步整理待核验问题清单，如配置、价格、政策和交付周期。",
        }
        self.record("sales_script_skill", "生成销售跟进话术", script)
        return script

    def compliance_check(self, text: str) -> str:
        replacements = {
            "绝对安全": "安全配置较完整",
            "自动驾驶": "辅助驾驶",
            "吊打": "在部分维度更有优势",
            "一定省钱": "长期使用成本有机会降低",
            "续航不打折": "实际续航受场景影响",
            "保证最低价": "价格以官方实时政策为准",
        }
        revised = text
        for old, new in replacements.items():
            revised = revised.replace(old, new)
        self.record("compliance_check_skill", "完成合规表达检查", {"changed": revised != text})
        return revised


skills = SkillRegistry()
