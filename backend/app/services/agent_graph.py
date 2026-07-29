"""LangGraph-based multi-agent recommendation graph.

Step 1: Linear pipeline that replicates the existing MultiAgentOrchestrator.recommend()
behavior exactly, using LangGraph's StateGraph as the execution framework.
"""

import json
import operator
import re
from typing import Annotated, Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from app.config import CHAT_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL, TEMPERATURE
from app.database import save_recommendation_log
from app.schemas import RecommendRequest, RecommendResponse, UserProfile
from app.services.agent_orchestrator import (
    build_web_search_query,
    detect_recommend_scene,
    join_sources,
    social_luxury_candidates,
)
from app.services.llm_client import openai_client
from app.services.skills import skills


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class RecommendState(TypedDict):
    query: str
    profile: Annotated[Dict[str, Any], "user profile dict"]
    top_k: int
    use_deep_search: bool

    # intermediate results
    scene: str
    recommendations: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]
    web_sources: List[Dict[str, Any]]
    sales_script: Dict[str, str]
    answer: str

    # trace (same format as before)
    agent_trace: Annotated[List[Dict[str, Any]], operator.add]


# ---------------------------------------------------------------------------
# Node functions – one per step of the original pipeline
# ---------------------------------------------------------------------------

def _flatten_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure profile dict keys are strings and values are JSON-safe."""
    return {str(k): v for k, v in profile.items()}


def node_reset_and_route(state: RecommendState) -> Dict[str, Any]:
    """Reset skills trace and perform LLM-based intent routing."""
    skills.reset()
    route = _llm_route_intent(state["query"])
    return {
        "agent_trace": [
            {"agent": "RouterAgent", "intent": route["intent"], "observation": route["observation"]}
        ]
    }


def _llm_route_intent(query: str) -> Dict[str, str]:
    """Use LLM to classify user intent. Falls back to keyword routing on failure."""
    if not OPENAI_API_KEY:
        return _keyword_route(query)

    client = openai_client()
    prompt = (
        "你是新能源汽车销售平台的意图分类器。"
        "分析用户输入，将其归入以下四类之一：\n"
        "- recommend: 用户想买车、求推荐、有预算/需求描述\n"
        "- compare: 用户在对比车型、问差异、怎么选\n"
        "- sales: 销售顾问询问话术、异议处理、客户沟通技巧\n"
        "- knowledge: 纯知识问答（技术原理、政策解读等），不涉及具体购车决策\n\n"
        "请只返回一个 JSON 对象，格式严格为："
        '{"intent":"<类别>","reason":"<一句话理由>"}\n\n'
        f"用户输入：{query}"
    )
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            timeout=15,
        )
        raw = resp.choices[0].message.content.strip()
        # extract JSON from response
        import json as _json
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        route = _json.loads(raw)
        intent = route.get("intent", "knowledge")
        reason = route.get("reason", "LLM 分类完成")
        valid = {"recommend", "compare", "sales", "knowledge"}
        if intent not in valid:
            intent = "knowledge"
        return {"intent": intent, "observation": f"LLM 路由识别为 {intent}：{reason}"}
    except Exception:
        return _keyword_route(query)


def _keyword_route(query: str) -> Dict[str, str]:
    """Fallback keyword-based routing."""
    if any(w in query for w in ["对比", "相比", "怎么选", "差异"]):
        return {"intent": "compare", "observation": "关键词路由识别为 compare"}
    if any(w in query for w in ["话术", "客户", "销售", "异议"]):
        return {"intent": "sales", "observation": "关键词路由识别为 sales"}
    if any(w in query for w in ["推荐", "买", "预算", "家用", "通勤"]):
        return {"intent": "recommend", "observation": "关键词路由识别为 recommend"}
    return {"intent": "knowledge", "observation": "关键词路由识别为 knowledge"}


def node_extract_profile(state: RecommendState) -> Dict[str, Any]:
    profile = skills.extract_profile(
        state["query"], UserProfile(**state["profile"])
    )
    return {
        "profile": profile,
        "agent_trace": [
            {"agent": "ProfileAgent", "observation": "完成用户画像抽取", "profile": profile}
        ],
    }


def node_detect_scene(state: RecommendState) -> Dict[str, Any]:
    scene = detect_recommend_scene(state["query"], state["profile"])
    return {
        "scene": scene,
        "agent_trace": [
            {"agent": "SceneAgent", "observation": f"识别推荐场景为 {scene}"}
        ],
    }


def node_vehicle_recall_and_rank(state: RecommendState) -> Dict[str, Any]:
    skills.vehicle_recall(state["profile"])
    ranked = skills.vehicle_rank(
        state["query"], UserProfile(**state["profile"]), state["top_k"]
    )
    return {
        "recommendations": ranked["recommendations"],
        "agent_trace": [
            {"agent": "RecommenderAgent", "observation": f"输出 Top {len(ranked['recommendations'])} 推荐车型"}
        ],
    }


def node_retrieve_sources(state: RecommendState) -> Dict[str, Any]:
    rag_sources = skills.rag_retrieve(state["query"], 6)
    web_sources = (
        skills.web_search(build_web_search_query(state["query"], state["scene"]), 6)
        if state["use_deep_search"]
        else []
    )
    sources = web_sources + rag_sources
    trace = []
    if state["use_deep_search"] and web_sources:
        if state["scene"] == "social_luxury":
            web_candidates = social_luxury_candidates(
                web_sources, state["recommendations"], state["profile"], state["query"],
                max(state["top_k"], 5),
            )
            if web_candidates:
                state["recommendations"] = (web_candidates + state["recommendations"])[
                    : max(state["top_k"], 5)
                ]
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
                "observation": f"联网搜索返回 {len(web_sources)} 条公开资料，作为模型综合推荐证据",
            })
    trace.append({
        "agent": "ResearchAgent",
        "observation": f"检索知识证据 {len(sources)} 条，其中联网结果 {len(web_sources)} 条",
    })
    return {
        "sources": sources,
        "web_sources": web_sources,
        "recommendations": state["recommendations"],
        "agent_trace": trace,
    }


def node_analysis_and_sales(state: RecommendState) -> Dict[str, Any]:
    recs = state["recommendations"]
    if recs:
        skills.range_estimate(recs[0], state["profile"])
        skills.budget_analysis(recs[0], state["profile"])
    script = skills.sales_script(state["profile"], recs)
    return {
        "sales_script": script,
        "agent_trace": [
            {"agent": "SalesAgent", "observation": "生成销售辅助话术"}
        ],
    }


def node_generate_answer(state: RecommendState) -> Dict[str, Any]:
    answer = _generate_answer(
        state["query"],
        state["profile"],
        state["recommendations"],
        state["sources"],
        state["sales_script"],
    )
    answer = skills.compliance_check(answer)
    return {
        "answer": answer,
        "agent_trace": [
            {"agent": "ReflectionAgent", "observation": "完成合规检查和风险表达修正"}
        ],
    }


def _generate_answer(
    query: str,
    profile: Dict,
    recommendations: List[Dict],
    sources: List[Dict],
    sales_script: Dict,
) -> str:
    top_lines = []
    for i, item in enumerate(recommendations[:5], 1):
        if item.get("source_type") == "web":
            top_lines.append(
                f"{i}. {item['brand']} {item['model']}：联网候选，来源：{item.get('source_title', '')}。"
                f"理由：{'；'.join(item['reasons'][:2])}"
            )
            continue
        top_lines.append(
            f"{i}. {item['brand']} {item['model']}：推荐分 {item['score']}，"
            f"价格 {item['price_min']//10000}-{item['price_max']//10000} 万，"
            f"{item['energy_type']}，CLTC {item['cltc_range']} km。"
            f"理由：{'；'.join(item['reasons'][:3])}"
        )

    social_note = ""
    concerns = profile.get("concerns", [])
    if any(x in concerns for x in ["社交形象", "内饰氛围", "品牌"]):
        social_note = (
            "你提到的“撩妹”我会按更专业的"
            "“社交形象/约会第一印象/个人气质匹配”来分析："
            "车辆能增强外在呈现，但真正的吸引力还取决于谈吐、审美和相处体验。\n\n"
        )

    fallback = (
        f"结论：根据当前画像，推荐优先关注 {recommendations[0]['brand']} {recommendations[0]['model']}。\n\n"
        f"用户画像摘要：预算上限 {profile.get('budget_max') or '未明确'}，"
        f"家庭人数 {profile.get('family_size') or '未明确'}，"
        f"通勤 {profile.get('commute_km') or '未明确'} km，"
        f"充电条件 {profile.get('has_home_charger')}，"
        f"关注点 {profile.get('concerns') or '综合体验'}。\n\n"
        f"{social_note}"
        f"Hybrid 检索补充：本次推荐同时参考本地结构化车型库、本地 RAG 知识库和 Web Search 公开资料；"
        f"卡片来自本地结构化车型库，联网资料用于补充价格、政策、口碑和实时信息判断，"
        f"最终仍建议以官方实时信息为准。\n\n"
        f"推荐车型：\n{chr(10).join(top_lines)}\n\n"
        f"销售建议：\n"
        f"- 开场：{sales_script.get('opening')}\n"
        f"- 推荐：{sales_script.get('recommendation')}\n"
        f"- 异议处理：{sales_script.get('objection')}\n"
        f"- 下一步：{sales_script.get('next_action')}\n\n"
        f"风险提示：实际价格、权益、续航和辅助驾驶可用范围以官方实时信息和试驾体验为准。"
    )

    if not OPENAI_API_KEY:
        return fallback

    allowed_models = [f"{item['brand']} {item['model']}" for item in recommendations[:5]]
    client = openai_client()
    prompt = (
        f"你是新能源汽车销售推荐专家。请基于用户画像、本地结构化车型评分、本地 RAG 知识库证据、"
        f"联网 Web Search 公开资料和销售话术，生成专业、克制、可执行的购车建议。\n\n"
        f"要求：\n"
        f"1. 先给明确推荐结论。\n"
        f"2. 说明用户画像和关键约束。\n"
        f"3. 输出 Top 车型推荐理由。\n"
        f"4. 车型推荐只能围绕“允许推荐车型列表”中的车型展开；"
        f"联网搜索资料只能作为补充证据，"
        f"不能把网页标题或未结构化车型当作推荐卡片。\n"
        f"5. 如果联网资料与本地车型库存在差异，请说明"
        f"“需以官方实时信息核验”，不要直接覆盖结构化参数。\n"
        f"6. 给出销售跟进话术。\n"
        f"7. 只有资料中存在明确 source/title/url 时才使用引用；不要编造 [1][2] 来源标题。\n"
        f"8. 避免绝对化表达，不要夸大辅助驾驶或续航。\n"
        f"9. 如果用户使用“撩妹”等说法，请转化为"
        f"“社交形象、约会第一印象、个人气质匹配”来专业回答。\n\n"
        f"用户问题：{query}\n"
        f"用户画像：{profile}\n"
        f"允许推荐车型列表：{allowed_models}\n"
        f"推荐结果：{recommendations[:5]}\n"
        f"销售话术：{sales_script}\n"
        f"知识库证据：\n{join_sources(sources)}\n"
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def node_save_log(state: RecommendState) -> Dict[str, Any]:
    save_recommendation_log(state["query"], state["profile"], state["recommendations"])
    return {}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

_builder = StateGraph(RecommendState)

_builder.add_node("reset_and_route", node_reset_and_route)
_builder.add_node("extract_profile", node_extract_profile)
_builder.add_node("detect_scene", node_detect_scene)
_builder.add_node("vehicle_recall_and_rank", node_vehicle_recall_and_rank)
_builder.add_node("retrieve_sources", node_retrieve_sources)
_builder.add_node("analysis_and_sales", node_analysis_and_sales)
_builder.add_node("generate_answer", node_generate_answer)
_builder.add_node("save_log", node_save_log)

_builder.add_edge(START, "reset_and_route")
_builder.add_edge("reset_and_route", "extract_profile")
_builder.add_edge("extract_profile", "detect_scene")
_builder.add_edge("detect_scene", "vehicle_recall_and_rank")
_builder.add_edge("vehicle_recall_and_rank", "retrieve_sources")
_builder.add_edge("retrieve_sources", "analysis_and_sales")
_builder.add_edge("analysis_and_sales", "generate_answer")
_builder.add_edge("generate_answer", "save_log")
_builder.add_edge("save_log", END)

recommend_graph = _builder.compile()


# ============================================================================
# Tool-based Agent Graph (ReAct pattern)
# ============================================================================

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool as langchain_tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from app.database import list_vehicles, match_mentioned_vehicles
from app.services.recommender import score_vehicle


# ---------------------------------------------------------------------------
# Tool definitions – each wraps one Skill
# ---------------------------------------------------------------------------

@langchain_tool
def extract_user_profile(query: str, budget_max: int = 0) -> str:
    """从用户的自然语言输入中抽取结构化购车画像。调用后返回用户画像JSON。

    Args:
        query: 用户的自然语言购车需求描述
        budget_max: 用户给出的预算上限（元），如未明确则为0
    """
    from app.schemas import UserProfile as _UP
    profile = _UP(budget_max=budget_max if budget_max > 0 else None)
    result = skills.extract_profile(query, profile)
    return json.dumps(result, ensure_ascii=False)


def _compact_vehicle_name(value: str) -> str:
    return re.sub(r"[\s\-_.]+", "", value).casefold()


def _resolve_named_vehicle(
    name: str,
    catalog: list[dict],
) -> dict | None:
    """Resolve a user-supplied model name without cross-brand model collisions."""
    requested = _compact_vehicle_name(name)
    exact_full_matches = []
    exact_model_matches = []
    partial_matches = []
    for vehicle in catalog:
        brand = _compact_vehicle_name(vehicle["brand"])
        model = _compact_vehicle_name(vehicle["model"])
        full_name = f"{brand}{model}"
        if requested == full_name:
            exact_full_matches.append(vehicle)
        elif requested == model:
            exact_model_matches.append(vehicle)
        elif brand in requested and model in requested:
            partial_matches.append(vehicle)

    for matches in (
        exact_full_matches,
        exact_model_matches,
        partial_matches,
    ):
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return None

    fallback_matches = match_mentioned_vehicles(name)
    if len(fallback_matches) == 1:
        return fallback_matches[0]
    return None


@langchain_tool
def search_and_rank_vehicles(budget_max: int = 0, preferred_type: str = "",
                              preferred_energy: str = "", concerns: str = "",
                              top_k: int = 5,
                              model_names: list[str] = []) -> str:
    """在本地38款新能源车型库中按画像检索和评分排序。返回Top推荐车型列表JSON。

    Args:
        budget_max: 预算上限（元）
        preferred_type: 偏好车型（SUV/轿车/MPV/跑车等），空字符串表示不限
        preferred_energy: 偏好能源（纯电/插混/增程），空字符串表示不限
        concerns: 用户关注点，逗号分隔，如“续航,空间,智驾”
        top_k: 返回Top N车型，默认5
        model_names: compare 场景下用户点名的车型名列表。存在时优先精确/模糊查库，
            返回 named_vehicles、supplemental_vehicles 与 named_vehicle_lookup。
    """
    from app.schemas import UserProfile as _UP
    concerns_list = [c.strip() for c in concerns.split(",") if c.strip()] if concerns else []
    profile = _UP(
        budget_max=budget_max if budget_max > 0 else None,
        preferred_type=preferred_type,
        preferred_energy=preferred_energy,
        concerns=concerns_list,
    )
    profile_dict = profile.model_dump()
    requested_names = [
        name.strip()
        for name in model_names
        if isinstance(name, str) and name.strip()
    ]
    energy_policy = _energy_policy(preferred_energy)
    vehicles = []
    for vehicle in list_vehicles():
        if budget_max and vehicle["price_min"] > budget_max * 1.25:
            continue
        if preferred_type and preferred_type not in vehicle["vehicle_type"]:
            continue
        energy_rank = energy_policy.get(vehicle["energy_type"])
        if energy_policy and energy_rank is None:
            continue
        vehicles.append(
            _score_tool_vehicle(
                vehicle,
                profile_dict,
                energy_rank or 0,
            )
        )
    if not vehicles:
        ranked = skills.vehicle_rank("", profile, top_k)
        recs = ranked["recommendations"]
    else:
        vehicles.sort(
            key=lambda item: (
                item.get("_energy_rank", 0),
                -float(item["score"]),
                item["price_min"],
                item["brand"],
                item["model"],
            )
        )
        recs = vehicles[:top_k]
    summary = [_tool_vehicle_card(r, i + 1) for i, r in enumerate(recs)]
    if not requested_names:
        return json.dumps(summary, ensure_ascii=False)

    named_rows: list[dict] = []
    missing_names: list[str] = []
    seen_named_ids: set[int] = set()
    catalog = list_vehicles()
    for name in requested_names:
        match = _resolve_named_vehicle(name, catalog)
        if match is not None and match["id"] in seen_named_ids:
            match = None
        if match is None:
            missing_names.append(name)
            continue
        seen_named_ids.add(match["id"])
        named_rows.append(_score_tool_vehicle(match, profile_dict, 0))

    named_cards = [
        _tool_vehicle_card(row, index + 1)
        for index, row in enumerate(named_rows)
    ]
    supplemental_cards = [
        card for card in summary
        if card["full_name"] not in {
            item["full_name"] for item in named_cards
        }
    ][:max(0, top_k - len(named_cards))]
    lookup = {
        "requested_model_names": requested_names,
        "resolved_model_names": [
            item["full_name"] for item in named_cards
        ],
        "missing_model_names": missing_names,
        "named_vehicle_missing": bool(missing_names),
    }
    return json.dumps(
        {
            "named_vehicle_lookup": lookup,
            "named_vehicles": named_cards,
            "supplemental_vehicles": supplemental_cards,
        },
        ensure_ascii=False,
    )


@langchain_tool
def retrieve_knowledge_base(query: str) -> str:
    """检索本地新能源汽车知识库（选购指南、技术路线、电池安全等12篇文档）。
    返回相关证据片段列表JSON。

    Args:
        query: 要检索的问题或关键词
    """
    sources = skills.rag_retrieve(query, 5)
    result = [{
        "rank": s["rank"],
        "source": s["source"],
        "domain": s["domain"],
        "content": s["content"][:200],
        "score": s["score"],
    } for s in sources]
    return json.dumps(result, ensure_ascii=False)


@langchain_tool
def search_web_info(query: str) -> str:
    """联网搜索新能源汽车最新价格、政策、口碑等公开信息。仅在需要实时数据时调用。
    返回搜索结果列表JSON。

    Args:
        query: 搜索关键词（建议包含车型名+关注维度）
    """
    results = skills.web_search(query, 5)
    if not results:
        return json.dumps([], ensure_ascii=False)
    return json.dumps([{
        "title": r.get("title", ""),
        "url": r.get("url", ""),
        "content": r.get("content", "")[:150],
    } for r in results], ensure_ascii=False)


@langchain_tool
def generate_sales_talk(budget_max: int = 0, concerns: str = "",
                         top_model: str = "") -> str:
    """根据用户画像和推荐车型生成销售顾问跟进话术。

    Args:
        budget_max: 预算上限（元）
        concerns: 关注点，逗号分隔
        top_model: 推荐的首选车型全名，如“小鹏 G6”
    """
    concerns_list = [c.strip() for c in concerns.split(",") if c.strip()] if concerns else []
    profile = {"budget_max": budget_max, "concerns": concerns_list}
    recs = [{"brand": top_model.split()[0] if top_model else "?",
              "model": " ".join(top_model.split()[1:]) if " " in top_model else top_model}]
    script = skills.sales_script(profile, recs if top_model else [])
    return json.dumps(script, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    query: str
    intent: str  # recommend / compare / knowledge / sales
    profile: Annotated[Dict[str, Any], lambda left, right: {**left, **right}]
    recommendations: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]
    answer: str
    agent_trace: Annotated[List[Dict[str, Any]], operator.add]


TOOLS = [extract_user_profile, search_and_rank_vehicles, retrieve_knowledge_base,
         search_web_info, generate_sales_talk]


def _energy_policy(preferred_energy: str) -> dict[str, int]:
    value = (preferred_energy or "").strip().casefold()
    if not value or value in {"新能源", "不限"}:
        return {}
    if "纯电" in value or "bev" in value:
        return {"纯电": 0}
    if "增程" in value:
        return {"增程": 0}
    if "插混" in value or "phev" in value or "混动" in value:
        return {"插混": 0, "增程": 1}
    return {preferred_energy: 0}


def _score_tool_vehicle(vehicle: dict, profile: dict, energy_rank: int) -> dict:
    scored = score_vehicle(vehicle, profile)
    scored["_energy_rank"] = energy_rank
    budget_max = profile.get("budget_max") or 300000
    mid_price = (vehicle["price_min"] + vehicle["price_max"]) / 2
    budget_distance = abs(mid_price - budget_max) / max(budget_max, 1)
    adjusted = float(scored["score"]) - min(budget_distance * 5, 6)
    if energy_rank:
        adjusted -= 4
    scored["score"] = round(max(0, min(100, adjusted)), 1)
    if energy_rank:
        scored["selection_note"] = (
            "增程算作广义插电备选；如用户严格要求 PHEV 可进一步筛选"
        )
        scored["reasons"] = [
            "增程属于广义插电范畴，作为插混需求的备选",
            *scored["reasons"],
        ]
    else:
        scored["selection_note"] = "能源类型精确匹配"
    return scored


def _wan(value: int | float) -> str:
    return f"{float(value) / 10000:.2f}".rstrip("0").rstrip(".")


def _energy_evidence(vehicle: dict) -> list[str]:
    text = f"{vehicle.get('suitable_scenarios', '')};{vehicle.get('highlights', '')}"
    return [
        item.strip()
        for item in text.split(";")
        if item.strip() and any(word in item for word in ("油耗", "能耗", "省油", "低使用成本"))
    ]


def _tool_vehicle_card(vehicle: dict, rank: int) -> dict:
    """Return grounded vehicle evidence with human-readable specs."""
    specs = {
        "price_range": f"{_wan(vehicle['price_min'])}-{_wan(vehicle['price_max'])}万",
        "match_score": f"{vehicle['score']}分",
        "cltc_range": f"{vehicle['cltc_range']}km",
        "battery": f"{vehicle['battery_kwh']}kWh",
        "fast_charge": f"{vehicle['fast_charge_minutes']}分钟",
        "seats": f"{vehicle['seats']}座",
        "drive_type": vehicle["drive_type"],
        "adas_level": vehicle["adas_level"],
        "smart_cockpit": vehicle["smart_cockpit"],
        "wheelbase": f"{vehicle['wheelbase']}mm",
        "trunk_volume": f"{vehicle['trunk_volume']}L",
        "safety_score": f"{vehicle['safety_score']}分",
        "monthly_sales": f"{vehicle['monthly_sales']}辆/月",
    }
    return {
        "rank": rank,
        "brand": vehicle["brand"],
        "model": vehicle["model"],
        "full_name": f"{vehicle['brand']}{vehicle['model']}",
        "score": vehicle["score"],
        "price": specs["price_range"],
        "energy": vehicle["energy_type"],
        "vehicle_type": vehicle["vehicle_type"],
        "range_km": vehicle["cltc_range"],
        "specs": specs,
        "energy_evidence": _energy_evidence(vehicle),
        "known_missing_specs": [
            "official_fuel_consumption_l_per_100km",
            "official_power_consumption_kwh_per_100km",
            "warranty_policy",
        ],
        "reasons": vehicle["reasons"][:3],
        "cautions": vehicle["cautions"][:2],
        "highlights": vehicle.get("highlights", ""),
        "weaknesses": vehicle.get("weaknesses", ""),
        "selection_note": vehicle.get("selection_note", ""),
    }

# Intent-specific tool subsets
TOOLS_BY_INTENT = {
    "recommend": TOOLS,
    "compare": [extract_user_profile, search_and_rank_vehicles, retrieve_knowledge_base, search_web_info],
    "knowledge": [retrieve_knowledge_base, search_web_info],
    "sales": [extract_user_profile, generate_sales_talk, retrieve_knowledge_base],
    "customer_service": [retrieve_knowledge_base, search_web_info],
    "deep_search": [extract_user_profile, search_and_rank_vehicles, retrieve_knowledge_base, search_web_info, generate_sales_talk],
}

PROMPTS_BY_INTENT = {
    "recommend": """你是新能源汽车智能推荐助手。用户想买车，请按以下步骤工作：

1. 先调用 extract_user_profile 抽取购车画像
2. 再调用 search_and_rank_vehicles 获取推荐车型
3. 根据需要调用 retrieve_knowledge_base 和 search_web_info 补充证据
4. 调用 generate_sales_talk 生成销售话术
5. 汇总输出专业购车推荐报告（包含画像摘要、Top车型对比、推荐理由、销售建议、风险提示）
6. 价格、续航、电池、快充、轴距、后备箱、座位数及其他一切硬指标，只能逐字使用 search_and_rank_vehicles 返回的 specs
7. 禁止凭模型记忆补充或改写任何精确数值；specs 未返回的字段一律标注“需官方核验”
8. 能源相关判断只能复述 energy_evidence，不得扩写成工具未返回的能耗或成本结论
9. 任何带单位数字必须整段复制某个 specs 显示值：价格只能复制完整 price_range，禁止抽取起售价、最低价或区间端点，禁止单位换算、阈值归纳、差值计算或百分比推导
10. monthly_sales 必须保持“辆/月”原文，不得换算成“万”；政策、权益、保修、交付等缺失项除“需官方核验”外不要展开
11. 强制输出结构：车型硬指标只能出现在 Top 车型表格中，表格单元格逐字复制 specs；表格之外不得出现任何车型数字、数值门槛、区间端点或“超过/以内”等派生数值结论
12. 车型名称中的数字不是规格，严禁从 L6、M7、009 等名称推导座位数、尺寸或版本；座位数只能逐字复制 specs.seats
13. 销售建议只允许建议试驾、补充用户需求和核对表格内车型，不得讨论价格优惠、金融政策、购车权益、保修或交付周期""",

    "compare": """你是新能源汽车竞品对比助手。用户想对比车型，请按以下步骤工作：

1. 调用 extract_user_profile 抽取用户画像
2. 调用 search_and_rank_vehicles，并将用户点名的两台车型以 model_names 数组传入，优先取回 named_vehicles 的真实 specs
3. 调用 retrieve_knowledge_base 获取车型技术对比信息
4. 如需要最新价格政策，调用 search_web_info
前 3 步均为不可省略的前置工具调用；未完成前不得输出最终答案
5. named_vehicle_missing=false 时，必须以 named_vehicles 的两台点名车做真并列对比表；supplemental_vehicles 仅作补充
6. named_vehicle_missing=true 时，必须明确写“库中无此车规格”，列出 missing_model_names，并把 supplemental_vehicles 标为邻近对比，不得编造缺车参数
7. 缺车分支的第一句必须为“库中无此车规格”，然后列出 missing_model_names；不得把缺车车型写入参数表
8. 真并列对比表只允许展示 named_vehicles 的 specs 原文；价格只能复制完整 price_range，禁止写预算可覆盖、起售价、价格带、差值、阈值、百分比或跨车型拼接区间
9. 表格之外不写任何带单位的车型数值；选车建议只写场景取舍，不讨论价格优惠、金融、权益、保修、交付或数字化门槛
10. 输出结构化的竞品对比报告（价格、续航、空间、智驾、亮点/短板），并给出选车建议

注意：不要生成销售话术，聚焦于客观对比分析。""",

    "knowledge": """你是新能源汽车知识问答助手。用户问的是技术/政策/原理类问题，请按以下步骤工作：

1. 调用 retrieve_knowledge_base 检索本地知识库
2. 如需要最新政策或实时信息，调用 search_web_info 联网搜索
3. 基于证据给出专业、准确的回答
4. 引用来源，标注 [1][2]
5. 不要做车型推荐，不要生成销售话术""",

    "sales": """你是新能源汽车销售话术助手。销售顾问需要应对客户异议或沟通技巧，请按以下步骤工作：

1. 调用 extract_user_profile 了解客户画像
2. 调用 retrieve_knowledge_base 检索相关销售话术和产品知识
3. 调用 generate_sales_talk 生成专业的销售跟进话术
4. 输出实用的沟通建议（开场白、异议处理、促成技巧）
5. 不要做车型推荐，聚焦于销售沟通技巧
6. 不得承诺或暗示任何未由工具明确返回的保修、质保、权益、补贴、赠品、免费服务、最低价、锁价、交付周期或优惠名额
7. 涉及保修、权益、补贴、价格、赠品、免费服务时，只能引导核验官方实时公开资料，不要写具体承诺话术
8. 除非工具结果逐字返回，不要在销售话术中写具体分钟、公里、百分比、金额、年限、次数等数字；看车路线、试驾安排和讲解流程用非数字化表达即可
9. 不要把禁止项写成具体示例清单，避免在最终话术中复述“最低价、锁价、赠送、免费”等未核验承诺词""",

    "customer_service": """你是新能源汽车智能客服助手。用户在咨询售后、政策、配置、使用边界或历史上下文问题。

1. 优先调用 retrieve_knowledge_base 查找可追溯资料
2. 只有涉及实时政策、价格、权益、门店、官方参数更新时才调用 search_web_info
3. 不要调用 search_and_rank_vehicles 或 generate_sales_talk，除非用户明确要求购车推荐；本意图默认不推荐车型
4. 回答要克制，无法由工具证据支持的信息要说明需以官方实时信息核验
5. 涉及辅助驾驶时必须说明不能替代驾驶员
6. 工具未返回保修、质保、权益、补贴、赠品、免费服务、交付或价格政策时，
   只能写“该项工具未返回，需以官方实时信息核验”；不得复述、评价、否定、
   概括或承诺用户提到的具体政策词，也不得给出查询渠道、适用条件或地区规则
7. 客服问题不得夹带车型、能源路线、预算或购车建议；除非工具逐字返回，否则
   不得将用户所在城市、家庭/通勤身份扩写为任何专属权益、政策或使用建议""",

    "deep_search": """你是新能源汽车深度检索助手。用户需要更充分的选车调研或复杂问题拆解。

必须先按顺序完成前四步，不要先查资料再补车型：
1. 调用 extract_user_profile 抽取画像和约束
2. 调用 search_and_rank_vehicles 获取结构化候选和 specs
3. 调用 retrieve_knowledge_base 检索本地知识库证据
4. 调用 search_web_info 获取必要的公开实时信息
5. 观察以上工具结果后，再决定是否调用 generate_sales_talk 补充跟进建议
6. 最终回答必须体现 observe -> decide 的多轮综合，而不是只做单轮推荐
7. 价格、续航、电池、快充、轴距、后备箱、座位数及其他一切硬指标，只能逐字使用 search_and_rank_vehicles 返回的 specs
8. 禁止凭模型记忆补充或改写任何精确数值；specs 未返回的字段一律标注“需官方核验”
9. 能源相关判断只能复述 energy_evidence，不得扩写成工具未返回的能耗或成本结论
10. 任何带单位数字必须整段复制某个 specs 显示值：价格只能复制完整 price_range，禁止抽取起售价、最低价或区间端点，禁止单位换算、阈值归纳、差值计算或百分比推导
11. monthly_sales 必须保持“辆/月”原文，不得换算成“万”；政策、权益、保修、交付等缺失项除“需官方核验”外不要展开
12. 强制输出结构：车型硬指标只能出现在候选车型表格中，表格单元格逐字复制 specs；表格之外不得出现任何车型数字、数值门槛、区间端点或“超过/以内”等派生数值结论
13. 车型名称中的数字不是规格，严禁从 L6、M7、009 等名称推导座位数、尺寸或版本；座位数只能逐字复制 specs.seats
14. 跟进建议只允许建议试驾、补充用户需求和核对表格内车型，不得讨论价格优惠、金融政策、购车权益、保修或交付周期
15. 用户未明确预算时不得自行设定任何预算档位、价格带或“某万级”标题；不得为通勤、长途、补能等场景自造公里、分钟、百分比或次数阈值
16. 分析正文不得出现任何带单位数字；座位数、尺寸、续航和价格只能在候选车型表格内按 specs 原文展示
17. 最终回答要结构化但精炼，保留关键证据和结论，不要长篇复述整段 observation""",
}


def _get_tools_for_intent(intent: str) -> list:
    return TOOLS_BY_INTENT.get(intent, TOOLS)


def _get_prompt_for_intent(intent: str) -> str:
    base = PROMPTS_BY_INTENT.get(intent, PROMPTS_BY_INTENT["recommend"])
    return base + (
        "\n\n约束：search_and_rank_vehicles 返回 specs 时，优先使用 specs 中的具体价格、"
        "续航、电池、快充、轴距、后备箱、座位数等字段给出具体建议；"
        "不得把工具中的精确数值四舍五入或粗略改写，例如不要把 12.98-16.98万"
        "写成 13-17万；"
        "只有工具未返回的价格、政策、权益、能耗、保修等字段才提示以官方实时信息核验。"
        "能耗相关非数值判断只能复述工具返回的 energy_evidence 原文，不要扩写成"
        "未返回的成本对比或程度判断。"
        "辅助驾驶不能替代驾驶员；"
        "避免绝对化表达；只有工具结果存在明确 source/title/url 时才使用引用，"
        "不得编造 [1][2] 来源标题。"
        "\n能源口径：增程算作广义插电范畴，可作为插混需求的备选并明确说明；"
        "如用户严格要求 PHEV，则只保留插混车型。纯电车型不能作为插混需求的推荐候选。"
    )


def _build_openai_messages(state: AgentState) -> list:
    """Build OpenAI-format messages from LangGraph state."""
    intent = state.get("intent", "recommend")
    prompt = _get_prompt_for_intent(intent)
    openai_msgs = [{"role": "system", "content": prompt}]

    if not state.get("messages"):
        openai_msgs.append({"role": "user", "content": state["query"]})
        return openai_msgs

    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            openai_msgs.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            entry = {"role": "assistant", "content": m.content or ""}
            reasoning = getattr(m, "reasoning_content", None)
            if reasoning:
                entry["reasoning_content"] = reasoning
            if hasattr(m, "tool_calls") and m.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc["id"] if isinstance(tc, dict) else getattr(tc, "id", "call_0"),
                        "type": "function",
                        "function": {
                            "name": tc["name"] if isinstance(tc, dict) else getattr(tc, "name", ""),
                            "arguments": json.dumps(
                                tc["args"] if isinstance(tc, dict) else getattr(tc, "args", {}),
                                ensure_ascii=False,
                            ),
                        },
                    }
                    for tc in m.tool_calls
                ]
            openai_msgs.append(entry)
        elif isinstance(m, ToolMessage):
            openai_msgs.append({
                "role": "tool",
                "content": m.content,
                "tool_call_id": m.tool_call_id,
            })
    return openai_msgs


def _build_tool_schemas(intent: str = "recommend") -> list:
    """Build OpenAI tool schemas from LangChain tools, filtered by intent."""
    tools = _get_tools_for_intent(intent)
    schemas = []
    for t in tools:
        params = {"type": "object", "properties": {}, "required": []}
        if hasattr(t, "args_schema") and t.args_schema:
            try:
                raw = t.args_schema.model_json_schema()
                params = {
                    "type": "object",
                    "properties": raw.get("properties", {}),
                    "required": raw.get("required", []),
                }
            except Exception:
                pass
        schemas.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": params,
            }
        })
    return schemas


def node_route_intent(state: AgentState) -> Dict[str, Any]:
    """Classify user intent before entering the agent loop.
    On subsequent turns: keeps existing intent, appends new query to messages."""
    existing_intent = state.get("intent", "")
    existing_messages = state.get("messages", [])
    query = state.get("query", "")

    if existing_intent and existing_messages:
        # Check if this query is already the last user message (checkpoint replay)
        last_user = ""
        for m in reversed(existing_messages):
            if isinstance(m, HumanMessage):
                last_user = m.content
                break
        if last_user != query and query:
            # Subsequent turn with new query
            return {
                "messages": [HumanMessage(content=query)],
                "agent_trace": [
                    {"agent": "RouterAgent", "intent": existing_intent,
                     "observation": f"多轮对话追加新问题，保持意图={existing_intent}"}
                ],
            }
        # Replay (already has this message) — keep intent
        return {
            "agent_trace": [
                {"agent": "RouterAgent", "intent": existing_intent,
                 "observation": f"多轮对话继续，意图={existing_intent}"}
            ],
        }

    # First turn: classify intent
    skills.reset()
    route = _llm_route_intent(query) if query else {"intent": "recommend", "observation": "默认意图"}
    return {
        "intent": route["intent"],
        "agent_trace": [
            {"agent": "RouterAgent", "intent": route["intent"],
             "observation": route["observation"]}
        ],
    }


def _agent_node(state: AgentState) -> Dict[str, Any]:
    """LLM agent node: decides which tools to call or generates final answer."""
    client = openai_client()

    intent = state.get("intent", "recommend")
    openai_msgs = _build_openai_messages(state)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=TEMPERATURE,
        messages=openai_msgs,
        tools=_build_tool_schemas(intent),
    )

    msg = resp.choices[0].message
    reasoning = getattr(msg, "reasoning_content", None)
    if msg.tool_calls:
        aim = AIMessage(
            content=msg.content or "",
            tool_calls=[{
                "id": tc.id,
                "name": tc.function.name,
                "args": json.loads(tc.function.arguments),
            } for tc in msg.tool_calls]
        )
        if reasoning:
            setattr(aim, "reasoning_content", reasoning)
        return {"messages": [aim]}
    aim = AIMessage(content=msg.content)
    if reasoning:
        setattr(aim, "reasoning_content", reasoning)
    return {"messages": [aim], "answer": msg.content}


def _should_continue(state: AgentState) -> str:
    """Decide whether to call tools or finish."""
    last_msg = state["messages"][-1] if state.get("messages") else None
    if last_msg and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "end"


def _finalize(state: AgentState) -> Dict[str, Any]:
    """Post-process: extract answer, run compliance check, build detailed trace."""
    answer = state.get("answer", "")
    if not answer and state.get("messages"):
        last = state["messages"][-1]
        answer = last.content if hasattr(last, "content") else ""
    answer = skills.compliance_check(answer)

    # Build detailed trace from messages
    trace = _build_detailed_trace(state)
    return {"answer": answer, "agent_trace": trace}


def _build_detailed_trace(state: AgentState) -> List[Dict[str, Any]]:
    """Parse messages list to generate a step-by-step tool call trace."""
    trace = []
    for m in state.get("messages", []):
        if not isinstance(m, AIMessage):
            continue
        tc = getattr(m, "tool_calls", None)
        if not tc:
            continue
        for call in tc:
            name = call["name"] if isinstance(call, dict) else getattr(call, "name", "?")
            args = call.get("args", {}) if isinstance(call, dict) else getattr(call, "args", {})
            # Build a human-readable observation
            obs = _format_tool_call(name, args)
            trace.append({"agent": "ToolCall", "observation": obs})

    trace.append({"agent": "ToolAgent",
                  "observation": "Tool 调用完成，合规检查通过，生成最终回答"})
    return trace


def _format_tool_call(name: str, args: Dict) -> str:
    """Format a tool call into a readable observation string."""
    if name == "extract_user_profile":
        budget = args.get("budget_max", 0)
        query_preview = (args.get("query", "") or "")[:40]
        return f"调用 extract_user_profile（budget_max={budget}，query={query_preview}...）"
    elif name == "search_and_rank_vehicles":
        return f"调用 search_and_rank_vehicles（budget_max={args.get('budget_max',0)}，type={args.get('preferred_type','')}，energy={args.get('preferred_energy','')}，concerns={args.get('concerns','')}，model_names={args.get('model_names',[])}，top_k={args.get('top_k',5)}）"
    elif name == "retrieve_knowledge_base":
        q = (args.get("query", "") or "")[:60]
        return f"调用 retrieve_knowledge_base（query={q}...）"
    elif name == "search_web_info":
        q = (args.get("query", "") or "")[:60]
        return f"调用 search_web_info（query={q}...）"
    elif name == "generate_sales_talk":
        return f"调用 generate_sales_talk（budget_max={args.get('budget_max',0)}，concerns={args.get('concerns','')}，top_model={args.get('top_model','')}）"
    else:
        return f"调用 {name}"


# ---------------------------------------------------------------------------
# Tool-based graph assembly (with conditional intent routing)
# ---------------------------------------------------------------------------

_tool_builder = StateGraph(AgentState)

_tool_builder.add_node("route_intent", node_route_intent)
_tool_builder.add_node("agent", _agent_node)
_tool_builder.add_node("tools", ToolNode(TOOLS))
_tool_builder.add_node("finalize", _finalize)

_tool_builder.add_edge(START, "route_intent")
_tool_builder.add_edge("route_intent", "agent")
_tool_builder.add_conditional_edges("agent", _should_continue, {"tools": "tools", "end": "finalize"})
_tool_builder.add_edge("tools", "agent")
_tool_builder.add_edge("finalize", END)

_memory_saver = MemorySaver()
tool_agent_graph = _tool_builder.compile(checkpointer=_memory_saver)
