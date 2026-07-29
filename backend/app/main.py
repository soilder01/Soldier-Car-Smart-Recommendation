import asyncio
import json as _json
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage

from app.config import CHAT_MODEL, CONTENT_GENERATION_MODEL, CONTENT_GENERATION_TASK_URL, CONTENT_GENERATION_TYPE, EMBEDDING_BASE_URL, EMBEDDING_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL, TAVILY_API_KEY
from app.database import clear_runtime_data, create_lead, init_db, list_leads, list_vehicles, recommendation_feedback_summary, save_recommendation_feedback
from app.schemas import ChatRequest, CompareRequest, DeepSearchRequest, LeadCreate, RecommendRequest, RecommendResponse, RecommendationFeedbackCreate, UserProfile
from app.services.acceptance_report import run_acceptance_report
from app.services.delivery_package import generate_delivery_package
from app.services.agent_graph import tool_agent_graph
from app.services.agent_orchestrator import recommend_with_orchestrator
from app.services.llm_client import check_chat_model, mask_secret
from app.services.multi_agent import supervisor_graph
from app.services.analytics import dashboard_summary
from app.services.rag import rag_service
from app.services.skills import skills
from app.services.obsidian_vault import graph as obsidian_graph, list_recommendation_cases, seed_from_project_data, save_recommendation_case
from app.services.demo_data import seed_demo_data
from app.services.recommender import profile_preview
from app.services.evaluation import run_agent_regression_evaluation, run_recommendation_evaluation
from app.services.feedback_notes import save_feedback_note
from app.services.fused_catalog import fused_catalog, recommend_fused
from app.services.optimization import generate_feedback_review, generate_optimization_insights
from app.services.real_data_governance import generate_real_data_governance
from app.services.real_world_data import real_world_overview
from app.services.real_world_recommender import recommend_real_world
from app.services.release_gate import release_gate
from app.services.system_readiness import system_readiness
from app.services.knowledge_fusion import knowledge_fusion_status, rebuild_knowledge_fusion


app = FastAPI(title="agent汽车智能推荐平台", version="2.0.0")
START_TIME = time.time()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    print(
        "effective_chat_config "
        f"base_url={OPENAI_BASE_URL} "
        f"model={CHAT_MODEL} "
        f"api_key={mask_secret(OPENAI_API_KEY)} "
        f"api_key_length={len(OPENAI_API_KEY or '')}",
        flush=True,
    )
    init_db()

def _extract_skill_outputs() -> dict:
    profile = {}
    recommendations = []
    sources = []
    for entry in skills.trace:
        name = entry.get("skill")
        payload = entry.get("payload")
        if name == "extract_profile_skill" and isinstance(payload, dict):
            profile = payload
        elif name == "vehicle_rank_skill" and isinstance(payload, list):
            recommendations = payload
        elif name in {"rag_retrieve_skill", "web_search_skill"} and isinstance(payload, list):
            sources.extend(payload)
    return {"profile": profile, "recommendations": recommendations, "sources": sources}


def _fallback_recommendation_result(req: RecommendRequest, error: Exception) -> dict:
    skills.reset()
    final_profile = skills.extract_profile(req.query, req.profile)
    ranked = skills.vehicle_rank(req.query, UserProfile(**final_profile), req.top_k)
    sources = skills.rag_retrieve(req.query, 4)
    recommendations = ranked["recommendations"]
    top_lines = []
    for index, item in enumerate(recommendations[:5], 1):
        reasons = "；".join(item.get("reasons", [])[:3])
        top_lines.append(
            f"{index}. {item['brand']} {item['model']}：推荐分 {item['score']}，{reasons}"
        )
    answer = (
        "当前已启用本地兜底推荐链路：Agent 大模型链路暂不可用，系统仍基于画像解析、车型库评分和 RAG 知识库完成推荐。\n\n"
        f"画像摘要：预算 {final_profile.get('budget_max') or '未明确'}，城市 {final_profile.get('city') or '未明确'}，"
        f"家庭人数 {final_profile.get('family_size') or '未明确'}，通勤 {final_profile.get('commute_km') or '未明确'} km，"
        f"关注点 {final_profile.get('concerns') or '综合体验'}。\n\n"
        f"推荐车型：\n{chr(10).join(top_lines)}\n\n"
        "风险提示：价格、权益、续航和辅助驾驶可用范围以官方实时信息及试驾体验为准。"
    )
    obsidian_note = save_recommendation_case(req.query, final_profile, recommendations, answer)
    return {
        "answer": answer,
        "recommendations": recommendations,
        "profile": final_profile,
        "sources": sources,
        "skill_trace": skills.trace,
        "obsidian_note": obsidian_note,
        "trace_count": 1,
        "fallback_reason": str(error),
    }


def _run_agent(query: str, profile: dict = None, thread_id: str = "",
               force_intent: str = "") -> dict:
    """Unified agent runner: all endpoints share this single LangGraph engine."""
    import uuid
    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    state_input = {"query": query}
    if not thread_id or force_intent:
        # First turn or forced intent: set profile and optional intent override
        state_input["profile"] = profile or {}
        state_input["answer"] = ""
        if force_intent:
            state_input["intent"] = force_intent

    state = tool_agent_graph.invoke(state_input, config)

    answer = state.get("answer", "")
    if not answer and state.get("messages"):
        for m in reversed(state["messages"]):
            if isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None):
                answer = m.content
                break
    return {
        "profile": state.get("profile", {}),
        "recommendations": state.get("recommendations", []),
        "answer": answer,
        "agent_trace": state.get("agent_trace", []),
        "skill_trace": skills.trace,
        "sources": state.get("sources", []),
    }


# ---- Unified endpoints ----

@app.get("/api/health")
def health_check():
    vehicles = list_vehicles()
    rag_stats = rag_service.stats()
    return {
        "status": "ok",
        "service": "Soldier-Car-Smart-Recommendation backend",
        "version": app.version,
        "uptime_seconds": round(time.time() - START_TIME),
        "vehicle_count": len(vehicles),
        "rag_chunks": rag_stats.get("chunks", 0),
        "openai_configured": bool(OPENAI_API_KEY),
        "chat_model": CHAT_MODEL,
    }

@app.get("/api/dashboard/summary")
def get_dashboard():
    return dashboard_summary()


@app.get("/api/vehicles")
def get_vehicles():
    return {"vehicles": list_vehicles()}


@app.post("/api/recommend")
def recommend(req: RecommendRequest):
    """LangGraph tool-calling Agent — LLM auto-detects intent and orchestrates tools."""
    try:
        result = _run_agent(req.query, req.profile.model_dump(), req.thread_id)
        return RecommendResponse(**result)
    except Exception as e:
        return _fallback_recommendation_result(req, e)


@app.post("/api/agent/recommend")
def agent_recommend(req: RecommendRequest):
    return recommend_with_orchestrator(req)


@app.post("/api/profile/preview")
def preview_profile(req: RecommendRequest):
    return profile_preview(req.query, req.profile)


@app.post("/api/recommend-multi")
def recommend_multi(req: RecommendRequest):
    """Multi-Agent Supervisor — 3 specialized Workers coordinated by Supervisor."""
    import uuid
    from langchain_core.messages import HumanMessage

    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    state = supervisor_graph.invoke(
        {"messages": [HumanMessage(content=req.query)]}, config)

    # Extract final answer from messages
    answer = ""
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage) and m.content and len(m.content) > 100:
            answer = m.content
            break

    return RecommendResponse(
        profile=req.profile.model_dump(),
        recommendations=[],
        answer=answer,
        agent_trace=[{"agent": "Supervisor",
                      "observation": "Multi-Agent: Supervisor + RecommendationWorker + KnowledgeWorker + SalesWorker"}],
        skill_trace=skills.trace,
        sources=[],
    )


@app.post("/api/recommend-stream")
async def recommend_stream(req: RecommendRequest):
    """SSE streaming endpoint — emits trace events progressively, result at end."""
    import uuid
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    state_input = {"query": req.query, "profile": req.profile.model_dump(), "answer": ""}

    async def event_generator():
        try:
            trace_count = 0
            answer_text = ""
            async for chunk in tool_agent_graph.astream(state_input, config, stream_mode="updates"):
                for node_name, node_output in chunk.items():
                    if node_name == "route_intent":
                        for t in node_output.get("agent_trace", []):
                            trace_count += 1
                            yield f"event: trace\ndata: {_json.dumps({'agent': t['agent'], 'observation': t['observation']}, ensure_ascii=False)}\n\n"
                    elif node_name == "tools":
                        yield f"event: trace\ndata: {_json.dumps({'agent': 'ToolNode', 'observation': 'Tool 执行完成，Agent 继续思考...'}, ensure_ascii=False)}\n\n"
                    elif node_name == "agent":
                        msgs = node_output.get("messages", [])
                        for m in msgs:
                            tc = getattr(m, "tool_calls", None) if hasattr(m, "tool_calls") else None
                            if tc:
                                for call in tc:
                                    name = call.get("name", "?") if isinstance(call, dict) else getattr(call, "name", "?")
                                    trace_count += 1
                                    yield f"event: trace\ndata: {_json.dumps({'agent': 'ToolCall', 'observation': f'调用 {name}'}, ensure_ascii=False)}\n\n"
                            elif getattr(m, "content", "") and not tc:
                                answer_text = m.content
                    elif node_name == "finalize":
                        answer_text = node_output.get("answer", answer_text)
                        final_trace = node_output.get("agent_trace", [])
                        for t in final_trace:
                            trace_count += 1
                            yield f"event: trace\ndata: {_json.dumps(t, ensure_ascii=False)}\n\n"

            skill_outputs = _extract_skill_outputs()
            final_profile = skill_outputs["profile"] or req.profile.model_dump()
            final_recommendations = skill_outputs["recommendations"]
            final_sources = skill_outputs["sources"]
            obsidian_note = {}
            if final_recommendations:
                obsidian_note = save_recommendation_case(
                    req.query, final_profile, final_recommendations, answer_text
                )

            result = {
                "answer": answer_text,
                "recommendations": final_recommendations,
                "profile": final_profile,
                "sources": final_sources,
                "skill_trace": skills.trace,
                "obsidian_note": obsidian_note,
                "trace_count": trace_count,
            }
            yield f"event: result\ndata: {_json.dumps(result, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: {}\n\n"

        except Exception as e:
            yield f"event: trace\ndata: {_json.dumps({'agent': 'FallbackAgent', 'observation': 'Agent 链路异常，已切换本地推荐兜底链路'}, ensure_ascii=False)}\n\n"
            result = _fallback_recommendation_result(req, e)
            yield f"event: result\ndata: {_json.dumps(result, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/api/customer-service/chat")
def customer_service_chat(req: ChatRequest):
    """LangGraph tool-calling Agent — forced customer_service intent."""
    result = _run_agent(query=req.query, thread_id="",
                        force_intent="customer_service")
    return {"answer": result["answer"], "sources": result["sources"],
            "agent_trace": result["agent_trace"], "skill_trace": result["skill_trace"]}


@app.post("/api/deep-search")
def deep_search(req: DeepSearchRequest):
    """LangGraph agent with deep search enabled (web search tool available)."""
    result = _run_agent(req.query)
    return {
        "query": req.query,
        "profile": result["profile"],
        "recommendations": result["recommendations"],
        "answer": result["answer"],
        "agent_trace": result["agent_trace"],
        "skill_trace": result["skill_trace"],
        "sources": result["sources"],
    }


# ---- Car image search ----

@app.get("/api/car-image")
def car_image(brand: str = "", model: str = ""):
    """Search for real car photos using Tavily image search."""
    from langchain_tavily import TavilySearch

    query = f"{brand} {model} 新能源汽车 官方图片"
    try:
        searcher = TavilySearch(
            tavily_api_key=TAVILY_API_KEY,
            max_results=3,
            include_images=True,
            include_answer=False,
        )
        raw = searcher.invoke(query)
        images = raw.get("images", []) if isinstance(raw, dict) else []
        if images:
            url = images[0] if isinstance(images[0], str) else images[0].get("url", "")
            return {"image_url": url, "query": query}
    except Exception:
        pass
    return {"image_url": "", "query": query}


# ---- Local-only endpoints (no LLM required) ----

@app.post("/api/compare")
def compare(req: CompareRequest):
    skills.reset()
    result = skills.compare_vehicle(req.models, req.profile)
    return {"result": result, "skill_trace": skills.trace}


@app.post("/api/rag/chat")
def rag_chat(req: ChatRequest):
    sources = rag_service.retrieve(req.query, req.top_k)
    answer = "根据知识库检索结果，建议从使用场景、预算、补能条件和安全配置四个维度综合判断。\n\n"
    for i, src in enumerate(sources[:4], 1):
        answer += f"{i}. {src['content'][:160]}... [{i}]\n"
    answer += "\n风险提示：价格、权益、质保和辅助驾驶范围以官方实时说明为准。"
    return {"answer": answer, "sources": sources}


# ---- Data / admin endpoints ----

@app.post("/api/leads")
def add_lead(req: LeadCreate):
    return create_lead(req.model_dump())


@app.get("/api/leads")
def get_leads():
    return {"leads": list_leads()}


@app.post("/api/recommendation-feedback")
def add_recommendation_feedback(req: RecommendationFeedbackCreate):
    feedback = save_recommendation_feedback(req.model_dump())
    note = save_feedback_note(feedback)
    return {"feedback": feedback, "obsidian_note": note, "summary": recommendation_feedback_summary()}


@app.get("/api/recommendation-feedback/summary")
def get_recommendation_feedback_summary():
    return recommendation_feedback_summary()


@app.post("/api/rag/rebuild")
def rebuild_rag():
    return rebuild_knowledge_fusion()


@app.get("/api/knowledge/fusion-status")
def get_knowledge_fusion_status():
    return knowledge_fusion_status()


@app.post("/api/evaluation/recommendation")
def evaluate_recommendation_quality():
    return run_recommendation_evaluation()


@app.post("/api/evaluation/agent-regression")
def evaluate_agent_regression():
    return run_agent_regression_evaluation()


@app.get("/api/real-world/overview")
def get_real_world_overview(limit: int = 30):
    return real_world_overview(limit)


@app.post("/api/real-world/governance")
def refresh_real_world_governance():
    return generate_real_data_governance(True)


@app.post("/api/real-world/recommend")
def real_world_recommend(req: RecommendRequest):
    return recommend_real_world(req)


@app.get("/api/catalog/fused")
def get_fused_catalog(limit_real: int = 220):
    catalog = fused_catalog(limit_real)
    return {"summary": catalog["summary"], "vehicles": catalog["vehicles"][:80]}


@app.post("/api/recommend-fused")
def fused_recommend(req: RecommendRequest):
    return recommend_fused(req)


@app.get("/api/feedback/review")
def get_feedback_review():
    return generate_feedback_review()


@app.get("/api/optimization/insights")
def get_optimization_insights():
    return generate_optimization_insights()


@app.get("/api/obsidian/graph")
def get_obsidian_graph():
    return obsidian_graph()


@app.get("/api/obsidian/recommendation-cases")
def get_obsidian_recommendation_cases(limit: int = 20):
    return list_recommendation_cases(limit)


@app.post("/api/obsidian/seed-project-data")
def seed_obsidian_project_data():
    return seed_from_project_data()


@app.get("/api/config/public")
def public_config():
    return {
        "base_url": "已配置（已隔离）" if OPENAI_BASE_URL else "未配置",
        "base_url_configured": bool(OPENAI_BASE_URL),
        "chat_model": "已配置（已隔离）" if CHAT_MODEL else "未配置",
        "chat_model_configured": bool(CHAT_MODEL),
        "api_key_configured": bool(OPENAI_API_KEY),
        "api_key_masked": mask_secret(OPENAI_API_KEY),
        "chat": {
            "configured": bool(OPENAI_API_KEY and OPENAI_BASE_URL and CHAT_MODEL),
            "base_url_configured": bool(OPENAI_BASE_URL),
            "model_configured": bool(CHAT_MODEL),
            "model": "已配置（已隔离）" if CHAT_MODEL else "未配置",
        },
        "embedding": {
            "base_url_configured": bool(EMBEDDING_BASE_URL),
            "model_configured": bool(EMBEDDING_MODEL),
            "model": "已配置（预留）" if EMBEDDING_MODEL else "未配置（当前RAG未使用embedding API）",
        },
        "content_generation": {
            "configured": bool(CONTENT_GENERATION_MODEL and CONTENT_GENERATION_TASK_URL),
            "type": CONTENT_GENERATION_TYPE,
            "model": "已配置（已隔离）" if CONTENT_GENERATION_MODEL else "未配置",
            "task_url": "已配置（已隔离）" if CONTENT_GENERATION_TASK_URL else "未配置",
        },
    }


@app.get("/api/config/llm-check")
def llm_check(model: str = ""):
    target_model = model or CHAT_MODEL
    result = check_chat_model(target_model)
    return {
        "base_url": "已配置（已隔离）" if OPENAI_BASE_URL else "未配置",
        "chat_model": "已配置（已隔离）" if target_model else "未配置",
        "api_key_configured": bool(OPENAI_API_KEY),
        "api_key_masked": mask_secret(OPENAI_API_KEY),
        "available": result.get("ok", False),
        "check": result,
    }


@app.get("/api/system/readiness")
def get_system_readiness():
    return system_readiness()


@app.get("/api/system/release-gate")
def get_release_gate():
    return release_gate()


@app.post("/api/system/acceptance-report")
def create_acceptance_report():
    return run_acceptance_report(True)


@app.post("/api/system/delivery-package")
def create_delivery_package():
    return generate_delivery_package(True)


@app.post("/api/demo/seed")
def seed_demo():
    result = seed_demo_data()
    return {"result": result, "summary": dashboard_summary()}


@app.post("/api/admin/clear-runtime-data")
def clear_data():
    return {"result": clear_runtime_data(), "summary": dashboard_summary()}
