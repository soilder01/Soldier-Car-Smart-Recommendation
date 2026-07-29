"""
True Multi-Agent system: Supervisor + 3 specialized Workers.

Architecture (langgraph 1.1.9, manually built supervisor):
    START -> Supervisor
                |
                +-> RecommendationWorker (profile + vehicle ranking)
                +-> KnowledgeWorker (RAG + web search)
                +-> SalesWorker (sales script)
                |
                <- back to Supervisor
                |
                -> END (final answer)
"""
import json
import operator
from typing import Annotated, Any, Dict, List, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from openai import OpenAI

from app.config import CHAT_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL, TEMPERATURE
from app.services.agent_graph import (
    extract_user_profile,
    generate_sales_talk,
    retrieve_knowledge_base,
    search_and_rank_vehicles,
    search_web_info,
)
from app.services.llm_client import openai_client


# ============================================================================
# Shared helpers
# ============================================================================

def _build_worker_messages(system_prompt: str, state_messages: list) -> list:
    """Convert LangGraph messages to OpenAI format for a worker with system prompt."""
    msgs = [{"role": "system", "content": system_prompt}]
    for m in (state_messages or []):
        if isinstance(m, HumanMessage):
            msgs.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            entry = {"role": "assistant", "content": m.content or ""}
            reasoning = getattr(m, "reasoning_content", None)
            if reasoning:
                entry["reasoning_content"] = reasoning
            if hasattr(m, "tool_calls") and m.tool_calls:
                entry["tool_calls"] = [
                    {"id": tc["id"] if isinstance(tc, dict) else getattr(tc, "id", ""),
                     "type": "function",
                     "function": {
                         "name": tc["name"] if isinstance(tc, dict) else getattr(tc, "name", ""),
                         "arguments": json.dumps(
                             tc["args"] if isinstance(tc, dict) else getattr(tc, "args", {}),
                             ensure_ascii=False)}
                    } for tc in m.tool_calls]
            msgs.append(entry)
        elif isinstance(m, ToolMessage):
            msgs.append({"role": "tool", "content": m.content,
                         "tool_call_id": m.tool_call_id})
    return msgs


def _build_tool_schemas(tools: list) -> list:
    """Build OpenAI tool schemas from LangChain tools."""
    schemas = []
    for t in tools:
        params = {"type": "object", "properties": {}, "required": []}
        if hasattr(t, "args_schema") and t.args_schema:
            try:
                raw = t.args_schema.model_json_schema()
                params = {"type": "object", "properties": raw.get("properties", {}),
                          "required": raw.get("required", [])}
            except Exception:
                pass
        schemas.append({"type": "function", "function": {
            "name": t.name, "description": t.description, "parameters": params}})
    return schemas


# ============================================================================
# Worker factory — each worker is a compiled StateGraph
# ============================================================================

WORKER_DEFS = {
    "recommendation": {
        "prompt": (
            "你是新能源汽车推荐专家。你的唯一任务是根据用户需求推荐最合适的车型。\n"
            "1. 先调用 extract_user_profile 从用户描述中抽取购车画像\n"
            "2. 再调用 search_and_rank_vehicles 获取评分排序后的推荐车型\n"
            "3. 输出简洁的推荐摘要：Top 3 车型名、价格区间、续航、推荐分、核心理由\n"
            "不要检索知识库，不要联网搜索，不要生成销售话术。只做推荐。"
        ),
        "tools": [extract_user_profile, search_and_rank_vehicles],
    },
    "knowledge": {
        "prompt": (
            "你是新能源汽车知识检索专家。你的唯一任务是通过检索回答用户的知识性问题。\n"
            "1. 调用 retrieve_knowledge_base 检索本地知识库\n"
            "2. 如果问题涉及最新政策、价格、实时信息，调用 search_web_info 联网搜索\n"
            "3. 输出基于证据的简洁回答，标注来源编号 [1][2]\n"
            "不要做车型推荐，不要生成销售话术。只做知识检索。"
        ),
        "tools": [retrieve_knowledge_base, search_web_info],
    },
    "sales": {
        "prompt": (
            "你是新能源汽车销售话术专家。你的唯一任务是为销售顾问生成沟通话术。\n"
            "1. 调用 extract_user_profile 了解客户基本信息\n"
            "2. 如需产品知识支撑，调用 retrieve_knowledge_base\n"
            "3. 调用 generate_sales_talk 生成销售话术\n"
            "4. 输出实用话术方案：开场白、异议处理、促成技巧\n"
            "不要做车型推荐。只做销售沟通指导。"
        ),
        "tools": [extract_user_profile, generate_sales_talk, retrieve_knowledge_base],
    },
}


class WorkerState(TypedDict):
    messages: Annotated[list, operator.add]


def _make_worker(name: str) -> StateGraph:
    """Build a compiled worker agent sub-graph."""
    cfg = WORKER_DEFS[name]
    tools = cfg["tools"]
    prompt = cfg["prompt"]

    def agent_node(state: WorkerState) -> Dict[str, Any]:
        client = openai_client()
        msgs = _build_worker_messages(prompt, state.get("messages", []))
        resp = client.chat.completions.create(
            model=CHAT_MODEL, temperature=TEMPERATURE,
            messages=msgs, tools=_build_tool_schemas(tools))

        msg = resp.choices[0].message
        reasoning = getattr(msg, "reasoning_content", None)
        if msg.tool_calls:
            aim = AIMessage(content=msg.content or "", tool_calls=[
                {"id": tc.id, "name": tc.function.name,
                 "args": json.loads(tc.function.arguments)} for tc in msg.tool_calls])
            if reasoning:
                setattr(aim, "reasoning_content", reasoning)
            return {"messages": [aim]}
        aim = AIMessage(content=msg.content)
        if reasoning:
            setattr(aim, "reasoning_content", reasoning)
        return {"messages": [aim]}

    builder = StateGraph(WorkerState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": END})
    builder.add_edge("tools", "agent")
    return builder.compile()


# ============================================================================
# Supervisor
# ============================================================================

SUPERVISOR_SYSTEM = (
    "你是新能源汽车销售平台的总协调员（Supervisor）。\n\n"
    "你有三个专家可以委派：\n"
    "- recommendation: 负责车型推荐（画像抽取+车型评分排序）\n"
    "- knowledge: 负责知识检索（RAG知识库+联网搜索）\n"
    "- sales: 负责销售话术（沟通技巧+异议处理）\n\n"
    "工作流程：\n"
    "1. 分析用户输入，判断需要哪些专家\n"
    "2. 将任务委派给第一个专家\n"
    "3. 收到专家报告后，判断是否还需要其他专家\n"
    "4. 所有必要专家完成后，汇总输出综合报告\n\n"
    "输出综合报告时，请整合所有专家的结论，标注信息来源。"
)


class SupervisorState(TypedDict):
    messages: Annotated[list, operator.add]
    next: str


def _supervisor_node(state: SupervisorState) -> Dict[str, Any]:
    """Supervisor: decide which worker to call next, or finish."""
    client = openai_client()

    msgs = [{"role": "system", "content": SUPERVISOR_SYSTEM}]
    for m in state.get("messages", []):
        if isinstance(m, HumanMessage):
            msgs.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            entry = {"role": "assistant", "content": m.content or ""}
            reasoning = getattr(m, "reasoning_content", None)
            if reasoning:
                entry["reasoning_content"] = reasoning
            msgs.append(entry)

    msgs.append({"role": "user", "content": (
        "请只回复一个词决定下一步：recommendation、knowledge、sales 或 FINISH。"
        "不要回复其他内容。"
    )})

    resp = client.chat.completions.create(
        model=CHAT_MODEL, temperature=0, messages=msgs, timeout=20)
    decision = resp.choices[0].message.content.strip()

    for option in ["recommendation", "knowledge", "sales"]:
        if option in decision.lower():
            return {"next": option}
    return {"next": "FINISH"}


def _supervisor_final(state: SupervisorState) -> Dict[str, Any]:
    """Generate final comprehensive report after all workers are done."""
    client = openai_client()
    msgs = [{"role": "system", "content": (
        "你是新能源汽车销售平台的总协调员。所有专家已完成工作，"
        "请整合以下专家报告，生成一份专业的综合购车建议报告。"
        "包含：画像摘要、推荐车型、关键证据、销售建议、风险提示。"
    )}]
    for m in state.get("messages", []):
        if isinstance(m, HumanMessage):
            msgs.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            entry = {"role": "assistant", "content": m.content or ""}
            reasoning = getattr(m, "reasoning_content", None)
            if reasoning:
                entry["reasoning_content"] = reasoning
            msgs.append(entry)

    resp = client.chat.completions.create(
        model=CHAT_MODEL, temperature=TEMPERATURE, messages=msgs, timeout=60)
    answer = resp.choices[0].message.content
    answer = skills.compliance_check(answer)
    return {"messages": [AIMessage(content=answer)]}


# ============================================================================
# Worker wrapper nodes
# ============================================================================

_recommendation_worker = _make_worker("recommendation")
_knowledge_worker = _make_worker("knowledge")
_sales_worker = _make_worker("sales")


def _call_recommendation(state: SupervisorState) -> Dict[str, Any]:
    result = _recommendation_worker.invoke({"messages": state["messages"]})
    last = result["messages"][-1] if result["messages"] else AIMessage(content="no result")
    return {"messages": [AIMessage(content=f"[推荐专家报告]\n{last.content}")]}


def _call_knowledge(state: SupervisorState) -> Dict[str, Any]:
    result = _knowledge_worker.invoke({"messages": state["messages"]})
    last = result["messages"][-1] if result["messages"] else AIMessage(content="no result")
    return {"messages": [AIMessage(content=f"[知识检索专家报告]\n{last.content}")]}


def _call_sales(state: SupervisorState) -> Dict[str, Any]:
    result = _sales_worker.invoke({"messages": state["messages"]})
    last = result["messages"][-1] if result["messages"] else AIMessage(content="no result")
    return {"messages": [AIMessage(content=f"[销售话术专家报告]\n{last.content}")]}


def _route_after_supervisor(state: SupervisorState) -> str:
    nxt = state.get("next", "FINISH")
    return nxt if nxt in ("recommendation", "knowledge", "sales") else "FINISH"


# ============================================================================
# Graph assembly
# ============================================================================

_sup_builder = StateGraph(SupervisorState)

_sup_builder.add_node("supervisor", _supervisor_node)
_sup_builder.add_node("recommendation", _call_recommendation)
_sup_builder.add_node("knowledge", _call_knowledge)
_sup_builder.add_node("sales", _call_sales)
_sup_builder.add_node("finalize", _supervisor_final)

_sup_builder.add_edge(START, "supervisor")
_sup_builder.add_conditional_edges("supervisor", _route_after_supervisor, {
    "recommendation": "recommendation",
    "knowledge": "knowledge",
    "sales": "sales",
    "FINISH": "finalize",
})
_sup_builder.add_edge("recommendation", "supervisor")
_sup_builder.add_edge("knowledge", "supervisor")
_sup_builder.add_edge("sales", "supervisor")
_sup_builder.add_edge("finalize", END)

_multi_memory = MemorySaver()
supervisor_graph = _sup_builder.compile(checkpointer=_multi_memory)
