import re
from typing import Dict, List

from openai import OpenAI

from app.config import CHAT_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL, TEMPERATURE
from app.schemas import ChatRequest
from app.services.agent_orchestrator import join_sources
from app.services.llm_client import openai_client
from app.services.skills import skills


def extract_vehicle_mentions(text: str) -> List[str]:
    patterns = [
        r"Model\s*3",
        r"Model\s*Y\s*L",
        r"Model\s*Y",
        r"BMW\s*i?[\w-]+",
        r"奔驰\s*[A-Za-z0-9\u4e00-\u9fa5-]+",
        r"宝马\s*[A-Za-z0-9\u4e00-\u9fa5-]+",
        r"问界\s*[A-Za-z0-9\u4e00-\u9fa5-]+",
        r"享界\s*[A-Za-z0-9\u4e00-\u9fa5-]+",
        r"尊界\s*[A-Za-z0-9\u4e00-\u9fa5-]+",
        r"小鹏\s*[A-Za-z0-9\u4e00-\u9fa5-]+",
        r"理想\s*[A-Za-z0-9\u4e00-\u9fa5-]+",
    ]
    mentions: List[str] = []
    seen = set()
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            name = re.sub(r"\s+", " ", match).strip()
            key = name.lower()
            if name and key not in seen:
                seen.add(key)
                mentions.append(name)
    return mentions[:6]


class CustomerServiceAgent:
    def answer(self, req: ChatRequest) -> Dict:
        skills.reset()
        history_text = "\n".join([f"{item.get('role', '')}: {item.get('content', '')}" for item in req.history[-8:]])
        user_history_text = "\n".join([item.get("content", "") for item in req.history[-8:] if item.get("role") == "user"])
        memory_vehicle_names = extract_vehicle_mentions(user_history_text or history_text)
        current_vehicle_names = extract_vehicle_mentions(req.query)
        active_vehicle_names = current_vehicle_names or memory_vehicle_names
        enriched_query = req.query
        if history_text:
            vehicle_context = f"\n上文涉及车型：{'、'.join(active_vehicle_names)}" if active_vehicle_names else ""
            enriched_query = f"历史对话：\n{history_text}{vehicle_context}\n\n当前问题：{req.query}"
        trace = [
            {"agent": "RouterAgent", "observation": "识别为新能源汽车智能客服咨询"},
            {"agent": "MemoryAgent", "observation": f"读取短期对话记忆 {len(req.history[-8:])} 条"},
            {"agent": "ServiceAgent", "observation": "准备调用 RAG 与 Web Search 获取可追溯资料"},
        ]
        rag_sources = skills.rag_retrieve(enriched_query, req.top_k)
        web_sources = skills.web_search(f"{enriched_query} 新能源汽车 官方 参数 政策", 4) if req.use_web_search else []
        sources: List[Dict] = web_sources + rag_sources
        trace.append({"agent": "ResearchAgent", "observation": f"获得 {len(sources)} 条证据，其中联网结果 {len(web_sources)} 条"})

        if OPENAI_API_KEY:
            client = openai_client()
            prompt = f"""你是新能源汽车企业智能客服，请基于 RAG 知识库和联网搜索证据回答用户问题。

要求：
1. 先给直接结论。
2. 再按要点解释。
3. 涉及价格、配置、权益、政策时提示以官方实时信息为准。
4. 涉及辅助驾驶时明确它不能替代驾驶员。
5. 使用 [1][2] 引用证据。

用户问题：{req.query}
短期对话记忆：
{history_text or "暂无"}
上文重点车型：{("、".join(active_vehicle_names)) if active_vehicle_names else "暂无"}

证据：
{join_sources(sources)}
"""
            resp = client.chat.completions.create(
                model=CHAT_MODEL,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = resp.choices[0].message.content
        else:
            answer = "结论：已结合本地知识库"
            if web_sources:
                answer += "和联网搜索结果"
            answer += "给出参考回答。\n\n"
            for i, src in enumerate(sources[:5], 1):
                if src.get("source") == "web":
                    answer += f"{i}. 联网资料：{src.get('title')} [{i}]\n"
                else:
                    answer += f"{i}. {src.get('content', '')[:180]}... [{i}]\n"
            answer += "\n风险提示：价格、配置、权益、政策和辅助驾驶功能边界以官方实时信息为准。"
            if history_text:
                context_line = f"本轮已继承上文车型：{'、'.join(active_vehicle_names)}。\n\n" if active_vehicle_names else ""
                answer = f"我已结合上一轮对话上下文进行理解。\n\n{context_line}{answer}"

        answer = skills.compliance_check(answer)
        trace.append({"agent": "ReflectionAgent", "observation": "完成客服回答合规检查"})
        return {"answer": answer, "sources": sources, "agent_trace": trace, "skill_trace": skills.trace}


customer_service_agent = CustomerServiceAgent()
