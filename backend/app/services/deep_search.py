from typing import Dict, List

from app.schemas import DeepSearchRequest, RecommendRequest, UserProfile
from app.services.agent_orchestrator import orchestrator
from app.services.skills import skills


class DeepSearchEngine:
    def decompose(self, query: str) -> List[str]:
        tasks = ["抽取购车画像", "检索新能源知识库", "召回车型库", "计算推荐评分", "生成竞品对比", "生成销售跟进建议", "执行合规检查"]
        if any(word in query for word in ["对比", "相比", "怎么选"]):
            tasks.insert(4, "识别用户指定车型并进行参数对比")
        if any(word in query for word in ["续航", "长途", "高速"]):
            tasks.insert(2, "检索续航与补能知识")
        return tasks

    def run(self, req: DeepSearchRequest) -> Dict:
        tasks = self.decompose(req.query)
        response = orchestrator.recommend(RecommendRequest(query=req.query, profile=UserProfile(), top_k=req.top_k, use_deep_search=True))
        stages = []
        for i, task in enumerate(tasks, 1):
            stages.append({
                "step": i,
                "task": task,
                "status": "completed",
                "observation": "已完成" if i < len(tasks) else "已汇总为最终报告",
            })
        return {
            "query": req.query,
            "tasks": stages,
            "profile": response.profile,
            "recommendations": response.recommendations,
            "answer": response.answer,
            "agent_trace": response.agent_trace,
            "skill_trace": skills.trace,
            "sources": response.sources,
        }


deep_search_engine = DeepSearchEngine()
