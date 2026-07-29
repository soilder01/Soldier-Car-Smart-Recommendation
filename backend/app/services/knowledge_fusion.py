from typing import Any, Dict

from app.services.obsidian_vault import graph
from app.services.rag import rag_service


def knowledge_fusion_status() -> Dict[str, Any]:
    rag_stats = rag_service.stats()
    vault = graph()
    obsidian_chunks = rag_stats.get("obsidian_chunks", 0)
    node_count = vault.get("stats", {}).get("node_count", 0)
    return {
        "summary": {
            "mode": "Obsidian长期记忆 + RAG即时检索",
            "rag_chunks": rag_stats.get("chunks", 0),
            "rag_documents": rag_stats.get("documents", 0),
            "obsidian_nodes": node_count,
            "obsidian_chunks": obsidian_chunks,
            "linked": obsidian_chunks > 0 and node_count > 0,
        },
        "roles": [
            {"name": "Obsidian", "role": "长期知识库", "description": "沉淀推荐案例、反馈复盘、测试评估、治理报告和交付报告"},
            {"name": "RAG", "role": "即时取证", "description": "从知识库与 Obsidian Vault 中检索相关片段，支撑 Agent 推荐解释"},
            {"name": "Agent", "role": "推理决策", "description": "结合画像、车型数据、RAG证据和反馈策略生成推荐"},
        ],
        "rag": rag_stats,
        "obsidian": vault.get("stats", {}),
    }


def rebuild_knowledge_fusion() -> Dict[str, Any]:
    rag_service.rebuild()
    return knowledge_fusion_status()
