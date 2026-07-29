from pathlib import Path

from app.services.knowledge_fusion import knowledge_fusion_status, rebuild_knowledge_fusion
from app.services.rag import rag_service

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUT_JSON = PROJECT_DIR / "data" / "real_world" / "knowledge_fusion_evaluation.json"
OUT_MD = PROJECT_DIR / "data" / "real_world" / "knowledge_fusion_evaluation.md"


def main():
    status = rebuild_knowledge_fusion()
    stats = status["rag"]
    query = "Obsidian 推荐案例 反馈复盘 Agent"
    results = rag_service.retrieve(query, 8)
    obsidian_hits = [item for item in results if str(item.get("domain", "")).startswith("Obsidian/")]
    checks = [
        {"name": "RAG统计包含Obsidian文档数", "passed": "obsidian_documents" in stats},
        {"name": "RAG统计包含Obsidian片段数", "passed": "obsidian_chunks" in stats},
        {"name": "Obsidian文档已被读取", "passed": stats.get("obsidian_documents", 0) > 0},
        {"name": "Obsidian片段已进入RAG", "passed": stats.get("obsidian_chunks", 0) > 0},
        {"name": "知识图谱节点可读取", "passed": status["obsidian"].get("node_count", 0) > 0},
        {"name": "联动状态为已联动", "passed": status["summary"].get("linked") is True},
        {"name": "RAG检索可命中Obsidian来源", "passed": len(obsidian_hits) > 0},
    ]
    passed = sum(1 for item in checks if item["passed"])
    report = {
        "stage": "H1 RAG 与 Obsidian 联动增强",
        "passed": passed,
        "total": len(checks),
        "success": passed == len(checks),
        "summary": status["summary"],
        "checks": checks,
        "sample_query": query,
        "sample_hits": obsidian_hits[:3],
    }
    OUT_JSON.write_text(__import__("json").dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# H1 RAG 与 Obsidian 联动增强评估",
        "",
        f"- 结论：{'通过' if report['success'] else '未通过'}",
        f"- 检查：{passed}/{len(checks)}",
        f"- RAG 总片段：{status['summary'].get('rag_chunks', 0)}",
        f"- Obsidian 入库片段：{status['summary'].get('obsidian_chunks', 0)}",
        f"- Obsidian 节点：{status['summary'].get('obsidian_nodes', 0)}",
        "",
        "## 检查项",
    ]
    for item in checks:
        lines.append(f"- [{'x' if item['passed'] else ' '}] {item['name']}")
    lines.extend(["", "## 检索样例", f"- Query：{query}"])
    for item in obsidian_hits[:3]:
        lines.append(f"- {item.get('domain')} / {item.get('source')}：{item.get('content', '')[:120]}")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(__import__("json").dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
