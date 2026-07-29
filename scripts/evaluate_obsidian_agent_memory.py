import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import PROJECT_DIR
from app.schemas import RecommendRequest, UserProfile
from app.services.agent_orchestrator import recommend_with_orchestrator
from app.services.obsidian_vault import VAULT_DIR, graph, split_frontmatter

OUT_JSON = ROOT / "data" / "real_world" / "obsidian_agent_memory_evaluation.json"
OUT_MD = ROOT / "data" / "real_world" / "obsidian_agent_memory_evaluation.md"

CASES = [
    ("家庭SUV", "预算25万以内，三口之家，上海通勤50公里，有家充，关注续航空间智驾", UserProfile(budget_max=250000, family_size=3, commute_km=50, has_home_charger=True, preferred_type="SUV", concerns=["续航", "空间", "智驾"])),
    ("无家充", "没有家充，预算30万，四口之家，通勤和周末长途都要稳定", UserProfile(budget_max=300000, family_size=4, has_home_charger=False, concerns=["补能", "空间"])),
    ("真实数据", "基于真实公开数据和海外车型规格，帮我看 Tesla 和 BMW 电动车", UserProfile(budget_max=500000, concerns=["真实数据", "续航"])),
    ("多人MPV", "二胎家庭六口人，预算40万以内，想要MPV或大六座", UserProfile(budget_max=400000, family_size=6, preferred_type="MPV", concerns=["空间", "安全"])),
    ("预算冲突", "预算15万但想要豪华品牌大空间SUV，最好续航长智驾强", UserProfile(budget_max=150000, preferred_type="SUV", concerns=["空间", "智驾", "品牌"])),
]

REQUIRED_SECTIONS = ["## 候选池决策", "## Top3 可解释对比", "## 风险核验清单", "## 试驾与跟进动作", "## 数据证据来源", "[[Agent 可解释推荐报告]]"]


def run_case(name, query, profile):
    result = recommend_with_orchestrator(RecommendRequest(query=query, profile=profile, top_k=5, candidate_pool_strategy="auto"))
    note = result.get("obsidian_note") or {}
    path = VAULT_DIR / note.get("path", "")
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    meta, _ = split_frontmatter(text)
    missing = [section for section in REQUIRED_SECTIONS if section not in text]
    return {
        "name": name,
        "passed": path.exists() and not missing and meta.get("type") == "recommendation-case" and bool(meta.get("candidate_pool")),
        "path": note.get("path"),
        "candidate_pool": meta.get("candidate_pool"),
        "top_model": meta.get("top_model"),
        "missing_sections": missing,
        "memory_sections": note.get("memory_sections", []),
    }


def main():
    before = graph()["stats"].get("recommendation_case_count", 0)
    rows = [run_case(*case) for case in CASES]
    after_graph = graph()
    passed = sum(1 for row in rows if row["passed"])
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "case_count": len(rows),
        "passed_count": passed,
        "pass_rate": round(passed / len(rows) * 100, 1),
        "recommendation_case_count_before": before,
        "recommendation_case_count_after": after_graph["stats"].get("recommendation_case_count", 0),
        "graph_edge_count": after_graph["stats"].get("edge_count", 0),
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Obsidian Agent 长期记忆评估", "", f"生成时间：{summary['generated_at']}", f"通过率：{summary['pass_rate']}%（{passed}/{len(rows)}）", f"案例数：{before} -> {summary['recommendation_case_count_after']}", f"图谱边数：{summary['graph_edge_count']}", ""]
    for row in rows:
        lines.append(f"## {row['name']}：{'通过' if row['passed'] else '未通过'}")
        lines.append(f"- 路径：{row['path']}")
        lines.append(f"- 候选池：{row['candidate_pool']}，首推：{row['top_model']}")
        if row["missing_sections"]:
            lines.append(f"- 缺失：{'、'.join(row['missing_sections'])}")
        lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
