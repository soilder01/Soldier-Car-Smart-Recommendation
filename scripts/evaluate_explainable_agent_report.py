import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.schemas import RecommendRequest, UserProfile
from app.services.agent_orchestrator import recommend_with_orchestrator

OUT_JSON = ROOT / "data" / "real_world" / "explainable_agent_report_evaluation.json"
OUT_MD = ROOT / "data" / "real_world" / "explainable_agent_report_evaluation.md"

CASES = [
    ("家庭SUV", "预算25万以内，三口之家，上海通勤50公里，有家充，关注续航空间智驾，推荐新能源SUV", UserProfile(budget_max=250000, family_size=3, commute_km=50, has_home_charger=True, preferred_type="SUV", concerns=["续航", "空间", "智驾"])),
    ("无家充", "没有家充，预算30万，四口之家，通勤和周末长途都要稳定", UserProfile(budget_max=300000, family_size=4, has_home_charger=False, concerns=["补能", "空间"])),
    ("真实数据", "基于真实公开数据和海外车型规格，帮我看 Tesla 和 BMW 电动车", UserProfile(budget_max=500000, concerns=["真实数据", "续航"])),
    ("多人MPV", "二胎家庭六口人，预算40万以内，想要MPV或大六座，关注空间安全", UserProfile(budget_max=400000, family_size=6, preferred_type="MPV", concerns=["空间", "安全"])),
    ("预算冲突", "预算15万但想要豪华品牌大空间SUV，最好续航长智驾强", UserProfile(budget_max=150000, preferred_type="SUV", concerns=["空间", "智驾", "品牌"])),
]


def check_case(name, query, profile):
    req = RecommendRequest(query=query, profile=profile, top_k=5, use_deep_search=False, candidate_pool_strategy="auto")
    result = recommend_with_orchestrator(req)
    exp = result.get("explainability") or {}
    comparisons = exp.get("top_comparisons") or []
    passed = bool(result.get("recommendations")) and len(comparisons) >= 3 and all(
        len(item.get("why_selected") or []) >= 3 and len(item.get("cautions") or []) >= 1 and item.get("why_not_others")
        for item in comparisons[:3]
    ) and bool(exp.get("not_recommended")) and bool(exp.get("follow_up_actions")) and bool(result.get("sources"))
    return {
        "name": name,
        "passed": passed,
        "selected_pool": result.get("pool_decision", {}).get("selected_pool"),
        "recommendation_count": len(result.get("recommendations", [])),
        "comparison_count": len(comparisons),
        "risk_count": len(exp.get("risk_checklist") or []),
        "action_count": len(exp.get("follow_up_actions") or []),
        "has_llm_report": any(item.get("agent") == "LLMReportAgent" and "已调用 ARK" in item.get("observation", "") for item in result.get("agent_trace", [])),
        "top_models": [item.get("model") for item in comparisons[:3]],
    }


def main():
    rows = [check_case(*case) for case in CASES]
    passed = sum(1 for row in rows if row["passed"])
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "case_count": len(rows),
        "passed_count": passed,
        "pass_rate": round(passed / len(rows) * 100, 1),
        "rows": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Agent 可解释推荐报告评估", "", f"生成时间：{summary['generated_at']}", f"通过率：{summary['pass_rate']}%（{passed}/{len(rows)}）", ""]
    for row in rows:
        lines.append(f"## {row['name']}：{'通过' if row['passed'] else '未通过'}")
        lines.append(f"- 候选池：{row['selected_pool']}")
        lines.append(f"- Top 对比数：{row['comparison_count']}")
        lines.append(f"- 风险数：{row['risk_count']}，动作数：{row['action_count']}，LLM 报告：{row['has_llm_report']}")
        lines.append(f"- Top 车型：{'、'.join(row['top_models'])}")
        lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
