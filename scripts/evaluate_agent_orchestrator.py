import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.schemas import RecommendRequest, UserProfile
from app.services.agent_orchestrator import recommend_with_orchestrator

OUT_JSON = ROOT / "data" / "real_world" / "agent_orchestrator_evaluation.json"
OUT_MD = ROOT / "data" / "real_world" / "agent_orchestrator_evaluation.md"

CASES = [
    {
        "name": "家庭SUV",
        "query": "预算25万以内，三口之家，上海通勤每天50公里，有家充，关注续航、空间和智驾，推荐新能源SUV",
        "profile": UserProfile(budget_max=250000, family_size=3, commute_km=50, has_home_charger=True, preferred_type="SUV", concerns=["续航", "空间", "智驾"]),
        "expected_pool": "local",
    },
    {
        "name": "无家充",
        "query": "没有家充，预算30万，四口之家，日常通勤和周末长途怎么选新能源车",
        "profile": UserProfile(budget_max=300000, family_size=4, has_home_charger=False, concerns=["补能", "空间"]),
        "expected_pool": "fused",
    },
    {
        "name": "豪华社交",
        "query": "预算60万，想要有社交形象和品牌感的新能源或豪华车，适合约会和商务",
        "profile": UserProfile(budget_max=600000, concerns=["社交形象", "品牌", "内饰氛围"]),
        "expected_pool": "fused",
    },
    {
        "name": "点名车型",
        "query": "特斯拉 Model Y 和小鹏 G6 怎么选，预算30万，主要城市通勤",
        "profile": UserProfile(budget_max=300000, commute_km=45, concerns=["智驾", "续航"]),
        "expected_pool": "fused",
    },
    {
        "name": "真实数据查询",
        "query": "基于真实公开数据和欧洲海外车型规格，帮我看 Tesla、BMW、Audi 电动车怎么选",
        "profile": UserProfile(budget_max=500000, concerns=["真实数据", "续航"]),
        "expected_pool": "real",
    },
    {
        "name": "预算冲突",
        "query": "预算15万但想要豪华品牌大空间SUV，最好续航长智驾强",
        "profile": UserProfile(budget_max=150000, preferred_type="SUV", concerns=["空间", "智驾", "品牌"]),
        "expected_pool": "fused",
    },
    {
        "name": "长续航通勤",
        "query": "每天通勤100公里，有固定车位家充，预算35万，想要纯电长续航",
        "profile": UserProfile(budget_max=350000, commute_km=100, has_home_charger=True, preferred_energy="纯电", concerns=["续航", "补能"]),
        "expected_pool": "local",
    },
    {
        "name": "多人MPV",
        "query": "二胎家庭六口人，预算40万以内，想要MPV或大六座，关注空间和安全",
        "profile": UserProfile(budget_max=400000, family_size=6, preferred_type="MPV", concerns=["空间", "安全"]),
        "expected_pool": "local",
    },
]


def summarize(case):
    req = RecommendRequest(query=case["query"], profile=case["profile"], top_k=5, use_deep_search=False, candidate_pool_strategy="auto")
    result = recommend_with_orchestrator(req)
    recs = result.get("recommendations", [])
    pool = result.get("pool_decision", {}).get("selected_pool")
    trace_agents = [item.get("agent") for item in result.get("agent_trace", [])]
    obsidian_path = result.get("obsidian_note", {}).get("path", "")
    passed = bool(recs) and pool == case["expected_pool"] and "CandidatePoolSelectorTool" in trace_agents and bool(obsidian_path)
    return {
        "name": case["name"],
        "passed": passed,
        "expected_pool": case["expected_pool"],
        "selected_pool": pool,
        "recommendation_count": len(recs),
        "top_models": [{"brand": item.get("brand"), "model": item.get("model"), "score": item.get("score"), "source": item.get("catalog_source") or item.get("source_type", "local")} for item in recs[:3]],
        "trace_agents": trace_agents,
        "risk_count": len(result.get("risks", [])),
        "obsidian_path": obsidian_path,
    }


def main():
    rows = [summarize(case) for case in CASES]
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
    lines = ["# Agent Orchestrator 真实后端评估报告", "", f"生成时间：{summary['generated_at']}", f"通过率：{summary['pass_rate']}%（{passed}/{len(rows)}）", ""]
    for row in rows:
        lines.append(f"## {row['name']}：{'通过' if row['passed'] else '未通过'}")
        lines.append(f"- 期望候选池：{row['expected_pool']}")
        lines.append(f"- 实际候选池：{row['selected_pool']}")
        lines.append(f"- Obsidian 沉淀：{row['obsidian_path']}")
        for item in row["top_models"]:
            lines.append(f"- {item['brand']} {item['model']}｜{item['source']}｜{item['score']}分")
        lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
