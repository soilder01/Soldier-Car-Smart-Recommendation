import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.database import init_db, save_recommendation_feedback
from app.schemas import RecommendRequest, UserProfile
from app.services.agent_orchestrator import recommend_with_orchestrator
from app.services.feedback_policy import apply_feedback_policy, build_feedback_policy

OUT_DIR = ROOT / "data" / "real_world"
OUT_JSON = OUT_DIR / "feedback_policy_evaluation.json"
OUT_MD = OUT_DIR / "feedback_policy_evaluation.md"

QUERY = "特斯拉 Model Y 和小鹏 G6 怎么选，预算30万，主要城市通勤，关注智驾、续航和空间"
PROFILE = UserProfile(
    budget_max=300000,
    commute_km=45,
    has_home_charger=True,
    preferred_type="SUV",
    concerns=["智驾", "续航", "空间"],
)
SEED_FEEDBACK = [
    ("特斯拉 Model Y", "negative", "阶段F评估：历史反馈认为该场景下价格波动和舒适性风险需降权"),
    ("特斯拉 Model Y", "negative", "阶段F评估：历史反馈认为预算边界下不能强推"),
    ("特斯拉 Model Y", "negative", "阶段F评估：历史反馈认为需更谨慎提示"),
    ("小鹏 G6", "positive", "阶段F评估：历史反馈认可智驾和快充解释"),
    ("小鹏 G6", "positive", "阶段F评估：历史反馈认可预算匹配"),
    ("小鹏 G6", "positive", "阶段F评估：历史反馈认可通勤场景适配"),
]


def seed_feedback() -> None:
    for model_name, rating, reason in SEED_FEEDBACK:
        brand, _, model = model_name.partition(" ")
        save_recommendation_feedback({
            "query": QUERY,
            "model_name": model_name,
            "rating": rating,
            "reason": reason,
            "candidate_pool": "fused",
            "scenario_tags": ["阶段F策略评估", "点名车型", "城市通勤"],
            "profile": PROFILE.model_dump(),
            "recommendation": {"brand": brand, "model": model, "score": 86, "candidate_pool": "fused"},
        })


def find_model(rows: list[dict], model_name: str) -> dict:
    for item in rows:
        if f"{item.get('brand')} {item.get('model')}" == model_name:
            return item
    return {}


def main() -> None:
    init_db()
    seed_feedback()
    policy = build_feedback_policy()
    direct_policy_result = apply_feedback_policy([
        {"brand": "特斯拉", "model": "Model Y", "score": 88.0, "reasons": ["原始推荐"], "cautions": []},
        {"brand": "小鹏", "model": "G6", "score": 84.0, "reasons": ["原始推荐"], "cautions": []},
        {"brand": "理想", "model": "L6", "score": 82.0, "reasons": ["原始推荐"], "cautions": []},
    ], "fused")
    direct_recs = direct_policy_result.get("recommendations", [])
    direct_model_y = find_model(direct_recs, "特斯拉 Model Y")
    direct_xpeng_g6 = find_model(direct_recs, "小鹏 G6")

    req = RecommendRequest(query=QUERY, profile=PROFILE, top_k=5, use_deep_search=False, candidate_pool_strategy="fused")
    result = recommend_with_orchestrator(req)
    recs = result.get("recommendations", [])
    trace_agents = [item.get("agent") for item in result.get("agent_trace", [])]
    applied_rules = result.get("feedback_policy", {}).get("applied_rules", [])
    adjusted_count = sum(1 for item in recs if item.get("feedback_score_delta"))

    checks = [
        {"name": "反馈策略生成正负车型规则", "passed": "特斯拉 Model Y" in policy.get("model_rules", {}) and "小鹏 G6" in policy.get("model_rules", {}), "actual": list(policy.get("model_rules", {}).keys())[:10]},
        {"name": "策略函数可对负反馈车型降权", "passed": direct_model_y.get("feedback_score_delta", 0) < 0, "actual": direct_model_y.get("feedback_score_delta")},
        {"name": "策略函数可对正反馈车型加权", "passed": direct_xpeng_g6.get("feedback_score_delta", 0) > 0, "actual": direct_xpeng_g6.get("feedback_score_delta")},
        {"name": "Agent Trace 包含 FeedbackPolicyTool", "passed": "FeedbackPolicyTool" in trace_agents, "actual": trace_agents},
        {"name": "真实 Agent 推荐结果已被反馈策略改分", "passed": adjusted_count > 0, "actual": adjusted_count},
        {"name": "API 返回反馈策略应用规则", "passed": len(applied_rules) > 0, "actual": len(applied_rules)},
        {"name": "解释性报告包含反馈策略", "passed": len(result.get("explainability", {}).get("feedback_policy", {}).get("applied_rules", [])) > 0, "actual": result.get("explainability", {}).get("feedback_policy", {})},
    ]
    passed = sum(1 for item in checks if item["passed"])
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "case_count": len(checks),
        "passed_count": passed,
        "pass_rate": round(passed / len(checks) * 100, 1),
        "applied_rule_count": len(applied_rules),
        "direct_policy_models": [
            {
                "model": f"{item.get('brand')} {item.get('model')}",
                "score": item.get("score"),
                "score_before_feedback": item.get("score_before_feedback"),
                "feedback_score_delta": item.get("feedback_score_delta", 0),
            }
            for item in direct_recs
        ],
        "agent_top_models": [
            {
                "model": f"{item.get('brand')} {item.get('model')}",
                "score": item.get("score"),
                "score_before_feedback": item.get("score_before_feedback"),
                "feedback_score_delta": item.get("feedback_score_delta", 0),
            }
            for item in recs[:5]
        ],
    }
    output = {"summary": summary, "checks": checks, "applied_rules": applied_rules, "direct_policy_rules": direct_policy_result.get("applied_rules", []), "policy": policy}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 阶段F：反馈策略反向影响 Agent 推荐评估",
        "",
        f"生成时间：{summary['generated_at']}",
        f"通过率：{summary['pass_rate']}%（{passed}/{len(checks)}）",
        f"真实 Agent 链路应用规则数：{summary['applied_rule_count']}",
        "",
        "## 检查明细",
        "",
    ]
    for item in checks:
        lines.append(f"- {'✅' if item['passed'] else '❌'} {item['name']}：{item['actual']}")
    lines.extend(["", "## 策略函数分数变化", ""])
    for item in summary["direct_policy_models"]:
        delta = item.get("feedback_score_delta") or 0
        lines.append(f"- {item['model']}：{item['score_before_feedback']} → {item['score']}（{delta:+}）")
    lines.extend(["", "## 真实 Agent 推荐分变化", ""])
    for item in summary["agent_top_models"]:
        delta = item.get("feedback_score_delta") or 0
        lines.append(f"- {item['model']}：{item['score_before_feedback']} → {item['score']}（{delta:+}）")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
