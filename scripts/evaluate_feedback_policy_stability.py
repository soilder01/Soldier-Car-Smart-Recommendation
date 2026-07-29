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
from app.services.feedback_policy import MIN_MODEL_SAMPLES, MIN_POOL_SAMPLES, apply_feedback_policy, build_feedback_policy

OUT_DIR = ROOT / "data" / "real_world"
OUT_JSON = OUT_DIR / "feedback_policy_stability_evaluation.json"
OUT_MD = OUT_DIR / "feedback_policy_stability_evaluation.md"

QUERY = "预算30万，城市通勤，关注智驾、续航和空间，比较特斯拉 Model Y 和小鹏 G6"
PROFILE = UserProfile(budget_max=300000, commute_km=45, has_home_charger=True, preferred_type="SUV", concerns=["智驾", "续航", "空间"])


def seed_stable_feedback() -> None:
    samples = [
        ("特斯拉 Model Y", "negative", "阶段G1评估：预算边界和舒适性反馈较差"),
        ("特斯拉 Model Y", "negative", "阶段G1评估：价格波动影响销售承诺"),
        ("特斯拉 Model Y", "negative", "阶段G1评估：需要更强风险提示"),
        ("小鹏 G6", "positive", "阶段G1评估：智驾和快充解释被认可"),
        ("小鹏 G6", "positive", "阶段G1评估：预算匹配被认可"),
        ("小鹏 G6", "positive", "阶段G1评估：通勤场景适配被认可"),
        ("智界 R7", "negative", "阶段G1评估：融合池候选风险校准"),
        ("腾势 N7", "negative", "阶段G1评估：融合池候选风险校准"),
        ("问界 M5", "positive", "阶段G1评估：融合池正反馈补充"),
    ]
    for model_name, rating, reason in samples:
        brand, _, model = model_name.partition(" ")
        save_recommendation_feedback({
            "query": QUERY,
            "model_name": model_name,
            "rating": rating,
            "reason": reason,
            "candidate_pool": "fused",
            "scenario_tags": ["阶段G1稳定性评估", "城市通勤", "策略置信度"],
            "profile": PROFILE.model_dump(),
            "recommendation": {"brand": brand, "model": model, "score": 86, "candidate_pool": "fused"},
        })


def main() -> None:
    init_db()
    seed_stable_feedback()
    policy = build_feedback_policy()
    direct = apply_feedback_policy([
        {"brand": "特斯拉", "model": "Model Y", "score": 88.0, "reasons": ["原始推荐"], "cautions": []},
        {"brand": "小鹏", "model": "G6", "score": 84.0, "reasons": ["原始推荐"], "cautions": []},
        {"brand": "样本不足", "model": "测试车", "score": 80.0, "reasons": ["原始推荐"], "cautions": []},
    ], "fused")
    result = recommend_with_orchestrator(RecommendRequest(query=QUERY, profile=PROFILE, top_k=5, use_deep_search=False, candidate_pool_strategy="fused"))
    model_rules = policy.get("model_rules", {})
    pool_rules = policy.get("pool_rules", {})
    applied = result.get("feedback_policy", {}).get("applied_rules", [])
    checks = [
        {"name": "策略声明车型样本阈值", "passed": policy.get("stability", {}).get("min_model_samples") == MIN_MODEL_SAMPLES, "actual": policy.get("stability", {})},
        {"name": "策略声明候选池样本阈值", "passed": policy.get("stability", {}).get("min_pool_samples") == MIN_POOL_SAMPLES, "actual": policy.get("stability", {})},
        {"name": "策略启用时间衰减", "passed": policy.get("stability", {}).get("uses_recency_decay") is True, "actual": policy.get("stability", {})},
        {"name": "策略启用置信度", "passed": policy.get("stability", {}).get("uses_confidence") is True, "actual": policy.get("stability", {})},
        {"name": "车型规则包含置信度和原因", "passed": all("confidence" in item and "reason" in item for item in model_rules.values()), "actual": list(model_rules.values())[:3]},
        {"name": "候选池规则满足样本阈值", "passed": all(item.get("total", 0) >= MIN_POOL_SAMPLES for item in pool_rules.values()), "actual": pool_rules},
        {"name": "正负反馈可稳定影响直接策略", "passed": any(item.get("feedback_score_delta", 0) > 0 for item in direct.get("recommendations", [])) and any(item.get("feedback_score_delta", 0) < 0 for item in direct.get("recommendations", [])), "actual": direct.get("recommendations", [])},
        {"name": "真实 Agent 链路返回可解释策略", "passed": bool(applied) and all("confidence" in item and "reason" in item for item in applied), "actual": applied[:5]},
    ]
    passed = sum(1 for item in checks if item["passed"])
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "case_count": len(checks),
        "passed_count": passed,
        "pass_rate": round(passed / len(checks) * 100, 1),
        "model_rule_count": len(model_rules),
        "pool_rule_count": len(pool_rules),
        "agent_applied_rule_count": len(applied),
    }
    output = {"summary": summary, "checks": checks, "policy_stability": policy.get("stability", {}), "model_rules": model_rules, "pool_rules": pool_rules, "agent_applied_rules": applied}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 阶段G1：反馈策略稳定性与可解释性评估",
        "",
        f"生成时间：{summary['generated_at']}",
        f"通过率：{summary['pass_rate']}%（{passed}/{len(checks)}）",
        f"车型规则数：{summary['model_rule_count']}，候选池规则数：{summary['pool_rule_count']}，Agent 应用规则数：{summary['agent_applied_rule_count']}",
        "",
        "## 检查明细",
        "",
    ]
    for item in checks:
        lines.append(f"- {'✅' if item['passed'] else '❌'} {item['name']}：{item['actual']}")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
