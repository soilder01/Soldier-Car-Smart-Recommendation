import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.database import init_db, recommendation_feedback_summary, save_recommendation_feedback
from app.services.feedback_notes import save_feedback_note
from app.services.optimization import generate_feedback_review

OUT_DIR = ROOT / "data" / "real_world"
OUT_JSON = OUT_DIR / "feedback_review_evaluation.json"
OUT_MD = OUT_DIR / "feedback_review_evaluation.md"

SAMPLES = [
    ("三口之家上海通勤，有家充，预算25万，关注空间和智驾", "比亚迪 宋PLUS DM-i", "positive", "local", ["家庭SUV", "智驾关注", "预算敏感"], "空间和补能匹配"),
    ("三口之家上海通勤，有家充，预算25万，关注空间和智驾", "理想 L6", "positive", "local", ["家庭SUV", "智驾关注"], "家庭舒适性解释充分"),
    ("无家充但想买纯电 SUV，预算22万", "小鹏 G6", "negative", "local", ["无家充补能", "预算敏感"], "无家充纯电风险提示不足"),
    ("希望看海外公开数据，优先长续航 SUV", "BMW iX", "neutral", "real", ["真实数据查询"], "真实数据可参考但价格地域需核验"),
    ("真实数据里找 30 万内高续航车型", "Hyundai IONIQ 5", "negative", "real", ["真实数据查询", "预算敏感"], "海外数据进入国内预算场景解释不足"),
    ("预算 18 万以内，需要通勤省心", "比亚迪 宋PLUS DM-i", "positive", "fused", ["预算敏感", "通用推荐"], "融合池给出了低预算稳定备选"),
    ("多人家庭六座七座，周末长途", "问界 M7", "positive", "local", ["家庭SUV"], "座位和长途补能匹配"),
    ("客户想要豪华品牌和社交形象", "奔驰 E300L", "positive", "local", ["豪华社交"], "保留了豪华品牌偏好"),
    ("预算 20 万但点名 Model Y", "特斯拉 Model Y", "negative", "fused", ["预算敏感", "预算冲突"], "超预算风险需要更靠前提示"),
    ("没有家充还经常长途，纠结纯电和增程", "理想 L6", "positive", "fused", ["无家充补能", "家庭SUV"], "增程备选解释清楚"),
]


def make_profile(query: str, tags: list[str]) -> dict:
    return {
        "budget_max": 250000 if "25" in query else 220000 if "22" in query else 200000 if "20" in query else 180000 if "18" in query else None,
        "city": "上海" if "上海" in query else "",
        "family_size": 3 if "三口" in query else None,
        "has_home_charger": False if "无家充" in query or "没有家充" in query else True if "有家充" in query else None,
        "concerns": tags,
    }


def main() -> None:
    init_db()
    created = []
    for query, model, rating, pool, tags, reason in SAMPLES:
        brand, _, model_part = model.partition(" ")
        feedback = save_recommendation_feedback({
            "query": query,
            "model_name": model,
            "rating": rating,
            "reason": f"阶段D模拟反馈：{reason}",
            "candidate_pool": pool,
            "scenario_tags": tags,
            "profile": make_profile(query, tags),
            "recommendation": {"brand": brand, "model": model_part or model, "score": 88, "candidate_pool": pool},
        })
        created.append(save_feedback_note(feedback))
    summary = recommendation_feedback_summary()
    review = generate_feedback_review(True)
    checks = [
        {"name": "构造反馈样本不少于10条", "passed": len(SAMPLES) >= 10, "actual": len(SAMPLES)},
        {"name": "候选池维度聚合存在", "passed": len(summary.get("pool_rows", [])) >= 3, "actual": len(summary.get("pool_rows", []))},
        {"name": "场景风险维度聚合存在", "passed": len(summary.get("scene_rows", [])) >= 3, "actual": len(summary.get("scene_rows", []))},
        {"name": "Agent复盘结论存在", "passed": len(review.get("insights", [])) >= 3, "actual": len(review.get("insights", []))},
        {"name": "Obsidian复盘记录已生成", "passed": bool(review.get("obsidian_note", {}).get("path")), "actual": review.get("obsidian_note", {}).get("path", "")},
    ]
    passed = sum(1 for item in checks if item["passed"])
    result = {
        "summary": {
            "case_count": len(checks),
            "passed_count": passed,
            "pass_rate": round(passed / len(checks) * 100, 1),
            "created_feedback_notes": len(created),
            "feedback_total_after": summary.get("total", 0),
        },
        "checks": checks,
        "feedback_summary": summary,
        "feedback_review": review,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 阶段D：反馈驱动 Agent 自我复盘评估",
        "",
        f"- 检查项：{len(checks)}",
        f"- 通过项：{passed}",
        f"- 通过率：{result['summary']['pass_rate']}%",
        f"- 累计反馈样本：{summary.get('total', 0)}",
        f"- Obsidian 复盘记录：{review.get('obsidian_note', {}).get('path', '')}",
        "",
        "## 检查明细",
        "",
    ]
    for item in checks:
        lines.append(f"- {'✅' if item['passed'] else '❌'} {item['name']}：{item['actual']}")
    lines.extend(["", "## Agent 复盘结论", ""])
    for item in review.get("insights", []):
        lines.append(f"- **{item['title']}**：{item['action']}（证据：{item['evidence']}）")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
