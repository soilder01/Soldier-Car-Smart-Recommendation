import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.schemas import RecommendRequest, UserProfile
from app.services.fused_catalog import recommend_fused
from app.services.real_world_recommender import recommend_real_world
from app.services.recommender import recommend

OUT_JSON = ROOT / "data" / "real_world" / "candidate_pool_switch_evaluation.json"
OUT_MD = ROOT / "data" / "real_world" / "candidate_pool_switch_evaluation.md"

REQUEST = RecommendRequest(
    query="预算35万，四口之家，想买新能源SUV，关注续航、空间和安全",
    profile=UserProfile(budget_max=350000, family_size=4, preferred_type="SUV", concerns=["续航", "空间", "安全"]),
    top_k=5,
)


def summarize(name, result):
    recs = result.get("recommendations", [])
    return {
        "pool": name,
        "passed": bool(recs) and len(recs) <= 5,
        "top_models": [{"brand": item["brand"], "model": item["model"], "score": item.get("score"), "source": item.get("catalog_source", item.get("source_type", "local"))} for item in recs],
        "source_count": len({item.get("catalog_source", item.get("source_type", "local")) for item in recs}),
    }


def main():
    rows = [
        summarize("local", recommend(REQUEST.query, REQUEST.profile, REQUEST.top_k)),
        summarize("real", recommend_real_world(REQUEST)),
        summarize("fused", recommend_fused(REQUEST)),
    ]
    passed = sum(1 for row in rows if row["passed"])
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "case_count": len(rows),
        "passed_count": passed,
        "pass_rate": round(passed / len(rows) * 100, 1),
        "rows": rows,
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 候选池切换评估报告", "", f"通过率：{summary['pass_rate']}%", ""]
    for row in rows:
        lines.append(f"## {row['pool']}：{'通过' if row['passed'] else '未通过'}")
        for item in row["top_models"]:
            lines.append(f"- {item['brand']} {item['model']}｜{item['source']}｜{item['score']}分")
        lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
