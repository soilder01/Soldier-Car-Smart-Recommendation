import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.schemas import RecommendRequest, UserProfile
from app.services.fused_catalog import fused_catalog, recommend_fused

OUT_JSON = ROOT / "data" / "real_world" / "fused_recommendation_evaluation.json"
OUT_MD = ROOT / "data" / "real_world" / "fused_recommendation_evaluation.md"

CASES = [
    {
        "name": "融合池-中文家庭SUV",
        "request": RecommendRequest(query="预算35万，四口之家，要SUV，关注空间和续航", profile=UserProfile(budget_max=350000, family_size=4, preferred_type="SUV", concerns=["空间", "续航"]), top_k=8),
        "expect": {"min_total": 200, "top8_has_local": True, "top8_has_real": True},
    },
    {
        "name": "融合池-豪华社交",
        "request": RecommendRequest(query="预算60万，关注品牌和社交形象，想要豪华新能源", profile=UserProfile(budget_max=600000, concerns=["社交形象", "品牌"]), top_k=8),
        "expect": {"top8_brand_any": ["宝马", "奔驰", "奥迪", "保时捷", "特斯拉"]},
    },
    {
        "name": "融合池-有家充长续航",
        "request": RecommendRequest(query="预算30万，有家充，通勤远，关注续航补能", profile=UserProfile(budget_max=300000, has_home_charger=True, commute_km=90, preferred_energy="纯电", concerns=["续航", "补能"]), top_k=8),
        "expect": {"top8_min_range": 550},
    },
]


def evaluate_case(case, catalog_summary):
    result = recommend_fused(case["request"])
    top = result["recommendations"]
    checks = []
    expect = case["expect"]
    if "min_total" in expect:
        checks.append({"name": "融合候选池规模", "passed": catalog_summary["total"] >= expect["min_total"], "actual": catalog_summary["total"], "expected": expect["min_total"]})
    if expect.get("top8_has_local"):
        actual = [item.get("catalog_source") for item in top]
        checks.append({"name": "Top8包含本地精选", "passed": "local_curated" in actual, "actual": actual, "expected": "local_curated"})
    if expect.get("top8_has_real"):
        actual = [item.get("catalog_source") for item in top]
        checks.append({"name": "Top8包含真实扩展", "passed": "real_world_enriched" in actual, "actual": actual, "expected": "real_world_enriched"})
    if "top8_brand_any" in expect:
        actual = [item["brand"] for item in top]
        checks.append({"name": "Top8包含目标品牌", "passed": any(brand in actual for brand in expect["top8_brand_any"]), "actual": actual, "expected": expect["top8_brand_any"]})
    if "top8_min_range" in expect:
        actual = max(item["cltc_range"] for item in top)
        checks.append({"name": "Top8包含长续航", "passed": actual >= expect["top8_min_range"], "actual": actual, "expected": expect["top8_min_range"]})
    return {
        "name": case["name"],
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "top_recommendations": [{"brand": item["brand"], "model": item["model"], "source": item.get("catalog_source"), "range": item["cltc_range"], "score": item["score"]} for item in top[:5]],
    }


def write_md(summary):
    lines = ["# 融合候选池推荐评估报告", "", f"生成时间：{summary['generated_at']}", f"融合候选池：{summary['catalog_summary']}", f"通过率：{summary['pass_rate']}%", ""]
    for case in summary["cases"]:
        lines.extend([f"## {case['name']}：{'通过' if case['passed'] else '未通过'}", ""])
        for item in case["top_recommendations"]:
            lines.append(f"- {item['brand']} {item['model']}｜{item['source']}｜续航 {item['range']}km｜{item['score']}分")
        for check in case["checks"]:
            lines.append(f"- {'✅' if check['passed'] else '❌'} {check['name']}：实际 {check['actual']}，期望 {check['expected']}")
        lines.append("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main():
    catalog_summary = fused_catalog()["summary"]
    cases = [evaluate_case(case, catalog_summary) for case in CASES]
    passed = sum(1 for case in cases if case["passed"])
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "catalog_summary": catalog_summary,
        "case_count": len(cases),
        "passed_count": passed,
        "pass_rate": round(passed / len(cases) * 100, 1),
        "cases": cases,
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
