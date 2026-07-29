import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app.services.recommender import score_vehicle

DATA_CSV = ROOT / "data" / "real_world" / "real_ev_specs_for_recommender.csv"
QUALITY_JSON = ROOT / "data" / "real_world" / "real_data_quality_report.json"
EVAL_JSON = ROOT / "data" / "real_world" / "real_recommendation_evaluation.json"
EVAL_MD = ROOT / "data" / "real_world" / "real_recommendation_evaluation.md"

NUMERIC_FIELDS = {
    "price_min": int,
    "price_max": int,
    "cltc_range": int,
    "battery_kwh": float,
    "fast_charge_minutes": int,
    "seats": int,
    "wheelbase": int,
    "trunk_volume": int,
    "safety_score": float,
    "monthly_sales": int,
}

CASES = [
    {
        "name": "真实样本-长续航纯电通勤",
        "profile": {"budget_max": 350000, "commute_km": 80, "family_size": 2, "has_home_charger": True, "preferred_energy": "纯电", "concerns": ["续航", "补能"]},
        "expect": {"min_top_score": 70, "top5_min_range": 450},
    },
    {
        "name": "真实样本-家庭SUV",
        "profile": {"budget_max": 400000, "commute_km": 50, "family_size": 4, "has_home_charger": True, "preferred_type": "SUV", "concerns": ["空间", "安全"]},
        "expect": {"min_top_score": 70, "top5_type": "SUV"},
    },
    {
        "name": "真实样本-多人家庭MPV",
        "profile": {"budget_max": 450000, "commute_km": 40, "family_size": 5, "has_home_charger": False, "preferred_type": "MPV", "concerns": ["空间", "补能"]},
        "expect": {"min_top_score": 65, "top10_type": "MPV"},
    },
    {
        "name": "真实样本-豪华社交形象",
        "profile": {"budget_max": 450000, "commute_km": 35, "family_size": 2, "has_home_charger": True, "concerns": ["社交形象", "品牌"]},
        "expect": {"min_top_score": 70, "top5_brand_any": ["Audi", "BMW", "Porsche", "Tesla"]},
    },
]


def load_vehicles():
    with DATA_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for field, cast in NUMERIC_FIELDS.items():
            row[field] = cast(float(row[field] or 0))
    return rows


def evaluate_case(case, vehicles):
    scored = [score_vehicle(vehicle, case["profile"]) for vehicle in vehicles]
    scored.sort(key=lambda item: item["score"], reverse=True)
    top10 = scored[:10]
    top5 = scored[:5]
    checks = []
    expect = case["expect"]
    checks.append({"name": "Top1分数达标", "passed": top10[0]["score"] >= expect["min_top_score"], "actual": top10[0]["score"], "expected": expect["min_top_score"]})
    if "top5_min_range" in expect:
        actual = max(item["cltc_range"] for item in top5)
        checks.append({"name": "Top5包含长续航车型", "passed": actual >= expect["top5_min_range"], "actual": actual, "expected": expect["top5_min_range"]})
    if "top5_type" in expect:
        actual = [item["vehicle_type"] for item in top5]
        checks.append({"name": "Top5包含目标车型", "passed": expect["top5_type"] in actual, "actual": actual, "expected": expect["top5_type"]})
    if "top10_type" in expect:
        actual = [item["vehicle_type"] for item in top10]
        checks.append({"name": "Top10包含目标车型", "passed": expect["top10_type"] in actual, "actual": actual, "expected": expect["top10_type"]})
    if "top5_brand_any" in expect:
        actual = [item["brand"] for item in top5]
        checks.append({"name": "Top5包含目标品牌带", "passed": any(brand in actual for brand in expect["top5_brand_any"]), "actual": actual, "expected": expect["top5_brand_any"]})
    return {
        "name": case["name"],
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "top_recommendations": [{"brand": item["brand"], "model": item["model"], "type": item["vehicle_type"], "range": item["cltc_range"], "score": item["score"]} for item in top10[:5]],
    }


def write_markdown(summary):
    lines = [
        "# 真实数据推荐测试评估报告",
        "",
        f"生成时间：{summary['generated_at']}",
        f"真实车型样本：{summary['data']['record_count']} 条，品牌 {summary['data']['unique_brand_count']} 个，车型 {summary['data']['unique_model_count']} 个。",
        f"测试用例：{summary['case_count']} 个，通过 {summary['passed_count']} 个，通过率 {summary['pass_rate']}%。",
        "",
        "## 数据概况",
        "",
        f"- 数据源：{summary['data']['source']}",
        f"- 年份范围：{summary['data']['year_range']}",
        f"- 车型分布：{summary['vehicle_type_distribution']}",
        "",
        "## 用例结果",
    ]
    for case in summary["cases"]:
        status = "通过" if case["passed"] else "未通过"
        lines.extend(["", f"### {case['name']}：{status}", "", "Top 推荐："])
        for item in case["top_recommendations"]:
            lines.append(f"- {item['brand']} {item['model']}｜{item['type']}｜续航 {item['range']}km｜得分 {item['score']}")
        lines.append("")
        lines.append("检查项：")
        for check in case["checks"]:
            lines.append(f"- {'✅' if check['passed'] else '❌'} {check['name']}：实际 {check['actual']}，期望 {check['expected']}")
    EVAL_MD.write_text("\n".join(lines), encoding="utf-8")


def main():
    vehicles = load_vehicles()
    data_report = json.loads(QUALITY_JSON.read_text(encoding="utf-8"))
    cases = [evaluate_case(case, vehicles) for case in CASES]
    passed_count = sum(1 for case in cases if case["passed"])
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "case_count": len(cases),
        "passed_count": passed_count,
        "pass_rate": round(passed_count / len(cases) * 100, 1),
        "data": data_report,
        "vehicle_type_distribution": Counter(vehicle["vehicle_type"] for vehicle in vehicles).most_common(),
        "cases": cases,
    }
    EVAL_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
