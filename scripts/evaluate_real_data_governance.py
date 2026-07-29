import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.services.real_data_governance import GOVERNANCE_REPORT, generate_real_data_governance

OUT_DIR = ROOT / "data" / "real_world"
OUT_JSON = OUT_DIR / "real_data_governance_evaluation.json"
OUT_MD = OUT_DIR / "real_data_governance_evaluation.md"


def main() -> None:
    result = generate_real_data_governance(True)
    summary = result.get("summary", {})
    checks = [
        {"name": "治理报告已生成", "passed": GOVERNANCE_REPORT.exists(), "actual": str(GOVERNANCE_REPORT)},
        {"name": "记录数不少于200", "passed": summary.get("record_count", 0) >= 200, "actual": summary.get("record_count", 0)},
        {"name": "重复检测存在", "passed": "duplicate_group_count" in summary, "actual": summary.get("duplicate_group_count")},
        {"name": "异常检测存在", "passed": "anomaly_record_count" in summary, "actual": summary.get("anomaly_record_count")},
        {"name": "来源可信度已计算", "passed": summary.get("trusted_source_rate", 0) > 0, "actual": summary.get("trusted_source_rate", 0)},
        {"name": "治理动作已生成", "passed": len(result.get("actions", [])) > 0, "actual": len(result.get("actions", []))},
        {"name": "治理评分合理", "passed": 0 < summary.get("quality_score", 0) <= 100, "actual": summary.get("quality_score", 0)},
    ]
    passed = sum(1 for item in checks if item["passed"])
    output = {"summary": {**summary, "generated_at_script": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "check_count": len(checks), "passed_checks": passed, "check_pass_rate": round(passed / len(checks) * 100, 1)}, "checks": checks, "governance": result}
    OUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 阶段G3：真实数据治理评估",
        "",
        f"生成时间：{output['summary']['generated_at_script']}",
        f"检查通过率：{output['summary']['check_pass_rate']}%（{passed}/{len(checks)}）",
        f"治理评分：{summary.get('quality_score', 0)}",
        f"重复组：{summary.get('duplicate_group_count', 0)}，异常记录：{summary.get('anomaly_record_count', 0)}，可信来源：{summary.get('trusted_source_rate', 0)}%",
        "",
        "## 检查明细",
        "",
    ]
    for item in checks:
        lines.append(f"- {'✅' if item['passed'] else '❌'} {item['name']}：{item['actual']}")
    lines.extend(["", "## 治理动作", ""])
    for item in result.get("actions", []):
        lines.append(f"- **{item['priority']} {item['title']}**：{item['action']}（证据：{item['evidence']}）")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
