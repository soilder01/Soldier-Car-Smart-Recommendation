import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.database import init_db
from app.services.acceptance_report import ACCEPTANCE_JSON, ACCEPTANCE_MD, run_acceptance_report

OUT_DIR = ROOT / "data" / "real_world"
OUT_JSON = OUT_DIR / "acceptance_report_evaluation.json"
OUT_MD = OUT_DIR / "acceptance_report_evaluation.md"


def main() -> None:
    init_db()
    report = run_acceptance_report(True)
    summary = report.get("summary", {})
    stages = report.get("stages", [])
    checks = [
        {"name": "验收报告JSON已生成", "passed": ACCEPTANCE_JSON.exists(), "actual": str(ACCEPTANCE_JSON)},
        {"name": "验收报告Markdown已生成", "passed": ACCEPTANCE_MD.exists(), "actual": str(ACCEPTANCE_MD)},
        {"name": "阶段汇总覆盖F-G5", "passed": len(stages) >= 6, "actual": len(stages)},
        {"name": "综合评分合理", "passed": 0 < summary.get("acceptance_score", 0) <= 100, "actual": summary.get("acceptance_score", 0)},
        {"name": "发布门禁已纳入", "passed": "release_gate" in report and bool(report.get("release_gate", {}).get("gate_items")), "actual": len(report.get("release_gate", {}).get("gate_items", []))},
        {"name": "下一步动作可读", "passed": len(report.get("next_actions", [])) > 0, "actual": report.get("next_actions", [])},
        {"name": "阶段报告路径可读", "passed": all(item.get("file") for item in stages), "actual": [item.get("file") for item in stages]},
    ]
    passed = sum(1 for item in checks if item["passed"])
    output = {
        "summary": {
            **summary,
            "generated_at_script": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "check_count": len(checks),
            "passed_checks": passed,
            "check_pass_rate": round(passed / len(checks) * 100, 1),
        },
        "checks": checks,
        "acceptance_report": report,
    }
    OUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 阶段G6：自动化验收报告评估",
        "",
        f"生成时间：{output['summary']['generated_at_script']}",
        f"验收综合评分：{summary.get('acceptance_score', 0)}%",
        f"检查通过率：{output['summary']['check_pass_rate']}%（{passed}/{len(checks)}）",
        f"验收报告：{summary.get('markdown_report', '')}",
        "",
        "## 检查明细",
        "",
    ]
    for item in checks:
        lines.append(f"- {'✅' if item['passed'] else '❌'} {item['name']}：{item['actual']}")
    lines.extend(["", "## 阶段汇总", ""])
    for item in stages:
        lines.append(f"- {item['stage']}：{item['pass_rate']}%，状态 {item['status']}")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
