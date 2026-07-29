import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.database import init_db
from app.services.delivery_package import DELIVERY_JSON, DELIVERY_MD, generate_delivery_package

OUT_DIR = ROOT / "data" / "real_world"
OUT_JSON = OUT_DIR / "delivery_package_evaluation.json"
OUT_MD = OUT_DIR / "delivery_package_evaluation.md"


def main() -> None:
    init_db()
    package = generate_delivery_package(True)
    summary = package.get("summary", {})
    checklist = package.get("checklist", [])
    release_notes = package.get("release_notes", [])
    key_files = package.get("key_files", [])
    checks = [
        {"name": "交付包JSON已生成", "passed": DELIVERY_JSON.exists(), "actual": str(DELIVERY_JSON)},
        {"name": "交付包Markdown已生成", "passed": DELIVERY_MD.exists(), "actual": str(DELIVERY_MD)},
        {"name": "版本说明覆盖阶段", "passed": len(release_notes) >= 8, "actual": len(release_notes)},
        {"name": "交付评分满分", "passed": summary.get("delivery_score", 0) == 100, "actual": summary.get("delivery_score", 0)},
        {"name": "验收评分达标", "passed": summary.get("acceptance_score", 0) >= 90, "actual": summary.get("acceptance_score", 0)},
        {"name": "关键文件清单存在", "passed": len(key_files) >= 8 and all(item.get("exists") for item in key_files), "actual": len(key_files)},
        {"name": "交付检查全部通过", "passed": all(item.get("passed") for item in checklist), "actual": [item.get("name") for item in checklist if not item.get("passed")]},
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
        "delivery_package": package,
    }
    OUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 阶段G7：版本说明与交付包评估",
        "",
        f"生成时间：{output['summary']['generated_at_script']}",
        f"交付评分：{summary.get('delivery_score', 0)}%",
        f"检查通过率：{output['summary']['check_pass_rate']}%（{passed}/{len(checks)}）",
        f"交付报告：{summary.get('markdown_report', '')}",
        "",
        "## 检查明细",
        "",
    ]
    for item in checks:
        lines.append(f"- {'✅' if item['passed'] else '❌'} {item['name']}：{item['actual']}")
    lines.extend(["", "## 版本说明", ""])
    for item in release_notes:
        lines.append(f"- {item['phase']} {item['title']}：{item['scope']}")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
