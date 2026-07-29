import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.database import init_db
from app.services.system_readiness import system_readiness

OUT_DIR = ROOT / "data" / "real_world"
OUT_JSON = OUT_DIR / "system_readiness_evaluation.json"
OUT_MD = OUT_DIR / "system_readiness_evaluation.md"


def main() -> None:
    init_db()
    result = system_readiness()
    summary = result.get("summary", {})
    checks = result.get("checks", [])
    files = result.get("files", [])
    risks = result.get("risks", [])
    names = {item.get("name") for item in checks}
    file_names = {item.get("name") for item in files}
    validation = [
        {"name": "健康检查项已生成", "passed": len(checks) >= 8, "actual": len(checks)},
        {"name": "就绪评分在合理区间", "passed": 0 < summary.get("readiness_score", 0) <= 100, "actual": summary.get("readiness_score", 0)},
        {"name": "车型库检查存在", "passed": "车型库可读取" in names and summary.get("vehicle_count", 0) > 0, "actual": summary.get("vehicle_count", 0)},
        {"name": "RAG索引检查存在", "passed": "RAG索引可用" in names and summary.get("rag_chunks", 0) > 0, "actual": summary.get("rag_chunks", 0)},
        {"name": "真实数据候选库被纳入检查", "passed": "真实数据候选库" in file_names, "actual": sorted(file_names)},
        {"name": "配置样例被纳入检查", "passed": "配置样例" in file_names, "actual": sorted(file_names)},
        {"name": "风险列表结构可读", "passed": isinstance(risks, list), "actual": len(risks)},
    ]
    passed = sum(1 for item in validation if item["passed"])
    output = {
        "summary": {
            **summary,
            "generated_at_script": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "validation_count": len(validation),
            "passed_validations": passed,
            "validation_pass_rate": round(passed / len(validation) * 100, 1),
        },
        "validation": validation,
        "readiness": result,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 阶段G4：工程化健康检查评估",
        "",
        f"生成时间：{output['summary']['generated_at_script']}",
        f"系统就绪评分：{summary.get('readiness_score', 0)}%（{summary.get('passed_count', 0)}/{summary.get('check_count', 0)}）",
        f"验证通过率：{output['summary']['validation_pass_rate']}%（{passed}/{len(validation)}）",
        "",
        "## 验证明细",
        "",
    ]
    for item in validation:
        lines.append(f"- {'✅' if item['passed'] else '❌'} {item['name']}：{item['actual']}")
    lines.extend(["", "## 工程化检查项", ""])
    for item in checks:
        lines.append(f"- {'✅' if item.get('passed') else '⚠️'} {item.get('name')}：{item.get('detail')}")
    lines.extend(["", "## 发布前风险", ""])
    if risks:
        for item in risks:
            lines.append(f"- **{item.get('level')} {item.get('title')}**：{item.get('action')}")
    else:
        lines.append("- 当前未发现阻断发布风险。")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
