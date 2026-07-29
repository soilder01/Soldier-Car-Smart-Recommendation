import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.database import init_db
from app.services.release_gate import release_gate

OUT_DIR = ROOT / "data" / "real_world"
OUT_JSON = OUT_DIR / "release_gate_evaluation.json"
OUT_MD = OUT_DIR / "release_gate_evaluation.md"


def main() -> None:
    init_db()
    result = release_gate()
    summary = result.get("summary", {})
    metrics = result.get("metrics", {})
    gate_items = result.get("gate_items", [])
    names = {item.get("name") for item in gate_items}
    validations = [
        {"name": "发布门禁项完整", "passed": len(gate_items) >= 6, "actual": len(gate_items)},
        {"name": "工程化就绪纳入门禁", "passed": "工程化就绪评分" in names, "actual": sorted(names)},
        {"name": "Agent回归纳入门禁", "passed": "Agent端到端回归" in names, "actual": sorted(names)},
        {"name": "真实数据治理纳入门禁", "passed": "真实数据治理评分" in names, "actual": sorted(names)},
        {"name": "发布结论可计算", "passed": summary.get("status") in {"pass", "warn", "blocked"}, "actual": summary.get("status")},
        {"name": "门禁评分合理", "passed": 0 < summary.get("gate_score", 0) <= 100, "actual": summary.get("gate_score", 0)},
        {"name": "真实样本量达标", "passed": metrics.get("real_record_count", 0) >= 200, "actual": metrics.get("real_record_count", 0)},
    ]
    passed = sum(1 for item in validations if item["passed"])
    output = {
        "summary": {
            **summary,
            "generated_at_script": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "validation_count": len(validations),
            "passed_validations": passed,
            "validation_pass_rate": round(passed / len(validations) * 100, 1),
        },
        "validations": validations,
        "release_gate": result,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 阶段G5：Agent运行监控与发布门禁评估",
        "",
        f"生成时间：{output['summary']['generated_at_script']}",
        f"发布结论：{'允许发布' if summary.get('release_allowed') else '暂缓发布'}（状态：{summary.get('status')}）",
        f"门禁评分：{summary.get('gate_score', 0)}%（{summary.get('passed_count', 0)}/{summary.get('gate_count', 0)}）",
        f"验证通过率：{output['summary']['validation_pass_rate']}%（{passed}/{len(validations)}）",
        "",
        "## 核心指标",
        "",
        f"- 工程化就绪：{metrics.get('readiness_score', 0)}%",
        f"- Agent回归通过率：{metrics.get('agent_pass_rate', 0)}%",
        f"- 真实数据治理评分：{metrics.get('governance_score', 0)}",
        f"- 真实数据样本量：{metrics.get('real_record_count', 0)}",
        f"- 反馈样本：{metrics.get('feedback_total', 0)}，正向率：{metrics.get('feedback_positive_rate', 0)}%",
        "",
        "## 验证明细",
        "",
    ]
    for item in validations:
        lines.append(f"- {'✅' if item['passed'] else '❌'} {item['name']}：{item['actual']}")
    lines.extend(["", "## 门禁项", ""])
    for item in gate_items:
        lines.append(f"- {'✅' if item.get('passed') else '⛔'} {item.get('name')}：当前 {item.get('actual')}，阈值 {item.get('threshold')}，动作：{item.get('action')}")
    lines.extend(["", "## 下一步动作", ""])
    for item in result.get("next_actions", []):
        lines.append(f"- {item}")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
