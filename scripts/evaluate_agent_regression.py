import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.database import init_db
from app.services.evaluation import run_agent_regression_evaluation

OUT_DIR = ROOT / "data" / "real_world"
OUT_JSON = OUT_DIR / "agent_regression_evaluation.json"
OUT_MD = OUT_DIR / "agent_regression_evaluation.md"


def main() -> None:
    init_db()
    result = run_agent_regression_evaluation(True)
    summary = result.get("summary", {})
    cases = result.get("cases", [])
    checks = [
        {"name": "端到端用例不少于4条", "passed": summary.get("case_count", 0) >= 4, "actual": summary.get("case_count", 0)},
        {"name": "Agent回归通过率达到100%", "passed": summary.get("pass_rate", 0) == 100, "actual": summary.get("pass_rate", 0)},
        {"name": "所有用例覆盖候选池选择", "passed": all(item.get("selected_pool") for item in cases), "actual": [item.get("selected_pool") for item in cases]},
        {"name": "所有用例覆盖FeedbackPolicyTool", "passed": all("FeedbackPolicyTool" in item.get("trace_agents", []) for item in cases), "actual": [item.get("trace_agents", []) for item in cases]},
        {"name": "所有用例写入Obsidian长期记忆", "passed": all(item.get("obsidian_note", {}).get("path") for item in cases), "actual": [item.get("obsidian_note", {}).get("path", "") for item in cases]},
        {"name": "回归报告写入Obsidian", "passed": bool(result.get("obsidian_note", {}).get("path")), "actual": result.get("obsidian_note", {}).get("path", "")},
    ]
    passed = sum(1 for item in checks if item["passed"])
    output = {"summary": {**summary, "generated_at_script": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "check_count": len(checks), "passed_checks": passed, "check_pass_rate": round(passed / len(checks) * 100, 1)}, "checks": checks, "agent_regression": result}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 阶段G2：Agent 端到端回归评估",
        "",
        f"生成时间：{output['summary']['generated_at_script']}",
        f"回归通过率：{summary.get('pass_rate', 0)}%",
        f"检查通过率：{output['summary']['check_pass_rate']}%（{passed}/{len(checks)}）",
        f"Obsidian回归报告：{result.get('obsidian_note', {}).get('path', '')}",
        "",
        "## 检查明细",
        "",
    ]
    for item in checks:
        lines.append(f"- {'✅' if item['passed'] else '❌'} {item['name']}：{item['actual']}")
    lines.extend(["", "## 回归用例", ""])
    for item in cases:
        lines.append(f"- {item['name']}：{item['score']} 分，候选池 {item['selected_pool']}，Top：{'、'.join(item['top_models'][:3])}")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
