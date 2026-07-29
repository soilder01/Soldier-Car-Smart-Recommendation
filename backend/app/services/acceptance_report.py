import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from app.config import DATA_DIR, PROJECT_DIR
from app.services.release_gate import release_gate

REPORT_DIR = DATA_DIR / "real_world"
ACCEPTANCE_JSON = REPORT_DIR / "acceptance_report.json"
ACCEPTANCE_MD = REPORT_DIR / "acceptance_report.md"

EVALUATION_FILES = [
    ("F 反馈策略闭环", "feedback_policy_evaluation.json"),
    ("G1 反馈策略稳定性", "feedback_policy_stability_evaluation.json"),
    ("G2 Agent端到端回归", "agent_regression_evaluation.json"),
    ("G3 真实数据治理", "real_data_governance_evaluation.json"),
    ("G4 工程化健康检查", "system_readiness_evaluation.json"),
    ("G5 发布门禁", "release_gate_evaluation.json"),
]


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _pick_rate(summary: Dict[str, Any]) -> float:
    for key in ("validation_pass_rate", "check_pass_rate", "pass_rate"):
        value = summary.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 100.0 if summary and summary.get("failed", 0) == 0 else 0.0


def _stage_rows() -> List[Dict[str, Any]]:
    rows = []
    for name, filename in EVALUATION_FILES:
        data = _load_json(REPORT_DIR / filename)
        summary = data.get("summary", {})
        pass_rate = _pick_rate(summary)
        rows.append({
            "stage": name,
            "file": f"data/real_world/{filename}",
            "exists": bool(data),
            "pass_rate": pass_rate,
            "status": "pass" if data and pass_rate >= 100 else "warn" if data and pass_rate >= 80 else "missing",
            "generated_at": summary.get("generated_at_script") or summary.get("generated_at") or "",
        })
    return rows


def _write_markdown(report: Dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# G6 自动化验收报告与发布前检查",
        "",
        f"生成时间：{summary['generated_at']}",
        f"验收结论：{'通过，可进入人工验收' if summary['accepted'] else '暂缓，需要处理阻断项'}",
        f"综合评分：{summary['acceptance_score']}%",
        f"阶段通过：{summary['passed_stage_count']}/{summary['stage_count']}，发布门禁：{summary['release_gate_status']}",
        "",
        "## 阶段评估汇总",
        "",
    ]
    for row in report["stages"]:
        lines.append(f"- {'✅' if row['status'] == 'pass' else '⚠️'} {row['stage']}：{row['pass_rate']}%，报告 {row['file']}")
    lines.extend(["", "## 发布门禁", ""])
    for item in report["release_gate"].get("gate_items", []):
        lines.append(f"- {'✅' if item.get('passed') else '⛔'} {item.get('name')}：当前 {item.get('actual')}，阈值 {item.get('threshold')}")
    lines.extend(["", "## 下一步动作", ""])
    for item in report.get("next_actions", []):
        lines.append(f"- {item}")
    ACCEPTANCE_MD.write_text("\n".join(lines), encoding="utf-8")


def run_acceptance_report(persist: bool = True) -> Dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stages = _stage_rows()
    gate = release_gate()
    passed_stages = sum(1 for row in stages if row["status"] == "pass")
    stage_score = round(passed_stages / max(len(stages), 1) * 100, 1)
    gate_score = float(gate.get("summary", {}).get("gate_score", 0) or 0)
    acceptance_score = round(stage_score * 0.55 + gate_score * 0.45, 1)
    accepted = passed_stages == len(stages) and gate.get("summary", {}).get("release_allowed", False)
    next_actions = []
    for row in stages:
        if row["status"] != "pass":
            next_actions.append(f"重新运行并修复 {row['stage']} 评估")
    next_actions.extend(gate.get("next_actions", []))
    if accepted and not next_actions:
        next_actions = ["自动化验收通过，可进入人工验收、版本说明和发布准备"]
    report = {
        "summary": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "accepted": accepted,
            "acceptance_score": acceptance_score,
            "stage_score": stage_score,
            "release_gate_score": gate_score,
            "stage_count": len(stages),
            "passed_stage_count": passed_stages,
            "release_gate_status": gate.get("summary", {}).get("status", "unknown"),
            "release_allowed": gate.get("summary", {}).get("release_allowed", False),
            "markdown_report": str(ACCEPTANCE_MD.relative_to(PROJECT_DIR)),
            "json_report": str(ACCEPTANCE_JSON.relative_to(PROJECT_DIR)),
        },
        "stages": stages,
        "release_gate": gate,
        "next_actions": next_actions,
    }
    if persist:
        ACCEPTANCE_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_markdown(report)
    return report
