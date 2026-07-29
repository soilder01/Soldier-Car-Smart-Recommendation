from datetime import datetime
from typing import Any, Dict, List

from app.database import recommendation_feedback_summary
from app.services.evaluation import run_agent_regression_evaluation
from app.services.real_data_governance import generate_real_data_governance
from app.services.system_readiness import system_readiness


def _gate_item(name: str, passed: bool, actual: Any, threshold: str, level: str, action: str) -> Dict[str, Any]:
    return {"name": name, "passed": passed, "actual": actual, "threshold": threshold, "level": level, "action": action}


def _safe_agent_regression() -> Dict[str, Any]:
    try:
        return run_agent_regression_evaluation(False)
    except Exception as exc:
        return {"summary": {"pass_rate": 0, "case_count": 0, "failed": 1, "error": str(exc)}, "cases": []}


def _safe_governance() -> Dict[str, Any]:
    try:
        return generate_real_data_governance(True)
    except Exception as exc:
        return {"summary": {"quality_score": 0, "record_count": 0, "error": str(exc)}, "actions": []}


def release_gate() -> Dict[str, Any]:
    readiness = system_readiness()
    agent_regression = _safe_agent_regression()
    governance = _safe_governance()
    feedback = recommendation_feedback_summary()
    readiness_summary = readiness.get("summary", {})
    agent_summary = agent_regression.get("summary", {})
    governance_summary = governance.get("summary", {})
    readiness_score = float(readiness_summary.get("readiness_score", 0) or 0)
    agent_pass_rate = float(agent_summary.get("pass_rate", 0) or 0)
    governance_score = float(governance_summary.get("quality_score", 0) or 0)
    record_count = int(governance_summary.get("record_count", 0) or 0)
    feedback_total = int(feedback.get("total", 0) or 0)
    feedback_positive_rate = float(feedback.get("positive_rate", 0) or 0)
    readiness_blockers = [item for item in readiness.get("risks", []) if item.get("level") in {"P0", "P1"}]
    gate_items: List[Dict[str, Any]] = [
        _gate_item("工程化就绪评分", readiness_score >= 95, readiness_score, ">= 95", "blocker", "修复系统健康检查未通过项"),
        _gate_item("Agent端到端回归", agent_pass_rate == 100, agent_pass_rate, "= 100", "blocker", "先修复失败回归用例再发布"),
        _gate_item("真实数据治理评分", governance_score >= 90, governance_score, ">= 90", "blocker", "处理重复、异常和缺失字段后重新治理"),
        _gate_item("真实数据样本量", record_count >= 200, record_count, ">= 200", "blocker", "继续补充真实车型数据"),
        _gate_item("发布阻断风险", len(readiness_blockers) == 0, len(readiness_blockers), "= 0", "blocker", "清理 P0/P1 风险后再发布"),
        _gate_item("人工反馈正向率", feedback_total < 3 or feedback_positive_rate >= 60, feedback_positive_rate if feedback_total else "样本不足", ">= 60% 或样本<3", "warning", "继续收集推荐反馈并复盘负反馈车型"),
    ]
    blockers = [item for item in gate_items if item["level"] == "blocker" and not item["passed"]]
    warnings = [item for item in gate_items if item["level"] == "warning" and not item["passed"]]
    passed = sum(1 for item in gate_items if item["passed"])
    gate_score = round(passed / len(gate_items) * 100, 1)
    release_allowed = len(blockers) == 0
    status = "pass" if release_allowed and not warnings else "warn" if release_allowed else "blocked"
    next_actions = [item["action"] for item in blockers + warnings]
    if not next_actions:
        next_actions = ["当前满足发布门禁，可进入人工验收和版本发布准备"]
    return {
        "summary": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": status,
            "release_allowed": release_allowed,
            "gate_score": gate_score,
            "passed_count": passed,
            "gate_count": len(gate_items),
            "blocker_count": len(blockers),
            "warning_count": len(warnings),
        },
        "metrics": {
            "readiness_score": readiness_score,
            "agent_pass_rate": agent_pass_rate,
            "governance_score": governance_score,
            "real_record_count": record_count,
            "feedback_total": feedback_total,
            "feedback_positive_rate": feedback_positive_rate,
        },
        "gate_items": gate_items,
        "blockers": blockers,
        "warnings": warnings,
        "next_actions": next_actions,
        "sources": {
            "readiness": readiness_summary,
            "agent_regression": agent_summary,
            "governance": governance_summary,
            "feedback": {"total": feedback_total, "positive_rate": feedback_positive_rate},
        },
    }
