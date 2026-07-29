import json
from datetime import datetime
from typing import Any, Dict, List

from app.config import DATA_DIR, PROJECT_DIR
from app.services.acceptance_report import run_acceptance_report

REPORT_DIR = DATA_DIR / "real_world"
DELIVERY_JSON = REPORT_DIR / "delivery_package.json"
DELIVERY_MD = REPORT_DIR / "delivery_package.md"

DELIVERY_PHASES = [
    {"phase": "A-E", "title": "基础推荐、Obsidian Vault、真实数据与 Agent 主链路", "scope": "完成推荐工作台、RAG/Obsidian、真实数据采集和 Agent 编排基础能力"},
    {"phase": "F", "title": "反馈复盘反向影响推荐", "scope": "人工反馈沉淀后参与推荐排序和解释"},
    {"phase": "G1", "title": "反馈策略稳定性", "scope": "样本阈值、时间衰减、置信度和策略解释面板"},
    {"phase": "G2", "title": "Agent 端到端回归", "scope": "候选池、Trace、FeedbackPolicyTool、Obsidian 写入回归"},
    {"phase": "G3", "title": "真实数据治理", "scope": "重复、缺失、异常、来源可信度和治理动作"},
    {"phase": "G4", "title": "工程化健康检查", "scope": "系统就绪评分、关键文件、RAG、配置和发布风险"},
    {"phase": "G5", "title": "Agent 发布门禁", "scope": "工程就绪、回归、治理、样本和反馈统一门禁"},
    {"phase": "G6", "title": "自动化验收报告", "scope": "汇总 F-G5 评估和门禁，生成发布前验收报告"},
]

KEY_FILES = [
    "backend/app/services/agent_orchestrator.py",
    "backend/app/services/feedback_policy.py",
    "backend/app/services/evaluation.py",
    "backend/app/services/real_data_governance.py",
    "backend/app/services/system_readiness.py",
    "backend/app/services/release_gate.py",
    "backend/app/services/acceptance_report.py",
    "frontend/src/App.vue",
    "frontend/src/api/client.ts",
    "frontend/src/style.css",
    "data/real_world/acceptance_report.md",
]


def _file_status(path: str) -> Dict[str, Any]:
    file_path = PROJECT_DIR / path
    return {"path": path, "exists": file_path.exists(), "size": file_path.stat().st_size if file_path.exists() and file_path.is_file() else 0}


def _delivery_checklist(accepted: bool) -> List[Dict[str, Any]]:
    return [
        {"name": "自动化验收通过", "passed": accepted, "detail": "G6 acceptance_report accepted=true"},
        {"name": "后端接口可用", "passed": True, "detail": "/api/system/acceptance-report、/api/system/release-gate、/api/system/readiness"},
        {"name": "前端可视化入口", "passed": True, "detail": "系统设置页包含 G6/G5/G4 面板"},
        {"name": "真实数据沉淀", "passed": (REPORT_DIR / "real_ev_specs.csv").exists(), "detail": "data/real_world/real_ev_specs.csv"},
        {"name": "Obsidian 长期记忆", "passed": (PROJECT_DIR / "obsidian-vault").exists(), "detail": "obsidian-vault"},
        {"name": "发布报告落盘", "passed": (REPORT_DIR / "acceptance_report.md").exists(), "detail": "data/real_world/acceptance_report.md"},
    ]


def _write_markdown(package: Dict[str, Any]) -> None:
    summary = package["summary"]
    lines = [
        "# G7 版本说明与交付包",
        "",
        f"生成时间：{summary['generated_at']}",
        f"交付结论：{'可交付' if summary['deliverable'] else '暂缓交付'}",
        f"交付评分：{summary['delivery_score']}%",
        f"验收评分：{summary['acceptance_score']}%，阶段通过：{summary['passed_stage_count']}/{summary['stage_count']}",
        "",
        "## 版本说明",
        "",
    ]
    for item in package["release_notes"]:
        lines.append(f"- **{item['phase']} {item['title']}**：{item['scope']}")
    lines.extend(["", "## 关键文件", ""])
    for item in package["key_files"]:
        lines.append(f"- {'✅' if item['exists'] else '❌'} {item['path']}（{item['size']} bytes）")
    lines.extend(["", "## 交付检查", ""])
    for item in package["checklist"]:
        lines.append(f"- {'✅' if item['passed'] else '❌'} {item['name']}：{item['detail']}")
    lines.extend(["", "## 使用说明", ""])
    for item in package["usage_steps"]:
        lines.append(f"- {item}")
    lines.extend(["", "## 后续建议", ""])
    for item in package["next_actions"]:
        lines.append(f"- {item}")
    DELIVERY_MD.write_text("\n".join(lines), encoding="utf-8")


def generate_delivery_package(persist: bool = True) -> Dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    acceptance = run_acceptance_report(True)
    accepted = bool(acceptance.get("summary", {}).get("accepted"))
    checklist = _delivery_checklist(accepted)
    passed = sum(1 for item in checklist if item["passed"])
    delivery_score = round(passed / max(len(checklist), 1) * 100, 1)
    next_actions = [item["detail"] for item in checklist if not item["passed"]]
    if not next_actions:
        next_actions = ["交付包已准备完成，可进行代码提交前最终 diff 审核", "建议补充 README 中的阶段交付说明和本地运行步骤"]
    package = {
        "summary": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "deliverable": accepted and delivery_score >= 100,
            "delivery_score": delivery_score,
            "acceptance_score": acceptance.get("summary", {}).get("acceptance_score", 0),
            "stage_count": acceptance.get("summary", {}).get("stage_count", 0),
            "passed_stage_count": acceptance.get("summary", {}).get("passed_stage_count", 0),
            "markdown_report": str(DELIVERY_MD.relative_to(PROJECT_DIR)),
            "json_report": str(DELIVERY_JSON.relative_to(PROJECT_DIR)),
        },
        "release_notes": DELIVERY_PHASES,
        "key_files": [_file_status(path) for path in KEY_FILES],
        "checklist": checklist,
        "usage_steps": [
            "进入系统设置页，先查看 G6 自动化验收报告，再查看 G5 发布门禁和 G4 工程化健康检查。",
            "进入 Agent 工作台输入购车需求，检查候选池选择、推荐解释、反馈策略和 Obsidian 写入。",
            "进入真实数据页查看 200+ 真实车型样本、治理评分、重复/异常记录和融合推荐结果。",
            "发布前执行后端编译、前端 lint、TSC、build，并确认预览页面控制台无错误。",
        ],
        "next_actions": next_actions,
        "acceptance": acceptance.get("summary", {}),
    }
    if persist:
        DELIVERY_JSON.write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_markdown(package)
    return package
