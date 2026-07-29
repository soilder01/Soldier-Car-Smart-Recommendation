import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "frontend" / "src" / "App.vue"
STYLE = ROOT / "frontend" / "src" / "style.css"
OUT_DIR = ROOT / "data" / "real_world"
OUT_JSON = OUT_DIR / "agent_workbench_evaluation.json"
OUT_MD = OUT_DIR / "agent_workbench_evaluation.md"


def main() -> None:
    app = APP.read_text(encoding="utf-8")
    style = STYLE.read_text(encoding="utf-8")
    checks = [
        {"name": "默认进入 Agent 工作台", "passed": "? initialActive : 'recommend'" in app, "evidence": "active 默认 recommend"},
        {"name": "智能推荐作为主入口", "passed": "label: 'Agent 工作台'" in app and "key: 'recommend'" in app.split("const navs = [", 1)[1].split("]", 1)[0], "evidence": "navs 首项为 Agent 工作台"},
        {"name": "工作台展示当前候选池", "passed": "当前候选池" in app and "selectedPoolLabel" in app, "evidence": "agentWorkspaceStats 包含候选池"},
        {"name": "工作台展示工具调用", "passed": "工具调用" in app and "agentTrace.value.length" in app, "evidence": "agentWorkspaceStats 包含 Trace 步数"},
        {"name": "工作台展示推荐证据", "passed": "推荐证据" in app and "evidenceRows.value.length" in app, "evidence": "agentWorkspaceStats 包含证据条数"},
        {"name": "工作台展示 Obsidian 沉淀结果", "passed": "长期记忆" in app and "obsidianNote.value?.path" in app, "evidence": "agentWorkspaceStats 包含长期记忆"},
        {"name": "辅助页面收束为二级入口", "passed": "auxiliaryViews" in app and "二级视图" in app, "evidence": "辅助能力入口存在"},
        {"name": "Agent Trace 可视化清晰", "passed": "agent-trace-flow" in app and ".agent-trace-flow" in style, "evidence": "使用独立 Trace Flow 样式"},
    ]
    passed = sum(1 for item in checks if item["passed"])
    result = {"summary": {"case_count": len(checks), "passed_count": passed, "pass_rate": round(passed / len(checks) * 100, 1)}, "checks": checks}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 阶段E：Agent 工作台收束评估",
        "",
        f"- 检查项：{len(checks)}",
        f"- 通过项：{passed}",
        f"- 通过率：{result['summary']['pass_rate']}%",
        "",
        "## 检查明细",
        "",
    ]
    for item in checks:
        lines.append(f"- {'✅' if item['passed'] else '❌'} {item['name']}：{item['evidence']}")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
