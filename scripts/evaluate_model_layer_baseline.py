import json
import os
import subprocess
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
OUT_MD = ROOT / "docs" / "model_layer_baseline.md"
PROGRESS = ROOT / "docs" / "model_layer_progress_log.md"
PROTECTED_SIDE_EFFECT_ROOTS = [
    ROOT / "data" / "real_world",
    ROOT / "obsidian-vault",
]


def classify_evaluation_outputs(scripts: Dict[str, Dict[str, Any]]) -> Dict[str, list[str]]:
    result = {"reward_compatible": [], "report_only": [], "engineering_gate": []}
    for name, meta in scripts.items():
        if meta.get("engineering_gate"):
            result["engineering_gate"].append(name)
        elif meta.get("numeric") and meta.get("deterministic"):
            result["reward_compatible"].append(name)
        else:
            result["report_only"].append(name)
    return {key: sorted(value) for key, value in result.items()}


def run_command(cmd: list[str]) -> Dict[str, Any]:
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    return {
        "cmd": " ".join(cmd),
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
        "passed": completed.returncode == 0,
    }


def normalize_release_gate_result(command_result: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(command_result)
    result["command_passed"] = command_result.get("returncode") == 0
    try:
        payload = json.loads(command_result.get("stdout", ""))
        summary = payload.get("summary", payload)
    except (json.JSONDecodeError, TypeError, AttributeError):
        summary = {}
    result.update(
        {
            "business_status": summary.get("status", "unknown"),
            "release_allowed": summary.get("release_allowed", False),
            "passed_count": summary.get("passed_count"),
            "gate_count": summary.get("gate_count"),
        }
    )
    result["business_passed"] = (
        result["business_status"] == "pass" and result["release_allowed"] is True
    )
    return result


def format_check_line(name: str, check: Dict[str, Any]) -> str:
    command_status = "success" if check.get("returncode") == 0 else "failed"
    fields = [f"command_status={command_status}"]
    if name == "release_gate":
        fields.extend(
            [
                f"business_status={check['business_status']}",
                f"release_allowed={str(check['release_allowed']).lower()}",
                f"门禁：{check['passed_count']}/{check['gate_count']}",
            ]
        )
    fields.append(f"命令 `{check['cmd']}`")
    return f"- {name}：" + "，".join(fields)


def snapshot_files(
    roots: list[Path], base: Path = ROOT
) -> Dict[str, Dict[str, Any]]:
    snapshot: Dict[str, Dict[str, Any]] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            content = path.read_bytes()
            stat = path.stat()
            snapshot[path.relative_to(base).as_posix()] = {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256(content).hexdigest(),
            }
    return snapshot


def compare_file_snapshots(
    before: Dict[str, Dict[str, Any]], after: Dict[str, Dict[str, Any]]
) -> Dict[str, list[str]]:
    return {
        "created": sorted(after.keys() - before.keys()),
        "overwritten": sorted(
            path
            for path in before.keys() & after.keys()
            if before[path] != after[path]
        ),
    }


def detect_teacher_config() -> Dict[str, Any]:
    providers = {
        "CHAT": ("CHAT_BASE_URL", "CHAT_MODEL", "CHAT_API_KEY"),
        "ARK": ("ARK_BASE_URL", "ARK_CHAT_MODEL", "ARK_API_KEY"),
        "OPENAI": ("OPENAI_BASE_URL", "OPENAI_CHAT_MODEL", "OPENAI_API_KEY"),
    }
    for prefix, variables in providers.items():
        values = [(os.getenv(variable) or "").strip() for variable in variables]
        if all(values):
            return {
                "configured": True,
                "status": "available_by_env",
                "provider": prefix.lower(),
                "note": "Phase 2 teacher endpoint has base URL, model, and API key configured.",
            }
    return {
        "configured": False,
        "status": "missing",
        "provider": None,
        "note": "Phase 2 requires a complete CHAT, ARK, or legacy OPENAI teacher configuration.",
    }


def build_baseline_report() -> Dict[str, Any]:
    before = snapshot_files(PROTECTED_SIDE_EFFECT_ROOTS)
    checks = {
        "backend_import": run_command(
            [
                "bash",
                "-lc",
                "PYTHONPATH=backend .venv/bin/python - <<'PY'\n"
                "import app.main\n"
                "print('app.main OK')\n"
                "PY",
            ]
        ),
        "health_testclient": run_command(
            [
                "bash",
                "-lc",
                "PYTHONPATH=backend .venv/bin/python - <<'PY'\n"
                "from fastapi.testclient import TestClient\n"
                "from app.main import app\n"
                "with TestClient(app) as client:\n"
                "    response = client.get('/api/health')\n"
                "    print(response.status_code)\n"
                "    print(response.json())\n"
                "    raise SystemExit(0 if response.status_code == 200 else 1)\n"
                "PY",
            ]
        ),
        "agent_regression": run_command(
            [
                "bash",
                "-lc",
                "PYTHONPATH=backend .venv/bin/python scripts/evaluate_agent_regression.py",
            ]
        ),
        "knowledge_fusion": run_command(
            [
                "bash",
                "-lc",
                "PYTHONPATH=backend .venv/bin/python scripts/evaluate_knowledge_fusion.py",
            ]
        ),
        "release_gate": run_command(
            [
                "bash",
                "-lc",
                "PYTHONPATH=backend .venv/bin/python scripts/evaluate_release_gate.py",
            ]
        ),
    }
    checks["release_gate"] = normalize_release_gate_result(checks["release_gate"])
    after = snapshot_files(PROTECTED_SIDE_EFFECT_ROOTS)
    split = classify_evaluation_outputs(
        {
            "evaluate_agent_regression.py": {
                "numeric": True,
                "deterministic": True,
                "engineering_gate": True,
            },
            "evaluate_release_gate.py": {
                "numeric": True,
                "deterministic": True,
                "engineering_gate": True,
            },
            "evaluate_knowledge_fusion.py": {
                "numeric": True,
                "deterministic": True,
                "engineering_gate": True,
            },
        }
    )
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checks": checks,
        "reward_split": split,
        "teacher": detect_teacher_config(),
        "side_effects": compare_file_snapshots(before, after),
        "notes": [
            "Reward-compatible cases are allowed for GRPO reward adapters.",
            "Engineering-gate scripts do not prove model quality.",
            "Phase 6 uses a separate model-output held-out dataset.",
        ],
    }


def write_markdown(report: Dict[str, Any]) -> None:
    lines = [
        "# 模型层改造 Phase 0 基线报告",
        "",
        f"生成时间：{report['generated_at']}",
        "",
        "## 评估脚本划分",
        "",
        f"- reward-compatible：{', '.join(report['reward_split']['reward_compatible']) or '无'}",
        f"- report-only：{', '.join(report['reward_split']['report_only']) or '无'}",
        f"- engineering-gate：{', '.join(report['reward_split']['engineering_gate']) or '无'}",
        "",
        "## 教师模型可用性",
        "",
        f"- 状态：{report['teacher']['status']}",
        f"- 说明：{report['teacher']['note']}",
        "",
        "## 检查结果",
        "",
    ]
    for name, check in report["checks"].items():
        lines.append(format_check_line(name, check))
    lines.extend(["", "## 评估脚本副作用", ""])
    lines.append("### overwritten")
    lines.append("")
    overwritten = report["side_effects"]["overwritten"]
    lines.extend(f"- `{path}`" for path in overwritten)
    if not overwritten:
        lines.append("- 无")
    lines.extend(["", "### created", ""])
    created = report["side_effects"]["created"]
    lines.extend(f"- `{path}`" for path in created)
    if not created:
        lines.append("- 无")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_progress(report: Dict[str, Any]) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as file:
        file.write(f"\n## {report['generated_at']} Phase 0\n\n")
        file.write("- 已生成基线报告。\n")
        file.write("- 当前工作区不是 Git 仓库时，以本文件记录进度。\n")
        file.write(
            "- Git unavailable: root workspace is not a git repository. "
            "Phase 0 artifacts written without commit.\n"
        )
        file.write("- 评估脚本副作用（overwritten）：\n")
        overwritten = report["side_effects"]["overwritten"]
        file.writelines(f"  - `{path}`\n" for path in overwritten)
        if not overwritten:
            file.write("  - 无\n")
        file.write("- 评估脚本副作用（created）：\n")
        created = report["side_effects"]["created"]
        file.writelines(f"  - `{path}`\n" for path in created)
        if not created:
            file.write("  - 无\n")


def main() -> None:
    report = build_baseline_report()
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(report)
    append_progress(report)
    print(
        json.dumps(
            {
                "generated_at": report["generated_at"],
                "reward_split": report["reward_split"],
                "teacher": report["teacher"],
                "release_gate": {
                    "command_status": (
                        "success"
                        if report["checks"]["release_gate"]["command_passed"]
                        else "failed"
                    ),
                    "business_status": report["checks"]["release_gate"][
                        "business_status"
                    ],
                    "release_allowed": report["checks"]["release_gate"][
                        "release_allowed"
                    ],
                    "passed_count": report["checks"]["release_gate"]["passed_count"],
                    "gate_count": report["checks"]["release_gate"]["gate_count"],
                },
                "side_effects": report["side_effects"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
