import json

import pytest

from scripts import evaluate_model_layer_baseline as baseline


def test_classify_evaluation_outputs_keeps_engineering_gates_separate():
    scripts = {
        "evaluate_agent_regression.py": {"numeric": True, "deterministic": True, "engineering_gate": True},
        "evaluate_knowledge_fusion.py": {"numeric": True, "deterministic": True, "engineering_gate": True},
        "evaluate_release_gate.py": {"numeric": True, "deterministic": True, "engineering_gate": True},
    }

    result = baseline.classify_evaluation_outputs(scripts)

    assert "reward_compatible" in result
    assert "report_only" in result
    assert "engineering_gate" in result
    assert "evaluate_agent_regression.py" in result["engineering_gate"]
    assert "evaluate_knowledge_fusion.py" in result["engineering_gate"]
    assert "evaluate_knowledge_fusion.py" not in result["reward_compatible"]


@pytest.mark.parametrize(
    ("prefix", "model_variable"),
    [
        ("CHAT", "CHAT_MODEL"),
        ("ARK", "ARK_CHAT_MODEL"),
        ("OPENAI", "OPENAI_CHAT_MODEL"),
    ],
)
def test_detect_teacher_config_requires_complete_provider_config(
    monkeypatch, prefix, model_variable
):
    for name in ("CHAT", "ARK", "OPENAI"):
        for field in ("BASE_URL", "MODEL", "CHAT_MODEL", "API_KEY"):
            monkeypatch.delenv(f"{name}_{field}", raising=False)

    monkeypatch.setenv(f"{prefix}_BASE_URL", "https://teacher.example/v1")
    monkeypatch.setenv(model_variable, "teacher-model")
    monkeypatch.setenv(f"{prefix}_API_KEY", "secret")

    result = baseline.detect_teacher_config()

    assert result["configured"] is True
    assert result["status"] == "available_by_env"
    assert result["provider"] == prefix.lower()


def test_detect_teacher_config_rejects_incomplete_or_mixed_config(monkeypatch):
    for name in ("CHAT", "ARK", "OPENAI"):
        for field in ("BASE_URL", "MODEL", "CHAT_MODEL", "API_KEY"):
            monkeypatch.delenv(f"{name}_{field}", raising=False)
    monkeypatch.setenv("CHAT_BASE_URL", "https://teacher.example/v1")
    monkeypatch.setenv("ARK_CHAT_MODEL", "teacher-model")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    result = baseline.detect_teacher_config()

    assert result["configured"] is False
    assert result["status"] == "missing"
    assert result["provider"] is None


def test_detect_teacher_config_rejects_whitespace_values(monkeypatch):
    for name in ("CHAT", "ARK", "OPENAI"):
        for field in ("BASE_URL", "MODEL", "CHAT_MODEL", "API_KEY"):
            monkeypatch.delenv(f"{name}_{field}", raising=False)
    monkeypatch.setenv("CHAT_BASE_URL", "   ")
    monkeypatch.setenv("CHAT_MODEL", "teacher-model")
    monkeypatch.setenv("CHAT_API_KEY", "secret")

    result = baseline.detect_teacher_config()

    assert result["configured"] is False
    assert result["status"] == "missing"


def test_release_gate_exit_zero_keeps_business_blocked_visible():
    command_result = {
        "cmd": "python scripts/evaluate_release_gate.py",
        "returncode": 0,
        "stdout": json.dumps(
            {
                "status": "blocked",
                "release_allowed": False,
                "passed_count": 4,
                "gate_count": 6,
            }
        ),
        "stderr": "",
        "passed": True,
    }

    result = baseline.normalize_release_gate_result(command_result)
    line = baseline.format_check_line("release_gate", result)

    assert result["command_passed"] is True
    assert result["business_status"] == "blocked"
    assert result["release_allowed"] is False
    assert result["passed_count"] == 4
    assert result["gate_count"] == 6
    assert result["business_passed"] is False
    assert "command_status=success" in line
    assert "business_status=blocked" in line
    assert "release_allowed=false" in line
    assert "门禁：4/6" in line
    assert "business_status=pass" not in line


def test_file_snapshot_detects_created_and_overwritten(tmp_path):
    protected = tmp_path / "data" / "real_world"
    protected.mkdir(parents=True)
    existing = protected / "existing.json"
    existing.write_text('{"version": 1}', encoding="utf-8")

    before = baseline.snapshot_files([protected], base=tmp_path)
    existing.write_text('{"version": 2}', encoding="utf-8")
    created = protected / "created.json"
    created.write_text("{}", encoding="utf-8")
    after = baseline.snapshot_files([protected], base=tmp_path)

    result = baseline.compare_file_snapshots(before, after)

    assert result["created"] == ["data/real_world/created.json"]
    assert result["overwritten"] == ["data/real_world/existing.json"]
