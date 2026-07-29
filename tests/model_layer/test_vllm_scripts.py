import json
import os
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
START_SCRIPT = ROOT / "scripts" / "start_vllm_qwen7b.sh"
SFT_START_SCRIPT = ROOT / "scripts" / "start_vllm_sft.sh"
GRPO_START_SCRIPT = ROOT / "scripts" / "start_vllm_grpo.sh"
CONFIG_EXAMPLE = ROOT / "backend" / "config" / "config.vllm.example.yaml"
CHECK_SCRIPT = ROOT / "scripts" / "check_vllm_tool_call.py"


def test_start_vllm_script_contains_required_tool_call_flags():
    script = START_SCRIPT.read_text(encoding="utf-8")

    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "--enable-auto-tool-choice" in script
    assert "--tool-call-parser hermes" in script
    assert "--max-model-len 8192" in script
    assert "--gpu-memory-utilization 0.90" in script
    assert "--served-model-name" in script
    assert "qwen7b-nev" in script
    assert "models/Qwen2.5-7B-Instruct" in script
    assert "exit 1" in script


def test_start_vllm_script_rejects_missing_local_model_without_starting_vllm(tmp_path):
    missing_model = tmp_path / "missing model"
    env = os.environ.copy()
    env["MODEL_PATH"] = str(missing_model)
    env["PATH"] = ""

    result = subprocess.run(
        ["/bin/bash", str(START_SCRIPT)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert str(missing_model) in result.stderr
    assert "不会静默下载模型" in result.stderr


def test_vllm_config_example_matches_current_yaml_contract():
    config = yaml.safe_load(CONFIG_EXAMPLE.read_text(encoding="utf-8"))

    assert config["llm"]["api_key"] == "dummy"
    assert config["llm"]["base_url"] == "http://127.0.0.1:8000/v1"
    assert config["llm"]["chat_model"] == "qwen7b-nev"
    assert config["llm"]["embedding_base_url"] == ""
    assert config["llm"]["embedding_model"] == ""


def test_tool_call_checker_defaults_to_single_smoke_tool():
    from scripts.check_vllm_tool_call import load_tools

    tools = load_tools(None)

    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "search_and_rank_vehicles"


def test_tool_call_checker_accepts_schema_file(tmp_path):
    from scripts.check_vllm_tool_call import load_tools

    expected = [
        {
            "type": "function",
            "function": {
                "name": "task5_tool",
                "description": "Schema supplied by the future Task 5 module.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    schema_file = tmp_path / "tools.json"
    schema_file.write_text(json.dumps({"tools": expected}), encoding="utf-8")

    assert load_tools(schema_file) == expected


def test_tool_call_checker_help_does_not_contact_server():
    result = subprocess.run(
        [str(ROOT / ".venv" / "bin" / "python"), str(CHECK_SCRIPT), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--schema-file" in result.stdout
    assert "smoke" in result.stdout.lower()


def test_adapter_vllm_scripts_contain_required_lora_tool_call_flags():
    expectations = {
        SFT_START_SCRIPT: ("checkpoints/sft", "qwen7b-sft", "8001"),
        GRPO_START_SCRIPT: ("checkpoints/grpo", "qwen7b-grpo", "8002"),
    }

    for path, (adapter_path, served_name, port) in expectations.items():
        script = path.read_text(encoding="utf-8")
        assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
        assert "--enable-lora" in script
        assert "--lora-modules" in script
        assert "--enable-auto-tool-choice" in script
        assert "--tool-call-parser hermes" in script
        assert "--max-model-len 8192" in script
        assert adapter_path in script
        assert served_name in script
        assert port in script
        assert "exit 1" in script
        assert "不会静默下载模型" in script


def test_adapter_vllm_scripts_reject_missing_adapter_without_starting_vllm(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    missing_adapter = tmp_path / "missing-adapter"

    for script in (SFT_START_SCRIPT, GRPO_START_SCRIPT):
        env = os.environ.copy()
        env["MODEL_PATH"] = str(model_dir)
        env["LORA_PATH"] = str(missing_adapter)
        env["PATH"] = ""

        result = subprocess.run(
            ["/bin/bash", str(script)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 1
        assert str(missing_adapter) in result.stderr
        assert "adapter 不存在" in result.stderr
