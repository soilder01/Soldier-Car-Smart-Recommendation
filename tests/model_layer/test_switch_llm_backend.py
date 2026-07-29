import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.switch_llm_backend import build_backend_env, render_exports


ROOT = Path(__file__).resolve().parents[2]
SWITCH_SCRIPT = ROOT / "scripts" / "switch_llm_backend.py"
CHAT_KEYS = {"CHAT_API_KEY", "CHAT_BASE_URL", "CHAT_MODEL"}
ARK_SYMBOLS = {
    "CHAT_API_KEY": "${ARK_API_KEY}",
    "CHAT_BASE_URL": "${ARK_BASE_URL}",
    "CHAT_MODEL": "${ARK_CHAT_MODEL}",
}


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        (
            "qwen_base_vllm",
            {
                "CHAT_API_KEY": "dummy",
                "CHAT_BASE_URL": "http://127.0.0.1:8000/v1",
                "CHAT_MODEL": "qwen7b-nev",
            },
        ),
        (
            "qwen_sft_vllm",
            {
                "CHAT_API_KEY": "dummy",
                "CHAT_BASE_URL": "http://127.0.0.1:8001/v1",
                "CHAT_MODEL": "qwen7b-sft",
            },
        ),
        (
            "qwen_grpo_vllm",
            {
                "CHAT_API_KEY": "dummy",
                "CHAT_BASE_URL": "http://127.0.0.1:8002/v1",
                "CHAT_MODEL": "qwen7b-grpo",
            },
        ),
    ],
)
def test_local_backend_triplets_only_contain_chat_settings(backend, expected):
    env = build_backend_env(backend)

    assert env == expected
    assert set(env) == CHAT_KEYS
    assert not any(key.startswith("EMBEDDING_") for key in env)


def test_ark_backend_returns_stable_symbols_without_reading_environment(monkeypatch):
    secrets = {
        "ARK_API_KEY": "ark-secret-must-not-leak",
        "ARK_BASE_URL": "https://secret.example/v1",
        "ARK_CHAT_MODEL": "secret-model",
    }
    for key, value in secrets.items():
        monkeypatch.setenv(key, value)

    with_secrets = build_backend_env("ark")
    for key in secrets:
        monkeypatch.delenv(key)
    without_secrets = build_backend_env("ark")

    assert with_secrets == ARK_SYMBOLS
    assert without_secrets == ARK_SYMBOLS
    assert set(with_secrets) == CHAT_KEYS
    assert not any(value in repr(with_secrets) for value in secrets.values())


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match=r"unknown backend: unknown"):
        build_backend_env("unknown")


def test_render_exports_is_shell_safe_for_special_characters(tmp_path):
    marker = tmp_path / "must-not-exist"
    expected = {
        "CHAT_API_KEY": f"key with spaces; $(touch {marker}) ' \" $HOME",
        "CHAT_BASE_URL": "https://example.invalid/v1?a=1&b=two words",
        "CHAT_MODEL": "model\nwith-newline",
    }

    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            'eval "$1" && "$2" -c "$3"',
            "bash",
            render_exports(expected),
            sys.executable,
            (
                "import json, os; "
                "print(json.dumps({key: os.environ[key] for key in "
                "('CHAT_API_KEY', 'CHAT_BASE_URL', 'CHAT_MODEL')}))"
            ),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected
    assert not marker.exists()


def test_render_exports_does_not_expand_an_untrusted_ark_like_mapping(tmp_path):
    marker = tmp_path / "must-not-exist"
    expected = dict(ARK_SYMBOLS)
    expected["CHAT_MODEL"] = f"$(touch {marker})"

    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            'eval "$1" && "$2" -c "$3"',
            "bash",
            render_exports(expected),
            sys.executable,
            (
                "import json, os; "
                "print(json.dumps({key: os.environ[key] for key in "
                "('CHAT_API_KEY', 'CHAT_BASE_URL', 'CHAT_MODEL')}))"
            ),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected
    assert not marker.exists()


def _ark_api_output() -> str:
    output = render_exports(build_backend_env("ark"))
    assert "EMBEDDING_" not in output
    return output


def _ark_cli_output() -> str:
    generation_env = os.environ.copy()
    generation_env.update(
        {
            "ARK_API_KEY": "generation-secret-must-not-leak",
            "ARK_BASE_URL": "https://generation.example/v1",
            "ARK_CHAT_MODEL": "generation-model",
        }
    )
    result = subprocess.run(
        [sys.executable, str(SWITCH_SCRIPT), "ark"],
        cwd=ROOT,
        env=generation_env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "generation-secret-must-not-leak" not in result.stdout
    assert "https://generation.example/v1" not in result.stdout
    assert "generation-model" not in result.stdout
    assert "EMBEDDING_" not in result.stdout
    return result.stdout.rstrip("\n")


def test_ark_cli_reuses_the_render_api_path_without_leaking_values():
    assert _ark_cli_output() == _ark_api_output()


def test_ark_render_api_binds_values_at_eval_time_without_executing_them(
    tmp_path,
):
    output = _ark_api_output()
    marker = tmp_path / "must-not-exist"
    eval_env = os.environ.copy()
    expected = {
        "CHAT_API_KEY": "eval secret $HOME; 'quoted'",
        "CHAT_BASE_URL": "https://eval.example/v1?a=1&b=two words",
        "CHAT_MODEL": f"eval-model $(touch {marker})",
    }
    eval_env.update(
        {
            "ARK_API_KEY": expected["CHAT_API_KEY"],
            "ARK_BASE_URL": expected["CHAT_BASE_URL"],
            "ARK_CHAT_MODEL": expected["CHAT_MODEL"],
        }
    )

    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            'eval "$1" && "$2" -c "$3"',
            "bash",
            output,
            sys.executable,
            (
                "import json, os; "
                "print(json.dumps({key: os.environ[key] for key in "
                "('CHAT_API_KEY', 'CHAT_BASE_URL', 'CHAT_MODEL')}))"
            ),
        ],
        cwd=ROOT,
        env=eval_env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)
    assert actual == expected
    assert all(not value.startswith("${ARK_") for value in actual.values())
    assert not marker.exists()


@pytest.mark.parametrize(
    ("invalid_name", "invalid_value"),
    [
        ("ARK_API_KEY", None),
        ("ARK_BASE_URL", None),
        ("ARK_CHAT_MODEL", None),
        ("ARK_API_KEY", ""),
        ("ARK_BASE_URL", " \t\n"),
        ("ARK_CHAT_MODEL", "   "),
    ],
)
@pytest.mark.parametrize("nounset", [False, True])
def test_ark_render_fails_atomically_for_unset_empty_or_blank_values(
    invalid_name,
    invalid_value,
    nounset,
):
    output = _ark_api_output()
    eval_env = os.environ.copy()
    eval_env.update(
        {
            "ARK_API_KEY": "ark-secret",
            "ARK_BASE_URL": "https://ark.example/v1",
            "ARK_CHAT_MODEL": "ark-model",
            "CHAT_API_KEY": "unchanged-key",
            "CHAT_BASE_URL": "unchanged-url",
            "CHAT_MODEL": "unchanged-model",
        }
    )
    if invalid_value is None:
        eval_env.pop(invalid_name, None)
    else:
        eval_env[invalid_name] = invalid_value

    result = subprocess.run(
        [
            "/bin/bash",
            "-c",
            (
                f"{'set -u; ' if nounset else ''}"
                'eval "$1"; status=$?; set +u; '
                'printf \'%s\\n\' "$status" "$CHAT_API_KEY" '
                '"$CHAT_BASE_URL" "$CHAT_MODEL"'
            ),
            "bash",
            output,
        ],
        cwd=ROOT,
        env=eval_env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "1",
        "unchanged-key",
        "unchanged-url",
        "unchanged-model",
    ]
    assert invalid_name in result.stderr
    assert "must be set and contain non-whitespace" in result.stderr
