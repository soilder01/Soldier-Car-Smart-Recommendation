#!/usr/bin/env python3
"""Render shell exports for selecting the application's chat backend."""

import argparse
import re
import shlex
from typing import Dict


CHAT_KEYS = ("CHAT_API_KEY", "CHAT_BASE_URL", "CHAT_MODEL")
ARK_SOURCES = {
    "CHAT_API_KEY": "ARK_API_KEY",
    "CHAT_BASE_URL": "ARK_BASE_URL",
    "CHAT_MODEL": "ARK_CHAT_MODEL",
}
ARK_PROFILE = {
    target: f"${{{source}}}" for target, source in ARK_SOURCES.items()
}
BACKENDS = {
    "ark": ARK_PROFILE,
    "qwen_base_vllm": {
        "CHAT_API_KEY": "dummy",
        "CHAT_BASE_URL": "http://127.0.0.1:8000/v1",
        "CHAT_MODEL": "qwen7b-nev",
    },
    "qwen_sft_vllm": {
        "CHAT_API_KEY": "dummy",
        "CHAT_BASE_URL": "http://127.0.0.1:8001/v1",
        "CHAT_MODEL": "qwen7b-sft",
    },
    "qwen_grpo_vllm": {
        "CHAT_API_KEY": "dummy",
        "CHAT_BASE_URL": "http://127.0.0.1:8002/v1",
        "CHAT_MODEL": "qwen7b-grpo",
    },
}
_SHELL_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def build_backend_env(name: str) -> Dict[str, str]:
    """Build one complete CHAT_* triplet without embedding settings."""
    if name not in BACKENDS:
        raise ValueError(f"unknown backend: {name}")

    return dict(BACKENDS[name] or {})


def render_exports(env: Dict[str, str]) -> str:
    """Render environment values as shell-safe export statements."""
    lines = []
    for key, value in env.items():
        if not _SHELL_NAME.fullmatch(key):
            raise ValueError(f"invalid environment variable name: {key!r}")
        if not isinstance(value, str):
            raise TypeError(f"environment variable {key} must be a string")
        lines.append(f"export {key}={shlex.quote(value)}")
    if env == ARK_PROFILE:
        return render_ark_exports()
    return "\n".join(lines)


def render_ark_exports() -> str:
    """Render fixed Ark forwarding that validates before one atomic export."""
    lines = []
    for index, source in enumerate(ARK_SOURCES.values()):
        keyword = "if" if index == 0 else "elif"
        lines.extend(
            [
                f'{keyword} [[ "${{{source}-}}" =~ ^[[:space:]]*$ ]]; then',
                (
                    "  printf '%s\\n' "
                    f"'switch_llm_backend: {source} must be set and contain "
                    "non-whitespace' "
                    ">&2"
                ),
                "  false",
            ]
        )
    lines.append("else")
    lines.extend(
        [
            '  export CHAT_API_KEY="${ARK_API_KEY}" \\',
            '    CHAT_BASE_URL="${ARK_BASE_URL}" \\',
            '    CHAT_MODEL="${ARK_CHAT_MODEL}"',
        ]
    )
    lines.append("fi")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print shell exports for one chat backend. Evaluate the output "
            "to apply it to the current shell."
        )
    )
    parser.add_argument(
        "backend",
        choices=sorted(BACKENDS),
        help="chat backend profile to render",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(render_exports(build_backend_env(args.backend)))


if __name__ == "__main__":
    main()
