#!/usr/bin/env python3
"""Run a smoke check against a vLLM OpenAI-compatible tool-call endpoint."""

import argparse
import json
import os
from pathlib import Path
from typing import Any


# Smoke-only fallback. Task 5 must replace this evaluation input with the exact
# five-tool schema from data_synth.tool_schemas (or an exported JSON file).
SMOKE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_and_rank_vehicles",
            "description": "Smoke-test vehicle search and ranking.",
            "parameters": {
                "type": "object",
                "properties": {
                    "budget_max": {"type": "integer"},
                    "preferred_type": {"type": "string"},
                    "preferred_energy": {"type": "string"},
                    "concerns": {"type": "string"},
                    "top_k": {"type": "integer"},
                },
            },
        },
    }
]


def load_tools(schema_file: Path | None) -> list[dict[str, Any]]:
    """Load OpenAI tool definitions, or return the one-tool smoke fallback."""
    if schema_file is None:
        return SMOKE_TOOLS

    payload = json.loads(schema_file.read_text(encoding="utf-8"))
    tools = payload.get("tools") if isinstance(payload, dict) else payload
    if not isinstance(tools, list) or not tools:
        raise ValueError("schema file must contain a non-empty tool list")
    if any(not isinstance(tool, dict) or tool.get("type") != "function" for tool in tools):
        raise ValueError("each tool must be an OpenAI function tool object")
    return tools


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-check vLLM tool calling. The default is one smoke tool, "
            "not the final Task 5 five-tool evaluation."
        )
    )
    parser.add_argument(
        "--schema-file",
        type=Path,
        help=(
            "JSON tool schema exported from data_synth.tool_schemas; "
            "defaults to one built-in smoke tool"
        ),
    )
    parser.add_argument(
        "--prompt",
        default="预算25万，三口之家，推荐新能源SUV",
        help="User prompt sent to the chat endpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tools = load_tools(args.schema_file)

    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("CHAT_API_KEY", "dummy"),
        base_url=os.getenv("CHAT_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    response = client.chat.completions.create(
        model=os.getenv("CHAT_MODEL", "qwen7b-nev"),
        messages=[{"role": "user", "content": args.prompt}],
        tools=tools,
        tool_choice="auto",
        temperature=0,
    )
    message = response.choices[0].message
    print(json.dumps(message.model_dump(), ensure_ascii=False, indent=2))
    if not message.tool_calls:
        raise SystemExit("vLLM did not return tool_calls")


if __name__ == "__main__":
    main()
