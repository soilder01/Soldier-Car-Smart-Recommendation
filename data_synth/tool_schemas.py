"""OpenAI function tool schemas for the recommendation agent."""

from copy import deepcopy


TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "extract_user_profile",
            "description": (
                "Extract a structured vehicle purchase profile from a user query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "budget_max": {"type": "integer", "default": 0},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_and_rank_vehicles",
            "description": (
                "Search and rank vehicles from the local vehicle database."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "budget_max": {"type": "integer", "default": 0},
                    "preferred_type": {"type": "string", "default": ""},
                    "preferred_energy": {"type": "string", "default": ""},
                    "concerns": {"type": "string", "default": ""},
                    "top_k": {"type": "integer", "default": 5},
                    "model_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_knowledge_base",
            "description": "Retrieve evidence from the local knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web_info",
            "description": "Search public web information about vehicles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_sales_talk",
            "description": "Generate sales talk for a recommended vehicle.",
            "parameters": {
                "type": "object",
                "properties": {
                    "budget_max": {"type": "integer", "default": 0},
                    "concerns": {"type": "string", "default": ""},
                    "top_model": {"type": "string", "default": ""},
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
]


def build_tool_schemas() -> list[dict]:
    """Return an isolated copy safe for per-request modification."""
    return deepcopy(TOOL_SCHEMAS)
