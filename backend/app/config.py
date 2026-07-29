from pathlib import Path
import os

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[2]
BASE_DIR = PROJECT_DIR / "backend"
DATA_DIR = PROJECT_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
CONFIG_DIR = BASE_DIR / "config"
DB_PATH = STORAGE_DIR / "nev_advisor.db"
VECTOR_DIR = STORAGE_DIR / "vector_store"
KNOWLEDGE_DIR = DATA_DIR / "knowledge_base"
VEHICLE_CSV = DATA_DIR / "vehicles" / "vehicle_database.csv"

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DIR.mkdir(parents=True, exist_ok=True)


def load_yaml():
    path = CONFIG_DIR / "config.yaml"
    if not path.exists():
        path = CONFIG_DIR / "config.example.yaml"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _select_chat_config(env, llm):
    providers = (
        (
            env.get("CHAT_API_KEY", ""),
            env.get("CHAT_BASE_URL", ""),
            env.get("CHAT_MODEL", ""),
        ),
        (
            env.get("ARK_API_KEY", ""),
            env.get("ARK_BASE_URL", ""),
            env.get("ARK_CHAT_MODEL", ""),
        ),
        (
            llm.get("api_key", ""),
            llm.get("base_url", ""),
            llm.get("chat_model", ""),
        ),
        (
            env.get("OPENAI_API_KEY", ""),
            env.get("OPENAI_BASE_URL", ""),
            env.get("OPENAI_CHAT_MODEL", ""),
        ),
    )
    for provider in providers:
        values = tuple(str(value).strip() if value is not None else "" for value in provider)
        if all(values):
            return values
    return "", "", ""


SETTINGS = load_yaml()
LLM = SETTINGS.get("llm", {})
CHAT_API_KEY, CHAT_BASE_URL, CHAT_MODEL = _select_chat_config(os.environ, LLM)

EMBEDDING_API_KEY = (
    os.getenv("EMBEDDING_API_KEY")
    or os.getenv("ARK_EMBEDDING_API_KEY")
    or LLM.get("embedding_api_key", "")
)
EMBEDDING_BASE_URL = (
    os.getenv("EMBEDDING_BASE_URL")
    or os.getenv("ARK_EMBEDDING_BASE_URL")
    or LLM.get("embedding_base_url", "")
).strip()
EMBEDDING_MODEL = (
    os.getenv("EMBEDDING_MODEL")
    or os.getenv("ARK_EMBEDDING_MODEL")
    or LLM.get("embedding_model", "")
).strip()

# Legacy aliases retained for existing application imports.
OPENAI_API_KEY = CHAT_API_KEY
OPENAI_BASE_URL = CHAT_BASE_URL
TEMPERATURE = float(LLM.get("temperature", 0.2))
TIMEOUT = int(LLM.get("timeout", 60))

CONTENT_GENERATION = SETTINGS.get("content_generation", {})
CONTENT_GENERATION_API_KEY = os.getenv("ARK_CONTENT_API_KEY") or CONTENT_GENERATION.get("api_key", "") or OPENAI_API_KEY
CONTENT_GENERATION_TASK_URL = (os.getenv("ARK_CONTENT_TASK_URL") or CONTENT_GENERATION.get("task_url", "")).strip()
CONTENT_GENERATION_MODEL = (os.getenv("ARK_CONTENT_MODEL") or CONTENT_GENERATION.get("model", "")).strip()
CONTENT_GENERATION_TYPE = CONTENT_GENERATION.get("type", "")

APP = SETTINGS.get("app", {})
WATERMARK = APP.get("watermark", "soldier_yhl")
TAVILY_API_KEY = APP.get("tavily_api_key", "")
