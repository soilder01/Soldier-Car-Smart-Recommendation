import httpx
from openai import OpenAI

from app.config import CHAT_API_KEY, CHAT_BASE_URL, TIMEOUT


def chat_client() -> OpenAI:
    return OpenAI(
        api_key=CHAT_API_KEY,
        base_url=CHAT_BASE_URL,
        http_client=httpx.Client(trust_env=False, timeout=TIMEOUT),
    )


def openai_client() -> OpenAI:
    return chat_client()


def mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 4:
        return "***"
    return f"***{value[-4:]}"


def check_chat_model(model: str, timeout: int = 20) -> dict:
    if not CHAT_API_KEY:
        return {"ok": False, "reason": "missing_api_key"}
    if not model or not CHAT_BASE_URL:
        return {"ok": False, "reason": "missing_model_or_base_url"}
    try:
        client = OpenAI(
            api_key=CHAT_API_KEY,
            base_url=CHAT_BASE_URL,
            http_client=httpx.Client(trust_env=False, timeout=timeout),
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "只回答：可用"}],
            temperature=0,
            timeout=timeout,
        )
        return {"ok": True, "model_configured": True, "sample": resp.choices[0].message.content[:20]}
    except Exception as exc:
        return {"ok": False, "model_configured": True, "reason": type(exc).__name__}
