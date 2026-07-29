import importlib

import pytest


CONFIG_ENV_KEYS = [
    "CHAT_API_KEY",
    "CHAT_BASE_URL",
    "CHAT_MODEL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_MODEL",
    "ARK_API_KEY",
    "ARK_BASE_URL",
    "ARK_CHAT_MODEL",
    "ARK_EMBEDDING_API_KEY",
    "ARK_EMBEDDING_BASE_URL",
    "ARK_EMBEDDING_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_CHAT_MODEL",
]


@pytest.fixture
def reload_config(monkeypatch):
    import app.config as config

    def reload_with(*, yaml_llm=None, **env):
        with monkeypatch.context() as scoped:
            for key in CONFIG_ENV_KEYS:
                scoped.delenv(key, raising=False)
            for key, value in env.items():
                scoped.setenv(key, value)
            if yaml_llm is not None:
                scoped.setattr(
                    config.yaml,
                    "safe_load",
                    lambda _stream: {"llm": yaml_llm},
                )
            return importlib.reload(config)

    yield reload_with
    importlib.reload(config)


def test_chat_config_prefers_chat_env_and_keeps_legacy_aliases(reload_config):
    config = reload_config(
        CHAT_API_KEY="chat-key",
        CHAT_BASE_URL="http://chat/v1",
        CHAT_MODEL="chat-model",
        ARK_API_KEY="ark-key",
        ARK_BASE_URL="http://ark/v1",
        ARK_CHAT_MODEL="ark-model",
    )

    assert config.CHAT_API_KEY == "chat-key"
    assert config.CHAT_BASE_URL == "http://chat/v1"
    assert config.CHAT_MODEL == "chat-model"
    assert config.OPENAI_API_KEY == "chat-key"
    assert config.OPENAI_BASE_URL == "http://chat/v1"


def test_chat_config_fallback_order_is_ark_yaml_then_legacy(reload_config):
    yaml_llm = {
        "api_key": "yaml-key",
        "base_url": "http://yaml/v1",
        "chat_model": "yaml-model",
    }
    config = reload_config(
        yaml_llm=yaml_llm,
        ARK_API_KEY="ark-key",
        ARK_BASE_URL="http://ark/v1",
        ARK_CHAT_MODEL="ark-model",
        OPENAI_API_KEY="legacy-key",
        OPENAI_BASE_URL="http://legacy/v1",
        OPENAI_CHAT_MODEL="legacy-model",
    )
    assert (config.CHAT_API_KEY, config.CHAT_BASE_URL, config.CHAT_MODEL) == (
        "ark-key",
        "http://ark/v1",
        "ark-model",
    )

    config = reload_config(
        yaml_llm=yaml_llm,
        OPENAI_API_KEY="legacy-key",
        OPENAI_BASE_URL="http://legacy/v1",
        OPENAI_CHAT_MODEL="legacy-model",
    )
    assert (config.CHAT_API_KEY, config.CHAT_BASE_URL, config.CHAT_MODEL) == (
        "yaml-key",
        "http://yaml/v1",
        "yaml-model",
    )

    config = reload_config(
        yaml_llm={},
        OPENAI_API_KEY="legacy-key",
        OPENAI_BASE_URL="http://legacy/v1",
        OPENAI_CHAT_MODEL="legacy-model",
    )
    assert (config.CHAT_API_KEY, config.CHAT_BASE_URL, config.CHAT_MODEL) == (
        "legacy-key",
        "http://legacy/v1",
        "legacy-model",
    )


def test_chat_config_does_not_mix_incomplete_providers(reload_config):
    config = reload_config(
        yaml_llm={},
        CHAT_BASE_URL="http://chat/v1",
        ARK_CHAT_MODEL="ark-model",
        OPENAI_API_KEY="legacy-key",
    )

    assert (config.CHAT_API_KEY, config.CHAT_BASE_URL, config.CHAT_MODEL) == (
        "",
        "",
        "",
    )
    assert config.OPENAI_API_KEY == ""
    assert config.OPENAI_BASE_URL == ""


def test_incomplete_high_priority_provider_falls_back_to_complete_provider(
    reload_config,
):
    config = reload_config(
        yaml_llm={},
        CHAT_API_KEY="chat-key",
        CHAT_BASE_URL="http://chat/v1",
        CHAT_MODEL=" ",
        ARK_API_KEY=" ark-key ",
        ARK_BASE_URL=" http://ark/v1 ",
        ARK_CHAT_MODEL=" ark-model ",
    )

    assert (config.CHAT_API_KEY, config.CHAT_BASE_URL, config.CHAT_MODEL) == (
        "ark-key",
        "http://ark/v1",
        "ark-model",
    )


def test_chat_config_rejects_whitespace_only_provider_values(reload_config):
    config = reload_config(
        yaml_llm={
            "api_key": " ",
            "base_url": "\t",
            "chat_model": "\n",
        },
        CHAT_API_KEY=" ",
        CHAT_BASE_URL="\t",
        CHAT_MODEL="\n",
        ARK_API_KEY=" ",
        ARK_BASE_URL="\t",
        ARK_CHAT_MODEL="\n",
        OPENAI_API_KEY=" ",
        OPENAI_BASE_URL="\t",
        OPENAI_CHAT_MODEL="\n",
    )

    assert (config.CHAT_API_KEY, config.CHAT_BASE_URL, config.CHAT_MODEL) == (
        "",
        "",
        "",
    )


def test_embedding_config_is_separate_and_optional(reload_config):
    config = reload_config(
        yaml_llm={},
        CHAT_API_KEY="chat-key",
        CHAT_BASE_URL="http://chat/v1",
        CHAT_MODEL="chat-model",
        EMBEDDING_MODEL="bge-test",
    )

    assert config.CHAT_BASE_URL == "http://chat/v1"
    assert config.EMBEDDING_API_KEY == ""
    assert config.EMBEDDING_BASE_URL == ""
    assert config.EMBEDDING_MODEL == "bge-test"


def test_embedding_config_uses_its_own_values(reload_config):
    config = reload_config(
        yaml_llm={},
        EMBEDDING_API_KEY="embedding-key",
        EMBEDDING_BASE_URL="http://embedding/v1",
        EMBEDDING_MODEL="bge-test",
    )

    assert config.EMBEDDING_API_KEY == "embedding-key"
    assert config.EMBEDDING_BASE_URL == "http://embedding/v1"
    assert config.EMBEDDING_MODEL == "bge-test"


def test_openai_client_remains_a_chat_client_alias(monkeypatch):
    from app.services import llm_client

    sentinel = object()
    monkeypatch.setattr(llm_client, "chat_client", lambda: sentinel)

    assert llm_client.openai_client() is sentinel


def test_public_config_reports_chat_and_embedding_separately(monkeypatch):
    from app import main

    monkeypatch.setattr(main, "OPENAI_API_KEY", "chat-key")
    monkeypatch.setattr(main, "OPENAI_BASE_URL", "http://chat/v1")
    monkeypatch.setattr(main, "CHAT_MODEL", "chat-model")
    monkeypatch.setattr(main, "EMBEDDING_BASE_URL", "")
    monkeypatch.setattr(main, "EMBEDDING_MODEL", "")

    result = main.public_config()

    assert result["chat"] == {
        "configured": True,
        "base_url_configured": True,
        "model_configured": True,
        "model": "已配置（已隔离）",
    }
    assert result["embedding"] == {
        "base_url_configured": False,
        "model_configured": False,
        "model": "未配置（当前RAG未使用embedding API）",
    }


def test_public_config_requires_complete_chat_provider(monkeypatch):
    from app import main

    monkeypatch.setattr(main, "OPENAI_API_KEY", "")
    monkeypatch.setattr(main, "OPENAI_BASE_URL", "http://chat/v1")
    monkeypatch.setattr(main, "CHAT_MODEL", "chat-model")

    assert main.public_config()["chat"]["configured"] is False
