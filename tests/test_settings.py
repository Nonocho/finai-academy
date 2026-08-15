import pytest

from finai_academy import Settings


def test_local_defaults() -> None:
    settings = Settings()

    assert settings.provider == "ollama"
    assert settings.chat_model == "qwen3:8b"
    assert settings.embedding_provider == "ollama"
    assert settings.embedding_model == "qwen3-embedding:0.6b"


def test_openai_environment_uses_openai_specific_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINAI_MODEL_PROVIDER", "openai")
    monkeypatch.delenv("FINAI_CHAT_MODEL", raising=False)
    monkeypatch.delenv("FINAI_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("FINAI_EMBEDDING_MODEL", raising=False)

    settings = Settings.from_environment()

    assert settings.provider == "openai"
    assert settings.chat_model == "gpt-5-mini"
    assert settings.embedding_provider == "openai"
    assert settings.embedding_model == "text-embedding-3-small"


def test_explicit_models_override_provider_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINAI_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("FINAI_CHAT_MODEL", "gpt-5.1")
    monkeypatch.setenv("FINAI_EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("FINAI_EMBEDDING_MODEL", "qwen3-embedding:4b")

    settings = Settings.from_environment()

    assert settings.chat_model == "gpt-5.1"
    assert settings.embedding_provider == "ollama"
    assert settings.embedding_model == "qwen3-embedding:4b"


def test_unsupported_provider_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINAI_MODEL_PROVIDER", "unsupported")

    with pytest.raises(ValueError, match="FINAI_MODEL_PROVIDER"):
        Settings.from_environment()
