import pytest

from finai_academy import Settings


def test_local_defaults() -> None:
    settings = Settings()

    assert settings.provider == "ollama"
    assert settings.chat_model == "qwen3:8b"
    assert settings.embedding_provider == "ollama"
    assert settings.embedding_model == "qwen3-embedding:0.6b"


def test_openai_environment_uses_openai_specific_defaults(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")
    monkeypatch.setenv("FINAI_MODEL_PROVIDER", "openai")
    monkeypatch.delenv("FINAI_CHAT_MODEL", raising=False)
    monkeypatch.delenv("FINAI_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("FINAI_EMBEDDING_MODEL", raising=False)

    settings = Settings.from_environment(env_file=env_file)

    assert settings.provider == "openai"
    assert settings.chat_model == "gpt-5.6-luna"
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


def test_settings_reject_unknown_reasoning_effort() -> None:
    with pytest.raises(ValueError, match="FINAI_REASONING_EFFORT"):
        Settings(provider="openai", reasoning_effort="extreme")


def test_settings_loads_an_explicit_env_file(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "FINAI_MODEL_PROVIDER=ollama\n"
        "FINAI_CHAT_MODEL=qwen3:4b\n"
        "FINAI_EMBEDDING_MODEL=qwen3-embedding:0.6b\n",
        encoding="utf-8",
    )
    for variable in (
        "FINAI_MODEL_PROVIDER",
        "FINAI_CHAT_MODEL",
        "FINAI_EMBEDDING_PROVIDER",
        "FINAI_EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(variable, raising=False)

    settings = Settings.from_environment(env_file=env_file)

    assert settings.chat_model == "qwen3:4b"
    assert settings.embedding_model == "qwen3-embedding:0.6b"


def test_shell_environment_overrides_env_file(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("FINAI_CHAT_MODEL=qwen3:4b\n", encoding="utf-8")
    monkeypatch.setenv("FINAI_CHAT_MODEL", "qwen3:8b")

    settings = Settings.from_environment(env_file=env_file)

    assert settings.chat_model == "qwen3:8b"
