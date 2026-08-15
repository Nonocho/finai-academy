from importlib.util import find_spec

import pytest

from finai_academy import providers
from finai_academy.settings import Settings


def test_provider_boundary_is_available() -> None:
    assert find_spec("finai_academy.providers") is not None


def test_provider_summary_never_exposes_api_keys() -> None:
    settings = Settings(provider="openai", chat_model="gpt-5.1")

    summary = providers.provider_summary(settings)

    assert summary == {
        "chat_provider": "openai",
        "chat_model": "gpt-5.1",
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-small",
    }


def test_openai_configuration_reports_a_missing_key_without_reading_the_network() -> None:
    settings = Settings(provider="openai")

    issues = providers.check_provider_configuration(settings, environ={})

    assert issues == ("OPENAI_API_KEY is required when FINAI_MODEL_PROVIDER=openai.",)


def test_model_run_rejects_negative_latency() -> None:
    with pytest.raises(ValueError, match="latency_ms"):
        providers.ModelRun(provider="ollama", model="qwen3:8b", text="answer", latency_ms=-1)


def test_chat_factory_builds_the_configured_ollama_model() -> None:
    model = providers.create_chat_model(Settings())

    assert model.__class__.__name__ == "ChatOllama"
    assert model.model == "qwen3:8b"


def test_chat_factory_builds_the_configured_openai_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    model = providers.create_chat_model(Settings(provider="openai"))

    assert model.__class__.__name__ == "ChatOpenAI"
    assert model.model_name == "gpt-5-mini"


@pytest.mark.parametrize(
    ("provider", "expected_class"),
    [("ollama", "OllamaEmbeddings"), ("openai", "OpenAIEmbeddings")],
)
def test_embedding_factory_builds_the_configured_provider(
    provider: str,
    expected_class: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    settings = Settings(provider=provider, embedding_provider=provider)

    embeddings = providers.create_embeddings(settings)

    assert embeddings.__class__.__name__ == expected_class
