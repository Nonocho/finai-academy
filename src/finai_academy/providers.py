"""Provider-neutral model and embedding boundaries for all course notebooks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from os import environ as process_environment
from typing import Any, Protocol

from finai_academy.measurement import TokenUsage
from finai_academy.settings import Settings


class EmbeddingModel(Protocol):
    """Provider-neutral embedding boundary used by retrieval and chunking."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class ModelRun:
    """Normalized metadata returned by the first model-gateway lesson."""

    provider: str
    model: str
    text: str
    latency_ms: float
    token_usage: TokenUsage | None = None
    prompt_version: str | None = None

    def __post_init__(self) -> None:
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be greater than or equal to zero")
        if self.prompt_version is not None and not self.prompt_version.strip():
            raise ValueError("prompt_version must not be empty when provided")


def normalize_token_usage(usage_metadata: Mapping[str, Any] | None) -> TokenUsage | None:
    """Normalize complete LangChain usage metadata without inventing missing counts."""

    if usage_metadata is None:
        return None
    required_fields = ("input_tokens", "output_tokens", "total_tokens")
    if any(field not in usage_metadata for field in required_fields):
        return None
    return TokenUsage(
        input_tokens=usage_metadata["input_tokens"],
        output_tokens=usage_metadata["output_tokens"],
        total_tokens=usage_metadata["total_tokens"],
    )


def provider_summary(settings: Settings) -> dict[str, str]:
    """Return safe, display-ready provider information without environment secrets."""

    return {
        "chat_provider": settings.provider,
        "chat_model": settings.chat_model,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
    }


def check_provider_configuration(
    settings: Settings,
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Report missing credentials without starting a network connection."""

    current_environment = process_environment if environ is None else environ
    uses_openai = "openai" in {settings.provider, settings.embedding_provider}
    if uses_openai and not current_environment.get("OPENAI_API_KEY"):
        return ("OPENAI_API_KEY is required when FINAI_MODEL_PROVIDER=openai.",)
    return ()


def create_chat_model(settings: Settings) -> Any:
    """Create the configured LangChain chat model without making a model call."""

    if settings.provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as error:  # pragma: no cover - optional dependency boundary
            raise RuntimeError("Ollama support is not installed. Run `uv sync --extra ai`.") from error

        return ChatOllama(
            model=settings.chat_model,
            base_url=settings.ollama_base_url,
        )

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as error:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("OpenAI support is not installed. Run `uv sync --extra ai`.") from error

    return ChatOpenAI(model=settings.chat_model)


def create_embeddings(settings: Settings) -> Any:
    """Create the configured embedding client without embedding any text."""

    if settings.embedding_provider == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError as error:  # pragma: no cover - optional dependency boundary
            raise RuntimeError("Ollama support is not installed. Run `uv sync --extra ai`.") from error

        return OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url,
        )

    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError as error:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("OpenAI support is not installed. Run `uv sync --extra ai`.") from error

    return OpenAIEmbeddings(model=settings.embedding_model)
