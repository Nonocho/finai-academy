"""Provider-neutral structured model boundary used by the capstone."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any, Literal, Protocol, TypeVar
from urllib.request import urlopen

from pydantic import BaseModel, ConfigDict, Field

from finai_academy.settings import Settings

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class ProviderReadiness(BaseModel):
    """Public, credential-free readiness for one explicitly selected route."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    provider: Literal["recorded", "ollama", "openai"]
    model: str = Field(min_length=1)
    available: bool
    status: Literal["available", "unavailable"]
    guidance: str = Field(min_length=1)
    fallback_provider: None = None


class StructuredModel(Protocol):
    """Smallest model interface the application needs in Module 00."""

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseT],
    ) -> ResponseT:
        """Return a response validated against ``response_model``."""


class ModelOutputError(RuntimeError):
    """Raised when a provider does not return the requested structured output."""


class OpenAIResponsesStructuredModel:
    """Adapt OpenAI Responses structured parsing to the capstone boundary."""

    def __init__(self, *, client: Any, model: str, reasoning_effort: str = "medium") -> None:
        self._client = client
        self._model = model
        self._reasoning_effort = reasoning_effort

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseT],
    ) -> ResponseT:
        response = self._client.responses.parse(
            model=self._model,
            reasoning={"effort": self._reasoning_effort},
            instructions=system_prompt,
            input=user_prompt,
            text_format=response_model,
            store=False,
        )
        if response.output_parsed is None:
            raise ModelOutputError("provider returned no structured output")
        return response_model.model_validate(response.output_parsed)


class LangChainStructuredModel:
    """Adapt a LangChain chat model to the capstone's narrow interface."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseT],
    ) -> ResponseT:
        with_structured_output = self._model.with_structured_output
        structured_model = with_structured_output(response_model)
        result = structured_model.invoke(
            [
                ("system", system_prompt),
                ("human", user_prompt),
            ]
        )
        if isinstance(result, response_model):
            return result
        return response_model.model_validate(result)


def create_structured_model(settings: Settings) -> StructuredModel:
    """Create the configured model while keeping provider imports optional."""

    provider = settings.provider.casefold()

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as error:  # pragma: no cover - depends on optional extras
            raise RuntimeError(
                "Ollama support is not installed. Run `uv sync --extra ai`."
            ) from error

        model = ChatOllama(
            model=settings.chat_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )
        return LangChainStructuredModel(model)

    if provider == "openai":
        try:
            from openai import OpenAI
        except ImportError as error:  # pragma: no cover - depends on optional extras
            raise RuntimeError(
                "OpenAI support is not installed. Run `uv sync --extra ai`."
            ) from error

        return OpenAIResponsesStructuredModel(
            client=OpenAI(),
            model=settings.chat_model,
            reasoning_effort=settings.reasoning_effort,
        )

    raise ValueError(
        f"Unsupported FINAI_MODEL_PROVIDER={settings.provider!r}. "
        "Choose 'ollama' or 'openai'."
    )


def provider_readiness(
    provider: str,
    model: str,
    *,
    settings: Settings | None = None,
    ollama_probe: Callable[[], bool] | None = None,
) -> ProviderReadiness:
    """Report readiness for exactly the requested provider without fallback."""

    selected_provider = provider.casefold().strip()
    selected_model = model.strip()
    if selected_provider not in {"recorded", "ollama", "openai"}:
        raise ValueError("provider must be recorded, ollama, or openai")
    if not selected_model:
        raise ValueError("model must not be blank")

    if selected_provider == "recorded":
        return ProviderReadiness(
            provider="recorded",
            model=selected_model,
            available=True,
            status="available",
            guidance="Recorded mode is ready for certified offline analysis.",
        )

    if selected_provider == "openai":
        available = bool(os.environ.get("OPENAI_API_KEY", "").strip())
        return ProviderReadiness(
            provider="openai",
            model=selected_model,
            available=available,
            status="available" if available else "unavailable",
            guidance=(
                "OpenAI is configured for the selected model."
                if available
                else "Set OPENAI_API_KEY to use the selected OpenAI model."
            ),
        )

    selected_settings = settings or Settings(
        provider="ollama",
        chat_model=selected_model,
    )
    probe = ollama_probe or (
        lambda: _probe_ollama(
            base_url=selected_settings.ollama_base_url,
            model=selected_model,
        )
    )
    try:
        available = probe() is True
    except Exception:  # noqa: BLE001 - readiness exposes no provider details
        available = False
    return ProviderReadiness(
        provider="ollama",
        model=selected_model,
        available=available,
        status="available" if available else "unavailable",
        guidance=(
            "Ollama is ready with the selected model."
            if available
            else "Start Ollama and run: ollama pull qwen3:4b"
        ),
    )


def _probe_ollama(*, base_url: str, model: str) -> bool:
    """Perform one bounded local health check and require the selected model."""

    endpoint = f"{base_url.rstrip('/')}/api/tags"
    with urlopen(endpoint, timeout=1.0) as response:
        payload = json.loads(response.read().decode("utf-8"))
    models = payload.get("models")
    if not isinstance(models, list):
        return False
    names = {
        item.get("name")
        for item in models
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    return model in names
