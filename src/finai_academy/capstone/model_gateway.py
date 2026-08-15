"""Provider-neutral structured model boundary used by the capstone."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from finai_academy.settings import Settings

ResponseT = TypeVar("ResponseT", bound=BaseModel)


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
            from langchain_openai import ChatOpenAI
        except ImportError as error:  # pragma: no cover - depends on optional extras
            raise RuntimeError(
                "OpenAI support is not installed. Run `uv sync --extra ai`."
            ) from error

        model = ChatOpenAI(model=settings.chat_model, temperature=0)
        return LangChainStructuredModel(model)

    raise ValueError(
        f"Unsupported FINAI_MODEL_PROVIDER={settings.provider!r}. "
        "Choose 'ollama' or 'openai'."
    )
