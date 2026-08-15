"""Provider-neutral settings shared by course examples."""

from dataclasses import dataclass
from os import getenv

SUPPORTED_PROVIDERS = frozenset({"ollama", "openai"})

CHAT_DEFAULTS = {
    "ollama": "qwen3:8b",
    "openai": "gpt-5-mini",
}

EMBEDDING_DEFAULTS = {
    "ollama": "qwen3-embedding:0.6b",
    "openai": "text-embedding-3-small",
}


@dataclass(frozen=True)
class Settings:
    """Runtime settings with a local-first default."""

    provider: str = "ollama"
    chat_model: str = ""
    embedding_provider: str = ""
    embedding_model: str = ""
    ollama_base_url: str = "http://localhost:11434"

    def __post_init__(self) -> None:
        provider = self.provider.casefold().strip()
        embedding_provider = (self.embedding_provider or provider).casefold().strip()

        self._validate_provider("FINAI_MODEL_PROVIDER", provider)
        self._validate_provider("FINAI_EMBEDDING_PROVIDER", embedding_provider)

        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "embedding_provider", embedding_provider)
        object.__setattr__(self, "chat_model", self.chat_model or CHAT_DEFAULTS[provider])
        object.__setattr__(
            self,
            "embedding_model",
            self.embedding_model or EMBEDDING_DEFAULTS[embedding_provider],
        )

    @staticmethod
    def _validate_provider(variable_name: str, provider: str) -> None:
        if provider not in SUPPORTED_PROVIDERS:
            choices = ", ".join(sorted(SUPPORTED_PROVIDERS))
            raise ValueError(f"Unsupported {variable_name}={provider!r}; choose {choices}.")

    @classmethod
    def from_environment(cls) -> "Settings":
        provider = getenv("FINAI_MODEL_PROVIDER", "ollama")
        return cls(
            provider=provider,
            chat_model=getenv("FINAI_CHAT_MODEL", ""),
            embedding_provider=getenv("FINAI_EMBEDDING_PROVIDER", provider),
            embedding_model=getenv("FINAI_EMBEDDING_MODEL", ""),
            ollama_base_url=getenv("FINAI_OLLAMA_BASE_URL", "http://localhost:11434"),
        )
