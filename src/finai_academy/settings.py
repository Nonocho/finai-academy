"""Provider-neutral settings shared by course examples."""

from dataclasses import dataclass
from os import getenv
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SUPPORTED_PROVIDERS = frozenset({"ollama", "openai"})
SUPPORTED_REASONING_EFFORTS = frozenset({"low", "medium", "high"})

CHAT_DEFAULTS = {
    "ollama": "qwen3:8b",
    "openai": "gpt-5.6-luna",
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
    reasoning_effort: str = "medium"

    def __post_init__(self) -> None:
        provider = self.provider.casefold().strip()
        embedding_provider = (self.embedding_provider or provider).casefold().strip()
        reasoning_effort = self.reasoning_effort.casefold().strip()

        self._validate_provider("FINAI_MODEL_PROVIDER", provider)
        self._validate_provider("FINAI_EMBEDDING_PROVIDER", embedding_provider)
        self._validate_reasoning_effort(reasoning_effort)

        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "embedding_provider", embedding_provider)
        object.__setattr__(self, "reasoning_effort", reasoning_effort)
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

    @staticmethod
    def _validate_reasoning_effort(reasoning_effort: str) -> None:
        if reasoning_effort not in SUPPORTED_REASONING_EFFORTS:
            choices = ", ".join(sorted(SUPPORTED_REASONING_EFFORTS))
            raise ValueError(
                "Unsupported FINAI_REASONING_EFFORT="
                f"{reasoning_effort!r}; choose {choices}."
            )

    @classmethod
    def from_environment(cls, env_file: str | Path | None = None) -> "Settings":
        load_dotenv(
            dotenv_path=Path(env_file) if env_file is not None else PROJECT_ROOT / ".env",
            override=False,
        )
        provider = getenv("FINAI_MODEL_PROVIDER", "ollama")
        return cls(
            provider=provider,
            chat_model=getenv("FINAI_CHAT_MODEL", ""),
            embedding_provider=getenv("FINAI_EMBEDDING_PROVIDER", provider),
            embedding_model=getenv("FINAI_EMBEDDING_MODEL", ""),
            ollama_base_url=getenv("FINAI_OLLAMA_BASE_URL", "http://localhost:11434"),
            reasoning_effort=getenv("FINAI_REASONING_EFFORT", "medium"),
        )
