"""Provider-neutral settings shared by course examples."""

from dataclasses import dataclass
from os import getenv


@dataclass(frozen=True)
class Settings:
    """Runtime settings with a local-first default."""

    provider: str = "ollama"
    chat_model: str = "qwen3:8b"
    embedding_model: str = "qwen3-embedding:0.6b"
    ollama_base_url: str = "http://localhost:11434"

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            provider=getenv("FINAI_MODEL_PROVIDER", cls.provider),
            chat_model=getenv("FINAI_CHAT_MODEL", cls.chat_model),
            embedding_model=getenv("FINAI_EMBEDDING_MODEL", cls.embedding_model),
            ollama_base_url=getenv("FINAI_OLLAMA_BASE_URL", cls.ollama_base_url),
        )
