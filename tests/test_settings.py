from finai_academy import Settings


def test_local_defaults() -> None:
    settings = Settings()

    assert settings.provider == "ollama"
    assert settings.chat_model
    assert settings.embedding_model
