from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel

from finai_academy.capstone import model_gateway
from finai_academy.capstone.model_gateway import provider_readiness
from finai_academy.capstone.models import ResearchRequest
from finai_academy.capstone.service import build_copilot_for_request
from finai_academy.settings import Settings

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class FailingStructuredModel:
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseT],
    ) -> ResponseT:
        del system_prompt, user_prompt, response_model
        raise RuntimeError(
            "provider failed with OPENAI_API_KEY=sk-secret-provider-value "
            "at /Users/example/private/request.json"
        )


class UnsafeWordingModel:
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseT],
    ) -> ResponseT:
        del system_prompt, user_prompt, response_model
        return {  # type: ignore[return-value]
            "executive_summary": "Use sk-secret-provider-value from /Users/private/file.",
            "cross_company_observations": ["Unsafe provider output."],
            "interpretation": ["Unsafe provider output."],
            "limitations": ["Unsafe provider output."],
        }


class WindowsPathWordingModel:
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseT],
    ) -> ResponseT:
        del system_prompt, user_prompt, response_model
        return {  # type: ignore[return-value]
            "executive_summary": r"Provider cache C:\Users\analyst\private\response.json.",
            "cross_company_observations": ["Unsafe provider output."],
            "interpretation": ["Unsafe provider output."],
            "limitations": ["Unsafe provider output."],
        }


def test_openai_without_key_is_unavailable_without_fallback(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    readiness = provider_readiness(provider="openai", model="gpt-5-mini")

    assert readiness.provider == "openai"
    assert readiness.model == "gpt-5-mini"
    assert not readiness.available
    assert readiness.status == "unavailable"
    assert "OPENAI_API_KEY" in readiness.guidance
    assert readiness.fallback_provider is None


def test_openai_with_key_is_available_without_serializing_the_key(monkeypatch) -> None:
    secret = "sk-injected-safe-test-value"
    monkeypatch.setenv("OPENAI_API_KEY", secret)

    readiness = provider_readiness(provider="openai", model="gpt-5-mini")
    payload = readiness.model_dump_json()

    assert readiness.available
    assert readiness.status == "available"
    assert readiness.fallback_provider is None
    assert secret not in payload
    assert "sk-" not in payload


def test_ollama_false_probe_uses_exact_guidance_and_no_fallback() -> None:
    readiness = provider_readiness(
        provider="ollama",
        model="qwen3:4b",
        ollama_probe=lambda: False,
    )

    assert not readiness.available
    assert readiness.guidance == "Start Ollama and run: ollama pull qwen3:4b"
    assert readiness.fallback_provider is None


def test_ollama_true_probe_is_available() -> None:
    readiness = provider_readiness(
        provider="ollama",
        model="qwen3:4b",
        ollama_probe=lambda: True,
    )

    assert readiness.available
    assert readiness.status == "available"
    assert readiness.fallback_provider is None


def test_recorded_factory_never_constructs_or_probes_a_live_model(monkeypatch) -> None:
    def fail_if_called(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("recorded route reached a live dependency")

    monkeypatch.setattr(model_gateway, "create_structured_model", fail_if_called)
    request = ResearchRequest.reference()

    result = build_copilot_for_request(
        request,
        Settings(),
        ollama_probe=fail_if_called,
    ).run(request)

    assert result.provider == "recorded"
    assert result.status == "completed"


def test_live_provider_failure_returns_only_a_generic_public_error(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-provider-value")
    monkeypatch.setattr(
        model_gateway,
        "create_structured_model",
        lambda settings: FailingStructuredModel(),
    )
    request = ResearchRequest.reference(
        provider="openai",
        model="gpt-5-mini",
    )

    result = build_copilot_for_request(request, Settings()).run(request)
    payload = result.model_dump_json()

    assert result.status == "provider_error"
    assert result.briefing is None
    assert result.trajectory[-1].failure_owner == "provider"
    assert "provider failed" not in payload
    assert "secret-provider-value" not in payload
    assert "/Users/" not in payload
    assert json.loads(payload)["status"] == "provider_error"


def test_unsafe_live_wording_is_rejected_as_a_sanitized_provider_error(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-configured-provider-value")
    monkeypatch.setattr(
        model_gateway,
        "create_structured_model",
        lambda settings: UnsafeWordingModel(),
    )
    request = ResearchRequest.reference(provider="openai", model="gpt-5-mini")

    result = build_copilot_for_request(request, Settings()).run(request)
    payload = result.model_dump_json()

    assert result.status == "provider_error"
    assert result.briefing is None
    assert "secret-provider-value" not in payload
    assert "/Users/" not in payload


def test_live_wording_with_a_windows_personal_path_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-configured-provider-value")
    monkeypatch.setattr(
        model_gateway,
        "create_structured_model",
        lambda settings: WindowsPathWordingModel(),
    )
    request = ResearchRequest.reference(provider="openai", model="gpt-5-mini")

    result = build_copilot_for_request(request, Settings()).run(request)

    assert result.status == "provider_error"
    assert result.briefing is None
    assert "C:\\Users\\" not in result.model_dump_json()
