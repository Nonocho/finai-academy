from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TypeVar

import pytest
from pydantic import BaseModel

from finai_academy.capstone import model_gateway
from finai_academy.capstone.model_gateway import (
    ModelOutputError,
    OpenAIResponsesStructuredModel,
    provider_readiness,
)
from finai_academy.capstone.models import ResearchRequest
from finai_academy.capstone.service import build_copilot_for_request
from finai_academy.settings import Settings

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class ExpectedBrief(BaseModel):
    answer: str


class FakeResponses:
    def __init__(self, output_parsed: ExpectedBrief | None) -> None:
        self.output_parsed = output_parsed
        self.last_call: dict[str, object] | None = None

    def parse(self, **kwargs: object) -> SimpleNamespace:
        self.last_call = kwargs
        return SimpleNamespace(output_parsed=self.output_parsed)


class FakeOpenAIClient:
    def __init__(self, output_parsed: ExpectedBrief | None) -> None:
        self.responses = FakeResponses(output_parsed)


class CompletedSmokeResult:
    status = "completed"
    evidence_gate = SimpleNamespace(passed=True)
    briefing = SimpleNamespace(cited_facts=(object(), object()))
    deterministic_evaluation = SimpleNamespace(
        metrics=(SimpleNamespace(name="citation_integrity", value=1.0),)
    )


def _load_smoke_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts/smoke_capstone_openai.py"
    spec = importlib.util.spec_from_file_location("smoke_capstone_openai", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_openai_smoke_prints_only_safe_completed_summary(monkeypatch, capsys) -> None:
    smoke_capstone_openai = _load_smoke_module()
    settings = Settings(provider="openai", chat_model="gpt-5.6-luna")
    monkeypatch.setattr(smoke_capstone_openai.Settings, "from_environment", lambda: settings)
    monkeypatch.setattr(
        smoke_capstone_openai,
        "build_copilot_for_request",
        lambda request, configured_settings: SimpleNamespace(
            run=lambda received_request: CompletedSmokeResult()
        ),
    )

    assert smoke_capstone_openai.main() == 0
    assert capsys.readouterr().out == (
        "provider=openai model=gpt-5.6-luna status=completed citations=2\n"
    )


def test_openai_smoke_fails_silently_for_a_non_openai_route(monkeypatch, capsys) -> None:
    smoke_capstone_openai = _load_smoke_module()
    monkeypatch.setattr(
        smoke_capstone_openai.Settings,
        "from_environment",
        lambda: Settings(provider="ollama"),
    )

    assert smoke_capstone_openai.main() == 1
    assert capsys.readouterr().out == ""


def test_openai_smoke_rejects_an_unexpected_citation_count(monkeypatch, capsys) -> None:
    smoke_capstone_openai = _load_smoke_module()
    settings = Settings(provider="openai", chat_model="gpt-5.6-luna")
    incomplete_result = CompletedSmokeResult()
    incomplete_result.briefing = SimpleNamespace(cited_facts=(object(),))
    monkeypatch.setattr(smoke_capstone_openai.Settings, "from_environment", lambda: settings)
    monkeypatch.setattr(
        smoke_capstone_openai,
        "build_copilot_for_request",
        lambda request, configured_settings: SimpleNamespace(
            run=lambda received_request: incomplete_result
        ),
    )

    assert smoke_capstone_openai.main() == 1
    assert capsys.readouterr().out == ""


def test_openai_adapter_uses_luna_medium_structured_responses() -> None:
    client = FakeOpenAIClient(output_parsed=ExpectedBrief(answer="Evidence is cited."))
    model = OpenAIResponsesStructuredModel(
        client=client,
        model="gpt-5.6-luna",
        reasoning_effort="medium",
    )

    result = model.generate(
        system_prompt="Use only cited evidence.",
        user_prompt="Evidence payload",
        response_model=ExpectedBrief,
    )

    assert isinstance(result, ExpectedBrief)
    assert client.responses.last_call == {
        "model": "gpt-5.6-luna",
        "reasoning": {"effort": "medium"},
        "instructions": "Use only cited evidence.",
        "input": "Evidence payload",
        "text_format": ExpectedBrief,
        "store": False,
    }


def test_openai_adapter_rejects_missing_structured_output() -> None:
    model = OpenAIResponsesStructuredModel(
        client=FakeOpenAIClient(output_parsed=None),
        model="gpt-5.6-luna",
    )

    with pytest.raises(ModelOutputError, match="no structured output"):
        model.generate(
            system_prompt="Use only cited evidence.",
            user_prompt="Evidence payload",
            response_model=ExpectedBrief,
        )


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


class StatementSelectionModel:
    def __init__(self, *, executive_summary_id: str = "executive_summary:1") -> None:
        self.executive_summary_id = executive_summary_id
        self.user_prompt = ""

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseT],
    ) -> ResponseT:
        del system_prompt
        self.user_prompt = user_prompt
        if "executive_summary" in response_model.model_fields:
            return response_model.model_validate(
                {
                    "executive_summary": self.executive_summary_id,
                    "cross_company_observations": ["Host-independent provider prose."],
                    "interpretation": ["Host-independent provider prose."],
                    "limitations": ["Host-independent provider prose."],
                }
            )
        return response_model.model_validate(
            {
                "executive_summary_id": self.executive_summary_id,
                "cross_company_observation_ids": ["cross_company_observation:1"],
                "interpretation_ids": ["interpretation:1"],
                "limitation_ids": ["limitation:2", "limitation:1"],
            }
        )


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


@pytest.mark.parametrize(
    "unsupported_selection",
    [
        "NVIDIA has an unassailable competitive moat.",
        "NVIDIA revenue will reach USD 500 billion.",
        "Investors should buy NVIDIA shares.",
        "NVIDIA price target is USD 250.",
    ],
    ids=["unsupported-claim", "fabricated-number", "recommendation", "price-target"],
)
def test_live_provider_cannot_turn_unsupported_prose_into_a_completed_briefing(
    monkeypatch,
    unsupported_selection: str,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-configured-provider-value")
    model = StatementSelectionModel(executive_summary_id=unsupported_selection)
    monkeypatch.setattr(
        model_gateway,
        "create_structured_model",
        lambda settings: model,
    )
    request = ResearchRequest.reference(provider="openai", model="gpt-5-mini")

    result = build_copilot_for_request(request, Settings()).run(request)
    payload = result.model_dump_json()

    assert result.status == "provider_error"
    assert result.briefing is None
    assert unsupported_selection not in payload


def test_valid_live_provider_selection_reconstructs_only_host_statement_units(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-configured-provider-value")
    model = StatementSelectionModel()
    monkeypatch.setattr(
        model_gateway,
        "create_structured_model",
        lambda settings: model,
    )
    request = ResearchRequest.reference(provider="openai", model="gpt-5-mini")
    host_briefing = build_copilot_for_request(
        ResearchRequest.reference(),
        Settings(),
    ).run(ResearchRequest.reference()).briefing

    result = build_copilot_for_request(request, Settings()).run(request)

    assert result.status == "completed"
    assert result.briefing is not None
    assert host_briefing is not None
    assert result.briefing.executive_summary == host_briefing.executive_summary
    assert result.briefing.cited_facts == host_briefing.cited_facts
    assert result.briefing.cross_company_observations == (
        host_briefing.cross_company_observations[0],
    )
    assert result.briefing.interpretation == (host_briefing.interpretation[0],)
    assert result.briefing.limitations == tuple(reversed(host_briefing.limitations))
    prompt = json.loads(model.user_prompt)
    assert {unit["id"] for unit in prompt["certified_statement_units"]} == {
        "executive_summary:1",
        "cross_company_observation:1",
        "interpretation:1",
        "limitation:1",
        "limitation:2",
    }
