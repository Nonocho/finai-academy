"""Run one explicit, secret-safe live smoke test for the OpenAI capstone route."""

from __future__ import annotations

from finai_academy.capstone import ResearchRequest, build_copilot_for_request
from finai_academy.settings import Settings


def _failure(*, stage: str, code: str) -> int:
    """Emit a fixed public diagnostic without exposing provider data."""

    print(f"status=failed stage={stage} code={code}")
    return 1


def main() -> int:
    """Run the fixed mission and print a summary only after every gate passes."""

    try:
        settings = Settings.from_environment()
    except Exception:  # noqa: BLE001 - environment details must remain private
        return _failure(stage="configuration", code="settings_unavailable")
    if settings.provider != "openai":
        return _failure(stage="configuration", code="openai_route_required")

    try:
        request = ResearchRequest.reference(provider="openai", model=settings.chat_model)
        result = build_copilot_for_request(request, settings).run(request)
    except Exception:  # noqa: BLE001 - live provider details must remain private
        return _failure(stage="provider", code="live_run_failed")
    if result.status == "provider_error":
        return _failure(stage="provider", code="provider_result_failed")
    if result.status != "completed":
        return _failure(stage="result", code="run_not_completed")
    if not result.evidence_gate.passed:
        return _failure(stage="validation", code="evidence_gate_failed")
    if result.briefing is None:
        return _failure(stage="validation", code="briefing_missing")
    if len(result.briefing.cited_facts) != 2:
        return _failure(stage="validation", code="citation_count_invalid")
    citation_integrity = next(
        (
            metric.value
            for metric in result.deterministic_evaluation.metrics
            if metric.name == "citation_integrity"
        ),
        0.0,
    )
    if citation_integrity != 1.0:
        return _failure(stage="validation", code="citation_integrity_failed")

    print(
        f"provider=openai model={request.model} status=completed "
        f"citations={len(result.briefing.cited_facts)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
