"""Run one explicit, secret-safe live smoke test for the OpenAI capstone route."""

from __future__ import annotations

from finai_academy.capstone import ResearchRequest, build_copilot_for_request
from finai_academy.settings import Settings


def main() -> int:
    """Run the fixed mission and print a summary only after every gate passes."""

    try:
        settings = Settings.from_environment()
        if settings.provider != "openai":
            return 1
        request = ResearchRequest.reference(provider="openai", model=settings.chat_model)
        result = build_copilot_for_request(request, settings).run(request)
        citation_integrity = next(
            (
                metric.value
                for metric in result.deterministic_evaluation.metrics
                if metric.name == "citation_integrity"
            ),
            0.0,
        )
        if not (
            result.status == "completed"
            and result.evidence_gate.passed
            and result.briefing is not None
            and len(result.briefing.cited_facts) == 2
            and citation_integrity == 1.0
        ):
            return 1
    except Exception:  # noqa: BLE001 - live provider details must remain private
        return 1

    print(
        f"provider=openai model={request.model} status=completed "
        f"citations={len(result.briefing.cited_facts)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
