"""Recorded and provider-neutral policies for Lesson 11 research runs."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from finai_academy.providers import create_chat_model
from finai_academy.research_planning import (
    AnalystBriefing,
    PlannerToolSpec,
    PlanStep,
    ReplanDecision,
    ResearchObservation,
    ResearchPlan,
)
from finai_academy.settings import Settings

MISSION = (
    "Produce a concise NVIDIA and Schneider Electric briefing. Compare their available "
    "valuation metrics and latest operating-growth evidence. Cite every factual claim "
    "and state which observations cannot be compared directly."
)

MAX_RESEARCH_STEPS = 6

INITIAL_RECORDED_STEPS = (
    PlanStep(
        step_id=1,
        capability="get_company_metric",
        arguments={"ticker": "NVDA", "metric": "P/E"},
        purpose="Collect NVIDIA valuation evidence.",
        expected_evidence=("NVIDIA P/E",),
    ),
    PlanStep(
        step_id=2,
        capability="get_company_metric",
        arguments={"ticker": "SU.PA", "metric": "P/E"},
        purpose="Collect Schneider Electric valuation evidence.",
        expected_evidence=("Schneider Electric P/E",),
    ),
    PlanStep(
        step_id=3,
        capability="get_company_metric",
        arguments={"ticker": "NVDA", "metric": "Revenue"},
        purpose="Attempt to collect NVIDIA revenue as a structured metric.",
        expected_evidence=("NVIDIA revenue",),
        depends_on=(1,),
    ),
    PlanStep(
        step_id=4,
        capability="search_financial_documents",
        arguments={"company": "Schneider Electric", "query": "revenue growth", "top_k": 2},
        purpose="Collect Schneider Electric operating evidence.",
        expected_evidence=("Schneider Electric revenue growth",),
        depends_on=(2,),
    ),
)

RECORDED_REPLACEMENT_STEPS = (
    PlanStep(
        step_id=5,
        capability="search_financial_documents",
        arguments={"company": "NVIDIA", "query": "revenue growth", "top_k": 2},
        purpose="Replace the unsupported metric with NVIDIA document evidence.",
        expected_evidence=("NVIDIA revenue growth",),
        depends_on=(1,),
    ),
    PlanStep(
        step_id=6,
        capability="search_financial_documents",
        arguments={"company": "Schneider Electric", "query": "energy management", "top_k": 2},
        purpose="Collect Schneider Electric operating-growth evidence.",
        expected_evidence=("Schneider Electric Energy Management growth",),
        depends_on=(2,),
    ),
)


async def recorded_planner(
    question: str, catalog: tuple[PlannerToolSpec, ...]
) -> ResearchPlan:
    """Return the maintained deterministic lesson plan without inspecting providers."""
    del catalog
    return ResearchPlan(goal=question, steps=INITIAL_RECORDED_STEPS)


async def recorded_replanner(state: Mapping[str, Any]) -> ReplanDecision:
    """Switch the maintained unsupported metric attempt to controlled document evidence."""
    observations = tuple(state.get("observations", ()))
    last = observations[-1] if observations else None
    if (
        isinstance(last, ResearchObservation)
        and last.step_id == 3
        and last.error_code == "unsupported_metric"
    ):
        return ReplanDecision(
            action="replace_remaining",
            reasoning="Use document search because Revenue is not a supported metric.",
            replacement_steps=RECORDED_REPLACEMENT_STEPS,
            limitations=("Revenue evidence comes from documents, not the metric snapshot.",),
        )
    if int(state.get("current_index", 0)) >= len(tuple(state.get("active_steps", ()))):
        return ReplanDecision(
            action="finish",
            reasoning="Every active research step has been attempted.",
        )
    return ReplanDecision(
        action="continue",
        reasoning="Continue with the next validated research step.",
    )


async def recorded_report_writer(
    question: str, observations: tuple[ResearchObservation, ...]
) -> AnalystBriefing:
    """Create the offline briefing from successful typed MCP observations only."""
    successful = tuple(item for item in observations if item.status == "ok")
    sources = tuple(
        dict.fromkeys(source for item in successful for source in item.source_references)
    )
    return briefing_from_verified_observations(question, successful, sources)


def briefing_from_verified_observations(
    question: str,
    observations: Sequence[ResearchObservation],
    sources: tuple[str, ...],
) -> AnalystBriefing:
    """Extract plain factual claims from successful metric and document observations."""
    facts: list[str] = []
    for observation in observations:
        if observation.status != "ok" or observation.result is None:
            continue
        result = observation.result
        if observation.capability == "get_company_metric":
            company = result.get("company")
            metric = result.get("metric")
            value = result.get("value")
            unit = result.get("unit")
            as_of = result.get("as_of")
            if all(isinstance(value_, str) for value_ in (company, metric, unit, as_of)) and isinstance(
                value, (int, float)
            ):
                facts.append(f"{company} {metric} was {value:g} {unit} as of {as_of}.")
        elif observation.capability == "search_financial_documents":
            company = result.get("company")
            hits = result.get("hits")
            if not isinstance(company, str) or not isinstance(hits, Sequence) or isinstance(
                hits, (str, bytes)
            ):
                continue
            for hit in hits:
                if not isinstance(hit, Mapping):
                    continue
                text = hit.get("text")
                period = hit.get("period")
                if isinstance(text, str) and isinstance(period, str):
                    facts.append(f"{company} ({period}): {text}")

    return AnalystBriefing(
        reported_facts=tuple(facts),
        cross_company_observations=(
            "The available valuation and operating-growth observations use different periods and scopes.",
        ),
        interpretation=(
            f"This briefing describes the verified evidence for the requested mission: {question}",
            "It is not investment advice.",
        ),
        limitations=(
            "The companies report in different currency units.",
            "The available observations cover different reporting periods.",
            "The companies have different business mixes, so operating evidence is not directly comparable.",
        ),
        source_references=sources,
    )


class _LivePolicy:
    """Lazy structured-output adapter shared by the three provider-neutral policies."""

    response_model: type[ResearchPlan | ReplanDecision | AnalystBriefing]

    def __init__(
        self,
        settings: Settings,
        *,
        model_factory: Callable[[Settings], Any] = create_chat_model,
    ) -> None:
        self._settings = settings
        self._model_factory = model_factory
        self._structured_model: Any | None = None

    async def _respond(self, system_prompt: str, context: Mapping[str, Any]) -> Any:
        if self._structured_model is None:
            model = self._model_factory(self._settings)
            self._structured_model = model.with_structured_output(self.response_model)
        result = await self._structured_model.ainvoke(
            [("system", system_prompt), ("human", json.dumps(context, sort_keys=True))]
        )
        if isinstance(result, self.response_model):
            return result
        return self.response_model.model_validate(result)


class LivePlanner(_LivePolicy):
    """Request a typed initial plan from the configured provider only when invoked."""

    response_model = ResearchPlan

    async def __call__(
        self,
        question: str,
        catalog: tuple[PlannerToolSpec, ...],
    ) -> ResearchPlan:
        result = await self._respond(
            "Create a concise, factual financial research plan. Use only the supplied allowlisted "
            "catalog, at most six steps, and no investment advice. Give observable concise reasons, "
            "not hidden reasoning.",
            {"mission": question, "allowlisted_catalog": _safe_catalog(catalog)},
        )
        return result


class LiveReplanner(_LivePolicy):
    """Request a typed decision from typed execution summaries without runtime internals."""

    response_model = ReplanDecision

    async def __call__(self, state: Mapping[str, Any]) -> ReplanDecision:
        result = await self._respond(
            "Choose whether to continue, replace remaining steps, finish, or stop. Respect the "
            "supplied graph step limit, reserved IDs, and remaining capacity. Use only allowlisted "
            "research steps, no investment advice, and observable concise reasons, not hidden "
            "reasoning. Do not repeat successful calls.",
            {
                "allowlisted_catalog": _safe_catalog_from_state(state),
                **_safe_replanning_progress(state),
                "typed_errors": _safe_error_summaries(state.get("observations")),
                "successful_observations": _safe_successful_observations(
                    state.get("observations")
                ),
                "public_source_references": _public_sources_from_object(
                    state.get("observations")
                ),
            },
        )
        return result


class LiveReportWriter(_LivePolicy):
    """Request a typed briefing grounded only in successful public research observations."""

    response_model = AnalystBriefing

    async def __call__(
        self,
        question: str,
        observations: tuple[ResearchObservation, ...],
    ) -> AnalystBriefing:
        result = await self._respond(
            "Write a concise factual briefing with no investment advice. Every reported fact must map "
            "to a supplied public source reference. State explicit comparison limitations for currency, "
            "reporting period, and business mix.",
            {
                "mission": question,
                "successful_observations": _safe_successful_observations(observations),
                "public_source_references": _public_sources(observations),
            },
        )
        return result


def build_live_plan_execute_policies(
    settings: Settings,
    *,
    model_factory: Callable[[Settings], Any] = create_chat_model,
) -> tuple[LivePlanner, LiveReplanner, LiveReportWriter]:
    """Build live adapters that share contracts but defer provider construction until use."""
    return (
        LivePlanner(settings, model_factory=model_factory),
        LiveReplanner(settings, model_factory=model_factory),
        LiveReportWriter(settings, model_factory=model_factory),
    )


def _safe_catalog(catalog: Sequence[PlannerToolSpec]) -> list[dict[str, Any]]:
    return [
        {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}
        for tool in catalog
    ]


def _safe_catalog_from_state(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    catalog = state.get("catalog")
    if isinstance(catalog, Sequence) and not isinstance(catalog, (str, bytes)):
        tools = tuple(tool for tool in catalog if isinstance(tool, PlannerToolSpec))
        if len(tools) == len(catalog):
            return _safe_catalog(tools)
    return []


def _safe_replanning_progress(state: Mapping[str, Any]) -> dict[str, int | list[int]]:
    reserved = state.get("all_step_ids")
    if not isinstance(reserved, Sequence) or isinstance(reserved, (str, bytes)):
        reserved_ids: list[int] = []
    else:
        reserved_ids = [
            step_id
            for step_id in reserved
            if isinstance(step_id, int) and not isinstance(step_id, bool) and step_id >= 1
        ]
    graph_step_limit = state.get("max_steps")
    valid_limit = (
        isinstance(graph_step_limit, int)
        and not isinstance(graph_step_limit, bool)
        and 1 <= graph_step_limit <= MAX_RESEARCH_STEPS
    )
    highest_reserved_id = max(reserved_ids, default=0)
    return {
        "reserved_step_ids": reserved_ids,
        "next_replacement_step_id": highest_reserved_id + 1,
        "max_step_budget": graph_step_limit if valid_limit else 0,
        "remaining_step_capacity": max(0, graph_step_limit - len(reserved_ids))
        if valid_limit
        else 0,
    }


def _safe_error_summaries(observations: object) -> list[dict[str, Any]]:
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        return []
    return [
        {
            "step_id": observation.step_id,
            "capability": observation.capability,
            "error_code": observation.error_code or observation.status,
        }
        for observation in observations
        if isinstance(observation, ResearchObservation) and observation.status != "ok"
    ]


def _safe_successful_observations(observations: object) -> list[dict[str, Any]]:
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        return []
    return [
        {
            "step_id": observation.step_id,
            "capability": observation.capability,
            "arguments": observation.arguments,
            "result": _safe_public_result(observation),
            "evidence_ids": observation.evidence_ids,
            "public_source_references": observation.source_references,
        }
        for observation in observations
        if isinstance(observation, ResearchObservation) and observation.status == "ok"
    ]


def _safe_public_result(observation: ResearchObservation) -> dict[str, Any]:
    result = observation.result
    if not isinstance(result, Mapping):
        return {}
    if observation.capability == "get_company_metric":
        return _selected_strings_and_numbers(
            result, ("company", "metric", "value", "unit", "as_of", "source")
        )
    if observation.capability == "search_financial_documents":
        safe_result = _selected_strings_and_numbers(result, ("company", "query"))
        hits = result.get("hits")
        if isinstance(hits, Sequence) and not isinstance(hits, (str, bytes)):
            safe_result["hits"] = [
                _selected_strings_and_numbers(
                    hit,
                    ("evidence_id", "text", "document_id", "section", "period", "source"),
                )
                for hit in hits
                if isinstance(hit, Mapping)
            ]
        return safe_result
    return {}


def _selected_strings_and_numbers(
    values: Mapping[str, Any], fields: tuple[str, ...]
) -> dict[str, Any]:
    return {
        field: value
        for field in fields
        if isinstance((value := values.get(field)), (str, int, float))
        and not isinstance(value, bool)
    }


def _public_sources(observations: Sequence[ResearchObservation]) -> list[str]:
    return _public_sources_from_object(observations)


def _public_sources_from_object(observations: object) -> list[str]:
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        return []
    return list(
        dict.fromkeys(
            source
            for observation in observations
            if isinstance(observation, ResearchObservation) and observation.status == "ok"
            for source in observation.source_references
        )
    )
