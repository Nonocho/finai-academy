"""Bounded application service for the recorded Financial Analyst Copilot."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from time import perf_counter
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from finai_academy.agent_evaluation import METRIC_NAMES, canonical_call_signature
from finai_academy.capstone.models import (
    CapstoneBriefing,
    CapstoneEvidenceHit,
    CitedFact,
    DeterministicEvaluation,
    EvidenceGateDecision,
    JudgeEvaluation,
    MetricEvaluation,
    PublicTraceEvent,
    ResearchRequest,
    ResearchRunResult,
    RunStatus,
    _clean_public_value,
)
from finai_academy.capstone.tools import (
    MANDATORY_ANALYST_TOOLS,
    AnalystToolRegistry,
    CertifiedRetriever,
    ToolOutcome,
    build_certified_retriever,
)
from finai_academy.financial_mcp_capabilities import (
    CapabilityValidationError,
    MetricResult,
)
from finai_academy.research_planning import (
    PlannerToolSpec,
    PlanStep,
    ResearchObservation,
    ResearchPlan,
    validate_plan,
)
from finai_academy.settings import Settings

if TYPE_CHECKING:
    from finai_academy.capstone.model_gateway import StructuredModel

_COMPANIES = ("NVIDIA", "Schneider Electric")

_INITIAL_PLAN = (
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
        arguments={"company": "NVIDIA", "query": "operating growth", "top_k": 2},
        purpose="Collect NVIDIA operating-growth document evidence.",
        expected_evidence=("NVIDIA operating growth",),
        depends_on=(1,),
    ),
    PlanStep(
        step_id=5,
        capability="search_financial_documents",
        arguments={
            "company": "Schneider Electric",
            "query": "operating growth",
            "top_k": 2,
        },
        purpose="Collect Schneider Electric operating-growth document evidence.",
        expected_evidence=("Schneider Electric operating growth",),
        depends_on=(2,),
    ),
)

_REPLACEMENT_TAIL = (
    PlanStep(
        step_id=4,
        capability="search_financial_documents",
        arguments={"company": "NVIDIA", "query": "revenue growth", "top_k": 2},
        purpose="Replace the unsupported metric with NVIDIA document evidence.",
        expected_evidence=("NVIDIA revenue growth",),
        depends_on=(1,),
    ),
    _INITIAL_PLAN[4],
)

_EXPECTED_CALLS = (
    ("get_company_metric", {"ticker": "NVDA", "metric": "P/E"}),
    ("get_company_metric", {"ticker": "SU.PA", "metric": "P/E"}),
    ("get_company_metric", {"ticker": "NVDA", "metric": "Revenue"}),
    (
        "search_financial_documents",
        {"company": "NVIDIA", "query": "revenue growth", "top_k": 2},
    ),
    (
        "search_financial_documents",
        {"company": "Schneider Electric", "query": "operating growth", "top_k": 2},
    ),
)

_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "get_company_metric": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string"},
            "metric": {"type": "string"},
        },
        "required": ["ticker", "metric"],
        "additionalProperties": False,
    },
    "search_financial_documents": {
        "type": "object",
        "properties": {
            "company": {"type": "string"},
            "query": {"type": "string"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 3},
        },
        "required": ["company", "query"],
        "additionalProperties": False,
    },
}


class _LiveReportSelection(BaseModel):
    """Provider-selected host statement IDs; the provider authors no final prose."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    executive_summary_id: str = Field(min_length=1)
    cross_company_observation_ids: tuple[str, ...] = Field(min_length=1)
    interpretation_ids: tuple[str, ...] = Field(min_length=1)
    limitation_ids: tuple[str, ...] = Field(min_length=1)


class Retriever(Protocol):
    """The company-bounded retrieval operation used by the service."""

    def search(
        self, company: str, query: str, top_k: int = 2
    ) -> tuple[CapstoneEvidenceHit, ...]: ...


class ToolRegistry(Protocol):
    """The fail-closed tool discovery and invocation boundary."""

    def discover(self) -> tuple[str, ...]: ...

    def invoke(self, name: str, arguments: Mapping[str, Any]) -> ToolOutcome: ...


class FinancialAnalystCopilot:
    """Run one bounded, fully public research trajectory."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        registry: ToolRegistry,
        run_id_factory: Callable[[], str] | None = None,
        clock: Callable[[], float] | None = None,
        initial_plan: Sequence[PlanStep] = _INITIAL_PLAN,
        replacement_tail: Sequence[PlanStep] = _REPLACEMENT_TAIL,
        structured_model: StructuredModel | None = None,
        provider_available: bool = True,
    ) -> None:
        self.retriever = retriever
        self._registry = registry
        self._run_id_factory = run_id_factory or (lambda: str(uuid4()))
        self._clock = clock or perf_counter
        self._initial_plan = tuple(initial_plan)
        self._replacement_tail = tuple(replacement_tail)
        self._structured_model = structured_model
        self._provider_available = provider_available

    def run(self, request: ResearchRequest) -> ResearchRunResult:
        """Execute host-controlled research with an optional wording provider."""

        run_started = self._clock()
        trajectory: list[PublicTraceEvent] = []
        observations: list[ResearchObservation] = []
        initial_plan: tuple[PlanStep, ...] = self._initial_plan
        final_plan: tuple[PlanStep, ...] = initial_plan
        replan_count = 0

        self._event(
            trajectory,
            phase="planning",
            status="ok",
            summary=f"Prepared {len(initial_plan)} bounded research steps.",
        )

        if request.data_mode != "certified":
            return self._result(
                request=request,
                status=RunStatus.PROVIDER_ERROR,
                initial_plan=(),
                final_plan=(),
                observations=(),
                trajectory=self._blocked_event(
                    trajectory,
                    summary="certified_data_required",
                    owner="provider",
                ),
                replan_count=0,
                run_started=run_started,
            )

        if request.provider != "recorded" and (
            not self._provider_available or self._structured_model is None
        ):
            return self._result(
                request=request,
                status=RunStatus.PROVIDER_ERROR,
                initial_plan=(),
                final_plan=(),
                observations=(),
                trajectory=self._provider_error_event(trajectory),
                replan_count=0,
                run_started=run_started,
            )

        if request.companies != _COMPANIES:
            return self._result(
                request=request,
                status=RunStatus.PLAN_BLOCKED,
                initial_plan=(),
                final_plan=(),
                observations=(),
                trajectory=self._blocked_event(
                    trajectory,
                    summary="company_universe_not_permitted",
                    owner="planner",
                ),
                replan_count=0,
                run_started=run_started,
            )

        catalog = self._catalog()
        if catalog is None or not self._plan_is_valid(initial_plan, catalog, request.max_steps):
            return self._result(
                request=request,
                status=RunStatus.PLAN_BLOCKED,
                initial_plan=(),
                final_plan=(),
                observations=(),
                trajectory=self._blocked_event(
                    trajectory,
                    summary="initial_plan_rejected",
                    owner="planner",
                ),
                replan_count=0,
                run_started=run_started,
            )

        self._event(
            trajectory,
            phase="policy",
            status="ok",
            summary="Initial research plan passed host validation.",
        )

        successful_signatures: set[str] = set()
        current_index = 0
        plan_revision = 0
        terminal_status: RunStatus | None = None

        while current_index < len(final_plan):
            step = final_plan[current_index]
            signature = canonical_call_signature(step.capability, step.arguments)
            if signature in successful_signatures:
                terminal_status = RunStatus.EXECUTION_STOPPED
                self._event(
                    trajectory,
                    phase="guardrail",
                    status="blocked",
                    summary="duplicate_successful_call",
                    capability=step.capability,
                    step_id=step.step_id,
                    plan_revision=plan_revision,
                    failure_owner="replanner",
                )
                break
            if len(observations) >= request.max_steps:
                terminal_status = RunStatus.EXECUTION_STOPPED
                self._event(
                    trajectory,
                    phase="guardrail",
                    status="blocked",
                    summary="step_budget_exhausted",
                    capability=step.capability,
                    step_id=step.step_id,
                    plan_revision=plan_revision,
                    failure_owner="replanner",
                )
                break

            observation = self._execute(
                step,
                attempt_id=len(observations) + 1,
                plan_revision=plan_revision,
            )
            observations.append(observation)
            self._event(
                trajectory,
                phase="execution",
                status=observation.status,
                summary=(
                    f"{step.capability} completed."
                    if observation.status == "ok"
                    else f"{step.capability} returned {observation.error_code}."
                ),
                capability=step.capability,
                step_id=step.step_id,
                attempt_id=observation.attempt_id,
                plan_revision=plan_revision,
                duration_ms=observation.duration_ms,
                error_code=observation.error_code,
                failure_owner=("tool_boundary" if observation.status != "ok" else None),
            )

            if observation.status == "ok":
                successful_signatures.add(signature)
                current_index += 1
                continue

            if observation.error_code != "unsupported_metric":
                terminal_status = RunStatus.EXECUTION_STOPPED
                break
            if replan_count >= request.max_replans:
                terminal_status = RunStatus.REPLAN_BUDGET_EXHAUSTED
                self._event(
                    trajectory,
                    phase="guardrail",
                    status="blocked",
                    summary="replan_budget_exhausted",
                    capability=step.capability,
                    step_id=step.step_id,
                    attempt_id=observation.attempt_id,
                    plan_revision=plan_revision,
                    failure_owner="replanner",
                )
                break

            candidate = final_plan[: current_index + 1] + self._replacement_tail
            if not self._replacement_is_valid(
                executed_prefix=final_plan[: current_index + 1],
                candidate=candidate,
                catalog=catalog,
                max_steps=request.max_steps,
                successful_signatures=successful_signatures,
            ):
                terminal_status = RunStatus.PLAN_BLOCKED
                self._event(
                    trajectory,
                    phase="guardrail",
                    status="blocked",
                    summary="replacement_plan_rejected",
                    capability=step.capability,
                    step_id=step.step_id,
                    attempt_id=observation.attempt_id,
                    plan_revision=plan_revision,
                    failure_owner="replanner",
                )
                break

            final_plan = candidate
            current_index += 1
            replan_count += 1
            plan_revision += 1
            self._event(
                trajectory,
                phase="replanning",
                status="ok",
                summary="Replaced the remaining tail with document search.",
                capability=step.capability,
                step_id=step.step_id,
                attempt_id=observation.attempt_id,
                plan_revision=plan_revision,
            )

        if terminal_status is not None:
            return self._result(
                request=request,
                status=terminal_status,
                initial_plan=initial_plan,
                final_plan=final_plan,
                observations=tuple(observations),
                trajectory=tuple(trajectory),
                replan_count=replan_count,
                run_started=run_started,
            )

        evidence_gate = _evaluate_evidence(tuple(observations))
        self._event(
            trajectory,
            phase="evidence_gate",
            status="ok" if evidence_gate.passed else "blocked",
            summary=(
                "Both companies have source-addressable document evidence."
                if evidence_gate.passed
                else "insufficient_evidence"
            ),
            plan_revision=plan_revision,
            failure_owner=None if evidence_gate.passed else "evidence_gate",
        )
        if not evidence_gate.passed:
            return self._result(
                request=request,
                status=RunStatus.INSUFFICIENT_EVIDENCE,
                initial_plan=initial_plan,
                final_plan=final_plan,
                observations=tuple(observations),
                trajectory=tuple(trajectory),
                replan_count=replan_count,
                run_started=run_started,
                evidence_gate=evidence_gate,
            )

        briefing = _build_briefing(tuple(observations), evidence_gate)
        if request.provider != "recorded":
            try:
                briefing = self._apply_live_wording(request, briefing)
            except Exception:  # noqa: BLE001 - provider details must never become public
                return self._result(
                    request=request,
                    status=RunStatus.PROVIDER_ERROR,
                    initial_plan=initial_plan,
                    final_plan=final_plan,
                    observations=tuple(observations),
                    trajectory=self._provider_error_event(trajectory),
                    replan_count=replan_count,
                    run_started=run_started,
                    evidence_gate=evidence_gate,
                )
        self._event(
            trajectory,
            phase="report",
            status="ok",
            summary="Created a briefing from verified public evidence.",
            plan_revision=plan_revision,
        )
        return self._result(
            request=request,
            status=RunStatus.COMPLETED,
            initial_plan=initial_plan,
            final_plan=final_plan,
            observations=tuple(observations),
            trajectory=tuple(trajectory),
            replan_count=replan_count,
            run_started=run_started,
            evidence_gate=evidence_gate,
            briefing=briefing,
        )

    def _apply_live_wording(
        self,
        request: ResearchRequest,
        briefing: CapstoneBriefing,
    ) -> CapstoneBriefing:
        """Reconstruct final prose exclusively from host-certified statement units."""

        if self._structured_model is None:
            raise RuntimeError("live provider is unavailable")
        statement_units = _certified_statement_units(briefing)
        prompt_payload = {
            "mission": request.question,
            "certified_cited_facts": [
                fact.model_dump(mode="json") for fact in briefing.cited_facts
            ],
            "certified_statement_units": [
                {"id": statement_id, "section": section, "text": text}
                for section, units in statement_units.items()
                for statement_id, text in units.items()
            ],
        }
        proposal = self._structured_model.generate(
            system_prompt=(
                "Select and order only the supplied certified statement IDs. Do not write or "
                "rewrite prose, add facts, add recommendations, or change citations. Return "
                "every supplied ID exactly once in its matching section."
            ),
            user_prompt=json.dumps(prompt_payload, sort_keys=True),
            response_model=_LiveReportSelection,
        )
        selection = _LiveReportSelection.model_validate(proposal)
        _clean_public_value(selection)
        executive_summary = _selected_statements(
            statement_units["executive_summary"],
            (selection.executive_summary_id,),
        )
        cross_company_observations = _selected_statements(
            statement_units["cross_company_observation"],
            selection.cross_company_observation_ids,
        )
        interpretation = _selected_statements(
            statement_units["interpretation"],
            selection.interpretation_ids,
        )
        limitations = _selected_statements(
            statement_units["limitation"],
            selection.limitation_ids,
        )
        return CapstoneBriefing.model_validate(
            {
                **briefing.model_dump(mode="python"),
                "executive_summary": executive_summary[0],
                "cross_company_observations": cross_company_observations,
                "interpretation": interpretation,
                "limitations": limitations,
            }
        )

    def _catalog(self) -> tuple[PlannerToolSpec, ...] | None:
        try:
            discovered = self._registry.discover()
        except Exception:  # noqa: BLE001 - injected discovery must fail closed
            return None
        if not isinstance(discovered, Sequence) or isinstance(discovered, (str, bytes)):
            return None
        if any(not isinstance(name, str) for name in discovered):
            return None
        return tuple(
            PlannerToolSpec(
                name=name,
                description=f"Certified read-only {name} capability.",
                input_schema=_TOOL_SCHEMAS[name],
            )
            for name in discovered
            if name in MANDATORY_ANALYST_TOOLS
        )

    @staticmethod
    def _plan_is_valid(
        steps: tuple[PlanStep, ...],
        catalog: tuple[PlannerToolSpec, ...],
        max_steps: int,
    ) -> bool:
        try:
            _clean_public_value(tuple(step.model_dump(mode="python") for step in steps))
            validate_plan(
                ResearchPlan(goal="Bounded two-company research.", steps=steps),
                catalog,
                max_steps,
            )
        except (TypeError, ValueError):
            return False
        return True

    def _replacement_is_valid(
        self,
        *,
        executed_prefix: tuple[PlanStep, ...],
        candidate: tuple[PlanStep, ...],
        catalog: tuple[PlannerToolSpec, ...],
        max_steps: int,
        successful_signatures: set[str],
    ) -> bool:
        if candidate[: len(executed_prefix)] != executed_prefix:
            return False
        if not self._plan_is_valid(candidate, catalog, max_steps):
            return False
        try:
            replacement_signatures = {
                canonical_call_signature(step.capability, step.arguments)
                for step in candidate[len(executed_prefix) :]
            }
        except (TypeError, ValueError):
            return False
        return not (replacement_signatures & successful_signatures)

    def _execute(
        self,
        step: PlanStep,
        *,
        attempt_id: int,
        plan_revision: int,
    ) -> ResearchObservation:
        started = self._clock()
        try:
            if step.capability == "get_company_metric":
                observation = self._execute_metric(step, attempt_id, plan_revision)
            elif step.capability == "search_financial_documents":
                observation = self._execute_documents(step, attempt_id, plan_revision)
            else:
                observation = _error_observation(
                    step,
                    attempt_id,
                    plan_revision,
                    error_code="unknown_capability",
                )
        except CapabilityValidationError as error:
            observation = _error_observation(
                step,
                attempt_id,
                plan_revision,
                error_code=error.error.error_code,
            )
        except Exception:  # noqa: BLE001 - tool and injected doubles must fail closed
            observation = _error_observation(
                step,
                attempt_id,
                plan_revision,
                error_code="tool_boundary_error",
            )
        duration_ms = max(0.0, (self._clock() - started) * 1000)
        return observation.model_copy(update={"duration_ms": duration_ms})

    def _execute_metric(
        self, step: PlanStep, attempt_id: int, plan_revision: int
    ) -> ResearchObservation:
        outcome = self._registry.invoke(step.capability, step.arguments)
        if not isinstance(outcome, ToolOutcome):
            return _error_observation(
                step,
                attempt_id,
                plan_revision,
                error_code="malformed_tool_outcome",
            )
        if outcome.status == "error":
            return _error_observation(
                step,
                attempt_id,
                plan_revision,
                error_code=outcome.error_code or "malformed_tool_outcome",
            )
        if not isinstance(outcome.payload, MetricResult):
            return _error_observation(
                step,
                attempt_id,
                plan_revision,
                error_code="malformed_tool_outcome",
            )
        payload = outcome.payload.model_dump(mode="python")
        return ResearchObservation(
            attempt_id=attempt_id,
            step_id=step.step_id,
            plan_revision=plan_revision,
            capability=step.capability,
            arguments=dict(step.arguments),
            status="ok",
            result=payload,
            source_references=(outcome.payload.source,),
            duration_ms=0,
        )

    def _execute_documents(
        self, step: PlanStep, attempt_id: int, plan_revision: int
    ) -> ResearchObservation:
        company = step.arguments.get("company")
        query = step.arguments.get("query")
        top_k = step.arguments.get("top_k", 2)
        if (
            not isinstance(company, str)
            or not isinstance(query, str)
            or not isinstance(top_k, int)
            or isinstance(top_k, bool)
        ):
            return _error_observation(
                step,
                attempt_id,
                plan_revision,
                error_code="invalid_arguments",
            )
        hits = self.retriever.search(company, query, top_k)
        if not isinstance(hits, Sequence) or isinstance(hits, (str, bytes)):
            return _error_observation(
                step,
                attempt_id,
                plan_revision,
                error_code="malformed_tool_outcome",
            )
        if any(not isinstance(hit, CapstoneEvidenceHit) or hit.company != company for hit in hits):
            return _error_observation(
                step,
                attempt_id,
                plan_revision,
                error_code="malformed_tool_outcome",
            )
        public_hits = tuple(hits)
        result_hits = tuple(
            {
                "evidence_id": hit.evidence_id,
                "text": hit.text,
                "document_id": hit.document_id,
                "section": hit.section,
                "period": hit.period,
                "source": hit.source_reference,
            }
            for hit in public_hits
        )
        return ResearchObservation(
            attempt_id=attempt_id,
            step_id=step.step_id,
            plan_revision=plan_revision,
            capability=step.capability,
            arguments=dict(step.arguments),
            status="ok",
            result={"company": company, "query": query, "hits": result_hits},
            evidence_ids=tuple(hit.evidence_id for hit in public_hits),
            source_references=tuple(hit.source_reference for hit in public_hits),
            duration_ms=0,
        )

    def _result(
        self,
        *,
        request: ResearchRequest,
        status: RunStatus,
        initial_plan: tuple[PlanStep, ...],
        final_plan: tuple[PlanStep, ...],
        observations: tuple[ResearchObservation, ...],
        trajectory: tuple[PublicTraceEvent, ...],
        replan_count: int,
        run_started: float,
        evidence_gate: EvidenceGateDecision | None = None,
        briefing: CapstoneBriefing | None = None,
    ) -> ResearchRunResult:
        gate = evidence_gate or _evaluate_evidence(observations)
        evaluation = _evaluate_run(
            request=request,
            status=status,
            observations=observations,
            replan_count=replan_count,
            evidence_gate=gate,
            briefing=briefing,
        )
        return ResearchRunResult(
            run_id=self._run_id_factory(),
            request=request,
            provider=request.provider,
            model=request.model,
            data_mode=request.data_mode,
            status=status,
            initial_plan=initial_plan,
            final_plan=final_plan,
            observations=observations,
            trajectory=trajectory,
            evidence_gate=gate,
            briefing=briefing,
            deterministic_evaluation=evaluation,
            judge_evaluation=JudgeEvaluation(),
            mlflow_run_id=None,
            mlflow_trace_id=None,
            replan_count=replan_count,
            total_duration_ms=max(0.0, (self._clock() - run_started) * 1000),
        )

    @staticmethod
    def _event(
        trajectory: list[PublicTraceEvent],
        *,
        phase: str,
        status: str,
        summary: str,
        capability: str | None = None,
        step_id: int | None = None,
        attempt_id: int | None = None,
        plan_revision: int = 0,
        duration_ms: float = 0,
        error_code: str | None = None,
        failure_owner: str | None = None,
    ) -> None:
        trajectory.append(
            PublicTraceEvent(
                index=len(trajectory) + 1,
                phase=phase,
                status=status,
                summary=summary,
                capability=capability,
                step_id=step_id,
                attempt_id=attempt_id,
                plan_revision=plan_revision,
                duration_ms=duration_ms,
                error_code=error_code,
                failure_owner=failure_owner,
            )
        )

    def _blocked_event(
        self,
        trajectory: list[PublicTraceEvent],
        *,
        summary: str,
        owner: str,
    ) -> tuple[PublicTraceEvent, ...]:
        self._event(
            trajectory,
            phase="guardrail",
            status="blocked",
            summary=summary,
            failure_owner=owner,
        )
        return tuple(trajectory)

    def _provider_error_event(
        self,
        trajectory: list[PublicTraceEvent],
    ) -> tuple[PublicTraceEvent, ...]:
        self._event(
            trajectory,
            phase="guardrail",
            status="error",
            summary="The selected provider could not complete structured generation.",
            failure_owner="provider",
        )
        return tuple(trajectory)


def _error_observation(
    step: PlanStep,
    attempt_id: int,
    plan_revision: int,
    *,
    error_code: str,
) -> ResearchObservation:
    return ResearchObservation(
        attempt_id=attempt_id,
        step_id=step.step_id,
        plan_revision=plan_revision,
        capability=step.capability,
        arguments=dict(step.arguments),
        status="error",
        error_code=error_code,
        duration_ms=0,
    )


def _certified_statement_units(
    briefing: CapstoneBriefing,
) -> dict[str, dict[str, str]]:
    """Assign stable section-scoped IDs to host-authored public prose."""

    return {
        "executive_summary": {"executive_summary:1": briefing.executive_summary},
        "cross_company_observation": {
            f"cross_company_observation:{index}": statement
            for index, statement in enumerate(briefing.cross_company_observations, start=1)
        },
        "interpretation": {
            f"interpretation:{index}": statement
            for index, statement in enumerate(briefing.interpretation, start=1)
        },
        "limitation": {
            f"limitation:{index}": statement
            for index, statement in enumerate(briefing.limitations, start=1)
        },
    }


def _selected_statements(
    available: Mapping[str, str],
    selected_ids: Sequence[str],
) -> tuple[str, ...]:
    """Require one permutation of all certified IDs and reconstruct their prose."""

    if len(selected_ids) != len(available) or set(selected_ids) != set(available):
        raise ValueError("provider selection must contain every certified statement ID once")
    return tuple(available[statement_id] for statement_id in selected_ids)


def _document_hits(
    observations: Sequence[ResearchObservation],
) -> tuple[CapstoneEvidenceHit, ...]:
    hits: list[CapstoneEvidenceHit] = []
    for observation in observations:
        if (
            observation.status != "ok"
            or observation.capability != "search_financial_documents"
            or observation.result is None
        ):
            continue
        company = observation.result.get("company")
        raw_hits = observation.result.get("hits")
        if (
            not isinstance(company, str)
            or not isinstance(raw_hits, Sequence)
            or isinstance(raw_hits, (str, bytes))
        ):
            continue
        for raw_hit in raw_hits:
            if not isinstance(raw_hit, Mapping):
                continue
            try:
                hit = CapstoneEvidenceHit(
                    company=company,
                    text=raw_hit.get("text"),
                    evidence_id=raw_hit.get("evidence_id"),
                    document_id=raw_hit.get("document_id"),
                    section=raw_hit.get("section"),
                    period=raw_hit.get("period"),
                    source_reference=raw_hit.get("source"),
                )
            except (TypeError, ValueError):
                continue
            if (
                hit.evidence_id in observation.evidence_ids
                and hit.source_reference in observation.source_references
            ):
                hits.append(hit)
    return tuple(hits)


def _evaluate_evidence(observations: Sequence[ResearchObservation]) -> EvidenceGateDecision:
    hits = _document_hits(observations)
    coverage: dict[str, tuple[str, ...]] = {}
    for company in _COMPANIES:
        kinds: list[str] = []
        if any(hit.company == company for hit in hits):
            kinds.append("document")
        if any(
            observation.status == "ok"
            and observation.capability == "get_company_metric"
            and observation.result is not None
            and observation.result.get("company") == company
            and bool(observation.source_references)
            for observation in observations
        ):
            kinds.append("metric")
        coverage[company] = tuple(kinds)
    missing = tuple(
        f"{company} document evidence"
        for company in _COMPANIES
        if "document" not in coverage[company]
    )
    return EvidenceGateDecision(
        passed=not missing,
        coverage=coverage,
        missing_requirements=missing,
        evidence_hits=hits,
    )


def _build_briefing(
    observations: Sequence[ResearchObservation],
    evidence_gate: EvidenceGateDecision,
) -> CapstoneBriefing:
    facts: list[CitedFact] = []
    for observation in observations:
        if observation.status != "ok" or observation.result is None:
            continue
        result = observation.result
        if observation.capability == "get_company_metric" and observation.source_references:
            company = result.get("company")
            metric = result.get("metric")
            value = result.get("value")
            unit = result.get("unit")
            as_of = result.get("as_of")
            if (
                isinstance(company, str)
                and isinstance(metric, str)
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and isinstance(unit, str)
                and isinstance(as_of, str)
            ):
                facts.append(
                    CitedFact(
                        claim=f"{company} {metric} was {value:g} {unit} as of {as_of}.",
                        company=company,
                        provenance_kind="metric",
                        source_reference=observation.source_references[0],
                    )
                )
    for hit in evidence_gate.evidence_hits:
        facts.append(
            CitedFact(
                claim=f"{hit.company} ({hit.period}): {hit.text}",
                company=hit.company,
                provenance_kind="document",
                source_reference=hit.source_reference,
                evidence_id=hit.evidence_id,
            )
        )
    sources = tuple(dict.fromkeys(fact.source_reference for fact in facts))
    return CapstoneBriefing(
        executive_summary="Evidence-backed comparison prepared for the bounded research request.",
        cited_facts=tuple(facts),
        company_evidence={
            company: tuple(hit for hit in evidence_gate.evidence_hits if hit.company == company)
            for company in _COMPANIES
        },
        cross_company_observations=(
            "Direct comparability is limited by different reporting periods, currencies, and business mixes.",
        ),
        interpretation=(
            "The verified evidence supports a qualified operating-growth comparison, not a like-for-like ranking.",
        ),
        limitations=(
            "The companies report in different currencies and periods.",
            "Their business mixes and disclosed operating measures are not directly comparable.",
        ),
        open_questions=(
            "Which aligned operating measure would be most useful for a later comparison?",
        ),
        aggregate_sources=sources,
    )


def _evaluate_run(
    *,
    request: ResearchRequest,
    status: RunStatus,
    observations: Sequence[ResearchObservation],
    replan_count: int,
    evidence_gate: EvidenceGateDecision,
    briefing: CapstoneBriefing | None,
) -> DeterministicEvaluation:
    observed_signatures = tuple(
        canonical_call_signature(item.capability, item.arguments) for item in observations
    )
    expected_signatures = tuple(
        canonical_call_signature(capability, arguments) for capability, arguments in _EXPECTED_CALLS
    )
    errors = tuple(item.error_code for item in observations if item.status == "error")
    correctness = (
        observed_signatures == expected_signatures
        and errors == ("unsupported_metric",)
        and replan_count == 1
    )
    successful_signatures = tuple(
        canonical_call_signature(item.capability, item.arguments)
        for item in observations
        if item.status == "ok"
    )
    efficiency = (
        len(observations) <= request.max_steps
        and replan_count <= request.max_replans
        and len(successful_signatures) == len(set(successful_signatures))
    )
    relevance = (
        status == RunStatus.COMPLETED
        and briefing is not None
        and {fact.company for fact in briefing.cited_facts} == set(_COMPANIES)
    )
    completeness = bool(
        briefing is not None
        and briefing.executive_summary
        and briefing.cross_company_observations
        and briefing.interpretation
        and briefing.limitations
        and all(briefing.company_evidence.get(company) for company in _COMPANIES)
    )
    citation_integrity = _citations_are_exact(observations, evidence_gate, briefing)
    values = (correctness, efficiency, relevance, completeness, citation_integrity)
    rationales = (
        "Expected call order, typed recovery, and replan count match.",
        "The run stays within both budgets and repeats no successful call.",
        "The briefing addresses the fixed two-company research universe.",
        "Both company sections, comparison, interpretation, and limitations are present.",
        "Every cited source and document evidence pair matches a collected observation.",
    )
    metrics = tuple(
        MetricEvaluation(name=name, value=float(value), rationale=rationale)
        for name, value, rationale in zip(METRIC_NAMES, values, rationales, strict=True)
    )
    return DeterministicEvaluation(
        metrics=metrics,
        release_passed=all(metric.value == 1.0 for metric in metrics),
    )


def _citations_are_exact(
    observations: Sequence[ResearchObservation],
    evidence_gate: EvidenceGateDecision,
    briefing: CapstoneBriefing | None,
) -> bool:
    if briefing is None or not evidence_gate.passed:
        return False
    metric_pairs = {
        (observation.result.get("company"), source)
        for observation in observations
        if observation.status == "ok"
        and observation.capability == "get_company_metric"
        and observation.result is not None
        for source in observation.source_references
    }
    document_pairs = {
        (hit.company, hit.source_reference, hit.evidence_id) for hit in evidence_gate.evidence_hits
    }
    for fact in briefing.cited_facts:
        if fact.provenance_kind == "metric":
            if (fact.company, fact.source_reference) not in metric_pairs:
                return False
        elif (
            fact.company,
            fact.source_reference,
            fact.evidence_id,
        ) not in document_pairs:
            return False
    expected_sources = tuple(dict.fromkeys(fact.source_reference for fact in briefing.cited_facts))
    return bool(briefing.cited_facts) and briefing.aggregate_sources == expected_sources


def build_reference_copilot(
    *,
    retriever: CertifiedRetriever | None = None,
    registry: AnalystToolRegistry | None = None,
    run_id_factory: Callable[[], str] | None = None,
    clock: Callable[[], float] | None = None,
) -> FinancialAnalystCopilot:
    """Compose the fully offline recorded reference service."""

    return FinancialAnalystCopilot(
        retriever=retriever or build_certified_retriever(),
        registry=registry or AnalystToolRegistry(discovered=tuple(MANDATORY_ANALYST_TOOLS)),
        run_id_factory=run_id_factory,
        clock=clock,
    )


def build_copilot_for_request(
    request: ResearchRequest,
    settings: Settings,
    *,
    ollama_probe: Callable[[], bool] | None = None,
) -> FinancialAnalystCopilot:
    """Compose exactly the requested provider route with no silent fallback."""

    if request.provider == "recorded":
        return build_reference_copilot()

    from finai_academy.capstone import model_gateway

    readiness = model_gateway.provider_readiness(
        request.provider,
        request.model,
        settings=settings,
        ollama_probe=ollama_probe,
    )
    if not readiness.available:
        return FinancialAnalystCopilot(
            retriever=build_certified_retriever(),
            registry=AnalystToolRegistry(discovered=tuple(MANDATORY_ANALYST_TOOLS)),
            provider_available=False,
        )

    routed_settings = Settings(
        provider=request.provider,
        chat_model=request.model,
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model,
        ollama_base_url=settings.ollama_base_url,
    )
    try:
        structured_model = model_gateway.create_structured_model(routed_settings)
    except Exception:  # noqa: BLE001 - construction failures become public typed results
        structured_model = None
    return FinancialAnalystCopilot(
        retriever=build_certified_retriever(),
        registry=AnalystToolRegistry(discovered=tuple(MANDATORY_ANALYST_TOOLS)),
        structured_model=structured_model,
        provider_available=structured_model is not None,
    )
