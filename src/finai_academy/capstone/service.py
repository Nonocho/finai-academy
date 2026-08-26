"""Bounded application service for the recorded Financial Analyst Copilot."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from finai_academy.agent_evaluation import METRIC_NAMES, canonical_call_signature
from finai_academy.capstone.document_models import FinancialChunk
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
    DocumentEvidenceOutcome,
    ReportedValue,
    ReportedValueComparison,
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
_QuestionIntent = Literal["reference", "operating_growth", "valuation", "revenue_growth"]

_DOCUMENT_PLAN = (
    PlanStep(step_id=1, capability="search_financial_documents", arguments={"company": "NVIDIA", "reporting_period": "FY2026", "query": "reported segment revenue", "element_type": "table", "top_k": 1}, purpose="Find NVIDIA's reported revenue table.", expected_evidence=("NVIDIA reported segment revenue table",)),
    PlanStep(step_id=2, capability="inspect_document_evidence", arguments={"chunk_id": "selected:NVIDIA"}, purpose="Inspect the selected NVIDIA table and its page context.", expected_evidence=("NVIDIA inspected document evidence",), depends_on=(1,)),
    PlanStep(step_id=3, capability="search_financial_documents", arguments={"company": "Schneider Electric", "reporting_period": "FY2025", "query": "reported revenue organic growth", "element_type": "table", "top_k": 1}, purpose="Find Schneider Electric's reported revenue table.", expected_evidence=("Schneider Electric reported revenue table",)),
    PlanStep(step_id=4, capability="inspect_document_evidence", arguments={"chunk_id": "selected:Schneider Electric"}, purpose="Inspect the selected Schneider Electric table and its page context.", expected_evidence=("Schneider Electric inspected document evidence",), depends_on=(3,)),
    PlanStep(step_id=5, capability="compare_reported_values", arguments={"left": "selected:NVIDIA", "right": "selected:Schneider Electric"}, purpose="Compare only displayed cited values and preserve non-comparability.", expected_evidence=("deterministic comparison limits",), depends_on=(2, 4)),
)

_REPLACEMENT_TAIL: tuple[PlanStep, ...] = ()
_EXPECTED_CALLS = tuple((step.capability, dict(step.arguments)) for step in _DOCUMENT_PLAN)

_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "search_financial_documents": {"type": "object", "properties": {"company": {"type": "string"}, "reporting_period": {"type": "string"}, "query": {"type": "string"}, "element_type": {"type": "string"}, "top_k": {"type": "integer", "minimum": 1, "maximum": 5}}, "required": ["company", "reporting_period", "query", "element_type", "top_k"], "additionalProperties": False},
    "inspect_document_evidence": {"type": "object", "properties": {"chunk_id": {"type": "string"}}, "required": ["chunk_id"], "additionalProperties": False},
    "compare_reported_values": {"type": "object", "properties": {"left": {"type": "string"}, "right": {"type": "string"}}, "required": ["left", "right"], "additionalProperties": False},
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
        initial_plan: Sequence[PlanStep] = _DOCUMENT_PLAN,
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
        self._selected_hits: dict[str, CapstoneEvidenceHit] = {}
        self._inspected_chunks: dict[str, FinancialChunk] = {}

    def run(self, request: ResearchRequest) -> ResearchRunResult:
        """Execute host-controlled research with an optional wording provider."""

        run_started = self._clock()
        self._selected_hits = {}
        self._inspected_chunks = {}
        trajectory: list[PublicTraceEvent] = []
        observations: list[ResearchObservation] = []
        question_intent = _classify_question_intent(request)
        initial_plan: tuple[PlanStep, ...] = self._plan_for_intent(question_intent)
        final_plan: tuple[PlanStep, ...] = initial_plan
        replacement_tail = self._replacement_for_intent(question_intent)
        replan_count = 0

        self._event(
            trajectory,
            phase="planning",
            status="ok",
            summary=f"Prepared {len(initial_plan)} bounded research steps.",
        )

        if question_intent is None:
            return self._result(
                request=request,
                status=RunStatus.PLAN_BLOCKED,
                initial_plan=(),
                final_plan=(),
                observations=(),
                trajectory=self._blocked_event(
                    trajectory,
                    summary="unsupported_question",
                    owner="planner",
                ),
                replan_count=0,
                run_started=run_started,
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

            if observation.error_code not in {
                "missing_contextual_table",
                "missing_evidence_metadata",
            }:
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

            candidate = final_plan[: current_index + 1] + replacement_tail
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
                summary="Removed the remaining document steps after missing contextual evidence.",
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

        briefing = _build_briefing(
            tuple(observations),
            evidence_gate,
            question_intent=question_intent,
        )
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

    def _plan_for_intent(
        self, question_intent: _QuestionIntent | None
    ) -> tuple[PlanStep, ...]:
        if question_intent in {None, "reference"}:
            return self._initial_plan
        label = _intent_label(question_intent)
        return tuple(
            step.model_copy(
                update={
                    "purpose": f"Use {step.expected_evidence[0]} for the submitted {label} question."
                }
            )
            for step in self._initial_plan
        )

    def _replacement_for_intent(
        self, question_intent: _QuestionIntent | None
    ) -> tuple[PlanStep, ...]:
        if question_intent in {None, "reference"}:
            return self._replacement_tail
        label = _intent_label(question_intent)
        return tuple(
            step.model_copy(
                update={
                    "purpose": f"Use {step.expected_evidence[0]} for the submitted {label} question."
                }
            )
            for step in self._replacement_tail
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
            if step.capability == "search_financial_documents":
                observation = self._execute_documents(step, attempt_id, plan_revision)
            elif step.capability == "inspect_document_evidence":
                observation = self._execute_inspection(step, attempt_id, plan_revision)
            elif step.capability == "compare_reported_values":
                observation = self._execute_comparison(step, attempt_id, plan_revision)
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
        reporting_period = step.arguments.get("reporting_period")
        query = step.arguments.get("query")
        element_type = step.arguments.get("element_type")
        top_k = step.arguments.get("top_k", 2)
        if (
            not isinstance(company, str)
            or not isinstance(reporting_period, str)
            or not isinstance(query, str)
            or element_type != "table"
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
        if any(
            not isinstance(hit, CapstoneEvidenceHit)
            or hit.company != company
            or hit.period != reporting_period
            for hit in hits
        ):
            return _error_observation(
                step,
                attempt_id,
                plan_revision,
                error_code="malformed_tool_outcome",
            )
        public_hits = tuple(hits)
        if not public_hits:
            return _error_observation(
                step, attempt_id, plan_revision, error_code="missing_contextual_table"
            )
        selected = public_hits[0]
        if (
            selected.element_type != "table"
            or selected.unit is None
            or not selected.element_ids
            or selected.physical_page <= 0
        ):
            return _error_observation(
                step, attempt_id, plan_revision, error_code="missing_evidence_metadata"
            )
        self._selected_hits[company] = selected
        result_hits = tuple(
            {
                "chunk_id": hit.chunk_id,
                "selection_reason": hit.selection_reason,
                "channel_ranks": hit.channel_ranks,
                "fused_score": hit.fused_score,
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
            result={
                "company": company,
                "reporting_period": reporting_period,
                "query": query,
                "element_type": element_type,
                "candidate_chunk_ids": tuple(hit.chunk_id for hit in public_hits),
                "selected_chunk_ids": (selected.chunk_id,),
                "hits": result_hits,
            },
            evidence_ids=tuple(hit.chunk_id for hit in public_hits),
            source_references=(),
            duration_ms=0,
        )

    def _execute_inspection(
        self, step: PlanStep, attempt_id: int, plan_revision: int
    ) -> ResearchObservation:
        requested = step.arguments.get("chunk_id")
        if not isinstance(requested, str) or not requested.startswith("selected:"):
            return _error_observation(step, attempt_id, plan_revision, error_code="invalid_arguments")
        company = requested.removeprefix("selected:")
        selected = self._selected_hits.get(company)
        if selected is None:
            return _error_observation(
                step, attempt_id, plan_revision, error_code="missing_contextual_table"
            )
        outcome = self._registry.invoke("inspect_document_evidence", {"chunk_id": selected.chunk_id})
        if not isinstance(outcome, ToolOutcome):
            return _error_observation(
                step, attempt_id, plan_revision, error_code="malformed_tool_outcome"
            )
        if outcome.status != "ok":
            return _error_observation(
                step, attempt_id, plan_revision, error_code="missing_evidence_metadata"
            )
        if not isinstance(outcome.payload, DocumentEvidenceOutcome):
            return _error_observation(
                step, attempt_id, plan_revision, error_code="malformed_tool_outcome"
            )
        chunk = outcome.payload.chunk
        if (
            chunk.chunk_id != selected.chunk_id
            or not chunk.source_element_ids
        ):
            return _error_observation(
                step, attempt_id, plan_revision, error_code="missing_evidence_metadata"
            )
        certified = _certified_hit_from_inspection(
            chunk,
            crop_asset_key=outcome.payload.crop_asset_key,
            selection_reason=selected.selection_reason,
            channel_ranks=selected.channel_ranks,
            fused_score=selected.fused_score,
        )
        self._selected_hits[company] = certified
        self._inspected_chunks[company] = chunk
        return ResearchObservation(
            attempt_id=attempt_id,
            step_id=step.step_id,
            plan_revision=plan_revision,
            capability=step.capability,
            arguments=dict(step.arguments),
            status="ok",
            result={
                "company": company,
                "hit": certified.model_dump(mode="json"),
            },
            evidence_ids=(certified.chunk_id,),
            source_references=(certified.source_reference,),
            duration_ms=0,
        )

    def _execute_comparison(
        self, step: PlanStep, attempt_id: int, plan_revision: int
    ) -> ResearchObservation:
        left = self._selected_hits.get("NVIDIA")
        right = self._selected_hits.get("Schneider Electric")
        left_chunk = self._inspected_chunks.get("NVIDIA")
        right_chunk = self._inspected_chunks.get("Schneider Electric")
        if left is None or right is None or left_chunk is None or right_chunk is None:
            return _error_observation(
                step, attempt_id, plan_revision, error_code="unavailable_comparison_input"
            )
        try:
            values = {
                "NVIDIA": _reported_value_from_inspected_table(left_chunk),
                "Schneider Electric": _reported_value_from_inspected_table(right_chunk),
            }
        except ValueError:
            return _error_observation(
                step, attempt_id, plan_revision, error_code="unavailable_comparison_input"
            )
        outcome = self._registry.invoke(
            "compare_reported_values", {"left": values["NVIDIA"], "right": values["Schneider Electric"]}
        )
        if not isinstance(outcome, ToolOutcome) or outcome.status != "ok" or not isinstance(
            outcome.payload, ReportedValueComparison
        ):
            return _error_observation(
                step, attempt_id, plan_revision, error_code="malformed_tool_outcome"
            )
        return ResearchObservation(
            attempt_id=attempt_id,
            step_id=step.step_id,
            plan_revision=plan_revision,
            capability=step.capability,
            arguments=dict(step.arguments),
            status="ok",
            result=outcome.payload.model_dump(mode="json"),
            evidence_ids=(left.chunk_id, right.chunk_id),
            source_references=(left.source_reference, right.source_reference),
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


def _certified_hit_from_inspection(
    chunk: FinancialChunk,
    *,
    crop_asset_key: str | None,
    selection_reason: str,
    channel_ranks: tuple[tuple[str, int], ...],
    fused_score: float,
) -> CapstoneEvidenceHit:
    """Expose only immutable inspection content while retaining search rank lineage."""

    unit = _displayed_unit(chunk)
    if unit is None:
        raise ValueError("inspected table has no displayed unit")
    return CapstoneEvidenceHit(
        company=chunk.context.company_name,
        text=chunk.text,
        chunk_id=chunk.chunk_id,
        element_ids=chunk.source_element_ids,
        document_id=chunk.context.document_id,
        document_sha256=chunk.context.document_sha256,
        section=" > ".join(chunk.context.heading_path) or chunk.element_type,
        period=chunk.context.reporting_period,
        unit=unit,
        physical_page=chunk.context.physical_page,
        printed_page=chunk.context.printed_page,
        element_type=chunk.element_type,
        bbox=chunk.context.bbox,
        source_reference=chunk.context.official_source_url,
        crop_asset_key=crop_asset_key,
        original_markdown=chunk.table.markdown if chunk.table is not None else None,
        selection_reason=selection_reason,
        channel_ranks=channel_ranks,
        fused_score=fused_score,
    )


def _reported_value_from_inspected_table(chunk: FinancialChunk) -> ReportedValue:
    """Extract one unambiguous displayed revenue input from an inspected target table."""

    table = chunk.table
    unit = _displayed_unit(chunk)
    if table is None or unit is None:
        raise ValueError("inspected evidence has no contextual table value")
    if chunk.context.company_name == "NVIDIA":
        return _nvidia_reported_value(chunk, unit)
    if chunk.context.company_name == "Schneider Electric":
        return _schneider_reported_value(chunk, unit)
    raise ValueError("inspected evidence company is not eligible for comparison")


def _nvidia_reported_value(chunk: FinancialChunk, unit: str) -> ReportedValue:
    table = chunk.table
    if table is None:
        raise ValueError("NVIDIA evidence has no table")
    headers = table.rows[0]
    revenue_rows = [row for row in table.rows if row and row[0].casefold() == "revenue"]
    if len(revenue_rows) != 3:
        raise ValueError("NVIDIA table does not contain one value per reported year")
    values = _numeric_cells(revenue_rows[0])
    if len(values) != 3:
        raise ValueError("NVIDIA reported revenue row is ambiguous")
    segment_positions = [index for index, header in enumerate(headers) if header.strip()]
    if len(segment_positions) != 3:
        raise ValueError("NVIDIA segment header is ambiguous")
    segment_index = segment_positions[0]
    amount = _parse_displayed_number(revenue_rows[0][segment_index])
    return ReportedValue(
        label=f"{headers[segment_index]} {revenue_rows[0][0].casefold()}",
        value=amount,
        unit=unit,
        chunk_id=chunk.chunk_id,
    )


def _schneider_reported_value(chunk: FinancialChunk, unit: str) -> ReportedValue:
    table = chunk.table
    if table is None or len(table.rows) < 3:
        raise ValueError("Schneider table has no reportable rows")
    period_row, metric_row, *data_rows = table.rows
    target_positions = [
        index
        for index, value in enumerate(period_row)
        if value.casefold().replace(" ", "") == chunk.context.reporting_period.casefold()
    ]
    if len(target_positions) != 1:
        raise ValueError("Schneider reporting-period column is ambiguous")
    amount_index = target_positions[0] - 1
    if amount_index < 0 or "revenue" not in metric_row[amount_index].casefold():
        raise ValueError("Schneider table has no reported revenue column")
    total_rows = [row for row in data_rows if row and row[0].casefold().startswith("total ")]
    if not total_rows:
        raise ValueError("Schneider table has no total row")
    row = total_rows[-1]
    amount = _parse_displayed_number(row[amount_index])
    metric = metric_row[amount_index].split("€", maxsplit=1)[0].strip().casefold()
    return ReportedValue(label=f"{row[0]} {metric}", value=amount, unit=unit, chunk_id=chunk.chunk_id)


def _displayed_unit(chunk: FinancialChunk) -> str | None:
    if chunk.financial.currency is None or chunk.financial.scale is None:
        return None
    return f"{chunk.financial.currency} {chunk.financial.scale}"


def _numeric_cells(row: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(cell for cell in row if re.search(r"\d", cell))


def _parse_displayed_number(value: str) -> float:
    matches = re.findall(r"(?:\d{1,3}(?:,\d{3})+|\d+)", value)
    if len(matches) != 1:
        raise ValueError("displayed table value is ambiguous")
    return float(matches[0].replace(",", ""))


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
            or observation.capability != "inspect_document_evidence"
            or observation.result is None
        ):
            continue
        raw_hit = observation.result.get("hit")
        if not isinstance(raw_hit, Mapping):
            continue
        try:
            hit = CapstoneEvidenceHit.model_validate(raw_hit)
        except (TypeError, ValueError):
            continue
        if (
            hit.chunk_id in observation.evidence_ids
            and hit.source_reference in observation.source_references
        ):
            hits.append(hit)
    return tuple(hits)


def _evaluate_evidence(observations: Sequence[ResearchObservation]) -> EvidenceGateDecision:
    return evaluate_evidence_gate(_document_hits(observations))


def evaluate_evidence_gate(hits: Sequence[CapstoneEvidenceHit]) -> EvidenceGateDecision:
    """Release only certified, contextual table evidence for each company."""

    certified_document_hashes = _certified_document_hashes()
    evidence_hits = tuple(hits)
    coverage: dict[str, tuple[str, ...]] = {}
    missing: list[str] = []
    for company in _COMPANIES:
        contextual = tuple(
            hit
            for hit in evidence_hits
            if hit.company == company
            and hit.element_type == "table"
            and hit.unit is not None
            and hit.chunk_id
            and hit.element_ids
            and hit.physical_page > 0
            and hit.document_sha256 in certified_document_hashes
        )
        coverage[company] = ("document",) if contextual else ()
        if not contextual:
            missing.append(f"{company} contextual table evidence")
    return EvidenceGateDecision(
        passed=not missing,
        coverage=coverage,
        missing_requirements=tuple(missing),
        evidence_hits=evidence_hits,
    )


def _certified_document_hashes() -> frozenset[str]:
    root = Path(__file__).resolve().parents[3]
    manifest = json.loads((root / "assets/course-data/manifest.json").read_text(encoding="utf-8"))
    return frozenset(str(record["sha256"]) for record in manifest["capstone_documents"])


def _classify_question_intent(request: ResearchRequest) -> _QuestionIntent | None:
    """Map a custom question to one small certified intent or reject it explicitly."""

    if request.mode == "reference":
        return "reference"
    question = request.question.casefold()
    padded = f" {question} "
    prohibited = (
        " buy ",
        " sell ",
        "recommend",
        "price target",
        "target price",
        "forecast",
        "predict",
    )
    if any(term in padded for term in prohibited):
        return None
    if any(term in question for term in ("revenue growth", "revenue evidence")):
        return "revenue_growth"
    if any(term in question for term in ("operating-growth", "operating growth", "growth evidence")):
        return "operating_growth"
    if any(term in question for term in ("valuation", "p/e", "price-to-earnings", "multiple")):
        return "valuation"
    return None


def _intent_label(question_intent: _QuestionIntent) -> str:
    return {
        "reference": "two-company research",
        "operating_growth": "operating-growth",
        "valuation": "valuation",
        "revenue_growth": "revenue-growth",
    }[question_intent]


def _build_briefing(
    observations: Sequence[ResearchObservation],
    evidence_gate: EvidenceGateDecision,
    *,
    question_intent: _QuestionIntent = "reference",
) -> CapstoneBriefing:
    del observations
    facts = [
        CitedFact(
            claim=f"{hit.company} ({hit.period}): {hit.text}",
            company=hit.company,
            provenance_kind="document",
            source_reference=hit.source_reference,
            chunk_id=hit.chunk_id,
            element_ids=hit.element_ids,
            physical_page=hit.physical_page,
        )
        for hit in evidence_gate.evidence_hits
    ]
    sources = tuple(dict.fromkeys(fact.source_reference for fact in facts))
    executive_summary = (
        "Evidence-backed comparison prepared for the bounded research request."
        if question_intent == "reference"
        else f"Evidence-backed {_intent_label(question_intent)} comparison prepared for the submitted question."
    )
    interpretation = (
        "The verified evidence supports a qualified operating-growth comparison, not a like-for-like ranking."
        if question_intent in {"reference", "operating_growth"}
        else f"The verified evidence supports a qualified {_intent_label(question_intent)} comparison, not an investment recommendation."
    )
    return CapstoneBriefing(
        executive_summary=executive_summary,
        cited_facts=tuple(facts),
        company_evidence={
            company: tuple(hit for hit in evidence_gate.evidence_hits if hit.company == company)
            for company in _COMPANIES
        },
        cross_company_observations=(
            "NVIDIA reports USD segment revenue for FY2026, while Schneider Electric reports EUR Group revenue for FY2025; their currencies, reporting scopes, and periods differ.",
        ),
        interpretation=(interpretation,),
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
    correctness = observed_signatures == expected_signatures and not errors and replan_count == 0
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
    question_intent = _classify_question_intent(request)
    relevance = bool(
        status == RunStatus.COMPLETED
        and briefing is not None
        and question_intent is not None
        and {fact.company for fact in briefing.cited_facts} == set(_COMPANIES)
        and (
            question_intent == "reference"
            or _intent_label(question_intent) in briefing.executive_summary.casefold()
        )
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
        "Expected document-tool call order completes without an artificial recovery.",
        "The run stays within both budgets and repeats no successful call.",
        "The briefing addresses the fixed two-company research universe.",
        "Both company sections, comparison, interpretation, and limitations are present.",
        "Every cited fact maps to a collected certified chunk, element, source, and page.",
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
    del observations
    hits_by_chunk_id = {hit.chunk_id: hit for hit in evidence_gate.evidence_hits}
    for fact in briefing.cited_facts:
        hit = hits_by_chunk_id.get(fact.chunk_id)
        if (
            hit is None
            or not set(fact.element_ids) <= set(hit.element_ids)
            or fact.company != hit.company
            or fact.source_reference != hit.source_reference
            or fact.physical_page != hit.physical_page
            or hit.document_sha256 not in _certified_document_hashes()
        ):
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
