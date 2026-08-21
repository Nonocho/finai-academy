"""Small deterministic helpers used by guided course notebooks."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

import numpy as np
from pydantic import BaseModel

from finai_academy.agent_workflows import (
    AgentDecision,
    ToolObservation,
    ToolRequest,
    TraceStep,
    WorkflowPlan,
)

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class ManifestLabelPassage(Protocol):
    """The provenance fields required for compact notebook labels."""

    company: str
    period: str
    passage_id: str


def normalize_rows(values: Sequence[Sequence[float]]) -> np.ndarray:
    """Return finite row-normalized vectors while preserving all-zero rows."""

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("values must be a two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("values must contain only finite numbers")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def compact_manifest_labels(
    passages: Sequence[ManifestLabelPassage],
) -> dict[str, str]:
    """Build provenance-rich labels with collision-safe stable-ID abbreviations."""

    passage_ids = [passage.passage_id for passage in passages]
    if len(passage_ids) != len(set(passage_ids)):
        raise ValueError("passage_id values must be unique")

    abbreviated_ids = {
        passage.passage_id: _abbreviate_manifest_id(passage.passage_id)
        for passage in passages
    }
    provisional_labels = {
        passage.passage_id: (
            f"{passage.company} · {passage.period} · "
            f"{abbreviated_ids[passage.passage_id]}"
        )
        for passage in passages
    }
    collisions = {
        label
        for label in provisional_labels.values()
        if tuple(provisional_labels.values()).count(label) > 1
    }
    return {
        passage.passage_id: (
            f"{passage.company} · {passage.period} · "
            f"{passage.passage_id if provisional_labels[passage.passage_id] in collisions else abbreviated_ids[passage.passage_id]}"
        )
        for passage in passages
    }


def spread_label_positions(
    desired_positions: Sequence[float],
    *,
    lower: float,
    upper: float,
    minimum_gap: float,
) -> tuple[float, ...]:
    """Place ordered annotation centers deterministically without vertical collisions."""

    if not all(np.isfinite(value) for value in (*desired_positions, lower, upper, minimum_gap)):
        raise ValueError("label positions and bounds must be finite")
    if upper <= lower or minimum_gap <= 0:
        raise ValueError("label bounds and minimum_gap must be positive")
    if len(desired_positions) > 1 and (len(desired_positions) - 1) * minimum_gap > upper - lower:
        raise ValueError("label bounds cannot accommodate the requested minimum_gap")
    if not desired_positions:
        return ()

    positions = [min(upper, max(lower, float(value))) for value in desired_positions]
    positions[0] = min(positions[0], upper - minimum_gap * (len(positions) - 1))
    for index in range(1, len(positions)):
        positions[index] = max(positions[index], positions[index - 1] + minimum_gap)
    if positions[-1] > upper:
        positions[-1] = upper
        for index in range(len(positions) - 2, -1, -1):
            positions[index] = min(positions[index], positions[index + 1] - minimum_gap)
    return tuple(positions)


def _abbreviate_manifest_id(passage_id: str) -> str:
    parts = passage_id.split("-")
    if len(passage_id) <= 20 or len(parts) < 2:
        return passage_id
    return f"{parts[0]}…{parts[-1]}"


@dataclass(frozen=True)
class RecordedMessage:
    """Provider-like message returned by the offline classroom fixture."""

    content: str
    response_metadata: dict[str, Any]


class RecordedChatModel:
    """Return stable responses when the notebook execution suite is offline."""

    def invoke(self, messages: list[tuple[str, str]]) -> RecordedMessage:
        question = messages[-1][1].casefold()
        if "nvidia" in question and "f1" in question:
            content = (
                "NVIDIA reported fiscal 2026 revenue of $215.9 billion, up 65% [F1]. "
                "Data Center revenue reached $193.7 billion, up 68% [F2], showing that "
                "the company's growth was concentrated in that end market. The supplied "
                "facts cannot establish valuation, a price target or management guidance."
            )
        else:
            content = (
                "AI demand can support growth, but the question does not specify a company, "
                "period, source or required evidence."
            )
        return RecordedMessage(content=content, response_metadata={"mode": "offline fixture"})


class RecordedStructuredModel:
    """Return one stable typed analyst brief through the production protocol."""

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseT],
    ) -> ResponseT:
        del system_prompt, user_prompt
        return response_model.model_validate(
            {
                "company": "NVIDIA",
                "reporting_period": "fiscal 2026",
                "executive_summary": (
                    "NVIDIA reported strong fiscal 2026 growth led by Data Center, "
                    "while the supplied evidence does not establish valuation."
                ),
                "findings": [
                    {
                        "statement": "Fiscal 2026 revenue reached $215.9 billion, up 65%.",
                        "category": "key_result",
                        "evidence_type": "reported_fact",
                        "source_excerpt": (
                            "Revenue for fiscal 2026 was $215.9 billion, up 65% "
                            "from a year ago."
                        ),
                    },
                    {
                        "statement": "Growth was concentrated in Data Center.",
                        "category": "risk",
                        "evidence_type": "interpretation",
                        "rationale": (
                            "Data Center revenue of $193.7 billion represented most of "
                            "the $215.9 billion total supplied in the evidence card."
                        ),
                    },
                ],
                "open_questions": [
                    "How much of the reported growth came from volume versus pricing?"
                ],
                "caveats": [
                    "The supplied evidence does not establish valuation or a price target."
                ],
            }
        )


class RecordedRagModel:
    """Return one stable answer for the offline naive RAG lesson."""

    def invoke(self, messages: list[tuple[str, str]]) -> RecordedMessage:
        del messages
        return RecordedMessage(
            content=(
                "NVIDIA's Data Center business drove fiscal 2026 growth. "
                "Data Center revenue reached $193.7 billion, up 68% [NVDA-F2], "
                "within total revenue of $215.9 billion, up 65% [NVDA-F1]. "
                "The retrieved evidence does not establish valuation or a price target."
            ),
            response_metadata={"mode": "offline RAG fixture"},
        )


class RecordedChunkingModel:
    """Return stable atomic propositions for the offline chunking laboratory."""

    def invoke(self, messages: list[tuple[str, str]]) -> RecordedMessage:
        text = messages[-1][1].casefold()
        if "193.7" in text or "16.0" in text:
            propositions = [
                "NVIDIA Data Center revenue was $193.7 billion in fiscal 2026.",
                "NVIDIA Data Center revenue increased 68% year on year.",
                "NVIDIA Gaming revenue was $16.0 billion in fiscal 2026.",
                "NVIDIA Gaming revenue increased 41% year on year.",
            ]
        elif "40.2" in text or "adjusted ebita" in text:
            propositions = [
                "Schneider Electric FY2025 revenue was EUR 40.2 billion.",
                "Schneider Electric FY2025 organic revenue growth was 8.9%.",
                "Schneider Electric FY2025 adjusted EBITA was EUR 7.5 billion.",
                "Schneider Electric FY2025 adjusted EBITA margin was 18.7%.",
            ]
        elif "nvidia" in text or "215.9" in text:
            propositions = [
                "NVIDIA reported fiscal 2026 revenue of $215.9 billion.",
                "NVIDIA fiscal 2026 revenue increased 65% from the prior year.",
            ]
        elif "data center" in text:
            propositions = [
                "Data Center revenue was $193.7 billion in NVIDIA fiscal 2026.",
                "NVIDIA Data Center revenue increased 68% year on year.",
            ]
        else:
            propositions = ["The source block contains one evidence-bound statement."]
        return RecordedMessage(
            content=json.dumps({"propositions": propositions}),
            response_metadata={"mode": "offline chunking fixture"},
        )


class RecordedContextualChunkingModel:
    """Return versioned contextual descriptions keyed by stable chunk identifier."""

    def __init__(self, contexts: Mapping[str, str]) -> None:
        self.contexts = {
            chunk_id.strip(): context.strip()
            for chunk_id, context in contexts.items()
            if chunk_id.strip() and context.strip()
        }

    def invoke(self, messages: list[tuple[str, str]]) -> RecordedMessage:
        payload = json.loads(messages[-1][1])
        chunk_id = payload.get("chunk_id")
        if not isinstance(chunk_id, str) or chunk_id not in self.contexts:
            raise ValueError("recorded context is unavailable for the supplied chunk_id")
        return RecordedMessage(
            content=json.dumps({"context": self.contexts[chunk_id]}),
            response_metadata={"mode": "offline contextual chunking fixture"},
        )


class RecordedLesson08Model:
    """Provide stable workflow and agent decisions for offline Lesson 08 runs."""

    mode = "offline fixture"

    def plan_workflow(self, question: str) -> WorkflowPlan:
        normalized = question.casefold()
        if "convert" in normalized or "euro" in normalized:
            return WorkflowPlan(
                route="unsupported_dependency",
                reason=(
                    "The conversion amount depends on the unseen price observation; "
                    "this one-pass workflow has no predefined second branch."
                ),
            )
        if "nvidia" in normalized or "nvda" in normalized:
            return WorkflowPlan(
                route="tool",
                request=ToolRequest(
                    name="get_market_price",
                    arguments={"ticker": "NVDA"},
                ),
                reason="A direct price lookup requires one predetermined tool call.",
            )
        return WorkflowPlan(
            route="finish",
            answer="The offline fixture supports NVIDIA and Schneider Electric only.",
            reason="The question falls outside the maintained course snapshot.",
        )

    def write_workflow_answer(
        self,
        question: str,
        observations: tuple[ToolObservation, ...],
    ) -> str:
        del question
        observation = observations[0]
        return (
            f"{observation.payload['company']}: {observation.payload['price']:.4f} "
            f"{observation.payload['currency']} as of {observation.payload['as_of']} "
            f"[{observation.payload['source']}]."
        )

    def decide_agent(
        self,
        question: str,
        trajectory: tuple[TraceStep, ...],
    ) -> AgentDecision:
        del question
        tool_steps = [step for step in trajectory if step.phase == "tool"]
        if not tool_steps:
            return AgentDecision(
                action="tool",
                request=ToolRequest(
                    name="get_market_price",
                    arguments={"ticker": "NVDA"},
                ),
            )

        latest = tool_steps[-1].observation
        if latest is None:
            raise ValueError("recorded policy requires visible tool observations")
        if latest.status == "error":
            return AgentDecision(
                action="finish",
                answer=f"The course tool returned an error: {latest.error}",
            )
        if latest.tool_name == "get_market_price":
            return AgentDecision(
                action="tool",
                request=ToolRequest(
                    name="convert_currency",
                    arguments={
                        "amount": latest.payload["price"],
                        "from_currency": latest.payload["currency"],
                        "to_currency": "EUR",
                    },
                ),
            )
        return AgentDecision(
            action="finish",
            answer=(
                f"NVIDIA: EUR {latest.payload['output_amount']:.4f} "
                f"using USD/EUR {latest.payload['rate']:.6f} "
                f"as of {latest.payload['rate_as_of']} [{latest.payload['source']}]."
            ),
        )


@dataclass(frozen=True)
class GroundingResult:
    """Observable checks for the Session 01 evidence-bounded answer."""

    checks: dict[str, bool]

    @property
    def score(self) -> int:
        return sum(self.checks.values())

    @property
    def maximum(self) -> int:
        return len(self.checks)

    @property
    def passed(self) -> bool:
        return self.score == self.maximum


def evaluate_grounding(text: str) -> GroundingResult:
    """Score four transparent behaviours without using another language model."""

    normalized = " ".join(text.casefold().split())
    metric_groups = (
        ("215.9", "65%"),
        ("193.7", "68%"),
        ("16.0", "41%"),
        ("4.5",),
    )
    metrics_used = sum(any(token in normalized for token in group) for group in metric_groups)
    citations_used = sum(f"[f{index}]" in normalized for index in range(1, 5))
    without_citations = re.sub(r"\[(?:f[1-4](?:/f[1-4])*)\]", "", normalized)
    numeric_tokens = set(re.findall(r"(?<![a-z])\d+(?:\.\d+)?", without_citations))
    allowed_numeric_tokens = {
        "2026",
        "215.9",
        "65",
        "193.7",
        "68",
        "16.0",
        "16",
        "41",
        "4.5",
    }
    limitation_markers = (
        "cannot",
        "does not establish",
        "not provided",
        "insufficient",
        "not enough evidence",
    )

    return GroundingResult(
        checks={
            "company and period": "nvidia" in normalized
            and ("fiscal 2026" in normalized or "fy2026" in normalized),
            "at least two evidence-bounded metrics": metrics_used >= 2
            and numeric_tokens <= allowed_numeric_tokens,
            "at least two evidence citations": citations_used >= 2,
            "explicit limitation": any(marker in normalized for marker in limitation_markers),
        }
    )
