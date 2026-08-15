"""Small deterministic helpers used by guided course notebooks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel

ResponseT = TypeVar("ResponseT", bound=BaseModel)


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
