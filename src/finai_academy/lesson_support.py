"""Small deterministic helpers used by guided course notebooks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


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
