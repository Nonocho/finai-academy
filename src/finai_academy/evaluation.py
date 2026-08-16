"""Deterministic retrieval and answer metrics for the Lesson 07 golden set."""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

from finai_academy.hybrid_retrieval import RetrievalFilters
from finai_academy.retrieval_pipeline import RetrievalResult

CITATION_PATTERN = re.compile(r"\[([A-Z0-9][A-Z0-9-]{2,})\]")
NORMALIZED_FAILURE_STAGES = frozenset(
    {"none", "abstention", "retrieval", "filter", "citation", "grounding"}
)


def _normalized_identifier(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _unique_tuple(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    normalized = tuple(
        _normalized_identifier(value, field_name=field_name) for value in values
    )
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} values must be unique")
    return normalized


@dataclass(frozen=True)
class EvaluationCase:
    """One versioned golden-set question with deterministic expectations."""

    case_id: str
    question: str
    filters: RetrievalFilters
    expected_evidence_ids: tuple[str, ...]
    expected_facts: tuple[str, ...]
    requires_abstention: bool
    tags: tuple[str, ...]

    def __post_init__(self) -> None:
        case_id = _normalized_identifier(self.case_id, field_name="case_id").casefold()
        question = _normalized_identifier(self.question, field_name="question")
        if not isinstance(self.filters, RetrievalFilters):
            raise TypeError("filters must be a RetrievalFilters instance")
        if not isinstance(self.requires_abstention, bool):
            raise TypeError("requires_abstention must be a boolean")
        expected_ids = _unique_tuple(
            self.expected_evidence_ids,
            field_name="expected_evidence_ids",
        )
        expected_facts = _unique_tuple(self.expected_facts, field_name="expected_facts")
        tags = tuple(
            tag.casefold()
            for tag in _unique_tuple(self.tags, field_name="tags")
        )
        if self.requires_abstention and expected_ids:
            raise ValueError("abstention cases must not declare expected evidence")
        if not self.requires_abstention and not expected_ids:
            raise ValueError("non-abstention cases require expected evidence")
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "question", question)
        object.__setattr__(self, "expected_evidence_ids", expected_ids)
        object.__setattr__(self, "expected_facts", expected_facts)
        object.__setattr__(self, "tags", tags)


@dataclass(frozen=True)
class CaseEvaluation:
    """Separated retrieval, citation, grounding and abstention results."""

    case_id: str
    retrieved_ids: tuple[str, ...]
    parsed_citations: tuple[str, ...]
    retrieval_recall_at_k: float
    reciprocal_rank: float
    filter_correctness: float
    citation_correctness: float
    grounded_fact_coverage: float
    abstention_correctness: float
    failure_stage: str

    def __post_init__(self) -> None:
        metrics = (
            self.retrieval_recall_at_k,
            self.reciprocal_rank,
            self.filter_correctness,
            self.citation_correctness,
            self.grounded_fact_coverage,
            self.abstention_correctness,
        )
        if any(not 0.0 <= value <= 1.0 for value in metrics):
            raise ValueError("evaluation metrics must be between 0 and 1")
        if self.failure_stage not in NORMALIZED_FAILURE_STAGES:
            raise ValueError("failure_stage is not normalized")


@dataclass(frozen=True)
class EvaluationSummary:
    """Arithmetic means across a non-empty aligned evaluation collection."""

    case_count: int
    retrieval_recall_at_k: float
    reciprocal_rank: float
    filter_correctness: float
    citation_correctness: float
    grounded_fact_coverage: float
    abstention_correctness: float


def load_evaluation_cases(
    path: Path,
    *,
    known_evidence_ids: Collection[str],
) -> tuple[EvaluationCase, ...]:
    """Load an immutable golden set and reject duplicate or unknown identifiers."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not str(payload.get("dataset_version", "")).strip():
        raise ValueError("evaluation dataset requires dataset_version")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("evaluation dataset requires a non-empty cases list")

    normalized_ids = [str(item.get("case_id", "")).strip().casefold() for item in raw_cases]
    if len(normalized_ids) != len(set(normalized_ids)):
        raise ValueError("duplicate case_id values are not allowed")

    known_ids = set(known_evidence_ids)
    cases = []
    for item in raw_cases:
        filters_payload = item.get("filters", {})
        if not isinstance(filters_payload, dict):
            raise TypeError("filters must be an object")
        expected_ids = tuple(item.get("expected_evidence_ids", ()))
        unknown_ids = sorted(set(expected_ids).difference(known_ids))
        if unknown_ids:
            raise ValueError(f"unknown evidence IDs: {', '.join(unknown_ids)}")
        cases.append(
            EvaluationCase(
                case_id=item.get("case_id", ""),
                question=item.get("question", ""),
                filters=RetrievalFilters(
                    company=filters_payload.get("company"),
                    period=filters_payload.get("period"),
                    document_type=filters_payload.get("document_type"),
                    section=filters_payload.get("section"),
                ),
                expected_evidence_ids=expected_ids,
                expected_facts=tuple(item.get("expected_facts", ())),
                requires_abstention=item.get("requires_abstention", False),
                tags=tuple(item.get("tags", ())),
            )
        )
    return tuple(cases)


def _normalized_phrase(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _fact_is_present(fact: str, answer: str) -> bool:
    normalized_fact = _normalized_phrase(fact)
    normalized_answer = _normalized_phrase(answer)
    if not normalized_fact:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(normalized_fact)}(?![a-z0-9])", normalized_answer) is not None


def _failure_stage(
    *,
    retrieval_recall_at_k: float,
    filter_correctness: float,
    citation_correctness: float,
    grounded_fact_coverage: float,
    abstention_correctness: float,
) -> str:
    if abstention_correctness < 1.0:
        return "abstention"
    if retrieval_recall_at_k < 1.0:
        return "retrieval"
    if filter_correctness < 1.0:
        return "filter"
    if citation_correctness < 1.0:
        return "citation"
    if grounded_fact_coverage < 1.0:
        return "grounding"
    return "none"


def evaluate_case(
    case: EvaluationCase,
    retrieval: RetrievalResult,
    answer: str,
) -> CaseEvaluation:
    """Evaluate one prediction without an LLM judge or hidden provider defaults."""

    normalized_answer = answer.strip()
    if not normalized_answer:
        raise ValueError("answer must not be empty")
    retrieved_ids = tuple(hit.passage.passage_id for hit in retrieval.reranked_hits)
    parsed_citations = tuple(dict.fromkeys(CITATION_PATTERN.findall(normalized_answer)))
    abstained = bool(retrieval.abstention_reason) and not retrieved_ids

    if case.requires_abstention:
        retrieval_recall_at_k = 1.0 if abstained else 0.0
        reciprocal_rank = retrieval_recall_at_k
        citation_correctness = 1.0 if not parsed_citations else 0.0
        grounded_fact_coverage = 1.0 if not case.expected_facts else 0.0
    else:
        expected = set(case.expected_evidence_ids)
        retrieval_recall_at_k = len(expected.intersection(retrieved_ids)) / len(expected)
        reciprocal_rank = next(
            (
                1.0 / rank
                for rank, passage_id in enumerate(retrieved_ids, start=1)
                if passage_id in expected
            ),
            0.0,
        )
        citation_correctness = (
            sum(
                citation in expected and citation in retrieved_ids
                for citation in parsed_citations
            )
            / len(parsed_citations)
            if parsed_citations
            else 0.0
        )
        grounded_fact_coverage = (
            sum(_fact_is_present(fact, normalized_answer) for fact in case.expected_facts)
            / len(case.expected_facts)
            if case.expected_facts
            else 1.0
        )

    filter_correctness = float(
        retrieval.filters == case.filters
        and all(case.filters.matches(hit.passage) for hit in retrieval.reranked_hits)
    )
    abstention_correctness = float(abstained == case.requires_abstention)
    failure_stage = _failure_stage(
        retrieval_recall_at_k=retrieval_recall_at_k,
        filter_correctness=filter_correctness,
        citation_correctness=citation_correctness,
        grounded_fact_coverage=grounded_fact_coverage,
        abstention_correctness=abstention_correctness,
    )
    return CaseEvaluation(
        case_id=case.case_id,
        retrieved_ids=retrieved_ids,
        parsed_citations=parsed_citations,
        retrieval_recall_at_k=retrieval_recall_at_k,
        reciprocal_rank=reciprocal_rank,
        filter_correctness=filter_correctness,
        citation_correctness=citation_correctness,
        grounded_fact_coverage=grounded_fact_coverage,
        abstention_correctness=abstention_correctness,
        failure_stage=failure_stage,
    )


def summarize_evaluations(results: Sequence[CaseEvaluation]) -> EvaluationSummary:
    """Return arithmetic means while preserving retrieval/answer separation."""

    if not results:
        raise ValueError("results must not be empty")
    case_ids = [result.case_id for result in results]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("result case_id values must be unique")

    def mean(field_name: str) -> float:
        return fmean(getattr(result, field_name) for result in results)

    return EvaluationSummary(
        case_count=len(results),
        retrieval_recall_at_k=mean("retrieval_recall_at_k"),
        reciprocal_rank=mean("reciprocal_rank"),
        filter_correctness=mean("filter_correctness"),
        citation_correctness=mean("citation_correctness"),
        grounded_fact_coverage=mean("grounded_fact_coverage"),
        abstention_correctness=mean("abstention_correctness"),
    )
