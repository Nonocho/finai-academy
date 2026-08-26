"""Deterministic evaluation contracts for the financial RAG golden set."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from finai_academy.evaluation import (
    EvaluationCase,
    evaluate_case,
    load_evaluation_cases,
    summarize_evaluations,
)
from finai_academy.hybrid_retrieval import (
    BM25Index,
    DenseIndex,
    DeterministicTeachingEmbeddings,
    RetrievalFilters,
)
from finai_academy.retrieval_pipeline import retrieve_evidence


def _retrieve(corpus, question: str, filters: RetrievalFilters, *, final_k: int = 2):
    embeddings = DeterministicTeachingEmbeddings()
    return retrieve_evidence(
        question,
        keyword_index=BM25Index(corpus),
        dense_index=DenseIndex(
            corpus,
            embeddings,
            provider="offline",
            model=embeddings.model_name,
            chunking_strategy="evaluation-test",
        ),
        filters=filters,
        candidate_k=4,
        final_k=final_k,
    )


def test_evaluation_case_requires_evidence_or_abstention():
    with pytest.raises(ValueError, match="expected evidence"):
        EvaluationCase(
            case_id="nvda-empty",
            question="Unsupported question",
            filters=RetrievalFilters(company="NVIDIA"),
            expected_evidence_ids=(),
            expected_facts=(),
            requires_abstention=False,
            tags=("negative",),
        )


def test_loading_rejects_duplicate_normalized_case_ids_and_unknown_evidence(tmp_path: Path):
    dataset = {
        "dataset_version": "test-v1",
        "cases": [
            {
                "case_id": "NVDA-FACT",
                "question": "Question one?",
                "filters": {"company": "NVIDIA"},
                "expected_evidence_ids": ["NVDA-TABLE"],
                "expected_facts": ["193.7 billion"],
                "requires_abstention": False,
                "tags": ["direct"],
            },
            {
                "case_id": "nvda-fact",
                "question": "Question two?",
                "filters": {"company": "NVIDIA"},
                "expected_evidence_ids": ["UNKNOWN-ID"],
                "expected_facts": ["41%"],
                "requires_abstention": False,
                "tags": ["duplicate"],
            },
        ],
    }
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate case_id"):
        load_evaluation_cases(path, known_evidence_ids={"NVDA-TABLE"})

    dataset["cases"][1]["case_id"] = "unknown-evidence"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown evidence"):
        load_evaluation_cases(path, known_evidence_ids={"NVDA-TABLE"})


def test_metrics_keep_retrieval_and_answer_quality_separate(corpus):
    filters = RetrievalFilters(company="NVIDIA", period="FY2026")
    retrieval = _retrieve(
        corpus,
        "Which NVIDIA business generated $193.7 billion?",
        filters,
        final_k=1,
    )
    case = EvaluationCase(
        case_id="nvda-data-center",
        question="Which NVIDIA business generated $193.7 billion?",
        filters=filters,
        expected_evidence_ids=("NVDA-TABLE",),
        expected_facts=("Data Center", "$193.7 billion"),
        requires_abstention=False,
        tags=("direct-fact",),
    )

    result = evaluate_case(
        case,
        retrieval,
        "Data Center generated the revenue [SU-TABLE].",
    )

    assert result.retrieval_recall_at_k == 1.0
    assert result.reciprocal_rank == 1.0
    assert result.filter_correctness == 1.0
    assert result.citation_correctness == 0.0
    assert result.grounded_fact_coverage == 0.5
    assert result.abstention_correctness == 1.0
    assert result.failure_stage == "citation"


def test_evaluation_recognizes_correct_abstention(corpus):
    filters = RetrievalFilters(company="Absent Company")
    retrieval = _retrieve(corpus, "What was revenue?", filters)
    case = EvaluationCase(
        case_id="unsupported-company",
        question="What was revenue?",
        filters=filters,
        expected_evidence_ids=(),
        expected_facts=(),
        requires_abstention=True,
        tags=("negative",),
    )

    result = evaluate_case(case, retrieval, "Insufficient evidence.")

    assert result.retrieval_recall_at_k == 1.0
    assert result.reciprocal_rank == 1.0
    assert result.citation_correctness == 1.0
    assert result.grounded_fact_coverage == 1.0
    assert result.abstention_correctness == 1.0
    assert result.failure_stage == "none"


def test_evaluation_scores_explicit_answer_abstention_when_retrieval_returns_passages(corpus):
    """Unsupported questions may retrieve plausible context and still require refusal."""

    filters = RetrievalFilters(company="NVIDIA", period="FY2026")
    retrieval = _retrieve(corpus, "What fair value should investors assign?", filters)
    case = EvaluationCase(
        case_id="unsupported-valuation",
        question="What fair value should investors assign?",
        filters=filters,
        expected_evidence_ids=(),
        expected_facts=(),
        requires_abstention=True,
        tags=("negative",),
    )

    assert retrieval.reranked_hits
    result = evaluate_case(
        case,
        retrieval,
        "The provided evidence does not establish fair value.",
        abstained=True,
    )

    assert result.retrieval_recall_at_k == 1.0
    assert result.abstention_correctness == 1.0
    assert result.failure_stage == "none"


def test_summary_uses_arithmetic_means(corpus):
    filters = RetrievalFilters(company="NVIDIA", period="FY2026")
    retrieval = _retrieve(corpus, "Which NVIDIA business generated $193.7 billion?", filters)
    case = EvaluationCase(
        case_id="nvda-summary",
        question="Which NVIDIA business generated $193.7 billion?",
        filters=filters,
        expected_evidence_ids=("NVDA-TABLE",),
        expected_facts=("Data Center", "$193.7 billion"),
        requires_abstention=False,
        tags=("summary",),
    )
    perfect = evaluate_case(
        case,
        retrieval,
        "Data Center generated $193.7 billion [NVDA-TABLE].",
    )
    incomplete = evaluate_case(case, retrieval, "Data Center [SU-TABLE].")

    summary = summarize_evaluations((perfect, replace(incomplete, case_id="nvda-summary-2")))

    assert summary.case_count == 2
    assert summary.retrieval_recall_at_k == 1.0
    assert summary.citation_correctness == 0.5
    assert summary.grounded_fact_coverage == 0.75


def test_course_golden_set_is_versioned_and_loadable():
    root = Path(__file__).resolve().parents[1]
    path = root / "assets" / "course-data" / "evaluation" / "rag_cases_v1.json"
    known_ids = {
        "NVDA-2026-10K-EXCERPT-CONTEXTUAL-001",
        "NVDA-2026-10K-EXCERPT-CONTEXTUAL-002",
        "NVDA-2026-10K-EXCERPT-CONTEXTUAL-003",
        "NVDA-2026-10K-EXCERPT-CONTEXTUAL-004",
        "SU-2025-FY-EXCERPT-CONTEXTUAL-001",
        "SU-2025-FY-EXCERPT-CONTEXTUAL-002",
        "SU-2025-FY-EXCERPT-CONTEXTUAL-003",
    }

    cases = load_evaluation_cases(path, known_evidence_ids=known_ids)

    assert len(cases) >= 8
    assert {case.case_id for case in cases} >= {
        "nvda-direct-fact",
        "nvda-exact-number",
        "nvda-semantic-paraphrase",
        "nvda-filter-safety",
        "schneider-filter-safety",
        "cross-company-leakage",
        "multi-evidence-comparison",
        "insufficient-evidence",
    }
