"""Local MLflow tracking and trace integration for Lesson 07."""

from __future__ import annotations

import mlflow

from finai_academy.evaluation import EvaluationCase
from finai_academy.hybrid_retrieval import (
    BM25Index,
    DenseIndex,
    DeterministicTeachingEmbeddings,
    RetrievalFilters,
)
from finai_academy.mlflow_evaluation import (
    EvaluationConfiguration,
    EvaluationPrediction,
    run_mlflow_evaluation,
)
from finai_academy.retrieval_pipeline import retrieve_evidence


def _configuration(configuration_id: str, *, bm25_weight: float = 1.0):
    return EvaluationConfiguration(
        configuration_id=configuration_id,
        dataset_version="rag-cases-v1",
        provider="offline",
        chat_model="recorded-answer-v1",
        embedding_model="financial-concepts-v1",
        index_version="test-index-v1",
        prompt_version="answer-with-citations-v1",
        candidate_k=4,
        final_k=2,
        rrf_weights={"bm25": bm25_weight, "dense": 1.0},
    )


def _cases():
    return (
        EvaluationCase(
            case_id="nvda-fact",
            question="Which NVIDIA business generated $193.7 billion?",
            filters=RetrievalFilters(company="NVIDIA", period="FY2026"),
            expected_evidence_ids=("NVDA-TABLE",),
            expected_facts=("Data Center", "$193.7 billion"),
            requires_abstention=False,
            tags=("direct",),
        ),
        EvaluationCase(
            case_id="unsupported-nvda-valuation",
            question="What fair value should investors assign to NVIDIA?",
            filters=RetrievalFilters(company="NVIDIA", period="FY2026"),
            expected_evidence_ids=(),
            expected_facts=(),
            requires_abstention=True,
            tags=("negative",),
        ),
    )


def _predictor(corpus, configuration: EvaluationConfiguration):
    embeddings = DeterministicTeachingEmbeddings()
    bm25_index = BM25Index(corpus)
    dense_index = DenseIndex(
        corpus,
        embeddings,
        provider="offline",
        model=embeddings.model_name,
        chunking_strategy="evaluation-test",
    )

    def predict(case, observer):
        retrieval = retrieve_evidence(
            case.question,
            keyword_index=bm25_index,
            dense_index=dense_index,
            filters=case.filters,
            candidate_k=configuration.candidate_k,
            final_k=configuration.final_k,
            weights=configuration.retrieval_weights,
            observer=observer,
        )
        with observer.span("context", inputs={"final_k": configuration.final_k}):
            contexts = tuple(hit.passage.text for hit in retrieval.reranked_hits)
        with observer.span("generation", inputs={"prompt_version": configuration.prompt_version}):
            answer = (
                "Data Center generated $193.7 billion [NVDA-TABLE]."
                if case.case_id == "nvda-fact"
                else "The provided evidence does not establish fair value."
            )
        return EvaluationPrediction(
            retrieval=retrieval,
            answer=answer,
            contexts=contexts,
            abstained=False,
        )

    return predict


def test_mlflow_run_logs_complete_reproducibility_metadata(tmp_path, corpus):
    configuration = _configuration("baseline")
    cases = _cases()

    summary = run_mlflow_evaluation(
        tracking_path=tmp_path / "mlruns",
        experiment_name="lesson-07-test",
        configuration=configuration,
        cases=cases,
        predict_fn=_predictor(corpus, configuration),
    )

    assert summary.parameters["dataset_version"] == "rag-cases-v1"
    assert summary.parameters["index_version"] == "test-index-v1"
    assert summary.parameters["prompt_version"] == "answer-with-citations-v1"
    assert summary.trace_count == len(cases)
    assert summary.metrics["retrieval_recall_at_k"] == 0.5
    assert summary.metrics["citation_correctness"] == 1.0

    experiment_id = mlflow.get_run(summary.run_id).info.experiment_id
    traces = mlflow.search_traces(
        run_id=summary.run_id,
        locations=[experiment_id],
        return_type="list",
        flush=True,
    )
    assert len(traces) == len(cases)
    span_names = {
        span.name
        for trace in traces
        for span in trace.data.spans
    }
    assert {"eligibility", "bm25", "dense", "fusion", "rerank"} <= span_names
    assert {"context", "generation"} <= span_names


def test_mlflow_comparison_keeps_cases_aligned_and_classifies_failure(tmp_path, corpus):
    baseline = _configuration("baseline")
    weighted = _configuration("bm25-weight-3", bm25_weight=3.0)
    cases = _cases()

    summaries = [
        run_mlflow_evaluation(
            tracking_path=tmp_path / "mlruns",
            experiment_name="lesson-07-compare",
            configuration=configuration,
            cases=cases,
            predict_fn=_predictor(corpus, configuration),
        )
        for configuration in (baseline, weighted)
    ]

    assert summaries[0].run_id != summaries[1].run_id
    assert [row["case_id"] for row in summaries[0].failure_rows] == [
        "unsupported-nvda-valuation"
    ]
    assert [row["case_id"] for row in summaries[1].failure_rows] == [
        "unsupported-nvda-valuation"
    ]
    assert set(summaries[0].failure_rows[0]) >= {
        "case_id",
        "configuration_id",
        "failure_stage",
        "expected_ids",
        "retrieved_ids",
        "citations",
        "retrieval_recall_at_k",
        "citation_correctness",
    }


def test_configuration_exposes_internal_retrieval_weights_without_leaking_old_labels():
    configuration = _configuration("bm25-baseline", bm25_weight=2.5)

    assert dict(configuration.rrf_weights) == {"bm25": 2.5, "dense": 1.0}
    assert configuration.retrieval_weights == {"keyword": 2.5, "dense": 1.0}
