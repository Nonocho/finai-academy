"""Integration tests for the metadata-safe hybrid retrieval boundary."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from finai_academy.hybrid_retrieval import (
    DenseIndex,
    DeterministicTeachingEmbeddings,
    KeywordIndex,
    RetrievalFilters,
)
from finai_academy.retrieval_pipeline import retrieve_evidence, verify_retrieval_runs


def build_indexes(corpus):
    """Build the offline indexes used by the complete pipeline contract."""

    embeddings = DeterministicTeachingEmbeddings()
    return (
        KeywordIndex(corpus),
        DenseIndex(
            corpus,
            embeddings,
            provider="offline",
            model=embeddings.model_name,
            chunking_strategy="contextual-structure",
        ),
    )


def test_pipeline_abstains_instead_of_broadening_empty_filters(corpus):
    """An impossible metadata constraint must return an explicit empty result."""

    keyword_index, dense_index = build_indexes(corpus)

    result = retrieve_evidence(
        "What was revenue?",
        keyword_index=keyword_index,
        dense_index=dense_index,
        filters=RetrievalFilters(company="Absent Company"),
        candidate_k=4,
        final_k=2,
    )

    assert result.reranked_hits == ()
    assert result.abstention_reason == "No passages matched the requested metadata filters."


def test_pipeline_abstention_checks_eligibility_before_searching_indexes(corpus):
    """The no-match safety barrier must prevent either retrieval stage from running."""

    class SearchForbiddenIndex:
        def __init__(self, passages):
            self.passages = passages

        def search(self, *args, **kwargs):
            raise AssertionError("search must not run when no passage is eligible")

    result = retrieve_evidence(
        "What was revenue?",
        keyword_index=SearchForbiddenIndex(corpus),
        dense_index=SearchForbiddenIndex(corpus),
        filters=RetrievalFilters(company="Absent Company"),
        candidate_k=4,
        final_k=2,
    )

    assert result.abstention_reason == "No passages matched the requested metadata filters."


def test_pipeline_keeps_all_rankings_visible_after_pre_filtering(corpus):
    """Results must show the filtered channel, fusion, and final evidence stages."""

    keyword_index, dense_index = build_indexes(corpus)

    result = retrieve_evidence(
        "What was revenue?",
        keyword_index=keyword_index,
        dense_index=dense_index,
        filters=RetrievalFilters(company="Schneider Electric"),
        candidate_k=4,
        final_k=1,
    )

    assert result.keyword_hits and result.dense_hits and result.fused_hits
    assert len(result.reranked_hits) == 1
    assert result.abstention_reason is None
    assert {hit.passage.company for hit in result.fused_hits} == {"Schneider Electric"}


def test_retrieval_result_exposes_every_stage_measurement(corpus):
    """Every executed retrieval stage must expose a finite non-negative duration."""

    keyword_index, dense_index = build_indexes(corpus)

    result = retrieve_evidence(
        "What was revenue?",
        keyword_index=keyword_index,
        dense_index=dense_index,
        filters=RetrievalFilters(company="Schneider Electric"),
        candidate_k=4,
        final_k=1,
    )

    assert tuple(result.stage_measurements) == (
        "eligibility",
        "keyword",
        "dense",
        "fusion",
        "rerank",
    )
    assert all(item.duration_ms >= 0 for item in result.stage_measurements.values())
    assert all(item.metadata.get("status") == "completed" for item in result.stage_measurements.values())


def test_abstention_records_skipped_retrieval_stage_measurements(corpus):
    """An eligibility abstention must not invent durations for stages that never ran."""

    keyword_index, dense_index = build_indexes(corpus)

    result = retrieve_evidence(
        "What was revenue?",
        keyword_index=keyword_index,
        dense_index=dense_index,
        filters=RetrievalFilters(company="Absent Company"),
        candidate_k=4,
        final_k=1,
    )

    assert result.stage_measurements["eligibility"].metadata["status"] == "completed"
    for stage in ("keyword", "dense", "fusion", "rerank"):
        assert result.stage_measurements[stage].duration_ms == 0.0
        assert result.stage_measurements[stage].metadata == {"status": "skipped"}


def test_pipeline_does_not_abstain_when_only_the_dense_index_has_eligible_passages(corpus):
    """Active dense retrieval must accept weights declared for both supported channels."""

    embeddings = DeterministicTeachingEmbeddings()
    keyword_index = KeywordIndex(corpus[:1])
    dense_index = DenseIndex(
        corpus[2:],
        embeddings,
        provider="offline",
        model=embeddings.model_name,
        chunking_strategy="contextual-structure",
    )

    result = retrieve_evidence(
        "What margin reached 18.7%?",
        keyword_index=keyword_index,
        dense_index=dense_index,
        filters=RetrievalFilters(company="Schneider Electric"),
        candidate_k=2,
        final_k=1,
        weights={"keyword": 1.0, "dense": 2.0},
    )

    assert result.abstention_reason is None
    assert result.keyword_hits == ()
    assert result.dense_hits
    assert result.reranked_hits[0].passage.company == "Schneider Electric"


def test_pipeline_validates_weights_before_an_empty_filter_result(corpus):
    """Unsupported channel weights must fail at the public orchestration boundary."""

    keyword_index, dense_index = build_indexes(corpus)

    with pytest.raises(ValueError, match="weights"):
        retrieve_evidence(
            "What was revenue?",
            keyword_index=keyword_index,
            dense_index=dense_index,
            filters=RetrievalFilters(company="Absent Company"),
            candidate_k=4,
            final_k=2,
            weights={"unsupported": 1.0},
        )


@pytest.mark.parametrize(("candidate_k", "final_k"), [(0, 1), (1, 0), (1, 2)])
def test_pipeline_rejects_an_invalid_candidate_and_final_budget(corpus, candidate_k, final_k):
    """The wider candidate stage cannot be smaller than the final evidence budget."""

    keyword_index, dense_index = build_indexes(corpus)

    with pytest.raises(ValueError, match="candidate_k"):
        retrieve_evidence(
            "What was revenue?",
            keyword_index=keyword_index,
            dense_index=dense_index,
            filters=RetrievalFilters(),
            candidate_k=candidate_k,
            final_k=final_k,
        )


def build_maintained_run(corpus):
    """Build one successful maintained run for verification-predicate tests."""

    keyword_index, dense_index = build_indexes(corpus)
    return retrieve_evidence(
        "What margin reached 18.7%?",
        keyword_index=keyword_index,
        dense_index=dense_index,
        filters=RetrievalFilters(company="Schneider Electric", period="FY2025"),
        candidate_k=2,
        final_k=2,
    )


def test_verification_report_requires_every_structural_and_offline_evidence_check(
    corpus, tmp_path
):
    """The PASS predicate must cover artifacts, stages, filters, order, IDs, and evidence."""

    run = build_maintained_run(corpus)
    manifest = tmp_path / "manifest.json"
    vectors = tmp_path / "vectors.npy"
    manifest.touch()
    vectors.touch()

    report = verify_retrieval_runs(
        {"se-margin": run},
        expected_evidence={"se-margin": "18.7%"},
        require_expected_evidence=True,
        required_artifacts=(manifest, vectors),
    )

    assert report.passed
    assert all(report.checks.values())


@pytest.mark.parametrize(
    ("mutate", "failed_check"),
    [
        (
            lambda run: replace(
                run,
                filters=RetrievalFilters(company="NVIDIA", period="FY2026"),
            ),
            "all stage hits satisfy their run filters",
        ),
        (
            lambda run: replace(run, fused_hits=(run.fused_hits[0], run.fused_hits[0])),
            "fused passage identifiers are unique per run",
        ),
        (
            lambda run: replace(run, fused_hits=tuple(reversed(run.fused_hits))),
            "fused hits are sorted by descending score then passage ID",
        ),
        (
            lambda run: replace(
                run,
                keyword_hits=(
                    SimpleNamespace(
                        passage=run.keyword_hits[0].passage,
                        score=float("nan"),
                    ),
                ),
            ),
            "all retrieval scores are finite",
        ),
    ],
)
def test_verification_report_rejects_each_regressed_run_invariant(
    corpus, tmp_path, mutate, failed_check
):
    """A regressed invariant must make the explicit verification predicate false."""

    manifest = tmp_path / "manifest.json"
    manifest.touch()
    run = mutate(build_maintained_run(corpus))

    report = verify_retrieval_runs(
        {"se-margin": run},
        expected_evidence={"se-margin": "18.7%"},
        require_expected_evidence=True,
        required_artifacts=(manifest,),
    )

    assert report.passed is False
    assert report.checks[failed_check] is False


def test_verification_report_rejects_missing_artifacts_and_expected_evidence(
    corpus, tmp_path
):
    """Offline PASS must fail when an artifact or final-k evidence token is absent."""

    report = verify_retrieval_runs(
        {"se-margin": build_maintained_run(corpus)},
        expected_evidence={"se-margin": "not present"},
        require_expected_evidence=True,
        required_artifacts=(tmp_path / "missing-manifest.json",),
    )

    assert report.passed is False
    assert report.checks["required index artifacts exist"] is False
    assert report.checks["offline expected evidence is recovered within final_k"] is False
