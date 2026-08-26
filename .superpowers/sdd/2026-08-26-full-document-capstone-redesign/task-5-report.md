# Task 5 report — certified document retrieval

## Summary

Implemented offline, metadata-first hybrid retrieval over the certified financial
chunk artifact. The index verifies the chunk artifact SHA-256 and source hash
set before parsing immutable chunks, applies company/period/document/element
filters and the financial-context boundary before either ranker is constructed,
then returns deterministic BM25+dense rank lineage and exact chunk inspection.

Exact financial terms receive a deterministic lexical preference in fusion so a
requested reported figure selects its atomic table even when the teaching dense
embedder gives every already-filtered table the same semantic score.

## Files changed

- `src/finai_academy/capstone/document_index.py`
- `tests/test_capstone_document_index.py`
- `.superpowers/sdd/2026-08-26-full-document-capstone-redesign/task-5-report.md`

## RED evidence

```text
$ UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra capstone pytest tests/test_capstone_document_index.py -q
ModuleNotFoundError: No module named 'finai_academy.capstone.document_index'
```

The source-hash regression was then verified separately with the verification
removed:

```text
FAILED: DID NOT RAISE CertifiedDocumentIndexError
```

## GREEN verification

```text
$ UV_CACHE_DIR=/tmp/finai-uv-cache uv run ruff check src/finai_academy/capstone/document_index.py tests/test_capstone_document_index.py
All checks passed!

$ UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra capstone pytest tests/test_capstone_document_index.py tests/test_hybrid_retrieval.py -q
47 passed in 3.95s
```

## Commit

`feat: retrieve contextual document evidence`

## Concerns and rulings

No unresolved concerns. The deterministic teaching embedder creates tied scores
for the NVIDIA-table-only numeric query, so rank fusion intentionally weights
the exact-term BM25 channel while retaining dense rank lineage on every hit.
Unitless numeric tables remain inspectable in the certified artifact but are
excluded from searchable contextual financial evidence.

## Fix round 1 — segment and document identity filters

### Summary

`DocumentFilters` now accepts case-folded exact `segment` values against
`FinancialMetadata.segments` and `document_id` values against the already
certified `ContextualMetadata.document_id`. The existing eligibility tuple in
`CertifiedDocumentIndex.search` remains the single input to both BM25 and dense
rankers, so these fields constrain each channel before ranking.

`element_type` accepts a public string filter rather than a case-sensitive
literal, while still matching only exact case-folded element values. The model
continues to inherit frozen, extra-forbidden public-contract validation from
`FrozenDocumentModel`.

### RED evidence

```text
$ UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra capstone pytest \
  tests/test_capstone_document_index.py::test_document_filters_match_segment_and_document_identity_case_insensitively \
  tests/test_capstone_document_index.py::test_segment_and_document_identity_filters_constrain_both_rankers_before_ranking -q
2 failed

ValidationError: element_type must be one of the case-sensitive literals;
segment and document_id were extra-forbidden inputs.
```

### GREEN verification

```text
$ UV_CACHE_DIR=/tmp/finai-uv-cache uv run ruff check \
  src/finai_academy/capstone/document_models.py \
  src/finai_academy/capstone/document_index.py \
  tests/test_capstone_document_index.py
All checks passed!

$ UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra capstone pytest \
  tests/test_capstone_document_index.py tests/test_hybrid_retrieval.py -q
49 passed in 4.80s

$ UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra capstone pytest \
  tests/test_capstone_document_ingestion.py \
  tests/test_capstone_document_chunking.py \
  tests/test_capstone_document_index.py \
  tests/test_hybrid_retrieval.py -q
75 passed in 6.11s
```

The broader Task 1 asset suite currently has four unrelated pre-existing
credential-pattern failures: its `api_key=super-secret` fixtures are shorter
than the established credential-value minimum. This filter-only change does
not alter that validator.

### Commit

`fix: filter certified document evidence by segment`
