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

`element_type` accepts mixed-case public input while matching exact
case-folded element values. The model continues to inherit frozen,
extra-forbidden public-contract validation from `FrozenDocumentModel`.

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

The broader Task 1 asset-suite failures discovered during this round were not
pre-existing or unrelated: the credential matcher incorrectly required a
16-character value even for explicit `api_key=...` assignments. They are fixed
in round 2 below.

### Commit

`fix: filter certified document evidence by segment`

## Fix round 2 — canonical element types and credential assignments

### Summary

`DocumentFilters.element_type` now canonicalizes mixed-case input to one of the
six supported `ElementType` values and rejects unknown values at public-model
validation time. This retains case-insensitive retrieval without letting a
misspelled element type silently produce an empty result set.

The public-string credential matcher now rejects any explicit API-key
assignment using `=` or `:`, including short values such as
`api_key=super-secret`. Its longer whitespace-separated branch remains in
place, so normal prose such as `API key performance` stays valid while truly
credential-like unassigned strings remain blocked. The same behavior is
revalidated through table and nested Pydantic `model_construct`/`model_copy`
paths.

### RED evidence

```text
$ UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra capstone pytest \
  tests/test_capstone_document_index.py::test_document_filters_canonicalize_supported_element_types_and_reject_unknown_values \
  tests/test_capstone_document_assets.py -q
5 failed, 42 passed

DocumentFilters(element_type="TABLE") retained "TABLE", and the short
explicit api_key assignment was accepted in direct, table, constructed, and
copied nested public contracts.
```

### GREEN verification

```text
$ UV_CACHE_DIR=/tmp/finai-uv-cache uv run ruff check \
  src/finai_academy/capstone/document_models.py \
  src/finai_academy/capstone/document_index.py \
  tests/test_capstone_document_assets.py \
  tests/test_capstone_document_index.py
All checks passed!

$ UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra capstone pytest \
  tests/test_capstone_document_assets.py \
  tests/test_capstone_document_ingestion.py \
  tests/test_capstone_document_chunking.py \
  tests/test_capstone_document_index.py \
  tests/test_hybrid_retrieval.py -q
122 passed in 7.57s
```

### Commit

`fix: validate document filter types and short credential assignments`
