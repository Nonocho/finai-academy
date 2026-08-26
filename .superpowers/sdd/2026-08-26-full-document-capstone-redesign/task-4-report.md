# Task 4 report

Summary: added deterministic full-PDF extraction, chunk, crop, and manifest generation. The builder certifies one NVIDIA 14x4 table on page 165 and three Schneider six-column tables on page 16.

Files changed: `scripts/build_capstone_document_assets.py`, `tests/test_capstone_document_artifacts.py`, `assets/course-data/manifest.json`, the four generated assets, plus narrow parser/privacy and full-corpus chunking fixes required for certified parsing.

RED evidence: `uv run --extra capstone pytest tests/test_capstone_document_artifacts.py -q` failed with missing `capstone_derived_artifacts` and missing `financial_chunks_v2.json`; the complete-builder test then failed on parser validation and missing local table units before the targeted fixes.

GREEN: `UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra capstone pytest tests/test_capstone_document_artifacts.py tests/test_capstone_document_ingestion.py tests/test_capstone_document_chunking.py -q` — 28 passed. Ruff passed for all Task 4 files. A second generation produced byte-identical assets.

Artifacts: elements `f70afed5f935c4a11b637b689b1bdb49f0c44ad66eff92bddaef53d61cc21082`; chunks `08f0d1408952624a64e1fd92a8d36e9df9e43fcdaea5901c4450a0137cc54fd2`; NVIDIA crop `8ef1de77ff829f97404644c64d2edc66f472767c2256ee07c233a59851f9b80f`; Schneider crop `1e12ca050cab7797749741ae061e27ceecce780df8a4d71191de665349d0044c`.

Corpus commit: `eaa42e1` (`feat: build offline capstone document corpus`).

Concerns: full reports include valid source strings such as financial labels with colon syntax and ordinary authorization wording; privacy detection now rejects credential-shaped assignments and actual drive-root paths without rejecting those source values. Some non-target numeric tables have no explicit unit in locally extracted context; bulk artifacts preserve them with `scale=None`, while direct strict table construction still fails closed.

Fix round 1 verification: `UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra capstone pytest tests/test_capstone_document_artifacts.py::test_builder_certifies_complete_document_target_tables -q` — 1 passed in 22.79s; `...::test_builder_regenerates_byte_identical_artifacts -q` — 1 passed in 21.96s; committed-artifact hash/privacy tests — 2 passed in 0.03s. Ruff passed for the artifact builder, artifact tests, document models, and chunking module.

Fix round 2: spaced API-key labels with credential-like values are rejected while `API key performance` remains accepted. `pytest tests/test_capstone_document_assets.py::test_public_contract_rejects_delimited_credential_labels tests/test_capstone_document_assets.py::test_public_contract_keeps_non_secret_filing_text tests/test_capstone_document_artifacts.py::test_committed_capstone_artifacts_match_manifest_hashes tests/test_capstone_document_artifacts.py::test_committed_chunks_contain_no_personal_paths_or_parser_filenames -q` — 11 passed; Ruff passed.
