# Session 06 Hybrid Retrieval Laboratory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a 45-minute visual laboratory that compares lexical, dense, metadata-filtered hybrid, and reranked evidence retrieval over the trusted Lesson 05 financial chunks.

**Architecture:** `hybrid_retrieval.py` owns metadata eligibility, provider-neutral dense and keyword indexes, index versioning, persistence, and rank fusion. `reranking.py` owns transparent second-stage feature scoring, while `retrieval_pipeline.py` composes both layers without circular imports. The notebook rebuilds the Lesson 05 structure-aware corpus, then changes one retrieval layer at a time while the deck teaches the same seven-step mental model.

**Tech Stack:** Python 3.11, NumPy, scikit-learn, pandas, matplotlib, provider-neutral LangChain embeddings, Jupyter, pytest, Ruff, `@oai/artifact-tool`.

## Global Constraints

- The session is Day 1, 15:15–16:00: 15 minutes slides and 30 minutes notebook.
- Parsing, structure-aware chunks, source fixtures, provenance, and maintained questions remain fixed from Lesson 05.
- The source notebook must run without network access in offline mode.
- The same notebook must support Ollama `qwen3-embedding:0.6b` and OpenAI `text-embedding-3-small` through `create_embeddings(Settings.from_environment())`.
- Offline dense vectors must be clearly labelled illustrative rather than production embeddings.
- Filters cover company, period, document type, and section using logical AND.
- The repository must not commit generated vectors or introduce an external vector database.
- Cosine similarity is a ranking score, never a probability or confidence measure.
- HyDE, ANN internals, generative reranking, and multi-model benchmarking remain out of scope.
- The footer is `First Finance - Arnaud Demes`.

---

### Task 1: Provider-neutral keyword and dense indexes with metadata eligibility

**Files:**
- Create: `src/finai_academy/hybrid_retrieval.py`
- Create: `tests/conftest.py`
- Create: `tests/test_hybrid_retrieval.py`

**Interfaces:**
- Consumes: `EvidencePassage`, `RetrievalHit`, `LexicalRetriever` from `finai_academy.retrieval` and any embedding object exposing `embed_documents(list[str])` plus `embed_query(str)`.
- Produces: `IndexedPassage`, `RetrievalFilters`, `DeterministicTeachingEmbeddings`, `KeywordIndex`, and `DenseIndex`.
- `IndexedPassage` extends `EvidencePassage` with `document_type` while preserving the Lesson 04 retrieval contract.
- `KeywordIndex.search(query, top_k, filters=None)` and `DenseIndex.search(query, top_k, filters=None)` both return `list[RetrievalHit]` sorted by descending score and then passage identifier.

- [ ] **Step 1: Write the index and filter contract tests**

Use a compact unit-test corpus with the same facts and provenance as the Lesson 05
fixtures. Put the `corpus` fixture below in `tests/conftest.py` so the index, reranker,
and pipeline tests share exactly the same evidence. The notebook later rebuilds all seven
chunks through the real parsers.

```python
@pytest.fixture
def corpus() -> tuple[IndexedPassage, ...]:
    return (
        IndexedPassage(
            passage_id="NVDA-TABLE",
            company="NVIDIA",
            period="FY2026",
            section="Revenue by business",
            text=(
                "Data Center revenue was $193.7 billion with 68% growth; "
                "Gaming revenue was $16.0 billion with 41% growth."
            ),
            source_url="https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm",
            document_type="10-K teaching extract",
        ),
        IndexedPassage(
            passage_id="NVDA-CONCENTRATION",
            company="NVIDIA",
            period="FY2026",
            section="Concentration question",
            text="Data Center represented most of total revenue and reported expansion.",
            source_url="https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm",
            document_type="10-K teaching extract",
        ),
        IndexedPassage(
            passage_id="SU-TABLE",
            company="Schneider Electric",
            period="FY2025",
            section="Key financial metrics",
            text=(
                "Revenue was EUR 40.2bn with 8.9% organic growth; Energy Management "
                "grew 10% organically; adjusted EBITA was EUR 7.5bn at an 18.7% margin."
            ),
            source_url="https://www.se.com/ww/en/assets/564/document/528237/release-fy-results-2025.pdf",
            document_type="Full-year results teaching extract",
        ),
        IndexedPassage(
            passage_id="SU-PARSING",
            company="Schneider Electric",
            period="FY2025",
            section="Key financial metrics",
            text="A naive character split can separate EUR 40.2bn from Revenue.",
            source_url="https://www.se.com/ww/en/assets/564/document/528237/release-fy-results-2025.pdf",
            document_type="Full-year results teaching extract",
        ),
    )
def test_company_and_period_filters_block_cross_company_candidates(corpus):
    index = DenseIndex(
        corpus,
        DeterministicTeachingEmbeddings(),
        provider="offline",
        model="financial-concepts-v1",
        chunking_strategy="contextual-structure",
    )

    hits = index.search(
        "data centre energy demand growth",
        top_k=3,
        filters=RetrievalFilters(company="NVIDIA", period="FY2026"),
    )

    assert hits
    assert {hit.passage.company for hit in hits} == {"NVIDIA"}
    assert {hit.passage.period for hit in hits} == {"FY2026"}


def test_keyword_retrieval_preserves_an_exact_financial_figure(corpus):
    hits = KeywordIndex(corpus).search("18.7% margin", top_k=1)

    assert hits[0].passage.company == "Schneider Electric"
    assert "18.7%" in hits[0].passage.text


@pytest.mark.parametrize(
    "filters,expected_ids",
    [
        (RetrievalFilters(company="NVIDIA"), {"NVDA-TABLE", "NVDA-CONCENTRATION"}),
        (RetrievalFilters(period="FY2025"), {"SU-TABLE", "SU-PARSING"}),
        (
            RetrievalFilters(document_type="10-K teaching extract"),
            {"NVDA-TABLE", "NVDA-CONCENTRATION"},
        ),
        (RetrievalFilters(section="Concentration question"), {"NVDA-CONCENTRATION"}),
    ],
)
def test_each_supported_metadata_filter_is_enforced(corpus, filters, expected_ids):
    hits = KeywordIndex(corpus).search("revenue growth", top_k=10, filters=filters)

    assert {hit.passage.passage_id for hit in hits} == expected_ids


def test_equal_dense_scores_use_passage_id_as_stable_tie_break(corpus):
    index = DenseIndex(
        corpus,
        DeterministicTeachingEmbeddings(),
        provider="offline",
        model="financial-concepts-v1",
        chunking_strategy="contextual-structure",
    )

    hits = index.search("vocabulary outside the teaching concepts", top_k=len(corpus))

    tied_ids = [hit.passage.passage_id for hit in hits if hit.score == 0.5]
    assert tied_ids == sorted(tied_ids)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `.venv/bin/pytest tests/test_hybrid_retrieval.py -q`

Expected: collection fails because `finai_academy.hybrid_retrieval` does not exist.

- [ ] **Step 3: Implement the minimal types, embedder, and indexes**

Implement these exact public shapes:

```python
class EmbeddingModel(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class IndexedPassage(EvidencePassage):
    document_type: str = "financial disclosure"


@dataclass(frozen=True)
class RetrievalFilters:
    company: str | None = None
    period: str | None = None
    document_type: str | None = None
    section: str | None = None

    def matches(self, passage: IndexedPassage) -> bool: ...


class DeterministicTeachingEmbeddings:
    model_name = "financial-concepts-v1"
    dimension = 12

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class KeywordIndex:
    def __init__(self, passages: Sequence[IndexedPassage]) -> None: ...
    def search(
        self,
        query: str,
        top_k: int,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievalHit]: ...


class DenseIndex:
    def __init__(
        self,
        passages: Sequence[IndexedPassage],
        embeddings: EmbeddingModel,
        *,
        provider: str,
        model: str,
        chunking_strategy: str,
    ) -> None: ...

    def search(
        self,
        query: str,
        top_k: int,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievalHit]: ...

    def cosine_scores(
        self,
        query: str,
        filters: RetrievalFilters | None = None,
    ) -> list[tuple[IndexedPassage, float]]: ...
```

The offline embedder uses a documented concept vocabulary for company, period, revenue,
growth, margin, data center, gaming, energy management, and adjusted EBITA. Numeric
tokens intentionally receive no semantic dimension so the exact-number failure remains
observable. Normalize non-zero vectors before cosine similarity. `DenseIndex.cosine_scores`
exposes raw cosine values in `[-1, 1]` for teaching visuals; `search` maps them to
`(cosine + 1) / 2` so the reused `RetrievalHit` contract stays in `[0, 1]` without
changing rank order. Reject empty corpora, duplicate passage identifiers, empty queries,
non-finite vectors, inconsistent vector dimensions, and non-positive `top_k` values.
When fewer passages are eligible than `top_k`, return every eligible passage.
Normalize filter values with `casefold().strip()` and compare exact normalized metadata;
do not use substring matching. `IndexedPassage.__post_init__` calls the parent validation
and rejects an empty document type.

- [ ] **Step 4: Run focused tests and lint**

Run: `.venv/bin/pytest tests/test_hybrid_retrieval.py -q`

Run: `.venv/bin/ruff check src/finai_academy/hybrid_retrieval.py tests/test_hybrid_retrieval.py`

Expected: all index and filter tests pass with no Ruff findings.

- [ ] **Step 5: Commit the retrieval foundations**

```bash
git add src/finai_academy/hybrid_retrieval.py tests/conftest.py tests/test_hybrid_retrieval.py
git commit -m "feat: add filtered keyword and dense retrieval"
```

---

### Task 2: Versioned local embedding-index persistence

**Files:**
- Modify: `src/finai_academy/hybrid_retrieval.py`
- Modify: `tests/test_hybrid_retrieval.py`

**Interfaces:**
- Consumes: `IndexedPassage`, `EmbeddingModel`, and the document matrix already built by `DenseIndex`.
- Produces: `EmbeddingIndexVersion`, `IndexVersionError`, `DenseIndex.save(directory)`, and `DenseIndex.from_artifact(...)`.
- The artifact contains `manifest.json` and `vectors.npy`; it never stores credentials or model responses.

- [ ] **Step 1: Write persistence and mismatch tests**

```python
def build_offline_dense_index(corpus):
    embeddings = DeterministicTeachingEmbeddings()
    return DenseIndex(
        corpus,
        embeddings,
        provider="offline",
        model=embeddings.model_name,
        chunking_strategy="contextual-structure",
    )


def test_dense_index_round_trip_preserves_vectors_and_version(corpus, tmp_path):
    embeddings = DeterministicTeachingEmbeddings()
    original = DenseIndex(
        corpus,
        embeddings,
        provider="offline",
        model=embeddings.model_name,
        chunking_strategy="contextual-structure",
    )
    original.save(tmp_path)

    restored = DenseIndex.from_artifact(
        tmp_path,
        corpus,
        embeddings,
        expected_version=original.version,
    )

    assert np.array_equal(restored.document_matrix, original.document_matrix)
    assert restored.version == original.version


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("provider", "different-provider"),
        ("model", "different-model"),
        ("dimension", 99),
        ("corpus_hash", "0" * 64),
        ("chunking_strategy", "fixed"),
    ],
)
def test_loading_rejects_an_incompatible_index(corpus, tmp_path, field, replacement):
    index = build_offline_dense_index(corpus)
    index.save(tmp_path)
    incompatible = replace(index.version, **{field: replacement})

    with pytest.raises(IndexVersionError, match=field):
        DenseIndex.from_artifact(
            tmp_path,
            corpus,
            DeterministicTeachingEmbeddings(),
            expected_version=incompatible,
        )
```

- [ ] **Step 2: Run the persistence tests and verify RED**

Run: `.venv/bin/pytest tests/test_hybrid_retrieval.py -k 'round_trip or incompatible' -q`

Expected: failures report missing `save`, `from_artifact`, or index-version types.

- [ ] **Step 3: Implement stable version identity and persistence**

Implement:

```python
@dataclass(frozen=True)
class EmbeddingIndexVersion:
    schema_version: int
    provider: str
    model: str
    dimension: int
    corpus_hash: str
    chunking_strategy: str
    passage_ids: tuple[str, ...]


class IndexVersionError(ValueError):
    """Raised when persisted vectors do not describe the requested corpus."""
```

Build the corpus hash from ordered passage ID, company, period, document type, section,
text, and source URL values separated by an explicit delimiter. Serialize the dataclass
as JSON and vectors with `numpy.save`. Write only inside the exact directory passed by the
caller. `from_artifact` validates every identity field, manifest passage order, matrix
shape, finite values, and query-vector dimension before returning an index.

- [ ] **Step 4: Run regression tests and lint**

Run: `.venv/bin/pytest tests/test_hybrid_retrieval.py tests/test_retrieval.py -q`

Run: `.venv/bin/ruff check src/finai_academy/hybrid_retrieval.py tests/test_hybrid_retrieval.py`

Expected: persistence, dense retrieval, and Lesson 04 lexical retrieval remain green.

- [ ] **Step 5: Commit index versioning**

```bash
git add src/finai_academy/hybrid_retrieval.py tests/test_hybrid_retrieval.py
git commit -m "feat: version and persist local embedding indexes"
```

---

### Task 3: Reciprocal-rank fusion, transparent reranking, and orchestration

**Files:**
- Modify: `src/finai_academy/hybrid_retrieval.py`
- Create: `src/finai_academy/reranking.py`
- Create: `src/finai_academy/retrieval_pipeline.py`
- Modify: `tests/test_hybrid_retrieval.py`
- Consume: `tests/conftest.py`
- Create: `tests/test_reranking.py`
- Create: `tests/test_retrieval_pipeline.py`

**Interfaces:**
- Consumes: keyword and dense `list[RetrievalHit]` rankings plus eligible `IndexedPassage` records.
- Produces: `FusedHit` and `reciprocal_rank_fusion` in `hybrid_retrieval.py`; `RerankFeatures`, `RerankedHit`, and `rerank_candidates` in `reranking.py`; `RetrievalResult` and `retrieve_evidence` in `retrieval_pipeline.py`.
- `retrieve_evidence` filters before both indexes, fuses a wider candidate set, reranks it, and returns an explicit abstention reason when no passage is eligible.

- [ ] **Step 1: Write fusion tests that prove formula, deduplication, weighting, and ties**

```python
def test_rrf_deduplicates_and_preserves_channel_ranks(corpus):
    lexical = [
        RetrievalHit(passage=corpus[0], score=0.9),
        RetrievalHit(passage=corpus[1], score=0.8),
    ]
    dense = [
        RetrievalHit(passage=corpus[1], score=0.95),
        RetrievalHit(passage=corpus[2], score=0.7),
    ]

    fused = reciprocal_rank_fusion(
        {"keyword": lexical, "dense": dense},
        k=60,
        weights={"keyword": 1.0, "dense": 1.0},
    )

    shared = next(hit for hit in fused if hit.passage.passage_id == corpus[1].passage_id)
    assert dict(shared.channel_ranks) == {"dense": 1, "keyword": 2}
    assert shared.rrf_score == pytest.approx(1 / 61 + 1 / 62)
    assert len({hit.passage.passage_id for hit in fused}) == len(fused)


def test_rrf_weight_changes_the_controlled_top_result(corpus):
    rankings = {
        "keyword": [
            RetrievalHit(passage=corpus[0], score=0.9),
            RetrievalHit(passage=corpus[1], score=0.8),
        ],
        "dense": [
            RetrievalHit(passage=corpus[1], score=0.95),
            RetrievalHit(passage=corpus[0], score=0.7),
        ],
    }

    keyword_heavy = reciprocal_rank_fusion(
        rankings, k=60, weights={"keyword": 2.0, "dense": 1.0}
    )
    dense_heavy = reciprocal_rank_fusion(
        rankings, k=60, weights={"keyword": 1.0, "dense": 2.0}
    )

    assert keyword_heavy[0].passage.passage_id != dense_heavy[0].passage.passage_id


def test_rrf_equal_scores_use_passage_id_as_stable_tie_break(corpus):
    rankings = {
        "keyword": [RetrievalHit(passage=corpus[0], score=0.9)],
        "dense": [RetrievalHit(passage=corpus[1], score=0.9)],
    }

    fused = reciprocal_rank_fusion(rankings, k=60)

    assert [hit.passage.passage_id for hit in fused] == sorted(
        [corpus[0].passage_id, corpus[1].passage_id]
    )
```

- [ ] **Step 2: Write reranking and orchestration tests**

The global `corpus` fixture from `tests/conftest.py` is available in both new test
modules. Put the reranker test in `tests/test_reranking.py` and the abstention test in
`tests/test_retrieval_pipeline.py`.

```python
def test_reranker_rewards_exact_numeric_evidence(corpus):
    candidates = [
        FusedHit(
            passage=corpus[0],
            rrf_score=0.032,
            channel_ranks=(("dense", 1), ("keyword", 2)),
        ),
        FusedHit(
            passage=corpus[2],
            rrf_score=0.031,
            channel_ranks=(("dense", 2), ("keyword", 1)),
        ),
    ]

    reranked = rerank_candidates("What margin reached 18.7%?", candidates, top_k=1)

    assert reranked[0].passage.company == "Schneider Electric"
    assert reranked[0].features.numeric_coverage == 1.0
    assert "18.7%" in reranked[0].passage.text


def test_pipeline_abstains_instead_of_broadening_empty_filters(corpus):
    embeddings = DeterministicTeachingEmbeddings()
    keyword_index = KeywordIndex(corpus)
    dense_index = DenseIndex(
        corpus,
        embeddings,
        provider="offline",
        model=embeddings.model_name,
        chunking_strategy="contextual-structure",
    )
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
```

- [ ] **Step 3: Run the new tests and verify RED**

Run: `.venv/bin/pytest tests/test_hybrid_retrieval.py tests/test_reranking.py -q`

Expected: collection or attribute failures identify the missing fusion, reranking, and
orchestration interfaces.

- [ ] **Step 4: Implement RRF with stable identifiers**

Implement:

```python
@dataclass(frozen=True)
class FusedHit:
    passage: IndexedPassage
    rrf_score: float
    channel_ranks: tuple[tuple[str, int], ...]


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[RetrievalHit]],
    *,
    k: int = 60,
    weights: Mapping[str, float] | None = None,
) -> list[FusedHit]: ...
```

Reject empty rankings, non-positive `k`, unknown weight keys, negative weights, duplicate
passage IDs inside one channel, and inconsistent passage content for the same ID. Sort by
descending RRF score and then passage ID. Store channel ranks alphabetically for stable
serialization.

- [ ] **Step 5: Implement transparent reranking**

Implement in `reranking.py`:

```python
@dataclass(frozen=True)
class RerankFeatures:
    lexical_coverage: float
    numeric_coverage: float
    section_overlap: float
    metadata_eligibility: float
    fusion_signal: float


@dataclass(frozen=True)
class RerankedHit:
    passage: IndexedPassage
    score: float
    features: RerankFeatures
    fused_hit: FusedHit


def rerank_candidates(
    query: str,
    candidates: Sequence[FusedHit],
    *,
    top_k: int,
) -> list[RerankedHit]: ...
```

Extract exact numeric tokens with a documented regular expression, normalize feature
values to `[0, 1]`, and calculate the final score as a fixed weighted sum declared once at
module level. Sort by descending score, descending fusion score, then passage ID. Return
feature contributions without calling a model.

- [ ] **Step 6: Implement the complete result boundary**

Implement the orchestration boundary in `retrieval_pipeline.py` so it can import both
index/fusion and reranking types without either lower-level module importing it.

```python
@dataclass(frozen=True)
class RetrievalResult:
    query: str
    filters: RetrievalFilters
    keyword_hits: tuple[RetrievalHit, ...]
    dense_hits: tuple[RetrievalHit, ...]
    fused_hits: tuple[FusedHit, ...]
    reranked_hits: tuple[RerankedHit, ...]
    abstention_reason: str | None = None


def retrieve_evidence(
    query: str,
    *,
    keyword_index: KeywordIndex,
    dense_index: DenseIndex,
    filters: RetrievalFilters,
    candidate_k: int,
    final_k: int,
    weights: Mapping[str, float] | None = None,
) -> RetrievalResult: ...
```

Require `candidate_k >= final_k >= 1`. Check eligibility before index calls. Keep every
intermediate ranking visible in the result so Lesson 07 can trace the pipeline.

- [ ] **Step 7: Run focused and regression tests**

Run: `.venv/bin/pytest tests/test_hybrid_retrieval.py tests/test_reranking.py tests/test_retrieval_pipeline.py tests/test_retrieval.py -q`

Run: `.venv/bin/ruff check src/finai_academy/hybrid_retrieval.py src/finai_academy/reranking.py src/finai_academy/retrieval_pipeline.py tests/test_hybrid_retrieval.py tests/test_reranking.py tests/test_retrieval_pipeline.py`

Expected: every retrieval stage passes with deterministic ordering and no lint findings.

- [ ] **Step 8: Commit the hybrid pipeline**

```bash
git add src/finai_academy/hybrid_retrieval.py src/finai_academy/reranking.py src/finai_academy/retrieval_pipeline.py tests/test_hybrid_retrieval.py tests/test_reranking.py tests/test_retrieval_pipeline.py
git commit -m "feat: fuse and rerank financial evidence"
```

---

### Task 4: Visual guided notebook, instructor chapter, and execution contract

**Files:**
- Create: `notebooks/06_hybrid_retrieval.ipynb`
- Create: `chapters/06-hybrid-retrieval.md`
- Modify: `tests/test_notebook_contracts.py`
- Modify: `tests/test_course_manifest.py`

**Interfaces:**
- Consumes: Lesson 05 source manifest, parsers, `structure_aware_chunks`, `contextualize_chunks`, shared embedding gateway, and Task 1–3 retrieval interfaces.
- Produces: an executable notebook with final marker `PASS — hybrid retrieval laboratory verified` and a minute-by-minute instructor chapter.

- [ ] **Step 1: Add a failing Lesson 06 notebook execution contract**

First add these reusable output helpers above the lesson execution tests:

```python
def count_png_outputs(notebook) -> int:
    return sum(
        "image/png" in output.get("data", {})
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") in {"display_data", "execute_result"}
    )


def stream_text(notebook) -> str:
    return "".join(
        output.get("text", "")
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )
```

Then add the Lesson 06 contract:

```python
def test_hybrid_retrieval_notebook_offline_run_is_visual_and_verified(tmp_path):
    notebook_path = ROOT / "notebooks" / "06_hybrid_retrieval.ipynb"
    output_dir = tmp_path / "executed"
    result = subprocess.run(
        [
            sys.executable,
            str(EXECUTOR),
            str(notebook_path),
            "--mode",
            "offline",
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    executed = nbformat.read(output_dir / notebook_path.name, as_version=4)
    assert count_png_outputs(executed) >= 8
    output_text = stream_text(executed)
    assert "Dense exact-term failure reproduced" in output_text
    assert "Cross-company leakage blocked" in output_text
    assert "Hybrid retrieval improves maintained recall" in output_text
    assert "PASS — hybrid retrieval laboratory verified" in output_text
```

Add `test_implemented_lesson_six_assets_exist` to `tests/test_course_manifest.py` using
the same manifest-driven pattern as Lessons 04 and 05.

- [ ] **Step 2: Run the contract and verify RED**

Run: `.venv/bin/pytest tests/test_notebook_contracts.py -k hybrid_retrieval -q`

Expected: failure because `notebooks/06_hybrid_retrieval.ipynb` does not exist.

- [ ] **Step 3: Author the 30-minute source notebook**

Generate the notebook with unique cell IDs, cleared outputs, the standard seven required
headings, `expected_runtime_minutes: 30`, and the signature
`First Finance - Arnaud Demes`. Use `FINAI_LIVE_MODE` from the shared executor:

```python
live_mode = os.getenv("FINAI_LIVE_MODE", "0") == "1"
settings = Settings.from_environment()
if live_mode:
    problems = check_provider_configuration(settings)
    if problems:
        raise RuntimeError(" ".join(problems))
    embeddings = create_embeddings(settings)
    embedding_provider = settings.embedding_provider
    embedding_model = settings.embedding_model
else:
    embeddings = DeterministicTeachingEmbeddings()
    embedding_provider = "offline"
    embedding_model = embeddings.model_name
```

Rebuild the seven contextual structure-aware chunks directly from the versioned source
manifest. Adapt each chunk into `IndexedPassage` without dropping company, period,
document type, section, text, ID, or URL. Use the four maintained questions from Lesson
05 and a versioned expected-evidence mapping.

Create at least eight complete figures: pipeline, 2D projection, similarity heatmap,
ranked ladders, exact-term failure, leakage barrier, RRF contributions, rerank features,
and four-stage scorecard. Make score types explicit and never label cosine similarity as
confidence. The challenge changes one RRF weight and prints which question ranking moved.

Persist the temporary index under
`Path(os.getenv("FINAI_INDEX_DIR", tempfile.gettempdir())) / "finai-lesson06-index"`.
Do not embed an absolute user path in the notebook.

- [ ] **Step 4: Author the instructor chapter**

Write `chapters/06-hybrid-retrieval.md` with:

- the exact 15-minute deck and 30-minute notebook pacing;
- the expected result of every checkpoint and marker;
- explanations for cosine similarity, pre-filtering, RRF, and reranking;
- the controlled exact-number and cross-company failures;
- offline, Ollama, and OpenAI run commands;
- challenge solution and likely student mistakes;
- index-version debugging guidance; and
- the transition to Lesson 07 evaluation and tracing.

- [ ] **Step 5: Execute and inspect every notebook visual**

Run:

```bash
.venv/bin/python scripts/execute_notebooks.py \
  notebooks/06_hybrid_retrieval.ipynb \
  --mode offline \
  --output-dir /private/tmp/finai-lesson06-executed
```

Extract every `image/png` output to `/private/tmp/finai-lesson06-figures`. Inspect all
figures individually at full size. Fix clipped labels, misleading scales, hidden ties,
ambiguous score names, and unreadable table text.

- [ ] **Step 6: Validate the source notebook and focused contracts**

Run: `.venv/bin/python scripts/validate_notebooks.py notebooks/06_hybrid_retrieval.ipynb`

Run: `.venv/bin/pytest tests/test_notebook_contracts.py tests/test_course_manifest.py -q`

Expected: the source notebook is output-free, the offline execution emits at least eight
PNG outputs and all four markers, and Lesson 06 manifest paths exist.

- [ ] **Step 7: Commit the guided lesson**

```bash
git add notebooks/06_hybrid_retrieval.ipynb chapters/06-hybrid-retrieval.md tests/test_notebook_contracts.py tests/test_course_manifest.py
git commit -m "lesson: build the hybrid retrieval laboratory"
```

---

### Task 5: Seven-slide companion diagram deck

**Files:**
- Create: `decks/06-hybrid-retrieval.pptx`

**Interfaces:**
- Mirrors the notebook pipeline, score semantics, metadata barrier, worked RRF example, reranking boundary, and Lesson 07 handoff.

- [ ] **Step 1: Load the presentation environment and reference system**

Call `codex_app__load_workspace_dependencies`. Read the presentation skill's complete
`style_guidelines.md`, `references/template-following.md`,
`artifact_tool_docs/API_QUICK_START.md`, and `artifact_tool_docs/api/API_DOCS.md`.
Inspect every slide in `decks/05-document-and-chunking-lab.pptx` and reuse its First
Finance visual hierarchy, typography, palette, footer, and page-marker system.
Use the dependency loader's exact absolute values as command-scoped `RUNTIME_NODE`,
`RUNTIME_NODE_MODULES`, and `RUNTIME_BIN_DIR`. Create a `node_modules` symlink under
`$TMP_DIR` pointing to `RUNTIME_NODE_MODULES`; do not modify the dependency directory.

- [ ] **Step 2: Define the seven-slide narrative and source notes**

Create `$TMP_DIR/source-notes.txt` and plan these audience-facing slides:

1. session outcome and 15/30-minute format;
2. embedding geometry and cosine similarity;
3. complementary keyword and dense candidate lists;
4. company/period/document filters as an eligibility barrier;
5. one worked reciprocal-rank-fusion calculation;
6. retrieve widely, rerank narrowly; and
7. final capstone retrieval pipeline and Lesson 07 transition.

Use `[Sources]` speaker-note blocks on every slide. Cite scikit-learn cosine-similarity
documentation, the NVIDIA and Schneider official disclosures, and course-owned diagrams
where applicable.

- [ ] **Step 3: Mark and create the PowerPoint exactly once**

Immediately before the first authoring command, run the presentation marker once with:

```bash
"$RUNTIME_NODE" "$SKILL_DIR/container_tools/mark_artifact_operation_started.mjs" \
  --operation-kind create \
  --expected-output-count 1 \
  --output-format pptx
```

Build the deck from a JavaScript ES module under `$TMP_DIR` using
`@oai/artifact-tool`. Create connectors before nodes. Keep title text on one line, body
text at 16pt or above, mid-level labels at 24pt or above, and the title slide at 50pt or
above. Export only the final PPTX to `decks/06-hybrid-retrieval.pptx`.

- [ ] **Step 4: Render and inspect every slide**

Render the final deck with `container_tools/render_slides.py`, create a montage for flow,
then inspect all seven final slide PNGs individually at full size. Fix every unintended
overlap, clipped label, connector crossing, inconsistent footer, unreadable formula, and
unexpected title wrap.

- [ ] **Step 5: Run the presentation overflow test**

Run:

```bash
/Users/arnauddemes/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  "$SKILL_DIR/container_tools/slides_test.py" \
  decks/06-hybrid-retrieval.pptx
```

Expected: `Test passed. No overflow detected.`

- [ ] **Step 6: Commit the companion deck**

```bash
git add decks/06-hybrid-retrieval.pptx
git commit -m "content: add the hybrid retrieval diagram deck"
```

---

### Task 6: Full repository and provider-path verification

**Files:**
- Verify only; modify the smallest responsible source and its failing test if a defect is found.

**Interfaces:**
- Proves the complete Lesson 06 acceptance criteria without broadening its scope.

- [ ] **Step 1: Run the complete automated suite**

Run: `.venv/bin/pytest -q`

Expected: all tests pass with zero failures.

- [ ] **Step 2: Run lint and repository validation**

Run: `.venv/bin/ruff check .`

Run: `.venv/bin/python scripts/validate_repo.py`

Run:

```bash
.venv/bin/python scripts/validate_notebooks.py \
  notebooks/01_model_gateway.ipynb \
  notebooks/02_prompts_and_structured_outputs.ipynb \
  notebooks/03_cag_financial_document.ipynb \
  notebooks/04_rag_from_scratch.ipynb \
  notebooks/05_document_and_chunking_lab.ipynb \
  notebooks/06_hybrid_retrieval.ipynb
```

Run: `git diff --check`

Expected: Ruff, structure validation, six notebook contracts, and whitespace checks all
pass.

- [ ] **Step 3: Diagnose both live embedding paths without exposing credentials**

Print only whether `OPENAI_API_KEY` exists, whether `langchain_openai` and
`langchain_ollama` import, and whether Ollama responds at `/api/tags`. If configured, run:

```bash
.venv/bin/python scripts/execute_notebooks.py \
  notebooks/06_hybrid_retrieval.ipynb \
  --mode live \
  --provider ollama \
  --output-dir /private/tmp/finai-lesson06-ollama

.venv/bin/python scripts/execute_notebooks.py \
  notebooks/06_hybrid_retrieval.ipynb \
  --mode live \
  --provider openai \
  --output-dir /private/tmp/finai-lesson06-openai
```

If a provider is unavailable, record the missing prerequisite accurately and rely on the
already verified offline path; do not claim a live execution.

- [ ] **Step 4: Review the scoped diff and acceptance matrix**

Confirm each design requirement maps to a passing test, executed notebook marker,
inspected figure, or inspected slide. Review `git status --short` so Lesson 06 changes are
distinguished from pre-existing Lesson 02–05 work. Do not delete or overwrite unrelated
changes.
