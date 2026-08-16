# MLE-Inspired Alignment and Lesson 07 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align Lessons 01–06 on one build-observe-improve-verify teaching contract and add a complete MLflow-centered Lesson 07 for financial RAG evaluation, with Ragas as an optional final comparison.

**Architecture:** Lessons 01–06 continue to return ordinary typed Python records, tables, figures, and PASS markers without importing MLflow. Lesson 07 wraps the real Lesson 06 retrieval pipeline in MLflow traces, evaluates a versioned NVIDIA/Schneider golden set with deterministic metrics, compares configurations, and optionally runs explicitly configured Ragas judge metrics.

**Tech Stack:** Python 3.11, Jupyter, Pydantic v2, LangChain provider adapters, Ollama, OpenAI, scikit-learn, NumPy, pandas, `mlflow>=3.15,<4`, optional `ragas>=0.4.3,<0.5`, pytest, Ruff, `@oai/artifact-tool` for PowerPoint.

## Global Constraints

- Keep the exact lesson slots in `course.yml`.
- Every live notebook supports Ollama and OpenAI through shared provider boundaries.
- Every notebook has a deterministic offline execution path.
- Lessons 01–06 do not import, configure, or require MLflow.
- Lesson 07 uses a local MLflow store; Docker is not required.
- Ragas is optional and runs only after the first-principles metrics.
- Do not require Supabase, PostgreSQL, pgvector, Docling, a VLM, FlashRank, or HyDE for classroom PASS contracts.
- Preserve NVIDIA and Schneider evidence provenance, filtering, abstention, and non-advice boundaries.
- Source notebooks committed to Git contain no outputs.
- Deck footer is `First Finance - Arnaud Demes`.
- Use tests before implementation changes and make a focused commit after every task.

## File and interface map

- `src/finai_academy/providers.py` owns normalized chat and embedding provider boundaries.
- `src/finai_academy/measurement.py` will own reusable provider-neutral run metadata only; it will not import MLflow.
- `src/finai_academy/context.py` owns CAG budgeting and routing decisions.
- `src/finai_academy/retrieval.py` owns the naive lexical baseline.
- `src/finai_academy/chunking.py` owns chunk construction and provider-neutral semantic/contextual enrichment.
- `src/finai_academy/retrieval_pipeline.py` owns the staged Lesson 06 retrieval result and stage measurements.
- `src/finai_academy/evaluation.py` will own golden cases and deterministic Lesson 07 metrics.
- `src/finai_academy/mlflow_evaluation.py` will be the only course module that imports MLflow.
- `src/finai_academy/ragas_evaluation.py` will be an optional adapter with explicit judge configuration.
- `notebooks/01_model_gateway.ipynb` through `notebooks/07_rag_evaluation.ipynb` are the canonical student notebooks.
- `chapters/01-model-gateway.md` through `chapters/07-rag-evaluation.md` are the canonical instructor guides.
- `decks/01-model-gateway.pptx` through `decks/07-rag-evaluation.pptx` are the canonical concept decks.

---

### Task 1: Provider-neutral measurement foundation

**Files:**
- Create: `src/finai_academy/measurement.py`
- Modify: `src/finai_academy/providers.py`
- Create: `tests/test_measurement.py`
- Modify: `tests/test_providers.py`

**Interfaces:**
- Consumes: provider metadata already returned by LangChain adapters or recorded offline models.
- Produces: `TokenUsage`, `RunMeasurement`, a provider-neutral `StageObserver`, and an enriched `ModelRun` used by Lessons 01–04 and later logged by Lesson 07.

- [ ] **Step 1: Write failing measurement validation tests**

```python
from finai_academy.measurement import RunMeasurement, TokenUsage


def test_token_usage_requires_consistent_non_negative_counts():
    usage = TokenUsage(input_tokens=10, output_tokens=4, total_tokens=14)
    assert usage.total_tokens == usage.input_tokens + usage.output_tokens


def test_run_measurement_rejects_negative_duration():
    with pytest.raises(ValueError, match="duration_ms"):
        RunMeasurement(stage="generate", duration_ms=-0.1)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `.venv/bin/pytest tests/test_measurement.py tests/test_providers.py -q`

Expected: collection fails because `finai_academy.measurement` does not exist.

- [ ] **Step 3: Implement the minimal typed records**

```python
@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        values = (self.input_tokens, self.output_tokens, self.total_tokens)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
            raise ValueError("token counts must be non-negative integers")
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("total_tokens must equal input_tokens + output_tokens")


@dataclass(frozen=True)
class RunMeasurement:
    stage: str
    duration_ms: float
    token_usage: TokenUsage | None = None
    metadata: Mapping[str, str | int | float | bool | None] = field(default_factory=dict)
```

Add a `StageObserver` protocol exposing `span(name, *, inputs)` as a context manager and a `NullStageObserver` implementation. This lets Lesson 06 expose real stage boundaries while keeping MLflow outside Lessons 01–06. Add optional `token_usage: TokenUsage | None` and `prompt_version: str | None` fields to `ModelRun`. Normalize LangChain `usage_metadata` only when all three counts are present; otherwise retain `None` rather than inventing zero usage.

- [ ] **Step 4: Run focused tests and Ruff**

Run: `.venv/bin/pytest tests/test_measurement.py tests/test_providers.py -q`

Run: `.venv/bin/ruff check src/finai_academy/measurement.py src/finai_academy/providers.py tests/test_measurement.py tests/test_providers.py`

Expected: all focused checks pass.

- [ ] **Step 5: Commit the measurement foundation**

```bash
git add src/finai_academy/measurement.py src/finai_academy/providers.py tests/test_measurement.py tests/test_providers.py
git commit -m "feat: add provider-neutral run measurements"
```

### Task 2: Align Lesson 01 with the toolkit progression

**Files:**
- Modify: `notebooks/01_model_gateway.ipynb`
- Modify: `chapters/01-model-gateway.md`
- Modify: `decks/01-model-gateway.pptx`
- Modify: `tests/test_notebook_contracts.py`

**Interfaces:**
- Consumes: `ModelRun`, `RunMeasurement`, shared settings, Ollama/OpenAI adapters, and the recorded offline model.
- Produces: normalized latency/token output, one observable streaming example, a three-question knowledge check, and the unchanged provider-neutral capstone boundary.

- [ ] **Step 1: Add failing notebook contract assertions**

```python
def test_lesson_01_teaches_measurement_streaming_and_toolkit_roadmap():
    notebook = load_notebook("01_model_gateway.ipynb")
    text = notebook_text(notebook)
    assert "Token usage" in text
    assert "Streaming is a delivery pattern" in text
    assert "## Knowledge check" in text
    assert "MLflow" not in executable_code(notebook)
```

Also assert that the final executed output still contains exactly one `PASS — provider-neutral model gateway verified` marker.

- [ ] **Step 2: Run the contract test and verify RED**

Run: `.venv/bin/pytest tests/test_notebook_contracts.py -k lesson_01 -q`

Expected: failure because the new measurement, streaming, and knowledge-check content is absent.

- [ ] **Step 3: Revise the notebook and instructor chapter**

Use this timed notebook sequence:

```text
0–3 min   configuration and safe diagnostics
3–7 min   messages and first measured call
7–10 min  token/latency output
10–13 min short streaming demonstration
13–17 min ambiguous finance failure
17–19 min grounded NVIDIA response
19–20 min verification and knowledge check
```

The streaming cell must use the selected provider in live mode and a recorded iterator offline. It prints partial text but returns the same final answer contract. Add expected-output guidance and troubleshooting for unavailable token counts.

- [ ] **Step 4: Revise the deck in place**

Keep the existing visual template. Add or reflow content so the deck communicates:

1. model gateway boundary;
2. explicit message contract;
3. latency and tokens are application data;
4. streaming changes delivery, not factual quality;
5. ambiguous finance failure;
6. evidence-grounded answer;
7. toolkit roadmap from structured output to MCP.

Use the Presentations skill, preserve inherited shapes, include one `[Sources]` block per slide, render every slide, and run `slides_test.py`.

- [ ] **Step 5: Execute and verify Lesson 01**

Run offline:

```bash
.venv/bin/python scripts/execute_notebooks.py notebooks/01_model_gateway.ipynb --mode offline --output-dir /private/tmp/finai-l01-offline
```

Run Ollama with the configured class model. Run OpenAI when `OPENAI_API_KEY` is available. Verify no source outputs were committed and visually inspect every notebook figure/output and deck slide.

- [ ] **Step 6: Commit Lesson 01**

```bash
git add notebooks/01_model_gateway.ipynb chapters/01-model-gateway.md decks/01-model-gateway.pptx tests/test_notebook_contracts.py
git commit -m "lesson: align the model gateway progression"
```

### Task 3: Rebuild Lesson 02 around progressive prompt engineering

**Files:**
- Modify: `src/finai_academy/capstone/briefing.py`
- Modify: `notebooks/02_prompts_and_structured_outputs.ipynb`
- Modify: `chapters/02-prompts-and-structured-outputs.md`
- Modify: `decks/02-prompts-and-structured-outputs.pptx`
- Modify: `tests/test_capstone_briefing.py`
- Modify: `tests/test_notebook_contracts.py`

**Interfaces:**
- Consumes: the Lesson 01 gateway, NVIDIA evidence card, `AnalystBrief`, and provider-neutral structured model.
- Produces: named prompt stages, prompt/schema metadata, a few-shot insufficient-evidence example, and the existing validated analyst brief.

- [ ] **Step 1: Add failing prompt-stage tests**

```python
def test_prompt_contains_six_part_contract_and_delimited_source():
    prompt = build_analyst_brief_prompt(
        company="NVIDIA",
        reporting_period="FY2026",
        source_text="Revenue was USD 215.9bn.",
    )
    assert "<task>" in prompt
    assert "<context>" in prompt
    assert "<instructions>" in prompt
    assert "<source_document>" in prompt
    assert "<output_criteria>" in prompt
    assert "<example>" in prompt
    assert "Revenue was USD 215.9bn." in prompt
```

Expose `build_analyst_brief_prompt(company: str, reporting_period: str, source_text: str) -> str` and `PROMPT_VERSION` from `briefing.py`. Keep trusted company and period overwrite behavior in `AnalystBriefService`.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `.venv/bin/pytest tests/test_capstone_briefing.py tests/test_notebook_contracts.py -k 'prompt or lesson_02' -q`

Expected: the builder and notebook stages are missing.

- [ ] **Step 3: Implement the staged prompt builder and notebook comparison**

The notebook must execute these stages over the same source:

```text
vague -> six_part -> delimited -> few_shot -> prompt_json -> schema_bound
```

Store a row per stage with `stage`, `prompt_version`, `valid_json`, `valid_schema`, `finance_accepted`, and `failure_reason`. Use recorded candidates offline so the comparison is deterministic. Make only the final schema-bound stage call the live structured model.

- [ ] **Step 4: Update slides and chapter**

The deck must include the six-part framework, the progression ladder, prompt injection through untrusted financial text, few-shot treatment of insufficient evidence, and the existing Pydantic/finance acceptance layers. The chapter provides exact 10/20 pacing and answer keys.

- [ ] **Step 5: Verify all provider modes and commit**

Run the focused tests, Ruff, offline notebook execution, Ollama, available OpenAI, deck render, overflow, template fidelity, and source-note audits.

```bash
git add src/finai_academy/capstone/briefing.py tests/test_capstone_briefing.py tests/test_notebook_contracts.py notebooks/02_prompts_and_structured_outputs.ipynb chapters/02-prompts-and-structured-outputs.md decks/02-prompts-and-structured-outputs.pptx
git commit -m "lesson: add progressive financial prompt engineering"
```

### Task 4: Apply focused alignment changes to Lessons 03 and 04

**Files:**
- Modify: `src/finai_academy/context.py`
- Modify: `notebooks/03_cag_financial_document.ipynb`
- Modify: `notebooks/04_rag_from_scratch.ipynb`
- Modify: `chapters/03-cag-financial-document.md`
- Modify: `chapters/04-rag-from-scratch.md`
- Modify: `decks/03-cag-financial-document.pptx`
- Modify: `decks/04-rag-from-scratch.pptx`
- Modify: `tests/test_context.py`
- Modify: `tests/test_notebook_contracts.py`

**Interfaces:**
- Consumes: existing CAG budget functions, lexical retriever, recorded/live models, and run measurements.
- Produces: `ContextDecision`, no-context/full-context/RAG comparison, and separately timed naive RAG stages without MLflow.

- [ ] **Step 1: Write the failing `ContextDecision` tests**

```python
def test_context_decision_records_route_and_budget_reason():
    decision = decide_context_route(
        document_tokens=7000,
        system_prompt_tokens=400,
        question_tokens=100,
        budget=ContextBudget(max_input_tokens=8192, reserved_output_tokens=1200),
    )
    assert decision.route == "rag"
    assert decision.estimated_input_tokens == 7500
    assert decision.available_input_tokens == 6992
    assert "exceeds" in decision.reason
```

Implement `ContextDecision` with `route: Literal["cag", "rag"]`, exact token fields, and a non-empty reason. Make `should_use_full_context` delegate to the new decision function so existing callers stay compatible.

- [ ] **Step 2: Add failing notebook contract assertions**

Assert Lesson 03 contains the five-way context/cache/memory/grounding/RAG comparison and Lesson 04 contains the no-context/full-context/RAG comparison plus one naive paragraph split. Assert neither notebook code imports MLflow.

- [ ] **Step 3: Implement the focused notebook and chapter changes**

Do not expand the timed scope. Lesson 03 adds one decision table and returns `ContextDecision`. Lesson 04 adds a recorded no-context answer and a full-context reference before executing the existing RAG pipeline. Add common knowledge checks and expected output blocks.

- [ ] **Step 4: Update the two decks in place**

Lesson 03 gains a context/cache/memory/grounding/RAG distinction. Lesson 04 gains the three-path comparison and preserves the strong current TF-IDF, top-k, verification, and roadmap slides.

- [ ] **Step 5: Verify and commit each lesson separately**

```bash
git add src/finai_academy/context.py tests/test_context.py tests/test_notebook_contracts.py notebooks/03_cag_financial_document.ipynb chapters/03-cag-financial-document.md decks/03-cag-financial-document.pptx
git commit -m "lesson: clarify the complete-context decision"
```

```bash
git add tests/test_notebook_contracts.py notebooks/04_rag_from_scratch.ipynb chapters/04-rag-from-scratch.md decks/04-rag-from-scratch.pptx
git commit -m "lesson: compare naive RAG with context baselines"
```

### Task 5: Add real semantic and LLM-contextual chunking

**Files:**
- Modify: `src/finai_academy/chunking.py`
- Modify: `src/finai_academy/providers.py`
- Modify: `tests/test_chunking.py`
- Modify: `tests/test_providers.py`

**Interfaces:**
- Consumes: `DocumentBlock`, `DocumentChunk`, provider embeddings, provider chat models, and recorded offline responses.
- Produces: `embedding_similarity_profile`, `contextual_enrich_chunks`, preserved `raw_text`, separate generated context, token/cost measurements, and complete provenance.

- [ ] **Step 1: Add failing semantic-embedding tests**

```python
def test_embedding_similarity_profile_uses_adjacent_sentence_vectors():
    model = RecordedEmbeddings([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])
    sentences, similarities = embedding_similarity_profile(blocks, model)
    assert len(sentences) == 3
    assert len(similarities) == 2
    assert similarities[0] > similarities[1]
```

Validate non-empty, finite, equal-dimension vectors and normalize safely before cosine calculation.

- [ ] **Step 2: Add failing contextual-enrichment tests**

```python
def test_contextual_enrichment_preserves_source_and_separates_generated_context():
    enriched = contextual_enrich_chunks(
        document_text="NVIDIA FY2026 filing excerpt",
        chunks=[source_chunk],
        model=RecordedChunkingModel({source_chunk.chunk_id: "This passage covers Data Center revenue."}),
    )
    assert enriched[0].raw_text == source_chunk.text
    assert enriched[0].generated_context == "This passage covers Data Center revenue."
    assert enriched[0].text.endswith(source_chunk.text)
    assert enriched[0].source_block_ids == source_chunk.source_block_ids
```

Add `generated_context: str | None = None` to `DocumentChunk`. The generated context must never replace `raw_text` or provenance fields.

- [ ] **Step 3: Run focused tests and verify RED**

Run: `.venv/bin/pytest tests/test_chunking.py tests/test_providers.py -q`

Expected: missing semantic and contextual-enrichment interfaces.

- [ ] **Step 4: Implement semantic and enrichment boundaries**

Use these exact public signatures:

- `embedding_similarity_profile(blocks: Sequence[DocumentBlock], embeddings: EmbeddingModel) -> tuple[list[str], list[float]]`
- `contextual_enrich_chunks(*, document_text: str, chunks: Sequence[DocumentChunk], model: ChunkingModel) -> list[DocumentChunk]`

The model prompt returns JSON with one `context` string. Validate JSON, reject empty context, preserve the original chunk, and set retrieval text to `generated_context + "\n\n" + raw_text`. Recorded offline output is keyed by stable chunk ID. Live models are selected by shared settings.

- [ ] **Step 5: Add cost and provenance regression tests**

Test token inflation from raw versus enriched retrieval text, deterministic chunk IDs, unchanged page and block IDs, and refusal of malformed live/recorded JSON.

- [ ] **Step 6: Run focused tests, Ruff, and commit**

```bash
git add src/finai_academy/chunking.py src/finai_academy/providers.py tests/test_chunking.py tests/test_providers.py
git commit -m "feat: add LLM-aware contextual chunking"
```

### Task 6: Rebuild Lesson 05 around the MLE-aligned chunking progression

**Files:**
- Modify: `notebooks/05_document_and_chunking_lab.ipynb`
- Modify: `chapters/05-document-and-chunking-lab.md`
- Modify: `decks/05-document-and-chunking-lab.pptx`
- Modify: `tests/test_notebook_contracts.py`

**Interfaces:**
- Consumes: Task 5 semantic and contextual chunking, real course fixtures, and the Lesson 04 lexical evaluator.
- Produces: parser ladder, structure/semantic/hierarchy/enrichment comparison, token inflation figure, retrieval comparison, and one provider-neutral PASS contract.

- [ ] **Step 1: Add failing Lesson 05 contracts**

Assert the notebook contains and executes markers for parser quality, provider-aware semantic boundaries, generated contextual enrichment, preserved raw text, token inflation, and retrieval comparison. Assert `proposition_chunks` is labelled optional and contextual enrichment is not called agentic chunking.

- [ ] **Step 2: Run the focused contract test and verify RED**

Run: `.venv/bin/pytest tests/test_notebook_contracts.py -k lesson_05 -q`

- [ ] **Step 3: Reorder the notebook to the exact 70-minute core path**

```text
0–12   PDF/HTML extraction and visual quality failure
12–20  canonical DocumentBlock boundary
20–30  fixed, recursive, and structure-aware chunks
30–40  provider-aware semantic boundaries
40–48  hierarchical parent-child chunks
48–58  deterministic prefix vs LLM contextual enrichment
58–65  tokens, latency, construction, and retrieval comparison
65–70  verification, knowledge check, and capstone handoff
```

Docling/VLM parsing, proposition chunking, and iterative agentic grouping are optional extensions after the core PASS marker.

- [ ] **Step 4: Update the chapter and deck**

Add a parser ladder, raw versus enriched chunk diagram, token-inflation visualization, and a decision matrix for structure-aware, semantic, hierarchical, and LLM contextual enrichment. Preserve the existing strong table/provenance slides.

- [ ] **Step 5: Execute all modes and visually inspect**

Offline must use versioned recorded embeddings and contexts. Ollama and OpenAI must use their configured chat and embedding models. Live ranking is observational; provenance and structural checks remain provider-invariant.

- [ ] **Step 6: Commit Lesson 05**

```bash
git add notebooks/05_document_and_chunking_lab.ipynb chapters/05-document-and-chunking-lab.md decks/05-document-and-chunking-lab.pptx tests/test_notebook_contracts.py
git commit -m "lesson: teach contextual financial chunking"
```

### Task 7: Add Lesson 06 measurements and production bridges

**Files:**
- Modify: `src/finai_academy/retrieval_pipeline.py`
- Modify: `tests/test_retrieval_pipeline.py`
- Modify: `notebooks/06_hybrid_retrieval.ipynb`
- Modify: `chapters/06-hybrid-retrieval.md`
- Modify: `decks/06-hybrid-retrieval.pptx`
- Modify: `tests/test_notebook_contracts.py`

**Interfaces:**
- Consumes: current `RetrievalResult`, `RunMeasurement`, persisted indexes, RRF, and transparent reranker.
- Produces: exact stage measurements, local-to-pgvector architecture bridge, optional HyDE/FlashRank discussion, and a trace handoff consumed by Lesson 07.

- [ ] **Step 1: Add failing stage-measurement tests**

```python
def test_retrieval_result_exposes_every_stage_measurement(retrieval_dependencies):
    result = retrieve_evidence(**retrieval_dependencies)
    assert tuple(result.stage_measurements) == (
        "eligibility",
        "keyword",
        "dense",
        "fusion",
        "rerank",
    )
    assert all(item.duration_ms >= 0 for item in result.stage_measurements.values())
```

Add `stage_measurements: Mapping[str, RunMeasurement]` to `RetrievalResult` and an optional `observer: StageObserver | None = None` parameter to `retrieve_evidence`. Wrap the actual eligibility, keyword, dense, fusion, and rerank work in observer spans and measure the same boundaries with `perf_counter`; do not import MLflow.

- [ ] **Step 2: Run focused tests and implement GREEN**

Run: `.venv/bin/pytest tests/test_retrieval_pipeline.py -q`

Implement exact measurements for normal and abstention paths. The abstention path records eligibility and zero-duration skipped-stage records with `metadata={"status": "skipped"}` rather than inventing hits.

- [ ] **Step 3: Update notebook, chapter, and deck**

Keep the current 15/30 core. Add one stage-timing figure, one local index versus pgvector/HNSW bridge, a documents/chunks/embeddings schema, and a final mapping from stage objects to future MLflow spans. FlashRank and HyDE remain optional and cannot affect the PASS marker.

- [ ] **Step 4: Verify Lesson 06 regression and commit**

Run all existing Lesson 06 unit and notebook tests, offline and Ollama execution, available OpenAI execution, nine-or-more figure visual QA, deck QA, full suite, and repo validator.

```bash
git add src/finai_academy/retrieval_pipeline.py tests/test_retrieval_pipeline.py tests/test_notebook_contracts.py notebooks/06_hybrid_retrieval.ipynb chapters/06-hybrid-retrieval.md decks/06-hybrid-retrieval.pptx
git commit -m "lesson: prepare hybrid retrieval for tracing"
```

### Task 8: Build deterministic RAG evaluation primitives

**Files:**
- Create: `src/finai_academy/evaluation.py`
- Create: `tests/test_evaluation.py`
- Modify: `assets/course-data/manifest.json`
- Create: `assets/course-data/evaluation/rag_cases_v1.json`

**Interfaces:**
- Consumes: `RetrievalResult`, maintained evidence IDs, NVIDIA/Schneider manifest metadata, and answer/citation text.
- Produces: `EvaluationCase`, `CaseEvaluation`, `EvaluationSummary`, deterministic metric functions, and a versioned JSON dataset.

- [ ] **Step 1: Write failing dataset-model tests**

```python
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
```

Define frozen dataclasses with unique, normalized IDs and immutable tuples. Dataset loading rejects duplicate case IDs and unknown source IDs.

- [ ] **Step 2: Write failing metric tests**

```python
def test_metrics_keep_retrieval_and_answer_quality_separate():
    result = evaluate_case(case, retrieval_result, answer)
    assert result.retrieval_recall_at_k == 1.0
    assert result.citation_correctness == 0.0
    assert result.grounded_fact_coverage == 0.5
```

Test recall@k, reciprocal rank, filter correctness, citation correctness, grounded-fact coverage, abstention correctness, finite values, and insufficient-evidence behavior.

- [ ] **Step 3: Implement the minimal evaluation API**

Use these exact public signatures:

- `load_evaluation_cases(path: Path, *, known_evidence_ids: Collection[str])` returns an immutable tuple of `EvaluationCase` values.
- `evaluate_case(case: EvaluationCase, retrieval: RetrievalResult, answer: str) -> CaseEvaluation`
- `summarize_evaluations(results: Sequence[CaseEvaluation]) -> EvaluationSummary`

`EvaluationCase` contains `case_id`, `question`, `filters`, `expected_evidence_ids`, `expected_facts`, `requires_abstention`, and `tags`. `CaseEvaluation` contains the case ID, retrieved IDs, parsed citations, recall@k, reciprocal rank, filter correctness, citation correctness, grounded-fact coverage, abstention correctness, and one normalized failure stage. `EvaluationSummary` contains the case count and the arithmetic mean of every numeric metric.

Citation correctness parses stable passage IDs only. Grounded-fact coverage checks maintained fact tokens or normalized numeric phrases; it does not use an LLM judge.

- [ ] **Step 4: Create the versioned golden set**

Create at least eight cases: direct fact, exact number, semantic paraphrase, NVIDIA filter, Schneider filter, multi-evidence, insufficient evidence, and controlled cross-company leakage. Register the dataset hash and version in the manifest.

- [ ] **Step 5: Run tests, manifest validation, Ruff, and commit**

```bash
git add src/finai_academy/evaluation.py tests/test_evaluation.py assets/course-data/manifest.json assets/course-data/evaluation/rag_cases_v1.json
git commit -m "feat: add a versioned financial RAG evaluation set"
```

### Task 9: Add MLflow experiment and trace integration for Lesson 07

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `src/finai_academy/mlflow_evaluation.py`
- Create: `tests/test_mlflow_evaluation.py`

**Interfaces:**
- Consumes: Task 7 `RetrievalResult.stage_measurements`, Task 8 evaluation cases/results, prompt/index/provider metadata.
- Produces: a local MLflow experiment, per-case traces, logged aggregate metrics, configuration runs, and an inline-comparable run summary.

- [ ] **Step 1: Add the optional evaluation dependencies**

Add an `evaluation` optional dependency group containing `mlflow>=3.15,<4` and `ragas>=0.4.3,<0.5`. Keep Ragas importable only through the optional adapter. Regenerate `uv.lock` with the bundled `uv` workflow already used by the repository.

- [ ] **Step 2: Write failing local-store and metadata tests**

```python
def test_mlflow_run_logs_complete_reproducibility_metadata(tmp_path):
    summary = run_mlflow_evaluation(
        tracking_path=tmp_path / "mlruns",
        experiment_name="lesson-07-test",
        configuration=config,
        cases=cases,
        predict_fn=predict_fn,
    )
    assert summary.parameters["dataset_version"] == "rag-cases-v1"
    assert summary.parameters["index_version"]
    assert summary.parameters["prompt_version"]
    assert summary.trace_count == len(cases)
```

Tests use a temporary local tracking directory and the deterministic offline pipeline. They do not start a server or open the UI.

- [ ] **Step 3: Implement one explicit tracing boundary**

`mlflow_evaluation.py` is the only shared module allowed to import MLflow. Implement:

```python
@dataclass(frozen=True)
class EvaluationConfiguration:
    configuration_id: str
    dataset_version: str
    provider: str
    chat_model: str
    embedding_model: str
    index_version: str
    prompt_version: str
    candidate_k: int
    final_k: int
    rrf_weights: Mapping[str, float]


def run_mlflow_evaluation(
    *,
    tracking_path: Path,
    experiment_name: str,
    configuration: EvaluationConfiguration,
    cases: Sequence[EvaluationCase],
    predict_fn: EvaluationPredictor,
) -> MLflowEvaluationSummary:
    """Trace every case, log reproducibility metadata, and return the run summary."""
```

Define `EvaluationPrediction` with `retrieval`, `answer`, and `contexts`; `EvaluationPredictor` receives the case and a `StageObserver`, then returns that record. Define `MLflowEvaluationSummary` with `run_id`, `parameters`, `metrics`, `trace_count`, and `failure_rows`. Implement `MLflowStageObserver` so the actual Lesson 06 work generates eligibility, keyword, dense, fusion, and rerank spans; the predictor uses the same observer for context and generation spans. Log inputs/outputs, durations, eligible company/period, result IDs, and scores without secrets or credentials.

- [ ] **Step 4: Add comparison and failure-table tests**

Run two deterministic configurations and assert stable run IDs, exact case alignment, aggregate metric comparison, and one classified failure. The output table columns are `case_id`, `configuration_id`, `failure_stage`, `expected_ids`, `retrieved_ids`, `citations`, and metric values.

- [ ] **Step 5: Run focused tests and commit**

```bash
git add pyproject.toml uv.lock src/finai_academy/mlflow_evaluation.py tests/test_mlflow_evaluation.py
git commit -m "feat: trace and compare financial RAG evaluations"
```

### Task 10: Add the optional explicit Ragas adapter

**Files:**
- Create: `src/finai_academy/ragas_evaluation.py`
- Create: `tests/test_ragas_evaluation.py`

**Interfaces:**
- Consumes: the same evaluation cases, predictions, contexts, and an explicit Ollama/OpenAI judge adapter.
- Produces: context-recall and faithfulness results suitable for logging into the Lesson 07 MLflow run.

- [ ] **Step 1: Write failing explicit-judge tests**

```python
def test_ragas_adapter_never_selects_a_default_judge():
    with pytest.raises(ValueError, match="judge"):
        evaluate_with_ragas(rows, judge=None)


def test_offline_mode_returns_recorded_or_skipped_metrics():
    result = evaluate_with_ragas(rows, judge=RecordedRagasJudge(metrics={}))
    assert result.status == "recorded_or_skipped"
```

- [ ] **Step 2: Implement the narrow adapter**

Expose only `evaluate_with_ragas(rows: Sequence[RagasEvaluationRow], *, judge: RagasJudge) -> RagasEvaluationResult`. `RagasEvaluationRow` contains the case ID, user input, retrieved contexts, response, and reference answer. `RagasJudge` is a protocol with explicit `provider`, `model`, and `evaluate(rows)` members. `RagasEvaluationResult` contains `status`, judge metadata, per-case context recall, per-case faithfulness, and aggregate means.

Convert the course rows to the currently supported Ragas evaluation dataset, run only context recall and faithfulness, record judge provider/model, and return explicit skipped status when offline judge outputs are unavailable. Do not let Ragas initialize a default model.

- [ ] **Step 3: Run focused tests and commit**

```bash
git add src/finai_academy/ragas_evaluation.py tests/test_ragas_evaluation.py
git commit -m "feat: compare RAG metrics with Ragas"
```

### Task 11: Build the complete Lesson 07 teaching assets

**Files:**
- Create: `notebooks/07_rag_evaluation.ipynb`
- Create: `chapters/07-rag-evaluation.md`
- Create: `decks/07-rag-evaluation.pptx`
- Modify: `course.yml`
- Modify: `tests/test_notebook_contracts.py`
- Modify: `tests/test_course_manifest.py`

**Interfaces:**
- Consumes: real Lesson 06 retrieval pipeline, versioned golden set, deterministic metrics, MLflow integration, optional Ragas adapter, Ollama/OpenAI gateways.
- Produces: one 45-minute lesson with 15-minute visual deck, 30-minute notebook, complete traces, two compared configurations, failure classification, and final PASS marker.

- [ ] **Step 1: Write failing Lesson 07 asset contracts**

Assert the course manifest points to all three assets, the notebook has unique cells and no outputs, and its source contains the golden-set, metric, trace, comparison, failure, MLflow, optional Ragas, knowledge-check, and capstone sections.

- [ ] **Step 2: Capture the RED contract**

Run: `.venv/bin/pytest tests/test_course_manifest.py tests/test_notebook_contracts.py -k lesson_07 -q`

Expected: asset paths are missing.

- [ ] **Step 3: Author the 30-minute notebook**

Use this exact sequence:

```text
0–4    load and inspect the versioned golden set
4–9    run one baseline case and separate retrieval/answer metrics
9–14   run the full deterministic metric suite
14–20  configure local MLflow and inspect one complete trace
20–25  compare two retrieval configurations
25–28  classify and explain one failed case
28–30  verify, knowledge check, capstone handoff
optional after PASS: Ragas context recall and faithfulness
```

The notebook displays inline equivalents of the essential MLflow trace and run comparison, so the PASS contract does not depend on a browser UI. Offline results are exact. Live provider metrics are labelled observations.

- [ ] **Step 4: Author the instructor chapter**

Include exact 15/30 pacing, setup commands, local tracking-path behavior, MLflow UI demonstration commands, deterministic metric formulas, judge limitations, expected outputs, common failures, checkpoint answers, challenge solution, provider modes, and transition to Day 2 LangGraph.

- [ ] **Step 5: Author the seven-slide concept deck**

Use the existing course template and these narrative jobs:

1. Evaluation locates the failing stage.
2. The golden set is a versioned engineering artifact.
3. Retrieval and answer quality are separate axes.
4. Deterministic metrics make the baseline reproducible.
5. A trace connects inputs, stages, outputs, and timing.
6. MLflow compares configurations, not isolated anecdotes.
7. Ragas adds convenient RAG judges after the fundamentals.

Every slide needs sources in speaker notes and the course footer. Render and inspect all slides full-size, then run overflow, template-fidelity, placeholder, source-note, font, and no-shrink audits.

- [ ] **Step 6: Execute offline, Ollama, and OpenAI**

Offline:

```bash
.venv/bin/python scripts/execute_notebooks.py notebooks/07_rag_evaluation.ipynb --mode offline --output-dir /private/tmp/finai-l07-offline
```

Ollama uses explicit chat and embedding models. OpenAI runs only when the key is available. The source notebook remains output-free after executions.

- [ ] **Step 7: Commit Lesson 07**

```bash
git add notebooks/07_rag_evaluation.ipynb chapters/07-rag-evaluation.md decks/07-rag-evaluation.pptx course.yml tests/test_notebook_contracts.py tests/test_course_manifest.py
git commit -m "lesson: build RAG evaluation and tracing"
```

### Task 12: Complete seven-lesson regression and grading

**Files:**
- Modify only files required by confirmed regression findings.
- Create: `docs/reviews/lessons-01-07-alignment-review.md`

**Interfaces:**
- Consumes: every asset and test from Tasks 1–11.
- Produces: a clean branch, complete verification evidence, per-lesson grade, and no unresolved blocker.

- [ ] **Step 1: Run all static and unit gates**

```bash
.venv/bin/ruff check .
.venv/bin/pytest -q
.venv/bin/python scripts/validate_repo.py
.venv/bin/python scripts/validate_notebooks.py
git diff --check
```

- [ ] **Step 2: Execute all seven notebooks offline**

Use `scripts/execute_notebooks.py` for each canonical notebook and require one exact final PASS marker per notebook. Extract and inspect every generated figure full-size.

- [ ] **Step 3: Execute affected notebooks live**

Run Ollama for Lessons 01–07. Run OpenAI when a key is available. Record provider, model, success marker, figures, and any provider-dependent observations without weakening provider-invariant contracts.

- [ ] **Step 4: Run complete slide QA**

Render all seven decks. Inspect every slide individually and as a contact sheet. Run overflow, template fidelity, placeholder, sources, footer, font, and no-shrink audits.

- [ ] **Step 5: Perform the beginner-readability review**

For each lesson, score 0–10 on prerequisites, objective clarity, visual explanation, expected outputs, failure diagnosis, challenge scope, finance continuity, provider neutrality, verification, and timing. Require at least 9.5/10 overall and document any remaining elective limitation.

- [ ] **Step 6: Fix only confirmed findings and rerun the affected gates**

Use test-driven fixes for code or contract findings and re-render any changed deck/notebook visual. Do not refactor unrelated legacy files.

- [ ] **Step 7: Commit the final review**

```bash
git add docs/reviews/lessons-01-07-alignment-review.md
git commit -m "docs: grade the aligned first-day curriculum"
```

## Plan self-review record

- Spec coverage: all global constraints and every Lesson 01–07 requirement map to Tasks 1–12.
- Scope: legacy duplicate cleanup remains a separate reviewed decision and is not mixed into lesson implementation.
- Type consistency: `TokenUsage`, `RunMeasurement`, `ContextDecision`, `DocumentChunk.generated_context`, `RetrievalResult.stage_measurements`, `EvaluationCase`, and `EvaluationConfiguration` retain one exact name across producer and consumer tasks.
- Provider boundary: offline, Ollama, and OpenAI behavior is explicit for every new LLM/embedding/judge path.
- MLflow boundary: only `mlflow_evaluation.py` and Lesson 07 import MLflow.
- Ragas boundary: only `ragas_evaluation.py` imports Ragas and requires an explicit judge.
- Placeholder scan: the plan contains no unspecified implementation placeholders or deferred core requirements.
