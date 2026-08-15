# FinAI Academy Two-Day Course Build and Class-Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, execute, and rehearse the complete two-day “AI Engineering for Asset Management” course, including the introduction deck, twelve detailed guided notebooks, the shared Financial Analyst Copilot, and verified Ollama/OpenAI execution paths.

**Architecture:** The notebooks are the teaching surface, while reusable logic lives in `src/finai_academy`. Every notebook imports one provider-neutral model and embedding boundary, uses deterministic local fixtures for automated tests, and supports an explicit live run with either Ollama or OpenAI. The capstone accumulates the same components taught in the notebooks; slide decks explain the mental model and failure mode without duplicating notebook code.

**Tech Stack:** Python 3.11+, `uv`, Jupyter/nbclient, Pydantic 2, LangChain provider adapters, Ollama, OpenAI API, scikit-learn, `pdfplumber`, LangGraph, FastAPI, Streamlit, Tavily, yfinance, MCP Python SDK, pytest, Ruff, PowerPoint generated with `@oai/artifact-tool`.

## Global Constraints

- Every LLM-dependent notebook must run unchanged with `FINAI_MODEL_PROVIDER=ollama` or `FINAI_MODEL_PROVIDER=openai`.
- Every embedding-dependent notebook must run unchanged with `FINAI_EMBEDDING_PROVIDER=ollama` or `FINAI_EMBEDDING_PROVIDER=openai`.
- Default local chat model: `qwen3:8b`; default local embedding model: `qwen3-embedding:0.6b`.
- Default hosted chat model: `gpt-4.1-mini`; default hosted embedding model: `text-embedding-3-small`.
- Model identifiers remain overrideable through environment variables.
- No API key, secret, generated credential, or private financial document is committed.
- Unit and notebook-contract tests run without network access, an API key, or an Ollama daemon.
- Live smoke tests are explicit and provider-specific; they never run inside the default test suite.
- NVIDIA is the principal US case and Schneider Electric is the principal European case.
- Claims derived from documents retain company, reporting period, document type, source URL, page or section, and retrieval identifier.
- The application separates reported facts, calculations, management claims, external facts, interpretations, and open questions.
- The core course uses one bounded agent; multi-agent work is an optional extension, not a live build requirement.
- Live course hours are 09:00-17:00 with lunch 12:00-13:30 and 15-minute morning and afternoon breaks.
- Each notebook contains: learning objectives, capstone position, concepts, guided build, failure lab, improvement, verification, challenge, integration checkpoint, and recap.
- Each technical block presents five to seven slides, including shared architecture and progress slides reused from the introduction deck; repeated shared slides are not replayed, keeping the live sequence compact.
- Visible learner materials are in English; instructor implementation notes may be in English or French.

---

## Target Repository Map

```text
finai-academy/
├── assets/
│   ├── brand/finai-academy-style.md
│   └── course-data/
│       ├── manifest.json
│       ├── nvidia/
│       └── schneider/
├── chapters/
│   ├── 00-course-introduction.md
│   └── 01-... through 12-...
├── decks/
│   ├── 00-course-introduction.pptx
│   └── 01-... through 12-...
├── notebooks/
│   ├── 01_model_gateway.ipynb
│   ├── 02_prompts_and_structured_outputs.ipynb
│   ├── 03_cag_financial_document.ipynb
│   ├── 04_rag_from_scratch.ipynb
│   ├── 05_document_and_chunking_lab.ipynb
│   ├── 06_hybrid_retrieval.ipynb
│   ├── 07_rag_evaluation.ipynb
│   ├── 08_langgraph_rag_workflow.ipynb
│   ├── 09_tools_and_workflows.ipynb
│   ├── 10_workflow_vs_agent.ipynb
│   ├── 11_reliable_agent.ipynb
│   └── 12_financial_mcp.ipynb
├── src/finai_academy/
│   ├── settings.py
│   ├── providers.py
│   ├── documents.py
│   ├── chunking.py
│   ├── retrieval.py
│   ├── evaluation.py
│   ├── tools.py
│   ├── workflows.py
│   ├── agents.py
│   ├── mcp_server.py
│   └── capstone/
├── scripts/
│   ├── setup_check.py
│   ├── fetch_course_data.py
│   ├── execute_notebooks.py
│   ├── validate_notebooks.py
│   └── rehearse_class.py
├── tests/
│   ├── fixtures/
│   ├── notebook_expectations.py
│   └── test_*.py
└── docs/
    ├── instructor-guide.md
    └── class-test-report.md
```

Notebook files are the learner-facing deliverables. Small reusable functions are moved into `src/finai_academy` only after their notebook implementation has made the mechanism visible.

---

### Task 1: Lock the Two-Day Curriculum Manifest

**Files:**
- Modify: `course.yml`
- Modify: `scripts/validate_repo.py`
- Create: `tests/test_course_manifest.py`
- Create: `chapters/00-course-introduction.md`

**Interfaces:**
- Consumes: the approved schedule in `docs/superpowers/specs/2026-08-14-two-day-ai-engineering-class-design.md`.
- Produces: twelve ordered lesson records plus an orientation record, each with `id`, `title`, `day`, `start`, `end`, `deck`, `notebook`, and `capstone_increment`.

- [ ] **Step 1: Write a failing manifest test**

```python
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def course_manifest() -> dict:
    return yaml.safe_load(Path("course.yml").read_text(encoding="utf-8"))


def test_two_day_manifest_has_twelve_notebooks_and_exact_live_hours(course_manifest):
    lessons = course_manifest["lessons"]
    assert len([lesson for lesson in lessons if lesson.get("notebook")]) == 12
    assert lessons[0]["start"] == "09:00"
    assert lessons[-1]["end"] == "17:00"
    assert {lesson["day"] for lesson in lessons} == {1, 2}
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run: `uv run --extra dev pytest tests/test_course_manifest.py -q`

Expected: failure because the current YAML contains eleven legacy chapter records and no timed lesson manifest.

- [ ] **Step 3: Replace the chapter-only YAML contract with orientation and twelve lessons**

Use the exact titles and times from the approved design. Preserve brand colors and the course name. Add `delivery.two_day` for break and lunch windows rather than encoding breaks as notebooks.

- [ ] **Step 4: Update repository validation to read the manifest rather than a hard-coded ID range**

```python
def expected_notebooks(manifest: dict) -> list[Path]:
    return [ROOT / lesson["notebook"] for lesson in manifest["lessons"] if lesson.get("notebook")]
```

- [ ] **Step 5: Run the manifest and structural tests**

Run: `uv run --extra dev pytest tests/test_course_manifest.py -q`

Run: `uv run python scripts/validate_repo.py`

Expected: both commands pass and report twelve canonical notebook paths.

- [ ] **Step 6: Commit the curriculum contract**

```bash
git add course.yml chapters/00-course-introduction.md scripts/validate_repo.py tests/test_course_manifest.py
git commit -m "curriculum: lock the two-day lesson manifest"
```

---

### Task 2: Build the Ollama/OpenAI Provider Contract

**Files:**
- Modify: `src/finai_academy/settings.py`
- Create: `src/finai_academy/providers.py`
- Modify: `src/finai_academy/capstone/model_gateway.py`
- Modify: `src/finai_academy/__init__.py`
- Create: `.env.example`
- Modify: `docs/model-strategy.md`
- Modify: `tests/test_settings.py`
- Create: `tests/test_providers.py`

**Interfaces:**
- Produces: `Settings.from_environment()`, `ModelRun`, `create_chat_model(settings)`, `create_embeddings(settings)`, `provider_summary(settings)`, and `check_provider_configuration(settings)`.
- `create_chat_model` returns a LangChain-compatible chat model.
- `create_embeddings` returns an object with `embed_documents(list[str]) -> list[list[float]]` and `embed_query(str) -> list[float]`.

- [ ] **Step 1: Write failing provider-default tests**

```python
def test_ollama_defaults(monkeypatch):
    monkeypatch.setenv("FINAI_MODEL_PROVIDER", "ollama")
    monkeypatch.delenv("FINAI_CHAT_MODEL", raising=False)
    settings = Settings.from_environment()
    assert settings.chat_model == "qwen3:8b"
    assert settings.embedding_model == "qwen3-embedding:0.6b"


def test_openai_defaults(monkeypatch):
    monkeypatch.setenv("FINAI_MODEL_PROVIDER", "openai")
    monkeypatch.setenv("FINAI_EMBEDDING_PROVIDER", "openai")
    monkeypatch.delenv("FINAI_CHAT_MODEL", raising=False)
    monkeypatch.delenv("FINAI_EMBEDDING_MODEL", raising=False)
    settings = Settings.from_environment()
    assert settings.chat_model == "gpt-4.1-mini"
    assert settings.embedding_model == "text-embedding-3-small"
```

- [ ] **Step 2: Run the tests and confirm provider-specific defaults fail**

Run: `uv run --extra dev pytest tests/test_settings.py tests/test_providers.py -q`

- [ ] **Step 3: Implement validated settings**

```python
SUPPORTED_PROVIDERS = frozenset({"ollama", "openai"})

CHAT_DEFAULTS = {"ollama": "qwen3:8b", "openai": "gpt-4.1-mini"}
EMBEDDING_DEFAULTS = {
    "ollama": "qwen3-embedding:0.6b",
    "openai": "text-embedding-3-small",
}
```

Reject unsupported provider values. Require `OPENAI_API_KEY` only when a live OpenAI factory is called, not when settings are parsed or offline tests run.

- [ ] **Step 4: Implement lazy provider factories**

Use `ChatOllama`/`OllamaEmbeddings` for Ollama and `ChatOpenAI`/`OpenAIEmbeddings` for OpenAI. Keep imports inside the factories so core tests run without AI extras.

- [ ] **Step 5: Preserve the narrow structured-output adapter used by the capstone**

`create_structured_model(settings)` delegates model creation to `create_chat_model(settings)` and wraps it with `LangChainStructuredModel`.

- [ ] **Step 6: Document the two execution commands**

```bash
FINAI_MODEL_PROVIDER=ollama uv run --extra ai jupyter lab
FINAI_MODEL_PROVIDER=openai OPENAI_API_KEY=... uv run --extra ai jupyter lab
```

- [ ] **Step 7: Run settings, provider, and capstone tests**

Run: `uv run --extra dev pytest tests/test_settings.py tests/test_providers.py tests/test_capstone_briefing.py -q`

- [ ] **Step 8: Commit the model boundary**

```bash
git add .env.example docs/model-strategy.md src/finai_academy tests/test_settings.py tests/test_providers.py tests/test_capstone_briefing.py
git commit -m "feat: support Ollama and OpenAI across course components"
```

---

### Task 3: Add Setup and Notebook Execution Harnesses

**Files:**
- Modify: `pyproject.toml`
- Create: `scripts/setup_check.py`
- Create: `scripts/execute_notebooks.py`
- Create: `scripts/validate_notebooks.py`
- Create: `tests/notebook_expectations.py`
- Create: `tests/test_notebook_contracts.py`
- Create: `tests/test_setup_check.py`
- Modify: `README.md`

**Interfaces:**
- `validate_notebook(path: Path) -> list[str]` returns human-readable contract violations.
- `execute_notebook(path: Path, provider: str, timeout: int) -> ExecutionResult` executes with `nbclient`.
- `setup_report(settings: Settings) -> SetupReport` checks Python, imports, provider reachability, model presence, and keys.

- [ ] **Step 1: Declare explicit authoring and execution dependencies**

Add `nbclient`, `nbformat`, and `pyyaml` to the appropriate dependency groups. Add `pytest`, `ruff`, and notebook validation tools to `dev`.

- [ ] **Step 2: Write failing notebook-contract tests**

```python
REQUIRED_HEADINGS = (
    "Learning objectives",
    "Where this fits",
    "Failure lab",
    "Verification",
    "Challenge",
    "Capstone integration",
    "Recap",
)


def test_every_notebook_has_the_teaching_contract(canonical_notebooks):
    for path in canonical_notebooks:
        assert validate_notebook(path) == []
```

- [ ] **Step 3: Implement static notebook validation**

Validate title, signature, required headings, kernelspec, cleared outputs, unique cell IDs, no embedded secrets, no absolute user paths, and explicit expected runtime metadata.

- [ ] **Step 4: Implement three execution modes**

```text
--mode offline  : use fixtures/fakes; no network and no local daemon
--provider ollama : execute live against configured Ollama models
--provider openai : execute live against OPENAI_API_KEY
```

Live mode sets `FINAI_LIVE_MODE=1`; notebooks select deterministic fixtures when it is absent.

- [ ] **Step 5: Implement setup diagnostics with actionable errors**

The report must distinguish missing optional dependency, missing API key, unreachable Ollama, and missing Ollama model. It must print the exact corrective command for each condition.

- [ ] **Step 6: Run harness tests**

Run: `uv run --extra dev pytest tests/test_setup_check.py tests/test_notebook_contracts.py -q`

- [ ] **Step 7: Commit the teaching harness**

```bash
git add pyproject.toml README.md scripts/setup_check.py scripts/execute_notebooks.py scripts/validate_notebooks.py tests
git commit -m "test: add course setup and notebook execution harnesses"
```

---

### Task 4: Build and Visually Verify the Introduction Deck

**Files:**
- Create: `decks/00-course-introduction.pptx`
- Modify: `decks/README.md`
- Modify: `chapters/00-course-introduction.md`

**Interfaces:**
- Consumes: `course.yml`, the approved two-day design, and `assets/brand/finai-academy-style.md`.
- Produces: a twelve-slide, 16:9 PowerPoint deck with speaker notes and sources blocks.

- [ ] **Step 1: Write the audience-facing slide narrative**

Use this exact sequence:

1. `AI Engineering for Asset Management` — minimal title and subtitle.
2. `In two days, one analyst copilot becomes a complete AI system` — learning promise.
3. `The product comes first` — final copilot preview and NVIDIA/Schneider mission.
4. `Each failure creates the need for the next engineering layer` — cumulative stack map.
5. `Every lesson alternates explanation, code, failure and verification` — teaching rhythm.
6. `Day 1 morning: model calls become evidence-grounded answers` — 09:00-12:00 schedule.
7. `Day 1 afternoon: document structure determines retrieval quality` — 13:30-17:00 schedule.
8. `Day 1 ends with an evaluated financial RAG pipeline` — deliverable.
9. `Day 2 morning: deterministic workflows become bounded agents` — 09:00-12:00 schedule.
10. `Day 2 afternoon: MCP and evaluation complete the application` — 13:30-17:00 schedule.
11. `The same notebooks run locally or through OpenAI` — provider contract and trade-offs.
12. `Success means a working, inspectable and defensible copilot` — final acceptance criteria.

- [ ] **Step 2: Build the deck with `@oai/artifact-tool`**

Use Deep Navy `#051C2A`, Royal Blue `#1F40CB`, Cyan `#00A2EB`, Signal Orange `#F07D00`, Off White `#F5F5F5`, Georgia/Arial/Consolas fallbacks, and the required FinAI Academy footer.

- [ ] **Step 3: Render all twelve slides to PNG**

Run the bundled `render_slides.py` against `decks/00-course-introduction.pptx`.

- [ ] **Step 4: Inspect the contact sheet and every slide at full size**

Check title wrapping, schedule legibility, footer consistency, contrast, alignment, and visual rhythm.

- [ ] **Step 5: Run overflow and overlap checks**

Run the bundled `slides_test.py`. Correct every unintended overlap or overflow before accepting the deck.

- [ ] **Step 6: Commit the verified introduction deck**

```bash
git add decks/00-course-introduction.pptx decks/README.md chapters/00-course-introduction.md
git commit -m "content: add the two-day course introduction deck"
```

---

### Task 5: Build Notebook 01 — Model Gateway

**Files:**
- Create: `notebooks/01_model_gateway.ipynb`
- Create: `chapters/01-model-gateway.md`
- Modify: `tests/notebook_expectations.py`
- Create: `tests/test_model_gateway.py`

**Interfaces:**
- Consumes: `Settings`, `create_chat_model`, `provider_summary`.
- Produces: one provider-neutral response, latency metadata, and the `ModelRun` record defined in `src/finai_academy/providers.py`.

- [ ] **Step 1: Add unit tests for normalized model-run metadata**

```python
def test_model_run_records_provider_model_and_latency():
    run = ModelRun(provider="ollama", model="qwen3:8b", latency_ms=125, text="answer")
    assert run.provider == "ollama"
    assert run.latency_ms >= 0
```

- [ ] **Step 2: Author the detailed notebook**

Teach messages, roles, tokens, temperature, context limits, local versus hosted trade-offs, configuration, the first call, timing, a deliberately vague prompt, and provider switching. The failure lab compares an underspecified financial question with a constrained one. The challenge asks the learner to add a second model configuration without adding provider-specific notebook code.

- [ ] **Step 3: Execute static and offline notebook tests**

Run: `uv run python scripts/validate_notebooks.py notebooks/01_model_gateway.ipynb`

Run: `uv run --extra dev pytest tests/test_model_gateway.py -q`

- [ ] **Step 4: Run live provider smoke tests**

Run: `uv run python scripts/execute_notebooks.py notebooks/01_model_gateway.ipynb --provider ollama`

Run: `uv run python scripts/execute_notebooks.py notebooks/01_model_gateway.ipynb --provider openai`

- [ ] **Step 5: Commit Notebook 01**

```bash
git add notebooks/01_model_gateway.ipynb chapters/01-model-gateway.md tests/notebook_expectations.py tests/test_model_gateway.py
git commit -m "lesson: build the provider-neutral model gateway notebook"
```

---

### Task 6: Build Notebook 02 — Prompts and Structured Outputs

**Files:**
- Replace: `notebooks/02-prompting-and-structured-outputs.ipynb` with `notebooks/02_prompts_and_structured_outputs.ipynb`
- Modify: `src/finai_academy/capstone/models.py`
- Modify: `src/finai_academy/capstone/briefing.py`
- Create: `chapters/02-prompts-and-structured-outputs.md`
- Modify: `tests/test_capstone_briefing.py`

**Interfaces:**
- Consumes: `StructuredModel`, `AnalystBriefService`.
- Produces: `AnalystBrief` with typed findings, evidence categories, caveats, and open questions.

- [ ] **Step 1: Extend the failing semantic-validation tests**

```python
def test_reported_fact_requires_a_source_excerpt():
    with pytest.raises(ValidationError):
        AnalystFinding(
            statement="Revenue increased.",
            category="key_result",
            evidence_type="reported_fact",
            source_excerpt=None,
        )
```

- [ ] **Step 2: Implement domain-level validators**

Require a source excerpt for reported facts and management claims; require rationale for interpretations; preserve trusted company and period inputs.

- [ ] **Step 3: Author the detailed notebook**

Teach instruction/context/example/constraint separation, delimiters, prompt injection resistance, few-shot examples, JSON versus schema validation, retry boundaries, and semantic checks. The failure lab uses valid JSON containing an unsupported financial claim. The challenge adds a `confidence_reason` without allowing an ungrounded numeric confidence score.

- [ ] **Step 4: Execute offline, Ollama, and OpenAI modes**

Run the focused pytest file, static notebook validation, and both provider executions through `scripts/execute_notebooks.py`.

- [ ] **Step 5: Commit Notebook 02**

```bash
git add notebooks/02_prompts_and_structured_outputs.ipynb chapters/02-prompts-and-structured-outputs.md src/finai_academy/capstone tests/test_capstone_briefing.py
git commit -m "lesson: teach financial prompts and structured outputs"
```

---

### Task 7: Build Notebook 03 — Cache-Augmented Generation

**Files:**
- Create: `notebooks/03_cag_financial_document.ipynb`
- Create: `src/finai_academy/context.py`
- Create: `tests/test_context.py`
- Create: `chapters/03-cag-financial-document.md`

**Interfaces:**
- Produces: `ContextBudget`, `build_full_context_prompt`, and `should_use_full_context`.

- [ ] **Step 1: Write context-budget tests**

```python
def test_full_context_is_rejected_above_budget():
    budget = ContextBudget(max_input_tokens=8_000, reserved_output_tokens=1_000)
    assert not should_use_full_context(document_tokens=7_500, budget=budget)
```

- [ ] **Step 2: Implement deterministic token-budget logic**

The decision must account for system prompt, user question, document text, and reserved output tokens. Token estimation may be approximate but must be deterministic and explicitly labeled.

- [ ] **Step 3: Author the detailed notebook**

Teach context engineering, full-document prompting, repeated-document caching, context dilution, lost-in-the-middle behavior, latency, privacy, and the CAG/RAG decision. The failure lab places the answer in the middle of a long synthetic filing. The challenge writes a decision record for CAG versus RAG.

- [ ] **Step 4: Verify tests and both live providers**

Run `tests/test_context.py`, static notebook validation, and the two provider notebook executions.

- [ ] **Step 5: Commit Notebook 03**

```bash
git add notebooks/03_cag_financial_document.ipynb chapters/03-cag-financial-document.md src/finai_academy/context.py tests/test_context.py
git commit -m "lesson: add cache-augmented financial document analysis"
```

---

### Task 8: Build Notebook 04 — RAG from First Principles

**Files:**
- Create: `notebooks/04_rag_from_scratch.ipynb`
- Create: `src/finai_academy/retrieval_baseline.py`
- Create: `tests/test_retrieval_baseline.py`
- Create: `chapters/04-rag-from-scratch.md`

**Interfaces:**
- Produces: `split_text`, `TfidfRetriever.fit`, `TfidfRetriever.search`, and `build_grounded_prompt`.

- [ ] **Step 1: Write retrieval and abstention tests**

```python
def test_tfidf_retriever_returns_the_supporting_passage():
    retriever = TfidfRetriever.fit(["Revenue grew 12%.", "The board changed."])
    assert retriever.search("How much did revenue grow?", k=1)[0].text == "Revenue grew 12%."
```

- [ ] **Step 2: Implement retrieval without a vector database or framework**

Use plain Python splitting, `TfidfVectorizer`, cosine similarity, stable result IDs, scores, and a minimum-score abstention rule.

- [ ] **Step 3: Author the detailed notebook**

Make every stage visible: corpus, chunks, matrix, query vector, similarity scores, top-k, grounded prompt, generated answer, and citation. The failure lab uses a semantic paraphrase that lexical retrieval misses. The challenge adds source identifiers and abstention.

- [ ] **Step 4: Verify unit, offline, Ollama, and OpenAI execution**

Run the focused test and three notebook modes.

- [ ] **Step 5: Commit Notebook 04**

```bash
git add notebooks/04_rag_from_scratch.ipynb chapters/04-rag-from-scratch.md src/finai_academy/retrieval_baseline.py tests/test_retrieval_baseline.py
git commit -m "lesson: build financial RAG from first principles"
```

---

### Task 9: Build the Real Financial Corpus and Notebook 05 — Documents and Chunking

**Files:**
- Create: `assets/course-data/manifest.json`
- Create: `scripts/fetch_course_data.py`
- Create: `tests/fixtures/documents/`
- Create: `src/finai_academy/documents.py`
- Create: `src/finai_academy/chunking.py`
- Create: `tests/test_documents.py`
- Create: `tests/test_chunking.py`
- Create: `notebooks/05_document_and_chunking_lab.ipynb`
- Create: `chapters/05-document-and-chunking-lab.md`

**Interfaces:**
- Produces: `DocumentBlock`, `DocumentChunk`, `parse_pdf`, `parse_html`, `fixed_chunks`, `recursive_chunks`, `structure_aware_chunks`, `semantic_chunks`, `hierarchical_chunks`, and `contextualize_chunks`.

- [ ] **Step 1: Define a versioned source manifest**

Each entry includes company, reporting period, document type, language, official URL, retrieval date, expected SHA-256 when stable, and local relative path. Full public documents are downloaded by the script; compact licensed test fixtures are committed.

- [ ] **Step 2: Write parser provenance tests**

```python
def test_pdf_blocks_keep_page_heading_and_source():
    blocks = parse_pdf(FIXTURES / "sample_financial_report.pdf", source=SOURCE)
    assert blocks[0].page_number == 1
    assert blocks[0].source_url.startswith("https://")
    assert blocks[0].block_id
```

- [ ] **Step 3: Write chunk integrity tests**

Assert that tables are not split row-by-row, every child links to a parent, every chunk keeps provenance, and overlapping fixed chunks do not duplicate more than the configured overlap.

- [ ] **Step 4: Implement parsers and six chunking strategies**

Use SEC HTML when available and `pdfplumber` for machine-generated PDFs. LLM-based contextualization uses the shared provider gateway and is skipped in offline mode through recorded contextual labels.

- [ ] **Step 5: Author the detailed comparative laboratory**

Use the same NVIDIA/Schneider questions, corpus, and scoring function for fixed, recursive, structural, semantic, hierarchical, contextual, and LLM-assisted chunking. Show chunk boundaries before retrieval results. The failure lab splits a financial table and a section heading from its content. The challenge selects and defends one production strategy.

- [ ] **Step 6: Verify parser, chunker, notebook, and both providers**

Run `tests/test_documents.py`, `tests/test_chunking.py`, static notebook validation, offline execution, and live Ollama/OpenAI executions.

- [ ] **Step 7: Commit Notebook 05 and the corpus contract**

```bash
git add assets/course-data scripts/fetch_course_data.py notebooks/05_document_and_chunking_lab.ipynb chapters/05-document-and-chunking-lab.md src/finai_academy/documents.py src/finai_academy/chunking.py tests
git commit -m "lesson: compare financial document chunking strategies"
```

---

### Task 10: Build Notebook 06 — Embeddings, Hybrid Retrieval, and Reranking

**Files:**
- Create: `src/finai_academy/retrieval.py`
- Create: `tests/test_retrieval.py`
- Create: `notebooks/06_hybrid_retrieval.ipynb`
- Create: `chapters/06-hybrid-retrieval.md`

**Interfaces:**
- Produces: `DenseIndex`, `KeywordIndex`, `reciprocal_rank_fusion`, `rerank`, `RetrievalFilters`, and `retrieve_evidence`.

- [ ] **Step 1: Write exact-term, semantic, fusion, filter, and deduplication tests**

```python
def test_company_filter_prevents_cross_company_leakage(index):
    results = index.search("data centre demand", filters=RetrievalFilters(company="NVIDIA"))
    assert {result.chunk.company for result in results} == {"NVIDIA"}
```

- [ ] **Step 2: Implement local in-memory indexes first**

Persist vectors and metadata to a versioned local artifact. Include provider, embedding model, dimension, corpus hash, and chunking strategy in the index version.

- [ ] **Step 3: Implement reciprocal-rank fusion and deterministic fallback reranking**

Use stable rank tie-breaking. The optional model reranker is an enhancement; offline tests use lexical cross-feature scoring.

- [ ] **Step 4: Author the detailed notebook**

Teach embedding intuition, cosine similarity, index versioning, dense versus lexical failures, ticker/accounting exact terms, RRF, metadata filters, reranking, and final context budgeting. The failure lab demonstrates cross-company evidence leakage. The challenge tunes fusion weights and explains the measured consequence.

- [ ] **Step 5: Verify unit tests and all notebook modes**

Run focused tests plus offline, Ollama, and OpenAI execution.

- [ ] **Step 6: Commit Notebook 06**

```bash
git add notebooks/06_hybrid_retrieval.ipynb chapters/06-hybrid-retrieval.md src/finai_academy/retrieval.py tests/test_retrieval.py
git commit -m "lesson: add hybrid retrieval and evidence reranking"
```

---

### Task 11: Build Notebook 07 — RAG Evaluation and Tracing

**Files:**
- Create: `assets/course-data/evals/rag_cases.jsonl`
- Create: `src/finai_academy/evaluation.py`
- Create: `src/finai_academy/tracing.py`
- Create: `tests/test_evaluation.py`
- Create: `notebooks/07_rag_evaluation.ipynb`
- Create: `chapters/07-rag-evaluation.md`

**Interfaces:**
- Produces: `RagCase`, `RetrievalMetrics`, `AnswerEvaluation`, `evaluate_retrieval`, `evaluate_answer`, and `TraceRecorder`.

- [ ] **Step 1: Create a maintained gold set**

Include answerable, unanswerable, cross-company, cross-period, table-derived, and management-claim questions. Each case records expected source IDs and allowed evidence types.

- [ ] **Step 2: Write metric tests using hand-calculated examples**

```python
def test_recall_at_k_matches_hand_calculation():
    metrics = evaluate_retrieval(retrieved=["a", "b"], relevant={"b", "c"}, k=2)
    assert metrics.recall_at_k == 0.5
```

- [ ] **Step 3: Implement deterministic metrics and provider-backed judges separately**

Deterministic metrics cover recall, precision, reciprocal rank, citation identity, and abstention. Provider-backed judges score groundedness and completeness with a strict structured schema and visible rubric.

- [ ] **Step 4: Author the detailed notebook**

Teach the evaluation dataset, retrieval/generation separation, chunking comparison, traces, deterministic versus LLM-as-judge evaluation, judge bias, and regression thresholds. The failure lab shows a fluent answer produced from failed retrieval. The challenge chooses one improvement from evidence rather than intuition.

- [ ] **Step 5: Verify metrics and all notebook modes**

Run focused tests plus offline and both live providers.

- [ ] **Step 6: Commit Notebook 07**

```bash
git add assets/course-data/evals notebooks/07_rag_evaluation.ipynb chapters/07-rag-evaluation.md src/finai_academy/evaluation.py src/finai_academy/tracing.py tests/test_evaluation.py
git commit -m "lesson: evaluate and trace the financial RAG pipeline"
```

---

### Task 12: Build Notebook 08 — Stateful LangGraph RAG Workflow

**Files:**
- Create: `src/finai_academy/rag_graph.py`
- Create: `tests/test_rag_graph.py`
- Create: `notebooks/08_langgraph_rag_workflow.ipynb`
- Create: `chapters/08-langgraph-rag-workflow.md`

**Interfaces:**
- Produces: `RagState`, `build_rag_graph`, and typed `RetrievalCompleted`, `CitationEmitted`, and `AnswerCompleted` events.

- [ ] **Step 1: Write graph transition and state tests**

Assert retrieve → rerank → generate ordering, typed citations, checkpoint recovery, and abstention routing.

- [ ] **Step 2: Implement the smallest deterministic graph**

Nodes call the existing retrieval and answer services. Graph state contains question, filters, candidates, evidence, answer, citations, errors, and trace ID.

- [ ] **Step 3: Author the detailed notebook**

Teach why stateful orchestration exists, typed state, nodes, edges, streaming, memory boundaries, checkpoints, and observable failure handling. The failure lab resumes after a simulated generation error. The challenge adds an evidence-sufficiency conditional edge.

- [ ] **Step 4: Verify unit tests and all notebook modes**

Run focused tests plus offline, Ollama, and OpenAI executions.

- [ ] **Step 5: Commit Notebook 08**

```bash
git add notebooks/08_langgraph_rag_workflow.ipynb chapters/08-langgraph-rag-workflow.md src/finai_academy/rag_graph.py tests/test_rag_graph.py
git commit -m "lesson: orchestrate stateful financial RAG with LangGraph"
```

---

### Task 13: Build Notebook 09 — Financial Tools and Workflows

**Files:**
- Create: `src/finai_academy/tools.py`
- Create: `src/finai_academy/workflows.py`
- Create: `tests/fixtures/market_data/`
- Create: `tests/fixtures/news/`
- Create: `tests/test_tools.py`
- Create: `tests/test_workflows.py`
- Create: `notebooks/09_tools_and_workflows.ipynb`
- Create: `chapters/09-tools-and-workflows.md`

**Interfaces:**
- Produces: `ToolSpec`, `ToolResult`, `ToolRegistry`, and tools named `search_financial_documents`, `get_financial_facts`, `get_market_prices`, `calculate_financial_metric`, and `search_company_news`.

- [ ] **Step 1: Write tool-contract tests**

Assert validated inputs, structured errors, source dates, units, currencies, URLs, deterministic calculation formulas, and partial results when one provider fails.

- [ ] **Step 2: Implement adapters with offline fixtures**

Use SEC/company sources for financial facts, yfinance for educational market prices, Tavily for news, and pure Python for calculations. Network adapters return the same schemas as fixture adapters.

- [ ] **Step 3: Implement deterministic routing**

Known question types route through explicit Python logic. A comparative update runs document search, facts, prices, calculations, and news with an inspectable order.

- [ ] **Step 4: Author the detailed notebook**

Teach tool schemas, descriptions, validation, observations, timeouts, partial failure, provenance, deterministic workflows, and why Python calculations outrank LLM arithmetic. The failure lab feeds an ambiguous period and unit into a tool. The challenge adds one validated financial ratio.

- [ ] **Step 5: Verify unit tests and all notebook modes**

Run focused tests plus offline, Ollama, and OpenAI executions. Tavily live calls run only when `TAVILY_API_KEY` is present; otherwise the notebook uses a clearly labeled recorded result.

- [ ] **Step 6: Commit Notebook 09**

```bash
git add notebooks/09_tools_and_workflows.ipynb chapters/09-tools-and-workflows.md src/finai_academy/tools.py src/finai_academy/workflows.py tests
git commit -m "lesson: build typed financial tools and workflows"
```

---

### Task 14: Build Notebook 10 — Workflow Versus Agent

**Files:**
- Create: `src/finai_academy/agents.py`
- Create: `tests/test_agent_loop.py`
- Create: `notebooks/10_workflow_vs_agent.ipynb`
- Create: `chapters/10-workflow-vs-agent.md`

**Interfaces:**
- Produces: `AgentState`, `AgentAction`, `AgentObservation`, `AgentLimits`, and `run_agent`.

- [ ] **Step 1: Write bounded-loop tests**

```python
def test_agent_stops_at_max_steps(fake_planner, registry):
    result = run_agent("research", fake_planner, registry, AgentLimits(max_steps=2))
    assert result.status == "budget_exhausted"
    assert len(result.trajectory) == 2
```

- [ ] **Step 2: Implement a minimal reason-select-act-observe-stop loop**

The model returns structured `AgentAction`; the registry validates and executes; the loop records every action and observation; stop conditions are explicit.

- [ ] **Step 3: Author the detailed notebook**

Begin with a fixed workflow, expose its failure on a dynamic multi-step research question, then introduce the bounded agent. Compare workflow, agent, and multi-agent systems on determinism, flexibility, latency, cost, and inspectability. The challenge improves a tool description and measures tool-selection impact.

- [ ] **Step 4: Verify unit tests and all notebook modes**

Run focused tests plus offline, Ollama, and OpenAI executions.

- [ ] **Step 5: Commit Notebook 10**

```bash
git add notebooks/10_workflow_vs_agent.ipynb chapters/10-workflow-vs-agent.md src/finai_academy/agents.py tests/test_agent_loop.py
git commit -m "lesson: compare workflows with a bounded financial agent"
```

---

### Task 15: Build Notebook 11 — Reliable Agent and Trajectory Evaluation

**Files:**
- Modify: `src/finai_academy/agents.py`
- Modify: `src/finai_academy/evaluation.py`
- Create: `assets/course-data/evals/agent_cases.jsonl`
- Create: `tests/test_agent_reliability.py`
- Create: `tests/test_trajectory_evaluation.py`
- Create: `notebooks/11_reliable_agent.ipynb`
- Create: `chapters/11-reliable-agent.md`

**Interfaces:**
- Produces: `AgentBudget`, `RetryPolicy`, `TrajectoryEvaluation`, and `evaluate_trajectory`.

- [ ] **Step 1: Write recovery and budget tests**

Cover invalid arguments, unavailable tools, one safe retry, repeated identical calls, time budget, token budget, cost budget, and explicit human escalation.

- [ ] **Step 2: Implement bounded recovery**

Tool errors return typed observations to the model. Retries require a changed action. Repeated identical failed actions terminate with `stalled`.

- [ ] **Step 3: Implement trajectory scoring**

Score tool choice, argument validity, ordering constraints, redundant calls, evidence coverage, completion status, and final-answer support.

- [ ] **Step 4: Author the detailed notebook**

Teach agent state, recovery, budgets, idempotency, guardrails, human control, and trajectory evaluation. The failure lab triggers a redundant-call loop. The challenge sets acceptable budgets for a real financial-research mission.

- [ ] **Step 5: Verify unit tests and all notebook modes**

Run focused tests plus offline, Ollama, and OpenAI executions.

- [ ] **Step 6: Commit Notebook 11**

```bash
git add assets/course-data/evals/agent_cases.jsonl notebooks/11_reliable_agent.ipynb chapters/11-reliable-agent.md src/finai_academy/agents.py src/finai_academy/evaluation.py tests
git commit -m "lesson: add reliable agent execution and trajectory evaluation"
```

---

### Task 16: Build Notebook 12 — Financial MCP

**Files:**
- Create: `src/finai_academy/mcp_server.py`
- Create: `src/finai_academy/mcp_client.py`
- Create: `tests/test_mcp_server.py`
- Create: `tests/test_mcp_client.py`
- Create: `notebooks/12_financial_mcp.ipynb`
- Create: `chapters/12-financial-mcp.md`

**Interfaces:**
- Produces: local MCP server tools `search_financial_documents`, `get_market_prices`, and `calculate_financial_metric`; `FinancialMcpClient.list_tools()` and `FinancialMcpClient.call_tool()`.

- [ ] **Step 1: Write server capability tests**

Assert exact tool names, schemas, structured results, error results, read-only behavior, and absence of any trade-execution capability.

- [ ] **Step 2: Write stdio client integration tests**

Start the local server, discover tools, call the calculator, call document search against fixtures, and close the session cleanly.

- [ ] **Step 3: Implement server and client by wrapping tested domain functions**

MCP is an interoperability boundary, not a second implementation of financial logic. Preserve provenance and trace IDs in tool results.

- [ ] **Step 4: Author the detailed notebook**

Teach host, client, server, tools, resources, prompts, stdio transport, capability discovery, direct function calling versus MCP, permissions, and trust boundaries. The failure lab exposes an overly broad tool schema. The challenge adds a safe resource listing available company documents.

- [ ] **Step 5: Verify unit, integration, offline, Ollama, and OpenAI execution**

Run MCP tests and all notebook modes. Model-provider choice affects the agent consuming MCP, not the MCP server’s deterministic calculation behavior.

- [ ] **Step 6: Commit Notebook 12**

```bash
git add notebooks/12_financial_mcp.ipynb chapters/12-financial-mcp.md src/finai_academy/mcp_server.py src/finai_academy/mcp_client.py tests
git commit -m "lesson: expose financial capabilities through MCP"
```

---

### Task 17: Integrate the Financial Analyst Copilot Capstone

**Files:**
- Modify: `final-project/app.py`
- Modify: `final-project/README.md`
- Modify: `final-project/PRODUCT_SPEC.md`
- Create: `final-project/api.py`
- Create: `final-project/docker-compose.yml`
- Create: `src/finai_academy/capstone/copilot.py`
- Create: `tests/test_capstone_copilot.py`
- Create: `tests/test_capstone_api.py`

**Interfaces:**
- Produces: `FinancialAnalystCopilot.answer(ResearchQuestion) -> ResearchAnswer`, FastAPI `/health`, `/research`, and `/research/stream`, and Streamlit rendering of evidence, calculations, news, and trajectory.

- [ ] **Step 1: Write an end-to-end fixture-backed capstone test**

```python
def test_comparative_answer_keeps_evidence_and_trajectory(copilot):
    answer = copilot.answer(ResearchQuestion(
        companies=["NVIDIA", "Schneider Electric"],
        question="Compare evidence of data-centre demand and key risks.",
    ))
    assert {citation.company for citation in answer.citations} == {"NVIDIA", "Schneider Electric"}
    assert answer.trajectory
    assert answer.status in {"complete", "partial"}
```

- [ ] **Step 2: Compose existing tested components**

The capstone uses the shared settings, ingestion, retrieval, evaluation, tools, bounded agent, and MCP client. It does not duplicate notebook functions in `final-project`.

- [ ] **Step 3: Implement evidence-aware presentation**

Render reported facts, calculations, management claims, external facts, interpretations, and open questions distinctly. Display calculation inputs and formulas, news URL and publication date, citations, agent budget consumption, and trace ID.

- [ ] **Step 4: Add health and streaming API tests**

Verify provider status is reported without exposing keys, SSE events are typed and ordered, and partial failures produce a usable partial research answer.

- [ ] **Step 5: Run the complete automated suite**

Run: `uv run --extra dev pytest -q`

Run: `uv run --extra dev ruff check .`

- [ ] **Step 6: Commit capstone integration**

```bash
git add final-project src/finai_academy/capstone tests/test_capstone_copilot.py tests/test_capstone_api.py
git commit -m "feat: integrate the Financial Analyst Copilot capstone"
```

---

### Task 18: Produce the Twelve Technical Micro-Decks

**Files:**
- Create: `decks/01-model-gateway.pptx` through `decks/12-financial-mcp.pptx`
- Modify: `decks/README.md`
- Modify: corresponding `chapters/*.md`

**Interfaces:**
- Each deck consumes its chapter contract and notebook.
- Each deck produces five to seven slides: problem, mental model, architecture, failure mode, trade-offs, notebook mission, and optional debrief.

- [ ] **Step 1: Write one takeaway-title outline per deck**

The outline must introduce only concepts used in that notebook and name the controlled failure that motivates the improvement.

- [ ] **Step 2: Build each deck with the shared FinAI Academy visual system**

Use PowerPoint speaker notes for teaching cues and `[Sources]` blocks. Keep code in notebooks; slides may show only short signatures or pseudocode.

- [ ] **Step 3: Render every slide and create one contact sheet per deck**

Inspect each full-size slide for wrapping, contrast, footer, progress indicator, and visual consistency.

- [ ] **Step 4: Run overflow/overlap tests on all thirteen decks**

No deck is accepted with unresolved overflow or unintended overlap.

- [ ] **Step 5: Commit decks in day-sized batches**

```bash
git add decks chapters
git commit -m "content: add the Day 1 technical micro-decks"
```

```bash
git add decks chapters
git commit -m "content: add the Day 2 technical micro-decks"
```

---

### Task 19: Run the Dual-Provider Course Acceptance Matrix

**Files:**
- Create: `scripts/run_acceptance.py`
- Create: `docs/class-test-report.md`
- Create: `tests/test_acceptance_report.py`

**Interfaces:**
- Produces: machine-readable JSON results and the human-readable class-test report.

- [ ] **Step 1: Define the acceptance matrix**

```text
static structure       × 12 notebooks
offline execution      × 12 notebooks
Ollama live execution  × 12 notebooks
OpenAI live execution  × 12 notebooks
unit/integration tests × complete repository
Ruff                   × complete repository
deck rendering         × 13 decks
deck overflow checks   × 13 decks
capstone smoke test    × offline, Ollama, OpenAI
```

- [ ] **Step 2: Implement resumable acceptance execution**

Store command, start/end time, provider, notebook, exit code, artifact paths, and a redacted failure excerpt. Never store prompts containing keys or environment values.

- [ ] **Step 3: Run the offline matrix**

Run: `uv run python scripts/run_acceptance.py --mode offline`

Expected: all unit tests, structural checks, offline notebooks, and deck checks pass.

- [ ] **Step 4: Run the Ollama matrix**

Run: `uv run python scripts/run_acceptance.py --provider ollama`

Expected: twelve notebooks and the capstone complete using configured local models.

- [ ] **Step 5: Run the OpenAI matrix**

Run: `uv run python scripts/run_acceptance.py --provider openai`

Expected: twelve notebooks and the capstone complete using `OPENAI_API_KEY`, with token usage and estimated cost recorded but no secret values.

- [ ] **Step 6: Commit the evidence-backed test report**

```bash
git add scripts/run_acceptance.py docs/class-test-report.md tests/test_acceptance_report.py
git commit -m "test: verify the complete course with Ollama and OpenAI"
```

---

### Task 20: Rehearse the Two-Day Class End to End

**Files:**
- Create: `docs/instructor-guide.md`
- Create: `docs/rehearsal-log.md`
- Create: `scripts/rehearse_class.py`
- Modify: `README.md`

**Interfaces:**
- Produces: a timed run sheet, instructor checkpoints, fallback paths, reset commands, expected learner outputs, and a go/no-go checklist.

- [ ] **Step 1: Create the instructor run sheet from `course.yml`**

For every block record start, stop, slide range, notebook cell range, expected learner artifact, likely failure, recovery path, and capstone checkpoint.

- [ ] **Step 2: Add a clean-machine rehearsal command**

The script validates setup, confirms the configured provider, clears only generated course outputs, recreates indexes, executes the selected notebook sequence, starts the capstone health check, and writes timings. Any cleanup target must be an explicit generated directory inside the project.

- [ ] **Step 3: Conduct a timed Day 1 rehearsal**

Record actual slide duration, notebook duration, model latency, debugging buffer consumed, learner decision points, and content to cut if the session exceeds the scheduled block.

- [ ] **Step 4: Conduct a timed Day 2 rehearsal**

Record the same measurements, including agent nondeterminism, MCP startup, and capstone integration time.

- [ ] **Step 5: Apply the timing rule**

No block may rely on typing boilerplate live. If a block exceeds its allocation by more than five minutes, reduce explanation or prebuild plumbing while preserving the learning decision, failure lab, and verification.

- [ ] **Step 6: Run final verification**

Run: `uv run --extra dev pytest -q`

Run: `uv run --extra dev ruff check .`

Run: `uv run python scripts/validate_repo.py`

Run: `uv run python scripts/run_acceptance.py --mode offline`

- [ ] **Step 7: Commit the class-ready instructor package**

```bash
git add README.md docs/instructor-guide.md docs/rehearsal-log.md scripts/rehearse_class.py
git commit -m "docs: add the tested two-day instructor runbook"
```

---

## Execution Checkpoints

### Checkpoint A — Foundations Ready

Tasks 1-5 are complete when the timed curriculum manifest, provider layer, notebook harness, introduction deck, and first live notebook are verified. This is the first useful internal pilot.

### Checkpoint B — Day 1 Ready

Tasks 6-11 are complete when the Financial Analyst Copilot can ingest real NVIDIA/Schneider documents, compare chunking strategies, retrieve evidence, generate citations, and report RAG metrics with both providers.

### Checkpoint C — Day 2 Ready

Tasks 12-16 are complete when the evaluated RAG workflow, financial tools, bounded agent, recovery logic, and MCP server/client work with both providers.

### Checkpoint D — Class Ready

Tasks 17-20 are complete when the integrated capstone, thirteen decks, dual-provider acceptance matrix, and two timed rehearsals pass their documented criteria.

## Definition of Done

- `uv run --extra dev pytest -q` reports zero failures.
- `uv run --extra dev ruff check .` reports zero errors.
- `uv run python scripts/validate_repo.py` reports a valid repository.
- Twelve canonical notebooks pass static validation and offline execution.
- Twelve canonical notebooks complete one live Ollama run.
- Twelve canonical notebooks complete one live OpenAI run.
- The capstone completes the maintained NVIDIA/Schneider comparative mission in offline, Ollama, and OpenAI modes.
- All thirteen decks render without overflow or unintended overlap.
- Day 1 and Day 2 each fit 09:00-17:00 with the approved lunch and breaks.
- The instructor guide contains a tested fallback for network failure, OpenAI quota failure, Ollama unavailability, Tavily unavailability, and slow learner machines.
- The final class-test report records dates, models, commands, results, durations, and known limitations without secrets.
