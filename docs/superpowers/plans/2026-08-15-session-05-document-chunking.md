# Session 05 Document and Chunking Laboratory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a 90-minute visual laboratory that parses real-source financial document fixtures, compares seven chunking strategies, and connects chunk construction to retrieval quality.

**Architecture:** `documents.py` normalizes HTML and PDF into ordered `DocumentBlock` records. `chunking.py` transforms those blocks into provenance-preserving `DocumentChunk` records through a common strategy interface. The notebook visualizes the pipeline and evaluates every strategy with the existing lexical retriever; the deck teaches the same mental model without duplicating code.

**Tech Stack:** Python 3.11, `beautifulsoup4`, `pdfplumber`, scikit-learn, pandas, matplotlib, Pydantic-compatible provider gateway, Jupyter, pytest, Ruff, `@oai/artifact-tool`.

## Global Constraints

- The session is Day 1, 13:30–15:00: 20 minutes slides and 70 minutes notebook.
- The repository must stay lightweight and the core lesson must run without network access.
- Parsing and deterministic chunking must not require an LLM.
- The LLM-assisted path must support offline recordings, Ollama and OpenAI.
- Every block and chunk must preserve stable source provenance.
- Production embeddings, hybrid retrieval and reranking remain Lesson 06 scope.
- The footer is `First Finance - Arnaud Demes`.

---

### Task 1: Versioned source fixtures and canonical document parsing

**Files:**
- Create: `assets/course-data/manifest.json`
- Create: `assets/course-data/fixtures/nvidia_fy2026_excerpt.html`
- Create: `assets/course-data/fixtures/schneider_fy2025_excerpt.pdf`
- Create: `scripts/fetch_course_data.py`
- Create: `src/finai_academy/documents.py`
- Create: `tests/test_documents.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `DocumentSource`, `DocumentBlock`, `parse_html(path, source)`, `parse_pdf(path, source)`, and `load_source_manifest(path)`.
- `DocumentBlock.table_rows` is an immutable tuple of row tuples and block order is stable.

- [ ] **Step 1: Write parser contract tests**

```python
def test_html_parser_preserves_heading_table_order_and_source(html_fixture, source):
    blocks = parse_html(html_fixture, source)
    assert [block.block_type for block in blocks][:3] == ["heading", "paragraph", "table"]
    assert blocks[2].table_rows[0] == ("Business", "Revenue", "Growth")
    assert all(block.source_url == source.source_url for block in blocks)


def test_pdf_parser_preserves_page_table_and_provenance(pdf_fixture, source):
    blocks = parse_pdf(pdf_fixture, source)
    assert {block.page_number for block in blocks} == {1, 2}
    assert any(block.block_type == "table" for block in blocks)
    assert all(block.block_id and block.source_id == source.source_id for block in blocks)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `uv run pytest tests/test_documents.py -q`

Expected: collection fails because `finai_academy.documents` does not exist.

- [ ] **Step 3: Add compact, clearly labelled fixtures and source manifest**

The HTML fixture retains representative SEC-style headings, paragraphs and one table.
The two-page PDF fixture is a compact classroom extract based on Schneider's official
FY2025 disclosure and explicitly identifies itself as a teaching extract. The manifest
contains official URLs, retrieval date, local paths and SHA-256 values.

- [ ] **Step 4: Implement the minimal canonical parser**

Use BeautifulSoup for ordered HTML elements. Use `pdfplumber` page text and table
extraction for the PDF, emitting table blocks before excluding duplicate table lines
from paragraph blocks. Validate non-empty identifiers and source fields in dataclass
post-init methods.

- [ ] **Step 5: Run parser tests and lint**

Run: `uv run pytest tests/test_documents.py -q`

Expected: all parser tests pass.

Run: `uv run ruff check src/finai_academy/documents.py tests/test_documents.py`

- [ ] **Step 6: Commit the parsing boundary**

```bash
git add assets/course-data scripts/fetch_course_data.py pyproject.toml uv.lock src/finai_academy/documents.py tests/test_documents.py
git commit -m "feat: add provenance-preserving financial document parsing"
```

---

### Task 2: Seven chunking strategies through one tested interface

**Files:**
- Create: `src/finai_academy/chunking.py`
- Create: `tests/test_chunking.py`
- Modify: `src/finai_academy/lesson_support.py`
- Modify: `tests/test_lesson_support.py`

**Interfaces:**
- Produces: `DocumentChunk`, `fixed_chunks`, `recursive_chunks`, `structure_aware_chunks`, `semantic_chunks`, `hierarchical_chunks`, `contextualize_chunks`, `proposition_chunks`, `compare_chunking_strategies`.
- LLM-assisted code consumes any object with `invoke(messages)` and preserves the input block identifiers in every proposition chunk.

- [ ] **Step 1: Write integrity and failure tests**

```python
def test_structure_aware_chunking_keeps_table_atomic(blocks):
    chunks = structure_aware_chunks(blocks, max_chars=500)
    table_chunk = next(chunk for chunk in chunks if "Data Center" in chunk.text)
    assert "Business | Revenue | Growth" in table_chunk.text
    assert table_chunk.source_block_ids


def test_hierarchical_children_link_to_parent(blocks):
    chunks = hierarchical_chunks(blocks, child_max_chars=160)
    children = [chunk for chunk in chunks if chunk.parent_id]
    assert children
    assert {child.parent_id for child in children} <= {chunk.chunk_id for chunk in chunks}


def test_fixed_size_reproduces_heading_table_split(blocks):
    chunks = fixed_chunks(blocks, chunk_size=90, overlap=10)
    assert any("Business | Revenue" in chunk.text and "Data Center" not in chunk.text for chunk in chunks)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `uv run pytest tests/test_chunking.py -q`

Expected: collection fails because `finai_academy.chunking` does not exist.

- [ ] **Step 3: Implement fixed, recursive and structure-aware strategies**

Keep deterministic stable identifiers, validate overlap bounds, preserve ordered source
block identifiers, keep table blocks atomic in structure-aware mode, and attach the most
recent heading path to content chunks.

- [ ] **Step 4: Implement semantic and hierarchical strategies**

`semantic_chunks` accepts an optional sentence-similarity sequence and otherwise derives
adjacent similarities with TF-IDF. A new chunk starts below the threshold.
`hierarchical_chunks` emits parent sections followed by child chunks that reference their
parent IDs.

- [ ] **Step 5: Implement contextual and LLM-assisted strategies**

`contextualize_chunks` adds a compact company/period/document/section prefix without
losing raw text or provenance. `RecordedChunkingModel` returns stable JSON propositions;
the live provider path uses the shared gateway and validates a list of strings.

- [ ] **Step 6: Implement the comparative scorecard**

Return strategy, chunk count, mean/max size, heading-retention rate, table-integrity rate,
provenance rate and retrieval recall at k. Adapt chunks to `EvidencePassage` only at the
retrieval boundary.

- [ ] **Step 7: Run focused and regression tests**

Run: `uv run pytest tests/test_chunking.py tests/test_lesson_support.py tests/test_retrieval.py -q`

Expected: all tests pass.

- [ ] **Step 8: Commit the strategy layer**

```bash
git add src/finai_academy/chunking.py src/finai_academy/lesson_support.py tests/test_chunking.py tests/test_lesson_support.py
git commit -m "feat: compare financial chunking strategies"
```

---

### Task 3: Visual guided notebook and course contract

**Files:**
- Create: `notebooks/05_document_and_chunking_lab.ipynb`
- Create: `chapters/05-document-and-chunking-lab.md`
- Modify: `tests/test_notebook_contracts.py`
- Modify: `tests/test_course_manifest.py`

**Interfaces:**
- Consumes all Task 1 and Task 2 public interfaces plus `LexicalRetriever`.
- Produces the final marker `PASS — document and chunking laboratory verified`.

- [ ] **Step 1: Add a failing notebook execution contract**

```python
def test_document_chunking_notebook_offline_run_is_visual_and_verified(tmp_path):
    result, executed = execute_lesson("05_document_and_chunking_lab.ipynb", tmp_path)
    assert result.returncode == 0, result.stderr
    assert count_png_outputs(executed) >= 8
    assert "Table integrity failure reproduced" in stream_text(executed)
    assert "PASS — document and chunking laboratory verified" in stream_text(executed)
```

- [ ] **Step 2: Run the contract and verify RED**

Run: `uv run pytest tests/test_notebook_contracts.py -k document_chunking -q`

Expected: failure because the notebook does not exist.

- [ ] **Step 3: Author the 70-minute notebook**

Use the required headings. Make parsing, block order, table reconstruction, chunk
boundaries, semantic threshold, parent-child links, scorecard, failure lab and Lesson 06
handoff visible. Core exercises run deterministically; the proposition stage selects
offline, Ollama or OpenAI from `FINAI_MODEL_MODE`.

- [ ] **Step 4: Author the instructor chapter**

Include minute-by-minute pacing, expected outputs, checkpoint answers, failure diagnosis,
challenge solution, source notes and explicit scope boundary.

- [ ] **Step 5: Execute and inspect every visual**

Run: `uv run python scripts/execute_notebooks.py notebooks/05_document_and_chunking_lab.ipynb --mode offline --output-dir /private/tmp/finai-lesson05-executed`

Extract every `image/png` output, inspect individually, and fix clipping, unreadable
labels or misleading scales.

- [ ] **Step 6: Validate source notebook and focused tests**

Run: `uv run python scripts/validate_notebooks.py notebooks/05_document_and_chunking_lab.ipynb`

Run: `uv run pytest tests/test_notebook_contracts.py tests/test_course_manifest.py -q`

- [ ] **Step 7: Commit the guided lesson**

```bash
git add notebooks/05_document_and_chunking_lab.ipynb chapters/05-document-and-chunking-lab.md tests/test_notebook_contracts.py tests/test_course_manifest.py
git commit -m "lesson: build financial document chunking laboratory"
```

---

### Task 4: Companion diagram deck

**Files:**
- Create: `decks/05-document-and-chunking-lab.pptx`

**Interfaces:**
- Mirrors the notebook's block model, chunk boundary patterns, hierarchy and scorecard.

- [ ] **Step 1: Read the presentation style and API references**

Read `style_guidelines.md`, `API_QUICK_START.md`, `API_DOCS.md` and the existing Lesson 04
deck. Use the explicit First Finance visual system rather than a new template.

- [ ] **Step 2: Create the eight-slide deck with `@oai/artifact-tool`**

Create editable diagrams, connectors before nodes, low-density labels, speaker notes with
`[Sources]` blocks and the required footer. Keep audience copy factual and professional.

- [ ] **Step 3: Render and inspect every slide**

Render the final PPTX, inspect all eight slides individually, and revise any overlap,
clipping, connector or hierarchy problem.

- [ ] **Step 4: Run the presentation overflow test**

Run: `python slides_test.py decks/05-document-and-chunking-lab.pptx`

Expected: no overflow detected.

- [ ] **Step 5: Commit the deck**

```bash
git add decks/05-document-and-chunking-lab.pptx
git commit -m "content: add lesson five document chunking deck"
```

---

### Task 5: Full verification and provider diagnosis

**Files:**
- Verify all changed Lesson 05 files.

- [ ] **Step 1: Run the complete automated suite**

Run: `uv run pytest -q`

Run: `uv run ruff check .`

Run: `uv run python scripts/validate_repo.py`

Run: `uv run python scripts/validate_notebooks.py notebooks/01_model_gateway.ipynb notebooks/02_prompts_and_structured_outputs.ipynb notebooks/03_cag_financial_document.ipynb notebooks/04_rag_from_scratch.ipynb notebooks/05_document_and_chunking_lab.ipynb`

Run: `git diff --check`

- [ ] **Step 2: Diagnose live provider availability**

Check whether `OPENAI_API_KEY` is configured and whether Ollama responds locally. Run live
notebook modes only when their prerequisite is present. Report an unavailable provider as
an untested path, never as a passing live test.

- [ ] **Step 3: Review Git scope**

Inspect `git status --short` and the Lesson 05 diff. Do not bundle unrelated existing
Lesson 02–04 changes into a broad commit.

- [ ] **Step 4: Hand off the verified lesson**

Provide absolute links to the notebook and chapter, cite the final deck exactly once,
state test counts and provider limits, and explain that Lesson 06 holds representation
and ranking constant until embeddings and hybrid retrieval are introduced.
