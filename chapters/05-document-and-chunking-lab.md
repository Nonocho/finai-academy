# Lesson 05 — Financial Documents and Chunking Laboratory

**Schedule:** Day 1, 13:30–15:00

**Format:** 20 minutes of concepts and diagrams, 70 minutes of guided notebook work

**Capstone increment:** configurable, provenance-preserving ingestion and chunking pipeline

## Teaching outcome

Students should leave with one operational principle:

> Retrieval cannot recover a relationship that parsing or chunking has already destroyed.

They parse compact real-source NVIDIA HTML and Schneider Electric PDF fixtures, normalize
both into ordered `DocumentBlock` records, compare seven core chunking strategies and
reproduce a table-integrity failure before retrieval or generation begins. Semantic
boundaries use the configured embedding provider; LLM contextual enrichment remains a
bounded, validated transformation rather than an autonomous agent.

## Connection to Lesson 04

Lesson 04 deliberately started with six prepared passages. State its hidden assumption at
the beginning of this session:

```text
raw document → parsing → canonical blocks → chunks → index → rank → context → answer
                                      Lesson 05 ↑       Lesson 04 ↑
```

Do not present parsing and chunking as optional preprocessing. They create the units that
every later representation and ranking system receives.

## Source policy

The core lab is offline and lightweight. It uses:

- an SEC-style HTML teaching extract based on NVIDIA's fiscal 2026 Form 10-K;
- a two-page machine-generated PDF teaching extract based on Schneider Electric's fiscal
  2025 full-year results; and
- a source manifest containing official URLs, retrieval dates, local paths and SHA-256
  hashes.

Every teaching fixture is labelled as an extract rather than an original filing page. The
optional `scripts/fetch_course_data.py` script retrieves the complete official sources for
extended work.

## Core architecture

### `DocumentSource`

Carries company, period, document type, language, official URL and fixture provenance.

### `DocumentBlock`

Provides the stable boundary between format-specific parsing and format-neutral chunking.
Each block contains:

- a stable block and source identifier;
- company, period, document type and language;
- ordered position and optional page number;
- heading path and block type;
- normalized text and optional table rows; and
- the official source URL.

### `DocumentChunk`

Adds strategy name, source block identifiers, optional parent link, role and retrieval
text. `raw_text` remains the source evidence; `generated_context` is stored separately.
Every chunk remains traceable to the exact blocks from which it was constructed.

## Concept sequence for the slides

### 1. Financial documents contain relationships, not only text

A PDF may visually associate a value with a row, column, unit, period and footnote. Linear
text extraction can reorder or separate those relationships. HTML often exposes more
structure, but still requires normalization and provenance.

### 2. Parsing and chunking are separate decisions

The parser describes what the document contains. The chunker decides which evidence units
will be indexed. Keeping one canonical block model lets the team change chunking policies
without rewriting every parser.

### 3. Fixed and recursive strategies are baselines

Fixed windows are transparent but can split anywhere. Recursive splitting prefers
paragraph, sentence and word boundaries, but has no financial understanding and may still
separate a heading or table from its context.

### 4. Structure-aware chunking protects document meaning

Tables stay atomic. Heading paths remain attached as metadata. Paragraphs are grouped only
within compatible sections. A structure-aware policy is a strong initial production
baseline for filings and annual reports.

### 5. Semantic boundaries depend on an embedding provider and threshold

The notebook embeds adjacent sentences with the provider selected by shared settings.
Offline execution uses the versioned `financial-concepts-v1` fixture; Ollama and OpenAI
use their configured embedding models. Provider, model, segmentation and threshold are
versioned engineering parameters, not universal constants.

### 6. Hierarchical retrieval separates matching from reading

Small children improve matching precision. Parent sections restore the evidence context
after retrieval. The parent-child link must be created during chunk construction.

### 7. Deterministic and generated contexts are different policies

A deterministic prefix copies trusted company, period and section metadata. LLM
contextual enrichment generates a short explanation of where the chunk sits in the
complete document, then indexes `generated_context + raw_text`. It never replaces raw
evidence or provenance. Proposition chunking is a separate optional extension after the
core PASS marker.

## Notebook pacing

| Time | Activity | Instructor emphasis |
|---:|---|---|
| 0–12 min | Parser ladder, fixture hashes and extraction failure | Extraction quality precedes chunk size. |
| 12–20 min | Inspect canonical `DocumentBlock` records | Parsing creates ordered evidence, not a text dump. |
| 20–30 min | Fixed, recursive and structure-aware chunks | Predict and inspect each boundary. |
| 30–40 min | Provider-aware semantic boundaries | Provider, model and threshold define the policy. |
| 40–48 min | Parent-child hierarchy | Retrieve a child, restore a verified parent. |
| 48–58 min | Prefix versus LLM contextual enrichment | Keep raw and generated fields separate. |
| 58–65 min | Token, latency, construction and retrieval comparison | Extra cost must earn measurable value. |
| 65–70 min | Verification, knowledge check and capstone handoff | Diagnose the stage before changing models. |

## Visual teaching contract

The notebook produces eleven executable figures:

1. source-to-retriever typed pipeline;
2. ordered block timelines for HTML and PDF;
3. raw PDF extraction versus canonical table block;
4. fixed, recursive and structure-aware boundary patterns;
5. embedding-based adjacent-similarity curve and threshold;
6. parent-child hierarchy;
7. raw evidence versus deterministic and generated context;
8. token inflation and construction latency;
9. chunk count and mean-size economics;
10. multi-metric strategy and retrieval scorecard; and
11. fixed table split versus atomic structural table.

The deck uses the same visual grammar. Students should be able to name the notebook state
that implements each deck diagram.

## Failure lab

The NVIDIA table contains business, revenue and year-on-year growth columns. With a
90-character fixed window, the table crosses a chunk boundary and the final Gaming cells
are separated.

Ask in order:

1. Did the parser recover the correct table? **Yes.**
2. Did fixed chunk construction keep the table atomic? **No.**
3. Should we change the retrieval model first? **No.**

The diagnosis is a chunk-construction failure. A dense embedding or larger LLM cannot
reconstruct the row/column relationship safely if the selected chunk does not contain it.

## Checkpoint questions and answers

### 1. Why attach metadata during parsing rather than after retrieval?

The parser knows the source, page, block order and structural position. Reconstructing that
information later is ambiguous and can produce incorrect citations or filters.

### 2. Is recursive chunking structure-aware?

Not necessarily. It prefers textual separators, but it does not inherently understand that
a table, heading-content pair or footnote reference must remain intact.

### 3. Why can contextual prefixes improve retrieval?

A short prefix can add missing company, period, document and section terms to an otherwise
ambiguous chunk. It also consumes index and context tokens, so its value must be measured.

### 4. Why is LLM contextual enrichment not agentic chunking?

It is one bounded call with one validated JSON output per chunk. It does not plan, choose
tools, iterate autonomously or decide when the task is complete.

### 5. What is the purpose of parent-child chunking?

Use a small representation for precise retrieval, then expand to a larger verified section
for interpretation and generation.

### 6. Why not use LLM propositions as the only evidence representation?

They are transformed claims rather than original evidence. The model may omit qualifiers,
units or relationships. Keep source links and original blocks, validate outputs, and compare
their retrieval benefit against cost and transformation risk.

### 7. Does 100% retrieval recall prove a strategy is production-ready?

No. The maintained set is deliberately small. A release decision also needs table,
provenance, citation, latency, cost and broader question coverage.

## Challenge solution

A defensible initial policy is:

> Index structure-aware child chunks with company, reporting period, document type,
> section, page and source metadata. Keep tables atomic and create parent section IDs for
> post-retrieval expansion. Add contextual prefixes only when they improve the maintained
> question set enough to justify the token cost. Evaluate proposition chunks as a secondary
> representation while retaining original evidence.

The correct answer is not one universal chunk size. It is a versioned policy plus an
evaluation set and explicit integrity constraints.

## Provider modes

The deterministic laboratory uses versioned offline embeddings and contextual responses.
In live mode, semantic boundaries and contextual enrichment cross the shared provider
boundary. Optional proposition construction runs only when `FINAI_RUN_OPTIONAL=1`.

### Offline

```bash
uv run python scripts/execute_notebooks.py \
  notebooks/05_document_and_chunking_lab.ipynb \
  --mode offline
```

### Ollama

```bash
FINAI_CHAT_MODEL=qwen3:8b uv run --extra ai python \
  scripts/execute_notebooks.py notebooks/05_document_and_chunking_lab.ipynb \
  --mode live --provider ollama
```

### OpenAI

```bash
OPENAI_API_KEY=... FINAI_CHAT_MODEL=gpt-5-mini uv run --extra ai python \
  scripts/execute_notebooks.py notebooks/05_document_and_chunking_lab.ipynb \
  --mode live --provider openai
```

If a live contextual response is malformed, keep the failure visible. The lesson is also
about validating transformations, not hiding provider variance.

## Instructor notes

- Ask learners to predict which fixed window will split the NVIDIA table.
- Keep the raw versus normalized PDF figure visible while defining `DocumentBlock`.
- Avoid saying that `pdfplumber` solves every PDF. OCR, scanned pages and complex layouts
  require additional extraction strategies.
- Do not describe cosine similarity as probability or confidence.
- Record the selected embedding provider and model before interpreting the semantic curve.
- Say explicitly that contextual enrichment is bounded context generation, not agentic
  chunking. Agentic grouping belongs only in the optional extension discussion.
- When reviewing the scorecard, ask which metrics are hard constraints and which are
  optimization targets.
- End with: “We now trust the units. Next we improve how those units are represented and
  ranked.”

## Scope boundary and transition

Lesson 05 does not implement OCR, XBRL fact reconciliation, a vector database, metadata
filters, hybrid fusion or reranking. Docling, VLM parsing, proposition chunking and
iterative agentic grouping remain optional extensions after the core PASS marker.

Lesson 06 keeps the parsed blocks, chunks, provenance and maintained questions. It changes:

```text
lexical representation → embeddings
single retriever        → lexical + dense fusion
unfiltered candidates   → metadata-aware candidates
simple top-k            → reranked evidence set
```

## Sources

- [NVIDIA fiscal 2026 Form 10-K](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm)
- [Schneider Electric 2025 full-year results](https://www.se.com/ww/en/assets/564/document/528237/release-fy-results-2025.pdf)
- [pdfplumber documentation](https://github.com/jsvine/pdfplumber)
- [Beautiful Soup documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [OpenAI embeddings guide](https://platform.openai.com/docs/guides/embeddings)
- [Ollama embeddings documentation](https://docs.ollama.com/capabilities/embeddings)
