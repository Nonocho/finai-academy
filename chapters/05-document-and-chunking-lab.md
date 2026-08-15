# Lesson 05 — Financial Documents and Chunking Laboratory

**Schedule:** Day 1, 13:30–15:00

**Format:** 20 minutes of concepts and diagrams, 70 minutes of guided notebook work

**Capstone increment:** configurable, provenance-preserving ingestion and chunking pipeline

## Teaching outcome

Students should leave with one operational principle:

> Retrieval cannot recover a relationship that parsing or chunking has already destroyed.

They parse compact real-source NVIDIA HTML and Schneider Electric PDF fixtures, normalize
both into ordered `DocumentBlock` records, compare seven chunking strategies and reproduce
a table-integrity failure before retrieval or generation begins.

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

Adds strategy name, source block identifiers, optional parent link, role and contextual
text. Every chunk remains traceable to the exact blocks from which it was constructed.

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

### 5. Semantic boundaries depend on a representation and threshold

The notebook uses adjacent TF-IDF sentence similarity to make the mechanism observable.
The threshold is a versioned engineering parameter, not a universal constant. Lesson 06
changes the representation to embeddings.

### 6. Hierarchical retrieval separates matching from reading

Small children improve matching precision. Parent sections restore the evidence context
after retrieval. The parent-child link must be created during chunk construction.

### 7. Contextual and LLM-assisted representations add cost

Contextual prefixes can reduce ambiguous matches by adding company, period and section.
LLM propositions can create atomic claims, but add latency, cost and a transformation that
must be validated against the original evidence.

## Notebook pacing

| Time | Activity | Instructor emphasis |
|---:|---|---|
| 0–5 min | Verify manifest and fixture hashes | Provenance begins before parsing. |
| 5–12 min | Parse NVIDIA HTML and Schneider PDF | Different formats, same canonical block boundary. |
| 12–20 min | Inspect block order and types | Parsing creates ordered evidence, not a text dump. |
| 20–25 min | Compare raw PDF text with normalized table | Row-column relationships must survive. |
| 25–35 min | Run fixed, recursive and structure-aware strategies | Predict where each boundary will occur. |
| 35–43 min | Inspect boundary map | Chunk size alone does not express evidence safety. |
| 43–48 min | Run semantic threshold experiment | Representation and threshold jointly create boundaries. |
| 48–53 min | Inspect parent-child hierarchy | Retrieve a child, restore a verified parent. |
| 53–57 min | Add contextual prefixes | Extra tokens must earn their place. |
| 57–62 min | Generate propositions offline or live | Preserve numbers and source identifiers. |
| 62–67 min | Compare scorecard and retrieval recall | Hold questions and retriever constant. |
| 67–70 min | Reproduce failure and select policy | Diagnose the failed stage before changing models. |

## Visual teaching contract

The notebook produces nine executable figures:

1. source-to-retriever typed pipeline;
2. ordered block timelines for HTML and PDF;
3. raw PDF extraction versus canonical table block;
4. fixed, recursive and structure-aware boundary patterns;
5. semantic adjacent-similarity curve and threshold;
6. parent-child hierarchy;
7. chunk count and mean-size economics;
8. multi-metric strategy scorecard; and
9. fixed table split versus atomic structural table.

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

### 4. What is the purpose of parent-child chunking?

Use a small representation for precise retrieval, then expand to a larger verified section
for interpretation and generation.

### 5. Why not use LLM propositions as the only evidence representation?

They are transformed claims rather than original evidence. The model may omit qualifiers,
units or relationships. Keep source links and original blocks, validate outputs, and compare
their retrieval benefit against cost and transformation risk.

### 6. Does 100% retrieval recall prove a strategy is production-ready?

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

The deterministic laboratory runs without a model. Only proposition construction crosses
the provider boundary.

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

If a live proposition response is malformed, keep the failure visible. The lesson is also
about validating transformations, not hiding provider variance.

## Instructor notes

- Ask learners to predict which fixed window will split the NVIDIA table.
- Keep the raw versus normalized PDF figure visible while defining `DocumentBlock`.
- Avoid saying that `pdfplumber` solves every PDF. OCR, scanned pages and complex layouts
  require additional extraction strategies.
- Do not describe cosine similarity as probability or confidence.
- Explain that semantic chunking in this notebook uses a lexical proxy so the mechanism is
  visible; provider embeddings arrive in Lesson 06.
- When reviewing the scorecard, ask which metrics are hard constraints and which are
  optimization targets.
- End with: “We now trust the units. Next we improve how those units are represented and
  ranked.”

## Scope boundary and transition

Lesson 05 does not implement OCR, XBRL fact reconciliation, a vector database, production
embeddings, metadata filters, hybrid fusion or reranking.

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
- [scikit-learn TF-IDF documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
