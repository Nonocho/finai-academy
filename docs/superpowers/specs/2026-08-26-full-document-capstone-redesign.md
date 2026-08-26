# Full-Document Financial Analyst Capstone Redesign

**Date:** 2026-08-26

**Status:** Approved in conversation; awaiting review of this written specification

**Course:** AI Engineering for Asset Management

**Scope:** Replace the capstone's small evidence catalog with a visible, table-aware pipeline over official NVIDIA and Schneider Electric documents.

## 1. Purpose

The capstone must prove that students can turn a real financial PDF into reliable AI evidence. The central learning journey is:

```text
official PDF
  -> layout and table extraction
  -> contextualized document elements
  -> table-aware chunks
  -> metadata-filtered hybrid retrieval
  -> cited structured answer
  -> trace, evaluation, and release decision
```

The application must make this journey visible. A learner should be able to point to one table in an original report, see how the system extracted and contextualized it, see the exact chunk retrieved, and verify the claim generated from it.

This redesign keeps the existing typed orchestration, evidence gate, recorded route, MLflow evaluation, and Streamlit delivery where they remain useful. It changes the data foundation and simplifies the user experience.

## 2. Success criteria

The redesign succeeds when:

1. The mandatory route uses complete, official NVIDIA and Schneider Electric PDF documents rather than a hand-written evidence catalog.
2. Complex tables retain their titles, headers, units, footnotes, values, page locations, and surrounding context.
3. Retrieved chunks include enough metadata to interpret every material number correctly.
4. Every generated factual claim links to a stable document element and visible PDF page or crop.
5. The user interface states the task in plain language and presents the answer before implementation detail.
6. A student can explain and inspect the entire evidence chain in a 60-minute capstone session.
7. The mandatory experience works offline from versioned extraction artifacts; a live OpenAI call is optional and explicit.
8. Extraction, retrieval, citation, and agent behavior have deterministic acceptance tests.

## 3. Product boundaries

### 3.1 Mandatory capstone

The mandatory capstone focuses on official financial documents for NVIDIA and Schneider Electric. It includes:

- real PDF ingestion;
- paragraph and table extraction;
- contextual metadata;
- table-aware chunking;
- metadata-filtered hybrid retrieval;
- bounded tool use;
- cited structured synthesis;
- a visible evidence chain;
- MLflow tracing and deterministic evaluation; and
- a recorded offline route.

### 3.2 Optional extensions

Market snapshots, current news, Ollama, and custom questions may remain available as clearly labelled extensions. They must not obscure the mandatory document-learning path or determine classroom success.

### 3.3 Non-goals

- Building a general-purpose document platform.
- Supporting arbitrary companies or every PDF layout.
- Promising perfect extraction from every financial report.
- Running OCR on every page.
- Autonomous trading, investment recommendations, or unrestricted browsing.
- Rebuilding a multi-agent architecture.
- Exposing low-level configuration on the first screen.

## 4. Authoritative document set

The certified document set contains:

1. NVIDIA FY2026 Annual Report from NVIDIA Investor Relations.
2. Schneider Electric FY2025 Full Year Results from Schneider Electric.

Each source record contains a stable `document_id`, company, document type, reporting period, publication date, official source URL, local asset path, byte size, and SHA-256 hash. The repository manifest verifies the downloaded assets before ingestion.

The full PDFs are the source of truth. Cached Markdown, JSON, embeddings, and page crops are derived artifacts and must retain a link to the source document hash and extraction version.

## 5. Architecture

```text
Official PDF assets
        |
        v
Document parser interface
  |- PyMuPDF4LLM primary adapter
  |- native-text-first OCR policy
  `- typed extraction diagnostics
        |
        v
Document elements + contextual metadata
        |
        v
Table-aware chunk builder
        |
        v
Versioned local retrieval index
  |- metadata filters
  |- BM25 rank
  `- dense rank + fusion
        |
        v
Bounded research service
  |- document search tool
  |- evidence inspection tool
  |- deterministic comparison tool
  `- evidence gate
        |
        v
Structured synthesis
  |- recorded offline route
  |- OpenAI Responses API route
  `- optional Ollama route
        |
        v
Simple Streamlit workspace
  |- Answer
  |- Evidence
  `- How it worked
        |
        v
MLflow trace + deterministic evaluation
```

Every boundary is typed and independently testable. The UI receives sanitized public view models and contains no parsing, retrieval, agent-policy, or evaluation logic.

## 6. Document extraction

### 6.1 Parser interface

The application depends on a parser protocol rather than directly on PyMuPDF4LLM:

```python
class FinancialDocumentParser(Protocol):
    def parse(self, document: DocumentSource) -> ParsedDocument: ...
```

The primary adapter uses PyMuPDF4LLM because the feasibility benchmark preserved the target NVIDIA table as one 14-by-4 table and the Schneider page as three separate 6-column tables. `pdfplumber` may remain a diagnostic or basic-text fallback, but it is not treated as an equivalent complex-table parser.

PyMuPDF and PyMuPDF4LLM licensing is AGPL or commercial. The dependency must be isolated behind the parser interface, its license must be documented, and distribution suitability must be reviewed before commercial course release.

### 6.2 OCR policy

The parser uses native PDF text first. OCR is attempted only when a page has no usable text layer or fails a defined text-quality check. OCR use, engine, language, and confidence are recorded in metadata. OCR failure produces a typed page diagnostic; it never silently substitutes uncertain text.

### 6.3 Immutable extraction

The original parser output is stored unchanged as a versioned extraction artifact. Enrichment creates additional fields but never overwrites the source text, Markdown table, bounding box, or page number.

Personal filesystem paths returned by parser libraries are removed before persistence, tracing, or UI rendering.

## 7. Typed document model

### 7.1 Source record

`DocumentSource` contains:

- `document_id`
- `company_name`
- `ticker`
- `document_type`
- `reporting_period`
- `publication_date`
- `official_source_url`
- `local_asset_key`
- `sha256`
- `page_count`

### 7.2 Extracted element

`DocumentElement` contains:

- stable `element_id` derived from document hash, page, type, and content;
- `document_id`;
- physical PDF page and optional printed page;
- element type: heading, paragraph, list, table, figure caption, header, footer, or footnote;
- bounding box;
- original text or Markdown;
- parser confidence and diagnostic flags;
- parent and neighboring element IDs; and
- table dimensions where applicable.

### 7.3 Contextual metadata

Contextual metadata is a required part of every retrievable chunk, not an optional tagging step.

**Source context**

- company and ticker;
- document type;
- reporting period and publication date;
- source URL and document hash.

**Structural context**

- physical and printed page;
- section and heading path;
- element type;
- table title, caption, and identifier;
- bounding box;
- parent, previous, and next element IDs.

**Financial context**

- normalized metric name when supported by evidence;
- period represented by each row or column;
- currency and scale, such as USD millions;
- segment, geography, or reporting scope;
- GAAP or non-GAAP classification when explicitly stated;
- audited status when explicitly stated; and
- associated footnotes.

**Lineage context**

- parser and parser version;
- extraction timestamp and extraction-schema version;
- native-text or OCR method;
- enrichment method and confidence;
- source element IDs;
- chunk content hash; and
- retrieval-index version.

Metadata from the document structure is deterministic. `gpt-5.6-luna` may normalize or enrich financial labels through a strict structured-output schema, but an inferred value must include supporting element IDs and confidence. Unsupported fields remain null; the model must not guess.

## 8. Table-aware chunking

Paragraphs are grouped under their nearest heading up to a defined token budget. Tables follow different rules:

1. A normal-sized table remains atomic.
2. Multi-row and multi-column headers remain attached to all values.
3. The chunk includes the table title, full heading path, units, currency, footnotes, page, source, and a short nearby explanatory paragraph.
4. Rows are not separated from their headers.
5. A very large table may be split only by logical row groups, with headers and context repeated in every child chunk.
6. The original Markdown table and structured cell matrix are both retained.

A table chunk therefore answers not only “what number was found?” but also “which company, metric, period, unit, segment, and source does it belong to?”

## 9. Retrieval and tools

### 9.1 Retrieval

The retriever applies company, reporting-period, document-type, and element-type filters before ranking. It then combines:

- BM25 for exact financial terms and reported figures;
- dense retrieval for semantic meaning; and
- deterministic rank fusion.

Each hit exposes rank components, fused score, chunk ID, element IDs, and source location. Table hits are explicitly identifiable so a query seeking a reported value can prefer table evidence without excluding relevant narrative context.

### 9.2 Mandatory tool registry

The bounded research service exposes a small registry:

1. `search_financial_documents` — returns ranked, metadata-rich chunks.
2. `inspect_document_evidence` — returns the exact element, table, page location, and neighboring context for a selected hit.
3. `compare_reported_values` — performs deterministic arithmetic only from displayed, cited inputs.

The document tools are exposed through the existing local MCP boundary so the capstone still demonstrates Lessons 10 and 11. Market and news tools remain optional extensions.

### 9.3 Evidence gate

No answer is released unless:

- each reported fact maps to at least one stable evidence element;
- every displayed value retains its period, unit, company, and source page;
- cross-company comparisons state material scope differences;
- interpretations are separated from reported facts; and
- cited element and document hashes match the certified index.

## 10. Model routes

The default OpenAI model is `gpt-5.6-luna`, called through the Responses API with strict structured outputs. The initial reasoning baseline is `medium`; `low` may replace it only after evaluation shows no material quality loss.

Luna is used for bounded tasks:

- optional metadata normalization with evidence references;
- query planning;
- selection of retrieved evidence;
- structured, cited synthesis; and
- explicit limitations.

It is not used to invent source metadata, parse table geometry, calculate displayed metrics, or decide deterministic release status.

The recorded route returns a clearly labelled, versioned result built from the same certified chunks and public view models. It remains the classroom and automated-test fallback. Ollama may remain an optional local route.

## 11. Simple user interface

The current interface gives configuration and process detail too much visual priority. The redesigned interface starts with the learner's job and uses plain language.

### 11.1 First screen

**Page title:** `Financial Document Analyst`

**Introductory text:** `Ask a financial question and see the exact report page and table behind the answer.`

The first screen contains only:

1. a clearly labelled example question;
2. one primary `Analyze the reports` button;
3. a one-sentence description of the NVIDIA and Schneider document set; and
4. a collapsed `Advanced settings` section.

Provider, model, data mode, readiness diagnostics, and reset controls move into `Advanced settings`. Defaults should allow the certified route to run without configuration.

### 11.2 Result order

After a run, the interface presents three plain-language tabs in this order:

1. `Answer`
2. `Evidence`
3. `How it worked`

**Answer** shows a short conclusion, company evidence, comparison limits, and citations. It does not begin with the research plan or tool log.

**Evidence** shows the original PDF page or cropped table beside the extracted Markdown/table. Below it, the interface shows the exact retrieved chunk and contextual metadata. Labels use learner language such as `Original report`, `Extracted table`, `Why this evidence was selected`, and `Source details`.

**How it worked** shows a compact five-step pipeline, followed by collapsed retrieval scores, tool activity, trace, MLflow link, and evaluation details. Advanced diagnostics do not compete with the answer.

### 11.3 Copy rules

- Every screen and section has a title that states its purpose.
- Introductory text is one or two short sentences.
- Buttons use verbs and describe the result of the action.
- Internal terms such as `trajectory`, `evidence gate`, `fusion score`, and `typed stop` are translated in the primary UI; exact technical names may appear in advanced diagnostics.
- Empty states explain what the learner should do next.
- Errors state what failed, what evidence is unavailable, and the safe next action.
- No title duplicates another title on the same screen.
- The course signature appears once in a quiet footer rather than competing with the page title.

### 11.4 Visual hierarchy

The primary desktop view uses one reading column for the question and answer. The evidence comparison may use two columns because the visual relationship is important. Tables must fit without horizontal ambiguity at the certified viewport, and long metadata is grouped rather than displayed as a raw dictionary.

## 12. Student learning experience

The 60-minute student task is to complete meaningful seams in the document pipeline rather than only wiring prebuilt application components.

The scaffold provides certified source documents, parser adapters, typed contracts, a partial index builder, and a working recorded model route. Students complete four bounded tasks:

1. preserve table context when creating a chunk;
2. attach required source and financial metadata;
3. apply metadata filters before hybrid ranking; and
4. map a cited answer back to its document element and page.

The diagnostic exercise gives students one deliberately broken table chunk whose number lacks a unit or header. They must explain why the answer is unsafe and repair the chunk or metadata mapping.

The complete reference solution stays available for instructor demonstration. Student verification runs without network access.

## 13. Error handling and observability

Typed failures include:

- source asset missing or hash mismatch;
- unsupported or encrypted PDF;
- unusable text layer;
- OCR unavailable or unsuccessful;
- table shape or header validation failure;
- required metadata missing;
- empty filtered retrieval set;
- provider unavailable;
- insufficient cited evidence; and
- trace or persistence failure.

A parser or provider failure cannot produce a released answer. The public UI receives sanitized explanations and next actions; raw exceptions, credentials, and personal file paths stay out of UI state and MLflow artifacts.

The MLflow trace records document and index versions, filters, retrieval candidates and scores, selected chunk IDs, tool calls, structured-output validation, evidence-gate results, latency, token use, and deterministic evaluation. It stores stable asset keys rather than personal paths.

## 14. Verification

### 14.1 Extraction contracts

- NVIDIA target page produces one 14-by-4 table.
- The NVIDIA FY2026 row contains Compute & Networking `193,479`, Graphics `22,459`, and Total `215,938`, with USD millions preserved.
- Schneider Electric target page produces three distinct six-column tables.
- The Schneider FY2025 table contains Energy Management `33,130` and `+10.3%`, Industrial Automation `7,022` and `+3.0%`, and Group `40,152` and `+8.9%`.
- All target tables retain title or heading context, page, source, dimensions, and bounding box.

### 14.2 Chunk and retrieval contracts

- No target table value appears in a retrievable chunk without company, period, unit, headers, page, and source.
- Stable IDs and hashes are reproducible across identical builds.
- Company and period filters run before ranking.
- Hybrid results expose BM25, dense, and fused rank information.
- A maintained query set retrieves the expected table or narrative element within the accepted rank threshold.

### 14.3 Answer and agent contracts

- Every reported claim maps to a valid evidence element.
- Calculations use displayed cited inputs and deterministic functions.
- Unsupported comparisons are qualified or stopped.
- Recorded and OpenAI routes produce the same public schema.
- Agent step, replan, duplication, and capability limits remain enforced.

### 14.4 Interface contracts

- The page opens with `Financial Document Analyst` and a one-sentence task explanation.
- The default route can run without changing Advanced settings.
- Results appear in `Answer`, `Evidence`, `How it worked` order.
- Evidence shows a PDF page or crop beside its parsed representation.
- Primary UI copy avoids unexplained implementation jargon.
- The certified desktop viewport has no clipped, overlapping, or unreadable content.
- Browser screenshots cover the initial state, completed result, evidence comparison, diagnostics, and representative failure state.

### 14.5 Release verification

- Unit and integration tests pass in isolated and full-suite execution.
- Notebook and capstone offline routes execute from a clean environment.
- Manifest verification passes for every source and derived artifact.
- No API key, personal path, or raw provider exception appears in repository files, MLflow output, screenshots, or public view models.

## 15. Repository impact

Implementation is expected to update:

- document assets and the course-data manifest;
- dependency declarations and licensing documentation;
- new focused ingestion, element, chunking, and indexing modules under `src/finai_academy/capstone/`;
- the existing capstone tool, service, model-gateway, view, and Streamlit layers;
- capstone student and instructor materials;
- recorded reference artifacts;
- extraction, retrieval, evidence, UI, and full-suite tests; and
- capstone README and product specification.

The existing capstone modules should be reused where their contracts remain valid. Parsing, chunking, retrieval, orchestration, presentation, and evaluation must stay in separate focused modules.

## 16. Delivery sequence

This capstone redesign is one architectural workstream. Course-wide packaging repairs discovered during the audit—missing notebook headings, deck-note contract drift, the Lesson 06 stale assertion, and full-suite notebook-render isolation—are a separate bounded release-cleanup workstream. They should be completed before declaring the whole course release-ready, but they must not be mixed into the capstone architecture implementation.

The capstone implementation sequence is:

1. certify the real PDF assets and extraction contracts;
2. build typed elements and contextual metadata;
3. build table-aware chunks and the versioned hybrid index;
4. connect document tools, evidence inspection, and the evidence gate;
5. standardize the Luna Responses API and recorded routes;
6. simplify the Streamlit views and copy;
7. update the student exercise and documentation; and
8. run extraction, retrieval, browser, privacy, and full-suite verification.
