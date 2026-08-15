# Session 05 — Financial Documents and Chunking Laboratory

## Purpose

Lesson 05 replaces the prepared-passage assumption from Lesson 04 with an observable
document-ingestion pipeline. Learners parse compact, real-source NVIDIA SEC HTML and
Schneider Electric PDF extracts, normalize them into one canonical block model, compare
seven chunking approaches, and measure how chunk boundaries change retrieval quality.

The session runs from 13:30 to 15:00 on Day 1: 20 minutes of slides and 70 minutes of
guided notebook work.

## Design decision

Three approaches were considered:

1. A broad conceptual survey of every chunking strategy. This fits the schedule but
   leaves learners unable to inspect the implementation.
2. A deep implementation of only fixed and recursive chunking. This is accessible but
   omits the semantic, hierarchical and LLM-assisted approaches that differentiate the
   course.
3. A controlled comparative laboratory. Learners implement and inspect the deterministic
   core, then run semantic, hierarchical, contextual and LLM-assisted variants through a
   common interface and evaluator.

The laboratory uses approach 3. It preserves technical depth while keeping the class
focused on engineering decisions rather than framework syntax.

## Learning outcome

At the end of the lesson, a learner can:

- explain why PDF text extraction is not document understanding;
- parse SEC-style HTML and a machine-generated financial PDF;
- preserve order, page, heading, table and source provenance;
- normalize heterogeneous inputs into `DocumentBlock` records;
- compare fixed, recursive, structure-aware, semantic, hierarchical, contextual and
  LLM-assisted chunking;
- diagnose a retrieval failure caused by a broken chunk boundary; and
- defend one production chunking policy using measured evidence.

## Financial source contract

The repository remains lightweight. It stores compact classroom fixtures derived from
official NVIDIA and Schneider Electric disclosures plus a versioned manifest. A fetch
script can download the complete official documents for extended experiments, but the
core notebook and tests never require network access.

Every source record contains:

- stable source identifier;
- company and reporting period;
- document type and language;
- official URL;
- retrieval date;
- local path and content hash; and
- a clear `fixture`, `downloaded`, or `recorded` provenance mode.

The notebook must never describe a recreated teaching extract as an original page. Any
compact fixture explicitly identifies how it relates to the official source.

## Canonical document model

`DocumentBlock` is the boundary between parsing and chunking. It contains:

- `block_id` and `source_id`;
- `company`, `period`, `document_type`, and `language`;
- `page_number` when available;
- ordered `section_path` headings;
- `block_type`: heading, paragraph, table, list, footnote, or page marker;
- normalized text;
- optional structured table rows; and
- `source_url` and ordinal position.

`DocumentChunk` contains a stable chunk identifier, strategy name, text, source block
identifiers, provenance metadata, optional parent identifier, and contextual prefix.
Every chunk must be traceable to at least one source block.

## Parsing pipeline

```text
official HTML / PDF
        ↓
raw elements, pages and tables
        ↓
normalization into ordered DocumentBlock records
        ↓
quality checks: order, headings, table integrity, provenance
        ↓
chunking strategy
```

The HTML parser recognizes heading hierarchy, paragraphs, lists and tables. The PDF
parser uses `pdfplumber`, keeps page numbers, extracts tables before plain text, and
removes duplicated table text where practical. OCR and scanned-document recovery are
named production concerns but remain out of scope for the 90-minute lesson.

## Chunking strategies

All strategies consume the same ordered blocks and return the same `DocumentChunk`
interface.

1. **Fixed-size with overlap** provides a deliberately naive baseline.
2. **Recursive** prefers paragraph and sentence boundaries before hard splitting.
3. **Structure-aware** keeps headings with their content and treats tables atomically.
4. **Semantic boundary** groups adjacent sentences until their similarity falls below a
   visible threshold. Offline mode uses a deterministic TF-IDF similarity proxy so the
   boundary algorithm remains observable; provider embeddings belong to Lesson 06.
5. **Hierarchical parent-child** indexes small child chunks while preserving larger parent
   sections for context expansion.
6. **Contextual enrichment** prefixes each chunk with compact company, period, document
   and section context.
7. **LLM-assisted propositions** converts a block into atomic, source-bound statements.
   OpenAI and Ollama use the shared model gateway; offline mode uses recorded propositions
   for the same source blocks.

The lesson does not claim that one strategy is universally best. Strategy selection is a
measured decision tied to document structure, question types, latency and cost.

## Controlled comparison

Every strategy is evaluated with the same fixture corpus, four maintained analyst
questions and the Lesson 04 lexical retriever. The scorecard shows:

- number of chunks;
- mean and maximum chunk size;
- percentage of chunks retaining headings;
- table-integrity pass rate;
- provenance completeness;
- retrieval recall at k; and
- context characters selected for the maintained questions.

This does not yet compare embedding models or hybrid ranking. Those variables remain
fixed until Lesson 06 so learners can attribute observed changes to chunk construction.

## Failure laboratory

A fixed-size boundary separates a financial table's column heading from the associated
row and splits a section heading from its explanatory paragraph. The resulting chunk is
lexically retrievable but insufficient to interpret the number safely.

Learners compare the same query under fixed, recursive and structure-aware chunking and
answer three diagnostic questions:

1. Was the source parsed incorrectly?
2. Was the evidence separated during chunking?
3. Did ranking fail despite a coherent chunk?

The intended diagnosis is a chunk-construction failure before retrieval and generation.

## Notebook visual contract

The executable notebook produces at least eight readable figures:

1. raw source formats flowing into the canonical block model;
2. page/order view of parsed block types;
3. a side-by-side raw PDF extraction and normalized table view;
4. visible chunk boundaries for fixed, recursive and structure-aware strategies;
5. semantic sentence-similarity curve with boundary threshold;
6. parent-child hierarchy for one financial section;
7. strategy scorecard and size distribution; and
8. retrieval outcome comparison plus the Lesson 06 handoff.

Figures show complete labels and source identifiers. Tables may be used where exact
values are more useful than decorative charts.

## Deck narrative

The companion deck contains eight low-density slides:

1. the lesson outcome and schedule;
2. why financial PDFs are visually structured but text extraction is linear;
3. the ingestion pipeline and canonical block boundary;
4. what metadata must survive parsing;
5. three core chunking boundary patterns;
6. semantic, hierarchical, contextual and LLM-assisted patterns;
7. the controlled failure and comparison scorecard; and
8. the production decision rule and transition to embeddings/hybrid retrieval.

Every diagram is editable and maps to a notebook output. The footer is
`First Finance - Arnaud Demes`.

## Provider behavior

Parsing and deterministic chunking never require an LLM. The LLM-assisted strategy uses
the existing provider-neutral gateway:

- `offline`: recorded, deterministic propositions;
- `ollama`: a configured local chat model; and
- `openai`: a configured OpenAI model and API key.

If a live provider is unavailable, the notebook explains the missing prerequisite and
continues in offline mode rather than breaking the laboratory.

## Testing and acceptance criteria

The implementation is complete when:

- parser tests prove page, section, order, table and source preservation;
- chunking tests prove overlap bounds, table atomicity, parent-child links and provenance;
- the controlled fixed-size failure is reproduced by a test;
- semantic boundaries are deterministic for a fixed similarity representation;
- recorded LLM propositions preserve source identifiers;
- the source notebook passes the course contract and executes end-to-end offline;
- OpenAI and Ollama code paths are present and diagnosed live when configured;
- all notebook figures render without clipping;
- every deck slide is rendered and inspected at full size;
- the presentation overflow test passes; and
- the full repository test, lint and validation suite remains green.

## Scope boundary

Lesson 05 does not implement OCR, XBRL fact reconciliation, vector-database persistence,
production embeddings, metadata filters, hybrid fusion or reranking. These are either
advanced ingestion concerns or Lesson 06 retrieval concerns. The capstone increment is
a configurable, provenance-preserving ingestion and chunking pipeline.
