# Lesson 04 — Naive RAG from First Principles

## Purpose and scope

Lesson 04 is a 30-minute bridge between the complete-context boundary demonstrated
in Lesson 03 and the document-ingestion laboratory in Lesson 05. Students build a
small retrieval-augmented generation pipeline that is intentionally naive and fully
observable.

The lesson must not imply that its prepared passages, lexical retrieval or simple
top-k policy are production quality. Its purpose is to expose the minimum RAG loop:

```text
prepared passages → index → retrieve → assemble context → generate → verify
```

The instructor and learner materials must consistently call this a **naive RAG
baseline**. The final synthesis must identify the limitations that Lessons 05 and 06
will improve.

## Learning outcome

Students can explain and implement RAG as two separately testable systems:

1. retrieval selects evidence from a corpus; and
2. generation answers from the selected evidence.

They can inspect why a passage was retrieved, distinguish retrieval failure from
generation failure, and explain why better parsing, chunking, embeddings, metadata
and ranking are needed.

## Financial case

The teaching corpus contains short, source-labelled evidence passages adapted from
official NVIDIA and Schneider Electric disclosures. The principal question concerns
the business segment that drove NVIDIA fiscal 2026 growth. Schneider passages act as
realistic cross-company distractors.

The corpus is already segmented for this lesson. This is an explicit simplification,
not an ingestion solution. Raw SEC HTML, PDF/XHTML parsing, table preservation,
metadata design and chunking strategy belong to Lesson 05.

## Technical design

The reusable implementation lives in `src/finai_academy/retrieval.py` and provides:

- an immutable evidence-passage model with stable source identifiers;
- a transparent TF-IDF index using `scikit-learn`;
- cosine-similarity ranking and deterministic top-k selection;
- prompt construction that preserves passage identifiers and provenance;
- small deterministic retrieval checks that do not require another model.

The notebook calls the existing provider-neutral model boundary. Its baseline path
runs offline, while Ollama and OpenAI remain selectable live providers.

No vector database, embedding API, reranker, query rewriting, agent or retrieval
framework is introduced in this lesson.

## Notebook narrative

The guided notebook takes approximately 22 minutes after an 8-minute conceptual
deck:

1. inspect the prepared NVIDIA–Schneider corpus;
2. build and inspect a TF-IDF representation;
3. calculate query-to-passage cosine similarity;
4. select the top-k evidence;
5. assemble a bounded, citation-bearing prompt;
6. generate an answer through offline, Ollama or OpenAI mode;
7. verify retrieval and grounding separately;
8. run a lexical-mismatch failure query;
9. map each observed limitation to Lessons 05 and 06.

## Visual teaching contract

The notebook must generate at least five meaningful executable visuals:

1. a corpus map separating NVIDIA and Schneider passages;
2. a query-term by passage heatmap derived from the TF-IDF matrix;
3. a similarity-ranking chart for every passage;
4. a top-k context-selection and token-allocation view;
5. a failure-to-improvement map connecting observed baseline weaknesses to the next
   lessons.

The deck must mirror these mechanisms with original editable diagrams. It should use
one primary claim per slide and finish with a clear boundary diagram:

```text
Naive RAG today
├── prepared passages
├── lexical similarity
└── simple top-k

Improved next
├── parsing and document structure       Lesson 05
├── chunking strategy and metadata       Lesson 05
└── embeddings, hybrid retrieval, rank   Lesson 06
```

Scores must never appear only as printed lists when a visual ranking can communicate
the same state.

## Failure laboratory

The failure query uses language that is semantically related to the relevant source
but shares weaker vocabulary with it. The visual ranking must show the relevant
passage losing position. The notebook must not invent a model-accuracy score or claim
that embeddings automatically solve every retrieval problem.

The exercise teaches that:

- retrieval quality depends on the representation of both source and question;
- top-k trades evidence coverage against context noise and cost;
- a fluent answer cannot repair missing evidence; and
- ingestion and retrieval improvements must be evaluated, not assumed.

## Deck sequence

1. RAG controls which evidence enters the prompt.
2. The naive baseline has five observable stages.
3. TF-IDF turns passages and the question into comparable weighted vectors.
4. Cosine similarity ranks evidence; top-k creates a decision boundary.
5. Retrieval and generation can fail independently.
6. The notebook builds the complete baseline on NVIDIA and Schneider.
7. Parsing, chunking and advanced retrieval improve specific weak points next.

Speaker notes contain source URLs for official financial facts and technical claims.
The footer is `First Finance - Arnaud Demes`.

## Verification requirements

- Unit tests cover ranking, deterministic ties, top-k validation, citation-preserving
  prompt construction and retrieval checks.
- The notebook restarts and runs in offline mode without network access.
- The executed notebook contains at least five code-generated figures.
- Its final output includes separate retrieval and grounding results plus an explicit
  `PASS — naive RAG baseline verified` marker.
- Repository validation, the full test suite and Ruff pass.
- Every slide is rendered and inspected individually for overflow, wrapping,
  connector order, source notes and footer consistency.

## Out of scope

- raw-document parsing;
- PDF table extraction;
- production chunking strategy;
- embeddings and vector indexes;
- filtered or hybrid retrieval;
- reranking and query transformation;
- production RAG evaluation.

These omissions must be presented as deliberate learning boundaries rather than
missing implementation work.
