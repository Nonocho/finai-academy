# MLE-Inspired Course Alignment and Lesson 07 Design

## Status

Validated direction from Arnaud Demes on 16 August 2026. This specification
defines the pedagogical alignment of Lessons 01–06 and the design of Lesson 07.
It does not reproduce MLExpert Academy code or copy its lesson text. It adapts
the effective top-down teaching progression to a two-day finance workshop with
original examples, implementation, diagrams, exercises, and verification.

## Goal

Make the first six lessons follow one consistent professional AI-engineering
progression, then add a complete RAG evaluation and tracing lesson with MLflow
as the primary platform and Ragas as a short comparison.

The finished sequence must remain accessible to technically curious finance
professionals, preserve the existing timetable, and build one cumulative
Financial Analyst Copilot over NVIDIA and Schneider Electric evidence.

## Non-negotiable constraints

- Keep the exact lesson slots in `course.yml`.
- Keep the workshop notebook-first and suitable for a two-day classroom.
- Every live notebook must support Ollama and OpenAI through shared provider
  boundaries.
- Every notebook must also have a deterministic offline execution path for
  regression testing and instructor recovery.
- Never require an OpenAI key for the offline or Ollama paths.
- Do not require Docker, Supabase, PostgreSQL, Docling, a VLM, or a hosted
  observability service for the classroom core.
- Production tools may appear as explicit extensions, comparison cells, or
  architecture bridges.
- Use NVIDIA as the principal US case and Schneider Electric as the principal
  European case.
- Preserve source provenance, evidence typing, abstention, and non-advice
  boundaries throughout.
- Keep slide copy factual, professional, and visually explanatory.
- Use the footer `First Finance - Arnaud Demes` on every course deck.
- Source notebooks committed to Git must have no executed outputs.

## Course-wide lesson contract

Every implemented lesson from 01 onward follows the same observable sequence:

1. **Why and architecture** — the business problem, product increment, and
   system boundary are shown visually before code.
2. **Build** — students run the smallest useful implementation.
3. **Observe** — a controlled failure exposes the next engineering need.
4. **Understand** — the mechanism and trade-off are made visible.
5. **Improve** — the baseline is changed in one inspectable way.
6. **Verify** — named deterministic checks produce one final PASS marker.
7. **Knowledge check** — three to five short questions test the decision rules.
8. **Engineering mission** — a bounded challenge extends the capstone.
9. **Professional extension** — optional production patterns are clearly
   separated from the timed classroom path.

Each notebook must contain learning objectives, prerequisites, expected output
guidance, failure lab, troubleshooting, verification, challenge, capstone
integration, and recap. Each instructor chapter must contain exact pacing,
checkpoint answers, provider commands, likely student errors, and transition
logic.

## Observability progression

MLflow is introduced early and taught deeply later:

| Lesson | Observable record |
|---|---|
| 01 | first provider-neutral model trace and normalized run metadata |
| 02 | prompt version, schema version, validation outcome |
| 03 | context budget, route, generation span |
| 04 | retrieval and generation spans |
| 05 | parser, chunking strategy, construction metrics |
| 06 | filter, lexical, dense, fusion, and reranking spans |
| 07 | datasets, complete traces, evaluation runs, comparisons, and analysis |

The core classroom setup uses a local MLflow tracking store. Docker deployment
is an architecture extension, not a prerequisite. The shared tracing boundary
must degrade explicitly and safely in deterministic offline tests; it must not
silently claim a trace that was not created.

## Lesson 01 — Local and hosted model gateway

### Keep

- One provider-neutral gateway for Ollama and OpenAI.
- Explicit chat-message roles.
- NVIDIA evidence card and the ambiguous-question failure lab.
- Grounding checks and normalized latency.

### Add

- Normalized input, output, and total token usage when the provider supplies it.
- Explicit temperature, seed, context-window, and output-budget explanations.
- One small streaming demonstration in the core notebook.
- A first local MLflow trace containing provider, model, latency, prompt
  version, token metadata, and outcome without secrets.
- A visual roadmap placing structured output, memory, tools, LangGraph,
  evaluation, and MCP in later lessons.
- Manual conversation history as an optional extension.

### Classroom boundary

The 30-minute lesson remains 10 minutes of concepts and 20 minutes of notebook
work. Streaming is observable but not developed into an API or UI.

## Lesson 02 — Prompt engineering and structured outputs

### Keep

- `AnalystBrief` as the finance product contract.
- Pydantic structured generation for Ollama and OpenAI.
- Syntax, schema, finance-semantic, and application-acceptance validation.
- Trusted company and reporting-period inputs.

### Add

- The six-part prompt framework: role, task, context, instructions, output,
  and examples.
- One visible progression using the same NVIDIA evidence:

  ```text
  vague request
    -> six-part prompt
    -> delimited untrusted source
    -> few-shot edge case
    -> prompt-only JSON
    -> schema-bound output
    -> finance acceptance
  ```

- A prompt-injection example that treats source contents as data.
- Positive, explicit instructions in place of ambiguous negative phrasing.
- A short model-selection decision between high-volume worker tasks and tasks
  requiring stronger reasoning.
- Prompt and schema versions in the MLflow record.
- A visual comparison of output acceptance across the prompt stages.

### Reasoning-language boundary

The lesson may teach task decomposition and explicit intermediate criteria. It
must not require the model to reveal hidden chain-of-thought. Students evaluate
the output and observable application state.

## Lesson 03 — Context engineering and CAG

### Keep

- The complete context-budget gate.
- Stable-prefix construction and repeated questions.
- The distinction between observed latency and proven cache use.
- Lost-in-the-middle and CAG-versus-RAG decision visuals.

### Add

- A minimal product architecture from source to complete-context answer.
- A decision table distinguishing context, prompt cache, conversation memory,
  grounding, and retrieval.
- MLflow attributes for estimated input allocation, route, provider, and
  observed latency.
- An optional beginning/middle/end evidence-position experiment whose live
  output is reported as an observation, not a guaranteed ranking.
- The common knowledge check and checkpoint-success structure.

## Lesson 04 — Naive RAG from first principles

### Keep

- Transparent TF-IDF and cosine similarity.
- Visible full ranking followed by top-k.
- Evidence-labelled context construction.
- Separate retrieval and answer verification.
- The lexical mismatch failure that motivates later embeddings.

### Add

- A no-context baseline before the RAG answer.
- A minimal, explicitly naive paragraph split so students see where the
  prepared evidence units originate.
- A three-path comparison: no context, full context, and retrieval.
- MLflow spans for retrieval, context assembly, and generation.
- A checkpoint that the student can explain Retrieve, Augment, Generate and
  name one failure in each stage.

## Lesson 05 — Financial documents and chunking laboratory

### Keep

- `pdfplumber` and BeautifulSoup as accessible baseline parsers.
- The canonical `DocumentSource`, `DocumentBlock`, and `DocumentChunk` models.
- HTML/PDF normalization, table failure, structure-aware chunking,
  parent-child chunks, provenance, and comparative retrieval evaluation.

### Parsing ladder

The core notebook demonstrates basic extraction and visual verification. The
deck explains when a financial document requires layout-aware parsing, OCR, or
a VLM. Docling may be demonstrated in an optional cell but must not be required
for the classroom PASS contract.

### Core chunking progression

1. Fixed-size failure.
2. Recursive fallback.
3. Structure-aware chunks with table protection.
4. Semantic boundaries using provider embeddings in live mode and versioned
   recorded similarities offline.
5. Hierarchical parent-child representation.
6. Deterministic metadata prefix baseline.
7. LLM-generated contextual enrichment.
8. Cost and retrieval comparison.

### LLM contextual enrichment contract

Add a provider-neutral function that receives document context, section scope,
and original chunk text. It returns one or two sentences that place the chunk
inside the complete document. The implementation must:

- support Ollama, OpenAI, and recorded offline responses;
- preserve the original chunk text separately;
- construct retrieval text as generated context plus original text;
- preserve block IDs, page numbers, section path, source URL, company, period,
  and document type;
- reject empty or structurally invalid outputs;
- measure enrichment latency and token inflation; and
- never present generated context as source evidence.

Proposition chunking remains an advanced extension. True iterative agentic
grouping is introduced as a separate pattern and is not mislabeled as
contextual enrichment.

### Classroom pacing

The 90-minute slot remains 20 minutes of concepts and 70 minutes of notebook
work. Proposition chunking and agentic grouping move outside the required timed
path if necessary to protect parsing, structure, semantic, hierarchical, and
contextual-enrichment understanding.

## Lesson 06 — Embeddings and hybrid retrieval

### Keep

- Provider-neutral embeddings with a deterministic offline index.
- Full-dimensional cosine scores and clearly labelled teaching projections.
- Metadata pre-filtering before retrieval.
- Lexical and dense channels, reciprocal-rank fusion, transparent reranking,
  provenance, abstention, and index versioning.

### Add

- A storage bridge comparing the classroom local index with PostgreSQL and
  pgvector.
- A minimal documents/chunks/embeddings schema diagram.
- A conceptual explanation of exact search versus HNSW approximate search.
- A conceptual comparison of TF-IDF, PostgreSQL full-text search, and BM25.
- MLflow spans for every visible retrieval stage.
- Optional FlashRank comparison without replacing the transparent classroom
  reranker.
- Optional HyDE query expansion that is evaluated in Lesson 07 rather than
  asserted as an automatic improvement.

Docker, Supabase, PostgreSQL, HNSW, FlashRank, and HyDE are professional
extensions. The core 45-minute PASS contract remains local and deterministic.

## Lesson 07 — RAG evaluation and tracing

### Schedule and outcome

- **Schedule:** Day 1, 16:00–16:45.
- **Format:** 15 minutes of concepts and diagrams, 30 minutes of guided notebook.
- **Capstone increment:** versioned RAG evaluation dataset, complete MLflow
  traces, comparative evaluation runs, and failure analysis.

Students must leave able to answer:

> When a financial RAG answer is wrong, did retrieval fail, did generation fail,
> or did the evaluation contract fail to represent the task?

### Evaluation architecture

```text
versioned evaluation cases
        |
        v
retrieval pipeline -> context -> model answer
        |                         |
        +---------- trace --------+
                    |
                    v
 deterministic metrics + optional LLM judges
                    |
                    v
        MLflow run comparison and failure table
```

### Golden dataset

The notebook builds a small versioned dataset using the same NVIDIA and
Schneider evidence used in Lessons 05 and 06. It includes:

- direct factual questions;
- exact-number and identifier questions;
- semantic paraphrases;
- company and period filtering cases;
- multi-evidence questions;
- insufficient-evidence cases requiring abstention; and
- one controlled cross-company leakage case.

Each case records stable case ID, question, filters, expected evidence IDs or
tokens, expected answer facts, whether abstention is required, and tags.

### Metrics taught from first principles

- Retrieval recall at k.
- Reciprocal rank or mean reciprocal rank for expected evidence.
- Metadata-filter correctness.
- Citation correctness against retrieved passage IDs.
- Deterministic grounded-fact coverage.
- Abstention correctness.
- Latency and, when available, token usage.

Retrieval metrics remain separate from answer metrics. A fluent final answer
cannot repair missing evidence, and perfect retrieval does not prove a grounded
answer.

### MLflow role

MLflow is the primary experiment and observability package. The lesson must:

- run with a local tracking store;
- record nested or linked spans for retrieval, context construction, and
  generation;
- log dataset version, provider, model, embedding index version, retrieval
  configuration, prompt version, and schema version;
- log per-case and aggregate metrics;
- compare at least two retrieval configurations;
- expose one failed trace and identify its failing stage; and
- produce an exportable failure-analysis table.

The core lesson must not depend on opening the MLflow browser UI to pass. The
notebook displays the same essential run and trace summary inline, while the UI
remains the instructor demonstration surface.

### LLM-as-judge boundary

The core evaluation uses deterministic checks. An optional judge may score
answer relevance or groundedness with Ollama or OpenAI, but it must record the
judge provider and model and must never silently select a default model.

Judge scores are presented as model-dependent measurements requiring
calibration, not objective truth.

### Ragas extension

Ragas appears only after students have implemented and understood the core
metrics. The extension:

- converts the same golden cases into a Ragas evaluation dataset;
- runs at most two RAG-specific metrics, initially context recall and
  faithfulness;
- receives an explicit Ollama or OpenAI judge configuration;
- uses recorded outputs or skips judge metrics in deterministic offline mode;
- logs the Ragas results back into the same MLflow experiment; and
- compares framework convenience with the transparency of the first-principles
  metrics.

Ragas is not the tracing, dataset-lifecycle, or experiment-comparison platform.
MLflow keeps that role.

### Controlled failure

The failure lab uses one configuration change that creates an observable
retrieval miss or filter error. Students inspect the trace, classify the failure
as retrieval or generation, change one variable, rerun the same versioned
cases, and compare the result without claiming universal improvement.

### Lesson 07 verification

The final PASS marker requires:

- a non-empty, versioned evaluation dataset;
- unique case IDs;
- both answerable and abstention cases;
- complete MLflow run metadata;
- visible retrieval and generation spans;
- finite deterministic metrics;
- exact case-to-result alignment;
- no cross-company leakage in filtered cases;
- citation checks only against retrieved evidence;
- one displayed configuration comparison; and
- one displayed failure classification.

Live Ollama and OpenAI metrics are observations unless the contract is
provider-invariant. The deterministic offline mode may assert exact expected
values.

## Repository structure and canonical paths

`course.yml` remains the source of truth. The implemented two-day path uses:

- `notebooks/01_model_gateway.ipynb` through
  `notebooks/07_rag_evaluation.ipynb`;
- `chapters/01-model-gateway.md` through
  `chapters/07-rag-evaluation.md`; and
- `decks/01-model-gateway.pptx` through
  `decks/07-rag-evaluation.pptx`.

Legacy notebooks and chapters with overlapping lesson numbers must not appear
as equally valid student choices. They will be audited and either removed in a
separate reviewed commit or moved to an explicitly non-student archive. No
destructive cleanup is part of an individual lesson change.

Shared implementation remains under `src/finai_academy`. Lesson 07 evaluation,
tracing, and metric code must be split by responsibility rather than embedded
entirely in the notebook.

## Verification strategy

Every adjustment uses tests before implementation changes. The completed work
must pass:

1. focused unit tests for every new shared boundary;
2. notebook contract tests for structure, expected markers, and forbidden
   committed outputs;
3. deterministic offline execution of Lessons 01–07;
4. live Ollama execution of every affected notebook;
5. live OpenAI execution when a key is available, with no claim when it is not;
6. repository validation;
7. Ruff;
8. the complete pytest suite;
9. slide overflow and template-fidelity checks;
10. full-size visual inspection of every revised slide and notebook figure; and
11. `git diff --check` plus an exact scope review before each commit.

## Implementation sequence

1. Add shared lesson-contract, tracing, and evaluation foundations.
2. Align Lessons 01 and 02.
3. Apply the smaller alignment changes to Lessons 03 and 04.
4. Upgrade Lesson 05 semantic and contextual chunking.
5. Add Lesson 06 production bridges and trace spans.
6. Build Lesson 07 metrics, tracing, notebook, chapter, and deck.
7. Perform a complete seven-lesson regression and beginner-readability review.
8. Review the legacy-path cleanup separately.

Each lesson or shared foundation must land as a focused commit with its own
verification evidence. Lesson 07 is not complete until it consumes the real
Lesson 06 retrieval pipeline and the maintained course evidence set.

## Success criteria

The design is successful when:

- a student sees the same build-observe-improve-verify rhythm in every lesson;
- the finance case remains continuous rather than decorative;
- Ollama and OpenAI use the same application contracts;
- the class core runs without Docker or paid infrastructure;
- MLflow makes every major system stage inspectable;
- Lesson 05 includes real LLM-aware contextual enrichment;
- Lesson 06 explains the path from a local index to production retrieval;
- Lesson 07 can diagnose a retrieval failure separately from an answer failure;
- Ragas is understandable as a convenience layer rather than a black box; and
- all seven lessons remain deliverable inside the published two-day timetable.
