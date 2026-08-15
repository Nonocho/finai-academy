# Session 06 — Embeddings, Hybrid Retrieval, and Reranking

## Purpose

Lesson 06 keeps the trusted document blocks, structure-aware chunks, provenance, and
maintained questions from Lesson 05. It changes only how evidence is represented,
filtered, combined, and ranked. Learners can therefore attribute each retrieval change
to one observable engineering decision.

The session runs from 15:15 to 16:00 on Day 1: 15 minutes of slides and 30 minutes of
guided notebook work.

## Approved design decision

Three approaches were considered:

1. A progressive comparison over one fixed corpus and question set.
2. A vector-database tutorial centered on one external product.
3. A mathematics-first treatment of embedding geometry.

The lesson uses approach 1. It gives learners production-relevant concepts without
turning a 45-minute session into framework setup or abstract mathematics. The notebook
compares four retrieval stages in order:

1. lexical retrieval;
2. dense retrieval;
3. metadata-filtered hybrid retrieval with reciprocal-rank fusion; and
4. deterministic reranking into a final evidence set.

## Learning outcome

At the end of the lesson, a learner can:

- explain an embedding as a numerical representation rather than a probability;
- interpret cosine similarity as a relative ranking signal;
- describe why dense and lexical retrieval fail on different questions;
- prevent cross-company and cross-period leakage with metadata filters;
- combine ranked lists with reciprocal-rank fusion;
- distinguish candidate retrieval from reranking;
- explain why an embedding index must be versioned; and
- defend a retrieval policy using measured evidence rather than one anecdotal query.

## Controlled corpus and question contract

The lesson consumes the NVIDIA FY2026 and Schneider Electric FY2025 fixtures introduced
in Lesson 05. It uses structure-aware child chunks with complete provenance and the same
four maintained questions:

1. Which NVIDIA business generated $193.7 billion?
2. How fast did NVIDIA Gaming revenue grow?
3. What was Schneider Electric FY2025 revenue?
4. What margin did Schneider adjusted EBITA reach?

The corpus also includes compact, provenance-labelled distractor chunks derived from the
same fixtures. Distractors expose exact-term and cross-company failure modes without
adding a third company or an uncontrolled data source.

The experiment holds parsing, chunk construction, corpus, questions, expected evidence,
and evaluation logic constant. Only representation, filtering, fusion, and ranking
change.

## Retrieval architecture

```text
trusted Lesson 05 chunks
          ↓
     metadata eligibility
          ↓
keyword candidates  +  dense candidates
          ↓                    ↓
       ranked IDs          ranked IDs
                 ↘        ↙
        reciprocal-rank fusion
                  ↓
         deterministic reranking
                  ↓
       budgeted, cited evidence set
```

The implementation adds a focused `hybrid_retrieval.py` module rather than expanding the
Lesson 04 lexical implementation into one large file. It reuses `EvidencePassage` and
`RetrievalHit` from `retrieval.py` and introduces the following public boundaries:

- `RetrievalFilters` for company, period, document type, and section eligibility;
- `EmbeddingIndexVersion` for provider, model, dimension, corpus hash, and chunking
  strategy identity;
- `KeywordIndex` for filtered TF-IDF candidates;
- `DenseIndex` for filtered cosine-similarity candidates;
- `reciprocal_rank_fusion` for deterministic list combination and deduplication;
- `rerank_candidates` for transparent second-stage evidence scoring; and
- `retrieve_evidence` for the complete filtered hybrid pipeline.

Every result retains its passage identifier and source URL. Stable passage identifiers
are the join key across lexical, dense, fused, and reranked views.

## Embedding behavior

The notebook supports three explicit modes:

- `offline`: a deterministic teaching embedder maps a documented financial concept
  vocabulary into dense vectors. It exists for repeatable tests and clearly identifies
  itself as an illustrative representation, not a production model;
- `ollama`: the shared provider gateway uses `qwen3-embedding:0.6b` by default; and
- `openai`: the shared provider gateway uses `text-embedding-3-small` by default.

The retrieval code consumes the provider-neutral methods
`embed_documents(list[str]) -> list[list[float]]` and
`embed_query(str) -> list[float]`. Provider selection never changes the indexing or
ranking code.

Cosine similarity is always labelled as a ranking score. The lesson never describes it
as confidence, correctness, or probability.

## Index versioning and persistence

The notebook writes a lightweight index artifact to its configured output directory. It
contains vectors plus a JSON manifest with:

- embedding provider and model;
- vector dimension;
- ordered passage identifiers;
- corpus SHA-256 hash;
- chunking strategy;
- index schema version; and
- creation mode: offline, Ollama, or OpenAI.

Loading rejects a dimension, corpus-hash, model, or chunking-strategy mismatch with a
specific version error. The repository does not commit generated vectors and does not
introduce an external vector database.

## Keyword and dense comparison

`KeywordIndex` preserves the Lesson 04 TF-IDF baseline. `DenseIndex` uses normalized
embeddings and cosine similarity. Both expose complete deterministic rankings before
top-k truncation and use passage identifier as the final tie break.

The comparison demonstrates two complementary failures:

- a semantically related query can succeed with dense retrieval despite limited literal
  overlap; and
- a ticker, percentage, accounting label, or exact figure can remain stronger in keyword
  retrieval.

The notebook shows ranked passages and scores side by side. It does not imply that one
retriever is universally superior.

## Metadata safety barrier

The production pipeline applies filters before keyword and dense scoring. The supported
fields are company, period, document type, and section. Multiple supplied fields use
logical AND.

The failure laboratory deliberately bypasses this production boundary for its first run:
an intentionally broad semantic query without filters returns a plausible but ineligible
chunk from the other company. Restoring the company and period filters removes it before
keyword and dense scoring, fusion, and reranking.

An empty eligible set is not silently broadened. The pipeline returns no evidence and an
explicit abstention reason.

## Reciprocal-rank fusion

Fusion operates on ranks rather than incomparable lexical and cosine scores. For each
passage identifier:

```text
RRF score = Σ  weight(retriever) / (k + rank(retriever))
```

The default uses equal lexical and dense weights with `k = 60`. The challenge changes one
weight and requires learners to explain which maintained question moved and why.

Duplicate passage identifiers collapse into one fused result with both contributing
ranks preserved. Equal fused scores use passage identifier as a stable tie break.

## Deterministic reranking

Reranking receives a wider fused candidate set and produces a smaller evidence set. The
offline scorer is transparent and combines:

- normalized query-to-passage lexical coverage;
- exact numeric and ticker preservation;
- section-title overlap;
- metadata eligibility; and
- the normalized fusion rank signal.

The feature values and final score remain visible in the notebook. A provider-based or
cross-encoder reranker is named as a production extension but is not required in this
45-minute session.

## Notebook narrative and timing

The 30-minute notebook follows one continuous experiment:

1. **3 minutes — Reuse trusted chunks:** load the same sources, chunks, and questions.
2. **5 minutes — See embeddings:** inspect vector dimensions, a two-dimensional
   projection, and query-to-chunk similarity.
3. **5 minutes — Compare retrievers:** view lexical and dense ranked ladders and expose
   the exact-term failure.
4. **5 minutes — Block leakage:** reproduce cross-company leakage, then apply company and
   period filters.
5. **5 minutes — Fuse candidates:** calculate RRF step by step and inspect deduplication.
6. **4 minutes — Rerank evidence:** inspect feature contributions and the final context
   budget.
7. **3 minutes — Challenge and verification:** change one fusion weight, explain the
   consequence, and run the maintained checks.

The source notebook has cleared outputs and runs end to end in offline mode. Ollama and
OpenAI use the same cells after environment configuration.

## Notebook visual contract

The executable notebook produces at least eight readable visuals:

1. the complete retrieval pipeline and its stable passage identifiers;
2. a two-dimensional projection of passages and one query;
3. a query-to-passage cosine-similarity heatmap;
4. lexical versus dense ranked ladders;
5. an exact-term failure comparison;
6. a cross-company leakage view before and after filtering;
7. an RRF rank-contribution table or diagram;
8. reranker feature contributions and the final evidence budget; and
9. a four-stage maintained-question scorecard when space permits.

Every visualization uses complete labels, distinguishes score types, and makes the
eligible company and period visible.

## Failure laboratory and expected markers

The notebook must print the following markers only after their conditions are verified:

- `Dense exact-term failure reproduced`;
- `Cross-company leakage blocked`;
- `Hybrid retrieval improves maintained recall`; and
- `PASS — hybrid retrieval laboratory verified`.

The final marker requires complete provenance, zero ineligible final passages, stable
fusion ordering, and successful retrieval of every maintained expected evidence item at
the configured final top-k.

## Deck narrative

The companion deck contains seven low-density slides:

1. lesson outcome and the 15/30-minute format;
2. embeddings and cosine similarity without probability language;
3. keyword and dense retrieval as complementary candidate generators;
4. metadata filters as an eligibility barrier;
5. reciprocal-rank fusion with one worked rank example;
6. candidate retrieval versus reranking; and
7. the capstone pipeline and transition to Lesson 07 evaluation.

Diagrams are editable, connectors sit behind nodes, and visible copy remains factual and
professional. Every slide contains a `[Sources]` block in speaker notes. The footer is
`First Finance - Arnaud Demes`.

## Testing and acceptance criteria

The implementation is accepted when:

- keyword and dense index tests prove stable ranking and top-k validation;
- filter tests prevent company, period, document-type, and section leakage;
- an empty eligible set returns an explicit abstention reason;
- RRF tests prove its formula, deduplication, weighting, and stable tie breaks;
- reranking tests prove exact numeric coverage and deterministic ordering;
- index tests reject provider, model, dimension, corpus, and chunking mismatches;
- the offline notebook executes end to end and produces at least eight PNG outputs;
- the source notebook passes the course notebook contract with cleared outputs;
- Ollama and OpenAI embedding paths use the shared provider gateway;
- all slide renders are inspected at full size and the overflow test passes; and
- the complete repository test, lint, and validation suite remains green.

## Scope boundary and capstone increment

Lesson 06 does not teach approximate-nearest-neighbor internals, operate an external
vector database, compare many embedding models, implement HyDE, use a generative
reranker, or tune production latency. Those topics would obscure the retrieval decisions
within the fixed 45-minute slot.

The capstone increment is a provider-neutral, versioned, metadata-filtered hybrid
retriever with deterministic reranking and traceable evidence identifiers. Lesson 07
keeps this pipeline fixed and evaluates retrieval relevance, groundedness, citation
correctness, completeness, and abstention.
