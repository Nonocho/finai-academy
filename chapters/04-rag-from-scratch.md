# Lesson 04 — Naive RAG from First Principles

**Schedule:** Day 1, 11:30–12:00

**Format:** 8 minutes of concepts and diagrams, 22 minutes of guided notebook work

**Capstone increment:** first retrieval-backed financial answer with separate retrieval
and grounding checks

## Teaching outcome

Students should leave with one operational definition:

> Retrieval-Augmented Generation is a retrieval system followed by controlled context
> construction and evidence-bounded generation.

They build a complete but intentionally naive baseline over prepared NVIDIA and
Schneider Electric passages. They can inspect the representation, ranking, selected
context and answer, then identify whether a failure occurred before or after the model
call.

The lesson keeps one question fixed and compares three application paths:

```text
No context → full context → naive RAG
```

- no context exposes why a fluent model cannot recover missing company evidence;
- full context supplies all six mixed-company passages and becomes the reference;
- naive RAG selects two passages before generation and makes that decision inspectable.

## State the limitation before teaching the mechanism

Call this a **naive RAG baseline** throughout the lesson. It is useful because every
stage is visible, not because it is ready for production.

The baseline assumes:

- six passages have already been extracted and segmented correctly;
- TF-IDF lexical overlap is an adequate representation;
- cosine similarity is the only ranking signal;
- top-k selection needs no metadata filter or reranker; and
- two retrieved passages provide enough context for the question.

Lesson 05 removes the first assumption by parsing real financial documents and
comparing chunking strategies. Lesson 06 improves representation and ranking with
embeddings, metadata, hybrid retrieval and reranking.

## Why this lesson comes before parsing and chunking

Students first need to understand the retrieval loop in isolation. Starting with raw
PDF complexity would combine several failure sources at once:

```text
raw document → parsing → structure → chunks → index → rank → context → answer
```

Lesson 04 deliberately begins at `chunks`. Once students can observe the index,
ranking and prompt, Lesson 05 can change the chunks while keeping the downstream
mechanism understandable.

For one visible baseline only, the notebook joins the six prepared passages and
applies a blank-line paragraph split. This is explicitly a **naive paragraph split**:
it ignores headings, tables, semantic boundaries and hierarchy. Lesson 05 replaces
that simplification rather than treating it as a production parser.

## Financial evidence pack

The corpus contains three NVIDIA fiscal 2026 passages and three Schneider Electric
fiscal 2025 passages.

NVIDIA evidence covers:

- total revenue of approximately $215.9 billion and 65% year-on-year growth;
- Data Center revenue of approximately $193.7 billion and 68% growth; and
- Gaming revenue of approximately $16.0 billion and 41% growth.

Schneider Electric evidence covers:

- fiscal 2025 revenue of approximately €40.2 billion and 8.9% organic growth;
- 10% organic growth in Energy Management, with Data Center demand leading fourth-
  quarter growth; and
- adjusted EBITA of approximately €7.5 billion at an 18.7% margin.

The passages are paraphrased teaching extracts. Their stable identifiers and official
source URLs remain attached throughout retrieval and prompt construction.

## Concept sequence

### 1. RAG controls evidence selection

A model does not query the corpus directly in this baseline. The application ranks all
passages, selects top-k, and constructs a new prompt. Any passage outside top-k is
invisible to the model call.

### 2. TF-IDF creates weighted lexical vectors

Term Frequency–Inverse Document Frequency gives higher weight to terms that help
distinguish one passage from the rest of the corpus. It does not represent financial
meaning or synonym relationships.

### 3. Cosine similarity creates a ranking

The query is transformed with the same vocabulary as the corpus. Cosine similarity
compares the query vector with each passage vector. The complete score distribution is
more informative than printing only the winner.

### 4. Top-k is a decision boundary

A smaller k reduces context size but can omit necessary evidence. A larger k may
recover more evidence but increases noise, token usage and cross-company leakage. The
notebook uses `top_k=2` because the question needs both total revenue and Data Center
evidence.

### 5. Context construction preserves provenance

Selected passages enter a prompt with passage identifiers, company, period, section
and source URL. Retrieved content is explicitly treated as untrusted data rather than
instructions.

### 6. Retrieval and generation are evaluated separately

Retrieval recall asks whether the expected passages entered the prompt. Grounding
checks ask whether the generated answer used those passages, preserved citations and
stated an evidence limitation.

### 7. A lexical failure creates the next engineering question

The failure query expresses a related idea with vocabulary absent from the corpus.
Relevant evidence loses its score. This does not prove that one embedding model will
solve the problem; it proves that the baseline representation has a measurable limit.

## Notebook pacing

| Time | Activity | Instructor emphasis |
|---:|---|---|
| 0–3 min | Compare no context and full context | Same question; different evidence boundary. |
| 3–5 min | Inspect the prepared corpus and paragraph split | Prepared passages are a deliberate simplification. |
| 5–9 min | Build TF-IDF and inspect the matrix | Representation determines which similarity signals exist. |
| 9–13 min | Rank every passage and apply top-k | The model has not been called yet. |
| 13–16 min | Assemble and visualize the prompt budget | Retrieval controls what the model can see. |
| 16–19 min | Generate with offline, Ollama or OpenAI | Add naive RAG to the three-path table. |
| 19–21 min | Run retrieval and grounding checks | Keep the two evaluation layers distinct. |
| 21–22 min | Trigger the lexical failure | Map each weakness to the next lesson. |

## Visual teaching contract

The notebook produces six executable figures:

1. the prepared NVIDIA–Schneider corpus and passage sizes;
2. selected TF-IDF term weights for every passage and the query;
3. the complete cosine-similarity ranking with a visible top-k boundary;
4. the final prompt allocation after retrieval;
5. ranking signals for explicit versus mismatched vocabulary; and
6. the map from naive components to Lessons 05 and 06.

The deck mirrors the same mechanism with editable diagrams. Students should be able
to point from a slide element to the notebook state that calculates it.

## Failure lab

Use the query:

> Which division supplied most of the expansion from machine-learning infrastructure?

The intended concept is NVIDIA Data Center, but the wording shares little or no useful
vocabulary with the prepared passages. Scores collapse to zero or near zero and the
stable identifier tie-break becomes visible.

Ask students:

1. Did the model fail?
2. Did top-k fail?
3. Or did the representation fail before both?

The correct diagnosis is a retrieval-representation failure. The model has not yet
received the relevant passage.

## Checkpoint questions and answers

### 1. Why keep retrieval evaluation separate from answer evaluation?

Because a wrong answer can originate from missing evidence or from poor use of correct
evidence. A single end-to-end score hides which component needs improvement.

### 2. Why is TF-IDF still useful if embeddings come next?

It is fast, local, transparent and strong for exact terms, accounting language,
product names and identifiers. It also provides a baseline that later methods must
beat on a maintained question set.

### 3. What does increasing top-k trade off?

It can improve evidence coverage, but it consumes more context and may add irrelevant,
conflicting or cross-company passages.

### 4. Does a high cosine score prove the passage supports the answer?

No. It measures vector similarity under the selected representation. Support and
grounding require separate evidence checks.

### 5. Why are source URLs included inside the context?

They preserve provenance through prompt construction and allow the final application
to emit traceable citations. A URL alone does not prove that the answer used the source
correctly.

### 6. Why compare no context, full context and RAG before tuning retrieval?

The comparison makes the engineering job explicit. No context tests the missing-
evidence boundary, full context tests whether the complete bounded corpus is usable,
and RAG tests whether selective evidence can preserve the answer with less context.
Without these references, a retrieval score has no application baseline.

## Challenge solution

With `top_k=1`, the Data Center passage is selected but the total-revenue passage is
lost. The answer cannot safely compare Data Center with the company total.

With `top_k=4`, additional NVIDIA or Schneider passages enter the context. Evidence
coverage rises, but so do noise and the risk of mixing companies or periods.

A defensible decision is:

> Use top-k=2 for this maintained question because the expected evidence set contains
> exactly the total-revenue and Data Center passages; re-evaluate the policy when the
> corpus or question set changes.

## Transition to the next lessons

```text
Observed naive component          Improvement
──────────────────────────────    ─────────────────────────────────
prepared evidence passages        parse SEC HTML and financial PDFs      Lesson 05
arbitrary passage boundaries       structural and semantic chunking       Lesson 05
metadata carried but not used      canonical blocks and filters           Lessons 05–06
lexical TF-IDF only                embeddings and hybrid retrieval        Lesson 06
simple top-k                       filtering and reranking                 Lesson 06
```

Do not describe parsing and chunking as optional polish. They determine whether the
retriever receives coherent financial evidence in the first place.

## Instructor notes

- Say “prepared passage” rather than “chunk” until the limitation is explicit.
- Ask learners to predict the top two passages before running the ranking cell.
- Keep the score chart on screen while discussing top-k.
- Do not attribute Schneider evidence to NVIDIA merely because both discuss Data
  Center demand.
- Do not present cosine similarity as probability or confidence.
- If a live model omits citations, keep the failure: retrieval may have passed while
  generation grounding failed.
- End before lunch with the raw-document question: “Who created these passages, and
  what might have been lost?” That question opens Lesson 05.

## Sources

- [NVIDIA fiscal 2026 Form 10-K](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm)
- [Schneider Electric 2025 full-year results](https://www.se.com/ww/en/assets/564/document/528237/release-fy-results-2025.pdf?p_File_Name=2025+FY+Results&p_enDocType=Financial+release)
- [scikit-learn `TfidfVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [scikit-learn cosine similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)
