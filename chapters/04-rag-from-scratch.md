# Lesson 04 — How RAG Chooses Evidence

**Schedule:** Day 1, 11:30–12:00

**Format:** 10 minutes of concepts and visual explanation, 20 minutes of guided notebook work

**Capstone increment:** first retrieval-backed answer over a complete official filing, with
retrieval and grounding evaluated separately

## Teaching outcome

Students should leave with one operational definition:

> RAG is an application pipeline that selects evidence before generation.

They build a deliberately naive but fully inspectable pipeline over NVIDIA's complete
FY2026 Form 10-K:

```text
official filing → flattened text → overlapping windows → TF-IDF rank
               → top-k evidence → controlled prompt → answer
```

The goal is not to present the baseline as production-ready. The goal is to make every
decision visible and diagnose the earliest stage that produced a failure.

## Why RAG comes after Lesson 03

Lesson 03 established the routing decision:

```text
bounded reusable context → CAG
large or changing evidence universe → RAG
```

Lesson 04 opens the RAG path. The complete filing contains roughly 90,000 estimated text
tokens after naive flattening, while one concise answer needs only a small evidence set.
Retrieval reduces that boundary before the model call.

RAG can therefore improve:

- **focus:** the model sees fewer irrelevant passages;
- **cost and latency:** a smaller prompt crosses the model boundary;
- **traceability:** selected evidence carries stable identifiers and source provenance; and
- **diagnosability:** retrieval can be evaluated independently from generation.

RAG also introduces a new failure mode: relevant evidence may exist in the corpus but
never enter top-k.

## Source policy

The notebook reads the committed official NVIDIA FY2026 Form 10-K, accession
`0001045810-26-000021`, from:

```text
assets/course-data/downloads/nvidia_fy2026_form_10k.html
```

It verifies the recorded byte count and SHA-256 through the course manifest before
parsing. There are no manually paraphrased teaching passages in this lesson.

## State the limitation before teaching the mechanism

Call this a **naive real-document RAG baseline** throughout the lesson.

The baseline intentionally:

- strips HTML/XBRL markup and collapses visible text into one string;
- discards table, row, column, heading and hierarchy boundaries;
- creates 1,600-character windows with 200-character overlap;
- represents windows with TF-IDF lexical weights;
- ranks with cosine similarity; and
- sends only the top two windows to the model.

These are inspectable choices, not best practices.

## Concept sequence

### 1. One answer does not need the complete filing

The complete source may fit inside a modern context window, but capacity is not the same
as evidence focus. Sending everything forces the model to navigate irrelevant material,
increases input size and weakens the application's control over citations.

### 2. Parsing determines what survives

The official filing contains 64 HTML tables and no semantic `h1`–`h6` headings. Naive
flattening preserves words and numbers but erases their structural relationships. A value
may remain present while its row, column, unit or section meaning becomes ambiguous.

### 3. Chunking creates the retrieval candidates

Overlapping character windows are simple and reproducible. They can also begin or end in
the middle of a sentence, table or financial relationship. Overlap reduces hard boundary
loss but does not make the chunks structure-aware.

### 4. Retrieval creates the model's evidence boundary

The application—not the model—represents every window, ranks it and applies top-k.
Everything below the selection line is invisible to generation.

### 5. A prompt is assembled after retrieval

Selected windows enter a controlled prompt with stable chunk identifiers and the official
SEC URL. Retrieved text is treated as untrusted data rather than instructions.

### 6. Retrieval and grounding are different checks

Retrieval asks whether the expected windows entered the prompt. Grounding asks whether the
answer used the supplied evidence, cited it and respected the evidence boundary. A single
end-to-end score hides which stage needs repair.

### 7. The model cannot repair missing evidence

For the maintained precise-revenue question, the window containing `$193,737 million`
ranks 21st under the fixed naive policy. With `top_k=2`, the required table never reaches
the model. The earliest measured failure is retrieval, influenced by the flattened parsing
and naive chunk representation.

## Notebook pacing

| Time | Activity | Instructor emphasis |
|---:|---|---|
| 0–3 min | Verify the official filing and compare its scale with one answer | RAG reduces evidence before generation. |
| 3–7 min | Flatten the real HTML and inspect the revenue table | Words survive; financial structure does not. |
| 7–10 min | Build overlapping character windows | Chunks are application-created evidence candidates. |
| 10–14 min | Rank every window and apply top-k | The model has not been called yet. |
| 14–17 min | Build the prompt and generate with the configured provider | Provenance must survive retrieval. |
| 17–19 min | Separate retrieval and grounding checks | Diagnose before changing the model. |
| 19–20 min | Reproduce the precise-table miss | Lessons 05–06 repair different stages. |

## Visual teaching contract

The notebook produces six executable figures:

1. complete filing tokens versus a concise-answer budget;
2. HTML structure counts beside the flattened revenue table;
3. overlapping character-window construction;
4. the complete visible top-k ranking boundary;
5. complete filing versus retrieved evidence and final prompt size; and
6. the precise Data Center table stranded outside top-k.

The presentation mirrors this progression with an official SEC screenshot, a published
RAG architecture visual and editable teaching diagrams.

## Maintained success case

Question:

> What drove NVIDIA revenue growth in fiscal 2026?

The fixed naive policy retrieves windows `NVDA-C152` and `NVDA-C160`. Together they contain
the fiscal-year summary and the discussion of accelerated computing, AI, Blackwell,
Data Center computing and networking growth.

The live model answer remains an observation. The deterministic retrieval check is the
gate: both expected windows must enter the prompt.

## Failure laboratory

Question:

> How large was Data Center revenue compared with total revenue in fiscal 2026?

The filing contains the necessary table, but its flattened window ranks outside top-k.
Ask students in order:

1. Does the evidence exist in the official document? **Yes.**
2. Did naive parsing preserve the table relationship? **Only as flattened text.**
3. Did the correct window enter top-k? **No.**
4. Can changing the generation model recover unseen evidence? **No.**

The correct diagnosis is a retrieval failure produced upstream of generation. Lesson 05
will test better parsing and chunk boundaries; Lesson 06 will improve representation and
ranking.

## Checkpoint questions

### Why use RAG if the complete filing fits in the model context window?

Context capacity does not guarantee focus, traceability or efficient repeated use. RAG
lets the application select and test the evidence boundary explicitly.

### Does overlap make character windows structure-aware?

No. It reduces abrupt boundary loss but does not understand headings, tables, units or
financial hierarchy.

### Is cosine similarity a confidence score?

No. It measures alignment under the chosen representation. A high score does not prove
that a passage supports the answer.

### Where should debugging begin when an answer omits a fact?

Check whether the fact was parsed, whether a coherent chunk contains it, whether that
chunk was retrieved, and only then inspect generation.

## Transition to Lessons 05 and 06

```text
Observed Lesson 04 baseline       Next controlled improvement
──────────────────────────────    ─────────────────────────────────
flattened HTML text               canonical headings/tables/blocks     Lesson 05
character windows                 structure-aware chunking              Lesson 05
lexical TF-IDF only               embeddings + hybrid retrieval         Lesson 06
simple global top-k               filters + fusion + reranking           Lesson 06
```

Do not collapse these into “better RAG.” Each improvement changes a distinct, measurable
stage.

## Provider modes

Offline execution uses `RecordedRagModel`. Live execution uses the shared provider gateway.
The course OpenAI default is `gpt-5.6-luna`.

```bash
uv run python scripts/execute_notebooks.py \
  notebooks/04_rag_from_scratch.ipynb --mode offline
```

```bash
FINAI_MODEL_PROVIDER=openai FINAI_CHAT_MODEL=gpt-5.6-luna \
uv run --extra ai python scripts/execute_notebooks.py \
  notebooks/04_rag_from_scratch.ipynb --mode live --provider openai
```

## Instructor notes

- Ask students to predict the top two windows before displaying the ranking.
- Say “window” rather than “good chunk.”
- Keep the top-k line visible while explaining the model boundary.
- Do not describe the live answer as proof that the retrieval policy is good.
- Preserve any live grounding miss as an observation.
- End with the question: “What document structure did our baseline destroy?”

## Sources

- [NVIDIA FY2026 Form 10-K](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [scikit-learn `TfidfVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [scikit-learn cosine similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)
- [OpenAI GPT-5.6 Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna)
