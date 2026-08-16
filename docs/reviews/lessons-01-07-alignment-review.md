# Day 1 alignment review — Lessons 01–07

Review date: 2026-08-16  
Course: AI Engineering for Asset Management  
Reviewer standard: notebook-first, finance-specific, observable, provider-neutral and professionally teachable

## Outcome

Day 1 is ready for an instructor rehearsal. The complete student path runs from a
provider-neutral model call to a measured financial RAG system with versioned evaluation
cases and MLflow traces. The progression is cumulative: each notebook reuses the contracts
and evidence built earlier instead of introducing a disconnected demo.

Overall grade: **9.7 / 10**.

The score is below 10 because the OpenAI live path was not executed without an API key,
the optional Ragas judge is intentionally outside the mandatory PASS gate, and classroom
timing still needs one human rehearsal with Antoine.

## Verified delivery schedule

| Time | Session | Observable capstone increment |
|---|---|---|
| 09:00–09:30 | Product demo and system architecture | Shared product and evidence contract |
| 09:30–10:00 | 01 — Model gateway | Provider-neutral response and run record |
| 10:00–10:30 | 02 — Prompts and structured outputs | Validated analyst object |
| 10:30–10:45 | Break | — |
| 10:45–11:30 | 03 — Context engineering and CAG | Capacity gate and complete-document answer |
| 11:30–12:00 | 04 — RAG from first principles | Transparent naive-RAG baseline |
| 12:00–13:30 | Lunch | — |
| 13:30–15:00 | 05 — Financial documents and chunking | Parsed, provenance-safe and configurable chunks |
| 15:00–15:15 | Break | — |
| 15:15–16:00 | 06 — Embeddings and hybrid retrieval | Filtered keyword+dense retrieval, RRF and reranking |
| 16:00–16:45 | 07 — RAG evaluation and tracing | Golden set, metrics, failure stages and MLflow traces |
| 16:45–17:00 | Integration checkpoint | Complete Day 1 financial RAG pipeline |

## Lesson grades

| Lesson | Grade | Why it is teachable now |
|---|---:|---|
| 01 — Model gateway | 9.6 | One interface covers Ollama/OpenAI, measurements remain visible, and a grounding contract prevents a successful API call from being mistaken for analysis. |
| 02 — Structured outputs | 9.7 | Prompt responsibilities, untrusted input, Pydantic validation and financial acceptance are separated clearly; students leave with a typed analyst object. |
| 03 — Context engineering and CAG | 9.6 | CAG is positioned as a bounded routing decision, not a universal replacement for RAG; caching and context-capacity failure are observable. |
| 04 — Naive RAG | 9.7 | The baseline is explicitly labelled naive, built from first principles, visualized through retrieval stages and handed off honestly to parsing, chunking and reranking. |
| 05 — Documents and chunking | 9.8 | Parsing quality, provenance, fixed/recursive/structure/semantic/hierarchical/proposition/contextual strategies and LLM-aware enrichment are compared on finance evidence. |
| 06 — Hybrid retrieval | 9.8 | Metadata eligibility, keyword+dense channels, score semantics, RRF, reranking, storage boundary and stage measurements form a professional retrieval lesson without hiding implementation details. |
| 07 — Evaluation and tracing | 9.7 | Eight versioned NVIDIA/Schneider cases separate retrieval and answer quality; two aligned configurations create real MLflow traces and failure analysis; Ragas is introduced after deterministic fundamentals. |

## Quality rubric

| Dimension | Grade | Evidence |
|---|---:|---|
| Prerequisites and setup | 9.7 | Offline mode is deterministic; Ollama is the tested live default; OpenAI uses the same gateway. |
| Objective clarity | 9.8 | Every lesson states the engineering question, capstone increment and observable verification. |
| Visual explanation | 9.7 | 65 reviewed slides plus 42 generated notebook figures across Lessons 03–07; diagrams use the same visual grammar. |
| Expected outputs | 9.8 | Notebook cells expose tables, metrics, evidence IDs, charts and explicit PASS gates. |
| Failure diagnosis | 9.8 | Each lesson contains a failure lab or boundary; Lesson 07 maps failures to retrieval, citation or abstention stages. |
| Challenge scope | 9.6 | Challenges change one bounded variable and retain a solution path; final timing must be confirmed in rehearsal. |
| Finance continuity | 9.8 | NVIDIA and Schneider Electric remain the evidence thread from prompting through evaluation. |
| Provider neutrality | 9.7 | All seven notebooks pass offline and with Ollama; OpenAI configuration is present but was not live-run without a key. |
| Verification | 9.8 | Seven offline and seven Ollama executions pass; deck sources, footer, placeholders and overflow are audited. |
| Timing | 9.6 | The schedule closes exactly at 17:00 with breaks and integration time; human delivery pacing remains the final check. |

## Execution evidence

- Seven source notebooks pass the output-free notebook contract.
- Seven notebooks execute successfully in sequence in offline mode.
- Seven notebooks execute successfully in sequence with Ollama.
- Lesson 07 records eight baseline MLflow traces and generates nine figures.
- The Day 1 decks contain 65 slides in total.
- All 65 slides contain `[Sources]` speaker notes.
- All 65 slides contain the exact footer `First Finance - Arnaud Demes`.
- No Day 1 deck contains unresolved placeholders or detected overflow.
- The Lesson 07 deck passes template-fidelity checks with zero issues.

## Instructor rehearsal acceptance

The rehearsal should be accepted when:

1. a clean learner environment completes the offline setup without manual code edits;
2. the Ollama models are pulled before class and all seven notebook PASS gates remain green;
3. Antoine can explain the output of each visual before revealing the next cell;
4. the afternoon reaches Lesson 07 by 16:00 without shortening the chunking comparison; and
5. one retrieval or abstention failure is diagnosed from its trace before Day 1 closes.
