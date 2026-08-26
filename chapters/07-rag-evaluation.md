# Lesson 07 — Find the Failure in a RAG System

**First Finance - Arnaud Demes**  
**Day 1 · 16:00–16:45 · 15 minutes concepts + 30 minutes notebook**

## Instructor outcome

Students finish Day 1 with a diagnosable financial RAG application, not a polished demo.
Given a failed case, they can identify whether retrieval, filters, citation, grounding or
abstention owns the failure and inspect the corresponding MLflow trace.

The teaching contract is:

```text
versioned cases → BM25+dense RAG → layer-specific metrics → trace → diagnosis
```

Lesson 07 evaluates the exact BM25+dense pipeline built in Lesson 06. It does not recreate
the old TF-IDF retriever or hide the application behind a judge score.

## Before class

Install the evaluation dependencies with the normal AI and RAG extras:

```bash
uv sync --extra ai --extra rag --extra evaluation --extra dev
```

Offline mode requires no server or API key. It writes a local SQLite MLflow database and
artifacts under `FINAI_MLFLOW_DIR`, or a temporary directory when the variable is absent.

Optional live-provider setup:

```bash
export OPENAI_API_KEY="..."
export FINAI_PROVIDER="openai"
export FINAI_CHAT_MODEL="gpt-5-mini"
export FINAI_EMBEDDING_MODEL="text-embedding-3-small"
export FINAI_LIVE_MODE="1"
```

Never store credentials in notebook output, MLflow parameters, span inputs or the
repository.

## Communication job of the concept deck

By the end of the deck, finance and AI practitioners should understand that “bad RAG
answer” is not a diagnosis: retrieval, generation and abstention require different tests,
and a trace connects the failed metric to the responsible application boundary.

The deck uses the learning progression:

```text
hidden failure → evaluation contract → metric layers → trace → comparison → repair decision
```

| Time | Slide | Instructor job |
|---:|---:|---|
| 0:00–1:15 | 1 | Frame evaluation as failure location, not model grading. |
| 1:15–2:30 | 2 | Show why one fluent answer cannot validate a RAG system. |
| 2:30–3:45 | 3 | Introduce the versioned golden-set coverage contract. |
| 3:45–5:15 | 4 | Demonstrate retrieval passing while the answer fails. |
| 5:15–6:45 | 5 | Assign one metric family to each application layer. |
| 6:45–8:45 | 6 | Read a real MLflow trace and its spans. |
| 8:45–10:15 | 7 | Interpret the case-by-metric heatmap. |
| 10:15–11:45 | 8 | Compare a one-variable RRF change at aggregate and case level. |
| 11:45–13:00 | 9 | Map failure stages to engineering owners. |
| 13:00–14:00 | 10 | Run the knowledge check. |
| 14:00–15:00 | 11 | Debrief the answers and bridge to the notebook. |

Use the official OpenAI evaluation process as methodological grounding, not as a required
course platform. The lesson code remains provider-neutral and local:
[OpenAI evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices).

## 30-minute notebook pacing

| Time | Work | Observable output |
|---:|---|---|
| 0:00–4:00 | Load `rag-cases-v1` and real Lesson 05 passages. | Coverage matrix, Figure 1. |
| 4:00–9:00 | Keep retrieval fixed and compare two answers. | Controlled metric contrast, Figure 2. |
| 9:00–14:00 | Evaluate all eight cases. | Case-by-metric heatmap, Figure 3. |
| 14:00–20:00 | Inspect one persisted MLflow trace. | Span table and durations, Figure 4. |
| 20:00–25:00 | Increase only the BM25 RRF weight. | Aggregate + rank comparison, Figure 5. |
| 25:00–28:00 | Inspect failure ownership and verify. | Acceptance scorecard, Figure 6. |
| 28:00–30:00 | Knowledge check and challenge. | Exact PASS marker. |
| Optional | Construct explicit Ragas rows. | Provider/model-labelled skipped or judged status. |

The notebook contains 18 cells, eight code cells and six figures. The main route is short
enough for students to read every block rather than scrolling through plotting utilities.

## The versioned golden set

The eight cases cover six distinct behaviors:

- direct fact retrieval;
- exact-number retrieval;
- semantic paraphrase;
- metadata filter safety;
- multi-evidence comparison; and
- explicit abstention on unsupported questions.

Every positive case stores stable evidence IDs and maintained facts. Every negative case
stores an explicit abstention requirement. A changed question, expected ID, fact or policy
requires a version decision; it is not an invisible notebook edit.

Figure 1 is a coverage matrix rather than a count chart. Students can see which behavior
each case protects and where the dataset remains thin.

## Retrieval can pass while the answer fails

Figure 2 evaluates two answers against the same retrieved passages:

- the supported answer cites the expected NVIDIA evidence ID and states both facts;
- the broken answer keeps retrieval recall and reciprocal rank at `1.0` but cites a
  Schneider Electric passage and omits the maintained number.

The visual makes the lesson's central distinction concrete. A single “RAG quality” score
would hide which subsystem changed.

## Deterministic metric layers

For expected evidence set `E` and ordered final retrieved IDs `R`:

```text
recall@k = |E ∩ R[:k]| / |E|
reciprocal rank = 1 / rank(first expected ID), else 0
```

Filter correctness is `1` only when the prediction uses the case filters and every final
passage satisfies them.

Citation correctness parses stable IDs from square brackets and measures the share that
are both expected and present in the retrieved evidence. Grounded-fact coverage measures
maintained facts present in the final answer.

Abstention is an explicit application decision. A system may retrieve plausible passages
and still correctly refuse a valuation question that those passages cannot support. Do
not infer abstention from an empty retrieval result.

The offline baseline intentionally keeps one known answer defect: the semantic-paraphrase
answer uses the correct fact but omits its citation. Figure 3 should therefore show:

- retrieval recall `1.0` for all six positive cases;
- abstention correctness `1.0` for both unsupported cases; and
- citation correctness `0.0` for `nvda-semantic-paraphrase`.

Evaluation passes by finding that maintained defect—not by pretending every answer is
perfect.

## Trace and run model

One MLflow run represents one complete configuration:

- dataset version;
- provider and chat model;
- embedding model and index version;
- prompt version;
- `candidate_k` and `final_k`; and
- BM25/dense RRF weights.

One trace represents one case. Its root records the question, filters, configuration,
answer, explicit abstention decision, retrieved IDs, citations, rerank scores, stage
timings and failure stage. The persisted child spans are:

```text
eligibility → bm25 → dense → fusion → rerank → context → generation
```

Figure 4 is generated from the actual persisted span objects. The concept deck additionally
shows the official MLflow trace interface so students recognize the professional tool they
will open after the notebook:
[MLflow trace UI](https://mlflow.org/docs/latest/genai/tracing/observe-with-traces/ui).

To launch the local interface, copy the command printed by the first notebook cell:

```bash
mlflow ui --backend-store-uri sqlite:////absolute/path/to/finai-lesson07-mlflow/mlflow.db
```

Then inspect the run table, one trace's span tree, inputs and outputs, duration and failure
stage. The UI is an inspection surface, not a notebook dependency.

## Comparing configurations

The baseline uses BM25/dense RRF weights `1:1`. The comparison uses `3:1`. Every other
declared variable stays fixed.

Figure 5 deliberately combines two levels:

- aggregate metrics answer whether the maintained contract changed; and
- case-level ranks answer whether the configuration had any observable effect.

Use this language:

> A moved rank proves that the policy changed behavior. It does not prove improvement.

A configuration is better only when aligned, decision-relevant cases improve without
violating retrieval, filters, citations, grounding or abstention contracts.

## Failure ownership

Map the first failed layer to the component that can repair it:

| Observed failure | Primary owner | First investigation |
|---|---|---|
| Retrieval recall / rank | Index, query, filters, fusion or reranker | Inspect eligible candidates and channel ranks. |
| Filter correctness | Metadata model and eligibility boundary | Inspect filters before retrieval. |
| Citation correctness | Prompt, structured answer schema or citation renderer | Compare cited IDs with final evidence IDs. |
| Grounded-fact coverage | Context assembly and generation | Compare maintained facts with answer claims. |
| Abstention correctness | Evidence-sufficiency policy | Inspect the explicit abstention decision. |

Figure 6 shows one known citation failure beside the acceptance scorecard. This is the
engineering decision surface: the prompt/citation path owns the repair, while the BM25
retriever does not need to be tuned for that defect.

## Challenge solution boundary

Students first repair the missing citation without editing the golden set. Then they may
add an evidence-sufficiency gate between reranking and generation. A valid extension:

1. receives the question and final evidence;
2. returns an explicit `abstained: bool` decision and public reason;
3. opens a `sufficiency` span;
4. preserves all six positive retrieval cases;
5. abstains on both unsupported cases; and
6. compares the new configuration on the same `rag-cases-v1` dataset.

## Ragas: optional extension

Ragas is introduced only after the deterministic contract:

- context recall asks whether retrieved context contains the reference information; and
- faithfulness asks whether response claims are supported by retrieved context.

The ID-based context-recall variant is particularly compatible with the course's stable
evidence IDs. Faithfulness normally introduces a model judge. See the current official
metric documentation:
[Ragas context recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/)
and [Ragas faithfulness](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/faithfulness/).

The course adapter never selects an implicit judge. Provider, model, prompt, latency and
cost are evaluation data. Ragas complements the deterministic retrieval and citation
checks; it does not replace them.

## Expected outputs

Offline execution must produce:

- eight versioned cases and seven provenance-preserving passages;
- six positive retrieval cases with expected evidence inside `final_k=2`;
- two correct explicit abstentions;
- one maintained citation defect;
- two aligned MLflow runs;
- eight baseline traces with all seven required child-span names;
- Figures 1–6; and
- one exact final marker:

```text
PASS — RAG evaluation and tracing verified
```

## Instructor cautions

- Do not collapse the six metric columns into one “quality” score.
- Do not repair a citation failure by tuning retrieval.
- Do not call a rerank score confidence.
- Do not use the golden labels to force live application behavior.
- Do not change more than one declared variable in a configuration comparison.
- Do not present judge metrics without naming their provider and model.
