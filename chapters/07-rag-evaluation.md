# Lesson 07 — RAG Evaluation and Tracing

**First Finance - Arnaud Demes**  
**Day 1 · 16:00–16:45 · 15 minutes concepts + 30 minutes notebook**

## Instructor outcome

Students finish Day 1 with a measured financial RAG application, not a single successful
demo. They can explain whether a failure came from retrieval, filters, citations,
grounding or abstention, and they can open the corresponding MLflow trace.

The teaching contract is:

```text
versioned cases → real pipeline → separated metrics → traces → comparison → decision
```

MLflow first appears here because Lessons 01–06 already expose ordinary typed results and
stage boundaries. Lesson 07 instruments those boundaries without rewriting the pipeline.

## Before class

Install the evaluation extra together with the normal AI dependencies:

```bash
uv sync --extra ai --extra rag --extra evaluation --extra dev
```

Offline mode needs no server and no API key. It writes a local SQLite database and local
artifacts below the directory selected by `FINAI_MLFLOW_DIR`; otherwise it uses the system
temporary directory.

Optional Ollama setup:

```bash
ollama pull qwen3:8b
ollama pull qwen3-embedding:0.6b
```

Optional OpenAI setup:

```bash
export OPENAI_API_KEY="..."
export FINAI_CHAT_MODEL="gpt-5-mini"
export FINAI_EMBEDDING_MODEL="text-embedding-3-small"
```

Never store credentials in MLflow parameters, span inputs, notebook output or the
repository.

## 15-minute concept deck

| Time | Slide | Instructor job |
|---:|---:|---|
| 0:00–2:00 | 1 | State that evaluation locates a failing stage; it is not one overall score. |
| 2:00–4:00 | 2 | Explain the golden set as versioned engineering data. |
| 4:00–6:00 | 3 | Separate retrieval, answer and abstention axes. |
| 6:00–8:00 | 4 | Work through deterministic metric formulas. |
| 8:00–10:30 | 5 | Read one trace from question and filters to answer. |
| 10:30–13:00 | 6 | Compare two configurations on identical cases. |
| 13:00–15:00 | 7 | Position Ragas after the baseline and state judge limitations. |

Do not spend deck time launching the UI. The notebook shows the same essential trace and
run-comparison information inline, so the class remains teachable without a browser.

## 30-minute notebook pacing

| Time | Work | Expected output |
|---:|---|---|
| 0:00–4:00 | Load `rag-cases-v1`. | Eight cases, seven known evidence IDs, Figure 1. |
| 4:00–9:00 | Run the first NVIDIA case. | Six separated metric values, Figure 2. |
| 9:00–14:00 | Evaluate the full baseline. | Case table and Figure 3 heatmap. |
| 14:00–20:00 | Inspect local MLflow. | Eight traces, seven required child-span names, Figure 4. |
| 20:00–25:00 | Change only keyword RRF weight. | Two run IDs, aligned comparison, Figures 5–6. |
| 25:00–28:00 | Inspect failure rows. | Explicit `abstention` classification, Figure 7. |
| 28:00–30:00 | Verify and debrief. | Figure 8 and one exact final PASS marker. |
| Optional | Run an explicit Ragas judge. | Figure 9 plus provider/model-labelled metrics or skipped status. |

## The versioned golden set

The JSON dataset contains at least one case for each required behavior:

- direct fact;
- exact number;
- semantic paraphrase;
- NVIDIA filter safety;
- Schneider Electric filter safety;
- controlled cross-company leakage;
- multi-evidence comparison; and
- insufficient evidence.

Every case stores stable evidence IDs, not text snippets. The manifest stores the dataset
version, path and SHA-256 hash. Changing a question, expected ID or expected fact creates a
new evaluation artifact and requires a deliberate version decision.

The two negative cases intentionally reveal a current capstone gap. Metadata eligibility
can find NVIDIA passages even when the question asks for unsupported valuation or Energy
Management information. The correct Lesson 07 output is a classified abstention failure,
not a silently edited golden set.

## Deterministic metric formulas

For expected evidence set `E` and ordered final retrieved IDs `R`:

```text
recall@k = |E ∩ R[:k]| / |E|
reciprocal rank = 1 / rank(first expected ID), else 0
```

Filter correctness is 1 only when the prediction used the case filters and every final
passage satisfies them.

Citation correctness parses stable IDs from square brackets. It measures the share of
citations that are both expected and present in the final retrieved set. A supported case
with no citation receives 0.

Grounded-fact coverage is:

```text
matched maintained facts / expected maintained facts
```

The implementation normalizes punctuation and financial numeric phrases. It does not ask
an LLM to judge its own answer.

Abstention correctness is 1 only when the case requirement and application behavior
agree. For a supported case, returning no evidence is wrong. For an unsupported case,
returning plausible evidence is also wrong.

## Trace and run model

One MLflow run is one complete configuration:

- dataset version;
- provider;
- chat model;
- embedding model;
- index version;
- prompt version;
- `candidate_k` and `final_k`; and
- keyword/dense RRF weights.

One trace is one case. Its root records the question, filters, configuration ID, answer,
retrieved IDs, citations, rerank scores, stage timings and failure stage. Child spans are:

```text
eligibility → keyword → dense → fusion → rerank → context → generation
```

The first five child spans come from the same observer boundary used in Lesson 06. Context
and generation use that observer too. Trace instrumentation therefore stays orthogonal to
retrieval behavior.

## Local MLflow behavior

The course uses a local SQLite backend because current MLflow tracing features no longer
treat the legacy filesystem tracking backend as the preferred path. Artifacts remain in a
local directory next to the database.

To launch the UI after executing the notebook, use the path printed by the setup cell:

```bash
mlflow ui --backend-store-uri sqlite:////absolute/path/to/finai-lesson07-mlflow/mlflow.db
```

Then open `http://127.0.0.1:5000`. Demonstrate:

1. the two configuration runs;
2. their parameter difference;
3. aggregate metrics;
4. one negative-case trace; and
5. the `abstention` failure row artifact.

The UI is an inspection surface, not a required notebook dependency.

## Comparing configurations

The baseline uses keyword/dense RRF weights `1:1`. The comparison uses `3:1`. Everything
else remains fixed. The lesson does not claim that a changed rank proves improvement.

Use this language:

> “A configuration is better only when the aligned cases and decision-relevant metrics
> improve without violating filter, evidence or abstention contracts.”

Aggregate metrics may tie while a passage rank changes. Figures 5 and 6 show both levels.

## Failure lab

Expected failure rows contain:

```text
case_id
configuration_id
failure_stage
expected_ids
retrieved_ids
citations
retrieval_recall_at_k
reciprocal_rank
filter_correctness
citation_correctness
grounded_fact_coverage
abstention_correctness
```

The current negative cases are classified as `abstention`. The next engineering step is
an evidence-sufficiency gate after reranking and before generation. Do not tune a global
similarity threshold on two examples and call the problem solved.

## Challenge solution

The student challenge is to add a third configuration with a sufficiency gate. A valid
solution:

1. receives the query, final evidence and transparent rerank features;
2. emits a typed `sufficient: bool` decision plus a reason;
3. opens a `sufficiency` span;
4. abstains before generation when insufficient;
5. preserves recall on all six positive cases;
6. passes both negative cases; and
7. compares the third MLflow run against the same `rag-cases-v1` dataset.

The gate may start deterministic. An LLM-assisted gate is an advanced alternative only if
its prompt, provider, model, cost and errors are logged and evaluated.

## Ragas: optional final comparison

Ragas is introduced for two judge-based metrics only:

- context recall; and
- faithfulness.

The course adapter accepts an explicit judge with declared provider and model. It never
selects a default model. Offline mode returns recorded metrics or an explicit skipped
status.

State the limitations clearly:

- judge metrics are model-dependent;
- prompts and metric implementations can change;
- an LLM judge adds latency and token cost;
- judge scores can disagree with maintained business rules; and
- Ragas complements, rather than replaces, deterministic retrieval and citation checks.

## Expected outputs

Offline:

- eight cases load;
- six positive retrieval cases recover expected evidence within `final_k=2`;
- two negative cases produce classified abstention failures;
- two MLflow runs are logged;
- each run contains eight traces;
- every trace exposes the seven required stage names;
- Figures 1–9 render; and
- the final line appears exactly once:

```text
PASS — RAG evaluation and tracing verified
```

Live Ollama/OpenAI:

- corpus, filters, case alignment, span structure and metadata still pass;
- generated answer metrics are labelled observations;
- no fixed provider ranking or judge score is required; and
- the same final PASS marker proves structural execution, not identical model quality.

## Common failures

### MLflow creates or looks for the wrong store

Check the printed `FINAI_MLFLOW_DIR` and use the exact SQLite URI when launching the UI.
Do not point the UI at a different working directory.

### Traces exist but are not associated with a run

Only pass `run_id` to the root span. Nested spans inherit the active trace. Passing a run
ID to nested spans is ignored by MLflow and produces warnings.

### The OpenAI run fails before evaluation

Confirm `OPENAI_API_KEY`, chat model and embedding model. The notebook never falls back to
Ollama or offline mode after an explicit OpenAI selection.

### A live answer has no citations

That is an answer-layer result, not a retrieval failure. Inspect the trace, then revise and
version the generation prompt rather than changing the retriever first.

### Ragas starts an unexpected model

Stop. The course adapter requires an explicit judge. Do not rely on library defaults or
ambient credentials.

## Checkpoint answers

1. **Why can recall@k equal 1 while citation correctness equals 0?**  
   Expected evidence was retrieved, but the answer cited nothing or the wrong ID.

2. **Why compare configurations on one golden-set version?**  
   Otherwise the experiment changes both the system and the exam.

3. **What does the trace add to a metric table?**  
   It connects one input to stage-level work, timing, outputs and the final result.

4. **Why are the negative cases valuable?**  
   They test refusal and evidence sufficiency, which fluent positive demos omit.

5. **Why is Ragas optional?**  
   Its judge metrics add useful semantic checks but also provider dependence, variance and
   cost.

## Provider modes

### Offline

```bash
python scripts/execute_notebooks.py notebooks/07_rag_evaluation.ipynb \
  --mode offline \
  --output-dir /tmp/finai-l07-offline
```

Answers are recorded and exact. Retrieval uses the deterministic teaching embeddings.

### Ollama

```bash
FINAI_CHAT_MODEL=qwen3:8b \
FINAI_EMBEDDING_MODEL=qwen3-embedding:0.6b \
python scripts/execute_notebooks.py notebooks/07_rag_evaluation.ipynb \
  --mode live --provider ollama \
  --output-dir /tmp/finai-l07-ollama
```

### OpenAI

```bash
OPENAI_API_KEY="..." \
FINAI_CHAT_MODEL=gpt-5-mini \
FINAI_EMBEDDING_MODEL=text-embedding-3-small \
python scripts/execute_notebooks.py notebooks/07_rag_evaluation.ipynb \
  --mode live --provider openai \
  --output-dir /tmp/finai-l07-openai
```

## Transition to Day 2

Close with:

> “Day 1 built and measured the RAG path. Day 2 will put these exact stages into state,
> branches and bounded control flow. We will not add agency where a deterministic workflow
> is easier to test.”

Lesson 08 begins with the classified abstention gap and turns the measured pipeline into a
stateful LangGraph workflow.
