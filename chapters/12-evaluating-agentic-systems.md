# Lesson 12 - Evaluating agentic systems with MLflow

**First Finance - Arnaud Demes**

**Day 2 | 14:30-15:30 | 12-minute concept deck + 40-minute notebook + 8-minute verification and debrief**

## Instructor outcome

Students evaluate both the public trajectory and the cited answer produced by the
NVIDIA and Schneider Electric analyst. They verify a six-case dataset, compare two
aligned configurations, inspect local MLflow runs and traces, diagnose failures by
owner, and apply a deterministic citation release gate.

The full Lesson 12 route is ready for an instructor-led offline test class. Use this
chapter with the companion
[Lesson 12 notebook](../notebooks/12_evaluating_agentic_systems.ipynb). The
[Lesson 12 concept deck](../decks/12-evaluating-agentic-systems.pptx) is the canonical
companion path. Its visual sequence uses notebook-derived evidence and official MLflow
interfaces to move from an evaluation run into one trace, its span tree, the failed
case, and the release decision.

```text
versioned cases -> public agent trajectory -> separate scorers -> local MLflow
-> per-case diagnosis -> release decision
```

The lesson evaluates public serializable state only. It does not request hidden
reasoning or private runtime objects. The classroom route is read-only analysis using
controlled evidence. It permits no trading, portfolio mutation, price target, or
investment recommendation.

## Before class

Run from the repository root:

```bash
uv sync --extra ai --extra evaluation --extra dev
uv run python scripts/build_lesson12_notebook.py
uv run python scripts/validate_notebooks.py notebooks/12_evaluating_agentic_systems.ipynb
FINAI_MLFLOW_DIR=/private/tmp/finai-lesson12-mlflow \
  uv run python scripts/execute_notebooks.py notebooks/12_evaluating_agentic_systems.ipynb \
  --mode offline --output-dir /private/tmp/finai-lesson12-offline
uv run jupyter lab
```

Open `notebooks/12_evaluating_agentic_systems.ipynb`. Confirm six cases, two run
IDs, twelve trace IDs, four figures, and one `LESSON_12_PASS`. The notebook prints the
resolved SQLite database and artifact directory. To inspect the same store in the
optional browser UI, replace the placeholder with that printed absolute path:

```bash
mlflow ui --backend-store-uri sqlite:////absolute/path/to/mlflow.db
```

The expected address is `http://127.0.0.1:5000`. Docker is not required. The browser
UI is not required because the notebook renders the essential scorecard, trace,
comparison, and failure views inline.

Do not configure a live judge for the core class. If an extension has been prepared,
select exactly one explicit URI through `FINAI_EVAL_JUDGE_MODEL`:

```text
FINAI_EVAL_JUDGE_MODEL=openai:/<model>
FINAI_EVAL_JUDGE_MODEL=ollama_chat:/<model>
```

OpenAI and Ollama are optional comparisons. Configuration text, installed packages,
ambient credentials, or a reachable service do not prove that a judge ran. Use the
exact outcome taxonomy:

- Missing configuration or an unavailable explicit provider, adapter, client, or service is `NOT RUN`.
- A completed scorer, including a low or disagreeing score, is `COMPLETED`.
- A timeout or ordinary runtime invocation failure is `ERROR`.
- All three outcomes are observational and never change deterministic metrics or `release_passed`.

Never substitute one provider for another or invent a score.

## Evaluation cases and aligned configurations

Both configurations use the exact bytes of `agent-cases-v1` and its verified SHA-256.
A version or hash mismatch stops evaluation before scoring. Never compare partially
aligned tables.

| Case ID | Diagnostic intent |
|---|---|
| `reference_completed` | Correct calls, one typed error, one bounded replan, and a complete cited briefing. |
| `unsupported_metric_not_recovered` | Makes unrevised strategy failure visible and requires a typed stop. |
| `redundant_metric_call` | Shows that a useful answer can follow an inefficient trajectory. |
| `missing_schneider_document` | Requires the evidence gate to block an incomplete cross-company briefing. |
| `document_fact_without_evidence_id` | Rejects a fluent document claim whose evidence ID is missing. |
| `wrong_source_evidence_pair` | Rejects individually real source and evidence values when their pair was never returned. |

| Configuration ID | Contract |
|---|---|
| `bounded-agent-v1` | Certified Lesson 11 behavior and guardrails, represented by maintained public fixtures. The real Lesson 11 offline reference must match `reference_completed` before comparison. |
| `regressed-agent-v0` | Maintained regression fixtures that expose redundant, incomplete, unrecovered, or untraceable behavior. They are teaching assets and do not replace the production graph. |

One MLflow run represents one complete configuration. One root trace represents one
case. The two runs must share all six case IDs, `agent-cases-v1`, and the same exact
hash. The expected result is six traces per run and twelve traces in total.

## Five deterministic metric formulas

All five scores are in `[0, 1]`. Unless noted otherwise, a checklist metric is:

```text
score = sum(earned check values) / number of checks
```

Boolean checks contribute 1 or 0. Coverage checks contribute the fraction covered.
The public rationale lists `Satisfied` and `Missing` conditions; a partial condition
includes its covered fraction. Read the rationale with the number. A mean can hide the
case that owns a release failure, and a metric pass count includes only cases at 1.0.

### `tool_call_correctness`

```text
correctness = earned checks / total checks
checks = each expected canonical call
       + each expected typed error
       + exact expected replan count
       + each declared prerequisite ordering
```

Capabilities and canonical validated arguments define call identity. Ordering matters
only for declared dependencies. The expected `unsupported_metric` error and bounded
replan are positive parts of the reference contract, not noise to remove.

### `tool_call_efficiency`

```text
penalties = redundant successful calls
          + calls above max_tool_calls
          + calls after a terminal gate
          + replans above the expected count

efficiency = max(0, 1 - penalties / max(1, max_tool_calls))
```

Repeated successful canonical calls count as redundant. A polished answer cannot erase
wasted calls, work after a blocked gate, or excess revisions.

### `answer_relevance`

For a case that permits a briefing:

```text
relevance = (required-company coverage
             + valuation-dimension check
             + operating-growth-dimension check)
            / (number of required companies + 2)
```

In the maintained deterministic contract, valuation is signalled by `valuation` or
`P/E`; operating growth is signalled by `operating growth`, `operating-growth`, or
`revenue growth`. This is mission coverage, not general semantic similarity. For a case
that forbids a briefing, relevance is the mean of two checks: the expected typed final
status and no briefing emitted.

### `answer_completeness`

```text
completeness = mean(
  expected final status,
  required evidence-ID coverage,
  required fact-kind coverage,
  required-company coverage,
  nonempty cross-company comparison,
  required-limitation coverage
)
```

Each coverage term is the matched fraction of its versioned requirement. Missing
Schneider evidence, a fact kind, a company, a comparison, or a required currency,
reporting-period, or business-definition limitation remains visible in the rationale.

### `citation_integrity`

```text
with a briefing:
  citation_integrity = 1 only when every fact is observation-backed
                       and aggregate sources are the exact ordered union;
                       otherwise 0

without a briefing:
  citation_integrity = 1 only for the expected no-briefing typed stop;
                       otherwise 0
```

This metric is deliberately binary because it is the finance-specific release gate.
Apply these exact citation rules:

- A metric fact has `provenance_kind="metric"`, exactly one source from a successful
  metric observation, and no evidence ID.
- A document fact has `provenance_kind="document"`, exactly one source and exactly one
  evidence ID, and that exact source/evidence-ID pair appeared in one successful
  document-search hit.
- Aggregate sources are the exact ordered, de-duplicated union of the sources cited by
  the individual facts.

A plausible claim with missing provenance, an unsupported source, or a cross-paired
source and evidence ID receives zero. The release also fails if a case that forbids a
briefing emits one.

## 12-minute concept deck route

Use this exact nine-slide script. The deck first establishes why answer-only evaluation
is unsafe, then shows learners how the same evidence appears in MLflow. Slides 4, 5,
and 9 use official MLflow interface views; slides 3 and 6 use outputs generated by the
notebook.

| Time | Slide | Instructor job |
|---:|---:|---|
| 0:00-1:00 | 1 | State the outcome: evaluate both the path and the answer during the 14:30-15:30 slot. |
| 1:00-2:15 | 2 | Contrast the same plausible answer reached by a bounded path and a redundant or unsafe path. |
| 2:15-3:45 | 3 | Separate trajectory correctness and efficiency from answer relevance, completeness, and citation integrity. |
| 3:45-5:15 | 4 | Read the MLflow evaluation overview: versioned cases, one root trace per case, scorer columns, and failed rows that can be opened. |
| 5:15-7:00 | 5 | Read an actual MLflow trace detail view: span tree, inputs and outputs, tool calls and typed errors, then assessments and expectations. |
| 7:00-8:30 | 6 | Read the per-case heatmap before the means, select the weakest case, and open its matching trace. |
| 8:30-9:45 | 7 | Compare deterministic release checks with optional semantic judges. State that absent evidence means `NOT RUN`. |
| 9:45-11:00 | 8 | Diagnose answer-good/path-bad, path-good/answer-incomplete, gate, and citation failures by earliest public owner. |
| 11:00-12:30 | 9 | Read the MLflow quality dashboard, apply the deterministic release checklist, and turn failures into capstone experiments. |

The route is approximately 12 minutes. The deck establishes the evaluation model; the notebook
supplies the observed evidence.

## 40-minute notebook route

Run the streamlined 19-cell notebook from top to bottom. The mapping names every cell
and all four figures.

| Time | Cells | Instructor action | Expected visible output |
|---:|---|---|---|
| 0:00-6:00 | `lesson12-000`, `lesson12-001`, `lesson12-002`, `lesson12-003` | State the public-state boundary, initialize the offline route, verify the case hash, and define the exam before scoring anything. Ask which expectation would block release. | Six versioned cases, two aligned configurations, the exact dataset SHA-256, and Figure 1, **Versioned expectations evaluate trajectory and answer separately**. |
| 6:00-14:00 | `lesson12-004`, `lesson12-005`, `lesson12-006`, `lesson12-007` | Run the real Lesson 11 mission once, inspect its public result, align expected and observed calls, and score its public trajectory. | `Reference public signature: MATCH`; five attempts; one `unsupported_metric`; one bounded replan; and Figure 2, **One public trace retains phase, attempt, revision, status, and latency**. |
| 14:00-27:00 | `lesson12-008`, `lesson12-009`, `lesson12-010`, `lesson12-011`, `lesson12-012`, `lesson12-013` | Ask which configuration is safer, calculate answer and citation scores, log both configurations, then read cases before aggregates. | Two run IDs, twelve traces, the case-level score table, Figure 3, **Per-case metrics reveal failures hidden by configuration means**, and Figure 4, **Aligned configurations compare all five means on one dataset hash**. |
| 27:00-37:00 | `lesson12-014`, `lesson12-015`, `lesson12-016`, `lesson12-017` | Ask which case blocks release, open its persisted root trace, inspect ordered phase and tool spans, assign the earliest public owner, and keep optional judges observational. | A notebook-visible trace table with run ID, trace/root IDs, span type, status, phase, attempt, revision, typed error, guardrail result, and owner; local MLflow UI command; exactly one `LESSON_12_PASS`. |
| 37:00-40:00 | `lesson12-018` | Run the knowledge check and state the capstone handoff without choosing the architecture for learners. | Learners explain why path and answer scores remain separate and name the evidence required for release. |

The rows total 40 minutes. Never compare wall-clock latency or private runtime objects
when checking the real Lesson 11 fixture identity.

## Reading MLflow evidence

Read evidence from broadest to narrowest:

1. **Run:** confirm one configuration ID, one dataset version/hash, six cases, aggregate
   means, pass counts, tool counts, latency summary, and required JSON artifacts.
2. **Root trace:** select a failed case and confirm the case ID, mission, expected and
   observed status, scores, public trajectory, and root run association.
3. **Phase span:** move through planning, plan gate, execution, replanning, evidence
   gate, and report. Repeated phases retain stable attempt or revision attributes.
4. **Tool attempt:** inspect only safe capability names, validated arguments, status,
   typed error code, evidence IDs, sources, and duration. Do not seek hidden reasoning.
5. **Failure row:** identify the earliest actionable owner: planner, tool boundary,
   replanner, evidence gate, report writer, dataset, or judge.
6. **Release decision:** inspect case-level citation integrity and briefing policy before
   looking at the mean. Any citation failure, or an impermissible briefing, blocks the
   deterministic release.

The notebook-visible tables are sufficient for the offline route, while the MLflow UI
makes the same hierarchy easier to navigate: evaluation row → root trace → phase span →
tool attempt → assessment. A run count without score rows and an inspectable trace is
not sufficient. Use the official [trace viewer](https://mlflow.org/docs/latest/genai/tracing/observe-with-traces/ui)
and [quality dashboard](https://mlflow.org/docs/latest/genai/tracing/observe-with-traces/dashboard/)
as the visual reference.

## 8-minute verification and debrief

| Time | Instructor action | Required learner evidence |
|---:|---|---|
| 0:00-2:00 | Ask why a relevant answer can fail trajectory correctness. | Names a wrong or missing canonical call, dependency, typed error, or replan transition. |
| 2:00-4:00 | Ask why a correct trajectory can fail answer completeness. | Names missing evidence IDs, fact kinds, companies, comparison content, limitations, or final status. |
| 4:00-6:00 | Open one failed root trace and assign ownership. | Moves from run to trace to phase/tool span, then selects planner, tool boundary, replanner, gate, report, dataset, or judge from public evidence. |
| 6:00-8:00 | Verify the marker and state the release and capstone boundaries. | Exactly one `LESSON_12_PASS`, a deterministic citation decision, and a list of undecided capstone inputs. |

The rows total 8 minutes. The full slot is 12 + 40 + 8 = 60 minutes, from
14:30-15:30.

## Knowledge check answer key

1. Why are tool-call correctness and answer relevance separate scores?

   **Answer:** A useful final answer does not prove that required tools, arguments,
   dependencies, typed errors, and replans were correct. Conversely, a correct path does
   not prove that the answer addressed the companies and mission dimensions.

2. Why must both configurations use the same dataset version and exact SHA-256?

   **Answer:** Alignment makes the comparison attributable to configuration behavior.
   The version names the contract and the hash proves the exact expectation bytes; a
   mismatch stops evaluation rather than producing an apples-to-oranges mean.

3. What does a public trace add to an aggregate score table?

   **Answer:** It exposes the event, attempt, revision, typed outcome, evidence, and
   latency that caused a case score, so the earliest actionable owner can be assigned.

4. Why is citation integrity a deterministic release gate for this financial mission?

   **Answer:** A plausible financial claim can still be unsupported. Exact metric-source
   and document-source/evidence pairing makes the decision reproducible and prevents a
   good aggregate mean from releasing untraceable claims.

5. When should an LLM judge be marked `NOT RUN` instead of being replaced silently?

   **Answer:** Use the exact three-way taxonomy: missing configuration or an unavailable
   explicit provider, adapter, client, or service is `NOT RUN`; a completed scorer,
   including a low or disagreeing score, is `COMPLETED`; and a timeout or ordinary
   runtime invocation failure is `ERROR`. All three outcomes are observational and
   never change deterministic metrics or `release_passed`. Never substitute another
   route or infer success from configuration inspection.

## Bounded challenge solution contract

Choose one deterministic scorer, such as maximum plan revisions. Add one versioned
expectation, add one regression case that initially fails, then implement the scorer
until the case passes. Expose a separately named MLflow metric and retain its public
rationale. The change is complete only when the expected failing test was observed,
the implementation passes, and the metric is visible. It must not change the existing
six cases, five core metrics, citation release gate, or 40-minute route.

The same contract applies to maximum latency, forbidden write tools, evidence freshness,
or mandatory currency and period caveats. A challenge experiment is not permission to
add a trading or mutation capability.

## Recovery paths

### Dataset/hash mismatch

Stop before scoring or logging. Confirm the manifest points to `agent-cases-v1`, restore
the canonical dataset bytes, rebuild the notebook, and rerun from a fresh kernel. Do not
rewrite the manifest hash around a local edit and do not compare a partial case set.

### Local SQLite failure

Print the resolved `FINAI_MLFLOW_DIR` and verify that the selected directory is local
and writable. Select a fresh explicit directory such as
`/private/tmp/finai-lesson12-mlflow-recovery`, restart the kernel, and rerun both
configurations. An in-memory score table does not prove MLflow persistence. This is the
required recovery for a local SQLite failure.

### Trace/run association failure

Confirm that each root trace carries its configuration run ID, child spans inherit the
root association, and asynchronous trace logging is flushed before counts are queried.
Rerun both complete configurations against a fresh local directory. Do not accept fewer
than six trace IDs per run or join traces to runs by display order.

### Missing provider

Keep the deterministic route green and show `NOT RUN` for the explicit OpenAI or Ollama
judge. Do not infer a score from an installed client, model URI, service health, or
credential. Continue the core lesson without the live extension.

### Judge timeout or disagreement

Record the provider, model, scorer, version, latency, status, and sanitized rationale as
an observation. Classify a timeout as `ERROR`; if the scorer completes with a low or
disagreeing score, classify it as `COMPLETED`. Neither outcome overwrites the five
deterministic scores, changes `release_passed`, or justifies editing the versioned
expectation during class.

### Suspected secret or private-data exposure

Stop execution and do not present or commit the affected store. Treat a suspected secret
or private-data exposure as an incident: rotate a real credential through the approved
process, select a fresh local directory, and verify that only safe public fields are
logged before rerunning. Block credential-shaped strings, environment dumps, raw
provider errors, personal paths, private document content, and hidden reasoning.

## No-network fallback

The full deterministic route requires no network, provider, Docker service, or browser:

```bash
FINAI_MLFLOW_DIR=/private/tmp/finai-lesson12-mlflow \
  uv run python scripts/execute_notebooks.py notebooks/12_evaluating_agentic_systems.ipynb \
  --mode offline --output-dir /private/tmp/finai-lesson12-offline
```

This route still verifies the dataset hash, runs the real local Lesson 11 MCP reference,
logs both configurations to local SQLite and artifacts, creates twelve traces, renders
all six figures and visible tables, diagnoses failures, and prints one
`LESSON_12_PASS`. OpenAI and Ollama remain `NOT RUN`; that is truthful provider
coverage, not a core-route failure.

## Skip if late

Keep the complete executable route but compress discussion:

1. Run all cells in order so hashing, persistence, trace counts, and final assertions
   remain valid.
2. At `lesson12-003` and `lesson12-007`, retain exact case hashing and the real Lesson 11
   public-signature alignment.
3. Use Figures 1 and 5 to preserve trajectory/answer separation and aligned comparison.
4. At `lesson12-014`, retain the citation gate and exact pairing rules.
5. At `lesson12-017` and `lesson12-020`, inspect at least one failed trace and assign its
   public owner.
6. Run `lesson12-024` and require the single `LESSON_12_PASS` marker.

Skip the extended Figure 2 and Figure 3 narration, the second failure pattern, the
browser UI, live judges, and the challenge. Do not skip case hashing, the real reference
match, path-versus-answer scoring, citation integrity, one failed trace, or verification.

## Capstone discussion, without an architecture decision

Lesson 12 contributes versioned regression cases, public agent traces, trajectory and
answer scorecards, the citation release gate, aligned MLflow run comparison, and
per-case failure ownership. These are discussion inputs, not an application design.

The instructor and course owner still decide:

- application surface;
- interaction model;
- read-only tool set;
- document corpus and data policy;
- evaluation thresholds and release policy; and
- demonstration mission.

Discuss how each option would consume the evaluation evidence. Do not choose a UI,
runtime topology, provider, orchestration framework, or deployment architecture in this
lesson.

## Sources

- [MLflow built-in LLM judges](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined/)
- [MLflow evaluating traces](https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/traces/)
- [MLflow Python API for `mlflow.genai`](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.genai.html)
- [Lesson 07 instructor chapter](07-rag-evaluation.md)
- [Lesson 11 instructor chapter](11-plan-and-execute-analyst.md)
- [Lesson 12 design](../docs/superpowers/specs/2026-08-22-lesson-12-evaluating-agentic-systems-with-mlflow-design.md)
