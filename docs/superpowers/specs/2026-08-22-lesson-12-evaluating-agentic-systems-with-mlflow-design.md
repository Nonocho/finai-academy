# Lesson 12 Evaluating Agentic Systems with MLflow Design

## Status

Proposed written specification for review. Arnaud Demes approved the in-chat
architecture on 22 August 2026. Implementation begins only after this written
specification is approved.

## Purpose

Lesson 12 closes the technical course sequence by evaluating the Financial Analyst
Copilot built in Lessons 08 to 11. It does not build another agent. It turns the public
trajectory and cited briefing from Lesson 11 into a versioned, repeatable MLflow
evaluation suite.

The lesson answers one engineering question:

> A financial agent returned a plausible briefing, but did it use the right tools, in
> the right order, without unnecessary calls, and did the final answer remain complete
> and traceable to evidence?

The teaching contract is:

```text
versioned cases -> public agent trajectory -> separate scorers -> MLflow traces ->
per-case diagnosis -> release decision
```

## Decision basis

The design adapts the progression of MLExpert Academy's **Evaluating Agentic Systems**
lesson without copying its source code, lesson copy, or assets. The reusable idea is to
score the path and the answer separately, then inspect individual failures rather than
trusting one aggregate score.

The design also reuses the course's existing Lesson 07 MLflow boundary:

- a local SQLite tracking database;
- local artifacts;
- notebook-visible scorecards that do not require the browser UI;
- explicit configuration and dataset versions;
- deterministic offline evaluation; and
- optional provider-backed semantic judges.

Lesson 12 uses MLflow only. Ragas is not part of this lesson.

## Position in the two-day course

| Time | Activity | Observable result |
|---|---|---|
| 14:30-14:42 | Concept deck | Learners separate trajectory quality from answer quality. |
| 14:42-15:22 | Guided notebook | Two agent configurations are logged, scored, traced, and compared. |
| 15:22-15:30 | Verification and debrief | One failing case is diagnosed and `LESSON_12_PASS` is verified. |

The complete route is 12 minutes of slides, 40 minutes in a prebuilt visual notebook,
and 8 minutes of verification and debrief. Learners do not implement MLflow plumbing or
rebuild the agent during class.

## Learning objectives

By the end of the lesson, a learner can:

1. explain why a strong final answer does not prove that an agent followed a safe or
   efficient path;
2. define a versioned agent evaluation case with explicit trajectory and answer
   expectations;
3. distinguish trajectory correctness, trajectory efficiency, answer relevance, answer
   completeness, and finance-specific citation integrity;
4. log one reproducible agent-evaluation configuration to MLflow;
5. inspect one trace from mission to plan, tool calls, replanning, evidence gate, and
   final briefing;
6. compare aligned cases across two configurations without changing the dataset;
7. identify whether a failure belongs to the planner, tool boundary, replanner, evidence
   gate, report writer, dataset, or judge; and
8. explain when a deterministic scorer is preferable to an LLM-as-judge and when a
   semantic judge adds useful information.

## Classroom boundaries

- MLflow is the only evaluation and observability framework introduced in Lesson 12.
- The core notebook runs offline with no API key and no external tracking server.
- OpenAI and Ollama are optional live judge routes, never silent fallbacks.
- The lesson evaluates public, serializable agent state. It does not request or store
  hidden chain-of-thought.
- No trading, portfolio mutation, price target, or investment recommendation is scored.
- NVIDIA and Schneider Electric remain the maintained financial examples.
- The source notebook is committed without outputs, secrets, absolute user paths, or
  provider-specific results.
- The optional challenge is nonessential and may be skipped without losing core concepts.
- Visible deck copy uses short professional English and contains no em dash characters.
- Every slide uses the footer `First Finance - Arnaud Demes`.

## Why this is not a second Lesson 07

Lesson 07 evaluates a RAG pipeline by retrieval, filters, citations, grounding, and
abstention. Lesson 12 evaluates an agentic system whose failure surface includes action
selection and control flow.

| Lesson 07 | Lesson 12 |
|---|---|
| Did retrieval find the right evidence? | Did the agent choose the right capabilities? |
| Did the answer cite retrieved evidence? | Did the agent avoid redundant or unsafe calls? |
| Did the pipeline abstain correctly? | Did replanning and the evidence gate behave correctly? |
| Which RAG stage failed? | Did the trajectory or the final briefing fail? |

Lesson 12 reuses MLflow concepts rather than reteaching experiment tracking from zero.

## Lesson 11 input contract

Lesson 12 receives the mission from the aligned `AgentEvaluationCase` and converts the
public `PlanExecuteResult` boundary into an `AgentEvaluationPrediction` without losing:

- initial and final plan steps;
- capability names and validated arguments;
- observation status, error code, duration, evidence IDs, and source references;
- public trajectory events;
- replan count;
- evidence-gate result;
- final `AnalystBriefing` sections;
- each fact's `provenance_kind`; and
- stage-level latency.

The evaluator never depends on the graph's private runtime objects, MCP session, provider
client, prompt internals, or hidden reasoning.

## Architecture

```text
agent-cases-v1 expectations               recorded agent outputs
              |                                   |
              +--------------+--------------------+
                             |
                    typed alignment gate
                             |
       +---------------------+---------------------+
       |                                           |
trajectory scorers                             answer scorers
       |                                           |
       +---------------------+---------------------+
                             |
                    MLflow run and traces
                             |
          inline scorecard + optional local UI
                             |
                    per-case diagnosis
```

Pure evaluation logic remains independent of MLflow. MLflow receives already validated
cases, predictions, scores, and public trace fields. Observability code must not change
agent behavior.

## Components and file boundaries

### `src/finai_academy/agent_evaluation.py`

Provider-neutral contracts and deterministic scorers:

- `AgentEvaluationCase`;
- `ExpectedToolCall`;
- `CandidateFact`;
- `AgentEvaluationPrediction`;
- `AgentCaseScores`;
- `AgentEvaluationSummary`;
- dataset loading and SHA-256 verification;
- public `PlanExecuteResult` serialization;
- exact case/prediction alignment;
- deterministic trajectory and answer scorers; and
- failure classification.

This module must not import MLflow.

`AgentEvaluationPrediction` is a safe evaluation record, not a second certified
`PlanExecuteResult`. A valid Lesson 11 result converts into it without information loss.
Recorded regression fixtures also use it so the evaluator can represent and score a
candidate claim with missing or incorrect provenance. `CandidateFact` therefore permits
the candidate fields needed for diagnosis, while `citation_integrity` performs the
strict Lesson 11 provenance check. The loader still rejects blank text, unknown fields,
secret-shaped content, and malformed container types.

### `src/finai_academy/mlflow_agent_evaluation.py`

MLflow-only integration:

- local SQLite and artifact initialization;
- reproducibility parameters;
- one root trace per case;
- public child spans reconstructed from the Lesson 11 trajectory;
- score and failure-row logging;
- aligned configuration comparison;
- optional MLflow GenAI scorer construction; and
- an inline summary equivalent to the essential UI views.

### Versioned data

Canonical files:

```text
assets/course-data/evaluation/agent_cases_v1.json
assets/course-data/evaluation/agent_runs_v1.json
assets/course-data/manifest.json
```

Expectations and observed outputs stay separate. A change to a mission, expected call,
required evidence ID, maximum call budget, or expected final status creates a new dataset
version and hash. Recorded outputs are labelled course fixtures, not live provider runs.

### Classroom artifacts

```text
scripts/build_lesson12_notebook.py
notebooks/12_evaluating_agentic_systems.ipynb
chapters/12-evaluating-agentic-systems.md
decks/12-evaluating-agentic-systems.pptx
docs/reviews/lesson-12-readiness.md
```

The implementation also updates the root, chapter, notebook, and deck indexes;
`assets/course-data/manifest.json`; `.env.example`; and the onboarding copy. The
evaluation extra remains the installation boundary. Add a narrowly pinned judge-client
dependency only if the installed MLflow version requires it for the explicit Ollama or
OpenAI model URI.

### Test surfaces

```text
tests/test_agent_evaluation.py
tests/test_mlflow_agent_evaluation.py
tests/test_lesson12_assets.py
tests/test_course_manifest.py
```

## Versioned evaluation dataset

The core dataset contains six small regression cases over the maintained NVIDIA and
Schneider Electric analyst mission. It evaluates the existing system rather than adding
a new general-purpose stock agent.

| Case | Intended diagnostic |
|---|---|
| `reference_completed` | Correct tools, one typed error, one bounded replan, full cited briefing. |
| `unsupported_metric_not_recovered` | Failure remains visible when strategy is not revised. |
| `redundant_metric_call` | Correct evidence can still come from an inefficient trajectory. |
| `missing_schneider_document` | The evidence gate must block an incomplete cross-company briefing. |
| `document_fact_without_evidence_id` | A fluent claim without document provenance must fail citation integrity. |
| `wrong_source_evidence_pair` | A real source and evidence ID still fail when their pair is invalid. |

Each case stores:

```text
case_id
mission
expected_final_status
expected_tool_calls
expected_error_codes
expected_replan_count
max_tool_calls
required_companies
required_evidence_ids
required_fact_kinds
required_limitations
allow_briefing
```

Each expected tool call stores a stable call ID, the capability, canonical validated
arguments, and any prerequisite call IDs. This makes dependency-aware ordering explicit
instead of inferring it from list position. The dataset never stores chain-of-thought or
secrets.

## Recorded configurations

The offline notebook compares two aligned configurations:

1. `bounded-agent-v1`: the certified Lesson 11 behavior and guardrails;
2. `regressed-agent-v0`: maintained public regression fixtures that demonstrate wrong,
   redundant, incomplete, or untraceable behavior.

Both use the same six expectation rows. The regression fixture is an evaluation teaching
asset. It does not weaken or replace the production Lesson 11 graph.

The notebook also runs the real certified Lesson 11 offline mission once. That result
must match the `reference_completed` fixture on public trajectory signatures, final
status, evidence IDs, and cited facts before the scorecard is trusted.

## Deterministic score contract

Every deterministic score is in `[0, 1]` and includes a short public rationale.

### `tool_call_correctness`

Score observed capability and canonical argument signatures against expected calls.
Ordering is enforced only where the case declares a dependency. A required typed failure
and expected replan transition are part of correctness.

### `tool_call_efficiency`

The score penalizes repeated successful canonical calls, calls above the case budget,
continued execution after a terminal gate, and replans above the declared limit. A
correct final briefing does not erase an inefficient path.

### `answer_relevance`

For the deterministic route, relevance means the output addresses the maintained mission
dimensions and companies, or returns the expected typed stop when reporting is not
permitted. It is not a general semantic-similarity claim.

### `answer_completeness`

Completeness checks required evidence IDs, fact kinds, comparison section, explicit
limitations, and expected final status. Missing Schneider evidence, a missing limitation,
or a missing requested company lowers this score.

### `citation_integrity`

This finance-specific custom scorer is a release gate, not an optional style metric. It
reuses the Lesson 11 provenance contract:

- metric fact: one source from a successful metric observation and no evidence ID;
- document fact: one exact returned source/evidence-ID pair; and
- aggregate sources: exact ordered union of cited facts.

Unsupported or cross-paired citations receive zero even when the sentence is plausible.

## Aggregate reporting

Aggregate means are useful for comparison but never replace per-case inspection. The
inline scorecard shows:

- metric mean by configuration;
- pass count by metric;
- per-case metric heatmap;
- total and redundant tool-call counts;
- mean and maximum latency;
- failure stage; and
- linkable MLflow run and trace IDs.

The release decision is blocked when any case fails citation integrity or when a case
that requires an evidence-gate stop produces a briefing.

## MLflow run model

One MLflow run represents one complete evaluation configuration. Required parameters:

```text
configuration_id
dataset_version
dataset_sha256
agent_version
provider
agent_model
judge_provider
judge_model
prompt_version
max_steps
max_replans
scorer_contract_version
```

Required aggregate metrics:

```text
tool_call_correctness_mean
tool_call_efficiency_mean
answer_relevance_mean
answer_completeness_mean
citation_integrity_mean
mean_tool_calls
mean_latency_ms
```

Required artifacts:

```text
evaluation/case_scores.json
evaluation/failure_rows.json
evaluation/dataset_manifest.json
```

No API key, raw environment dump, complete prompt, personal path, or private document
content may be logged.

## MLflow trace model

One root trace represents one evaluation case. It records case ID and mission, expected
and observed status, public plan revisions, tool-call signatures, observation status and
error codes, evidence IDs and sources, evidence-gate result, public briefing sections,
final scores, and case latency.

Child spans mirror the public Lesson 11 phases:

```text
planning -> plan_gate -> execution -> replanning -> evidence_gate -> report
```

Planning, plan gate, replanning, evidence gate, and report are chain spans. Each observed
execution attempt is a `TOOL` span whose safe inputs contain the capability and validated
arguments and whose safe outputs contain status, typed error code, evidence IDs, source
references, and duration. This gives MLflow tool-call scorers a real trace surface rather
than a text summary.

Repeated execution and replanning phases receive stable attempt/revision attributes. A
recorded offline trace is explicitly labelled `recorded`. A live trace is explicitly
labelled `openai` or `ollama`.

## Local MLflow behavior

The core uses the same local SQLite pattern as Lesson 07. The notebook creates a
directory selected by `FINAI_MLFLOW_DIR` or a safe temporary directory, then prints the
exact database path and UI command.

```bash
mlflow ui --backend-store-uri sqlite:////absolute/path/to/mlflow.db
```

The expected UI is `http://127.0.0.1:5000`. The UI is optional during class because the
notebook renders the scorecard, trace timeline, comparison, and failure table inline.

Docker is not required for the core route. A later deployment or capstone extension may
point the same client at a shared MLflow server.

## Optional MLflow GenAI judges

The live extension uses the current MLflow GenAI scorer interface and four built-in
judges:

- `ToolCallCorrectness`;
- `ToolCallEfficiency`;
- `RelevanceToQuery`; and
- `Completeness`.

These judges are provider-dependent and some tool-call scorer APIs are experimental. The
lesson treats them as an observed comparison, not as the deterministic release gate.

The judge model must be explicit:

```text
FINAI_EVAL_JUDGE_MODEL=openai:/<model>
FINAI_EVAL_JUDGE_MODEL=ollama_chat:/<model>
```

Recommended classroom values are configured through `.env`, not hard-coded into the
notebook. No provider is selected from ambient credentials.

Judge results log provider, model, scorer name, MLflow version, latency, and status. If a
judge is unavailable, the result is `NOT RUN` with a reason. The notebook never
substitutes another provider or awards judge credit from static configuration.

## Notebook design

Canonical file: `notebooks/12_evaluating_agentic_systems.ipynb`

The source notebook contains 24 to 28 stable, output-free cells and at least six rendered
PNG visuals. It follows this sequence:

1. title, outcome, prerequisites, timing, and visible outputs;
2. final Financial Analyst Copilot evaluation increment;
3. two-layer evaluation architecture;
4. offline/OpenAI/Ollama mode selection;
5. load and hash `agent-cases-v1`;
6. inspect one expectation row;
7. run the real Lesson 11 offline reference mission;
8. align the reference result with its recorded public fixture;
9. inspect one public plan and trajectory;
10. score tool-call correctness;
11. score tool-call efficiency;
12. score answer relevance and completeness;
13. enforce citation integrity;
14. log `bounded-agent-v1` to local MLflow;
15. log `regressed-agent-v0` against the same dataset;
16. compare aggregate scores;
17. inspect a per-case score heatmap;
18. open one trace timeline inline;
19. diagnose one answer-good/path-bad case;
20. diagnose one path-good/answer-incomplete case;
21. show the optional MLflow judge configuration;
22. print the MLflow UI command;
23. run deterministic verification;
24. knowledge check;
25. optional custom-scorer challenge;
26. capstone handoff; and
27. exact final marker `LESSON_12_PASS`.

### Required notebook visuals

1. trajectory versus answer evaluation architecture;
2. expectation versus observed tool-call sequence;
3. one agent trace timeline;
4. per-case metric heatmap;
5. aligned configuration comparison; and
6. failure-diagnosis matrix.

The scorecard and cited briefing remain visible as tables or text in addition to the
figures. Counts alone do not satisfy the notebook contract.

## Forty-minute notebook pacing

| Time | Work | Expected output |
|---|---|---|
| 0:00-4:00 | Load and inspect the versioned cases. | Six cases, version, hash, Figure 1. |
| 4:00-9:00 | Run and align the Lesson 11 reference. | Public plan, one typed error, one replan. |
| 9:00-15:00 | Score the trajectory. | Correctness and efficiency, Figures 2-3. |
| 15:00-21:00 | Score the briefing. | Relevance, completeness, citation integrity. |
| 21:00-27:00 | Log both configurations to MLflow. | Two run IDs and aligned cases. |
| 27:00-33:00 | Compare scorecards. | Heatmap and configuration chart, Figures 4-5. |
| 33:00-37:00 | Diagnose two failure patterns. | Failure matrix, Figure 6. |
| 37:00-40:00 | Verify, show optional judge route, and pass. | One `LESSON_12_PASS`. |

## Concept deck design

Canonical file: `decks/12-evaluating-agentic-systems.pptx`

The deck follows the Lesson 10 and Lesson 11 visual system and contains nine slides:

1. lesson title, outcome, and timing;
2. same answer, different path;
3. trajectory versus answer recap table;
4. versioned case and expectation contract;
5. MLflow run and trace architecture;
6. anatomy of one agent trace;
7. deterministic scorers versus LLM judges;
8. scorecard patterns and failure ownership; and
9. release gate and capstone handoff.

Every mechanism slide uses an original diagram. Slides 3, 7, and 8 use concise comparison
tables. All slides contain source notes and the exact footer.

## Twelve-minute deck pacing

| Time | Slide | Instructor job |
|---|---:|---|
| 0:00-1:00 | 1 | State the observable outcome. |
| 1:00-2:15 | 2 | Show why final-answer quality is insufficient. |
| 2:15-3:45 | 3 | Separate trajectory and answer metrics. |
| 3:45-5:00 | 4 | Explain the versioned evaluation case. |
| 5:00-6:30 | 5 | Map cases, scorers, runs, and traces. |
| 6:30-8:00 | 6 | Read one trace from plan to report. |
| 8:00-9:30 | 7 | Compare deterministic rules and semantic judges. |
| 9:30-11:00 | 8 | Diagnose four score patterns. |
| 11:00-12:00 | 9 | State the release gate and capstone handoff. |

## Eight-minute verification and debrief

| Time | Instructor action | Learner evidence |
|---|---|---|
| 0:00-2:00 | Ask why a relevant answer can fail trajectory correctness. | Names wrong or missing tool calls. |
| 2:00-4:00 | Ask why a correct trajectory can fail completeness. | Names missing facts or limitations. |
| 4:00-6:00 | Open one failed trace and assign ownership. | Planner, tool, gate, report, dataset, or judge. |
| 6:00-8:00 | Verify the pass marker and state the capstone boundary. | One `LESSON_12_PASS` and a release decision. |

## Knowledge check

1. Why are tool-call correctness and answer relevance separate scores?
2. Why must two configurations use the same dataset version and hash?
3. What does a trace add to an aggregate score table?
4. Why is citation integrity a deterministic release gate for this financial use case?
5. When should an LLM judge be marked `NOT RUN` rather than replaced silently?

## Optional challenge

Add one custom deterministic MLflow scorer for maximum latency, forbidden write tools,
evidence freshness, mandatory currency and period caveats, or maximum plan revisions.

The challenge is complete only when it adds a versioned expectation, a failing regression
case, a passing implementation, and a visible MLflow metric. It must not change the core
40-minute route.

## Failure handling

### Dataset mismatch

Fail before scoring when case IDs, prediction IDs, dataset version, or SHA-256 do not
align. Never compare partially aligned tables.

### MLflow backend failure

Print the resolved local directory and actionable setup error, then fail verification.
An in-memory table does not prove MLflow integration.

### Missing provider or judge

Keep the deterministic route green and label the live judge `NOT RUN`. Do not infer a
score from configuration.

### Judge timeout or disagreement

Log the judge failure or rationale as an observation. Do not overwrite the deterministic
release gate or edit the golden set during the lesson.

### Secret or private data exposure

Block credential-shaped fields, environment dumps, raw provider errors, personal paths,
and hidden reasoning. Public fixture paths, evidence IDs, safe capability names, and
typed error codes are permitted.

### Trace/run association failure

The root trace receives the run ID. Child spans inherit it. Flush asynchronous trace
logging before querying counts or ending the notebook.

## Testing strategy

Implementation follows test-driven development.

### Pure evaluation tests

- strict dataset schema and nonblank fields;
- unique case IDs and deterministic hash verification;
- exact case/prediction alignment;
- canonical tool-call signatures and dependency-aware ordering;
- redundancy, budget, and typed-stop scoring;
- answer relevance and completeness;
- citation integrity; and
- stable failure classification.

### MLflow integration tests

- local SQLite database and artifacts;
- one run per configuration and one trace per case;
- required public child spans;
- required parameters, metrics, and artifacts;
- no secret-shaped logged values;
- aligned comparison output;
- explicit judge `NOT RUN`; and
- trace flushing before inspection.

### Notebook and classroom tests

- 24 to 28 stable output-free cells;
- six required PNG visuals;
- exact timing and headings;
- real Lesson 11 offline reference execution;
- two MLflow runs on the same six cases;
- visible per-case scorecard and trace;
- one exact `LESSON_12_PASS` marker;
- OpenAI and Ollama instructions; and
- no mandatory Docker, browser UI, or live provider for the core route.

### Deck tests

- exactly nine slides and exact footer on every slide;
- no visible em dash and concise English copy;
- required comparison tables and diagrams;
- directly relevant source notes;
- no overflow, clipping, collision, or placeholder content; and
- visual consistency with Lessons 10 and 11.

### Final certification

- full repository tests and Ruff;
- notebook and repository validators;
- fresh offline notebook execution and inspection of every figure;
- PowerPoint structure, render, overflow, and full-size visual review;
- optional Ollama run when available;
- optional OpenAI run only when configured;
- timed learner rehearsal if performed; and
- a readiness report that separates lesson quality from provider coverage.

No readiness report may award provider or rehearsal credit without observed evidence.

## Acceptance criteria

Lesson 12 is ready for an instructor-led offline test class when:

- the six-case dataset and recorded outputs are versioned and hash-verified;
- the real Lesson 11 offline reference matches its maintained public fixture;
- trajectory and answer scores are computed separately;
- citation integrity blocks untraceable financial claims;
- two aligned configurations are logged to local MLflow;
- every case has an inspectable trace and score row;
- the notebook renders at least six readable visuals;
- exactly one `LESSON_12_PASS` appears;
- the nine-slide deck passes visual and structural QA;
- all automated tests and validators pass; and
- an independent review scores lesson quality at 9.5/10 or higher with no unresolved
  Important or Critical finding.

Live provider certification and timed rehearsal are reported separately from lesson
quality and the offline result.

## Capstone handoff

After Lesson 12 is certified, no capstone architecture is assumed automatically. The
instructor and course owner first brainstorm the Financial Analyst Copilot together.

Lesson 12 contributes:

```text
versioned regression cases
public agent traces
trajectory scorecard
answer scorecard
citation release gate
MLflow run comparison
per-case failure ownership
```

The capstone discussion decides the final application surface, interaction model, tool
set, document corpus, evaluation threshold, and demonstration mission.

## Sources

- [MLExpert Academy: Evaluating Agentic Systems](https://www.mlexpert.io/academy/v1/ai-agents/agent-evaluation)
- [MLExpert Academy: Plan and Execute Agent](https://www.mlexpert.io/academy/v1/ai-agents/planning-agent)
- [MLflow: Tracing LangGraph](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/langgraph)
- [MLflow: Built-in LLM Judges](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined/)
- [MLflow: Evaluating Traces](https://mlflow.org/docs/latest/genai/eval-monitor/running-evaluation/traces/)
- [MLflow Python API: `mlflow.genai`](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.genai.html)
- [Lesson 07 instructor chapter](../../../chapters/07-rag-evaluation.md)
- [Lesson 11 instructor chapter](../../../chapters/11-plan-and-execute-analyst.md)
- [Day 2 progression design](2026-08-21-day-two-agent-progression-and-lesson-08-design.md)
- [Lesson 11 design](2026-08-22-lesson-11-plan-and-execute-financial-analyst-design.md)
