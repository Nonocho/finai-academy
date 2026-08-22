# Lesson 11 Plan-and-Execute Financial Analyst Design

## Status

Approved by Arnaud Demes on 22 August 2026. This specification is the authoritative
contract for the Lesson 11 implementation plan and classroom artifacts.

Lesson 11 adapts the progression of the MLExpert Academy Plan and Execute Agent
lesson without reproducing its code, copy, diagrams, datasets, or exercises. The
course uses original NVIDIA and Schneider Electric evidence, the real read-only MCP
boundary built in Lesson 10, original visuals, and provider-neutral execution.

## Lesson position

| Field | Contract |
| --- | --- |
| Lesson | 11 - Plan-and-execute financial analyst |
| Day | 2 |
| Time | 13:30-14:30 |
| Duration | 60 minutes |
| Prerequisites | Lessons 01-10, especially LangGraph recovery and financial MCP discovery |
| Capstone increment | Planned multi-source financial research with citations and a full trajectory |
| Next lesson | 12 - Evaluating agentic systems |

Lesson 08 established when bounded autonomy is justified. Lesson 09 added typed
recovery inside a LangGraph loop. Lesson 10 replaced direct tool coupling with runtime
MCP discovery and host-owned permission. Lesson 11 coordinates several independent
research steps before synthesis. Lesson 12 will score both the resulting answer and the
trajectory that produced it.

## Engineering question

Lesson 11 answers one question:

> How can a financial analyst application plan several evidence-gathering steps,
> execute them through discovered read-only capabilities, revise only the unfinished
> work, and produce a cited briefing without surrendering control to the model?

The lesson must not imply that planning always improves an agent. Plan-and-execute is
appropriate when a request has a useful decomposition into multiple research steps and
the final synthesis benefits from seeing the complete evidence set. A direct workflow or
small ReAct loop remains preferable for simple requests.

## Learning objectives

By the end of the lesson, a learner can:

1. distinguish ReAct's next-action loop from an upfront plan-and-execute pattern;
2. represent a financial research plan with validated Pydantic models;
3. separate model-owned proposals from host-owned validation and execution;
4. initialize an executor from the tools discovered through the Lesson 10 MCP server;
5. execute one plan step at a time while preserving evidence IDs, dates, and sources;
6. replan only the unexecuted tail after a typed tool failure or changed observation;
7. enforce tool, step, replan, and evidence boundaries;
8. generate a briefing that separates facts, comparison, interpretation, and limitations;
9. inspect plan changes and the complete agent trajectory visually; and
10. identify the answer and trajectory fields that Lesson 12 will evaluate.

## Scope boundaries

- The classroom core uses one plan-and-execute graph, not a multi-agent team.
- The planner, replanner, and report writer are distinct roles inside one bounded system.
- The planner and replanner may use an LLM; the executor and policy gates are deterministic.
- The executor may call only runtime-discovered and statically allowlisted read-only tools.
- The MCP prompt `compare_companies` is user-controlled prompt material, not an executable tool.
- No trade, recommendation, portfolio mutation, transaction, credential, or private file is used.
- Controlled course evidence is not presented as live market data or investment advice.
- The core executor is sequential so every state transition remains visible.
- Parallel execution is an optional challenge, not a classroom dependency.
- The notebook works without a paid API. Live Ollama and OpenAI are explicit extensions.
- Source notebooks committed to Git contain no outputs or secrets.

## Financial research mission

The maintained mission is:

> Produce a concise NVIDIA and Schneider Electric briefing. Compare their available
> valuation metrics and latest operating-growth evidence. Cite every factual claim and
> state which observations cannot be compared directly.

The mission is intentionally analytical rather than advisory. It combines:

- NVIDIA (`NVDA`) and Schneider Electric (`SU.PA`);
- `P/E` and `EPS` from the controlled dated metric snapshot;
- NVIDIA Data Center, total revenue, and Gaming evidence;
- Schneider Electric revenue, adjusted EBITA margin, and Energy Management evidence;
- cross-company limitations caused by different currencies, reporting periods, business
  mixes, and metric definitions; and
- an explicit refusal to convert these observations into a buy, sell, or hold recommendation.

The current Lesson 10 evidence catalog remains authoritative. Lesson 11 does not add
unsupported facts merely to make the final report richer.

## Architecture decision

### Selected approach: MCP-native plan-and-execute

Lesson 11 uses the same real local `stdio` MCP server created in Lesson 10. Runtime
discovery produces a tool catalog. Host policy intersects that catalog with the static
allowlist before the planner sees executable capability metadata. The planner proposes a
typed plan, Python validates it, the deterministic executor calls one allowed MCP tool,
and the replanner reviews the resulting scratchpad.

```text
user mission
    |
    v
runtime MCP discovery -> host allowlist -> planner-visible catalog
    |                                      |
    |                                      v
    |                              structured research plan
    |                                      |
    +------------------------------> plan policy gate
                                           |
                                           v
                                  deterministic executor
                                           |
                                           v
                                    evidence scratchpad
                                           |
                                           v
                                      replanner
                                  / continue | finish \
                                 v           v          v
                            next step   evidence gate   stop
                                             |
                                             v
                                        report writer
                                             |
                                             v
                                      cited briefing
```

### Rejected alternatives

#### Direct Python ToolRegistry

A direct registry closely mirrors the reference tutorial and is easier to implement, but
it would disconnect Lesson 11 from the MCP lifecycle and permission boundary taught
immediately before it. The course will still teach the registry concept, but its entries
come from MCP discovery rather than duplicated direct imports.

#### Supervisor multi-agent system

A supervisor plus specialist agents introduces more model calls, prompts, routing, and
failure surfaces without improving the one-hour learning objective. It remains a capstone
extension after the single-system trajectory is evaluated.

## Component design

### 1. Discovered tool catalog

The Lesson 11 executor consumes the Lesson 10 discovery result and exposes planner-safe
metadata for exactly two tools:

```text
get_company_metric
search_financial_documents
```

The catalog contains only name, description, and input schema. It never contains process
arguments, environment values, API keys, stderr, or arbitrary server content. A discovered
tool is not permitted unless its name also appears in the static host allowlist.

### 2. Typed plan contracts

The implementation defines these stable Pydantic contracts:

```python
class PlanStep(BaseModel):
    step_id: int
    capability: str
    arguments: dict[str, Any]
    purpose: str
    expected_evidence: tuple[str, ...]
    depends_on: tuple[int, ...] = ()


class ResearchPlan(BaseModel):
    goal: str
    steps: tuple[PlanStep, ...]


class ReplanDecision(BaseModel):
    action: Literal["continue", "replace_remaining", "finish", "stop"]
    reasoning: str
    replacement_steps: tuple[PlanStep, ...] = ()
    limitations: tuple[str, ...] = ()
```

Validation requires:

- unique initial `step_id` values beginning at 1 and increasing by one;
- replacement-step IDs greater than every ID already proposed or attempted;
- capabilities present in the permitted discovered catalog;
- arguments accepted by the discovered input schema and local business rules;
- dependencies that refer only to earlier plan steps;
- a maximum of six total execution steps;
- a maximum of one plan revision in the maintained classroom route; and
- purposes and expected-evidence labels that are non-empty learner-readable text.

The LLM proposes data. It does not create a Python callable or bypass validation.

### 3. Observation and trajectory contracts

Each executed step produces an immutable observation:

```python
class ResearchObservation(BaseModel):
    attempt_id: int
    step_id: int
    plan_revision: int
    capability: str
    arguments: dict[str, Any]
    status: Literal["ok", "error", "blocked"]
    result: dict[str, Any] | None
    error_code: str | None
    evidence_ids: tuple[str, ...]
    source_references: tuple[str, ...]
    duration_ms: float
```

`attempt_id` is globally sequential within one research run. `step_id` identifies the
proposed unit of work, and `plan_revision` records which plan introduced it. A corrected
retry receives a new `step_id`; the rejected step remains visible in the trajectory rather
than being overwritten.

The trajectory separately records planning, policy, execution, replanning, evidence gate,
report, and guardrail events. It must be safe to display and must not expose prompts,
credentials, environment variables, subprocess configuration, or hidden reasoning.

### 4. Graph state

The graph uses an explicit `PlanExecuteState` with:

```text
question
permitted_catalog
plan
current_step
scratchpad
replan_count
trajectory
status
briefing
limitations
```

State updates are returned by nodes rather than mutated in place. The executed prefix of
the plan and its observations are append-only.

### 5. Graph nodes and routing

The conceptual architecture preserves the four roles used by the reference pattern:

1. **Planner**: returns a validated `ResearchPlan` from the mission and permitted catalog.
2. **Executor**: deterministically dispatches one permitted MCP tool call by name.
3. **Replanner**: reviews the scratchpad and unfinished steps, then returns a typed decision.
4. **Report writer**: synthesizes the complete verified scratchpad into a structured briefing.

The production teaching graph adds two deterministic controls:

- **Plan policy gate** between planner and executor; and
- **Evidence gate** before report generation.

Conditional routing after replanning chooses one of four outcomes:

```text
continue -> execute next validated step
replace_remaining -> validate revised tail, then execute
finish -> evidence gate
stop -> bounded failure result
```

### 6. Deterministic MCP executor

The executor holds one local MCP client lifecycle for a complete research run:

```text
start subprocess -> initialize -> discover -> execute steps -> close
```

It never asks the LLM how to invoke a tool after planning. It reads `capability` and
`arguments` from the validated step and calls the discovered allowlisted tool directly.
Typed MCP errors become `ResearchObservation(status="error")`; they do not crash the graph.

The executor preserves these fields whenever the server returns them:

```text
ticker, company, metric, value, unit, as_of, source,
evidence_id, document_id, section, period, trace_id
```

### 7. Evidence gate and report contract

The evidence gate requires:

- at least one successful metric observation with a source reference for each company;
- at least one document evidence hit whose returned source/evidence-ID pair exactly
  matches the observation provenance for each company;
- no factual report claim whose source cannot be traced to a successful observation; and
- explicit limitations for incompatible periods, currencies, and business definitions.

Each reported fact is a typed `CitedFact` containing non-blank claim text and
source references. Metric facts use source provenance without an evidence ID.
A document-backed fact contains exactly one source and its one evidence ID, which
prevents ambiguous cross-pairing. The pure
`validate_briefing_support()` boundary rejects sources, evidence IDs, and
exact source/evidence pairings absent from returned successful document hits. The final
`AnalystBriefing` separates:

```text
reported_facts: tuple[CitedFact, ...]
cross_company_observations
interpretation
limitations
source_references
```

The aggregate `source_references` tuple is the stable first-seen union of the
sources cited by `reported_facts`; extra, missing, duplicated, or reordered
aggregate references are invalid.

The report writer receives only the user mission and verified scratchpad. It does not
receive an unrestricted filesystem, raw MCP client, or write-capable tool.

## Maintained replanning failure lab

The deterministic classroom plan deliberately proposes a schema-valid but
domain-invalid metric request:

```text
get_company_metric(ticker="NVDA", metric="Revenue")
```

The server returns the typed `unsupported_metric` error and identifies `EPS` and `P/E`
as the metric tool's valid values. The replanner must:

1. keep every successfully executed step unchanged;
2. retain the failed attempt in the immutable trajectory;
3. supersede the failed step and replace only the unfinished tail;
4. assign new monotonic IDs to every replacement step;
5. replace the failed metric lookup with
   `search_financial_documents(company="NVIDIA", query="revenue growth", top_k=2)`;
6. keep the original research goal;
7. increment `replan_count` exactly once; and
8. continue without duplicating successful tool calls.

This differs from Lesson 09's metric-alias correction. Lesson 09 retries the same tool
with a corrected alias. Lesson 11 changes the research strategy from a structured metric
lookup to document evidence while preserving completed work.

If revised steps still fail, the graph stops with a typed bounded status. It does not
silently omit missing evidence or generate a complete-looking report.

## Provider strategy

### Offline maintained route

The default route uses recorded planner and replanner policies with the real local MCP
server and real LangGraph transitions. It is deterministic, requires no network or model,
and is explicitly labelled:

```text
offline fixture · deterministic planner and replanner · real local MCP execution
```

It verifies orchestration and evidence contracts, not live model quality.

### Ollama live route

With `FINAI_LIVE_MODE=1` and `FINAI_MODEL_PROVIDER=ollama`, the shared model gateway uses
the configured local model, defaulting to `qwen3:8b`. Planner, replanner, and report writer
use structured output where the role requires it. Python applies the same plan and policy
validation as the offline route.

### OpenAI live route

With `FINAI_LIVE_MODE=1`, `FINAI_MODEL_PROVIDER=openai`, and `OPENAI_API_KEY` configured,
the same notebook uses the shared OpenAI gateway, defaulting to `gpt-5-mini`. No provider-
specific planning or validation logic is allowed in learner cells.

Malformed model output is a typed provider failure. The instructor retries once, then
returns to the maintained offline route rather than debugging a provider during class.

## Notebook design

Canonical file: `notebooks/11_plan_and_execute_analyst.ipynb`

The source notebook is generated by `scripts/build_lesson11_notebook.py`, committed without
outputs, and contains 24 to 28 cells with stable `lesson11-###` identifiers.

Required sequence:

1. title, outcome, prerequisites, duration, and safety boundary;
2. capstone position and final observable product;
3. provider and live-mode setup;
4. ReAct versus plan-and-execute comparison;
5. Lesson 10 MCP discovery and permitted catalog;
6. Pydantic plan, replan, observation, and briefing contracts;
7. graph anatomy and host/model ownership;
8. maintained financial mission;
9. initial structured research plan;
10. plan policy validation;
11. deterministic step execution through the real MCP lifecycle;
12. evidence scratchpad growth;
13. maintained `unsupported_metric` planning failure;
14. tail-only plan replacement;
15. completion of the corrected plan;
16. evidence gate;
17. structured cited briefing;
18. optional live Ollama or OpenAI route;
19. trajectory and operational summary;
20. deterministic assertions and `LESSON_11_PASS`;
21. knowledge check with answers;
22. engineering challenge;
23. capstone increment; and
24. Lesson 12 handoff.

### Required notebook visuals

At least six code-generated figures are required:

1. **Autonomy pattern comparison**: workflow, ReAct, and plan-and-execute control flow.
2. **Plan dependency map**: step IDs, tool names, and dependencies.
3. **Graph state machine**: planner, policy gate, executor, replanner, evidence gate, report.
4. **Execution timeline**: step status, duration, and evidence count.
5. **Plan revision diff**: executed prefix, rejected step, and replacement tail.
6. **Evidence coverage matrix**: company by metric/document evidence before report generation.

Each visual must have a learner-readable title, axis or legend where relevant, and one
sentence explaining what decision it supports. Printed dictionaries alone do not satisfy
the visual contract.

## Slide design

Canonical file: `decks/11-plan-and-execute-analyst.pptx`

Target: nine concise slides using the established Lesson 08-10 template.

1. **One mission needs several coordinated evidence steps**: lesson outcome and finance case.
2. **Plan-and-execute separates strategy from action**: ReAct comparison diagram.
3. **The model proposes; the host controls**: ownership and trust boundary.
4. **A typed plan makes research inspectable**: plan schema and dependency example.
5. **Four roles share one bounded state**: planner, executor, replanner, report writer.
6. **MCP discovery becomes a permitted execution catalog**: Lesson 10 to Lesson 11 bridge.
7. **Replanning replaces only unfinished work**: before-and-after plan visual.
8. **The evidence gate prevents fluent incompleteness**: coverage matrix and stop rule.
9. **Lesson 12 will evaluate both path and answer**: trajectory and evaluation handoff.

Deck constraints:

- 16:9 widescreen;
- one primary claim per slide;
- visible learner copy in plain professional English;
- no visible em dash;
- exact footer `First Finance - Arnaud Demes`;
- citations and external sources in speaker notes;
- original diagrams rather than screenshots of MLExpert Academy; and
- no paragraph-heavy slide or generic closing slide.

## Instructor chapter

Canonical file: `chapters/11-plan-and-execute-analyst.md`

The chapter records:

- exact pre-class commands for offline, Ollama, and OpenAI runs;
- a 12-minute slide route, 40-minute notebook route, and 8-minute debrief;
- expected discovery, plan, error, replan, execution, and briefing outputs;
- the distinction between local action recovery and plan-tail revision;
- static recovery material for each diagram and plan table;
- no-network and no-model fallback instructions;
- skip-if-late guidance that preserves the plan revision and evidence gate;
- knowledge-check answers;
- challenge solution guidance; and
- the Lesson 12 evaluation handoff.

## Timing contract

### Slides: 12 minutes

| Time | Slides | Instructor outcome |
| --- | --- | --- |
| 0:00-2:00 | 1-2 | Establish why a multi-step mission differs from one-step ReAct. |
| 2:00-5:00 | 3-4 | Separate model proposal from host policy and inspect the typed plan. |
| 5:00-8:00 | 5-6 | Explain the graph and connect it to discovered MCP tools. |
| 8:00-10:00 | 7-8 | Show tail-only replanning and the evidence gate. |
| 10:00-12:00 | 9 | Define the answer-versus-trajectory handoff to Lesson 12. |

### Notebook: 40 minutes

| Time | Activity | Required observable evidence |
| --- | --- | --- |
| 0:00-5:00 | Setup, mission, and pattern comparison | Provider label and Figure 1 |
| 5:00-10:00 | Discover MCP tools and inspect contracts | Catalog table and ownership map |
| 10:00-16:00 | Create and validate the plan | Plan table, Figure 2, policy result |
| 16:00-23:00 | Execute initial steps | Scratchpad entries and Figures 3-4 |
| 23:00-29:00 | Replace the failed `Revenue` metric step | Typed error and Figure 5 |
| 29:00-34:00 | Complete the corrected plan | No duplicated successful calls |
| 34:00-37:00 | Run evidence gate and report writer | Figure 6 and cited briefing |
| 37:00-40:00 | Verify, challenge, and handoff | `LESSON_11_PASS` and Lesson 12 fields |

### Debrief: 8 minutes

Learners must answer:

1. Which parts were proposed by the model?
2. Which parts were controlled by Python and MCP policy?
3. What changed during replanning, and what stayed immutable?
4. Why can a factually plausible report still fail the evidence gate?
5. Which trajectory properties should Lesson 12 score?

## Failure and safety behavior

| Condition | Required behavior |
| --- | --- |
| Unknown or undiscovered capability | Block before execution and record `capability_not_permitted`. |
| Invalid arguments | Record typed tool or policy error; never coerce silently. |
| More than six steps | Reject the plan before the first tool call. |
| More than one replan | Stop with `replan_budget_exhausted`. |
| Revised tail duplicates successful work | Reject replacement and preserve the original scratchpad. |
| Missing company evidence | Stop at evidence gate; do not write a complete briefing. |
| Malformed live model output | Record provider failure, retry once, then use offline route. |
| MCP subprocess failure | Close lifecycle, show static recovery catalog, and avoid claims of live execution. |
| Any write-capable or trading tool | Exclude from the catalog and block by host policy. |

## Testing strategy

### Pure unit tests

`tests/test_plan_execute_analyst.py` covers:

- valid structured plans;
- duplicate, non-sequential, or invalid dependencies;
- monotonic replacement IDs and globally sequential attempt IDs;
- unknown capabilities and invalid arguments;
- maximum step and replan budgets;
- append-only observations;
- successful sequential execution;
- `unsupported_metric` strategy revision from metric tool to document search;
- no duplicate successful calls after replanning;
- evidence-gate success and failure;
- cited briefing inputs; and
- safe trajectory serialization.

Pure tests use a deterministic fake executor and require no subprocess, model, or network.

### MCP integration tests

The integration package uses the real local Lesson 10 server over `stdio` and verifies:

- runtime discovery of the two allowed tools;
- one lifecycle for a complete research run;
- metric and document observations with provenance;
- typed MCP error conversion;
- corrected execution after replanning; and
- clean shutdown after success or failure.

### Notebook asset tests

`tests/test_lesson11_assets.py` verifies:

- output-free source notebook;
- stable cell IDs and required teaching markers;
- at least six code-generated figures;
- offline restart-and-run-all;
- real local MCP use in the offline route;
- explicit Ollama and OpenAI configuration paths;
- one `LESSON_11_PASS` marker;
- chapter timing and fallback material;
- discoverable notebook, chapter, and deck indexes; and
- exact course manifest alignment.

### Deck tests and visual review

Automated checks require:

- exactly nine slides and nine speaker-note blocks;
- exact footer on every slide;
- expected capability and architecture terms;
- no visible em dash;
- no overflow; and
- successful template-plan and template-fidelity checks.

Every slide is rendered to PNG and inspected at full size before readiness is claimed.

## Readiness evidence

Canonical report: `docs/reviews/lesson-11-readiness.md`

Required evidence:

- focused unit and integration tests;
- full repository test suite;
- Ruff and Git whitespace checks;
- offline notebook execution and figure inspection;
- Ollama live run when the local model is configured;
- OpenAI live run only when `OPENAI_API_KEY` is configured;
- deck render, overflow, template, and visual inspection;
- timed learner rehearsal if performed; and
- explicit qualifications for any route that could not be executed.

No readiness report may infer a provider pass from static configuration alone.

## Capstone increment and Lesson 12 handoff

Lesson 11 adds a reusable research trajectory to the Financial Analyst Copilot:

```text
mission -> discovered catalog -> structured plan -> validated calls ->
observations -> revised plan -> evidence gate -> cited briefing
```

Lesson 12 consumes, without transformation:

- original mission;
- initial and final plan;
- capability names and arguments;
- observation status and error codes;
- tool-call order and count;
- replan count;
- evidence IDs and source references;
- final briefing sections; and
- latency per stage.

It will score trajectory correctness and efficiency separately from answer relevance,
completeness, grounding, and citation quality.

## Sources

- [MLExpert Academy Plan and Execute Agent](https://www.mlexpert.io/academy/v1/ai-agents/planning-agent)
- [MLExpert Academy Evaluating Agentic Systems](https://www.mlexpert.io/academy/v1/ai-agents/agent-evaluation)
- [Plan-and-Solve Prompting](https://arxiv.org/abs/2305.04091)
- [LangGraph documentation](https://docs.langchain.com/oss/python/langgraph/overview)
- [Model Context Protocol documentation](https://modelcontextprotocol.io/)
- [Official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Lesson 10 Financial MCP design](2026-08-21-lesson-10-financial-mcp-design.md)
- [Day 2 progression design](2026-08-21-day-two-agent-progression-and-lesson-08-design.md)
