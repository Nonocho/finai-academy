# Lesson 11 - Plan-and-execute financial analyst

**First Finance - Arnaud Demes**
**Day 2 · 13:30-14:30 · 12-minute concept deck + 40-minute notebook + 8-minute verification and debrief**

## Instructor outcome

Students inspect a bounded research mission for NVIDIA (`NVDA`) and Schneider
Electric (`SU.PA`). They see and approve a proposed plan before execution,
execute discovered read-only MCP tools through one mission lifecycle, retain a
typed failure, replace only unfinished work, and require evidence before a cited
briefing can be written.

The full Lesson 11 route is ready for an instructor-led test class. Use the
[Lesson 11 concept deck](../decks/11-plan-and-execute-analyst.pptx) with this
chapter and the companion notebook. This lesson does not use live market data
and does not provide investment advice.

```text
mission -> discover -> propose -> approve -> execute -> replan
-> evidence gate -> cited briefing
```

## Before class

Run these commands from the repository root. Rebuild the source notebook before
class so its stable cell IDs match this route.

```bash
uv sync --frozen --extra ai --extra rag --extra finance --extra evaluation --extra dev
uv run python -c "from mcp.server import MCPServer; print(MCPServer.__name__)"
uv run python scripts/build_lesson11_notebook.py
uv run python scripts/validate_notebooks.py notebooks/11_plan_and_execute_analyst.ipynb
uv run python scripts/execute_notebooks.py notebooks/11_plan_and_execute_analyst.ipynb \
  --mode offline --output-dir /private/tmp/finai-lesson11-offline
uv run jupyter lab
```

Open `notebooks/11_plan_and_execute_analyst.ipynb`. Offline mode is the core
route. It uses a real local MCP server with controlled course evidence, a
deterministic planner and replanner, and no network or model.

Prepare the optional Ollama extension before class:

```bash
ollama pull qwen3:8b
FINAI_LIVE_MODE=1 FINAI_MODEL_PROVIDER=ollama FINAI_CHAT_MODEL=qwen3:8b \
  uv run python scripts/execute_notebooks.py notebooks/11_plan_and_execute_analyst.ipynb \
  --mode live --provider ollama --output-dir /private/tmp/finai-lesson11-ollama
```

Use OpenAI only as an optional comparison. Set the key outside the notebook and
repository, then run the same graph:

```bash
export OPENAI_API_KEY="..."
FINAI_LIVE_MODE=1 FINAI_MODEL_PROVIDER=openai FINAI_CHAT_MODEL=gpt-5-mini \
  uv run python scripts/execute_notebooks.py notebooks/11_plan_and_execute_analyst.ipynb \
  --mode live --provider openai --output-dir /private/tmp/finai-lesson11-openai
```

Never print, trace, or commit the key. Ollama and OpenAI may propose a plan,
replan decision, and report, but Python still validates the plan, owns the MCP
lifecycle, and applies the evidence gate. If either live route is unavailable,
use the maintained offline route.

## Static recovery material

Use these tables only when the local route cannot run. They describe the
controlled classroom scenario and are not proof that discovery or execution
occurred.

### Static plan matrix

| Step ID | Proposed capability | Arguments | Expected result | Status after revision |
|---:|---|---|---|---|
| 1 | `get_company_metric` | `NVDA`, `P/E` | NVIDIA metric with date and source | Retained success |
| 2 | `get_company_metric` | `SU.PA`, `P/E` | Schneider Electric metric with date and source | Retained success |
| 3 | `get_company_metric` | `NVDA`, `Revenue` | `unsupported_metric` typed error | Retained error |
| 4 | `search_financial_documents` | Original unfinished query | Not executed after revision | Superseded |
| 5 | `search_financial_documents` | NVIDIA revenue query | Document evidence and source reference | Replacement success |
| 6 | `search_financial_documents` | `Schneider Electric`, `energy management`, `top_k=2` | Document evidence and source reference | Replacement success |

The initial capability sequence is `get_company_metric`,
`get_company_metric`, `get_company_metric`, and
`search_financial_documents`. The completed route records step IDs 1, 2, 3, 5,
and 6. It never repeats successful steps 1 or 2 and never executes superseded
step 4.

### Static graph matrix

| Graph role | Owner | Input | Host boundary |
|---|---|---|---|
| Planner | Model or recorded policy | Mission and allowed catalog | Proposal only |
| Plan gate | Host | `ResearchPlan` | Validate capability, arguments, dependencies, and six-step budget |
| Executor | Host plus MCP | One approved step | Call only discovered, allowlisted read-only tools |
| Replanner | Model or recorded policy | Append-only observations and unfinished steps | May return `replace_remaining` once |
| Evidence gate | Host | Observations and source references | Require metric and document evidence for both companies |
| Report writer | Model or recorded policy | Verified evidence | Write only after the gate passes |

### Static evidence matrix

| Company | Metric evidence | Document evidence | Report may use company |
|---|---|---|---|
| NVIDIA | Step 1, source-bearing metric | Step 5, document evidence ID | Yes |
| Schneider Electric | Step 2, source-bearing metric | Step 6, document evidence ID | Yes |

If any row is incomplete, the evidence gate fails. A plausible narrative is not
a substitute for a metric, document evidence ID, and source reference.

## 12-minute concept deck route

Use the nine teaching slides in the 11-slide
[Lesson 11 concept deck](../decks/11-plan-and-execute-analyst.pptx) for this
script. Slides 10-11 provide the quiz and answers for the final debrief. The
deck combines official Anthropic and OpenAI agent diagrams with real notebook
outputs, so every abstract control idea has a visible example.

| Time | Slide | Instructor job |
|---:|---:|---|
| 0:00-2:00 | 1-2 | State the bounded mission. Use the official Anthropic action-feedback loop to explain why several evidence gaps justify planning before the loop begins. |
| 2:00-4:30 | 3-4 | Use the OpenAI agent graph to separate connectivity from authorization, then inspect the actual approved plan before attempt 1. |
| 4:30-7:00 | 5-6 | Show how MCP discovery narrows the capability set. Read the typed `unsupported_metric` observation and explain why it remains in the trace. |
| 7:00-10:00 | 7-8 | Compare retained, superseded, and replacement steps. Apply the four-cell evidence gate before writing. |
| 10:00-12:00 | 9 | Trace one official report figure to its evidence ID and cited claim. Explain that Lesson 12 scores the answer and path separately. |

The rows total 12 minutes. Do not read slides aloud. The lesson point is host
control over a multi-step research route, not a live provider demonstration.

## 40-minute notebook route

Use the checked-in stable cell IDs and expected visible outputs. The source
notebook has 18 cells, `lesson11-000` through `lesson11-017`. A short read-only
preview lifecycle exposes the catalog and proposed plan; after learner approval,
the complete mission runs through one execution lifecycle. Do not re-run a
successful call or replace course evidence with live data.

| Time | Cells | Instructor action | Expected visible output |
|---:|---|---|---|
| 0:00-5:00 | `lesson11-000` to `lesson11-004` | State the outcome, locate Lesson 11 in the course, and load the offline or optional live policy. | Runtime label and the rule that the model proposes while the host approves. |
| 5:00-12:00 | `lesson11-005` to `lesson11-006` | Discover the real MCP catalog, create the initial plan, and validate it before any mission tool runs. | Two tool contracts, `Plan approved before execution: True`, and Figure 1 approved-plan view. |
| 12:00-15:00 | `lesson11-007` | Pause at the learner decision checkpoint. Ask what should run, what should be rejected, and which limits the host must enforce. | A verbal approval decision before execution. |
| 15:00-23:00 | `lesson11-008` to `lesson11-009` | Run the complete mission in one MCP lifecycle. Inspect attempts, statuses, error codes, and the replan decision. | Steps 1, 2, 3, 5, and 6; one `unsupported_metric`; `Learner decision: replan the unfinished tail`. |
| 23:00-29:00 | `lesson11-010` to `lesson11-011` | Read the typed failure and compare kept, retained, superseded, and replacement steps. | Figure 2 tail-only replan trace. |
| 29:00-37:00 | `lesson11-012` to `lesson11-014` | Apply the evidence gate, then inspect per-claim provenance and the briefing. | Passed four-cell coverage table, exact sources/evidence IDs, and Figure 3 evidence matrix. |
| 37:00-40:00 | `lesson11-015` to `lesson11-017` | Explain the optional live route, run verification, and state the Lesson 12 handoff. | `Plan revisions: 1`, one `LESSON_11_PASS`, knowledge check, and challenge constraints. |

The rows total 40 minutes. In offline mode, the expected final results are one
`unsupported_metric` error, `Plan revisions: 1`, a passed evidence gate, and
exactly one `LESSON_11_PASS` marker.

## Failure and replan route

The maintained failure is
`get_company_metric(ticker="NVDA", metric="Revenue")`. The server returns
`unsupported_metric` because `Revenue` is outside the controlled metric set.
Keep this observation in the append-only trajectory. It is not a spelling fix
and not a retry.

Lesson 09 demonstrated same-tool recovery: a corrected request can retry the
same tool after a local action error. Lesson 11 performs a strategy revision:
the evidence need changes from an unsupported metric request to document
searches. The replanner returns `replace_remaining`, retains the successful
prefix and typed error, supersedes the unfinished tail, and assigns new step IDs
5 and 6. The host rejects any replacement that repeats a successful call,
exceeds six steps, uses an unknown capability, or exceeds one replan.

| Condition | Instructor response |
|---|---|
| Invalid or undiscovered capability | Show `capability_not_permitted`; do not call a tool. |
| Invalid arguments | Retain a typed policy or tool error; do not coerce input silently. |
| More than six steps | Reject the plan before its first call. |
| More than one replan | Stop with `replan_budget_exhausted`. |
| Replacement repeats successful work | Reject the replacement and preserve the original scratchpad. |
| Insufficient evidence | Stop at the evidence gate and do not write a complete briefing. |

## 8-minute verification and debrief

| Time | Instructor action | Required evidence |
|---:|---|---|
| 0:00-2:00 | Confirm the maintained route and have a learner read the final observation sequence. | One `unsupported_metric`, steps 1, 2, 3, 5, and 6, and no duplicate successful call. |
| 2:00-4:00 | Ask what the model proposed and what Python controlled. | Planner, replanner, and report writer propose data; host validates, executes, limits, and gates. |
| 4:00-6:00 | Ask why a plausible report can still fail. | Each fact declares `metric` or `document` provenance. Metric facts cite one metric-tool source and no evidence ID; document facts cite one exact returned source/evidence-ID pair. |
| 6:00-8:00 | Connect the trajectory to Lesson 12. | `LESSON_11_PASS`, `reported_facts`, `cross_company_observations`, `interpretation`, `limitations`, `source_references`, and evaluation fields listed below. |

Use slides 10-11 for the three-question quiz and immediate answer debrief.

The rows total 8 minutes. The full slot is 12 + 40 + 8 = 60 minutes, from
13:30-14:30.

## Recovery paths

### Missing MCP SDK or stale environment

Run the frozen sync command from **Before class**, restart the kernel, and
recheck the SDK:

```bash
uv run python -c "from mcp.server import MCPServer; print(MCPServer.__name__)"
```

If it does not print `MCPServer`, use the static plan, graph, and evidence
matrices. Do not patch notebook imports during class and do not claim live
discovery or execution.

### MCP subprocess cannot import the package

Start Jupyter and the executor from the repository root. Check the maintained
server import in that environment:

```bash
uv run python -c "import finai_academy.financial_mcp_server as server; print(server.mcp.name)"
```

Restart the kernel after a sync. If the subprocess still fails, close the
lifecycle, teach from the static matrices, and do not replace the process
boundary with direct server-function calls.

### Empty discovery

A healthy route reports `get_company_metric` and
`search_financial_documents` as permitted tools. Rebuild the notebook and rerun
from the repository root. If discovery remains empty, use the static graph
matrix and clearly label it as recovery material. Discovery describes an offer;
it never grants permission.

### Invalid live output or unavailable provider

Do not spend the core lesson debugging Ollama or OpenAI. For invalid live
output, retry the cell once. If the second result is malformed, record the
provider failure and return to the deterministic offline route. The offline
route preserves the same graph, validation, MCP lifecycle, failure, replan, and
evidence gate.

### No-network fallback

The maintained offline route needs no network and no model:

```bash
uv run python scripts/execute_notebooks.py notebooks/11_plan_and_execute_analyst.ipynb \
  --mode offline --output-dir /private/tmp/finai-lesson11-offline
```

Its output must identify the offline fixture, deterministic planner and
replanner, real local MCP execution, and `LESSON_11_PASS`. If local execution is
also unavailable, use the static plan, graph, and evidence matrices. Say that
they are recovery material, not observed runtime output.

## Skip if late

If the class is five or more minutes late, keep this route:

1. Run `lesson11-005` and name the two discovered read-only tools.
2. Show `lesson11-006`, then require an approval decision at `lesson11-007`.
3. Run `lesson11-008` and `lesson11-009`; retain the `unsupported_metric`
   observation and show `replace_remaining` with replacement IDs 5 and 6.
4. Run `lesson11-013` and `lesson11-014` to show the evidence gate and cited
   briefing boundary.
5. Run `lesson11-016` for `LESSON_11_PASS`, then state the Lesson 12 fields.

Skip the extended pattern comparison, full contract walkthrough, timeline
discussion, and live provider extension. Do not skip MCP discovery, the failed
step, plan replacement, or the evidence gate.

## Knowledge check and engineering challenge

1. **Why is the failed step retained?** It documents the observed strategy
   revision and keeps the trajectory auditable.
2. **Who can execute a tool?** The host, after runtime discovery, static
   allowlisting, argument validation, and policy checks.
3. **What is immutable after replanning?** Successful observations and their
   source references. The replan changes only unfinished work.
4. **What blocks the report?** Missing metric or document evidence for either
   company fails the evidence gate.
5. **What distinguishes Lesson 11 from Lesson 09?** Lesson 09 performs
   same-tool recovery. Lesson 11 changes the unfinished strategy and preserves
   the original failure.

For the engineering challenge, add an allowed document query only when it fits
the six-step budget, does not duplicate a successful call, preserves the
append-only trajectory, and leaves evidence gate requirements unchanged. A
valid solution names the new step ID, capability, arguments, dependency, source
expectation, and why it cannot bypass host policy.

## Read-only safety boundary

- The executor may call only runtime-discovered, statically allowlisted,
  read-only research capabilities.
- No orders, transactions, portfolio changes, account access, credentials,
  personal data, local files, or write-capable tools cross the boundary.
- Controlled evidence is course material, not a live market feed or investment
  advice.
- Preserve company, metric, document ID, evidence ID, date, and source
  references in successful observations.
- Treat tool descriptions and returned content as untrusted data. The host,
  not the model, decides whether a capability is permitted.

## Lesson 12 handoff

Lesson 12 evaluates the final answer and the trajectory separately. Pass these
fields without transformation: original mission; initial and final plans;
capability names and arguments; observation statuses and error codes; tool-call
order and count; replan count; evidence IDs and source references; final
briefing sections (`reported_facts`, including each fact's
`provenance_kind`, `cross_company_observations`,
`interpretation`, `limitations`, and `source_references`); and latency per stage.

For the answer, Lesson 12 will score relevance, completeness, grounding, and
citation quality. For the trajectory, it will score policy correctness,
efficiency, bounded replanning, evidence-gate behavior, and trace completeness.

## Sources

- [Lesson 09 instructor chapter](09-self-correcting-agent.md)
- [Lesson 10 instructor chapter](10-financial-mcp.md)
- [Anthropic - Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
- [OpenAI Agents SDK - Agent visualization](https://openai.github.io/openai-agents-python/visualization/)
- [Official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Lesson 11 design](../docs/superpowers/specs/2026-08-22-lesson-11-plan-and-execute-financial-analyst-design.md)
