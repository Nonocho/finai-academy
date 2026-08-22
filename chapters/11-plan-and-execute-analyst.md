# Lesson 11 - Plan-and-execute financial analyst

**First Finance - Arnaud Demes**
**Day 2 · 13:30-14:30 · 12-minute concept deck + 40-minute notebook + 8-minute verification and debrief**

## Instructor outcome

Students inspect a bounded research mission for NVIDIA (`NVDA`) and Schneider
Electric (`SU.PA`). They validate a proposed plan, execute discovered read-only
MCP tools through one lifecycle, retain a typed failure, replace only unfinished
work, and require evidence before a cited briefing can be written.

The Lesson 11 chapter and notebook are available for instructor-led testing. The
deck remains planned at `decks/11-plan-and-execute-analyst.pptx`; Task 7 creates
and certifies that file before it is linked as a delivery asset. This lesson does
not use live market data and does not provide investment advice.

```text
mission -> discovered catalog -> validated plan -> observations -> revised tail
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
| 6 | `search_financial_documents` | Schneider revenue query | Document evidence and source reference | Replacement success |

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

The nine-slide companion deck is planned, not yet a delivery file. When Task 7
has certified it, use this script. Until then, teach this sequence from the
static matrices and notebook figures without implying that a deck exists.

| Time | Slide | Instructor job |
|---:|---:|---|
| 0:00-2:00 | 1-2 | Contrast a fixed workflow and one-action ReAct loop with a coordinated research plan. State the bounded mission and read-only boundary. |
| 2:00-5:00 | 3-4 | Separate a model proposal from the host plan gate. Show that a valid schema is not sufficient business permission. |
| 5:00-8:00 | 5-6 | Walk through planner, policy gate, executor, replanner, evidence gate, and report writer. Connect execution to the discovered MCP catalog. |
| 8:00-10:00 | 7-8 | Surface the `Revenue` failure and show tail-only `replace_remaining`. Explain why the two successful metric calls stay immutable. |
| 10:00-12:00 | 9 | Explain that Lesson 12 scores answer quality and trajectory quality separately. |

The rows total 12 minutes. Do not read slides aloud. The lesson point is host
control over a multi-step research route, not a live provider demonstration.

## 40-minute notebook route

Use the checked-in stable cell IDs and expected visible outputs. The source
notebook has cells `lesson11-000` through `lesson11-027`; do not insert a second
server, re-run a successful call, or replace course evidence with live data.

| Time | Cells | Instructor action | Expected visible output |
|---:|---|---|---|
| 0:00-5:00 | `lesson11-000` to `lesson11-005` | State the mission, boundary, and why this task needs a plan. Run setup and compare workflow, ReAct, and plan-and-execute. | Offline provider label and Figure 1 control-pattern comparison. |
| 5:00-10:00 | `lesson11-006` to `lesson11-008` | Run the one-lifecycle mission, inspect the discovered catalog and four contracts, then show the initial plan. | Real MCP server name, permitted tools, contract table, and Figure 2 initial-plan dependencies. |
| 10:00-16:00 | `lesson11-009` to `lesson11-011` | Validate the proposed plan before a call. Identify model-owned proposal and host-owned policy. | Initial and final step IDs, plan-gate result, and Figure 3 six-node graph. |
| 16:00-23:00 | `lesson11-012` to `lesson11-014` | Expose the controlled third attempt. Read the `unsupported_metric` error rather than correcting it in place. | Typed error table and Figure 4 attempt timeline. |
| 23:00-29:00 | `lesson11-015` to `lesson11-017` | Request the recorded replan. Read `replace_remaining`; compare executed prefix, superseded tail, and replacements. | Replacement IDs 5 and 6, retained failure, and Figure 5 tail revision. |
| 29:00-34:00 | `lesson11-018` to `lesson11-020` | Confirm the completed sequence, one replan, and no duplicate successful calls. | Steps 1, 2, 3, 5, and 6 plus Figure 6 evidence coverage matrix. |
| 34:00-37:00 | `lesson11-021` to `lesson11-024` | Run the evidence gate. Only then inspect the cited briefing and optional live route configuration. | Passed coverage table, briefing field counts, and trajectory table. |
| 37:00-40:00 | `lesson11-025` to `lesson11-027` | Run verification, answer the knowledge check, and state the Lesson 12 handoff. | One `LESSON_11_PASS` marker and challenge constraints. |

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
| 4:00-6:00 | Ask why a plausible report can still fail. | Each company needs both metric and document evidence with sources. |
| 6:00-8:00 | Connect the trajectory to Lesson 12. | `LESSON_11_PASS`, cited briefing fields, and evaluation fields listed below. |

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

1. Run `lesson11-007` and name the two discovered read-only tools.
2. Run `lesson11-013` and retain the `unsupported_metric` failed step.
3. Run `lesson11-016` and `lesson11-017` to show `replace_remaining`, the
   immutable successful prefix, and replacement IDs 5 and 6.
4. Run `lesson11-020` through `lesson11-022` to show the evidence gate and the
   cited briefing boundary.
5. Run `lesson11-026` for `LESSON_11_PASS`, then state the Lesson 12 fields.

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
briefing sections; and latency per stage.

For the answer, Lesson 12 will score relevance, completeness, grounding, and
citation quality. For the trajectory, it will score policy correctness,
efficiency, bounded replanning, evidence-gate behavior, and trace completeness.

## Sources

- [Lesson 09 instructor chapter](09-self-correcting-agent.md)
- [Lesson 10 instructor chapter](10-financial-mcp.md)
- [Official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Lesson 11 design](../docs/superpowers/specs/2026-08-22-lesson-11-plan-and-execute-financial-analyst-design.md)
