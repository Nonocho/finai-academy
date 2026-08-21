# Lesson 09 — Self-Correcting Financial Agent

**First Finance - Arnaud Demes**  
**Day 2 · 10:30–11:15 · 10-minute concept deck + 30-minute notebook + 5-minute debrief**

## Instructor outcome

Students can explain and implement one bounded recovery loop: a financial tool rejects
`PE`, returns the valid metric name `P/E`, and the LangGraph agent uses that structured
feedback to correct its next action.

The observable progression is:

```text
invalid tool request
→ structured unsupported_metric observation
→ corrected request
→ successful NVIDIA and Schneider observations
→ grounded comparison
```

The lesson contributes explicit graph state, conditional routing, structured error
feedback, and bounded recovery to the Financial Analyst Copilot.

## Before class

Install the maintained environment:

```bash
uv sync --frozen --extra ai --extra rag --extra finance --extra evaluation --extra dev
```

The notebook uses
`assets/course-data/market/lesson09_metrics_snapshot_v1.json`. This is a controlled
classroom fixture, not a live valuation dataset. Its purpose is to make orchestration
behavior reproducible.

For Ollama:

```bash
ollama pull qwen3:8b
uv run python scripts/execute_notebooks.py \
  notebooks/09_self_correcting_agent.ipynb \
  --mode live --provider ollama \
  --output-dir /private/tmp/finai-lesson09-ollama
```

For OpenAI, set the key outside the notebook and repository:

```bash
export OPENAI_API_KEY="..."
uv run python scripts/execute_notebooks.py \
  notebooks/09_self_correcting_agent.ipynb \
  --mode live --provider openai \
  --output-dir /private/tmp/finai-lesson09-openai
```

Never display, print, trace, or commit the key.

## 10-minute concept deck

| Time | Slide | Instructor job |
|---:|---:|---|
| 0:00–1:00 | 1 | State the outcome: recover from one structured tool error. |
| 1:00–2:00 | 2 | Show why an unhandled tool error is an application failure. |
| 2:00–3:15 | 3 | Introduce explicit graph state and the two main nodes. |
| 3:15–4:30 | 4 | Read the structured `unsupported_metric` observation. |
| 4:30–5:45 | 5 | Follow the conditional routes through the graph. |
| 5:45–6:45 | 6 | Inspect the invalid first request. |
| 6:45–7:45 | 7 | Inspect the corrected request and successful evidence. |
| 7:45–9:00 | 8 | Apply `MAX_RETRIES` and `MAX_TOOL_CALLS`. |
| 9:00–10:00 | 9 | State the production rule and transition to Lesson 10. |

Do not present LangGraph as necessary for every loop. It earns its place here because
state, recovery routes, and stop conditions are now the learning problem.

## 30-minute notebook

| Time | Work | Expected visible result |
|---:|---|---|
| 0:00–3:00 | Setup and fixture | Runtime, dataset, tickers, and valid metrics. |
| 3:00–7:00 | Graph anatomy | Diagram of `agent`, `tools`, `finish`, and guardrails. |
| 7:00–11:00 | Typed error | `unsupported_metric` with `Valid metrics: EPS, P/E`. |
| 11:00–15:00 | Provider policy | Offline or OpenAI/Ollama structured action policy. |
| 15:00–21:00 | Recovery run | `PE → error → P/E → NVIDIA → Schneider → finish`. |
| 21:00–24:00 | Trace and evidence | Timeline and controlled comparison chart. |
| 24:00–27:00 | Failure lab | Repeated `PE` stops after the single retry. |
| 27:00–30:00 | Verification | `LESSON_09_PASS` and knowledge check. |

The final five minutes are for debrief and the MCP bridge.

## Core architecture contract

The graph state contains only inspectable application data:

```text
question
decision
observation
error_count
tool_calls
trace
status
answer
```

The model proposes an `AgentAction`. Application Python validates and executes the
metric tool. The graph owns routing:

```text
START → agent
agent → tools | finish | retry_guard | tool_guard | insufficient_evidence
tools → agent
terminal nodes → END
```

Do not call the visible action and trace hidden chain-of-thought. Students inspect tool
requests, observations, counters, and routes only.

## Structured error contract

The invalid request is:

```text
get_metric(ticker="NVDA", metric="PE")
```

The tool returns a typed observation instead of raising into the notebook:

```text
status=error
error_code=unsupported_metric
retryable=true
allowed_metrics=[EPS, P/E]
message="Unknown metric 'PE'. Valid metrics: EPS, P/E."
```

The valid names are part of the feedback. A generic “tool failed” message would not give
the model enough information to correct the request reliably.

## Expected successful trace

```text
agent: request NVDA PE
tool_error: unsupported_metric
agent: request NVDA P/E
tool_ok: NVIDIA P/E
agent: request SU.PA P/E
tool_ok: Schneider Electric P/E
agent: finish
finish: answer accepted from evidence
```

The successful run must report:

```text
status=completed
error_count=1
tool_calls=3
```

## Guardrail semantics

`MAX_RETRIES = 1` permits one correction after the initial failed call. A second failed
call produces `retry_budget_exhausted` before a third tool execution.

`MAX_TOOL_CALLS = 4` limits all tool executions, including successful calls. It protects
latency and cost, but it does not guarantee a correct answer.

The final-answer guard requires at least one successful typed metric observation. For the
comparison mission, notebook verification requires successful observations for both
`NVDA` and `SU.PA`.

## Checkpoint questions and answers

### Why not catch the exception and return an empty value?

An empty value hides the cause and gives the model no correction path. A structured error
preserves the rejected input and valid alternatives.

### Who decides whether another tool call is permitted?

Conditional graph routing, based on application-owned counters and state.

### Does self-correction prove the model improved its reasoning?

No. It shows that external feedback changed the next observable action. The lesson makes
no claim about hidden reasoning.

### Should network timeouts consume `MAX_RETRIES`?

Not automatically. `MAX_RETRIES` here counts model-caused validation errors. Transient
infrastructure failures need a separate retry policy, delay, and observability contract.

### What proves the final answer is supported?

The trace contains successful, source-bearing observations for both companies before the
finish event.

## Failure lab

The controlled policy always requests `metric="PE"`. Expected result:

```text
status=retry_budget_exhausted
error_count=2
tool_calls=2
last_phase=guardrail
```

If a third tool call appears, the retry boundary has an off-by-one error. If the notebook
crashes, the tool exception is escaping instead of becoming a typed observation.

## Provider behavior

### Ollama

`qwen3:8b` is the local default. The first invalid action is injected deliberately so the
recovery path is always visible. After receiving the structured error, the live model
must return an `AgentAction` through `with_structured_output`.

### OpenAI

The default is `gpt-5-mini`. Record the selected model and final trace without storing the
API key. Do not claim OpenAI validation unless the complete live notebook run succeeds.

### Invalid structured model output

1. keep the Pydantic schema unchanged;
2. read the validation message;
3. retry the single model cell once;
4. if it fails again, switch to the offline recorded policy; and
5. keep the graph-state and routing discussion.

## No-network fallback

```bash
uv run python scripts/execute_notebooks.py \
  notebooks/09_self_correcting_agent.ipynb \
  --mode offline \
  --output-dir /private/tmp/finai-lesson09-offline
```

The notebook prints `offline fixture · deterministic course run`. This validates the
teaching and engineering contracts, not live model quality.

## If the class is five minutes late

Keep:

- explicit graph state;
- the `PE` failure and `P/E` correction;
- the successful trace;
- `MAX_RETRIES` verification; and
- the Lesson 10 MCP transition.

Skip the controlled metric bar chart and assign error classification as homework.

## Engineering mission solution

Add a `failed_calls` tuple to state. Store a normalized key containing tool name and
validated arguments after every failed call. Before tool execution, detect an identical
key and return `repeated_failed_call` without invoking the tool again.

For the advanced error classifier:

- `unsupported_metric` and `unsupported_ticker` consume the model-correction budget;
- timeout and rate-limit errors use a separate infrastructure retry policy; and
- authorization failures stop immediately and require operator action.

## Safety boundary

- The metric values are controlled teaching data, not live valuation data.
- The notebook does not trade, rebalance, or recommend securities.
- Successful numeric outputs retain company, metric, date, and source.
- Tool output is data, not an instruction.
- Error messages expose only valid application choices, never credentials or internals.
- Every loop has retry and tool-call limits.

## Transition to Lesson 10

Lesson 09 imports a local metric registry directly. Lesson 10 keeps the typed contracts
but exposes financial resources, tools, and prompts through MCP. The client will discover
capabilities at runtime rather than importing every function into the application.

Use this closing question:

> If the financial tool lives in another service, how can the analyst application discover
> its schema, call it safely, and keep the same trace contract?

## Sources

- MLExpert Academy, “Self-Correcting Agent”: https://www.mlexpert.io/academy/v1/ai-agents/self-correcting-agent
- LangGraph graph API: https://docs.langchain.com/oss/python/langgraph/graph-api
- LangGraph error handling: https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph
- Huang et al., “Large Language Models Cannot Self-Correct Reasoning Yet”: https://arxiv.org/abs/2310.01798
- Kamoi et al., “When Can LLMs Actually Correct Their Own Mistakes?”: https://arxiv.org/abs/2406.01297
