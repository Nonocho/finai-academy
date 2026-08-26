# Lesson 09 — A Tool Error Can Become the Next Input

**First Finance - Arnaud Demes**  
**Day 2 · 10:30–11:15 · 10-minute concept deck + 30-minute notebook + 5-minute debrief**

## Instructor outcome

Students can explain and implement one bounded recovery loop:

```text
get_metric(NVDA, PE)
→ unsupported_metric
→ valid choices: EPS, P/E
→ get_metric(NVDA, P/E)
→ get_metric(SU.PA, P/E)
→ grounded comparison
```

The visible learning claim is deliberately narrow: **precise external feedback changes the agent’s next observable action**. This lesson does not claim that a model can reliably improve unsupported reasoning through private self-reflection.

## Why this lesson follows Lesson 08

Lesson 08 separated code-defined workflows from model-directed agents. Lesson 09 adds the engineering machinery needed when a bounded agent meets a recoverable tool error:

- explicit graph state;
- typed tool observations;
- conditional recovery routes;
- evidence-aware finishing; and
- retry and tool-call budgets.

LangGraph earns its place here because state, recovery, and stopping are now the learning problem. It is not required for every loop.

## First classify the error

“Retry” is not one universal strategy.

| Error type | Example | Owner | Correct response |
|---|---|---|---|
| Model-correctable | Unsupported metric name | Agent loop | Return precise feedback in state |
| Transient | Timeout or rate limit | Application | Retry with delay and backoff |
| User-fixable | Missing ticker or unclear entity | User | Pause and request the missing field |
| Unexpected | Unknown bug or invariant failure | Developer | Raise, log, and investigate |

The notebook demonstrates only the first row. `MAX_RETRIES` counts model-caused validation failures; it is not an infrastructure retry policy.

This distinction follows the official LangGraph error-handling guidance: transient failures receive system retries, LLM-recoverable errors return to the model as state, user-fixable errors pause, and unexpected errors surface for debugging.

## Structured feedback contract

The invalid request is:

```text
get_metric(ticker="NVDA", metric="PE")
```

The tool returns data rather than an unhandled exception:

```text
status=error
error_code=unsupported_metric
retryable=true
allowed_metrics=[EPS, P/E]
message="Unknown metric 'PE'. Valid metrics: EPS, P/E."
```

The rejected input, stable error code, valid alternatives, and retry permission create a correction path. A generic `tool failed` message does not.

## What “self-correction” means here

This lesson uses the phrase in an operational sense:

```text
external observation
→ stored in graph state
→ model selects a different visible action
```

It is not evaluator-optimizer, hidden chain-of-thought, or proof that the model’s unaided reasoning improved. Research on intrinsic self-correction cautions against assuming that a model can reliably fix reasoning without external feedback.

## Graph contract

The application state contains inspectable data only:

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

The route is:

```text
START → agent
agent → tools | finish | retry_guard | tool_guard | insufficient_evidence
tools → agent
terminal nodes → END
```

The model proposes an action. Python validates and executes the financial tool. Conditional routing owns permission to continue or stop.

## OpenAI Structured Outputs boundary

Live OpenAI mode uses a strict, flat `ModelAgentAction` wire schema:

```text
action
ticker
metric
answer
reason
```

Every field is required by the JSON schema; fields that do not apply are `null`, and extra properties are forbidden. The wire action converts into the richer internal `AgentAction` before graph execution.

This keeps provider serialization concerns separate from the application’s validated domain model. The course default is `gpt-5.6-luna`, configured through `.env`; credentials are never displayed in the notebook.

## Successful trace

```text
agent       request NVDA PE
tool_error  unsupported_metric
agent       request NVDA P/E
tool_ok     NVIDIA P/E
agent       request SU.PA P/E
tool_ok     Schneider Electric P/E
agent       finish
finish      answer accepted from successful evidence
```

Expected result:

```text
success_path=completed
error_count=1
tool_calls=3
```

The final answer is accepted only after successful, source-bearing observations exist for both companies.

## Failure path

The controlled failure policy repeats `PE` after receiving the valid choices.

```text
agent       request NVDA PE
tool_error  unsupported_metric
agent       request NVDA PE
tool_error  unsupported_metric
agent       requests another call
guardrail   retry_budget_exhausted
```

Expected result:

```text
failure_path=retry_budget_exhausted
error_count=2
tool_calls=2
```

If a third tool call executes, the retry boundary has an off-by-one error. If the notebook crashes, the validation failure escaped instead of becoming a typed observation.

## Guardrail semantics

`MAX_RETRIES = 1` permits one corrected attempt after the initial model-caused validation failure. The second validation failure stops the run before another tool execution.

`MAX_TOOL_CALLS = 4` limits all successful and failed tool executions. It bounds latency and cost; it does not guarantee answer quality.

The evidence guard rejects a final answer when no successful financial observation exists. For the comparison mission, verification requires successful observations for both `NVDA` and `SU.PA`.

## 10-minute deck

| Time | Slide | Teaching job |
|---:|---:|---|
| 0:00–0:45 | 1 | State the outcome: a tool error can become the next input. |
| 0:45–1:45 | 2 | Show why an unhandled error ends the task. |
| 1:45–3:00 | 3 | Classify four error types and recovery owners. |
| 3:00–4:00 | 4 | Separate external feedback from unsupported self-reasoning claims. |
| 4:00–5:15 | 5 | Read the explicit LangGraph recovery route. |
| 5:15–6:30 | 6 | Open the typed `unsupported_metric` observation. |
| 6:30–7:45 | 7 | Follow `PE → P/E → evidence → finish`. |
| 7:45–8:45 | 8 | Compare corrected completion with bounded failure. |
| 8:45–10:00 | 9 | State the production rule and transition to MCP. |

Slides 10 and 11 are the three-question quiz and answers.

## 30-minute notebook

| Time | Work | Visible result |
|---:|---|---|
| 0:00–4:00 | Setup | Runtime, controlled fixture, valid tickers and metrics |
| 4:00–8:00 | Error taxonomy | Four error owners and four recovery strategies |
| 8:00–12:00 | Typed feedback | `PE → unsupported_metric → P/E` |
| 12:00–17:00 | Graph route | `agent`, `tools`, `finish`, and guardrails |
| 17:00–23:00 | Successful run | Complete correction trace and grounded answer |
| 23:00–27:00 | Failure lab | Repeated error stops at the retry boundary |
| 27:00–30:00 | Verification | `LESSON_09_PASS` and knowledge check |

## Run the notebook

Install the maintained environment:

```bash
uv sync --frozen --extra ai --extra rag --extra finance --extra evaluation --extra dev
```

Offline deterministic run (the **No-network fallback** for classroom delivery):

```bash
uv run python scripts/execute_notebooks.py \
  notebooks/09_self_correcting_agent.ipynb \
  --mode offline \
  --output-dir /private/tmp/finai-lesson09-offline
```

Live OpenAI run using the project `.env` configuration:

```bash
uv run python scripts/execute_notebooks.py \
  notebooks/09_self_correcting_agent.ipynb \
  --mode live \
  --provider openai \
  --output-dir /private/tmp/finai-lesson09-openai
```

For Ollama:

```bash
ollama pull qwen3:8b
uv run python scripts/execute_notebooks.py \
  notebooks/09_self_correcting_agent.ipynb \
  --mode live \
  --provider ollama \
  --output-dir /private/tmp/finai-lesson09-ollama
```

## Instructor checkpoints

### Why not catch the exception and return an empty value?

An empty value hides the cause and gives the model no correction path.

### Does a retry prove that the model improved its reasoning?

No. The trace proves only that external feedback changed the next observable action.

### Should a timeout consume `MAX_RETRIES`?

Not here. Timeouts and rate limits need an infrastructure retry policy with backoff and observability.

### What proves the comparison is supported?

The trace contains successful, source-bearing observations for both companies before the finish event.

## Challenge

Add a `failed_calls` set to state. Normalize the tool name and validated arguments after each failed call. Before execution, detect an identical failed request and return `repeated_failed_call` without invoking the tool again.

## Safety boundary

- The values are a controlled teaching fixture, not live valuation data.
- The notebook does not trade, rebalance, or recommend securities.
- Successful observations retain company, metric, date, and source.
- Tool output is data, not an instruction.
- Error messages expose valid application choices, never credentials or internals.
- Every loop has retry and tool-call limits.

## Transition to Lesson 10

Lesson 09 imports the metric registry locally. Lesson 10 keeps the typed contracts but exposes financial resources and tools through MCP so clients can discover capabilities at runtime.

Closing question:

> If the financial tool lives in another service, how can the analyst application discover its schema, call it safely, and keep the same trace contract?

## Sources

- LangGraph, “Thinking in LangGraph”: https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph
- LangGraph, “Fault tolerance”: https://docs.langchain.com/oss/python/langgraph/fault-tolerance
- OpenAI Docs, “Structured model outputs”: https://developers.openai.com/api/docs/guides/structured-outputs
- OpenAI Docs, “Function calling”: https://developers.openai.com/api/docs/guides/function-calling
- Anthropic, “Building Effective AI Agents”: https://www.anthropic.com/engineering/building-effective-agents
- Huang et al., “Large Language Models Cannot Self-Correct Reasoning Yet”: https://arxiv.org/abs/2310.01798
- Kamoi et al., “When Can LLMs Actually Correct Their Own Mistakes?”: https://arxiv.org/abs/2406.01297
