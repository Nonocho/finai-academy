# Lesson 08 — Workflows Versus Agents

**First Finance - Arnaud Demes**  
**Day 2 · 09:30–10:15 · 10-minute concept deck + 30-minute notebook + 5-minute debrief**

## Instructor outcome

Students can choose the lowest useful autonomy for an analyst task. They do not leave
with the claim that agents are generally superior to workflows.

The observable progression is:

```text
direct question → fixed workflow succeeds
dependent question → unsupported_dependency
same dependent question → bounded agent observes, acts again, then stops
```

The lesson contributes typed tools, one normalized trajectory format, and a minimal
bounded agent loop to the Financial Analyst Copilot.

## Before class

Install the maintained environment:

```bash
uv sync --frozen --extra ai --extra rag --extra finance --extra evaluation --extra dev
```

The notebook reads the checked-in snapshot
`assets/course-data/market/lesson08_market_snapshot_v1.json`. It does not fetch market
data during class. The snapshot contains a retrieval date, source URL, and manifest hash.

For Ollama:

```bash
ollama pull qwen3:8b
uv run python scripts/execute_notebooks.py \
  notebooks/08_workflows_vs_agents.ipynb \
  --mode live --provider ollama \
  --output-dir /private/tmp/finai-lesson08-ollama
```

For OpenAI, set the key outside the notebook and repository:

```bash
export OPENAI_API_KEY="..."
uv run python scripts/execute_notebooks.py \
  notebooks/08_workflows_vs_agents.ipynb \
  --mode live --provider openai \
  --output-dir /private/tmp/finai-lesson08-openai
```

Never display, print, trace, or commit the key.

## 10-minute concept deck

| Time | Slide | Instructor job |
|---:|---:|---|
| 0:00–1:00 | 1 | State the decision: use the lowest useful autonomy. |
| 1:00–2:15 | 2 | Move across function, workflow, bounded agent, and multi-agent. |
| 2:15–3:15 | 3 | Compare determinism, flexibility, latency, and failure surface. |
| 3:15–4:15 | 4 | Establish typed tools as the shared application boundary. |
| 4:15–5:30 | 5 | Walk through the one-pass workflow. |
| 5:30–6:45 | 6 | Expose the unseen conversion amount; do not fabricate it. |
| 6:45–8:00 | 7 | Add the reason–act–observe–stop loop and `MAX_STEPS`. |
| 8:00–9:00 | 8 | Compare the two recorded trajectories. |
| 9:00–10:00 | 9 | Apply the decision rule and transition to Lesson 09. |

Do not explain LangGraph in this deck. The plain-Python loop makes the mechanics visible;
LangGraph arrives when state and recovery routing become the learning problem.

## 30-minute notebook

| Time | Work | Expected visible result |
|---:|---|---|
| 0:00–3:00 | Setup and snapshot | Runtime label, dataset ID, two tool names. |
| 3:00–6:00 | Autonomy spectrum | Figure 1 and architecture decision language. |
| 6:00–9:00 | Typed tools | NVIDIA, Schneider Electric, and USD/EUR observation table. |
| 9:00–13:00 | Direct workflow | `workflow_direct_status=completed`. |
| 13:00–17:00 | Dependency failure | `workflow_dependency_status=unsupported_dependency`, Figures 2–3. |
| 17:00–22:00 | Bounded agent | Figures 4–5 and ordered two-tool trajectory. |
| 22:00–25:00 | Budget failure | `Stopped after MAX_STEPS=2`, Figure 6. |
| 25:00–28:00 | Verification | Grounding, order, budget, metadata assertions. |
| 28:00–30:00 | Knowledge check | Students state the architecture decision rule. |

The remaining five minutes belong to debrief and the Lesson 09 bridge. Do not consume
them by typing boilerplate live.

## Core architecture contract

Both systems use the same deterministic registry:

```text
get_market_price(ticker)
convert_currency(amount, from_currency, to_currency)
```

The model may produce a typed request. Python validates and executes it. A model sentence
never becomes a market observation.

The one-pass workflow selects one route before it has any observation:

```text
question → plan once → zero or one tool → final answer
```

The bounded agent can select another action from visible results:

```text
question + trajectory → typed action → tool observation → updated trajectory
                                      ↘ finish or MAX_STEPS stop
```

Do not call structured next-action output “chain-of-thought.” Students inspect actions,
arguments, observations, and stop conditions; hidden model reasoning is neither requested
nor required.

## Exact classroom trajectories

### Direct workflow

Question:

> What is NVIDIA's latest available share price?

Expected phases:

```text
plan → get_market_price(NVDA) → finish
```

Expected outcome: `completed`, with price, USD, observation date, and source URL.

### Unsupported dependency

Question:

> What is NVIDIA's latest available share price converted to euros?

Expected workflow outcome:

```text
unsupported_dependency
```

The amount passed to `convert_currency` does not exist until the price observation is
available. The workflow must not fabricate it.

State this qualification clearly:

> A developer can add a deterministic two-step conversion branch. The limitation is not
> that workflows cannot chain operations; it is that every dependency shape needs a
> predefined route.

### Bounded agent

Expected tool order:

```text
get_market_price(NVDA)
→ convert_currency(observed_price, observed_currency, EUR)
→ finish
```

The verification cell recomputes the dependency: the conversion input must equal the
recorded price observation.

## Checkpoint questions and answers

### Why not start with an agent for every question?

Known routes are easier to test, cheaper to run, and more predictable as workflows.
Autonomy must solve a real dynamic-decision problem.

### Who executes the tool?

Application Python. The model only proposes a typed action.

### What makes the EUR answer grounded?

The trajectory contains a successful price observation followed by a successful
conversion observation whose input amount equals that price.

### What does `MAX_STEPS` guarantee?

It bounds model decisions and tool calls. It does not guarantee answer quality.

### When should this agent become a workflow?

When production traces reveal a stable, finite dependency pattern that can be encoded and
tested explicitly.

## Failure lab

The looping policy always requests another NVIDIA price. With `MAX_STEPS=2`, the expected
result is:

```text
status=step_budget_exhausted
last_phase=guardrail
summary="Stopped after MAX_STEPS=2."
```

If it completes, the policy or runner contract has been changed incorrectly. If it keeps
running, interrupt the kernel and return to `run_bounded_agent` before discussing models.

## Provider behavior

### Ollama

`qwen3:8b` is the tested local default. Structured output can be slower than the offline
fixture. A first response that violates the Pydantic schema is a provider observation,
not permission to weaken the schema.

### OpenAI

The default is `gpt-5-mini`. Record the selected model and latency without storing the
key. Do not describe OpenAI as validated unless the complete notebook was actually run
with a configured key.

### Invalid structured output

1. read the Pydantic validation message;
2. confirm the provider and model displayed in the setup cell;
3. retry the single failing cell once;
4. if it fails again, use the offline fixture and keep the architecture discussion;
5. record the provider failure for post-class investigation.

Do not change `AgentDecision` fields during class.

## No-network fallback

Run:

```bash
uv run python scripts/execute_notebooks.py \
  notebooks/08_workflows_vs_agents.ipynb \
  --mode offline \
  --output-dir /private/tmp/finai-lesson08-offline
```

The notebook prints `offline fixture · deterministic course run`. Keep that label visible.
The fallback proves code and teaching contracts; it is not evidence of live model quality.

## If the class is five minutes late

Keep:

- the autonomy spectrum;
- direct workflow success;
- `unsupported_dependency` failure;
- ordered agent trajectory; and
- `MAX_STEPS` verification.

Skip live discussion of the snapshot table and assign the second challenge as homework.
Never skip the final architecture decision rule.

## Engineering mission solution

For the deterministic conversion branch:

1. route price-to-EUR questions to `price_then_fx`;
2. call `get_market_price`;
3. pass the returned amount and currency into `convert_currency`;
4. preserve both observations in the trace;
5. stop on either structured error; and
6. compare its fixed two-tool trajectory with the agent.

The preferred answer is conditional. For this single stable route, the workflow is easier
to test. The agent earns its cost only when additional tools or dependency shapes remain
open-ended.

## Safety boundary

- The snapshot is educational and not a live quote.
- The notebook does not trade, rebalance, or recommend securities.
- Every numeric output retains currency, date, and source.
- Tool output is data, not an instruction.
- Unsupported requests produce explicit errors or abstention.

## Transition to Lesson 09

Lesson 08 records an invalid tool request as a typed error observation, but the core
example does not yet teach systematic recovery. Lesson 09 introduces a LangGraph state,
agent and tool nodes, conditional routing, structured error feedback, retry counters, and
controlled self-correction.

Use this closing question:

> If a model requests `PE` but the valid metric is `P/E`, should the notebook crash, hide
> the error, or let the model correct its next action from structured feedback?

## Sources

- LangChain, structured output documentation: https://docs.langchain.com/oss/python/langchain/structured-output
- LangChain, tools documentation: https://docs.langchain.com/oss/python/langchain/tools
- Yahoo Finance historical observations used by the checked-in snapshot; exact URLs and
  retrieval date are recorded in `assets/course-data/manifest.json`.
