# Lesson 08 — Who Chooses the Next Step?

**First Finance - Arnaud Demes**  
**Day 2 · 09:30–10:15 · 10-minute concept deck + 30-minute notebook + 5-minute debrief**

## Learning outcome

Students leave with one reliable distinction:

> A workflow follows control flow defined in code. An agent lets a model choose the next action inside application-owned limits.

Tool count is not the distinction. A workflow can chain calls, branch on results, retry, and loop. The question is who owns the route.

## Lesson backbone

The lab compares three cases using the same typed finance tools.

1. **Direct lookup:** one-step workflow succeeds and is preferred.
2. **Price → FX:** a deterministic two-step workflow passes the observed price into conversion, succeeds, and is preferred.
3. **Model-directed loop:** a bounded agent performs the same fixed task, but its extra model decisions do not improve the result.

The final rule is deliberately demanding: use an agent only when **model-directed control flow creates measurable value** on an open-ended task.

## Why this correction matters

A dependent step does not automatically require an agent. If the dependency is known—retrieve a price, then convert that price—code can express it directly:

```text
question
   ↓
get_market_price
   ↓ observed price + currency
convert_currency
   ↓
grounded answer
```

This deterministic two-step workflow is easier to test, cheaper to run, and more predictable than asking a model to rediscover the same route at every step.

An agent earns its complexity when the route cannot be completely specified before the run. A reconciliation investigation is a better candidate: the evidence returned at each step may determine which ledger, policy, transaction, or exception to inspect next.

## Shared tool boundary

Both designs use the same application-owned tools:

- `get_market_price(ticker)` returns company, price, currency, date, and source;
- `convert_currency(amount, from_currency, to_currency)` returns input amount, output amount, rate, date, and source.

The model never executes Python. It can return a typed request. The application validates the request, invokes the registered tool, records the observation, and enforces grounding.

The OpenAI live path uses strict model-facing schemas with closed fields. Unused fields are nullable but required, and arbitrary `arguments` objects are not exposed to Structured Outputs.

## Agent boundaries

The bounded loop records every model choice and tool observation. Application code owns:

- the tool allowlist;
- argument validation;
- the requirement that conversion amount and source currency match the successful price observation;
- `MAX_STEPS` termination; and
- final-answer grounding checks.

These controls are executable guarantees, not prose in a prompt.

## Instructor run of show

| Time | Teaching move | Visible evidence |
|---|---|---|
| 09:30–09:40 | Use the concept deck to establish who controls execution | Workflow and agent diagrams |
| 09:40–09:46 | Inspect typed market and FX observations | Versioned snapshot table |
| 09:46–09:51 | Run direct lookup | `workflow_direct_status=completed` |
| 09:51–09:58 | Run price → FX workflow | `workflow_compound_status=completed` and exact tool order |
| 09:58–10:07 | Run the bounded agent on the same task | Same tools, more model route decisions |
| 10:07–10:12 | Trigger the step budget | `MAX_STEPS=2` guardrail |
| 10:12–10:15 | Debrief | `preferred_architecture=workflow` and `LESSON_08_PASS` |

## Provider modes

- **OpenAI:** set `FINAI_MODEL_PROVIDER=openai`; the default course model is `gpt-5.6-luna`.
- **Ollama:** set `FINAI_MODEL_PROVIDER=ollama` and ensure the configured local model is available.
- **No-network fallback:** run the labelled offline fixture. It reproduces the same decisions deterministically for teaching and recovery.

Run through the course executor:

```bash
python scripts/execute_notebooks.py notebooks/08_workflows_vs_agents.ipynb --mode offline
python scripts/execute_notebooks.py notebooks/08_workflows_vs_agents.ipynb --mode live --provider openai
```

## Verification contract

The final cell must prove all of the following:

- the direct workflow completes;
- the deterministic two-step workflow completes;
- its tool order is `get_market_price → convert_currency`;
- conversion input equals the observed price;
- the agent completes with the same tool order;
- a looping policy stops at `MAX_STEPS`; and
- the notebook emits `LESSON_08_PASS` exactly once.

## Knowledge check

1. Can a workflow use one tool result as the input to another tool?  
   **Yes.** Code can pass observations through a predefined chain, branch, or loop.

2. What makes the bounded example an agent?  
   The model—not Python—chooses the next action after each visible observation.

3. Why is the workflow preferred for price → FX?  
   The route is known, testable, and gains no quality from repeated model decisions.

4. What owns termination?  
   Application code through `MAX_STEPS`.

5. What would justify an agent?  
   Measured quality gains on tasks where the correct path genuinely cannot be specified in advance.

## Challenge and capstone bridge

Add a predefined Schneider Electric branch, then describe one open-ended finance investigation. State the evaluation metric that would show whether the agent earns its added latency, cost, and variability.

Lesson 09 keeps the same typed requests, observations, trace, and hard stop. It adds explicit state and self-correction only after Lesson 08 establishes the simpler baseline.

## Sources

- Anthropic, [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents).
- LangChain, [Workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents).
- OpenAI, [A practical guide to building agents](https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/).
- OpenAI, [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
- OpenAI, [Function calling](https://developers.openai.com/api/docs/guides/function-calling).
- Course snapshot provenance is recorded in `assets/course-data/manifest.json`.
