# Day 2 Agent Progression and Lesson 08 Design

## Status

Approved direction from Arnaud Demes on 21 August 2026. This specification
supersedes the Day 2 lesson ordering in the original two-day class design while
preserving its timing, finance scope, provider neutrality, and capstone goal.

Implementation must not reproduce MLExpert Academy code or lesson copy. The
course adapts its progression from constrained workflows to evaluated agentic
systems using original NVIDIA and Schneider Electric examples, diagrams,
exercises, typed contracts, and verification.

## Decision basis

The following MLExpert Academy lessons were reviewed as a connected sequence:

1. Workflows vs Agents;
2. Vectorless RAG;
3. Self-Correcting Agent;
4. Build an MCP Agent;
5. Plan and Execute Agent;
6. Thinking and Acting — Build an AI Agent;
7. Agentic RAG — Building an AI Financial Analyst Team; and
8. Evaluating Agentic Systems.

The reusable teaching pattern is to expose a specific limitation before adding
autonomy. The two-day workshop therefore keeps the smallest sufficient system at
each stage and makes the reason for the next stage observable.

## Approved Day 2 progression

| Time | Lesson | Observable engineering result |
|---|---|---|
| 09:00–09:30 | Debugging review | One measured Day 1 retrieval failure corrected |
| 09:30–10:15 | 08 — Workflows versus agents | Fixed workflow and minimal bounded agent compared |
| 10:15–10:30 | Break | — |
| 10:30–11:15 | 09 — Self-correcting financial agent | LangGraph agent recovers from a structured tool error |
| 11:15–12:00 | 10 — Build a financial MCP | Client discovers financial resources, tools, and prompts |
| 12:00–13:30 | Lunch | — |
| 13:30–14:30 | 11 — Plan-and-execute financial analyst | Structured multi-source plan executes and replans |
| 14:30–15:30 | 12 — Evaluating agentic systems | Final answers and tool trajectories are scored separately |
| 15:30–16:30 | Capstone integration | Financial Analyst Copilot assembled from course components |
| 16:30–17:00 | Demonstration and review | NVIDIA–Schneider mission demonstrated and reviewed |

### Scope decisions

- Vectorless tree retrieval belongs in the advanced hierarchical section of
  Lesson 05, not as a second Day 2 RAG lesson.
- The SQL agent is an optional homework pattern because it repeats the core
  tool-selection loop in a different data domain.
- A supervisor multi-agent system is a capstone extension. The classroom core
  uses one bounded agent with narrow tools.
- Lesson 12 reuses MLflow concepts from Lesson 07 and adds trajectory-specific
  evaluation rather than introducing another observability platform.

## Lesson 08 purpose

Lesson 08 answers one engineering question:

> When is a deterministic workflow sufficient, and when does the next action
> genuinely need to depend on an intermediate observation?

Students must not leave with the rule that agents are inherently better. They
must be able to choose the lowest useful level of autonomy and explain the cost
of moving upward.

## Learning objectives

By the end of the 45-minute block, a learner can:

1. distinguish a normal function, fixed workflow, bounded agent, and multi-agent
   system;
2. define typed financial tools with explicit argument and result contracts;
3. inspect a one-pass workflow and identify its dependency limitation;
4. implement a transparent reason–act–observe–stop loop;
5. enforce maximum-step and explicit-stop boundaries;
6. compare workflow and agent execution using visible trajectory, latency, and
   call-count records; and
7. justify which architecture should handle a given analyst task.

## Classroom boundaries

- Core duration is 45 minutes: 10 minutes slides, 30 minutes guided notebook,
  and 5 minutes verification and debrief.
- The agent loop is implemented transparently without LangGraph. Lesson 09
  introduces LangGraph when recovery and state routing justify it.
- The core has two tools. Additional tools appear only in the challenge or later
  lessons.
- No trade execution, investment recommendation, or portfolio mutation occurs.
- Live LLM execution must work through the shared OpenAI or Ollama gateway.
- Deterministic fixtures support automated tests and classroom recovery, but are
  clearly labelled and are not presented as a live model run.
- Source notebooks committed to Git contain no executed outputs or secrets.

## Finance scenario and tool contracts

The lesson uses NVIDIA (`NVDA`) and Schneider Electric (`SU.PA`) as the continuing
course cases. The core question is deliberately dependency-bearing:

> What is NVIDIA's latest available share price converted to euros?

The first tool returns an observation required by the second tool. The model
cannot correctly supply the conversion amount before seeing the price result.

### `get_market_price`

Input:

```text
ticker: supported ticker symbol
```

Typed result fields:

```text
ticker, company, price, currency, as_of, source, status, error
```

### `convert_currency`

Input:

```text
amount, from_currency, to_currency
```

Typed result fields:

```text
input_amount, output_amount, rate, from_currency, to_currency,
rate_as_of, source, status, error
```

The classroom implementation uses deterministic course market observations for
repeatability and may expose a labelled live-data adapter as an extension. The
lesson teaches orchestration, not market-data vendor integration.

## Teaching sequence

### 1. Frame the autonomy spectrum

Show the same analyst request implemented as a function, fixed workflow, bounded
agent, and multi-agent system. Compare determinism, flexibility, latency,
debuggability, cost, and failure surface.

### 2. Build the smallest useful workflow

The one-pass workflow:

1. receives the user question;
2. asks the model for a structured tool request;
3. validates and executes that request;
4. gives the observation to the model for a final response; and
5. records a typed trace.

It succeeds on a direct price lookup.

### 3. Observe a real limitation

The workflow receives the price-in-euros question. Its one-pass design can make
one tool decision but cannot use the price observation to construct a second
tool call. It must return a structured `unsupported_dependency` outcome rather
than fabricate a converted price.

The instructor must state explicitly that a deterministic developer can add a
known conversion branch. The failure demonstrates that every new dependency
shape needs another predefined path; it does not prove that workflows cannot
chain operations.

### 4. Add a bounded agent loop

The agent repeats:

```text
reason over visible messages
  -> select one tool or finish
  -> validate the request
  -> execute the tool
  -> append the observation
  -> enforce MAX_STEPS
```

The expected successful trajectory is:

```text
get_market_price(NVDA)
  -> observe USD price
  -> convert_currency(observed_price, USD, EUR)
  -> observe EUR amount
  -> final grounded response
```

### 5. Compare and decide

Students compare both paths using the same trace schema. The decision rule is:

- use a workflow when routes and dependencies are known and stable;
- use an agent when intermediate observations determine an open-ended next
  action;
- keep the agent bounded, inspectable, and replaceable by a workflow when a
  stable pattern emerges.

## Notebook design

Canonical file: `notebooks/08_workflows_vs_agents.ipynb`

The notebook follows the course-wide observable lesson contract:

1. title, outcome, prerequisites, and expected visible results;
2. final product increment and architecture diagram;
3. autonomy-spectrum comparison;
4. shared provider setup for Ollama or OpenAI;
5. typed tool arguments, results, and trace records;
6. deterministic course observations and tool execution;
7. one-pass workflow implementation;
8. successful direct-query run;
9. controlled compound-query failure;
10. reason–act–observe–stop loop;
11. successful bounded-agent run;
12. side-by-side trajectory, call-count, latency, and outcome table;
13. architecture decision matrix;
14. failure lab for a step-budget exhaustion or invalid tool request;
15. deterministic verification checks and final `LESSON_08_PASS` marker;
16. three- to five-question knowledge check;
17. bounded engineering mission; and
18. capstone handoff to Lesson 09.

The notebook must render its architecture and trajectory visuals from code so
students can see the system evolve without opening the slide deck.

## Slide design

Canonical file: `decks/08-workflows-vs-agents.pptx`

Target: nine concise slides.

1. lesson question and observable outcome;
2. autonomy spectrum;
3. architecture decision matrix;
4. typed tool boundary;
5. one-pass workflow sequence;
6. dependency failure diagram;
7. bounded agent loop;
8. workflow-versus-agent trace comparison; and
9. decision rule and Lesson 09 transition.

Each conceptual relationship must be expressed visually, not as a paragraph.
Every slide uses the exact footer `First Finance - Arnaud Demes`, contains source
notes, and follows the established course palette and typography.

## Instructor chapter

Canonical file: `chapters/08-workflows-vs-agents.md`

The chapter records:

- minute-by-minute pacing;
- slide-to-notebook transitions;
- exact expected tool trajectories;
- checkpoint answers;
- likely Ollama and OpenAI tool-calling differences;
- recovery instructions for invalid structured output;
- a no-network fallback using deterministic fixtures;
- material to skip if the class is five minutes late; and
- the conceptual bridge to LangGraph recovery in Lesson 09.

## Failure and safety contract

- Unknown tool names and invalid arguments become typed observations, never
  unhandled notebook crashes.
- The agent cannot exceed `MAX_STEPS`.
- A final response cannot claim a converted amount unless both required tool
  observations exist in the trajectory.
- Model text never becomes a market-data observation.
- Every numeric result retains currency, timestamp, and source metadata.
- Tool outputs and retrieved content are treated as data, not instructions.
- The lesson states that all examples are educational and not investment advice.

## Verification and acceptance criteria

Implementation is complete only when:

- source notebook structure passes the repository notebook contract;
- offline fixtures execute deterministically for automated regression;
- one Ollama live run and one OpenAI live run are attempted through the shared
  gateway, with unavailable credentials reported rather than concealed;
- the direct workflow query succeeds;
- the dependency query returns the expected structured workflow limitation;
- the bounded agent calls price before currency conversion;
- the agent stops within its configured budget;
- the final numeric answer can be recomputed from the recorded observations;
- all notebook figures render without external files;
- the deck passes overflow, source-note, footer, and visual-fidelity checks;
- unit tests cover tool validation, unsupported dependency, tool ordering,
  stopping, and grounded finalization; and
- the final notebook marker is exactly `LESSON_08_PASS`.

## Engineering mission

Students receive a scaffold for one of two bounded changes:

1. add a deterministic currency-conversion branch to the workflow and compare
   it with the agent; or
2. add a `calculate_return` tool and make the agent compare a supported NVIDIA
   or Schneider Electric period without exceeding the step budget.

The required reflection is architectural: which implementation is easier to
test, and does the added autonomy earn its latency and failure surface?

## Capstone handoff

Lesson 08 contributes typed tools, a common trajectory record, and a minimal
bounded agent loop. Lesson 09 retains those contracts, adds LangGraph state,
feeds structured tool errors back to the model, and demonstrates controlled
self-correction. Later lessons expose the tools through MCP, plan multi-source
research, and evaluate the resulting trajectory.
