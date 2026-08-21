# Lesson 08 Workflows Versus Agents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify the complete 45-minute Lesson 08 package that compares a one-pass financial workflow with a transparent bounded agent using the same typed tools.

**Architecture:** A small reusable `agent_workflows` module owns typed market observations, tool execution, workflow and agent run records, and the bounded loop. The notebook makes those mechanics visible with deterministic fixtures in offline mode and the shared Ollama/OpenAI gateway in live mode. The deck mirrors the notebook’s autonomy spectrum, dependency failure, agent loop, and decision rule; the instructor chapter fixes pacing and recovery paths.

**Tech Stack:** Python 3.11+, Pydantic 2, LangChain chat/tool messages, Ollama or OpenAI through `finai_academy.providers`, pandas, matplotlib, nbformat/nbclient, python-pptx or the bundled presentation runtime, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-21-day-two-agent-progression-and-lesson-08-design.md`

## Global Constraints

- Preserve the canonical 09:30–10:15 slot in `course.yml`.
- Use `NVDA` and `SU.PA`; do not execute trades or produce investment advice.
- The classroom live path must support both Ollama and OpenAI through the shared provider boundary.
- The offline path is a labelled deterministic fixture for regression and instructor recovery only.
- Implement the agent loop transparently without LangGraph; Lesson 09 introduces LangGraph.
- Use two core tools: `get_market_price` and `convert_currency`.
- Enforce `MAX_STEPS`, explicit stop behavior, typed errors, currency, timestamp, and source metadata.
- Never claim a converted amount unless the trajectory contains both successful observations.
- Source notebooks committed to Git contain no output, execution count, secret, or absolute user path.
- Use original course explanations, examples, code, diagrams, and exercises.
- Every slide contains `[Sources]` speaker notes and the exact footer `First Finance - Arnaud Demes`.
- Run implementation edits in the existing branch; no isolated worktree is required by the user.

## File map

### Create

- `src/finai_academy/agent_workflows.py` — typed tools, observations, workflow results, bounded agent loop, and live model adapter boundary.
- `tests/test_agent_workflows.py` — unit contracts for validation, dependency failure, tool order, budgets, and grounded completion.
- `scripts/refresh_lesson08_market_snapshot.py` — reproducibly capture the last complete Yahoo Finance observations used by the lab.
- `data/course/lesson08_market_snapshot_v1.json` — labelled deterministic NVIDIA, Schneider Electric, and FX course observations with provenance.
- `notebooks/08_workflows_vs_agents.ipynb` — complete guided student lab with executable diagrams.
- `chapters/08-workflows-vs-agents.md` — instructor pacing, answers, provider guidance, and recovery paths.
- `decks/08-workflows-vs-agents.pptx` — nine-slide concept deck.
- `tests/test_lesson08_assets.py` — canonical asset, notebook output, deck footer, source-note, and copy contracts.
- `docs/reviews/lesson-08-readiness.md` — evidence-backed readiness score and known limitations.

### Modify

- `src/finai_academy/lesson_support.py` — add the labelled recorded policy used only by the offline notebook path.
- `tests/test_notebook_contracts.py` — execute Lesson 08 offline and assert its visuals, trace, and exact PASS marker.
- `tests/test_course_manifest.py` — require canonical Lesson 08 assets and timing.
- `notebooks/README.md` — list Lesson 08 as the first completed Day 2 lab.
- `chapters/README.md` — link the Lesson 08 instructor chapter.
- `decks/README.md` — link the Lesson 08 deck and preserve planned status for 09–12.

---

### Task 1: Typed Financial Tool Boundary

**Files:**
- Create: `src/finai_academy/agent_workflows.py`
- Create: `tests/test_agent_workflows.py`
- Create: `scripts/refresh_lesson08_market_snapshot.py`
- Create: `data/course/lesson08_market_snapshot_v1.json`

**Interfaces:**
- Produces: `ToolRequest`, `ToolObservation`, `MarketPrice`, `CurrencyConversion`, `ToolRegistry`, `load_course_market_snapshot`, `build_course_tool_registry`.
- Consumes: Pydantic 2 and a versioned JSON snapshot below `data/course`.

- [ ] **Step 1: Write failing contracts for typed observations and registry errors**

Add tests equivalent to:

```python
def test_market_price_retains_provenance() -> None:
    registry = build_course_tool_registry(SNAPSHOT)
    result = registry.invoke(ToolRequest(name="get_market_price", arguments={"ticker": "NVDA"}))
    assert result.status == "ok"
    assert result.payload["currency"] == "USD"
    assert result.payload["as_of"]
    assert result.payload["source"]


def test_registry_returns_actionable_unknown_tool_error() -> None:
    registry = build_course_tool_registry(SNAPSHOT)
    result = registry.invoke(ToolRequest(name="get_price", arguments={"ticker": "NVDA"}))
    assert result.status == "error"
    assert "get_market_price" in result.error
    assert "convert_currency" in result.error


def test_currency_conversion_rejects_missing_or_non_positive_amount() -> None:
    registry = build_course_tool_registry(SNAPSHOT)
    result = registry.invoke(
        ToolRequest(
            name="convert_currency",
            arguments={"amount": 0, "from_currency": "USD", "to_currency": "EUR"},
        )
    )
    assert result.status == "error"
    assert "positive" in result.error.casefold()
```

- [ ] **Step 2: Run the new tests and confirm the missing-module failure**

Run: `.venv/bin/pytest -q tests/test_agent_workflows.py`

Expected: collection fails because `finai_academy.agent_workflows` does not exist.

- [ ] **Step 3: Add the reproducible versioned snapshot**

Create `scripts/refresh_lesson08_market_snapshot.py` with `yfinance`. Fetch seven calendar days for `NVDA`, `SU.PA`, and `EURUSD=X`, select the last non-null complete close for each series, and write this exact schema:

```python
snapshot = {
    "dataset_id": "lesson08-market-snapshot-v1",
    "notice": "Checked-in course snapshot; not a live quote or investment recommendation.",
    "prices": {
        "NVDA": {
            "company": "NVIDIA",
            "price": last_close("NVDA"),
            "currency": "USD",
            "as_of": last_date("NVDA"),
            "source": "https://finance.yahoo.com/quote/NVDA/history/",
        },
        "SU.PA": {
            "company": "Schneider Electric",
            "price": last_close("SU.PA"),
            "currency": "EUR",
            "as_of": last_date("SU.PA"),
            "source": "https://finance.yahoo.com/quote/SU.PA/history/",
        },
    },
    "fx": {
        "USD_EUR": {
            "rate": 1.0 / last_close("EURUSD=X"),
            "as_of": last_date("EURUSD=X"),
            "source": "https://finance.yahoo.com/quote/EURUSD%3DX/history/",
        }
    },
}
```

Round stored prices to four decimals and the FX rate to six decimals. Sort JSON keys, use UTF-8, and end the file with a newline. Reject empty series, non-finite values, non-positive closes, and dates that differ by more than seven calendar days across the three observations. Run the script once and commit its generated snapshot; notebook execution reads the checked-in file and never fetches the network implicitly.

- [ ] **Step 4: Implement the typed boundary**

Define these exact public models and methods:

```python
class ToolRequest(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolObservation(BaseModel):
    tool_name: str
    status: Literal["ok", "error"]
    payload: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class MarketPrice(BaseModel):
    ticker: str
    company: str
    price: float = Field(gt=0)
    currency: str
    as_of: str
    source: str


class CurrencyConversion(BaseModel):
    input_amount: float = Field(gt=0)
    output_amount: float = Field(gt=0)
    rate: float = Field(gt=0)
    from_currency: str
    to_currency: str
    rate_as_of: str
    source: str


class ToolRegistry:
    @property
    def names(self) -> tuple[str, ...]: ...
    def invoke(self, request: ToolRequest) -> ToolObservation: ...


def load_course_market_snapshot(path: Path) -> dict[str, Any]: ...
def build_course_tool_registry(snapshot: Mapping[str, Any]) -> ToolRegistry: ...
```

`ToolRegistry.invoke` catches unknown names, missing arguments, unsupported tickers or currency pairs, non-finite amounts, and validation failures. It returns a structured error naming valid alternatives; it never hides a failure behind an empty payload.

- [ ] **Step 5: Run tests and lint the module**

Run: `.venv/bin/pytest -q tests/test_agent_workflows.py`

Expected: all Task 1 tests pass.

Run: `.venv/bin/ruff check src/finai_academy/agent_workflows.py tests/test_agent_workflows.py`

Expected: zero errors.

- [ ] **Step 6: Commit the typed tool boundary**

```bash
git add src/finai_academy/agent_workflows.py tests/test_agent_workflows.py scripts/refresh_lesson08_market_snapshot.py data/course/lesson08_market_snapshot_v1.json
git commit -m "feat: add typed financial agent tools"
```

---

### Task 2: One-Pass Workflow and Bounded Agent

**Files:**
- Modify: `src/finai_academy/agent_workflows.py`
- Modify: `tests/test_agent_workflows.py`
- Modify: `src/finai_academy/lesson_support.py`

**Interfaces:**
- Consumes: `ToolRegistry`, `ToolRequest`, and `ToolObservation` from Task 1.
- Produces: `WorkflowPlan`, `AgentDecision`, `TraceStep`, `OrchestrationResult`, `run_one_pass_workflow`, `run_bounded_agent`, `RecordedLesson08Model`.

- [ ] **Step 1: Write failing orchestration tests**

Cover these exact behaviors:

```python
def test_one_pass_workflow_handles_direct_price_request(): ...
def test_one_pass_workflow_returns_unsupported_dependency_without_fabrication(): ...
def test_bounded_agent_calls_price_before_conversion(): ...
def test_bounded_agent_stops_at_max_steps(): ...
def test_bounded_agent_rejects_ungrounded_converted_answer(): ...
def test_tool_error_is_retained_as_an_observation(): ...
```

The successful agent assertion must inspect tool names, not only final text:

```python
tool_steps = [step.tool_name for step in result.trajectory if step.phase == "tool"]
assert tool_steps == ["get_market_price", "convert_currency"]
assert result.status == "completed"
assert result.answer
```

- [ ] **Step 2: Run tests and confirm missing orchestration symbols**

Run: `.venv/bin/pytest -q tests/test_agent_workflows.py`

Expected: failures identify undefined workflow and agent interfaces.

- [ ] **Step 3: Implement the run records and planner boundaries**

Use these exact contracts:

```python
class WorkflowPlan(BaseModel):
    route: Literal["tool", "unsupported_dependency", "finish"]
    request: ToolRequest | None = None
    answer: str | None = None
    reason: str


class AgentDecision(BaseModel):
    action: Literal["tool", "finish"]
    request: ToolRequest | None = None
    answer: str | None = None


class TraceStep(BaseModel):
    index: int = Field(ge=1)
    phase: Literal["plan", "tool", "finish", "guardrail"]
    summary: str
    tool_name: str | None = None
    request: ToolRequest | None = None
    observation: ToolObservation | None = None


class OrchestrationResult(BaseModel):
    architecture: Literal["workflow", "agent"]
    status: Literal[
        "completed", "unsupported_dependency", "step_budget_exhausted", "error"
    ]
    answer: str | None
    trajectory: tuple[TraceStep, ...]
    latency_ms: float = Field(ge=0)
```

The runners accept callable boundaries so deterministic tests do not invoke a network:

```python
def run_one_pass_workflow(
    question: str,
    *,
    planner: Callable[[str], WorkflowPlan],
    answer_writer: Callable[[str, tuple[ToolObservation, ...]], str],
    registry: ToolRegistry,
) -> OrchestrationResult: ...


def run_bounded_agent(
    question: str,
    *,
    policy: Callable[[str, tuple[TraceStep, ...]], AgentDecision],
    registry: ToolRegistry,
    max_steps: int = 4,
) -> OrchestrationResult: ...
```

Before accepting a final answer that contains a converted amount, the agent runner verifies that the trajectory contains successful price and conversion observations. If not, it returns `status="error"` and a guardrail trace step.

- [ ] **Step 4: Add the recorded offline model**

`RecordedLesson08Model` provides three deterministic callables:

```python
model.plan_workflow(question) -> WorkflowPlan
model.write_workflow_answer(question, observations) -> str
model.decide_agent(question, trajectory) -> AgentDecision
```

It recognizes the direct NVDA lookup and the NVDA-to-EUR dependency question. It is labelled `offline fixture` in every notebook-visible record and cannot be mistaken for an Ollama or OpenAI execution.

- [ ] **Step 5: Verify orchestration tests and complete regression**

Run: `.venv/bin/pytest -q tests/test_agent_workflows.py tests/test_lesson_support.py`

Expected: all tests pass.

Run: `.venv/bin/ruff check src/finai_academy/agent_workflows.py src/finai_academy/lesson_support.py tests/test_agent_workflows.py`

Expected: zero errors.

- [ ] **Step 6: Commit orchestration**

```bash
git add src/finai_academy/agent_workflows.py src/finai_academy/lesson_support.py tests/test_agent_workflows.py
git commit -m "feat: compare fixed and bounded orchestration"
```

---

### Task 3: Guided Lesson 08 Notebook

**Files:**
- Create: `notebooks/08_workflows_vs_agents.ipynb`
- Modify: `tests/test_notebook_contracts.py`
- Create: `tests/test_lesson08_assets.py`

**Interfaces:**
- Consumes: all Task 1–2 public interfaces and `create_chat_model(Settings.from_environment())`.
- Produces: an output-free canonical notebook and an executed evidence copy outside Git.

- [ ] **Step 1: Write failing notebook asset and execution tests**

Require:

```python
assert NOTEBOOK.is_file()
assert "## Learning objectives" in source
assert "## Where this fits" in source
assert "## Failure lab" in source
assert "## Verification" in source
assert "## Challenge" in source
assert "## Capstone integration" in source
assert "## Recap" in source
assert "LESSON_08_PASS" in source
assert "unsupported_dependency" in source
assert "MAX_STEPS" in source
```

The offline execution test must assert:

```python
assert result.returncode == 0, result.stderr
assert count_png_outputs(executed) >= 4
assert "LESSON_08_PASS" in stream_text(executed)
assert "get_market_price" in stream_text(executed)
assert "convert_currency" in stream_text(executed)
assert "unsupported_dependency" in stream_text(executed)
```

- [ ] **Step 2: Run tests and confirm the notebook is missing**

Run: `.venv/bin/pytest -q tests/test_lesson08_assets.py tests/test_notebook_contracts.py -k lesson08`

Expected: failure because the canonical notebook does not exist.

- [ ] **Step 3: Build the notebook in the approved sequence**

Create unique cell IDs and `metadata.finai.expected_runtime_minutes: 30`. Include the 18 sections listed in the spec. Live mode must instantiate the configured provider and use structured output for `WorkflowPlan` and `AgentDecision`; offline mode uses `RecordedLesson08Model`.

Generate at least these four figures from executed state:

1. autonomy spectrum with determinism and autonomy axes;
2. one-pass workflow sequence and dependency stop;
3. bounded agent loop with `MAX_STEPS` guardrail;
4. aligned workflow-versus-agent trajectory comparison.

Also render a pandas table containing architecture, status, tool calls, steps, latency, and grounded outcome. Every chart title and caption must state whether the run is an offline fixture, Ollama, or OpenAI.

- [ ] **Step 4: Implement live provider adapters in notebook cells**

The live workflow planner uses:

```python
structured_planner = chat_model.with_structured_output(WorkflowPlan)
```

The live agent policy uses:

```python
structured_policy = chat_model.with_structured_output(AgentDecision)
```

Prompts contain valid tool names and schemas, visible prior observations, the remaining step budget, and the instruction to finish only from tool observations. Do not request or display hidden chain-of-thought; the model returns only the next typed action or grounded final answer.

- [ ] **Step 5: Execute and inspect the notebook offline**

Run outside a socket-restricted sandbox when necessary:

```bash
.venv/bin/python scripts/execute_notebooks.py \
  notebooks/08_workflows_vs_agents.ipynb \
  --mode offline \
  --output-dir /private/tmp/finai-lesson08-offline
```

Expected: one PASS line, at least four PNG outputs, the dependency failure, the ordered two-tool agent trace, and exactly one `LESSON_08_PASS` marker.

- [ ] **Step 6: Validate the source notebook and tests**

Run: `.venv/bin/python scripts/validate_notebooks.py notebooks/08_workflows_vs_agents.ipynb`

Expected: `1 notebook passed the course notebook contract.`

Run: `.venv/bin/pytest -q tests/test_agent_workflows.py tests/test_lesson08_assets.py tests/test_notebook_contracts.py -k 'agent_workflows or lesson08'`

Expected: all selected tests pass.

- [ ] **Step 7: Commit the notebook**

```bash
git add notebooks/08_workflows_vs_agents.ipynb tests/test_notebook_contracts.py tests/test_lesson08_assets.py
git commit -m "feat: add workflows versus agents notebook"
```

---

### Task 4: Instructor Chapter and Repository Navigation

**Files:**
- Create: `chapters/08-workflows-vs-agents.md`
- Modify: `notebooks/README.md`
- Modify: `chapters/README.md`
- Modify: `decks/README.md`
- Modify: `tests/test_course_manifest.py`

**Interfaces:**
- Consumes: the final notebook cell order and expected outputs from Task 3.
- Produces: teachable timing, answer key, provider commands, fallback instructions, and discoverable Lesson 08 assets.

- [ ] **Step 1: Add failing manifest and navigation tests**

Require the 09:30–10:15 window, canonical paths, asset existence, README links, and exact title `Workflows versus agents`.

- [ ] **Step 2: Run the focused tests and confirm missing chapter/deck links**

Run: `.venv/bin/pytest -q tests/test_course_manifest.py tests/test_lesson08_assets.py`

Expected: Lesson 08 asset/navigation assertions fail.

- [ ] **Step 3: Write the instructor chapter**

The chapter includes:

- 10-minute slide pacing and 30-minute notebook pacing;
- five-minute verification/debrief;
- expected direct workflow and compound-agent trajectories;
- checkpoint answers and architecture decision rule;
- Ollama and OpenAI launch commands;
- invalid structured-output, slow-model, and no-network recovery;
- the exact content cut if five minutes late;
- engineering mission solution for the deterministic conversion branch;
- safety language and non-advice boundary; and
- transition to Lesson 09 structured-error recovery with LangGraph.

- [ ] **Step 4: Update the three asset indexes**

List Lesson 08 as completed and Lessons 09–12 as planned. Do not describe unfinished files as available.

- [ ] **Step 5: Verify documentation and commit**

Run: `.venv/bin/pytest -q tests/test_course_manifest.py tests/test_lesson08_assets.py`

Expected at this point: only deck-specific assertions may remain failing.

```bash
git add chapters/08-workflows-vs-agents.md notebooks/README.md chapters/README.md decks/README.md tests/test_course_manifest.py tests/test_lesson08_assets.py
git commit -m "docs: add Lesson 08 instructor guidance"
```

---

### Task 5: Nine-Slide Visual Deck

**Files:**
- Create: `decks/08-workflows-vs-agents.pptx`
- Modify: `tests/test_lesson08_assets.py`

**Interfaces:**
- Consumes: the exact mechanisms and labels rendered by the notebook.
- Produces: a nine-slide 16:9 deck with sourced notes and exact footer.

- [ ] **Step 1: Read the Presentations skill and load the bundled workspace dependencies**

Use `presentations:Presentations` before any deck write. Inspect `decks/07-rag-evaluation.pptx`, the brand style, and the slide authoring guide. Preserve the established visual grammar rather than inventing a new template.

- [ ] **Step 2: Add failing deck package assertions**

Assert nine slides; footer and `[Sources]` notes on all slides; and the presence of these concepts in visible XML: `autonomy`, `unsupported dependency`, `MAX_STEPS`, `get_market_price`, `convert_currency`, and `lowest useful autonomy`.

- [ ] **Step 3: Create the nine-slide deck**

Use these exact slide responsibilities:

1. question and observable outcome;
2. autonomy spectrum;
3. architecture decision matrix;
4. typed financial tool boundary;
5. one-pass workflow sequence;
6. dependency failure;
7. bounded agent loop;
8. aligned execution comparison; and
9. decision rule and Lesson 09 bridge.

Use diagrams, short labels, and one primary claim per slide. Do not use screenshots from MLExpert Academy.

- [ ] **Step 4: Render and visually inspect every slide**

Use the bundled render tool to create PNGs in a temporary directory. Inspect the montage and individual slides for overflow, unintended overlap, weak hierarchy, clipped notes, tiny text, and inconsistent alignment. Iterate until all nine pass.

- [ ] **Step 5: Run deck validation**

Run the bundled `slides_test.py` against `decks/08-workflows-vs-agents.pptx` and run:

```bash
.venv/bin/pytest -q tests/test_lesson08_assets.py tests/test_course_manifest.py
```

Expected: zero overflow errors and all Lesson 08 package assertions pass.

- [ ] **Step 6: Commit the deck**

```bash
git add decks/08-workflows-vs-agents.pptx tests/test_lesson08_assets.py decks/README.md
git commit -m "slides: add workflows versus agents deck"
```

---

### Task 6: Live Provider Runs, Final QA, and Readiness Grade

**Files:**
- Create: `docs/reviews/lesson-08-readiness.md`
- Modify only if evidence reveals a defect: Lesson 08 implementation files from Tasks 1–5.

**Interfaces:**
- Consumes: complete Lesson 08 package.
- Produces: reproducible verification evidence, honest provider status, and a scored review.

- [ ] **Step 1: Run the full static suite**

Run:

```bash
.venv/bin/ruff check .
.venv/bin/pytest -q
.venv/bin/python scripts/validate_repo.py
.venv/bin/python scripts/validate_notebooks.py notebooks/08_workflows_vs_agents.ipynb
```

Expected: zero Ruff errors, zero test failures, valid repository structure, and one valid source notebook.

- [ ] **Step 2: Execute the Ollama path**

Run:

```bash
.venv/bin/python scripts/execute_notebooks.py \
  notebooks/08_workflows_vs_agents.ipynb \
  --mode live \
  --provider ollama \
  --output-dir /private/tmp/finai-lesson08-ollama
```

Expected: PASS with the configured local model. Record actual model, duration, trace ordering, and any structured-output retry. Do not claim success if Ollama is unavailable.

- [ ] **Step 3: Execute the OpenAI path when a key is available**

Check configuration without printing the key. If available, run:

```bash
.venv/bin/python scripts/execute_notebooks.py \
  notebooks/08_workflows_vs_agents.ipynb \
  --mode live \
  --provider openai \
  --output-dir /private/tmp/finai-lesson08-openai
```

Record pass/fail, model, duration, and tool order. If the key is absent, record `not run — OPENAI_API_KEY unavailable`; never infer a pass from the Ollama run.

- [ ] **Step 4: Grade against the specification**

Create `docs/reviews/lesson-08-readiness.md` with evidence for:

| Dimension | Weight |
|---|---:|
| Conceptual clarity and progression | 20% |
| Notebook usability and visuals | 20% |
| Technical correctness and safety | 20% |
| Provider neutrality and recovery | 15% |
| Deck quality and fidelity | 15% |
| Timing and instructor readiness | 10% |

Do not assign 9.5/10 or higher unless every weighted dimension is at least 9.3, all deterministic checks pass, the deck has been visually inspected, and at least one live provider run succeeds.

- [ ] **Step 5: Commit readiness evidence**

```bash
git add docs/reviews/lesson-08-readiness.md
git commit -m "docs: certify Lesson 08 readiness"
```

- [ ] **Step 6: Final clean-tree verification**

Run: `git status --short`

Expected: no output.

Run: `git log --oneline -7`

Expected: the design commit followed by the Lesson 08 implementation commits above.

## Self-review

- Spec coverage: tool contracts, dependency failure, bounded loop, provider neutrality, offline fallback, four notebook visuals, nine slides, instructor timing, verification marker, safety, and capstone handoff each map to a task.
- Placeholder scan: implementation steps contain exact file paths, commands, public interfaces, assertions, slide responsibilities, and acceptance outputs. The market snapshot is produced by a specified reproducible fetch script and validated before commit.
- Type consistency: `ToolRequest`, `ToolObservation`, `WorkflowPlan`, `AgentDecision`, `TraceStep`, and `OrchestrationResult` are defined once in Tasks 1–2 and consumed unchanged in later tasks.
