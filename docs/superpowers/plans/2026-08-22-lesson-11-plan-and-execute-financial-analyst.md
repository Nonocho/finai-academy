# Lesson 11 Plan-and-Execute Financial Analyst Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and certify a 60-minute Lesson 11 in which learners plan, execute, revise, and synthesize a cited NVIDIA and Schneider Electric research mission through the real Lesson 10 MCP boundary.

**Architecture:** Keep plan contracts and validation pure, hold one real MCP `stdio` lifecycle inside a narrow async executor, and inject planner, replanner, and report policies into one bounded LangGraph. The maintained offline route uses deterministic policies with the real MCP server; optional Ollama and OpenAI policies use the same graph, validation, evidence gate, and trajectory schema.

**Tech Stack:** Python 3.11+, Pydantic 2, LangGraph, MCP Python SDK v2, LangChain provider adapters, Ollama or OpenAI through the existing gateway, `uv`, Jupyter/nbclient, Matplotlib, Pandas, pytest, Ruff, and PowerPoint through the established presentation workflow.

**Spec:** `docs/superpowers/specs/2026-08-22-lesson-11-plan-and-execute-financial-analyst-design.md`

## Global Constraints

- Preserve the canonical Day 2 slot: 13:30-14:30.
- Core format: 12-minute deck, 40-minute notebook, and 8-minute verification and debrief.
- Use one bounded plan-and-execute graph, not a multi-agent team.
- The planner, replanner, and report writer may use an LLM; policy gates and execution remain deterministic.
- Execute only the runtime-discovered and statically allowlisted `get_company_metric` and `search_financial_documents` tools.
- Use one local MCP `stdio` lifecycle for a complete research run.
- Reuse NVIDIA (`NVDA`) and Schneider Electric (`SU.PA`) and preserve dates, sources, document IDs, and evidence IDs.
- The maintained mission compares available valuation and operating-growth evidence without issuing investment advice.
- The maintained failure asks `get_company_metric(ticker="NVDA", metric="Revenue")`, then changes strategy to document search after `unsupported_metric`.
- Allow at most six initial or replacement execution steps and one plan revision.
- Never duplicate a successful tool call after replanning.
- Require at least one metric observation and one document evidence hit per company before report generation.
- Keep the checked-in notebook output-free, deterministic offline, and free of secrets.
- Every LLM-dependent path must support `FINAI_MODEL_PROVIDER=ollama` or `FINAI_MODEL_PROVIDER=openai` without provider-specific learner code.
- Use short factual English in slides and learner-facing notebook copy; do not use visible em dashes.
- Use the footer `First Finance - Arnaud Demes` on every slide.
- Add a `[Sources]` block and closing `[/Sources]` tag to every slide's speaker notes.
- Do not reproduce MLExpert Academy code, lesson copy, diagrams, data, or screenshots.

## File Map

| File | Responsibility |
| --- | --- |
| `src/finai_academy/research_planning.py` | Pydantic plan, observation, briefing, evidence-gate, and validation contracts. |
| `src/finai_academy/planning_mcp_executor.py` | One-lifecycle MCP discovery, allowlisting, typed execution, and safe result parsing. |
| `src/finai_academy/plan_execute_graph.py` | LangGraph state, routing, budgets, append-only trajectory, and run result. |
| `src/finai_academy/plan_execute_policies.py` | Deterministic classroom policies and provider-neutral live model policies. |
| `tests/test_research_planning.py` | Pure plan, replacement, argument, evidence, and serialization tests. |
| `tests/test_planning_mcp_executor.py` | Real local `stdio` lifecycle, discovery, provenance, and error tests. |
| `tests/test_plan_execute_graph.py` | Offline graph, revision, budget, deduplication, and evidence-gate tests. |
| `tests/test_plan_execute_policies.py` | Recorded policy and fake-model structured-output tests. |
| `scripts/build_lesson11_notebook.py` | Stable output-free notebook generator. |
| `notebooks/11_plan_and_execute_analyst.ipynb` | Student-facing 40-minute guided notebook with six generated visuals. |
| `chapters/11-plan-and-execute-analyst.md` | Instructor timing, expected outputs, recovery routes, and debrief. |
| `decks/11-plan-and-execute-analyst.pptx` | Nine-slide sourced concept deck. |
| `tests/test_lesson11_assets.py` | Notebook, chapter, deck, index, and execution contracts. |
| `docs/reviews/lesson-11-readiness.md` | Evidence-based delivery score and provider status. |

---

### Task 1: Build the pure research-plan and evidence contracts

**Files:**
- Create: `src/finai_academy/research_planning.py`
- Create: `tests/test_research_planning.py`

**Interfaces:**
- Consumes: Pydantic 2 only.
- Produces: `PlannerToolSpec`, `PlanStep`, `ResearchPlan`, `ReplanDecision`, `ResearchObservation`, `TrajectoryEvent`, `AnalystBriefing`, `EvidenceGateResult`, `validate_plan()`, `validate_replacement()`, and `evaluate_evidence_gate()`.

- [ ] **Step 1: Write failing plan and replacement tests**

Create these tests:

```python
from finai_academy.research_planning import (
    PlanStep,
    PlannerToolSpec,
    ResearchPlan,
    validate_plan,
    validate_replacement,
)


def tool_catalog() -> tuple[PlannerToolSpec, ...]:
    return (
        PlannerToolSpec(
            name="get_company_metric",
            description="Return one controlled company metric.",
            input_schema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "metric": {"type": "string"},
                },
                "required": ["ticker", "metric"],
            },
        ),
        PlannerToolSpec(
            name="search_financial_documents",
            description="Search controlled financial evidence.",
            input_schema={
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 3},
                },
                "required": ["company", "query"],
            },
        ),
    )


def valid_plan() -> ResearchPlan:
    return ResearchPlan(
        goal="Compare available NVIDIA and Schneider Electric evidence.",
        steps=(
            PlanStep(
                step_id=1,
                capability="get_company_metric",
                arguments={"ticker": "NVDA", "metric": "P/E"},
                purpose="Collect NVIDIA valuation evidence.",
                expected_evidence=("NVDA P/E",),
            ),
            PlanStep(
                step_id=2,
                capability="get_company_metric",
                arguments={"ticker": "SU.PA", "metric": "P/E"},
                purpose="Collect Schneider Electric valuation evidence.",
                expected_evidence=("SU.PA P/E",),
            ),
        ),
    )


def test_valid_plan_accepts_discovered_allowlisted_tools() -> None:
    checked = validate_plan(valid_plan(), tool_catalog(), max_steps=6)
    assert checked.steps[0].step_id == 1
    assert checked.steps[1].depends_on == ()


def test_plan_rejects_unknown_capability_before_execution() -> None:
    plan = valid_plan().model_copy(
        update={
            "steps": (
                valid_plan().steps[0].model_copy(update={"capability": "delete_portfolio"}),
            )
        }
    )
    with pytest.raises(ValueError, match="capability_not_permitted"):
        validate_plan(plan, tool_catalog(), max_steps=6)


def test_plan_rejects_non_sequential_initial_ids() -> None:
    plan = valid_plan().model_copy(
        update={"steps": (valid_plan().steps[0].model_copy(update={"step_id": 2}),)}
    )
    with pytest.raises(ValueError, match="initial_step_ids"):
        validate_plan(plan, tool_catalog(), max_steps=6)


def test_replacement_requires_new_monotonic_ids() -> None:
    replacement = (
        PlanStep(
            step_id=3,
            capability="search_financial_documents",
            arguments={"company": "NVIDIA", "query": "revenue growth", "top_k": 2},
            purpose="Replace the unsupported revenue metric with document evidence.",
            expected_evidence=("NVIDIA revenue evidence",),
            depends_on=(1,),
        ),
    )
    checked = validate_replacement(
        replacement,
        catalog=tool_catalog(),
        prior_step_ids=(1, 2),
        successful_step_ids=(1,),
        max_total_steps=6,
    )
    assert checked[0].step_id == 3
```

- [ ] **Step 2: Run focused tests and confirm RED**

```bash
.venv/bin/pytest -q tests/test_research_planning.py
```

Expected: collection fails because `finai_academy.research_planning` does not exist.

- [ ] **Step 3: Implement the typed contracts**

Create these public models and validators:

```python
class PlannerToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]


class PlanStep(BaseModel):
    step_id: int = Field(ge=1)
    capability: str = Field(min_length=1)
    arguments: dict[str, Any]
    purpose: str = Field(min_length=1)
    expected_evidence: tuple[str, ...]
    depends_on: tuple[int, ...] = ()


class ResearchPlan(BaseModel):
    goal: str = Field(min_length=1)
    steps: tuple[PlanStep, ...]


class ReplanDecision(BaseModel):
    action: Literal["continue", "replace_remaining", "finish", "stop"]
    reasoning: str = Field(min_length=1)
    replacement_steps: tuple[PlanStep, ...] = ()
    limitations: tuple[str, ...] = ()


class ResearchObservation(BaseModel):
    attempt_id: int = Field(ge=1)
    step_id: int = Field(ge=1)
    plan_revision: int = Field(ge=0)
    capability: str
    arguments: dict[str, Any]
    status: Literal["ok", "error", "blocked"]
    result: dict[str, Any] | None = None
    error_code: str | None = None
    evidence_ids: tuple[str, ...] = ()
    source_references: tuple[str, ...] = ()
    duration_ms: float = Field(ge=0)


class TrajectoryEvent(BaseModel):
    index: int = Field(ge=1)
    phase: Literal[
        "planning",
        "policy",
        "execution",
        "replanning",
        "evidence_gate",
        "report",
        "guardrail",
    ]
    status: Literal["ok", "error", "blocked"]
    summary: str = Field(min_length=1)
    step_id: int | None = None
    attempt_id: int | None = None
    duration_ms: float = Field(default=0, ge=0)


class AnalystBriefing(BaseModel):
    reported_facts: tuple[str, ...]
    cross_company_observations: tuple[str, ...]
    interpretation: tuple[str, ...]
    limitations: tuple[str, ...]
    source_references: tuple[str, ...]


class EvidenceGateResult(BaseModel):
    passed: bool
    coverage: dict[str, tuple[str, ...]]
    missing_requirements: tuple[str, ...] = ()
```

Reject an observation with `status="ok"` and no result, or `status="error"` and no `error_code`. Reject a briefing with no reported facts, no limitations, or no source references.

Implement a limited JSON-object schema validator that supports the MCP schemas used here: `required`, `additionalProperties`, `type` for string/integer/number/array/object, `enum`, `minimum`, and `maximum`. Do not add another dependency.

- [ ] **Step 4: Add evidence-gate tests and implementation**

Add a fixture with successful metric and document observations for both companies, then assert:

```python
gate = evaluate_evidence_gate(observations)
assert gate.passed is True
assert gate.missing_requirements == ()
assert gate.coverage == {
    "NVIDIA": ("document", "metric"),
    "Schneider Electric": ("document", "metric"),
}
```

Remove Schneider Electric's document observation and assert `passed is False` with `Schneider Electric document evidence` in `missing_requirements`.

Implement `evaluate_evidence_gate()` by reading successful observations only. Metric results identify companies through `result["company"]`; document results identify them through `result["company"]` and non-empty `result["hits"]`.

- [ ] **Step 5: Run pure tests and lint**

```bash
.venv/bin/pytest -q tests/test_research_planning.py
.venv/bin/ruff check src/finai_academy/research_planning.py tests/test_research_planning.py
```

Expected: all pure contract tests pass and Ruff reports no issues.

- [ ] **Step 6: Commit the pure planning boundary**

```bash
git add src/finai_academy/research_planning.py tests/test_research_planning.py
git commit -m "feat: add typed financial research plans"
```

---

### Task 2: Build the one-lifecycle MCP research executor

**Files:**
- Create: `src/finai_academy/planning_mcp_executor.py`
- Create: `tests/test_planning_mcp_executor.py`

**Interfaces:**
- Consumes: `financial_stdio_transport()` and `ALLOWED_TOOLS` from `finai_academy.financial_mcp_client`; `PlannerToolSpec`, `PlanStep`, and `ResearchObservation` from Task 1.
- Produces: `FinancialMcpPlanningExecutor`, `extract_structured_tool_result()`, and `extract_capability_error()`.

- [ ] **Step 1: Write failing real-stdio executor tests**

Create synchronous pytest tests that use `asyncio.run()` around one executor context:

```python
def revenue_metric_step() -> PlanStep:
    return PlanStep(
        step_id=1,
        capability="get_company_metric",
        arguments={"ticker": "NVDA", "metric": "Revenue"},
        purpose="Attempt to collect NVIDIA revenue as a structured metric.",
        expected_evidence=("NVIDIA revenue",),
    )


def test_executor_discovers_only_permitted_read_tools() -> None:
    async def scenario() -> None:
        async with FinancialMcpPlanningExecutor() as executor:
            assert tuple(item.name for item in executor.catalog) == (
                "get_company_metric",
                "search_financial_documents",
            )
            assert executor.server_name == "First Finance Research"

    asyncio.run(scenario())


def test_executor_preserves_metric_and_document_provenance() -> None:
    async def scenario() -> tuple[ResearchObservation, ResearchObservation]:
        async with FinancialMcpPlanningExecutor() as executor:
            metric = await executor.execute(
                PlanStep(
                    step_id=1,
                    capability="get_company_metric",
                    arguments={"ticker": "NVDA", "metric": "P/E"},
                    purpose="Collect valuation.",
                    expected_evidence=("NVDA P/E",),
                ),
                attempt_id=1,
                plan_revision=0,
            )
            document = await executor.execute(
                PlanStep(
                    step_id=2,
                    capability="search_financial_documents",
                    arguments={"company": "NVIDIA", "query": "revenue growth", "top_k": 2},
                    purpose="Collect operating evidence.",
                    expected_evidence=("NVIDIA revenue",),
                ),
                attempt_id=2,
                plan_revision=0,
            )
            return metric, document

    metric, document = asyncio.run(scenario())
    assert metric.status == "ok" and metric.source_references
    assert document.status == "ok" and document.evidence_ids


def test_executor_converts_unsupported_metric_to_observation() -> None:
    async def scenario() -> ResearchObservation:
        async with FinancialMcpPlanningExecutor() as executor:
            return await executor.execute(
                revenue_metric_step(), attempt_id=1, plan_revision=0
            )

    observation = asyncio.run(scenario())
    assert observation.status == "error"
    assert observation.error_code == "unsupported_metric"
    assert observation.result is not None
    assert "P/E" in observation.result["valid_values"]
```

- [ ] **Step 2: Run executor tests and confirm RED**

```bash
.venv/bin/pytest -q tests/test_planning_mcp_executor.py
```

Expected: collection fails because `planning_mcp_executor.py` does not exist.

- [ ] **Step 3: Implement discovery and lifecycle ownership**

Use one `Client` context for the whole run:

```python
class FinancialMcpPlanningExecutor:
    def __init__(self) -> None:
        self._client_context: Client | None = None
        self._client: Client | None = None
        self.catalog: tuple[PlannerToolSpec, ...] = ()
        self.server_name = ""

    async def __aenter__(self) -> FinancialMcpPlanningExecutor:
        self._client_context = Client(financial_stdio_transport())
        self._client = await self._client_context.__aenter__()
        self.server_name = self._client.server_info.name
        discovered = await self._client.list_tools()
        self.catalog = tuple(
            PlannerToolSpec(
                name=tool.name,
                description=tool.description or "",
                input_schema=dict(tool.input_schema),
            )
            for tool in discovered.tools
            if tool.name in ALLOWED_TOOLS
        )
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        if self._client_context is not None:
            await self._client_context.__aexit__(exc_type, exc, traceback)
```

Fail closed if discovery contains no permitted tools or if an `execute()` call names a tool absent from both discovery and `ALLOWED_TOOLS`.

- [ ] **Step 4: Implement typed execution and safe parsing**

`execute()` must time the call, parse `structured_content` for success, parse the JSON `CapabilityError` from text content for failure, and return `ResearchObservation`. Extract metric source values and document hit evidence IDs without logging raw protocol blocks.

Use this dispatch boundary:

```python
async def execute(
    self,
    step: PlanStep,
    *,
    attempt_id: int,
    plan_revision: int,
) -> ResearchObservation:
    if self._client is None:
        raise RuntimeError("executor must be opened before execution")
    permitted = {item.name for item in self.catalog}
    if step.capability not in ALLOWED_TOOLS or step.capability not in permitted:
        return blocked_observation(step, attempt_id, plan_revision, "capability_not_permitted")
    started = perf_counter()
    result = await self._client.call_tool(step.capability, step.arguments)
    duration_ms = (perf_counter() - started) * 1000
    return observation_from_call_result(
        step=step,
        result=result,
        attempt_id=attempt_id,
        plan_revision=plan_revision,
        duration_ms=duration_ms,
    )
```

- [ ] **Step 5: Verify one lifecycle and safe failures**

Add a test spy around `financial_stdio_transport()` that counts one open and one close for four sequential calls. Add tests for execution before opening and execution of `delete_portfolio`. Then run:

```bash
.venv/bin/pytest -q tests/test_planning_mcp_executor.py tests/test_financial_mcp_client.py tests/test_financial_mcp_server.py
.venv/bin/ruff check src/finai_academy/planning_mcp_executor.py tests/test_planning_mcp_executor.py
```

Expected: all tests pass; no subprocess configuration or environment data appears in serialized observations.

- [ ] **Step 6: Commit the MCP executor**

```bash
git add src/finai_academy/planning_mcp_executor.py tests/test_planning_mcp_executor.py
git commit -m "feat: add persistent MCP research executor"
```

---

### Task 3: Build the bounded plan-execute-replan graph

**Files:**
- Create: `src/finai_academy/plan_execute_graph.py`
- Create: `tests/test_plan_execute_graph.py`

**Interfaces:**
- Consumes: all contracts and validators from Task 1; an injected executor matching `execute(step, attempt_id, plan_revision)` and exposing `catalog`.
- Produces: `PlanExecuteState`, `PlanExecuteResult`, `PlannerPolicy`, `ReplannerPolicy`, `ReportPolicy`, `build_plan_execute_graph()`, and `run_plan_execute()`.

- [ ] **Step 1: Write the failing successful-revision graph test**

Create a deterministic fake executor with the same public interface as Task 2. It returns success for `P/E`, `unsupported_metric` for `Revenue`, and document hits for both companies. Assert:

```python
result = asyncio.run(
    run_plan_execute(
        question=MISSION,
        executor=fake_executor,
        planner=recorded_planner,
        replanner=recorded_replanner,
        report_writer=recorded_report_writer,
    )
)

assert result.status == "completed"
assert result.replan_count == 1
assert [item.attempt_id for item in result.observations] == [1, 2, 3, 4, 5]
assert [item.step_id for item in result.observations] == [1, 2, 3, 5, 6]
assert result.observations[2].error_code == "unsupported_metric"
assert result.evidence_gate.passed is True
assert result.briefing is not None
```

The initial plan IDs are `1, 2, 3, 4`; the replacement tail IDs are `5, 6`. Calls for successful steps `1` and `2` must occur exactly once.

- [ ] **Step 2: Run graph tests and confirm RED**

```bash
.venv/bin/pytest -q tests/test_plan_execute_graph.py
```

Expected: collection fails because `plan_execute_graph.py` does not exist.

- [ ] **Step 3: Implement state, protocols, and graph nodes**

Use these interfaces:

```python
class PlannerPolicy(Protocol):
    async def __call__(
        self, question: str, catalog: tuple[PlannerToolSpec, ...]
    ) -> ResearchPlan:
        raise NotImplementedError


class ReplannerPolicy(Protocol):
    async def __call__(self, state: Mapping[str, Any]) -> ReplanDecision:
        raise NotImplementedError


class ReportPolicy(Protocol):
    async def __call__(
        self, question: str, observations: tuple[ResearchObservation, ...]
    ) -> AnalystBriefing:
        raise NotImplementedError


class PlanExecuteState(TypedDict, total=False):
    question: str
    catalog: tuple[PlannerToolSpec, ...]
    initial_plan: ResearchPlan
    active_steps: tuple[PlanStep, ...]
    all_step_ids: tuple[int, ...]
    current_index: int
    observations: tuple[ResearchObservation, ...]
    trajectory: tuple[TrajectoryEvent, ...]
    plan_revision: int
    replan_count: int
    status: str
    briefing: AnalystBriefing | None
    evidence_gate: EvidenceGateResult


class PlanExecuteResult(BaseModel):
    status: Literal[
        "completed",
        "plan_blocked",
        "execution_stopped",
        "replan_budget_exhausted",
        "insufficient_evidence",
        "provider_error",
    ]
    initial_plan: ResearchPlan
    final_steps: tuple[PlanStep, ...]
    observations: tuple[ResearchObservation, ...]
    trajectory: tuple[TrajectoryEvent, ...]
    replan_count: int = Field(ge=0)
    evidence_gate: EvidenceGateResult
    briefing: AnalystBriefing | None = None


class ResearchExecutor(Protocol):
    catalog: tuple[PlannerToolSpec, ...]

    async def execute(
        self,
        step: PlanStep,
        *,
        attempt_id: int,
        plan_revision: int,
    ) -> ResearchObservation:
        raise NotImplementedError


def build_plan_execute_graph(
    *,
    executor: ResearchExecutor,
    planner: PlannerPolicy,
    replanner: ReplannerPolicy,
    report_writer: ReportPolicy,
    max_steps: int = 6,
    max_replans: int = 1,
) -> Any:
    workflow = StateGraph(PlanExecuteState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("plan_gate", plan_gate_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("replanner", replanner_node)
    workflow.add_node("evidence_gate", evidence_gate_node)
    workflow.add_node("report", report_node)
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "plan_gate")
    workflow.add_conditional_edges(
        "plan_gate",
        route_after_plan_gate,
        {"execute": "executor", "stop": END},
    )
    workflow.add_edge("executor", "replanner")
    workflow.add_conditional_edges(
        "replanner",
        route_after_replanning,
        {
            "execute": "executor",
            "evidence_gate": "evidence_gate",
            "stop": END,
        },
    )
    workflow.add_conditional_edges(
        "evidence_gate",
        route_after_evidence_gate,
        {"report": "report", "stop": END},
    )
    workflow.add_edge("report", END)
    return workflow.compile()


async def run_plan_execute(
    *,
    question: str,
    executor: ResearchExecutor,
    planner: PlannerPolicy,
    replanner: ReplannerPolicy,
    report_writer: ReportPolicy,
    max_steps: int = 6,
    max_replans: int = 1,
) -> PlanExecuteResult:
    graph = build_plan_execute_graph(
        executor=executor,
        planner=planner,
        replanner=replanner,
        report_writer=report_writer,
        max_steps=max_steps,
        max_replans=max_replans,
    )
    final_state = await graph.ainvoke(
        {
            "question": question,
            "catalog": executor.catalog,
            "observations": (),
            "trajectory": (),
            "plan_revision": 0,
            "replan_count": 0,
        }
    )
    return PlanExecuteResult(
        status=final_state["status"],
        initial_plan=final_state["initial_plan"],
        final_steps=final_state["active_steps"],
        observations=tuple(final_state.get("observations", ())),
        trajectory=tuple(final_state.get("trajectory", ())),
        replan_count=final_state.get("replan_count", 0),
        evidence_gate=final_state.get(
            "evidence_gate",
            EvidenceGateResult(passed=False, coverage={}, missing_requirements=("not run",)),
        ),
        briefing=final_state.get("briefing"),
    )
```

Implement the referenced node and routing functions as closures inside
`build_plan_execute_graph()` so they can use the injected executor and policies without
placing non-serializable runtime objects inside graph state.

Use these exact node contracts:

```python
async def planner_node(state: PlanExecuteState) -> dict[str, Any]:
    plan = await planner(state["question"], state["catalog"])
    return {
        "initial_plan": plan,
        "active_steps": plan.steps,
        "all_step_ids": tuple(step.step_id for step in plan.steps),
        "current_index": 0,
        "status": "planning",
    }


def plan_gate_node(state: PlanExecuteState) -> dict[str, Any]:
    try:
        checked = validate_plan(state["initial_plan"], state["catalog"], max_steps=max_steps)
    except ValueError as error:
        trajectory = tuple(state.get("trajectory", ()))
        event = TrajectoryEvent(
            index=len(trajectory) + 1,
            phase="guardrail",
            status="blocked",
            summary=str(error),
        )
        return {"status": "plan_blocked", "trajectory": trajectory + (event,)}
    return {"active_steps": checked.steps, "status": "executing"}


async def executor_node(state: PlanExecuteState) -> dict[str, Any]:
    step = state["active_steps"][state["current_index"]]
    observation = await executor.execute(
        step,
        attempt_id=len(state.get("observations", ())) + 1,
        plan_revision=state.get("plan_revision", 0),
    )
    return {
        "observations": tuple(state.get("observations", ())) + (observation,),
        "current_index": state["current_index"] + 1,
    }
```

`replanner_node()` must validate a replacement before updating state, append new IDs to
`all_step_ids`, set `current_index` to the executed-prefix length, and increment
`plan_revision` plus `replan_count` only for `replace_remaining`. `evidence_gate_node()`
calls `evaluate_evidence_gate()`. `report_node()` calls the report policy only after a
passed gate. Routing functions return only `execute`, `evidence_gate`, `report`, or `stop`
from the state status and current index.

The compiled graph contains `planner`, `plan_gate`, `executor`, `replanner`, `evidence_gate`, and `report` nodes. `executor` performs one call per visit. `replanner` routes to the next execution, replacement validation, evidence gate, or typed stop.

- [ ] **Step 4: Enforce append-only successful work and budgets**

Add tests for:

```python
assert duplicate_successful_call_result.status == "plan_blocked"
assert excessive_step_result.status == "plan_blocked"
assert second_replan_result.status == "replan_budget_exhausted"
assert missing_document_result.status == "insufficient_evidence"
assert missing_document_result.briefing is None
```

Implement call signatures as normalized `(capability, canonical JSON arguments)` pairs. Reject a replacement tail that repeats any successful signature. Count a revision only when `action="replace_remaining"`; a normal `continue` review does not consume the revision budget.

- [ ] **Step 5: Add safe trajectory serialization**

Require sequential event indexes and phases from this set:

```text
planning
policy
execution
replanning
evidence_gate
report
guardrail
```

Test that `json.dumps(result.model_dump(mode="json"))` contains no `OPENAI_API_KEY`, `StdioServerParameters`, `sys.executable`, or environment mapping.

- [ ] **Step 6: Run graph, pure-contract, and lint checks**

```bash
.venv/bin/pytest -q tests/test_research_planning.py tests/test_plan_execute_graph.py
.venv/bin/ruff check src/finai_academy/research_planning.py src/finai_academy/plan_execute_graph.py tests/test_research_planning.py tests/test_plan_execute_graph.py
```

Expected: graph completes the maintained revision path, budget failures stop cleanly, and lint passes.

- [ ] **Step 7: Commit the bounded graph**

```bash
git add src/finai_academy/plan_execute_graph.py tests/test_plan_execute_graph.py
git commit -m "feat: add bounded plan-execute research graph"
```

---

### Task 4: Add deterministic and live provider policies

**Files:**
- Create: `src/finai_academy/plan_execute_policies.py`
- Create: `tests/test_plan_execute_policies.py`

**Interfaces:**
- Consumes: `Settings`, `create_chat_model()`, Task 1 contracts, and Task 3 policy protocols.
- Produces: `recorded_planner()`, `recorded_replanner()`, `recorded_report_writer()`, `LivePlanner`, `LiveReplanner`, `LiveReportWriter`, and `build_live_plan_execute_policies()`.

- [ ] **Step 1: Write failing deterministic policy tests**

Assert the maintained plan and revision:

```python
plan = asyncio.run(recorded_planner(MISSION, catalog))
assert [step.step_id for step in plan.steps] == [1, 2, 3, 4]
assert plan.steps[2].arguments == {"ticker": "NVDA", "metric": "Revenue"}

failed_observation = ResearchObservation(
    attempt_id=3,
    step_id=3,
    plan_revision=0,
    capability="get_company_metric",
    arguments={"ticker": "NVDA", "metric": "Revenue"},
    status="error",
    result={"valid_values": ["EPS", "P/E"]},
    error_code="unsupported_metric",
    duration_ms=1.0,
)
decision = asyncio.run(
    recorded_replanner(
        {
            "observations": (failed_observation,),
            "active_steps": INITIAL_RECORDED_STEPS,
            "current_index": 3,
        }
    )
)
assert decision.action == "replace_remaining"
assert [step.step_id for step in decision.replacement_steps] == [5, 6]
assert decision.replacement_steps[0].capability == "search_financial_documents"
assert decision.replacement_steps[0].arguments["company"] == "NVIDIA"
```

Require the recorded report to contain facts for both companies, at least two source references, and explicit limitations for period, currency, and business-mix differences.

- [ ] **Step 2: Run policy tests and confirm RED**

```bash
.venv/bin/pytest -q tests/test_plan_execute_policies.py
```

Expected: collection fails because `plan_execute_policies.py` does not exist.

- [ ] **Step 3: Implement the deterministic classroom policies**

Create the exact initial plan:

```python
MISSION = (
    "Produce a concise NVIDIA and Schneider Electric briefing. Compare their available "
    "valuation metrics and latest operating-growth evidence. Cite every factual claim "
    "and state which observations cannot be compared directly."
)


INITIAL_RECORDED_STEPS = (
    PlanStep(
        step_id=1,
        capability="get_company_metric",
        arguments={"ticker": "NVDA", "metric": "P/E"},
        purpose="Collect NVIDIA valuation evidence.",
        expected_evidence=("NVIDIA P/E",),
    ),
    PlanStep(
        step_id=2,
        capability="get_company_metric",
        arguments={"ticker": "SU.PA", "metric": "P/E"},
        purpose="Collect Schneider Electric valuation evidence.",
        expected_evidence=("Schneider Electric P/E",),
    ),
    PlanStep(
        step_id=3,
        capability="get_company_metric",
        arguments={"ticker": "NVDA", "metric": "Revenue"},
        purpose="Attempt to collect NVIDIA revenue as a structured metric.",
        expected_evidence=("NVIDIA revenue",),
        depends_on=(1,),
    ),
    PlanStep(
        step_id=4,
        capability="search_financial_documents",
        arguments={"company": "Schneider Electric", "query": "revenue growth", "top_k": 2},
        purpose="Collect Schneider Electric operating evidence.",
        expected_evidence=("Schneider Electric revenue growth",),
        depends_on=(2,),
    ),
)


RECORDED_REPLACEMENT_STEPS = (
    PlanStep(
        step_id=5,
        capability="search_financial_documents",
        arguments={"company": "NVIDIA", "query": "revenue growth", "top_k": 2},
        purpose="Replace the unsupported metric with NVIDIA document evidence.",
        expected_evidence=("NVIDIA revenue growth",),
        depends_on=(1,),
    ),
    PlanStep(
        step_id=6,
        capability="search_financial_documents",
        arguments={"company": "Schneider Electric", "query": "energy management", "top_k": 2},
        purpose="Collect Schneider Electric operating-growth evidence.",
        expected_evidence=("Schneider Electric Energy Management growth",),
        depends_on=(2,),
    ),
)
```

When `unsupported_metric` appears for step 3, return replacement steps 5 and 6 for NVIDIA `revenue growth` and Schneider Electric `energy management`, both using `top_k=2`. When all active steps have successful observations, return `finish`.

Use these exact public signatures:

```python
async def recorded_planner(
    question: str, catalog: tuple[PlannerToolSpec, ...]
) -> ResearchPlan:
    return ResearchPlan(goal=question, steps=INITIAL_RECORDED_STEPS)


async def recorded_replanner(state: Mapping[str, Any]) -> ReplanDecision:
    observations = tuple(state.get("observations", ()))
    last = observations[-1] if observations else None
    if last is not None and last.error_code == "unsupported_metric":
        return ReplanDecision(
            action="replace_remaining",
            reasoning="Use document search because Revenue is not a supported metric.",
            replacement_steps=RECORDED_REPLACEMENT_STEPS,
            limitations=("Revenue evidence comes from documents, not the metric snapshot.",),
        )
    if int(state.get("current_index", 0)) >= len(tuple(state.get("active_steps", ()))):
        return ReplanDecision(
            action="finish",
            reasoning="Every active research step has been attempted.",
        )
    return ReplanDecision(
        action="continue",
        reasoning="Continue with the next validated research step.",
    )


async def recorded_report_writer(
    question: str, observations: tuple[ResearchObservation, ...]
) -> AnalystBriefing:
    successful = tuple(item for item in observations if item.status == "ok")
    sources = tuple(
        dict.fromkeys(source for item in successful for source in item.source_references)
    )
    return briefing_from_verified_observations(question, successful, sources)
```

Define `RECORDED_REPLACEMENT_STEPS` with step IDs 5 and 6 and implement
`briefing_from_verified_observations()` by extracting only successful metric values and
document-hit text. It must add the three maintained limitations: different currencies,
different reporting periods, and different business mixes.

- [ ] **Step 4: Write fake-model live policy tests**

Inject a `model_factory` returning a fake object whose `with_structured_output(schema)` records the requested schema and whose `ainvoke(payload)` returns a valid instance. Assert:

```python
planner, replanner, writer = build_live_plan_execute_policies(
    Settings(provider="ollama"),
    model_factory=fake_model_factory,
)
assert isinstance(await planner(MISSION, catalog), ResearchPlan)
assert isinstance(await replanner(replan_state), ReplanDecision)
assert isinstance(await writer(MISSION, observations), AnalystBriefing)
assert fake_model_factory.calls == 3
```

Also assert that prompts contain only the mission, planner-safe catalog, typed error summaries, successful observations, and public source references. They must not contain environment variables or subprocess configuration.

- [ ] **Step 5: Implement provider-neutral live policies**

Each class creates the shared model lazily and calls `with_structured_output()` with the relevant Pydantic schema. Planner and replanner prompts define maximum steps, allowlisted catalog, no investment advice, and observable rationale rather than hidden chain-of-thought. The report prompt requires every fact to map to a supplied source reference and requires explicit limitations.

Use the existing boundary exactly:

```python
def build_live_plan_execute_policies(
    settings: Settings,
    *,
    model_factory: Callable[[Settings], Any] = create_chat_model,
) -> tuple[LivePlanner, LiveReplanner, LiveReportWriter]:
    return (
        LivePlanner(settings, model_factory=model_factory),
        LiveReplanner(settings, model_factory=model_factory),
        LiveReportWriter(settings, model_factory=model_factory),
    )
```

- [ ] **Step 6: Run policy and provider regression tests**

```bash
.venv/bin/pytest -q tests/test_plan_execute_policies.py tests/test_providers.py tests/test_settings.py
.venv/bin/ruff check src/finai_academy/plan_execute_policies.py tests/test_plan_execute_policies.py
```

Expected: recorded and fake-live policies pass without a network, Ollama daemon, or OpenAI key.

- [ ] **Step 7: Commit policy adapters**

```bash
git add src/finai_academy/plan_execute_policies.py tests/test_plan_execute_policies.py
git commit -m "feat: add offline and live research policies"
```

---

### Task 5: Build and execute the visual Lesson 11 notebook

**Files:**
- Create: `tests/test_lesson11_assets.py`
- Create: `scripts/build_lesson11_notebook.py`
- Create: `notebooks/11_plan_and_execute_analyst.ipynb`
- Modify: `tests/test_course_manifest.py`

**Interfaces:**
- Consumes: `FinancialMcpPlanningExecutor`, `run_plan_execute()`, recorded policies, live policy builder, `Settings`, and the existing notebook executor.
- Produces: a 24-to-28-cell output-free notebook with at least six PNG visuals and the exact marker `LESSON_11_PASS`.

- [ ] **Step 1: Write failing notebook asset tests**

Require:

```python
assert notebook.metadata["finai"]["expected_runtime_minutes"] == 40
assert all(not cell.get("outputs") for cell in notebook.cells if cell.cell_type == "code")
assert all(cell.get("execution_count") is None for cell in notebook.cells if cell.cell_type == "code")
assert len({cell.id for cell in notebook.cells}) == len(notebook.cells)
assert 24 <= len(notebook.cells) <= 28
assert nbformat.writes(_build_notebook()) == nbformat.writes(notebook)
```

Require headings `Learning objectives`, `Where this fits`, `Failure lab`, `Verification`, `Knowledge check`, `Challenge`, `Capstone integration`, and `Recap`. Require markers `ResearchPlan`, `ReplanDecision`, `FinancialMcpPlanningExecutor`, `unsupported_metric`, `FINAI_LIVE_MODE`, `Ollama`, `OpenAI`, `Lesson 12`, and `LESSON_11_PASS`.

The offline execution test must require at least six PNG outputs, a real MCP server label, the offline-route qualification, one revision, a passed evidence gate, and one `LESSON_11_PASS` marker.

- [ ] **Step 2: Run notebook tests and confirm RED**

```bash
.venv/bin/pytest -q tests/test_lesson11_assets.py -k notebook
```

Expected: failure because the notebook and builder are absent.

- [ ] **Step 3: Implement the stable notebook builder**

Follow the stable-cell pattern used by Lessons 08-10. Build this exact sequence:

1. title, 60-minute slot, prerequisite, outcome, and read-only boundary;
2. objectives and cumulative architecture position;
3. provider-neutral setup and explicit offline label;
4. Figure 1: workflow, ReAct, and plan-and-execute comparison;
5. open one `FinancialMcpPlanningExecutor` and display discovered permitted tools;
6. inspect the four Pydantic contracts;
7. Figure 2: initial plan dependency map;
8. display plan-policy validation;
9. Figure 3: six-node graph and model-versus-host ownership;
10. execute the first three attempts and expose `unsupported_metric`;
11. Figure 4: execution timeline with status, duration, and evidence count;
12. display the typed replan decision;
13. Figure 5: initial tail, rejected step, and replacement tail;
14. finish corrected execution without repeating successful calls;
15. Figure 6: company-by-evidence coverage matrix;
16. run the evidence gate and display the structured briefing;
17. optional live Ollama or OpenAI route through the same graph;
18. trajectory summary and Lesson 12 field handoff;
19. verification assertions and `LESSON_11_PASS`;
20. knowledge check, challenge, capstone integration, and recap.

Use a single async function per full run so the MCP context is opened and closed once:

```python
async def run_lesson11(*, live_mode: bool) -> PlanExecuteResult:
    settings = Settings.from_environment()
    if live_mode:
        planner, replanner, report_writer = build_live_plan_execute_policies(settings)
    else:
        planner = recorded_planner
        replanner = recorded_replanner
        report_writer = recorded_report_writer
    async with FinancialMcpPlanningExecutor() as executor:
        return await run_plan_execute(
            question=MISSION,
            executor=executor,
            planner=planner,
            replanner=replanner,
            report_writer=report_writer,
        )
```

- [ ] **Step 4: Generate, validate, execute, and inspect all six visuals**

```bash
.venv/bin/python scripts/build_lesson11_notebook.py
.venv/bin/python scripts/validate_notebooks.py notebooks/11_plan_and_execute_analyst.ipynb
.venv/bin/python scripts/execute_notebooks.py notebooks/11_plan_and_execute_analyst.ipynb --mode offline --output-dir /private/tmp/finai-lesson11-offline
```

Expected: source validation passes, execution uses the real local MCP server, six or more PNGs render, evidence gate passes, and `LESSON_11_PASS` appears exactly once. Inspect every PNG at full size and correct crowding, clipped labels, weak contrast, or ambiguous ownership.

- [ ] **Step 5: Add manifest and regression coverage**

Extend `tests/test_course_manifest.py` with:

```python
def test_implemented_lesson_eleven_assets_exist() -> None:
    manifest = load_course_manifest()
    lesson = next(item for item in manifest["lessons"] if item["id"] == "11")
    assert lesson["title"] == "Plan-and-execute financial analyst"
    assert lesson["start"] == "13:30"
    assert lesson["end"] == "14:30"
    assert (ROOT / lesson["notebook"]).is_file()
```

Run:

```bash
.venv/bin/pytest -q tests/test_lesson11_assets.py -k notebook tests/test_course_manifest.py tests/test_research_planning.py tests/test_planning_mcp_executor.py tests/test_plan_execute_graph.py tests/test_plan_execute_policies.py
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit the notebook increment**

```bash
git add scripts/build_lesson11_notebook.py notebooks/11_plan_and_execute_analyst.ipynb tests/test_lesson11_assets.py tests/test_course_manifest.py
git commit -m "lesson: add visual plan-execute analyst notebook"
```

---

### Task 6: Write the instructor chapter and expose Lesson 11 in indexes

**Files:**
- Create: `chapters/11-plan-and-execute-analyst.md`
- Modify: `chapters/README.md`
- Modify: `notebooks/README.md`
- Modify: `decks/README.md`
- Modify: `README.md`
- Modify: `tests/test_lesson11_assets.py`

**Interfaces:**
- Consumes: exact notebook cells, output markers, diagrams, and failure route from Task 5.
- Produces: the complete 13:30-14:30 instructor route and discoverable Lesson 11 links.

- [ ] **Step 1: Extend failing chapter and index tests**

Require the chapter to contain:

```text
13:30-14:30
12-minute concept deck
40-minute notebook
8-minute verification and debrief
get_company_metric
search_financial_documents
unsupported_metric
replace_remaining
evidence gate
Ollama
OpenAI
No-network fallback
Skip if late
Lesson 12
LESSON_11_PASS
```

Require exact Lesson 11 paths in chapter, notebook, deck, and root indexes. Reject stale text claiming Lessons 11-12 are both planned.

- [ ] **Step 2: Run chapter tests and confirm RED**

```bash
.venv/bin/pytest -q tests/test_lesson11_assets.py -k "chapter or discoverable"
```

Expected: failure because the chapter and links are absent.

- [ ] **Step 3: Write the complete instructor chapter**

Include:

- lesson purpose, prerequisites, and exact `uv` startup commands;
- offline, Ollama, and OpenAI execution commands;
- a 12-minute slide script;
- a 40-minute notebook script mapped to stable cell IDs and expected visuals;
- an 8-minute verification and debrief;
- exact expected initial plan, `Revenue` failure, replacement IDs, observations, evidence gate, and report sections;
- the difference between Lesson 09 same-tool recovery and Lesson 11 strategy revision;
- recovery for missing MCP SDK, subprocess import, empty discovery, invalid live output, and insufficient evidence;
- static plan, graph, and evidence matrices for no-model or no-network fallback;
- a skip-if-late route that retains MCP discovery, the failed step, plan replacement, and evidence gate;
- knowledge-check answer key and engineering challenge guidance;
- read-only safety and no-investment-advice boundary; and
- the exact Lesson 12 trajectory and answer fields.

- [ ] **Step 4: Update indexes and validate the instructor route**

Add Lesson 11 links and state that Lessons 08-11 are ready for instructor-led testing while Lesson 12 remains planned. Run:

```bash
.venv/bin/pytest -q tests/test_lesson11_assets.py -k "chapter or discoverable"
.venv/bin/ruff check .
```

Expected: chapter and index tests pass; no stale Lesson 11 status remains.

- [ ] **Step 5: Commit instructor materials**

```bash
git add chapters/11-plan-and-execute-analyst.md chapters/README.md notebooks/README.md decks/README.md README.md tests/test_lesson11_assets.py
git commit -m "docs: add lesson 11 instructor route"
```

---

### Task 7: Create and visually certify the nine-slide deck

**Files:**
- Create: `decks/11-plan-and-execute-analyst.pptx`
- Modify: `tests/test_lesson11_assets.py`
- Use ignored QA workspace: `.artifacts/lesson11-deck/`

**Interfaces:**
- Consumes: Lesson 11 chapter language and the visual system from `decks/10-financial-mcp.pptx`.
- Produces: exactly nine sourced slides with the required footer, original diagrams, and no overflow.

- [ ] **Step 1: Invoke the presentation workflow and add failing deck tests**

Read the presentation skill completely before editing the deck. Add tests requiring exactly nine slide XML parts and nine notes parts, the exact footer nine times, `[Sources]` blocks, no visible em dash, and these case-insensitive markers:

```text
Plan-and-Execute Financial Analyst
REACT
PLAN
EXECUTE
REPLAN
REPORT
HOST POLICY
MCP DISCOVERY
unsupported_metric
EVIDENCE GATE
LESSON 12
```

- [ ] **Step 2: Run the deck test and confirm RED**

```bash
.venv/bin/pytest -q tests/test_lesson11_assets.py -k deck
```

Expected: failure because `decks/11-plan-and-execute-analyst.pptx` is absent.

- [ ] **Step 3: Build the deck from the Lesson 10 template**

Use the exact nine-slide narrative from the spec:

1. `One mission needs several evidence steps` - NVIDIA and Schneider Electric mission.
2. `Plan-and-execute separates strategy from action` - workflow, ReAct, plan-execute comparison.
3. `The model proposes; the host controls` - ownership and trust boundary.
4. `A typed plan makes research inspectable` - step schema and dependency example.
5. `Four roles share one bounded state` - planner, executor, replanner, report writer.
6. `MCP discovery becomes a permitted catalog` - Lesson 10 bridge.
7. `Replanning replaces only unfinished work` - failed Revenue metric and replacement document search.
8. `The evidence gate prevents fluent incompleteness` - two-company coverage matrix.
9. `Lesson 12 evaluates path and answer` - trajectory and briefing layers.

Use original vector diagrams and short English copy. Every notes block cites the Lesson 11 chapter plus the relevant official LangGraph, MCP, Plan-and-Solve, or MLExpert inspiration source.

- [ ] **Step 4: Render and inspect every slide**

Render the deck, create a montage, inspect the montage, then inspect all nine full-size slide PNGs. Correct any collision, clipping, low contrast, awkward wrap, unexplained acronym, inconsistent arrow direction, or dense paragraph.

- [ ] **Step 5: Run automated deck QA**

Run the current presentation runtime's `slides_test.py`, the template-plan check, the template-fidelity check against Lesson 10, and:

```bash
.venv/bin/pytest -q tests/test_lesson11_assets.py -k deck
```

Expected: zero overflow, nine slides, nine source-note blocks, exact footer, all required markers, and preserved template geometry.

- [ ] **Step 6: Commit the certified deck**

```bash
git add decks/11-plan-and-execute-analyst.pptx tests/test_lesson11_assets.py
git commit -m "docs: add lesson 11 plan-execute deck"
```

---

### Task 8: Run live providers, full regression, and record readiness

**Files:**
- Create: `docs/reviews/lesson-11-readiness.md`
- Modify: only files required to fix failures discovered by verification.

**Interfaces:**
- Consumes: all Lesson 11 code, notebook, chapter, deck, tests, and existing course validation scripts.
- Produces: an evidence-based readiness decision, weighted score, clean branch, and Lesson 12 handoff.

- [ ] **Step 1: Execute the complete targeted package**

```bash
.venv/bin/pytest -q tests/test_research_planning.py tests/test_planning_mcp_executor.py tests/test_plan_execute_graph.py tests/test_plan_execute_policies.py tests/test_lesson11_assets.py tests/test_course_manifest.py
.venv/bin/ruff check .
.venv/bin/python scripts/validate_notebooks.py notebooks/11_plan_and_execute_analyst.ipynb
```

Expected: all targeted tests, Ruff, and notebook validation pass.

- [ ] **Step 2: Execute and inspect the offline notebook**

```bash
.venv/bin/python scripts/execute_notebooks.py notebooks/11_plan_and_execute_analyst.ipynb --mode offline --output-dir /private/tmp/finai-lesson11-offline
```

Record runtime, provider label, MCP server name, attempted step IDs, one revision, evidence-gate result, PNG count, and `LESSON_11_PASS`. Inspect all six visuals at full size.

- [ ] **Step 3: Execute the configured Ollama route**

When Ollama and `qwen3:8b` are available:

```bash
FINAI_LIVE_MODE=1 FINAI_MODEL_PROVIDER=ollama FINAI_CHAT_MODEL=qwen3:8b \
.venv/bin/python scripts/execute_notebooks.py notebooks/11_plan_and_execute_analyst.ipynb --mode live --provider ollama --output-dir /private/tmp/finai-lesson11-ollama
```

Require a validated plan, allowlisted tool calls, bounded completion or an honestly recorded typed provider failure, and no secret or hidden-reasoning output.

- [ ] **Step 4: Execute OpenAI only when configured**

If `OPENAI_API_KEY` is present:

```bash
FINAI_LIVE_MODE=1 FINAI_MODEL_PROVIDER=openai FINAI_CHAT_MODEL=gpt-5-mini \
.venv/bin/python scripts/execute_notebooks.py notebooks/11_plan_and_execute_analyst.ipynb --mode live --provider openai --output-dir /private/tmp/finai-lesson11-openai
```

If the key is absent, record `NOT CONFIGURED`; do not infer a pass from source code or settings tests.

- [ ] **Step 5: Run deck QA and full repository regression**

Run the presentation overflow, template-plan, and template-fidelity checks, then:

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/python scripts/validate_repo.py
git diff --check
git status --short
```

Expected: the complete repository passes, deck QA is clean, structure validation passes, and the worktree contains only the readiness report before its commit.

- [ ] **Step 6: Write the evidence-based readiness report**

Record exact commands and results under:

```text
Scope
Environment
Unit and integration tests
Offline notebook
Ollama live route
OpenAI live route
Notebook visual review
Deck automated and visual review
Instructor timing and fallback review
Known qualifications
Weighted score
Decision
```

Use the established weighting: learner usability 25%, technical correctness and safety 20%, conceptual progression 20%, live delivery 15%, visuals 10%, repository quality 10%. Do not award provider evidence that was not executed.

- [ ] **Step 7: Commit readiness evidence and reverify the exact commit**

```bash
git add docs/reviews/lesson-11-readiness.md
git commit -m "docs: certify lesson 11 readiness"
git show --check HEAD
.venv/bin/pytest -q tests/test_research_planning.py tests/test_planning_mcp_executor.py tests/test_plan_execute_graph.py tests/test_plan_execute_policies.py tests/test_lesson11_assets.py tests/test_course_manifest.py
git status --short
```

Expected: the committed report matches observed evidence, focused tests pass on the exact commit, `git show --check` is clean, and the worktree is clean.
