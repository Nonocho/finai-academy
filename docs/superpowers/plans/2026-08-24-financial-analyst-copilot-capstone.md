# Financial Analyst Copilot Capstone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and certify a complete offline-first Streamlit Financial Analyst Copilot reference application, then derive a bounded student integration challenge that can be completed alone or in pairs in 60 minutes.

**Architecture:** A thin Streamlit layer calls a typed `FinancialAnalystCopilot` application service. The service composes the existing hybrid retrieval, local financial MCP capabilities, bounded plan-and-execute policies, deterministic evaluation, and MLflow persistence without duplicating their algorithms. A recorded provider and certified repository fixtures guarantee the full NVIDIA versus Schneider Electric mission without network access; Ollama, OpenAI, live market data, and Tavily are explicit optional enrichments.

**Tech Stack:** Python 3.11+, Pydantic 2, Streamlit, existing FinAI Academy retrieval/MCP/agent/evaluation modules, MLflow 3.15+, pytest, Streamlit AppTest, uv.

**Spec:** `docs/superpowers/specs/2026-08-24-financial-analyst-copilot-capstone-design.md`

## Global Constraints

- The mandatory UI is Streamlit; FastAPI, SSE, authentication, deployment, persistent chat memory, and multi-agent supervisors are non-goals.
- The fixed reference mission compares exactly NVIDIA and Schneider Electric and cites every factual claim.
- The certified route must run with no network, API key, or Ollama process.
- Providers are explicit: `recorded`, `ollama`, or `openai`; no silent fallback is permitted.
- Data modes are explicit: `certified` or `live_enrichment`; live data never silently replaces certified facts.
- The mandatory capability registry contains only `search_financial_documents` and `get_company_metric`.
- Agent ceilings are `max_steps <= 6` and `max_replans <= 1`; duplicate successful tool signatures are forbidden.
- The unsupported NVIDIA `Revenue` metric must produce a typed error and one bounded replan to document search.
- A failed evidence gate produces a typed stop and no briefing.
- Public state must contain no credentials, private reasoning, raw clients, model objects, or personal filesystem paths.
- Deterministic release uses exactly five metrics: tool-call correctness, tool-call efficiency, answer relevance, answer completeness, and citation integrity.
- Optional LLM-judge results remain separate and never change deterministic release status.
- The student starter must launch before completion, expose four bounded integration seams, and print exactly one `CAPSTONE_PASS` only after all public contracts pass.
- Every user-visible page uses the footer `First Finance - Arnaud Demes`.
- Use TDD for production behavior: failing focused test, observed failure, minimal implementation, passing focused test, then commit.

---

## File Structure

### Core application

- `src/finai_academy/capstone/models.py`: strict public request, evidence, trace, briefing, evaluation, and run-result contracts while retaining the Module 00 models.
- `src/finai_academy/capstone/tools.py`: company-filtered certified retrieval and allowlisted financial capability adapters.
- `src/finai_academy/capstone/live_news.py`: optional Tavily adapter with typed unavailable/error states.
- `src/finai_academy/capstone/service.py`: orchestration, recorded reference route, evidence gate, briefing assembly, evaluation, and optional live-provider boundary.
- `src/finai_academy/capstone/views.py`: conversion from strict domain results to safe serializable UI view models.
- `src/finai_academy/capstone/streamlit_ui.py`: shared Streamlit rendering that accepts a service factory.
- `src/finai_academy/capstone/__init__.py`: stable public exports.

### Capstone products

- `final-project/shared/reference_mission.json`: versioned mission, expected companies, capability path, and certified display copy.
- `final-project/reference/streamlit_app.py`: thin complete-correction entry point.
- `final-project/reference/README.md`: correction run and architecture guide.
- `final-project/student/integration.py`: four intentionally incomplete public integration functions.
- `final-project/student/streamlit_app.py`: thin student entry point that launches before seams are completed.
- `final-project/student/verify.py`: public verifier that prints exactly one pass marker on success.
- `final-project/student/README.md`: student run instructions and constraints.
- `final-project/student/CHECKLIST.md`: timeboxed completion checklist.
- `final-project/STUDENT_BRIEF.md`: challenge handout.
- `final-project/INSTRUCTOR_GUIDE.md`: timing, hints, expected outputs, diagnostic, and recovery route.
- `final-project/README.md`: canonical capstone landing page.

### Tests and certification

- `tests/test_capstone_models.py`: request limits, provenance, public-state, and result invariants.
- `tests/test_capstone_tools.py`: retrieval separation, evidence IDs, capability intersection, and optional news degradation.
- `tests/test_capstone_service.py`: reference route, replan, evidence stop, evaluation, and provider behavior.
- `tests/test_capstone_views.py`: safe public transformation and judge/release separation.
- `tests/test_capstone_streamlit.py`: AppTest success and typed-stop journeys.
- `tests/test_capstone_student.py`: starter shape, four contract groups, and exact pass-marker rules.
- `tests/test_capstone_docs.py`: commands, timing, footer, macOS/Windows, and recovery-route documentation.
- `scripts/certify_capstone.py`: deterministic end-to-end certification command and artifact writer.
- `artifacts/capstone/`: generated certification JSON, screenshot, and readiness report; no credentials or machine-specific paths.

---

### Task 1: Public contracts and certified reference mission

**Files:**
- Modify: `src/finai_academy/capstone/models.py`
- Modify: `src/finai_academy/capstone/__init__.py`
- Create: `final-project/shared/reference_mission.json`
- Create: `tests/test_capstone_models.py`

**Interfaces:**
- Consumes: existing `EvidenceType`, `FindingCategory`, `AnalystFinding`, and `AnalystBrief` contracts.
- Produces: `ResearchMode`, `CapstoneProvider`, `DataMode`, `RunStatus`, `ResearchRequest`, `CapstoneEvidenceHit`, `CitedFact`, `CapstoneBriefing`, `EvidenceGateDecision`, `PublicTraceEvent`, `MetricEvaluation`, `DeterministicEvaluation`, `JudgeEvaluation`, and `ResearchRunResult`.

- [ ] **Step 1: Write failing validation tests**

```python
def test_reference_request_locks_company_universe_and_safety_limits() -> None:
    request = ResearchRequest.reference()
    assert request.companies == ("NVIDIA", "Schneider Electric")
    assert request.max_steps == 6
    assert request.max_replans == 1

    with pytest.raises(ValidationError):
        ResearchRequest.reference(max_steps=7)
    with pytest.raises(ValidationError):
        ResearchRequest.reference(max_replans=2)


def test_completed_result_requires_a_passing_gate_and_briefing() -> None:
    with pytest.raises(ValidationError, match="completed run"):
        ResearchRunResult.model_validate(
            make_result_payload(status="completed", evidence_gate={"passed": False}, briefing=None)
        )


def test_cited_fact_requires_matching_document_provenance() -> None:
    with pytest.raises(ValidationError, match="evidence_id"):
        CitedFact(
            claim="Data Center revenue increased.",
            company="NVIDIA",
            provenance_kind="document",
            source_reference="NVIDIA FY2026 results",
            evidence_id=None,
        )
```

- [ ] **Step 2: Run tests and observe the missing-contract failure**

Run: `uv run pytest tests/test_capstone_models.py -q`

Expected: collection fails because the new models are not exported.

- [ ] **Step 3: Implement strict contracts and the reference factory**

Use frozen, `extra="forbid"` Pydantic models. `ResearchRequest.reference()` must set the exact mission text from the spec, companies `(“NVIDIA”, “Schneider Electric”)`, `mode="reference"`, `provider="recorded"`, `data_mode="certified"`, `max_steps=6`, and `max_replans=1`. A model validator must reject any other company tuple in reference mode, steps above six, replans above one, blank questions, and `include_news=True` unless data mode is live enrichment.

`CitedFact` must require `evidence_id` for document facts and must forbid it for metric facts. `ResearchRunResult` must enforce these state relationships:

```python
if self.status == RunStatus.COMPLETED:
    if not self.evidence_gate.passed or self.briefing is None:
        raise ValueError("completed run requires a passing evidence gate and briefing")
if not self.evidence_gate.passed and self.briefing is not None:
    raise ValueError("failed evidence gate cannot expose a briefing")
if self.deterministic_evaluation.release_passed != all(
    metric.value == 1.0 for metric in self.deterministic_evaluation.metrics
):
    raise ValueError("release decision must match deterministic metrics")
```

Add recursive rejection for credential-shaped strings and absolute personal paths in public trace, errors, briefing, and serialized metadata.

`CapstoneEvidenceHit` is the public retrieval record used by every later task. It contains `company`, `text`, `evidence_id`, `document_id`, `section`, `period`, and `source_reference`; all fields are nonblank and no field may contain a personal path or credential-shaped text.

- [ ] **Step 4: Add and validate the mission fixture**

Create JSON with `schema_version: 1`, mission ID `nvidia-schneider-reference-v1`, the exact mission, both companies, provider `recorded`, data mode `certified`, maximums `6` and `1`, and required capabilities in this order:

```json
["get_company_metric", "search_financial_documents"]
```

The tests must load it and assert equality with `ResearchRequest.reference()` rather than maintaining a second mission string.

- [ ] **Step 5: Run focused tests and commit**

Run: `uv run pytest tests/test_capstone_models.py tests/test_capstone_briefing.py -q`

Expected: all pass.

Commit: `feat: define capstone public contracts`

---

### Task 2: Certified retrieval, MCP tool registry, and optional Tavily boundary

**Files:**
- Create: `src/finai_academy/capstone/tools.py`
- Create: `src/finai_academy/capstone/live_news.py`
- Create: `tests/test_capstone_tools.py`

**Interfaces:**
- Consumes: `build_financial_capability_registry()`, `ALLOWED_TOOLS`, `DocumentSearchResult`, `MetricResult`, `IndexedPassage`, `RetrievalFilters`, and existing hybrid retrieval primitives.
- Produces: `CertifiedRetriever.search(company: str, query: str, top_k: int = 2) -> tuple[CapstoneEvidenceHit, ...]`, `AnalystToolRegistry.discover() -> tuple[str, ...]`, `AnalystToolRegistry.invoke(name: str, arguments: Mapping[str, Any]) -> ToolOutcome`, and `TavilyNewsAdapter.search(company: str, query: str) -> NewsSearchOutcome`.

- [ ] **Step 1: Write failing certified-tool tests**

```python
def test_retriever_never_crosses_company_boundary() -> None:
    retriever = build_certified_retriever()
    hits = retriever.search("NVIDIA", "growth revenue", top_k=3)
    assert hits
    assert {hit.company for hit in hits} == {"NVIDIA"}
    assert all(hit.evidence_id and hit.source_reference for hit in hits)


def test_registry_intersects_discovery_with_static_policy() -> None:
    registry = AnalystToolRegistry(
        discovered=("search_financial_documents", "get_company_metric", "place_order")
    )
    assert registry.discover() == ("get_company_metric", "search_financial_documents")
    with pytest.raises(ValueError, match="not allowlisted"):
        registry.invoke("place_order", {})


def test_tavily_without_key_is_typed_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    result = TavilyNewsAdapter.from_environment().search("NVIDIA", "AI demand")
    assert result.status == "unavailable"
    assert result.items == ()
```

- [ ] **Step 2: Run tests and observe missing adapters**

Run: `uv run pytest tests/test_capstone_tools.py -q`

Expected: collection fails because `capstone.tools` and `capstone.live_news` do not exist.

- [ ] **Step 3: Implement certified retrieval and typed tool outcomes**

Build the retriever from the versioned MCP evidence catalog. Convert every catalog record to an `IndexedPassage` while preserving company, period, section, source, document ID, and evidence ID. Search with exact `RetrievalFilters(company=company)` and return public hits containing both source reference and evidence ID.

`AnalystToolRegistry` must sort the intersection of runtime discovery and this immutable set:

```python
MANDATORY_ANALYST_TOOLS = frozenset(
    {"get_company_metric", "search_financial_documents"}
)
```

Unknown, undiscovered, mutation, trading, and code-execution capabilities fail closed. Capability validation errors become `ToolOutcome(status="error", error_code=..., retryable=...)`; they are never converted to success.

- [ ] **Step 4: Implement the optional Tavily adapter**

The adapter accepts an injected callable in tests. Without `TAVILY_API_KEY`, return `status="unavailable"`. On runtime failure, return `status="error"` with the stable message `News enrichment failed; certified analysis remains available.` On success, retain only title, URL, publication date when supplied, provider `tavily`, and an ISO-8601 retrieval timestamp. Do not add the tool to the mandatory registry.

- [ ] **Step 5: Run focused tests and commit**

Run: `uv run pytest tests/test_capstone_tools.py tests/test_hybrid_retrieval.py tests/test_financial_mcp_capabilities.py -q`

Expected: all pass.

Commit: `feat: add certified capstone tools`

---

### Task 3: Bounded application service and deterministic reference run

**Files:**
- Create: `src/finai_academy/capstone/service.py`
- Modify: `src/finai_academy/capstone/__init__.py`
- Create: `tests/test_capstone_service.py`

**Interfaces:**
- Consumes: Task 1 public contracts; Task 2 `CertifiedRetriever`, `AnalystToolRegistry`, and optional news boundary; existing plan, observation, evidence, and agent-evaluation contracts.
- Produces: `FinancialAnalystCopilot.run(request: ResearchRequest) -> ResearchRunResult`, `build_reference_copilot() -> FinancialAnalystCopilot`, and deterministic recorded reference behavior.

- [ ] **Step 1: Write failing reference-journey tests**

```python
def test_recorded_reference_run_is_complete_cited_and_bounded() -> None:
    result = build_reference_copilot().run(ResearchRequest.reference())
    assert result.status == "completed"
    assert result.replan_count == 1
    assert len(result.final_plan) <= 6
    assert result.evidence_gate.passed
    assert result.briefing is not None
    assert {fact.company for fact in result.briefing.cited_facts} == {
        "NVIDIA", "Schneider Electric"
    }
    assert all(fact.source_reference for fact in result.briefing.cited_facts)
    assert all(
        fact.evidence_id for fact in result.briefing.cited_facts
        if fact.provenance_kind == "document"
    )
    assert result.deterministic_evaluation.release_passed


def test_unsupported_revenue_metric_replans_to_document_search() -> None:
    result = build_reference_copilot().run(ResearchRequest.reference())
    errors = [event for event in result.trajectory if event.status == "error"]
    assert [event.error_code for event in errors] == ["unsupported_metric"]
    assert result.initial_plan != result.final_plan
    assert result.replan_count == 1


def test_failed_evidence_gate_returns_no_briefing() -> None:
    service = build_reference_copilot(retriever=MissingSchneiderRetriever())
    result = service.run(ResearchRequest.reference())
    assert result.status == "insufficient_evidence"
    assert not result.evidence_gate.passed
    assert result.briefing is None
```

- [ ] **Step 2: Run tests and observe the absent service**

Run: `uv run pytest tests/test_capstone_service.py -q`

Expected: collection fails because `FinancialAnalystCopilot` is not defined.

- [ ] **Step 3: Implement the deterministic plan and execution loop**

The recorded reference plan must visibly include metric retrieval for both tickers, the intentionally unsupported NVIDIA `Revenue` call, and company-filtered document searches. Execute at most one step per loop and record public trace events with stable attempt IDs, plan revision, duration, status, error code, and failure owner. When `Revenue` returns `unsupported_metric`, replace only the remaining tail with an NVIDIA revenue/growth document search. Reject repeated successful signatures and stop at the hard budgets.

Use a protocol-based constructor so tests inject retrievers, tool registries, clock, run ID factory, and persistence. Do not copy the full Lesson 11 graph; reuse its validation and evaluation helpers where their public contracts fit, and keep capstone-specific adaptation inside `service.py`.

- [ ] **Step 4: Implement the evidence gate and briefing assembly**

The gate passes only when both companies have at least one successful document hit, each document hit has a source reference and evidence ID, and no cited fact points outside the collected evidence. When it fails, return `status="insufficient_evidence"` and `briefing=None`.

The recorded briefing must contain:

- concise executive summary;
- separate NVIDIA and Schneider evidence sections;
- cross-company observations that explicitly state direct comparability limits;
- interpretation separated from reported facts;
- limitations and open questions;
- ordered aggregate sources.

Every factual sentence must exist as a `CitedFact`; prose sections may refer to those facts but must not introduce an uncited number.

- [ ] **Step 5: Implement the five deterministic scores**

Return all five named metrics in the fixed order. The reference run earns `1.0` only when expected calls, budgets, both-company coverage, required sections, and exact source/evidence pairing pass. `release_passed` is true only when all five values equal `1.0`. Judge status defaults to `not_run` and remains separate.

- [ ] **Step 6: Run focused and regression tests, then commit**

Run: `uv run pytest tests/test_capstone_service.py tests/test_plan_execute_graph.py tests/test_agent_evaluation.py -q`

Expected: all pass.

Commit: `feat: orchestrate recorded analyst copilot`

---

### Task 4: Explicit Ollama/OpenAI routes and MLflow evidence persistence

**Files:**
- Modify: `src/finai_academy/capstone/model_gateway.py`
- Modify: `src/finai_academy/capstone/service.py`
- Create: `src/finai_academy/capstone/persistence.py`
- Create: `tests/test_capstone_providers.py`
- Create: `tests/test_capstone_persistence.py`

**Interfaces:**
- Consumes: `Settings`, `create_structured_model`, Task 3 service result, and existing MLflow helpers.
- Produces: `ProviderReadiness`, `build_copilot_for_request(request, settings)`, and `CapstoneRunStore.persist(result) -> PersistedRunReferences`.

- [ ] **Step 1: Write failing provider and persistence tests**

```python
def test_openai_without_key_is_disabled_without_fallback(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    readiness = provider_readiness(provider="openai", model="gpt-5-mini")
    assert not readiness.available
    assert readiness.fallback_provider is None


def test_recorded_route_never_constructs_a_live_model(monkeypatch) -> None:
    monkeypatch.setattr(model_gateway, "create_structured_model", fail_if_called)
    result = build_copilot_for_request(ResearchRequest.reference(), Settings()).run(
        ResearchRequest.reference()
    )
    assert result.provider == "recorded"


def test_persisted_run_and_trace_share_identifiers(tmp_path: Path) -> None:
    store = CapstoneRunStore(tracking_directory=tmp_path)
    refs = store.persist(build_reference_copilot().run(ResearchRequest.reference()))
    assert refs.run_id
    assert refs.trace_id
    assert refs.tracking_uri.startswith("sqlite:")
```

- [ ] **Step 2: Run tests and observe missing readiness/persistence APIs**

Run: `uv run pytest tests/test_capstone_providers.py tests/test_capstone_persistence.py -q`

Expected: missing symbols fail collection.

- [ ] **Step 3: Implement explicit provider readiness and model routing**

Recorded mode always uses deterministic policies. Ollama readiness checks configuration and a short injected health probe; unavailable state says `Start Ollama and run: ollama pull qwen3:4b`. OpenAI requires a nonblank `OPENAI_API_KEY`; the error never includes the key. `build_copilot_for_request` must not substitute a different provider.

Live LLM use is bounded to planning/report wording over collected certified evidence. Tool authorization, evidence gating, provenance, budgets, and deterministic release remain host-controlled. Invalid structured output or provider failure returns a sanitized `provider_error` result.

- [ ] **Step 4: Implement local MLflow persistence**

Use an explicit repository-local SQLite URI under `artifacts/capstone/mlflow/`. Persist provider, model, data mode, mission ID, budgets, final status, five metric values, release decision, sanitized trace JSON, briefing JSON when present, and dataset identities. Return public run and trace identifiers. If MLflow is unavailable, return a typed `unavailable` persistence result for the UI while leaving deterministic analysis intact; certification with the evaluation extra requires persistence success.

- [ ] **Step 5: Run focused tests and commit**

Run: `uv run pytest tests/test_capstone_providers.py tests/test_capstone_persistence.py tests/test_mlflow_agent_evaluation.py -q`

Expected: all pass with the evaluation extra installed; provider tests use fakes and make no network calls.

Commit: `feat: add capstone providers and MLflow evidence`

---

### Task 5: Safe view models and complete Streamlit reference application

**Files:**
- Create: `src/finai_academy/capstone/views.py`
- Create: `src/finai_academy/capstone/streamlit_ui.py`
- Create: `final-project/reference/streamlit_app.py`
- Create: `final-project/reference/README.md`
- Create: `tests/test_capstone_views.py`
- Create: `tests/test_capstone_streamlit.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `ResearchRunResult`, provider readiness, and a zero-argument service factory.
- Produces: `CapstoneRunView`, `to_run_view(result)`, and `render_capstone(service_factory, integration_status=None)`.

- [ ] **Step 1: Add Streamlit as a bounded project dependency**

Add a `capstone` optional dependency group containing `streamlit>=1.40,<2` and `mlflow>=3.15,<4`. Update the lock file with `uv lock` and verify `uv sync --extra capstone --extra ai` resolves.

- [ ] **Step 2: Write failing view and AppTest checks**

```python
def test_view_contains_only_public_serializable_state() -> None:
    view = to_run_view(build_reference_copilot().run(ResearchRequest.reference()))
    payload = view.model_dump_json()
    assert "OPENAI_API_KEY" not in payload
    assert "/Users/" not in payload
    assert "api_key" not in payload.casefold()


def test_reference_app_renders_complete_recorded_journey() -> None:
    app = AppTest.from_file("final-project/reference/streamlit_app.py")
    app.run(timeout=30)
    assert not app.exception
    assert any("Reference mission" in item.label for item in app.tabs)
    app.button(key="run_reference").click().run(timeout=30)
    assert not app.exception
    rendered = " ".join(item.value for item in app.markdown)
    assert "Executive briefing" in rendered
    assert "Citation integrity" in rendered
    assert "Release passed" in rendered
    assert "First Finance - Arnaud Demes" in rendered
```

- [ ] **Step 3: Run tests and observe missing view/UI modules**

Run: `uv run pytest tests/test_capstone_views.py tests/test_capstone_streamlit.py -q`

Expected: collection fails on missing modules or Streamlit dependency.

- [ ] **Step 4: Implement safe view transformation**

Create presentation-focused Pydantic models for readiness, plan rows, tool rows, cited-fact rows, trace rows, score rows, and release state. Convert the strict result; never expose raw exceptions, clients, prompts, credentials, chain-of-thought, or filesystem locations. Format duration, timestamps, units, and status labels in this layer.

- [ ] **Step 5: Implement the shared Streamlit page**

Render:

- sidebar title, provider, model, data mode, Tavily status, safety statement, reset action, and footer;
- tabs `Reference mission` and `Ask the analyst`;
- readiness strip;
- read-only fixed mission and keyed `run_reference` button;
- custom question input bounded to the two-company universe;
- plan, tool activity, typed error and replan, evidence gate, five briefing sections, evidence table, trace expander, five-metric scorecard, optional judge panel, and deterministic release decision.

All success/warning/error copy must be factual and simple. The recorded mode and certified snapshots must be visible on the result. Failed evidence runs must not render the briefing container. Store only serialized public view data and selections in `st.session_state`.

- [ ] **Step 6: Build the thin reference entry point and run tests**

The entry point imports the shared renderer and reference service factory; it contains no business logic.

Run: `uv run pytest tests/test_capstone_views.py tests/test_capstone_streamlit.py -q`

Expected: all pass.

Run: `uv run streamlit run final-project/reference/streamlit_app.py --server.headless true --server.port 8509`

Expected: server reaches a healthy local URL and displays both tabs.

- [ ] **Step 7: Commit**

Commit: `feat: build Streamlit analyst copilot`

---

### Task 6: Student integration scaffold and exact verifier

**Files:**
- Create: `final-project/student/integration.py`
- Create: `final-project/student/streamlit_app.py`
- Create: `final-project/student/verify.py`
- Create: `final-project/student/README.md`
- Create: `final-project/student/CHECKLIST.md`
- Create: `tests/test_capstone_student.py`

**Interfaces:**
- Consumes: certified Task 2–5 functions and the shared Streamlit renderer.
- Produces four student functions: `wire_retriever`, `register_analyst_capabilities`, `evaluate_student_evidence_gate`, and `assemble_public_briefing_view`; plus a verifier whose only success marker is `CAPSTONE_PASS`.

- [ ] **Step 1: Write the public student-contract tests**

Group tests under markers or classes `retriever`, `capabilities`, `evidence_gate`, and `public_view`. Test each seam independently against the complete correction behavior. Add a subprocess verifier test that asserts:

```python
completed = subprocess.run(
    [sys.executable, "final-project/student/verify.py"],
    text=True,
    capture_output=True,
    check=False,
)
assert completed.returncode == 0
assert completed.stdout.splitlines().count("CAPSTONE_PASS") == 1
assert "CAPSTONE_PASS" not in completed.stderr
```

Maintain a separate starter-mode test that expects four named `StudentIntegrationIncomplete` statuses without broken imports or missing data.

- [ ] **Step 2: Derive the correction-backed function bodies**

First implement all four functions with the certified reference calls and confirm the public tests pass. Save the exact completed bodies in the instructor guide’s solution section. Then replace only those four bodies in the committed student file with a typed `StudentIntegrationIncomplete(seam=...)` result that the UI can render. Do not leave syntax errors, missing imports, or dependency failures.

- [ ] **Step 3: Implement the student application and verifier**

The app must launch in recorded mode, show the mission and four seam statuses, and explain the next failing contract without revealing solution code. The verifier runs the four public contract groups, the reference mission, citation integrity, deterministic release, and persistence check. It prints diagnostic lines followed by exactly one marker only if every condition passes; otherwise it exits nonzero and never prints the marker.

- [ ] **Step 4: Prove both starter and solved states**

Run the starter checks and observe exactly four intended incomplete groups. Apply the saved correction bodies in a temporary copy under pytest’s `tmp_path`, run the verifier there, and assert one `CAPSTONE_PASS`. This test must not modify the committed starter.

- [ ] **Step 5: Commit**

Run: `uv run pytest tests/test_capstone_student.py -q`

Expected: all meta-tests pass; the committed starter remains intentionally incomplete but launchable.

Commit: `feat: add bounded capstone student challenge`

---

### Task 7: Student brief, instructor guide, setup, and classroom recovery

**Files:**
- Rewrite: `final-project/README.md`
- Create: `final-project/STUDENT_BRIEF.md`
- Create: `final-project/INSTRUCTOR_GUIDE.md`
- Modify: `final-project/reference/README.md`
- Modify: `final-project/student/README.md`
- Modify: `final-project/student/CHECKLIST.md`
- Create: `tests/test_capstone_docs.py`

**Interfaces:**
- Consumes: final commands, UI labels, provider behavior, student seams, and verification marker from Tasks 1–6.
- Produces: one canonical landing page, printable student instructions, and a complete instructor runbook.

- [ ] **Step 1: Write failing documentation-contract tests**

Assert that the documents contain the exact commands:

```text
uv sync --extra capstone --extra ai
uv run streamlit run final-project/reference/streamlit_app.py
uv run streamlit run final-project/student/streamlit_app.py
uv run python final-project/student/verify.py
```

Also assert the fixed mission, 60-minute student timing, 30-minute demonstration, individual/pair modes, macOS and Windows environment instructions, Ollama model `qwen3:4b`, explicit OpenAI key setup, `.env` handling, certified offline fallback, Tavily optional status, four seams, diagnostic task, hints, solution discussion, and footer.

- [ ] **Step 2: Run tests and observe missing materials**

Run: `uv run pytest tests/test_capstone_docs.py -q`

Expected: fails because the handouts and complete runbook do not exist.

- [ ] **Step 3: Write the canonical README and student brief**

Lead with the learning outcome and the shortest recorded-mode command. Keep language factual and professional. Explain reference versus student folders, the two UI tabs, certified versus live data, OpenAI/Ollama/recorded choices, and that outputs are research support rather than investment advice.

The student brief must show the exact timetable:

- 15:30–15:40 understand mission;
- 15:40–16:10 complete four seams;
- 16:10–16:25 evaluate and diagnose;
- 16:25–16:30 prepare demo;
- 16:30–17:00 demonstration and architecture review.

- [ ] **Step 4: Write the instructor guide**

Include prerequisites, pre-class certification, expected reference output, a minute-by-minute facilitation table, pair rotation, one progressive hint per seam, complete correction bodies, the deliberately regressed diagnostic, MLflow trace inspection, common failures, recorded fallback, skip-if-late route, discussion questions, production non-goals, and a reset procedure that does not delete learner work.

- [ ] **Step 5: Run docs tests and commit**

Run: `uv run pytest tests/test_capstone_docs.py tests/test_onboarding_docs.py -q`

Expected: all pass.

Commit: `docs: complete capstone classroom materials`

---

### Task 8: End-to-end certification, visual evidence, and repository gate

**Files:**
- Create: `scripts/certify_capstone.py`
- Create: `tests/test_capstone_certification.py`
- Create or update generated outputs under: `artifacts/capstone/`
- Modify only if certification exposes a defect: files created in Tasks 1–7, with a focused regression test.

**Interfaces:**
- Consumes: complete reference application, student meta-tests, MLflow store, manifest validation, repository test suite, and Streamlit runtime.
- Produces: `artifacts/capstone/certification.json`, `artifacts/capstone/readiness.md`, and a desktop screenshot with sanitized paths and no secrets.

- [ ] **Step 1: Write the certification contract test**

```python
def test_certification_artifact_records_all_mandatory_gates() -> None:
    payload = json.loads(Path("artifacts/capstone/certification.json").read_text())
    assert payload["schema_version"] == 1
    assert payload["reference_mission"]["status"] == "completed"
    assert payload["reference_mission"]["citation_integrity"] == 1.0
    assert payload["reference_mission"]["release_passed"] is True
    assert payload["streamlit"]["app_test_passed"] is True
    assert payload["student"]["starter_launches"] is True
    assert payload["student"]["solved_marker_count"] == 1
    assert payload["mlflow"]["persisted"] is True
```

- [ ] **Step 2: Implement the deterministic certification command**

The script runs the recorded mission, validates all citations, persists MLflow evidence, runs the reference AppTest, validates starter/solved student states, and writes stable JSON plus a concise readiness report. Optional live checks must be labeled `NOT RUN`, `AVAILABLE`, `PASS`, or `ERROR`; they cannot change the offline gate. Never serialize environment values or absolute paths.

- [ ] **Step 3: Run focused certification**

Run: `uv run python scripts/certify_capstone.py`

Expected: exit zero and artifact records every mandatory gate as passed.

Run: `uv run pytest tests/test_capstone_certification.py -q`

Expected: all pass.

- [ ] **Step 4: Start Streamlit and capture the desktop reference journey**

Run the reference app headlessly on a fixed local port, execute the recorded mission, and capture a 1440×1000 desktop screenshot. Inspect it for hierarchy, clipping, evidence readability, trace readability, clear status colors, provider/data labels, judge/release separation, and the footer. Record the observed checks in `readiness.md`; do not claim Ollama, OpenAI, Tavily, or timed classroom rehearsal unless actually observed.

- [ ] **Step 5: Run complete quality gates**

Run:

```bash
uv run ruff check src tests scripts final-project
uv run pytest -q
uv run python scripts/validate_repo.py
git diff --check
```

Expected: lint passes, the complete suite passes, repository validation passes, and no whitespace errors are reported.

- [ ] **Step 6: Commit certification evidence**

Commit: `test: certify Financial Analyst Copilot capstone`

---

## Final independent review gate

After all eight tasks pass their scoped reviews, dispatch a fresh high-capability reviewer over the full diff from the plan’s starting commit. The reviewer must score these weighted dimensions:

| Dimension | Weight |
| --- | ---: |
| Technical correctness and evidence safety | 25% |
| Student feasibility within 60 minutes | 20% |
| Product usefulness and finance realism | 20% |
| Offline reliability and diagnostics | 15% |
| Streamlit usability and visual quality | 10% |
| Repository, tests, and instructor materials | 10% |

Acceptance requires at least `9.5/10` and no unresolved Critical or Important finding. One final fix wave and one scoped re-review are allowed by the Subagent-Driven Development workflow. Report optional-provider availability and timed rehearsal separately from the weighted score.
