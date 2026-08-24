# Financial Analyst Copilot Capstone Design

**Date:** 2026-08-24
**Status:** Proposed for user review
**Course:** AI Engineering for Asset Management
**Class slot:** Day 2, 15:30-16:30 capstone challenge; 16:30-17:00 demonstration and architecture review

## 1. Purpose

Build one complete, professional, but deliberately bounded AI engineering application that integrates the course from model gateway through agent evaluation.

The capstone is a Streamlit Financial Analyst Copilot. It helps an analyst compare NVIDIA and Schneider Electric using official financial-document evidence and selected company metrics. It shows the plan, tool activity, evidence, final briefing, limitations, trace, and evaluation decision.

The capstone has two products in the same repository:

1. A complete reference application that is implemented and certified before the class.
2. A student scaffold with four bounded integration gaps and one diagnostic exercise.

The student exercise must be achievable individually or in pairs in 60 minutes. Students integrate certified components; they do not rebuild the complete RAG, MCP, agent, UI, and evaluation stack from scratch.

## 2. Approved product decisions

- The user interface is Streamlit.
- The core application does not require FastAPI, SSE, authentication, or deployment infrastructure.
- The code uses a typed application-service layer so a future API can be added without rewriting the AI system.
- The mandatory mission is a cross-company NVIDIA and Schneider Electric briefing.
- The certified route uses versioned local documents and market snapshots.
- Optional live market data and Tavily news enrich the product but never determine classroom success.
- The application supports Ollama and OpenAI through explicit configuration.
- The reference application and student scaffold live in the existing repository.
- The same challenge works for individuals and pairs.
- The complete correction is visible in the repository; classroom instructions ask students to work only in the student folder before review.

## 3. Reference mission

The mandatory mission is:

> Compare NVIDIA and Schneider Electric using official documents and selected financial metrics. Identify the main operating-growth evidence, explain why direct comparison is limited, and cite every factual claim.

The reference mission is fixed because it provides:

- a reproducible user journey;
- known expected tool calls and evidence;
- a cross-market US and European comparison;
- a typed recovery path for unsupported metrics;
- a deterministic evidence gate;
- a maintained evaluation suite; and
- a final result that resembles professional analyst work without becoming investment advice.

Custom questions are available only after the reference mission is operational. They use the same bounded system but are not part of the mandatory pass decision.

## 4. User experience

### 4.1 Page structure

The Streamlit page uses one wide research workspace with a persistent sidebar.

**Sidebar**

- Application title and course signature.
- Provider: `Ollama`, `OpenAI`, or `Recorded demo`.
- Model name, shown explicitly.
- Data mode: `Certified snapshots` or `Optional live enrichment`.
- Optional Tavily availability status.
- Safety statement: research support, not investment advice.
- Reset-session action.

**Main workspace**

- Tab 1: `Reference mission`.
- Tab 2: `Ask the analyst`.
- A compact system-readiness strip.
- Mission input panel.
- Execution status panel.
- Results area.

### 4.2 Reference Mission tab

The reference mission text is visible and read-only. The user selects `Run reference mission`.

The application then shows:

1. Validated research request.
2. Initial research plan.
3. Tool activity and typed errors.
4. Replan, when required.
5. Evidence-gate decision.
6. Final structured briefing or a typed stop.
7. Source and evidence table.
8. Execution trace.
9. Five-metric evaluation scorecard.
10. Release decision.

### 4.3 Ask the Analyst tab

The user can enter a custom question within the fixed NVIDIA and Schneider Electric universe.

The interface behaves like a chat, but each submitted message creates a bounded research run rather than an unrestricted conversational agent. Conversation history is kept only in Streamlit session state. Persistent memory is a non-goal.

Custom questions must still produce:

- a typed status;
- a bounded plan;
- source-backed factual claims;
- explicit limitations;
- an inspectable trace; and
- no investment recommendation or price target.

### 4.4 Result presentation

The final result uses five sections:

1. `Executive briefing`
2. `Company evidence`
3. `Cross-company comparison`
4. `Limitations and open questions`
5. `Sources and execution`

Each factual claim displays its provenance kind, source reference, and evidence ID when applicable. Metric facts and document facts remain visually distinct.

The trace is shown in an expander with phase, tool, attempt ID, plan revision, status, typed error, duration, and failure owner.

The evaluation scorecard displays:

- tool-call correctness;
- tool-call efficiency;
- answer relevance;
- answer completeness; and
- citation integrity.

Optional LLM judges appear in a separate area and never change deterministic release status.

## 5. Technical architecture

```text
Streamlit UI
├── Reference mission
└── Ask the analyst
          │
          ▼
FinancialAnalystCopilot service
├── validate request
├── select certified or live data mode
├── create bounded plan-and-execute run
├── enforce evidence gate
├── return typed public result
└── persist evaluation evidence
          │
          ├── Ollama / OpenAI model gateway
          ├── hybrid document retrieval
          ├── local financial MCP client
          ├── optional Tavily news adapter
          └── MLflow SQLite tracing and evaluation
```

The Streamlit layer contains no retrieval, tool-selection, agent-policy, evidence, or evaluation logic. It receives typed public view models from the application service.

### 5.1 Application service

The public service contract is:

```python
class FinancialAnalystCopilot(Protocol):
    def run(self, request: ResearchRequest) -> ResearchRunResult: ...
```

The service owns orchestration only. It composes existing certified course modules rather than copying their algorithms.

### 5.2 Core contracts

`ResearchRequest`

- `mode`: `reference` or `custom`
- `question`
- `companies`: exactly NVIDIA and Schneider Electric for the reference mission
- `provider`: `recorded`, `ollama`, or `openai`
- `model`
- `data_mode`: `certified` or `live_enrichment`
- `include_news`
- `max_steps`: at most 6
- `max_replans`: at most 1

`ResearchRunResult`

- stable run ID
- request
- final status
- initial and final plan
- public observations
- public trajectory
- evidence-gate result
- optional analyst briefing
- deterministic evaluation
- optional judge evaluation
- MLflow run and trace identifiers
- total duration

`CapstoneBriefing`

- executive summary
- typed cited facts
- company evidence sections
- cross-company observations
- interpretation
- limitations
- open questions
- ordered aggregate sources

The capstone reuses the strict provenance rules already certified in Lessons 11 and 12. It does not introduce a second incompatible citation model.

### 5.3 Core capabilities

The mandatory tool registry contains only:

1. `search_financial_documents`
2. `get_company_metric`

Both are exposed through the existing local financial MCP server and discovered by the existing MCP client.

Optional live enrichment adds:

3. `search_company_news`

The news adapter uses Tavily only when an explicit key is available. It remains outside the mandatory reference trajectory and retains URL, title, publication date when available, and retrieval timestamp.

A deterministic internal comparison function may calculate display differences from already retrieved metric inputs. It is not an unrestricted code-execution tool.

### 5.4 Agent policy

The capstone uses the certified plan-and-execute graph:

```text
planning
  -> plan gate
  -> tool execution
  -> bounded replanning when needed
  -> evidence gate
  -> report or typed stop
```

Hard limits:

- maximum 6 planned steps;
- maximum 1 replan;
- no duplicate successful tool signature;
- capability allowlist from MCP discovery intersected with static policy;
- no report after a failed evidence gate;
- no trading, order, portfolio-mutation, code-execution, or unrestricted browsing tool.

The unsupported NVIDIA Revenue metric remains the reference typed error. The valid path replans into document search rather than fabricating a metric.

### 5.5 Model providers

The application supports:

- Ollama for local use;
- OpenAI with an explicit API key and model; and
- a clearly labelled recorded-demo mode for guaranteed classroom demonstration and automated testing.

Recorded mode never presents recorded text as a live model result. The UI labels the provider and data source on every run.

## 6. Data architecture

### 6.1 Certified data

The mandatory route uses versioned assets already present in the repository:

- NVIDIA official financial-document fixture;
- Schneider Electric official financial-document fixture;
- versioned market and metric snapshots;
- financial MCP evidence catalog;
- agent expectation dataset;
- agent recorded-run fixture.

Every canonical asset is verified against the repository manifest before use.

### 6.2 Optional live enrichment

Live enrichment is read-only and best-effort.

- Market data may refresh selected display metrics through a read-only `yfinance` adapter. Returned values retain symbol, field, currency when available, provider, and retrieval timestamp.
- Tavily may add recent company-news context.
- Live results are never silently substituted for certified facts.
- The UI labels source, timestamp, and availability.
- Provider failure leaves the certified reference mission operational.

### 6.3 No-network behavior

With no internet, no API key, and no Ollama process, the recorded-demo route still demonstrates the complete reference mission, citations, trace, and evaluation.

With Ollama available but no network, the complete local model route runs against bundled documents and local MCP tools.

## 7. Repository structure

```text
final-project/
├── README.md
├── PRODUCT_SPEC.md
├── STUDENT_BRIEF.md
├── INSTRUCTOR_GUIDE.md
├── reference/
│   ├── streamlit_app.py
│   └── README.md
├── student/
│   ├── streamlit_app.py
│   ├── integration.py
│   ├── README.md
│   └── CHECKLIST.md
└── shared/
    └── reference_mission.json

src/finai_academy/capstone/
├── models.py
├── model_gateway.py
├── briefing.py
├── service.py
├── tools.py
├── live_news.py
├── views.py
└── streamlit_ui.py

tests/
├── test_capstone_service.py
├── test_capstone_tools.py
├── test_capstone_views.py
├── test_capstone_streamlit.py
└── test_capstone_student.py
```

The shared Streamlit rendering layer receives a service factory. The reference and student entry points remain thin and visibly different only at the four student integration seams.

The existing Module 00 CLI remains supported as a small historical vertical slice. The Streamlit application becomes the canonical integrated capstone.

## 8. Reference application first

The complete correction is built before deriving the student exercise.

Implementation sequence:

1. Extend and reconcile the capstone domain contracts with Lessons 11 and 12.
2. Build the complete typed application service.
3. Integrate document retrieval and the local financial MCP.
4. Add recorded, Ollama, and OpenAI routes.
5. Build and visually certify the Streamlit interface.
6. Add deterministic capstone evaluation and MLflow evidence.
7. Execute and certify the complete reference mission.
8. Derive the student scaffold from the certified integration layer.
9. Verify that the starter initially fails only the intended public checks.
10. Verify that the completed student tasks reach the same reference acceptance contract.

No student scaffold TODO is implemented until the corresponding reference behavior has a passing test and observable UI result. The four task boundaries in this specification define the teaching intent; their exact removed lines are derived only from certified reference code.

## 9. Student challenge

### 9.1 Four implementation tasks

Students complete four bounded seams in `final-project/student/integration.py`.

#### Task A: Wire the retriever

Connect company and document filters to the provided hybrid retrieval pipeline. The implementation must keep NVIDIA and Schneider evidence separate and preserve evidence IDs.

#### Task B: Register the analyst capabilities

Build the allowed capability set from local MCP discovery. Only `search_financial_documents` and `get_company_metric` may enter the mandatory registry.

#### Task C: Complete the evidence gate

Require successful document evidence for both companies and valid provenance before report generation. A failed gate returns a typed stop and no briefing.

#### Task D: Assemble the public briefing view

Convert the typed agent result into the displayed briefing, citations, limitations, trace summary, and evaluation scorecard without leaking private runtime state.

### 9.2 Diagnostic task

After completing the four seams, students run the maintained capstone evaluation. One deliberately regressed configuration must fail.

Students must:

1. identify the failed case;
2. inspect its MLflow trace;
3. assign the failure owner;
4. correct the intended student integration defect; and
5. rerun the gate.

The diagnostic task uses a failure already taught in Lesson 12. It does not introduce a new framework.

### 9.3 Student pass marker

The student route passes only when the verification command prints exactly one:

```text
CAPSTONE_PASS
```

The marker requires:

- all four public integration contracts pass;
- the reference mission completes;
- the evidence gate passes;
- every factual claim has valid provenance;
- citation integrity equals 1.0;
- the deterministic release decision passes;
- the reference run and trace are persisted; and
- no secret or personal path appears in serialized evidence.

## 10. Classroom timing

### 15:30-15:40: Understand the mission

- Run the incomplete student app.
- Read the four TODO contracts.
- Inspect the expected reference output.
- Pairs select a driver and navigator.

### 15:40-16:10: Complete the four integration seams

- Approximately 7 minutes per seam.
- Pairs switch driver after Tasks B or C.
- Instructor uses tests and visible outputs as checkpoints.

### 16:10-16:25: Evaluate and diagnose

- Run public tests.
- Execute the reference mission.
- Inspect the failed evaluation and trace.
- Correct the intended defect.
- Reach `CAPSTONE_PASS`.

### 16:25-16:30: Prepare the demonstration

- Select one briefing claim and its evidence.
- Select one recovered failure.
- Select one architectural trade-off.

### 16:30-17:00: Demonstration and architecture review

- Instructor runs the complete reference application.
- Students or pairs present their evidence path and diagnostic.
- Class reviews what would be required for production deployment.

## 11. Error handling and truthful states

The UI never converts missing capability into apparent success.

| Condition | Required behavior |
| --- | --- |
| Ollama unavailable | Show setup guidance; offer recorded demo |
| OpenAI key missing | Disable OpenAI run; do not fall back silently |
| Tavily key missing | Mark live news unavailable; continue certified route |
| MCP server unavailable | Show typed capability error; do not generate briefing |
| Dataset/hash mismatch | Stop before execution and name the affected asset |
| Retrieval lacks one company | Evidence gate fails; no briefing |
| Unsupported metric | Record typed error and permit one bounded replan |
| Invalid citation pair | Citation integrity fails; release blocked |
| Judge provider unavailable | `NOT RUN` with sanitized reason |
| Judge invocation timeout/runtime failure | `ERROR` with sanitized reason |
| Low or disagreeing judge score | `COMPLETED`; display separately from deterministic release |
| Unexpected application error | Sanitized UI error; detailed safe local log; no secret echo |

Streamlit session state retains only public serializable results and UI selections. Model objects, clients, credentials, and private reasoning are never stored in the session result.

## 12. Testing strategy

### 12.1 Pure domain and service tests

- Request validation and hard limits.
- Provider selection and no silent fallback.
- Reference mission contract.
- Evidence-gate pass and stop paths.
- Exact source/evidence pairing.
- No report after failed gate.
- Public-state secret rejection.

### 12.2 Integration tests

- Local MCP discovery and allowlist intersection.
- Hybrid retrieval keeps company and evidence metadata.
- Recorded reference mission produces the certified public signature.
- Local SQLite run and trace association.
- Six acceptance cases and five deterministic metrics.
- Optional judge metadata remains separate and self-consistent.

### 12.3 Streamlit tests

Use Streamlit's application testing interface for the complete recorded reference route and the principal typed-stop route.

- Both tabs render.
- Reference mission can be launched in recorded mode.
- Provider and data labels are visible.
- Successful run displays briefing, evidence, trace, metrics, and release.
- Failed evidence gate displays no briefing.
- Optional live controls degrade truthfully.
- No credentials or personal paths render.

### 12.4 Student scaffold tests

- Starter fails exactly the intended four contract groups and diagnostic case.
- No failure is caused by missing dependencies, broken imports, or absent data.
- Reference implementation passes the same public contract.
- Completed student solution reaches exactly one `CAPSTONE_PASS`.

### 12.5 Visual verification

The reference app is captured at a standard desktop viewport and inspected for:

- readable hierarchy;
- unclipped briefing and evidence tables;
- clear success, warning, stop, and unavailable colors;
- visible provenance;
- trace readability;
- deterministic and optional judge separation; and
- no raw JSON wall as the primary learner experience.

## 13. Acceptance criteria

The capstone is ready for the classroom only when all of the following are observed.

### Reference application

- `streamlit run final-project/reference/streamlit_app.py` starts successfully.
- Recorded mode completes without network, API key, or Ollama.
- Ollama runs locally with the documented model when available.
- OpenAI configuration is explicit and does not affect the offline route.
- The reference mission produces the expected NVIDIA and Schneider evidence.
- The unsupported metric is recovered through the bounded replan.
- Every factual claim has valid provenance.
- The evidence gate prevents unsupported reporting.
- MLflow contains the run, trace, five deterministic metrics, and artifacts.
- The Streamlit result clearly separates deterministic release from optional judges.
- Automated tests, repository validation, and visual review pass.

### Student challenge

- The starter launches before any TODO is completed.
- The four TODOs are bounded and independently testable.
- Error messages point to concepts, not solution code.
- One individual or pair can complete the challenge in 60 minutes.
- The public verification ends with exactly one `CAPSTONE_PASS`.
- The correction explains each task and the diagnostic, not only the final code.

### Instructor materials

- Student brief with mission, constraints, tasks, commands, and checklist.
- Instructor guide with timing, expected outputs, hints, solution discussion, and recovery route.
- Reference screenshot or short demonstration sequence.
- Setup instructions for macOS and Windows.
- Recorded fallback for every optional external provider.

## 14. Explicit non-goals

- Separate FastAPI service in the mandatory project.
- Server-sent event infrastructure.
- Authentication or user accounts.
- Persistent chat memory.
- Cloud deployment.
- Portfolio construction, orders, or autonomous trading.
- Price targets or investment recommendations.
- Multi-agent supervisor architecture.
- Arbitrary web browsing.
- Arbitrary code execution.
- More than three user-visible research tools.
- Building embeddings, vector storage, MCP protocol code, or MLflow infrastructure from scratch during the final hour.

FastAPI, deployment, persistent sessions, and additional live-data connectors may be documented as stretch goals after the certified classroom route is complete.

## 15. Quality gate

The final capstone requires an independent weighted score of at least 9.5/10 with no unresolved Critical or Important finding.

Rubric:

| Dimension | Weight |
| --- | ---: |
| Technical correctness and evidence safety | 25% |
| Student feasibility within 60 minutes | 20% |
| Product usefulness and finance realism | 20% |
| Offline reliability and diagnostics | 15% |
| Streamlit usability and visual quality | 10% |
| Repository, tests, and instructor materials | 10% |

Live-provider availability and a timed classroom rehearsal are reported separately. They cannot substitute for the offline quality gate and cannot be claimed unless observed.

## 16. Delivery boundary

This specification authorizes the design only. Implementation starts after explicit user approval of this written specification and creation of a separate implementation plan.
