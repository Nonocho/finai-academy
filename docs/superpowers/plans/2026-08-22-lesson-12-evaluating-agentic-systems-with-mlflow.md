# Lesson 12 Evaluating Agentic Systems with MLflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and certify a 60-minute Lesson 12 that evaluates the public trajectory and cited answer of the Lesson 11 Financial Analyst Copilot with six versioned cases, five deterministic metrics, and two aligned local MLflow runs.

**Architecture:** Keep provider-neutral evaluation records, alignment, deterministic scorers, and failure ownership in `agent_evaluation.py`; convert certified `PlanExecuteResult` values losslessly into that safe boundary while loading intentionally regressed candidate facts directly into the same evaluation-only record. Put SQLite tracking, reconstructed public traces, artifacts, comparisons, and explicit optional MLflow GenAI judges behind `mlflow_agent_evaluation.py`, so observability never changes agent behavior and the complete classroom route remains offline-first.

**Tech Stack:** Python 3.11+, Pydantic 2, MLflow 3.15+, SQLite, LangGraph Lesson 11 public contracts, `uv`, pytest, Ruff, Jupyter/nbclient/nbformat, Pandas, Matplotlib, and PowerPoint through the established presentation workflow.

**Spec:** `docs/superpowers/specs/2026-08-22-lesson-12-evaluating-agentic-systems-with-mlflow-design.md`

## Global Constraints

- Preserve the canonical Day 2 slot: 14:30-15:30.
- Preserve the complete 12-minute deck + 40-minute notebook + 8-minute verification and debrief route.
- MLflow is the only evaluation and observability framework introduced in Lesson 12; do not add Ragas.
- The core route must run offline with no API key, external tracking server, browser UI, or Docker.
- Use local SQLite plus local artifacts selected by `FINAI_MLFLOW_DIR` or a safe temporary directory.
- Evaluate public, serializable Lesson 11 state only; never request or log hidden chain-of-thought, complete prompts, raw environment dumps, credentials, private document content, or personal paths.
- Keep expectations and observed outputs separate in `agent_cases_v1.json` and `agent_runs_v1.json`; verify the canonical case bytes with SHA-256 before scoring.
- Preserve exactly six aligned cases for both `bounded-agent-v1` and `regressed-agent-v0`.
- Preserve the five deterministic scores `tool_call_correctness`, `tool_call_efficiency`, `answer_relevance`, `answer_completeness`, and `citation_integrity`, each in `[0, 1]` with a public rationale.
- Citation integrity remains the deterministic finance release gate; a source/evidence pair must match one returned document hit exactly.
- A required evidence-gate stop that emits a briefing blocks release even if other means are high.
- OpenAI and Ollama judges are optional, explicit routes and never silent fallbacks; unavailable judges report `NOT RUN` with a reason.
- A valid Lesson 11 `PlanExecuteResult` must convert without information loss; regression records use permissive `CandidateFact` values so missing or incorrect provenance is scored, not rejected by the Lesson 11 production model.
- The notebook must contain 24 to 28 stable output-free cells, at least six rendered PNG visuals, two local MLflow runs over the same six cases, and exactly one `LESSON_12_PASS` marker.
- The concept deck must contain exactly nine slides, original mechanism diagrams, comparison tables on slides 3, 7, and 8, directly relevant source notes, and the footer `First Finance - Arnaud Demes` on every slide.
- Learner-facing notebook and visible deck copy must use short professional English and contain no visible em dash character.
- Do not reproduce MLExpert Academy source code, lesson copy, diagrams, data, or screenshots.
- Final lesson-quality certification requires an independent score of at least 9.5/10 and no unresolved Important or Critical finding; provider coverage and timed rehearsal are recorded separately and cannot lower or inflate the offline lesson-quality score.

## File Map

| File | Responsibility |
| --- | --- |
| `src/finai_academy/agent_evaluation.py` | Evaluation-only Pydantic contracts, safe Lesson 11 conversion, strict loaders, alignment, five deterministic scorers, summaries, and failure ownership. |
| `tests/test_agent_evaluation.py` | Contract, conversion, schema, hash, alignment, scorer, citation, release-gate, and failure-classification tests. |
| `assets/course-data/evaluation/agent_cases_v1.json` | Six canonical expectation rows and dependency-aware expected calls. |
| `assets/course-data/evaluation/agent_runs_v1.json` | Two labelled configurations with six public candidate predictions each. |
| `assets/course-data/manifest.json` | Canonical case and recorded-run paths, versions, and SHA-256 values. |
| `src/finai_academy/mlflow_agent_evaluation.py` | Local SQLite initialization, run/trace logging, artifacts, inline summaries, aligned comparison, and optional judge adapters. |
| `tests/test_mlflow_agent_evaluation.py` | Local store, parameters, metrics, artifacts, trace topology, sanitization, alignment, flush, comparison, and optional-judge tests. |
| `.env.example` | Explicit `FINAI_EVAL_JUDGE_MODEL` examples for OpenAI and Ollama with no enabled default. |
| `scripts/build_lesson12_notebook.py` | Deterministic source-notebook generator with stable cell IDs. |
| `notebooks/12_evaluating_agentic_systems.ipynb` | Forty-minute offline-first guided lab and six required visual explanations. |
| `tests/test_lesson12_assets.py` | Notebook, execution, chapter, indexes, deck structure, notes, footer, and copy contracts. |
| `tests/test_course_manifest.py` | Canonical Lesson 12 slot and staged asset-presence contract. |
| `course.yml` | Canonical Lesson 12 title and chapter/notebook/deck paths from the approved specification. |
| `chapters/12-evaluating-agentic-systems.md` | Instructor script, timing, expected evidence, recovery routes, answer key, and capstone handoff. |
| `chapters/README.md` | Discoverable Lesson 12 instructor link and completed Day 2 status. |
| `notebooks/README.md` | Discoverable Lesson 12 notebook link and execution marker. |
| `decks/README.md` | Discoverable Lesson 12 deck link and readiness status. |
| `README.md` | Complete Lessons 08-12 route, Lesson 12 links, and local evaluation setup. |
| `docs/getting-started.md` | Evaluation-extra installation, local MLflow directory/UI command, and optional judge configuration. |
| `decks/12-evaluating-agentic-systems.pptx` | Nine-slide sourced concept deck following the Lesson 10/11 visual system. |
| `docs/reviews/lesson-12-readiness.md` | Exact certification evidence, independent findings, quality score, provider coverage, rehearsal status, and release decision. |

---

### Task 1: Define pure evaluation contracts and the lossless Lesson 11 conversion

**Files:**
- Create: `src/finai_academy/agent_evaluation.py`
- Create: `tests/test_agent_evaluation.py`

**Interfaces:**
- Consumes: `PlanExecuteResult` from `finai_academy.plan_execute_graph`; `ResearchPlan`, `PlanStep`, `ResearchObservation`, `TrajectoryEvent`, and `EvidenceGateResult` from `finai_academy.research_planning`.
- Produces: `ExpectedToolCall`, `AgentEvaluationCase`, `CandidateFact`, `CandidateBriefing`, `AgentEvaluationPrediction`, `MetricScore`, `AgentCaseScores`, `AgentEvaluationSummary`, `FailureStage`, `prediction_from_plan_execute_result(result: PlanExecuteResult, *, case_id: str, dataset_version: str, dataset_sha256: str, configuration_id: str, agent_version: str, provider: Literal["recorded", "openai", "ollama"], agent_model: str, prompt_version: str, max_steps: int, max_replans: int) -> AgentEvaluationPrediction`, and `canonical_call_signature(capability: str, arguments: Mapping[str, Any]) -> str`.

- [ ] **Step 1: Write failing contract and conversion tests**

Create fixtures from the existing Lesson 11 `MISSION`, `initial_plan()`, `ResearchObservation`, `TrajectoryEvent`, `EvidenceGateResult`, and `AnalystBriefing` helpers. Add these tests with exact names:

```python
def test_prediction_conversion_preserves_every_public_plan_execute_field() -> None:
    result = completed_plan_execute_result()
    prediction = prediction_from_plan_execute_result(
        result,
        case_id="reference_completed",
        dataset_version="agent-cases-v1",
        dataset_sha256="a" * 64,
        configuration_id="bounded-agent-v1",
        agent_version="lesson11-certified-v1",
        provider="recorded",
        agent_model="recorded-public-fixture-v1",
        prompt_version="lesson11-recorded-policies-v1",
        max_steps=6,
        max_replans=1,
    )
    assert prediction.status == result.status
    assert prediction.initial_plan == result.initial_plan
    assert prediction.final_steps == result.final_steps
    assert prediction.observations == result.observations
    assert prediction.trajectory == result.trajectory
    assert prediction.replan_count == result.replan_count
    assert prediction.evidence_gate == result.evidence_gate
    assert prediction.briefing is not None and result.briefing is not None
    assert prediction.briefing.reported_facts[0].claim == result.briefing.reported_facts[0].claim
    assert prediction.briefing.reported_facts[0].provenance_kind == result.briefing.reported_facts[0].provenance_kind
    assert prediction.briefing.source_references == result.briefing.source_references


def test_candidate_fact_can_represent_missing_document_provenance_for_scoring() -> None:
    fact = CandidateFact(
        claim="Schneider Electric revenue grew in the maintained evidence.",
        provenance_kind="document",
        source_references=("assets/course-data/fixtures/schneider_fy2025_excerpt.pdf",),
        evidence_ids=(),
    )
    assert fact.evidence_ids == ()


def test_candidate_fact_rejects_blank_text_and_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        CandidateFact.model_validate({"claim": " ", "invented": True})


def test_canonical_call_signature_sorts_nested_arguments() -> None:
    left = canonical_call_signature(
        "get_company_metric", {"metric": "P/E", "ticker": "NVDA"}
    )
    right = canonical_call_signature(
        "get_company_metric", {"ticker": "NVDA", "metric": "P/E"}
    )
    assert left == right == 'get_company_metric:{"metric":"P/E","ticker":"NVDA"}'
```

- [ ] **Step 2: Run the focused tests and confirm RED**

```bash
.venv/bin/pytest -q tests/test_agent_evaluation.py -k "conversion or candidate_fact or canonical_call_signature"
```

Expected: pytest collection fails with `ModuleNotFoundError: No module named 'finai_academy.agent_evaluation'`.

- [ ] **Step 3: Implement strict models with a deliberately permissive candidate fact**

Use Pydantic models with `model_config = ConfigDict(extra="forbid", frozen=True)`, strip every string, reject blank tuple members, preserve tuple order, and keep `agent_evaluation.py` free of any MLflow import. Implement these exact shapes:

```python
FinalStatus = Literal[
    "completed", "plan_blocked", "execution_stopped",
    "replan_budget_exhausted", "insufficient_evidence", "provider_error",
]
FailureStage = Literal[
    "none", "planner", "tool_boundary", "replanner", "evidence_gate",
    "report_writer", "dataset", "judge",
]
MetricName = Literal[
    "tool_call_correctness", "tool_call_efficiency", "answer_relevance",
    "answer_completeness", "citation_integrity",
]


class ExpectedToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    call_id: str
    capability: str
    arguments: dict[str, Any]
    prerequisite_call_ids: tuple[str, ...] = ()


class AgentEvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    case_id: str
    mission: str
    expected_final_status: FinalStatus
    expected_tool_calls: tuple[ExpectedToolCall, ...]
    expected_error_codes: tuple[str, ...]
    expected_replan_count: int = Field(ge=0)
    max_tool_calls: int = Field(ge=0)
    required_companies: tuple[str, ...]
    required_evidence_ids: tuple[str, ...]
    required_fact_kinds: tuple[Literal["metric", "document"], ...]
    required_limitations: tuple[str, ...]
    allow_briefing: bool


class CandidateFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    claim: str
    provenance_kind: Literal["metric", "document"] | None = None
    source_references: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()


class CandidateBriefing(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    reported_facts: tuple[CandidateFact, ...]
    cross_company_observations: tuple[str, ...]
    interpretation: tuple[str, ...]
    limitations: tuple[str, ...]
    source_references: tuple[str, ...]


class AgentEvaluationPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    case_id: str
    dataset_version: str
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration_id: str
    agent_version: str
    provider: Literal["recorded", "openai", "ollama"]
    agent_model: str
    prompt_version: str
    max_steps: int = Field(ge=1)
    max_replans: int = Field(ge=0)
    status: FinalStatus
    initial_plan: ResearchPlan
    final_steps: tuple[PlanStep, ...]
    observations: tuple[ResearchObservation, ...]
    trajectory: tuple[TrajectoryEvent, ...]
    replan_count: int = Field(ge=0)
    evidence_gate: EvidenceGateResult
    briefing: CandidateBriefing | None
```

`CandidateFact` intentionally does not reuse `CitedFact`: missing evidence IDs, missing provenance kind, or an incorrect but well-formed source/evidence pair must survive fixture loading so `citation_integrity` can return zero. It still rejects blank strings, unknown keys, non-container inputs, and secret-shaped values matching `(?i)(api[_-]?key|authorization|bearer\s+[a-z0-9._-]+|sk-[a-z0-9]{12,})` anywhere in candidate fields.

- [ ] **Step 4: Implement and test the one-way conversion boundary**

Map valid Lesson 11 facts into permissive candidate facts; never attempt to convert a regression `CandidateBriefing` back into `AnalystBriefing`:

```python
def prediction_from_plan_execute_result(result: PlanExecuteResult, **metadata: object) -> AgentEvaluationPrediction:
    briefing = None
    if result.briefing is not None:
        briefing = CandidateBriefing(
            reported_facts=tuple(
                CandidateFact(
                    claim=fact.claim,
                    provenance_kind=fact.provenance_kind,
                    source_references=fact.source_references,
                    evidence_ids=fact.evidence_ids,
                )
                for fact in result.briefing.reported_facts
            ),
            cross_company_observations=result.briefing.cross_company_observations,
            interpretation=result.briefing.interpretation,
            limitations=result.briefing.limitations,
            source_references=result.briefing.source_references,
        )
    return AgentEvaluationPrediction(
        **metadata,
        status=result.status,
        initial_plan=result.initial_plan,
        final_steps=result.final_steps,
        observations=result.observations,
        trajectory=result.trajectory,
        replan_count=result.replan_count,
        evidence_gate=result.evidence_gate,
        briefing=briefing,
    )
```

Add a JSON round-trip assertion on the converted prediction and a test showing that a missing-provenance regression prediction validates only as `AgentEvaluationPrediction`, while `AnalystBriefing.model_validate()` rejects its briefing payload.

- [ ] **Step 5: Add score containers and pure-module guards**

Implement:

```python
class MetricScore(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    value: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1)


class AgentCaseScores(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    case_id: str
    configuration_id: str
    tool_call_correctness: MetricScore
    tool_call_efficiency: MetricScore
    answer_relevance: MetricScore
    answer_completeness: MetricScore
    citation_integrity: MetricScore
    failure_stage: FailureStage
    release_passed: bool
    total_tool_calls: int = Field(ge=0)
    redundant_tool_calls: int = Field(ge=0)
    latency_ms: float = Field(ge=0)


class AgentEvaluationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    configuration_id: str
    dataset_version: str
    dataset_sha256: str
    case_count: int = Field(ge=1)
    metric_means: dict[MetricName, float]
    metric_pass_counts: dict[MetricName, int]
    mean_tool_calls: float = Field(ge=0)
    mean_latency_ms: float = Field(ge=0)
    max_latency_ms: float = Field(ge=0)
    release_passed: bool
```

Add `test_agent_evaluation_module_does_not_import_mlflow()` using `inspect.getsource()` and assert neither `import mlflow` nor `from mlflow` appears.

- [ ] **Step 6: Run and commit the pure contract boundary**

```bash
.venv/bin/pytest -q tests/test_agent_evaluation.py -k "conversion or contract or candidate_fact or canonical_call_signature"
.venv/bin/ruff check src/finai_academy/agent_evaluation.py tests/test_agent_evaluation.py
git add src/finai_academy/agent_evaluation.py tests/test_agent_evaluation.py
git commit -m "feat: add agent evaluation contracts"
```

Expected: focused tests pass; Ruff reports `All checks passed!`; the commit contains only the pure module and tests.

---

### Task 2: Add versioned cases, recorded configurations, alignment, and five deterministic scorers

**Files:**
- Modify: `src/finai_academy/agent_evaluation.py`
- Modify: `tests/test_agent_evaluation.py`
- Create: `assets/course-data/evaluation/agent_cases_v1.json`
- Create: `assets/course-data/evaluation/agent_runs_v1.json`
- Modify: `assets/course-data/manifest.json`

**Interfaces:**
- Consumes: all Task 1 contracts and existing Lesson 10/11 fixture identifiers and public observations.
- Produces: `AgentEvaluationDataset`, `RecordedAgentRuns`, `load_agent_evaluation_dataset(path: Path, *, expected_sha256: str) -> AgentEvaluationDataset`, `load_recorded_agent_runs(path: Path, *, cases: AgentEvaluationDataset, expected_sha256: str) -> RecordedAgentRuns`, `align_cases_and_predictions(cases: Sequence[AgentEvaluationCase], predictions: Sequence[AgentEvaluationPrediction], *, dataset_version: str, dataset_sha256: str) -> tuple[tuple[AgentEvaluationCase, AgentEvaluationPrediction], ...]`, `score_agent_case(case: AgentEvaluationCase, prediction: AgentEvaluationPrediction) -> AgentCaseScores`, `summarize_agent_evaluation(scores: Sequence[AgentCaseScores], *, dataset_version: str, dataset_sha256: str) -> AgentEvaluationSummary`, and `classify_failure(case: AgentEvaluationCase, prediction: AgentEvaluationPrediction, scores: AgentCaseScores) -> FailureStage`.

- [ ] **Step 1: Write failing dataset, hash, and exact-alignment tests**

Add these named tests:

```python
def test_agent_cases_v1_has_exactly_six_required_cases_and_expected_calls() -> None:
    dataset = load_agent_evaluation_dataset(CASES_PATH, expected_sha256=manifest_case_hash())
    assert dataset.dataset_version == "agent-cases-v1"
    assert tuple(case.case_id for case in dataset.cases) == (
        "reference_completed",
        "unsupported_metric_not_recovered",
        "redundant_metric_call",
        "missing_schneider_document",
        "document_fact_without_evidence_id",
        "wrong_source_evidence_pair",
    )
    assert all(call.call_id for case in dataset.cases for call in case.expected_tool_calls)


def test_dataset_loader_rejects_byte_hash_mismatch_before_parsing(tmp_path: Path) -> None:
    path = tmp_path / "agent_cases_v1.json"
    path.write_bytes(CASES_PATH.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="dataset SHA-256 mismatch"):
        load_agent_evaluation_dataset(path, expected_sha256=manifest_case_hash())


def test_recorded_runs_have_two_configurations_and_six_aligned_predictions_each() -> None:
    dataset = load_agent_evaluation_dataset(CASES_PATH, expected_sha256=manifest_case_hash())
    runs = load_recorded_agent_runs(
        RUNS_PATH, cases=dataset, expected_sha256=manifest_run_hash()
    )
    assert tuple(config.configuration_id for config in runs.configurations) == (
        "bounded-agent-v1", "regressed-agent-v0"
    )
    assert all(len(config.predictions) == 6 for config in runs.configurations)


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "wrong_hash", "wrong_version"])
def test_alignment_rejects_partial_or_mismatched_prediction_tables(mutation: str) -> None:
    cases, predictions = aligned_fixture()
    changed = mutate_alignment(predictions, mutation)
    with pytest.raises(ValueError, match="alignment"):
        align_cases_and_predictions(
            cases, changed, dataset_version="agent-cases-v1", dataset_sha256="a" * 64
        )
```

- [ ] **Step 2: Run data tests and confirm RED**

```bash
.venv/bin/pytest -q tests/test_agent_evaluation.py -k "agent_cases_v1 or dataset_loader or recorded_runs or alignment"
```

Expected: failures report missing canonical JSON files and undefined loader functions.

- [ ] **Step 3: Create the canonical expectation and prediction schemas**

Add these exact version/container models before loading the files:

```python
class AgentEvaluationDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1]
    dataset_version: Literal["agent-cases-v1"]
    scorer_contract_version: Literal["agent-scorers-v1"]
    dataset_sha256: str
    cases: tuple[AgentEvaluationCase, ...]


class RecordedAgentConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    configuration_id: Literal["bounded-agent-v1", "regressed-agent-v0"]
    agent_version: str
    provider: Literal["recorded"]
    agent_model: str
    prompt_version: str
    max_steps: int = Field(ge=1)
    max_replans: int = Field(ge=0)
    predictions: tuple[AgentEvaluationPrediction, ...]


class RecordedAgentRuns(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1]
    fixture_version: Literal["agent-runs-v1"]
    dataset_version: Literal["agent-cases-v1"]
    dataset_sha256: str
    configurations: tuple[RecordedAgentConfiguration, ...]
```

Write `agent_cases_v1.json` as a stable UTF-8 JSON object with this top-level shape and all six fully populated rows:

```json
{
  "schema_version": 1,
  "dataset_version": "agent-cases-v1",
  "scorer_contract_version": "agent-scorers-v1",
  "cases": [
    {
      "case_id": "reference_completed",
      "mission": "Produce a concise NVIDIA and Schneider Electric briefing. Compare their available valuation metrics and latest operating-growth evidence. Cite every factual claim and state which observations cannot be compared directly.",
      "expected_final_status": "completed",
      "expected_tool_calls": [
        {"call_id": "metric-nvda-pe", "capability": "get_company_metric", "arguments": {"ticker": "NVDA", "metric": "P/E"}, "prerequisite_call_ids": []},
        {"call_id": "metric-su-pe", "capability": "get_company_metric", "arguments": {"ticker": "SU.PA", "metric": "P/E"}, "prerequisite_call_ids": []},
        {"call_id": "metric-nvda-revenue-error", "capability": "get_company_metric", "arguments": {"ticker": "NVDA", "metric": "Revenue"}, "prerequisite_call_ids": []},
        {"call_id": "document-nvda-growth", "capability": "search_financial_documents", "arguments": {"company": "NVIDIA", "query": "revenue growth", "top_k": 2}, "prerequisite_call_ids": ["metric-nvda-pe"]},
        {"call_id": "document-su-growth", "capability": "search_financial_documents", "arguments": {"company": "Schneider Electric", "query": "energy management", "top_k": 2}, "prerequisite_call_ids": ["metric-su-pe", "document-nvda-growth"]}
      ],
      "expected_error_codes": ["unsupported_metric"],
      "expected_replan_count": 1,
      "max_tool_calls": 5,
      "required_companies": ["NVIDIA", "Schneider Electric"],
      "required_evidence_ids": ["NVDA-FY2026-DATA-CENTER-001", "SU-FY2025-ENERGY-MANAGEMENT-001"],
      "required_fact_kinds": ["metric", "document"],
      "required_limitations": ["currencies", "reporting periods", "business definitions"],
      "allow_briefing": true
    }
  ]
}
```

Complete the six expectation rows with this exact matrix; call-set letters expand to the fully specified `ExpectedToolCall` objects below the table:

| Case | Status | Calls | Errors | Replans | Budget | Required evidence IDs | Fact kinds | Limits | Briefing |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| `reference_completed` | `completed` | A, B, C, D, E | `unsupported_metric` | 1 | 5 | NVIDIA Data Center 001; Schneider Energy Management 001 | metric, document | currencies; reporting periods; business definitions | yes |
| `unsupported_metric_not_recovered` | `execution_stopped` | A, B, C | `unsupported_metric` | 0 | 3 | NVIDIA Data Center 001; Schneider Energy Management 001 | metric, document | none | no |
| `redundant_metric_call` | `completed` | A, B, C, D, E | `unsupported_metric` | 1 | 5 | NVIDIA Data Center 001; Schneider Energy Management 001 | metric, document | currencies; reporting periods; business definitions | yes |
| `missing_schneider_document` | `insufficient_evidence` | A, B, C, D | `unsupported_metric` | 1 | 4 | NVIDIA Data Center 001; Schneider Energy Management 001 | metric, document | none | no |
| `document_fact_without_evidence_id` | `completed` | A, B, C, D, E | `unsupported_metric` | 1 | 5 | NVIDIA Data Center 001; Schneider Energy Management 001 | metric, document | currencies; reporting periods; business definitions | yes |
| `wrong_source_evidence_pair` | `completed` | A, B, C, D, E | `unsupported_metric` | 1 | 5 | NVIDIA Data Center 001; Schneider Energy Management 001 | metric, document | currencies; reporting periods; business definitions | yes |

Use these exact call expansions: A = `metric-nvda-pe`; B = `metric-su-pe`; C = `metric-nvda-revenue-error`; D = `document-nvda-growth`; E = `document-su-growth`. The canonical evidence IDs are `NVDA-FY2026-DATA-CENTER-001` and `SU-FY2025-ENERGY-MANAGEMENT-001`. Every row keeps `required_companies=["NVIDIA", "Schneider Electric"]`; an empty value in the table serializes as `[]`. Keep call dependencies in `prerequisite_call_ids`; do not infer order from array position.

Construct the `agent_runs_v1.json` top level from the finalized case bytes so the stored hash can never be an explanatory token. Its exact top-level fields are `schema_version=1`, `fixture_version="agent-runs-v1"`, `dataset_version="agent-cases-v1"`, `dataset_sha256`, and `configurations`; each configuration contains every `RecordedAgentConfiguration` field and six full prediction objects.

```python
case_path = Path("assets/course-data/evaluation/agent_cases_v1.json")
case_sha256 = sha256(case_path.read_bytes()).hexdigest()
assert re.fullmatch(r"[0-9a-f]{64}", case_sha256)
```

Populate both arrays in case-file order with six complete `AgentEvaluationPrediction.model_dump(mode="json")` objects: public plans, observations, trajectory events, evidence-gate results, stage durations, and candidate briefing sections. Use this behavior matrix and the exact configuration metadata stated in `RecordedAgentConfiguration` tests (`lesson11-certified-v1` / `lesson11-regression-fixtures-v0`, `recorded-public-fixture-v1`, the two prompt versions, six steps, one replan):

| Case | `bounded-agent-v1` | `regressed-agent-v0` |
| --- | --- | --- |
| `reference_completed` | exact certified Lesson 11 public fixture | same valid answer and path, retained as the regression control row |
| `unsupported_metric_not_recovered` | typed `execution_stopped`, no briefing | continues without a strategy revision and produces an impermissible briefing |
| `redundant_metric_call` | certified five-attempt path | repeats successful A once before completing |
| `missing_schneider_document` | typed `insufficient_evidence`, no briefing | same missing evidence but emits a briefing after the blocked gate |
| `document_fact_without_evidence_id` | complete exact document citation pairs | one document `CandidateFact` has its `evidence_ids` emptied |
| `wrong_source_evidence_pair` | complete exact document citation pairs | one NVIDIA source is paired with the Schneider evidence ID |

Do not change `plan_execute_graph.py` or its production policies to create these regression records.

- [ ] **Step 4: Implement byte hashing, strict loading, manifest entries, and alignment**

Hash the exact bytes before JSON parsing:

```python
def _verified_json(path: Path, expected_sha256: str) -> object:
    payload = path.read_bytes()
    actual = sha256(payload).hexdigest()
    if not compare_digest(actual, expected_sha256):
        raise ValueError(f"dataset SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    return json.loads(payload.decode("utf-8"))
```

Reject duplicate normalized IDs, unknown fields, blank content, secret-shaped values, malformed arrays/objects, unknown prerequisite call IDs, cycles, prediction IDs that differ from case IDs, differing versions/hashes, and incomplete configuration tables. Preserve the case-file order in the aligned result.

Append an expectation entry to `evaluation_datasets` and add a separate `evaluation_run_fixtures` array so expectations and observations remain visibly distinct. Construct the exact objects from verified bytes:

```python
case_entry = {
    "dataset_version": "agent-cases-v1",
    "path": "assets/course-data/evaluation/agent_cases_v1.json",
    "sha256": sha256(case_path.read_bytes()).hexdigest(),
}
run_path = Path("assets/course-data/evaluation/agent_runs_v1.json")
run_entry = {
    "fixture_version": "agent-runs-v1",
    "dataset_version": "agent-cases-v1",
    "path": "assets/course-data/evaluation/agent_runs_v1.json",
    "sha256": sha256(run_path.read_bytes()).hexdigest(),
}
```

- [ ] **Step 5: Write RED tests for all deterministic metrics and failure ownership**

Add a local helper and these exact tests:

```python
def fixture_pair(case_id: str, configuration_id: str):
    dataset = load_agent_evaluation_dataset(CASES_PATH, expected_sha256=manifest_case_hash())
    runs = load_recorded_agent_runs(RUNS_PATH, cases=dataset, expected_sha256=manifest_run_hash())
    case = next(item for item in dataset.cases if item.case_id == case_id)
    configuration = next(
        item for item in runs.configurations if item.configuration_id == configuration_id
    )
    prediction = next(item for item in configuration.predictions if item.case_id == case_id)
    return case, prediction


def test_tool_call_correctness_is_dependency_aware_not_list_position_based() -> None:
    case, prediction = fixture_pair("reference_completed", "bounded-agent-v1")
    independent = tuple(reversed(prediction.observations[:2])) + prediction.observations[2:]
    changed = prediction.model_copy(update={"observations": independent})
    assert score_agent_case(case, changed).tool_call_correctness.value == 1.0


def test_tool_call_correctness_requires_expected_typed_error_and_replan() -> None:
    case, prediction = fixture_pair("reference_completed", "bounded-agent-v1")
    changed = prediction.model_copy(update={"replan_count": 0})
    score = score_agent_case(case, changed).tool_call_correctness
    assert 0.0 <= score.value < 1.0
    assert "replan" in score.rationale.casefold()


def test_tool_call_efficiency_penalizes_duplicate_budget_and_post_terminal_calls() -> None:
    case, prediction = fixture_pair("redundant_metric_call", "regressed-agent-v0")
    score = score_agent_case(case, prediction)
    assert score.tool_call_efficiency.value < 1.0
    assert score.redundant_tool_calls >= 1


def test_answer_relevance_scores_expected_typed_stop_without_a_briefing() -> None:
    case, prediction = fixture_pair("missing_schneider_document", "bounded-agent-v1")
    assert prediction.briefing is None
    assert score_agent_case(case, prediction).answer_relevance.value == 1.0


def test_answer_completeness_requires_companies_evidence_fact_kinds_comparison_and_limits() -> None:
    case, prediction = fixture_pair("reference_completed", "bounded-agent-v1")
    assert score_agent_case(case, prediction).answer_completeness.value == 1.0
    assert prediction.briefing is not None
    changed = prediction.model_copy(
        update={"briefing": prediction.briefing.model_copy(update={"limitations": ()})}
    )
    assert score_agent_case(case, changed).answer_completeness.value < 1.0


def test_citation_integrity_accepts_metric_source_and_exact_document_pair() -> None:
    case, prediction = fixture_pair("reference_completed", "bounded-agent-v1")
    assert score_agent_case(case, prediction).citation_integrity.value == 1.0


def test_citation_integrity_returns_zero_for_document_fact_without_evidence_id() -> None:
    case, prediction = fixture_pair("document_fact_without_evidence_id", "regressed-agent-v0")
    assert score_agent_case(case, prediction).citation_integrity.value == 0.0


def test_citation_integrity_returns_zero_for_cross_paired_source_and_evidence() -> None:
    case, prediction = fixture_pair("wrong_source_evidence_pair", "regressed-agent-v0")
    assert score_agent_case(case, prediction).citation_integrity.value == 0.0


def test_release_fails_when_required_gate_stop_emits_a_briefing() -> None:
    case, prediction = fixture_pair("missing_schneider_document", "regressed-agent-v0")
    assert prediction.briefing is not None
    assert score_agent_case(case, prediction).release_passed is False


def test_summary_preserves_per_case_failures_and_computes_five_means() -> None:
    dataset = load_agent_evaluation_dataset(CASES_PATH, expected_sha256=manifest_case_hash())
    runs = load_recorded_agent_runs(RUNS_PATH, cases=dataset, expected_sha256=manifest_run_hash())
    configuration = runs.configurations[0]
    aligned = align_cases_and_predictions(
        dataset.cases, configuration.predictions,
        dataset_version=dataset.dataset_version, dataset_sha256=dataset.dataset_sha256,
    )
    scores = tuple(score_agent_case(case, prediction) for case, prediction in aligned)
    summary = summarize_agent_evaluation(
        scores, dataset_version=dataset.dataset_version, dataset_sha256=dataset.dataset_sha256
    )
    assert summary.case_count == 6
    assert set(summary.metric_means) == set(METRIC_NAMES)


def test_failure_classification_assigns_expected_fixture_owners() -> None:
    expected = {
        "unsupported_metric_not_recovered": "replanner",
        "redundant_metric_call": "replanner",
        "missing_schneider_document": "evidence_gate",
        "document_fact_without_evidence_id": "report_writer",
        "wrong_source_evidence_pair": "report_writer",
    }
    for case_id, owner in expected.items():
        case, prediction = fixture_pair(case_id, "regressed-agent-v0")
        scores = score_agent_case(case, prediction)
        assert scores.failure_stage == owner
```

Run:

```bash
.venv/bin/pytest -q tests/test_agent_evaluation.py -k "correctness or efficiency or relevance or completeness or citation or release or summary or classification"
```

Expected: failures report undefined scorer and summary functions.

- [ ] **Step 6: Implement the five scores and stable release decision**

Use these deterministic rules:

- `tool_call_correctness`: match capability plus canonical arguments; score matched expected calls, expected typed errors, expected replan count, and declared dependency order as equally weighted satisfied checks. Unrelated array order does not count.
- `tool_call_efficiency`: start at `1.0`; subtract one normalized penalty unit for each repeated successful signature, call above `max_tool_calls`, call after a terminal evidence-gate/guardrail event, and replan above `expected_replan_count`; divide penalties by `max(1, max_tool_calls)` and clamp to zero.
- `answer_relevance`: for an allowed briefing, average coverage of required companies and the maintained mission dimensions `valuation` and `operating growth`; for `allow_briefing=false`, award `1.0` only when the expected typed status occurs and no briefing exists.
- `answer_completeness`: average expected status, required evidence-ID coverage, required fact-kind coverage, required-company coverage, presence of a cross-company comparison, and required limitation phrase coverage. A correctly typed stop can therefore remain relevant while exposing incomplete evidence.
- `citation_integrity`: validate every candidate fact against successful observations; metric facts require exactly one successful metric source and no evidence ID; document facts require exactly one exact `(source, evidence_id)` pair from a returned hit; aggregate sources must equal the stable ordered union of fact sources. Return zero for any invalid fact or union; return one for the expected no-briefing stop.

Every `MetricScore.rationale` names satisfied and missing public checks. `release_passed` implements the specification's hard blockers exactly: citation integrity must equal `1.0`, and a case with `allow_briefing=false` must have no briefing. The other four scores remain visible diagnostics rather than undocumented hard thresholds. Classify the earliest actionable public owner in this order: dataset alignment (raised before scoring), planner, tool boundary, replanner, evidence gate, report writer; judge ownership is reserved for Task 4 results and never replaces deterministic ownership.

- [ ] **Step 7: Verify both fixture configurations and commit**

```bash
.venv/bin/pytest -q tests/test_agent_evaluation.py
.venv/bin/ruff check src/finai_academy/agent_evaluation.py tests/test_agent_evaluation.py
.venv/bin/python -c 'import hashlib,pathlib; p=pathlib.Path("assets/course-data/evaluation/agent_cases_v1.json"); print(hashlib.sha256(p.read_bytes()).hexdigest())'
.venv/bin/python -c 'import hashlib,pathlib; p=pathlib.Path("assets/course-data/evaluation/agent_runs_v1.json"); print(hashlib.sha256(p.read_bytes()).hexdigest())'
git diff --check
git add src/finai_academy/agent_evaluation.py tests/test_agent_evaluation.py assets/course-data/evaluation/agent_cases_v1.json assets/course-data/evaluation/agent_runs_v1.json assets/course-data/manifest.json
git commit -m "feat: add versioned agent evaluation suite"
```

Expected: six cases load, both configurations align on all six IDs and one hash, all five scores stay within `[0, 1]`, the bounded reference case releases, every intended regression is visible, printed hashes match the manifest exactly, and the commit contains only Task 2 files.

---

### Task 3: Log aligned evaluations and public traces to local MLflow

**Files:**
- Create: `src/finai_academy/mlflow_agent_evaluation.py`
- Create: `tests/test_mlflow_agent_evaluation.py`

**Interfaces:**
- Consumes: Task 2 loaders, `AgentEvaluationCase`, `AgentEvaluationPrediction`, `AgentCaseScores`, `AgentEvaluationSummary`, `align_cases_and_predictions()`, `score_agent_case()`, and `summarize_agent_evaluation()`.
- Produces: `AgentEvaluationConfiguration`, `LocalMLflowStore`, `MLflowAgentEvaluationSummary`, `AgentEvaluationComparison`, `initialize_local_mlflow(tracking_directory: Path | None = None) -> LocalMLflowStore`, `run_mlflow_agent_evaluation(*, tracking_directory: Path | None, experiment_name: str, configuration: AgentEvaluationConfiguration, cases: Sequence[AgentEvaluationCase], predictions: Sequence[AgentEvaluationPrediction]) -> MLflowAgentEvaluationSummary`, and `compare_agent_configurations(summaries: Sequence[MLflowAgentEvaluationSummary]) -> AgentEvaluationComparison`.

- [ ] **Step 1: Write failing local-store and run-contract tests**

Use `tmp_path` and the real MLflow client. Add:

```python
def test_local_store_uses_resolved_sqlite_database_and_local_artifacts(tmp_path: Path) -> None:
    store = initialize_local_mlflow(tmp_path / "lesson12-mlflow")
    assert store.database_path == (tmp_path / "lesson12-mlflow" / "mlflow.db").resolve()
    assert store.tracking_uri == f"sqlite:///{store.database_path}"
    assert store.artifact_directory == (tmp_path / "lesson12-mlflow" / "artifacts").resolve()
    assert store.ui_command == f"mlflow ui --backend-store-uri sqlite:///{store.database_path}"


def test_one_configuration_logs_required_parameters_metrics_and_artifacts(tmp_path: Path) -> None:
    summary = run_fixture_configuration(tmp_path, "bounded-agent-v1")
    run = mlflow.get_run(summary.run_id)
    assert set(run.data.params) >= {
        "configuration_id", "dataset_version", "dataset_sha256", "agent_version",
        "provider", "agent_model", "judge_provider", "judge_model",
        "prompt_version", "max_steps", "max_replans", "scorer_contract_version",
    }
    assert set(run.data.metrics) >= {
        "tool_call_correctness_mean", "tool_call_efficiency_mean",
        "answer_relevance_mean", "answer_completeness_mean",
        "citation_integrity_mean", "mean_tool_calls", "mean_latency_ms",
    }
    assert set(summary.artifact_paths) == {
        "evaluation/case_scores.json",
        "evaluation/failure_rows.json",
        "evaluation/dataset_manifest.json",
    }
```

- [ ] **Step 2: Run MLflow tests and confirm RED**

```bash
.venv/bin/pytest -q tests/test_mlflow_agent_evaluation.py -k "local_store or required_parameters"
```

Expected: collection fails because `finai_academy.mlflow_agent_evaluation` does not exist.

- [ ] **Step 3: Implement configuration, local initialization, and safe payload guards**

Use exact contracts:

```python
class AgentEvaluationConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    configuration_id: str
    dataset_version: str
    dataset_sha256: str
    agent_version: str
    provider: Literal["recorded", "openai", "ollama"]
    agent_model: str
    judge_provider: Literal["none", "openai", "ollama"] = "none"
    judge_model: str = "none"
    prompt_version: str
    max_steps: int = Field(ge=1)
    max_replans: int = Field(ge=0)
    scorer_contract_version: str = "agent-scorers-v1"


class LocalMLflowStore(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    root_directory: Path
    database_path: Path
    artifact_directory: Path
    tracking_uri: str
    ui_command: str
```

Resolve `tracking_directory` from its argument, then `FINAI_MLFLOW_DIR`, then `Path(tempfile.mkdtemp(prefix="finai-lesson12-mlflow-"))`. Reject a non-local or relative artifact URI and never serialize the root directory into MLflow parameters, trace payloads, or artifacts. Reuse Task 1's secret detector recursively on every value sent to MLflow. Wrap backend initialization failure in a `RuntimeError` that includes the resolved database path and sanitized original reason; never continue with only an in-memory score table.

- [ ] **Step 4: Write failing trace-topology, sanitization, flush, and comparison tests**

Add:

```python
def test_each_case_has_one_root_trace_and_required_public_child_spans(tmp_path: Path) -> None:
    summary = run_fixture_configuration(tmp_path, "bounded-agent-v1")
    traces = mlflow.search_traces(run_id=summary.run_id, return_type="list", flush=True)
    assert len(traces) == 6
    assert set(summary.trace_ids) == set(summary.case_scores_by_id)
    required_chain = {"planning", "plan_gate", "replanning", "evidence_gate", "report"}
    for trace in traces:
        names = {span.name for span in trace.data.spans}
        assert required_chain <= names
        assert sum(span.span_type == "TOOL" for span in trace.data.spans) >= 1


def test_trace_contains_only_public_safe_agent_fields(tmp_path: Path) -> None:
    summary = run_fixture_configuration(tmp_path, "bounded-agent-v1")
    payload = serialized_traces(summary.run_id)
    assert "OPENAI_API_KEY" not in payload
    assert "Authorization" not in payload
    assert "/Users/" not in payload
    assert "chain-of-thought" not in payload.casefold()
    assert "unsupported_metric" in payload
    assert "NVDA-FY2026-DATA-CENTER-001" in payload


def test_trace_logging_flushes_before_counting_or_returning(monkeypatch, tmp_path: Path) -> None:
    events: list[str] = []
    monkeypatch.setattr(mlflow, "flush_trace_async_logging", lambda: events.append("flush"))
    summary = run_fixture_configuration(tmp_path, "bounded-agent-v1")
    assert events == ["flush"]
    assert summary.trace_count == 6


def test_logging_rejects_secret_shaped_payload_before_starting_a_run(tmp_path: Path) -> None:
    configuration, cases, predictions = fixture_run_inputs("bounded-agent-v1")
    unsafe = predictions[0].model_copy(update={"agent_model": "sk-secret-shaped-value"})
    with pytest.raises(ValueError, match="secret-shaped"):
        run_mlflow_agent_evaluation(
            tracking_directory=tmp_path, experiment_name="lesson-12-secret-test",
            configuration=configuration, cases=cases,
            predictions=(unsafe,) + predictions[1:],
        )


def test_backend_failure_names_resolved_store_and_fails_verification(monkeypatch, tmp_path: Path) -> None:
    configuration, cases, predictions = fixture_run_inputs("bounded-agent-v1")
    monkeypatch.setattr(mlflow, "set_tracking_uri", Mock(side_effect=OSError("store unavailable")))
    with pytest.raises(RuntimeError, match=r"lesson12-store.*mlflow\.db.*store unavailable"):
        run_mlflow_agent_evaluation(
            tracking_directory=tmp_path / "lesson12-store",
            experiment_name="lesson-12-backend-test", configuration=configuration,
            cases=cases, predictions=predictions,
        )


def test_comparison_rejects_different_case_hashes_and_returns_heatmap_rows(tmp_path: Path) -> None:
    bounded, regressed = run_both_fixture_configurations(tmp_path)
    comparison = compare_agent_configurations((bounded, regressed))
    assert len(comparison.case_metric_rows) == 12
    assert comparison.configuration_ids == ("bounded-agent-v1", "regressed-agent-v0")
```

- [ ] **Step 5: Implement root traces, reconstructed child spans, artifacts, and comparison**

One MLflow run represents one configuration. For each aligned pair, start one root `CHAIN` span named `case:{case_id}` with `run_id`, then emit public child spans in Lesson 11 order:

```python
PHASE_SPAN_TYPES = {
    "planning": "CHAIN",
    "plan_gate": "CHAIN",
    "execution": "TOOL",
    "replanning": "CHAIN",
    "evidence_gate": "CHAIN",
    "report": "CHAIN",
}
```

Map Lesson 11 `policy` events to `plan_gate` and `guardrail` events to the owning preceding phase. Each observation becomes its own `execution:{attempt_id}` TOOL span with safe inputs `{capability, arguments, step_id, attempt_id, plan_revision}` and safe outputs `{status, error_code, evidence_ids, source_references, duration_ms}`. Add stable `attempt_id` and `plan_revision` attributes. Root outputs contain observed/expected status, public plan revisions, evidence-gate result, candidate briefing sections, five scores, failure stage, and case latency. Set `source_kind` to `recorded`, `openai`, or `ollama` from the explicit configuration.

Log three exact JSON artifacts. `case_scores.json` contains all six full score rows; `failure_rows.json` contains case/configuration/failure stage plus five scalar values and rationales; `dataset_manifest.json` contains only dataset version/hash, scorer contract version, case IDs, fixture version, and configuration ID. Flush traces before querying trace IDs/counts and before ending the notebook-visible call.

Return these immutable shapes:

```python
class MLflowAgentEvaluationSummary(BaseModel):
    run_id: str
    experiment_id: str
    tracking_uri: str
    trace_count: int
    trace_ids: dict[str, str]
    parameters: dict[str, str | int]
    metrics: dict[str, float]
    artifact_paths: tuple[str, ...]
    case_scores_by_id: dict[str, AgentCaseScores]
    failure_rows: tuple[dict[str, object], ...]


class AgentEvaluationComparison(BaseModel):
    configuration_ids: tuple[str, ...]
    dataset_version: str
    dataset_sha256: str
    metric_mean_rows: tuple[dict[str, object], ...]
    metric_pass_rows: tuple[dict[str, object], ...]
    case_metric_rows: tuple[dict[str, object], ...]
    tool_call_rows: tuple[dict[str, object], ...]
    latency_rows: tuple[dict[str, object], ...]
    failure_rows: tuple[dict[str, object], ...]
```

- [ ] **Step 6: Run integration coverage and commit**

```bash
.venv/bin/pytest -q tests/test_mlflow_agent_evaluation.py
.venv/bin/pytest -q tests/test_mlflow_evaluation.py
.venv/bin/ruff check src/finai_academy/mlflow_agent_evaluation.py tests/test_mlflow_agent_evaluation.py
git diff --check
git add src/finai_academy/mlflow_agent_evaluation.py tests/test_mlflow_agent_evaluation.py
git commit -m "feat: trace agent evaluations in MLflow"
```

Expected: each configuration produces one SQLite-backed run, six associated root traces with public child spans, all required parameters/metrics/artifacts, no unsafe values, and aligned comparison rows; Lesson 07 MLflow tests remain green.

---

### Task 4: Add explicit optional OpenAI and Ollama MLflow judge routes

**Files:**
- Modify: `src/finai_academy/mlflow_agent_evaluation.py`
- Modify: `tests/test_mlflow_agent_evaluation.py`
- Modify: `.env.example`

**Interfaces:**
- Consumes: MLflow's installed `mlflow.genai.scorers` interface and Task 3 traces/configuration.
- Produces: `JudgeConfiguration`, `JudgeResult`, `JudgeScorerSet`, `load_judge_configuration(environment: Mapping[str, str] = os.environ) -> JudgeConfiguration | None`, `build_optional_genai_scorers(configuration: JudgeConfiguration | None) -> JudgeScorerSet`, and `run_optional_judges(*, run_id: str, configuration: JudgeConfiguration | None, traces: Sequence[object]) -> tuple[JudgeResult, ...]`; the latter logs `evaluation/judge_results.json` on the supplied run without altering deterministic metrics or release status.

- [ ] **Step 1: Write failing explicit-configuration and unavailable-judge tests**

```python
def test_no_explicit_judge_model_returns_four_not_run_results(monkeypatch) -> None:
    logged: list[tuple[str, object]] = []
    monkeypatch.setattr(
        MlflowClient, "log_dict",
        lambda self, run_id, dictionary, artifact_file: logged.append((artifact_file, dictionary)),
    )
    assert load_judge_configuration({"OPENAI_API_KEY": "present-but-ambient"}) is None
    results = run_optional_judges(run_id="test-run", configuration=None, traces=())
    assert [result.scorer_name for result in results] == [
        "ToolCallCorrectness", "ToolCallEfficiency", "RelevanceToQuery", "Completeness"
    ]
    assert all(result.status == "NOT RUN" for result in results)
    assert all(result.score is None for result in results)
    assert logged[0][0] == "evaluation/judge_results.json"


@pytest.mark.parametrize(
    ("uri", "provider", "model"),
    [
        ("openai:/gpt-5-mini", "openai", "gpt-5-mini"),
        ("ollama_chat:/qwen3:8b", "ollama", "qwen3:8b"),
    ],
)
def test_judge_model_uri_selects_exactly_one_explicit_provider(uri, provider, model) -> None:
    config = load_judge_configuration({"FINAI_EVAL_JUDGE_MODEL": uri})
    assert config is not None
    assert (config.provider, config.model_uri, config.model) == (provider, uri, model)


def test_invalid_or_credential_only_judge_configuration_never_falls_back() -> None:
    with pytest.raises(ValueError, match="FINAI_EVAL_JUDGE_MODEL"):
        load_judge_configuration({"FINAI_EVAL_JUDGE_MODEL": "gpt-5-mini"})
```

- [ ] **Step 2: Run judge tests and confirm RED**

```bash
.venv/bin/pytest -q tests/test_mlflow_agent_evaluation.py -k judge
```

Expected: failures report undefined judge configuration and result types.

- [ ] **Step 3: Implement narrow configuration and current MLflow scorer construction**

Use:

```python
class JudgeConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    provider: Literal["openai", "ollama"]
    model_uri: str
    model: str


class JudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    scorer_name: Literal[
        "ToolCallCorrectness", "ToolCallEfficiency", "RelevanceToQuery", "Completeness"
    ]
    provider: Literal["openai", "ollama"] | None
    model: str | None
    mlflow_version: str
    latency_ms: float = Field(ge=0)
    status: Literal["COMPLETED", "ERROR", "NOT RUN"]
    score: float | None = Field(default=None, ge=0, le=1)
    rationale: str


class JudgeScorerSet(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    configuration: JudgeConfiguration | None
    scorers: tuple[object, ...]
```

Import `ToolCallCorrectness`, `ToolCallEfficiency`, `RelevanceToQuery`, and `Completeness` only inside `build_optional_genai_scorers()`. Under the repository's installed MLflow 3.15.1 API, construct each as `ScorerClass(model=configuration.model_uri)`; focused tests inspect the four signatures and protect this call shape. The four classes already import under the existing `evaluation` extra, and the OpenAI/Ollama clients are already bounded by the existing `ai` extra, so this plan does not add another dependency. If the module/API is unavailable or a call times out/errors, emit `NOT RUN` or `ERROR` with the public exception class and sanitized reason; do not substitute a provider, fabricate a score, edit deterministic scores, or change release status. `run_optional_judges()` writes four rows containing scorer name, provider, model, MLflow version, latency, status, score, and rationale to `evaluation/judge_results.json`; only completed scores receive separately prefixed `judge_*` MLflow metrics.

- [ ] **Step 4: Test scorer construction without network and logging separation**

Monkeypatch the four scorer constructors, `MlflowClient.log_dict`, and MLflow version. Assert all constructors receive the explicit URI, the returned names are stable, and the artifact has four provider/model/scorer/version/latency/status rows. Assert an `ERROR` or `NOT RUN` result leaves `citation_integrity`, the five deterministic run metrics, and deterministic `release_passed` unchanged.

Run:

```bash
.venv/bin/pytest -q tests/test_mlflow_agent_evaluation.py -k judge
.venv/bin/ruff check src/finai_academy/mlflow_agent_evaluation.py tests/test_mlflow_agent_evaluation.py
```

Expected: all judge tests pass without a daemon, network, or API key.

- [ ] **Step 5: Document explicit judge examples and commit**

Append disabled examples to `.env.example`:

```dotenv
# Optional Lesson 12 MLflow judge. Leave unset for the deterministic offline route.
# FINAI_EVAL_JUDGE_MODEL=openai:/gpt-5-mini
# FINAI_EVAL_JUDGE_MODEL=ollama_chat:/qwen3:8b
```

Then:

```bash
git add src/finai_academy/mlflow_agent_evaluation.py tests/test_mlflow_agent_evaluation.py .env.example
git commit -m "feat: add explicit agent evaluation judges"
```

Expected: the committed default selects no judge; both documented values select exactly one provider; core deterministic tests require no provider client beyond the existing evaluation/AI extras.

---

### Task 5: Build and execute the 40-minute visual notebook

**Files:**
- Create: `scripts/build_lesson12_notebook.py`
- Create: `notebooks/12_evaluating_agentic_systems.ipynb`
- Create: `tests/test_lesson12_assets.py`
- Modify: `tests/test_course_manifest.py`
- Modify: `course.yml`

**Interfaces:**
- Consumes: Task 1-4 public APIs; `FinancialMcpPlanningExecutor`, `run_plan_execute()`, and recorded Lesson 11 policies for one real offline reference run; the existing notebook validator/executor.
- Produces: deterministic `build_notebook() -> nbformat.NotebookNode`, canonical Lesson 12 manifest paths, 24-28 source cells, six or more runtime PNGs, two aligned MLflow runs, visible score/trace/failure tables, and exactly one `LESSON_12_PASS`.

- [ ] **Step 1: Write failing source-notebook and staged-manifest tests**

Require:

```python
def test_lesson12_notebook_is_output_free_stable_and_contains_the_teaching_contract() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    assert notebook.metadata["finai"]["expected_runtime_minutes"] == 40
    assert 24 <= len(notebook.cells) <= 28
    assert len({cell.id for cell in notebook.cells}) == len(notebook.cells)
    assert all(not cell.get("outputs") for cell in notebook.cells if cell.cell_type == "code")
    assert all(cell.get("execution_count") is None for cell in notebook.cells if cell.cell_type == "code")
    assert nbformat.writes(_build_notebook()) == nbformat.writes(notebook)
    assert source.count("LESSON_12_PASS") == 1
    assert "—" not in source
    for marker in (
        "bounded-agent-v1", "regressed-agent-v0", "agent-cases-v1",
        "tool_call_correctness", "tool_call_efficiency", "answer_relevance",
        "answer_completeness", "citation_integrity", "FINAI_EVAL_JUDGE_MODEL",
        "openai:/", "ollama_chat:/", "mlflow ui --backend-store-uri sqlite:///",
    ):
        assert marker in source
    assert "docker" not in executable_source(notebook).casefold()
```

Add `test_implemented_lesson_twelve_notebook_exists_before_chapter_and_deck()` to `tests/test_course_manifest.py`. It must assert title `Evaluating agentic systems with MLflow`, 14:30-15:30, and only the notebook's existence during this task.

- [ ] **Step 2: Run source tests and confirm RED**

```bash
.venv/bin/pytest -q tests/test_lesson12_assets.py -k "output_free or teaching_contract"
.venv/bin/pytest -q tests/test_course_manifest.py -k twelve
```

Expected: Lesson 12 asset test cannot find the builder/notebook; manifest test fails because `course.yml` still points to the pre-spec filenames.

- [ ] **Step 3: Implement the stable 27-cell notebook sequence**

Follow the existing stable-ID builder pattern and emit exactly these 27 cells:

1. `lesson12-000`: title, 14:30-15:30 slot, 12+40+8 timing, outcome, prerequisites, visible outputs, and public-state boundary.
2. `lesson12-001`: learning objectives and Lesson 11-to-12 capstone increment.
3. `lesson12-002`: offline/OpenAI/Ollama selection; deterministic route always runs; judge URI remains explicit.
4. `lesson12-003`: load manifest, verify `agent-cases-v1`, load both recorded configurations, initialize local SQLite.
5. `lesson12-004`: Figure 1, trajectory-versus-answer architecture.
6. `lesson12-005`: display six case IDs, version/hash, and one complete expectation row.
7. `lesson12-006`: run the real Lesson 11 offline mission once through `FinancialMcpPlanningExecutor`.
8. `lesson12-007`: convert the real result and assert its public signature equals `reference_completed`: initial/final call signatures, final status, evidence IDs, facts, provenance kinds, sources, and one replan.
9. `lesson12-008`: display the public plan, trajectory, typed error, gate result, and briefing.
10. `lesson12-009`: Figure 2, expected-versus-observed dependency-aware tool-call sequence.
11. `lesson12-010`: compute and explain tool-call correctness.
12. `lesson12-011`: compute and explain tool-call efficiency.
13. `lesson12-012`: Figure 3, one public trace timeline with phase, attempt/revision, status, and latency.
14. `lesson12-013`: compute answer relevance and completeness.
15. `lesson12-014`: enforce citation integrity and show metric/document rules.
16. `lesson12-015`: log all six `bounded-agent-v1` cases to local MLflow.
17. `lesson12-016`: log all six `regressed-agent-v0` cases against the identical version/hash.
18. `lesson12-017`: display run IDs, trace IDs, five means/pass counts, tool counts, mean/max latency, and failure ownership.
19. `lesson12-018`: Figure 4, per-case five-metric heatmap.
20. `lesson12-019`: Figure 5, aligned configuration comparison with all five means.
21. `lesson12-020`: inspect one answer-good/path-bad case and one path-good/answer-incomplete case.
22. `lesson12-021`: Figure 6, failure-diagnosis matrix for planner/tool/replanner/gate/report/dataset/judge.
23. `lesson12-022`: optional MLflow judge configuration and explicit `NOT RUN` display.
24. `lesson12-023`: print resolved database path, artifact directory, expected UI URL, and exact UI command.
25. `lesson12-024`: deterministic verification assertions and the single exact `LESSON_12_PASS` print.
26. `lesson12-025`: five-question knowledge check plus optional custom-scorer challenge conditions.
27. `lesson12-026`: recap and capstone handoff without assuming the capstone architecture.

The real reference comparison must derive canonical signatures and facts from public fields. It must not compare wall-clock latency or private runtime objects. Use the dataset hash from verified file bytes for both run configurations.

- [ ] **Step 4: Update the canonical course paths and generate the source notebook**

Change the Lesson 12 `course.yml` entry to:

```yaml
  - id: "12"
    title: Evaluating agentic systems with MLflow
    day: 2
    start: "14:30"
    end: "15:30"
    deck: decks/12-evaluating-agentic-systems.pptx
    notebook: notebooks/12_evaluating_agentic_systems.ipynb
    chapter: chapters/12-evaluating-agentic-systems.md
    capstone_increment: Versioned agent trajectory and answer evaluation suite
```

Generate and validate source determinism:

```bash
.venv/bin/python scripts/build_lesson12_notebook.py
cp notebooks/12_evaluating_agentic_systems.ipynb /private/tmp/lesson12-first.ipynb
.venv/bin/python scripts/build_lesson12_notebook.py
cmp /private/tmp/lesson12-first.ipynb notebooks/12_evaluating_agentic_systems.ipynb
.venv/bin/python scripts/validate_notebooks.py notebooks/12_evaluating_agentic_systems.ipynb
```

Expected: `cmp` emits no output; validator reports `1 notebook passed the course notebook contract.`

- [ ] **Step 5: Execute offline and inspect every required output**

```bash
.venv/bin/python scripts/execute_notebooks.py notebooks/12_evaluating_agentic_systems.ipynb --mode offline --output-dir /private/tmp/finai-lesson12-offline
.venv/bin/pytest -q tests/test_lesson12_assets.py -k notebook
```

The execution test must assert six cases per configuration, two distinct MLflow run IDs on one dataset version/hash, twelve traces total, all five score columns, a visible per-case table, a visible public trace, at least six PNG outputs, the exact SQLite/UI command, explicit OpenAI and Ollama judge instructions, and exactly one `LESSON_12_PASS`. Extract and inspect every PNG full-size; correct clipped labels, weak contrast, ambiguous arrows, unreadable annotations, or tables represented only as counts.

- [ ] **Step 6: Respect the staged asset boundary and commit**

At this point the canonical notebook exists but the Task 6 chapter and Task 7 deck intentionally do not. Run only the staged manifest assertion and targeted Lesson 12 code/notebook tests; defer `scripts/validate_repo.py` and the full repository suite until the chapter and deck exist.

```bash
.venv/bin/pytest -q tests/test_agent_evaluation.py tests/test_mlflow_agent_evaluation.py tests/test_lesson12_assets.py -k "not deck and not chapter and not indexes"
.venv/bin/pytest -q tests/test_course_manifest.py -k "not repository_validator"
.venv/bin/ruff check scripts/build_lesson12_notebook.py tests/test_lesson12_assets.py tests/test_course_manifest.py
git add scripts/build_lesson12_notebook.py notebooks/12_evaluating_agentic_systems.ipynb tests/test_lesson12_assets.py tests/test_course_manifest.py course.yml
git commit -m "lesson: add visual MLflow agent evaluation notebook"
```

Expected: staged tests pass; the full repository validator is deliberately deferred only because the newly canonical chapter/deck paths are not created until Tasks 6-7.

---

### Task 6: Write the instructor chapter, onboarding, and completed-course indexes

**Files:**
- Create: `chapters/12-evaluating-agentic-systems.md`
- Modify: `chapters/README.md`
- Modify: `notebooks/README.md`
- Modify: `decks/README.md`
- Modify: `README.md`
- Modify: `docs/getting-started.md`
- Modify: `tests/test_lesson12_assets.py`

**Interfaces:**
- Consumes: Task 5 cell IDs, figures, commands, score tables, trace labels, pass marker, and Task 4 explicit judge routes.
- Produces: the exact 14:30-15:30 instructor route, recovery playbook, answer key, local MLflow instructions, capstone handoff, and discoverable Lesson 12 links.

- [ ] **Step 1: Add failing chapter, onboarding, and index contracts**

Require the chapter to contain the exact timing labels, all 27 cell IDs, six figure purposes, six case IDs, both configuration IDs, five metric names, local SQLite/UI command, `No-network fallback`, `Skip if late`, OpenAI/Ollama `NOT RUN`, knowledge-check answers, `LESSON_12_PASS`, and the sentence `full Lesson 12 route is ready for an instructor-led offline test class`.

Require exact links:

```text
chapters/README.md -> [Evaluating agentic systems with MLflow](12-evaluating-agentic-systems.md)
notebooks/README.md -> [Evaluating agentic systems with MLflow](12_evaluating_agentic_systems.ipynb)
decks/README.md -> [Evaluating agentic systems with MLflow](12-evaluating-agentic-systems.pptx)
README.md -> Lesson 12 chapter, notebook, and concept deck links
```

Reject stale text stating Lesson 12 remains planned. Require `docs/getting-started.md` to state that Docker and the browser UI are not required for Lesson 12 and that judge URIs are explicit.

- [ ] **Step 2: Run chapter/index tests and confirm RED**

```bash
.venv/bin/pytest -q tests/test_lesson12_assets.py -k "chapter or onboarding or indexes"
```

Expected: failures report the absent chapter, absent links, and stale planned status.

- [ ] **Step 3: Write the complete instructor chapter**

Include:

- purpose, prerequisites, `uv sync --extra ai --extra evaluation --extra dev`, offline execution, and exact local UI command;
- the 12-minute nine-slide script from the spec;
- the 40-minute notebook pacing table mapped to `lesson12-000` through `lesson12-026` and visible outputs;
- the 8-minute verification/debrief with expected learner answers;
- exact six-case diagnostic intent and both aligned configuration contracts;
- formulas and rationale interpretation for all five deterministic metrics;
- the strict metric-fact/document-fact/aggregate-source citation rules;
- how to read one run, root trace, phase span, tool attempt, failure row, and release decision;
- recovery for dataset/hash mismatch, local SQLite failure, trace/run association failure, missing provider, judge timeout/disagreement, and suspected secret/private-data exposure;
- no-network fallback that still runs the full deterministic route;
- a skip-if-late route that preserves case hashing, real Lesson 11 alignment, trajectory/answer separation, citation gate, one failed trace, and the pass marker;
- the five knowledge-check answers and a bounded custom-scorer challenge solution contract;
- safety: public serializable state only, read-only analysis, no trading/portfolio mutation/price target/investment recommendation;
- truthful provider language: missing routes are `NOT RUN`, never passed by configuration inspection; and
- capstone discussion inputs without choosing a final application architecture.

- [ ] **Step 4: Update onboarding and all four indexes**

State that Lessons 08-12 are ready for an instructor-led offline test class and remove the obsolete planned wording. In `docs/getting-started.md`, add a Lesson 12 section that prints the resolved `FINAI_MLFLOW_DIR`, uses local SQLite/artifacts, lists `http://127.0.0.1:5000` as optional, and documents both explicit judge URIs without enabling either.

Run:

```bash
.venv/bin/pytest -q tests/test_lesson12_assets.py -k "chapter or onboarding or indexes"
.venv/bin/ruff check .
git diff --check
```

Expected: chapter/onboarding/index tests pass, Ruff is clean, no stale Lesson 12 planned status remains, and no visible em dash exists in the Lesson 12 chapter.

- [ ] **Step 5: Commit instructor materials**

```bash
git add chapters/12-evaluating-agentic-systems.md chapters/README.md notebooks/README.md decks/README.md README.md docs/getting-started.md tests/test_lesson12_assets.py
git commit -m "docs: add lesson 12 instructor route"
```

Expected: the commit exposes a complete teachable route but does not claim that the deck or final independent review has passed.

---

### Task 7: Create and visually certify the nine-slide concept deck

**Files:**
- Create: `decks/12-evaluating-agentic-systems.pptx`
- Modify: `tests/test_lesson12_assets.py`
- Use ignored QA workspace: `.artifacts/lesson12-deck/`

**Interfaces:**
- Consumes: Lesson 12 chapter terminology, notebook figures/score patterns, and the visual system from `decks/10-financial-mcp.pptx` and `decks/11-plan-and-execute-analyst.pptx`.
- Produces: exactly nine editable, sourced, visually reviewed slides with exact footer, three comparison tables, original diagrams, no visible em dash, and no overflow, collision, or unfinished template content.

- [ ] **Step 1: Invoke the presentation workflow and add failing structural tests**

Read the presentation skill completely before creating or editing the deck. Add tests that open the PPTX package and require:

```python
assert len(slide_names) == 9
assert len(notes_names) == 9
assert visible_text.count("First Finance - Arnaud Demes") == 9
assert "—" not in visible_text
assert notes_text.count("[Sources]") == 9
assert notes_text.count("[/Sources]") == 9
```

Require visible markers `Evaluating Agentic Systems with MLflow`, `SAME ANSWER`, `DIFFERENT PATH`, `TRAJECTORY`, `ANSWER`, `agent-cases-v1`, `MLFLOW RUN`, `ROOT TRACE`, `TOOL`, all five metric labels, `DETERMINISTIC RELEASE GATE`, `LLM JUDGE`, `CITATION INTEGRITY`, and `CAPSTONE`. Parse slides 3, 7, and 8 and assert each contains an actual PowerPoint table element. Require source notes to cite the Lesson 12 chapter and directly relevant official MLflow tracing/judge/evaluation pages.

- [ ] **Step 2: Run deck tests and confirm RED**

```bash
.venv/bin/pytest -q tests/test_lesson12_assets.py -k deck
```

Expected: failure because `decks/12-evaluating-agentic-systems.pptx` is absent.

- [ ] **Step 3: Build the exact nine-slide narrative**

Use the Lesson 10/11 page size, theme, footer geometry, typography hierarchy, and source-note convention. Create original vector content for:

1. `Evaluate the path and the answer` - outcome, 14:30-15:30, 12+40+8.
2. `Same answer, different path` - two trajectories reaching plausible copy, one redundant/unsafe.
3. `Trajectory quality and answer quality are separate` - concise comparison table.
4. `A versioned case defines the exam` - case ID, mission, dependency-aware expected calls, evidence, gate, and budget.
5. `One configuration is one MLflow run` - cases to pure scorers to root traces/artifacts/scorecard architecture.
6. `Read one trace from plan to report` - planning, plan gate, TOOL attempts, replanning, evidence gate, report with one typed error.
7. `Deterministic rules decide release` - table comparing deterministic scorers with optional provider judges.
8. `Score patterns assign failure ownership` - table with answer-good/path-bad, path-good/answer-incomplete, citation failure, and correct stop.
9. `Release the evidence, then design the capstone` - citation gate, no forbidden briefing after stop, six aligned cases, capstone handoff.

Keep visible copy concise. Each mechanism slide uses an original diagram; slides 3, 7, and 8 use native editable tables. Every notes block includes the instructor purpose, the planned timing, `[Sources]`, the chapter path, directly relevant source URLs, and `[/Sources]`.

- [ ] **Step 4: Render, montage, and inspect all nine slides**

Use the presentation workflow to render the deck into `.artifacts/lesson12-deck/rendered/`, create a montage, inspect it, then inspect each slide PNG at full size. Correct overflow, collisions, clipping, low contrast, dense copy, inconsistent arrows, unexplained abbreviations, footer drift, and misleading metric hierarchy.

- [ ] **Step 5: Run automated structural, overflow, and template QA**

Run the presentation runtime's `slides_test.py`, template-plan validator, and template-fidelity validator using Lesson 11 as the starter visual reference, followed by:

```bash
.venv/bin/pytest -q tests/test_lesson12_assets.py -k deck
.venv/bin/python scripts/validate_repo.py
git diff --check
```

Expected: exactly nine slides and notes blocks, three native comparison tables, nine exact footers, directly relevant sources, zero overflow/collision/fidelity issues, no visible em dash, and repository structure valid now that all canonical Lesson 12 assets exist.

- [ ] **Step 6: Commit the certified deck**

```bash
git add decks/12-evaluating-agentic-systems.pptx tests/test_lesson12_assets.py
git commit -m "docs: add lesson 12 MLflow evaluation deck"
```

Expected: ignored render/QA artifacts stay uncommitted; the commit contains only the deck and its structural tests.

---

### Task 8: Run full certification and enforce the independent 9.5 quality gate

**Files:**
- Create: `docs/reviews/lesson-12-readiness.md`
- Modify: only files required to resolve failures found by certification.

**Interfaces:**
- Consumes: all Lesson 12 code, versioned data, MLflow runs/traces, optional judge routes, notebook, chapter, deck, indexes, tests, and repository validators.
- Produces: exact offline evidence, separately labelled provider/rehearsal coverage, independent findings, lesson-quality score `>= 9.5/10`, no unresolved Important/Critical finding, release decision, and a clean verified commit.

- [ ] **Step 1: Run the complete targeted package**

```bash
.venv/bin/pytest -q tests/test_agent_evaluation.py tests/test_mlflow_agent_evaluation.py tests/test_lesson12_assets.py tests/test_course_manifest.py tests/test_plan_execute_graph.py tests/test_research_planning.py tests/test_lesson11_assets.py tests/test_mlflow_evaluation.py
.venv/bin/ruff check .
.venv/bin/python scripts/validate_notebooks.py notebooks/12_evaluating_agentic_systems.ipynb
```

Expected: all targeted tests pass, Lesson 07 and 11 regressions remain green, Ruff reports no issues, and the Lesson 12 source notebook passes the repository contract.

- [ ] **Step 2: Execute a fresh offline notebook and inspect its complete evidence**

```bash
.venv/bin/python scripts/execute_notebooks.py notebooks/12_evaluating_agentic_systems.ipynb --mode offline --output-dir /private/tmp/finai-lesson12-certification
```

Record exact case/run/trace counts, dataset version/hash, real Lesson 11 signature match, five metric means and pass counts, per-case failures, SQLite path and UI command, PNG count, and marker count. Inspect all six or more PNGs full-size plus visible scorecard, cited briefing, trace table, and failure rows. Required evidence: two run IDs, six cases each, twelve traces total, one dataset/hash, at least six readable PNGs, and one `LESSON_12_PASS`.

- [ ] **Step 3: Exercise optional providers only when actually available**

If Ollama responds and the configured judge model exists:

```bash
FINAI_EVAL_JUDGE_MODEL=ollama_chat:/qwen3:8b .venv/bin/python scripts/execute_notebooks.py notebooks/12_evaluating_agentic_systems.ipynb --mode offline --output-dir /private/tmp/finai-lesson12-ollama-judge
```

If `OPENAI_API_KEY` is configured:

```bash
FINAI_EVAL_JUDGE_MODEL=openai:/gpt-5-mini .venv/bin/python scripts/execute_notebooks.py notebooks/12_evaluating_agentic_systems.ipynb --mode offline --output-dir /private/tmp/finai-lesson12-openai-judge
```

Record observed scorer names, provider/model, MLflow version, status, latency, and rationales. If unavailable, record `NOT AVAILABLE / NOT RUN` for Ollama or `NOT CONFIGURED / NOT RUN` for OpenAI. Do not award provider credit from imports, settings, credentials, or static source.

- [ ] **Step 4: Re-run deck QA and inspect the final presentation**

Re-render the committed deck, run overflow/template-plan/template-fidelity checks, inspect the montage and all nine slides full-size, and run:

```bash
.venv/bin/pytest -q tests/test_lesson12_assets.py -k deck
```

Expected: all structural and visual checks remain clean on the final deck bytes; notes, tables, diagrams, footer, and visible-copy constraints are verified.

- [ ] **Step 5: Run the full repository certification**

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/python scripts/validate_notebooks.py
.venv/bin/python scripts/validate_repo.py
git diff --check
git status --short
```

Expected: the full test suite, Ruff, every source notebook, repository paths, and whitespace checks pass. Before the readiness report is added, the worktree is clean.

- [ ] **Step 6: Conduct the independent lesson-quality review**

Give an independent reviewer the approved spec, final notebook execution, chapter, deck renders, relevant code/tests, and certification outputs. Require findings classified as Critical, Important, Minor, or Observation and this offline lesson-quality rubric:

```text
Technical correctness and safety: 25%
Learner usability and pacing: 20%
Conceptual progression: 20%
Offline reliability and diagnosability: 15%
Notebook and deck visual quality: 10%
Repository and test quality: 10%
```

The gate passes only when the weighted lesson-quality score is at least `9.5/10`, there is no unresolved Critical or Important finding, the six acceptance cases and five metrics are covered, and all certification commands are evidenced. Provider availability and timed rehearsal appear in separate sections and contribute no points to this rubric. Resolve any gate failure, rerun affected checks, and repeat the independent review until the threshold is observed.

- [ ] **Step 7: Write the readiness report from observed evidence**

Use these exact sections:

```text
Scope
Environment
Versioned data and alignment
Pure scorer results
Local MLflow runs and traces
Offline notebook execution
Notebook visual review
Deck automated and visual review
Ollama judge coverage
OpenAI judge coverage
Timed rehearsal coverage
Full repository regression
Independent findings and resolutions
Lesson-quality rubric and weighted score
Known qualifications
Decision
```

State an offline release only if the independent gate is at least 9.5 with no unresolved Important/Critical finding. Never claim optional judge or rehearsal evidence that was not observed.

- [ ] **Step 8: Commit readiness evidence and verify the exact commit**

```bash
git add docs/reviews/lesson-12-readiness.md
git commit -m "docs: certify lesson 12 MLflow evaluation"
git show --check HEAD
.venv/bin/pytest -q tests/test_agent_evaluation.py tests/test_mlflow_agent_evaluation.py tests/test_lesson12_assets.py tests/test_course_manifest.py
.venv/bin/ruff check .
.venv/bin/python scripts/validate_repo.py
git status --short
```

Expected: the readiness report contains only observed results, the exact commit passes focused tests/Ruff/repository validation, `git show --check` is clean, and the worktree is clean.

---

## Execution Order and Review Gates

- Review Task 1 for type safety and the one-way `PlanExecuteResult` conversion before accepting regression fixtures.
- Review Task 2 for byte-level versioning, exact six-by-two alignment, metric formulas, citation pairing, and release behavior before MLflow work.
- Review Task 3 for safe local persistence and truthful public trace topology before optional judges.
- Review Task 4 for explicit provider selection and deterministic-score isolation before notebook authoring.
- Review Task 5 as a runnable notebook increment; its canonical notebook may temporarily precede the Task 6 chapter and Task 7 deck, so only staged validation is valid there.
- Review Task 6 for a teachable 12+40+8 route and truthful indexes.
- Review Task 7 for structural and full-size visual quality before final certification.
- Review Task 8 only from fresh observed evidence and enforce the independent 9.5/10 gate.
