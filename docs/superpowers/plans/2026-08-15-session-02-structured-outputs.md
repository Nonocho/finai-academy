# Session 02 Structured Outputs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify the thirty-minute Prompt Engineering and Structured Outputs lesson around the capstone's provider-neutral `AnalystBrief` contract.

**Architecture:** Extend the existing capstone Pydantic models with finance-specific validators, provide one deterministic offline `StructuredModel`, and teach the same `AnalystBriefService` used by Ollama and OpenAI. Keep the notebook as the student-facing executable narrative and the micro-deck as the ten-minute conceptual setup.

**Tech Stack:** Python 3.11+, Pydantic 2, LangChain provider adapters, Jupyter/nbclient, pytest, PowerPoint generated with `@oai/artifact-tool`.

## Global Constraints

- Preserve every unrelated working-tree modification.
- Use `First Finance - Arnaud Demes` in student-facing footers and signatures.
- Use NVIDIA fiscal 2026 evidence from Lesson 1; do not introduce new live data dependencies.
- Support offline, Ollama, and OpenAI execution through the existing provider boundary.
- Never embed API keys or absolute user paths.
- Keep the live lesson inside the 10:00-10:30 delivery window.

---

### Task 1: Enforce the financial output contract

**Files:**
- Modify: `tests/test_capstone_briefing.py`
- Modify: `src/finai_academy/capstone/models.py`

**Interfaces:**
- Consumes: `AnalystFinding`, `EvidenceType`, and `FindingCategory`.
- Produces: Pydantic validation that requires excerpts for reported facts and management claims and rationale for interpretations.

- [ ] **Step 1: Write the failing reported-fact test**

```python
def test_reported_fact_requires_a_source_excerpt() -> None:
    with pytest.raises(ValidationError, match="source_excerpt"):
        AnalystFinding(
            statement="Revenue increased.",
            category=FindingCategory.KEY_RESULT,
            evidence_type=EvidenceType.REPORTED_FACT,
        )
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `uv run pytest tests/test_capstone_briefing.py::test_reported_fact_requires_a_source_excerpt -q`

Expected: FAIL because the current model accepts the missing excerpt.

- [ ] **Step 3: Add the minimal model-level validator**

Use a Pydantic `model_validator(mode="after")` on `AnalystFinding` and raise a
field-specific `ValueError` when `evidence_type` is `REPORTED_FACT` or
`MANAGEMENT_CLAIM` and `source_excerpt` is empty.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run: `uv run pytest tests/test_capstone_briefing.py::test_reported_fact_requires_a_source_excerpt -q`

Expected: PASS.

- [ ] **Step 5: Write the failing interpretation test**

```python
def test_interpretation_requires_a_rationale() -> None:
    with pytest.raises(ValidationError, match="rationale"):
        AnalystFinding(
            statement="Growth is concentrated.",
            category=FindingCategory.RISK,
            evidence_type=EvidenceType.INTERPRETATION,
        )
```

- [ ] **Step 6: Run the focused test and verify RED**

Run: `uv run pytest tests/test_capstone_briefing.py::test_interpretation_requires_a_rationale -q`

Expected: FAIL because the current model accepts the missing rationale.

- [ ] **Step 7: Extend the validator minimally**

Reject `INTERPRETATION` findings when `rationale` is empty, without imposing
requirements on other evidence types.

- [ ] **Step 8: Run the complete capstone test file**

Run: `uv run pytest tests/test_capstone_briefing.py -q`

Expected: all tests pass.

### Task 2: Provide a deterministic structured-output classroom fixture

**Files:**
- Modify: `tests/test_lesson_support.py`
- Modify: `src/finai_academy/lesson_support.py`

**Interfaces:**
- Consumes: `StructuredModel.generate(system_prompt, user_prompt, response_model)`.
- Produces: `RecordedStructuredModel`, which returns a deterministic instance of the requested Pydantic model.

- [ ] **Step 1: Write the failing fixture test**

Instantiate `RecordedStructuredModel`, call `generate` with `AnalystBrief`, and
assert the returned brief is typed, names NVIDIA, includes at least one sourced
reported fact, and contains a caveat.

- [ ] **Step 2: Run the fixture test and verify RED**

Run: `uv run pytest tests/test_lesson_support.py::test_recorded_structured_model_returns_a_valid_financial_brief -q`

Expected: FAIL because `RecordedStructuredModel` does not exist.

- [ ] **Step 3: Implement the fixture**

Return a literal `AnalystBrief` payload through `response_model.model_validate`.
Use the Lesson 1 NVIDIA evidence card, including fiscal 2026 revenue, Data Center
revenue, a growth-concentration interpretation with rationale, one open
question, and one caveat.

- [ ] **Step 4: Run the lesson-support tests**

Run: `uv run pytest tests/test_lesson_support.py -q`

Expected: all tests pass.

### Task 3: Author and execute the student notebook

**Files:**
- Create: `notebooks/02_prompts_and_structured_outputs.ipynb`
- Create: `chapters/02-prompts-and-structured-outputs.md`
- Modify: `tests/test_notebook_contracts.py`
- Preserve: `notebooks/02-prompting-and-structured-outputs.ipynb`

**Interfaces:**
- Consumes: `AnalystBriefService`, `RecordedStructuredModel`, `Settings`, and `create_structured_model`.
- Produces: a validated NVIDIA `AnalystBrief` and the marker `PASS — structured financial brief verified`.

- [ ] **Step 1: Write the failing offline notebook contract test**

Execute `notebooks/02_prompts_and_structured_outputs.ipynb` through
`scripts/execute_notebooks.py --mode offline` and assert the stored execution
contains both `Validation caught the unsupported candidate` and
`PASS — structured financial brief verified`.

- [ ] **Step 2: Run the focused notebook test and verify RED**

Run: `uv run pytest tests/test_notebook_contracts.py::test_structured_outputs_offline_run_reaches_the_validation_target -q`

Expected: FAIL because the canonical notebook does not exist.

- [ ] **Step 3: Create the notebook narrative**

Create a clean notebook with unique cell IDs, Python 3 kernelspec metadata,
`finai.expected_runtime_minutes: 20`, and these learner-facing sections:
Learning objectives, Where this fits, prompt anatomy, Failure lab, Pydantic
contract, provider selection, structured generation, Verification, Challenge,
Capstone integration, and Recap.

- [ ] **Step 4: Create the instructor chapter**

Document the 10/20-minute timing, learning objectives, facilitation cues,
expected failure, verification rubric, provider commands, challenge answer, and
transition into context engineering.

- [ ] **Step 5: Validate the source notebook**

Run: `uv run python scripts/validate_notebooks.py notebooks/02_prompts_and_structured_outputs.ipynb`

Expected: `1 notebook passed the course notebook contract.`

- [ ] **Step 6: Execute offline and verify GREEN**

Run: `uv run python scripts/execute_notebooks.py notebooks/02_prompts_and_structured_outputs.ipynb --mode offline --output-dir .artifacts/session-02/offline`

Expected: PASS and both required notebook markers in the executed output.

- [ ] **Step 7: Run the focused notebook test**

Run: `uv run pytest tests/test_notebook_contracts.py::test_structured_outputs_offline_run_reaches_the_validation_target -q`

Expected: PASS.

### Task 4: Build and inspect the micro-deck

**Files:**
- Create: `decks/02-prompts-and-structured-outputs.pptx`
- Use as visual source: `decks/01-model-gateway.pptx`
- Modify: `course.yml` only if its path differs from the final output.

**Interfaces:**
- Consumes: the existing Session 1 visual system and the Session 2 teaching narrative.
- Produces: six 16:9 slides with speaker notes, source blocks, and `First Finance - Arnaud Demes` footers.

- [ ] **Step 1: Inspect every Session 1 source slide**

Run the presentation skill's template inspection tool and review all rendered
slides, layout JSON files, inspect records, master/layout structure, fonts, and
placeholders.

- [ ] **Step 2: Create the template audit and frame map**

Map each of the six output slides to one Session 1 source slide and classify
every edited inherited element explicitly.

- [ ] **Step 3: Create the starter deck**

Run the template starter helper and verify that all mapped slides are duplicated
from the source deck before editing.

- [ ] **Step 4: Edit with `@oai/artifact-tool`**

Create the six-slide progression specified in the approved design. Preserve the
source deck's typography and footer. Add `[Sources]` notes for the OpenAI
Structured Outputs guide and NVIDIA evidence where used.

- [ ] **Step 5: Render and inspect each slide**

Export every slide at full size plus a montage. Check hierarchy, wrapping,
overlap, clipping, footer consistency, and source notes.

- [ ] **Step 6: Run structural deck QA**

Run the presentation overflow test and template-fidelity check. Correct every
unintended issue before retaining the final PPTX.

### Task 5: Complete Lesson 2 verification

**Files:**
- Verify all files changed by Tasks 1-4.

**Interfaces:**
- Consumes: the complete Lesson 2 artifact set.
- Produces: evidence that code, notebook, and deck satisfy the accepted design.

- [ ] **Step 1: Run focused tests**

Run: `uv run pytest tests/test_capstone_briefing.py tests/test_lesson_support.py tests/test_notebook_contracts.py -q`

- [ ] **Step 2: Run the complete test suite**

Run: `uv run pytest -q`

- [ ] **Step 3: Re-run notebook validation and offline execution**

Run the canonical Lesson 2 notebook validator and executor from Task 3 using a
fresh output directory.

- [ ] **Step 4: Record live-provider status accurately**

Run Ollama if its server and configured model are available. Run OpenAI only
when `OPENAI_API_KEY` is available. Otherwise record the exact skipped condition
without claiming a live-provider pass.

- [ ] **Step 5: Review the accepted design line by line**

Confirm the thirty-minute pacing, NVIDIA continuity, shared Pydantic contract,
failure lab, deterministic verification, dual-provider setup, and branded deck
are all represented in the delivered artifacts.
