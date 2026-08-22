# Lesson 11 delivery readiness

Review date: 2026-08-22
Lesson: Plan-and-execute financial analyst

## Scope

This review covers the Lesson 11 notebook, chapter, plan-execute graph and
policies, real local Lesson 10 MCP boundary, nine-slide deck, indexes, and
course validation. The maintained offline route is the certification baseline.
Provider credit is awarded only for commands executed during this review.

## Environment

macOS on Apple Silicon; repository virtual environment with Python 3.13; local
Jupyter kernel permission was approved for notebook-execution tests. The source
notebook is built deterministically: two consecutive builder runs produced the
same SHA-256, `fbaac48b4f9e643e552e0d640e4320790cf5fba042b9869eb28b4f252f9cd710`.

## Unit and integration tests

The required targeted package passed with approved local-kernel permission:

```bash
.venv/bin/pytest -q tests/test_research_planning.py tests/test_planning_mcp_executor.py tests/test_plan_execute_graph.py tests/test_plan_execute_policies.py tests/test_lesson11_assets.py tests/test_course_manifest.py
# 73 passed in 6.70s
```

The full repository regression passed after a failure-driven correction to the
Lesson 10 index test, whose assertion had retained superseded per-lesson wording
after Lesson 11 consolidated all four indexes to the Lesson 08-11 readiness
statement:

```bash
.venv/bin/pytest -q
# 314 passed in 52.04s

.venv/bin/ruff check .
# All checks passed!

.venv/bin/python scripts/validate_notebooks.py notebooks/11_plan_and_execute_analyst.ipynb
# 1 notebook passed the course notebook contract.

.venv/bin/python scripts/validate_repo.py
# FinAI Academy repository structure is valid.

git diff --check
# no output
```

The original whole-repository Ruff run correctly found four unused imports in
generated cell `lesson11-003`. The root cause was the builder emitting symbols
only named later as teaching text. The builder and regenerated notebook now
remove only those imports; whole-repository Ruff and the targeted package pass.

## Offline notebook

Fresh execution used a new explicit output directory:

```bash
.venv/bin/python scripts/execute_notebooks.py notebooks/11_plan_and_execute_analyst.ipynb --mode offline --output-dir /private/tmp/finai-lesson11-offline-task8-20260822
# PASS notebooks/11_plan_and_execute_analyst.ipynb -> /private/tmp/finai-lesson11-offline-task8-20260822/11_plan_and_execute_analyst.ipynb
```

Observed runtime evidence: `offline fixture · deterministic planner and
replanner · real local MCP execution`; server `First Finance Research`; permitted
tools `get_company_metric` and `search_financial_documents`; attempted step IDs
`1, 2, 3, 5, 6`; one retained `unsupported_metric` error; one revision; evidence
gate `True`; six PNG outputs; and exactly one `LESSON_11_PASS` marker.

## Ollama live route

**NOT AVAILABLE / NOT RUN.** `ollama` was present, but its local daemon did not
respond to `ollama list`; consequently the availability of `qwen3:8b` could not
be established. No model was downloaded or pulled, and the live command was not
attempted. No Ollama provider credit is claimed.

## OpenAI live route

**NOT CONFIGURED / NOT RUN.** `OPENAI_API_KEY` was absent. The key value was not
read or printed, the OpenAI command was not attempted, and no OpenAI provider
credit is claimed.

## Notebook visual review

All six PNG outputs from the fresh offline artifact were extracted and inspected
at full size: control-pattern comparison, initial-plan dependencies, ownership
graph, logarithmic execution timeline, tail-only replan, and evidence-coverage
matrix. Labels, connector direction, short-duration visibility, retained failure,
replacement IDs, and dark-cell contrast are readable. No clipping or collision
was observed.

## Deck automated and visual review

The bundled presentation runtime produced these current checks:

```bash
slides_test.py decks/11-plan-and-execute-analyst.pptx
# Test passed. No overflow detected.

validate_template_plan.mjs --workspace .artifacts/lesson11-deck --map .artifacts/lesson11-deck/template-frame-map.json --source-slide-count 9
# initial retained inventory failed: 12 unknown inherited shape IDs on source/output slide 6

task8_inspect_source.mjs with presentation.inspect({ kind: 'slide,textbox,shape,image,table,chart', maxChars: 200000 })
# fresh inventory: inspectTruncated=false; 9 slides; 273 records; 35 slide-6 records

validate_template_plan.mjs --workspace .artifacts/lesson11-deck --map .artifacts/lesson11-deck/template-frame-map.json --inspect .artifacts/lesson11-deck/task8-template-inspect/template-inspect.ndjson --source-slide-count 9
# status: pass; issueCount: 0

check_template_fidelity.mjs --workspace .artifacts/lesson11-deck --final-pptx decks/11-plan-and-execute-analyst.pptx --map .artifacts/lesson11-deck/template-frame-map.json --starter-pptx .artifacts/lesson11-deck/template-starter.pptx --starter-layout-dir .artifacts/lesson11-deck/template-starter-layout --final-layout-dir .artifacts/lesson11-deck/layout/final --edit-dir .artifacts/lesson11-deck
# status: pass; issueCount: 0
```

The package contains nine slide XML parts and zero empty structural placeholders.
Its `ppt/theme/theme1.xml` SHA-256 is
`8b500abccb3a86061340d95e2edfe2ca62da665f2741801d8790930dba1507a0`, identical
to the Lesson 10 source template. Task 7's final render review inspected all nine
slides at full size; this review found no deck-content change since then. The
retained inspector had used `max_chars` rather than the API's `maxChars`, which
made its inventory truncated and omitted shape IDs; the complete fresh inventory
above is the passing template-plan evidence.

## Instructor timing and fallback review

The chapter defines the complete 13:30-14:30 route: 12-minute deck, 40-minute
notebook, and 8-minute verification/debrief. Its five-minutes-late path preserves
MCP discovery, the retained failure, tail replacement, evidence gate, and pass
marker. Offline execution is the documented fallback for unavailable providers;
static matrices are explicitly labelled recovery material rather than observed
execution.

## Known qualifications

- Ollama live evidence is unavailable because the daemon was unavailable; `qwen3:8b` was not pulled or run.
- OpenAI live evidence is unavailable because `OPENAI_API_KEY` is not configured.
- No timed learner rehearsal was performed.

## Weighted score

| Dimension | Weight | Score | Basis |
| --- | ---: | ---: | --- |
| Learner usability | 25% | 9.2/10 | Deterministic 40-minute offline notebook, visible artifacts, complete instructor route; no timed learner rehearsal. |
| Technical correctness and safety | 20% | 9.7/10 | 314 passing tests, real local MCP lifecycle, typed failure/replan/evidence gates, repository validation. |
| Conceptual progression | 20% | 9.5/10 | Deck, chapter, and notebook move from proposal to host control, retained failure, evidence gate, and Lesson 12 handoff. |
| Live delivery | 15% | 4.0/10 | Offline delivery is observed; neither live provider was available/configured, so no live-provider pass is awarded. |
| Visuals | 10% | 9.7/10 | Six current notebook figures and the nine-slide deck pass full-size review, overflow, template-plan, fidelity, placeholder, and theme checks. |
| Repository quality | 10% | 9.7/10 | Full tests, Ruff, notebook validator, repository validator, whitespace, deterministic build, and refreshed deck QA pass. |

Weighted readiness score: `(9.2 × 0.25) + (9.7 × 0.20) + (9.5 × 0.20) + (4.0 × 0.15) + (9.7 × 0.10) + (9.7 × 0.10) =` **8.68/10**.

## Decision

**Conditionally ready for an offline instructor-led test class.** The maintained
offline route and full repository are verified. It is not fully certified for
live-provider delivery because neither optional provider route was available for
this review.
