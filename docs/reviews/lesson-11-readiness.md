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
Jupyter kernel permission was approved for notebook-execution tests. After the
claim-level provenance fix, two consecutive builder runs produced the same source
notebook SHA-256,
`6c71234d394b6508c5247af774bcb1317a72b7581e7740d306bd6042b40b481d`.

## Post-certification provenance and display review

The final whole-lesson review added a typed `CitedFact` boundary. Every reported
fact now carries a required `provenance_kind` (`metric` or `document`), non-empty
source references, and the evidence-ID shape required by that kind. The graph
validates capability-specific provenance before it can return `completed`:
metric facts cite exactly one source from a successful `get_company_metric`
observation and no evidence ID; document facts cite exactly one source and one
evidence ID whose exact pair occurs in a returned hit from a successful
`search_financial_documents` observation. A source-only document claim and a
metric claim citing document-only provenance are both rejected. Aggregate
briefing sources must exactly match the stable first-seen union of cited-fact
sources. Metric evidence cannot satisfy the evidence gate without source
references, and document evidence counts only when at least one returned hit
exactly matches the observation's declared source/evidence-ID pair.

Notebook cell `lesson11-022` now prints every complete factual claim with its
provenance kind, source references, and evidence IDs, followed by cross-company
observations, interpretation, limitations, and aggregate sources. The 40-minute notebook and
12 + 40 + 8 lesson timebox are unchanged.

## Unit and integration tests

The required targeted package passed with approved local-kernel permission:

```bash
.venv/bin/pytest -q tests/test_research_planning.py tests/test_planning_mcp_executor.py tests/test_plan_execute_graph.py tests/test_plan_execute_policies.py tests/test_lesson11_assets.py tests/test_course_manifest.py
# 89 passed in 7.00s
```

The full repository regression passed after a failure-driven correction to the
Lesson 10 index test, whose assertion had retained superseded per-lesson wording
after Lesson 11 consolidated all four indexes to the Lesson 08-11 readiness
statement:

```bash
.venv/bin/pytest -q
# 330 passed in 51.80s

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

Fresh post-fix execution used a new explicit output directory:

```bash
.venv/bin/python scripts/execute_notebooks.py notebooks/11_plan_and_execute_analyst.ipynb --mode offline --output-dir /private/tmp/finai-lesson11-pairing-0Ep6Il
# PASS notebooks/11_plan_and_execute_analyst.ipynb -> /private/tmp/finai-lesson11-pairing-0Ep6Il/11_plan_and_execute_analyst.ipynb
```

Observed runtime evidence: `offline fixture · deterministic planner and
replanner · real local MCP execution`; server `First Finance Research`; permitted
tools `get_company_metric` and `search_financial_documents`; attempted step IDs
`1, 2, 3, 5, 6`; one retained `unsupported_metric` error; one revision; evidence
gate `True`; six PNG outputs; and exactly one `LESSON_11_PASS` marker. The cited
briefing visibly contains two sourced metric claims and four document claims with
their exact evidence IDs, one cross-company observation, two interpretation
statements, three limitations, and the three aggregate source references.

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

All six PNG outputs from the post-fix offline artifact were extracted and inspected
at full size: control-pattern comparison, initial-plan dependencies, ownership
graph, logarithmic execution timeline, tail-only replan, and evidence-coverage
matrix. Labels, connector direction, short-duration visibility, retained failure,
replacement IDs, and dark-cell contrast are readable. No clipping or collision
was observed.

## Deck automated and visual review

The final provenance round did not modify the deck. A fresh structural regression
of the two Lesson 11 deck contracts passed. A fresh invocation of the bundled
overflow helper could not start because the workspace dependency loader was not
available in this agent context and the project environment does not include
`pdf2image`; no new overflow result is claimed. The prior certification evidence
below remains applicable to the byte-unchanged deck.

The original certification produced these retained checks:

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
- The fresh deck overflow helper could not start without the unavailable bundled
  dependency loader; the deck is byte-unchanged and its structural regression passed.

## Weighted score

| Dimension | Weight | Score | Basis |
| --- | ---: | ---: | --- |
| Learner usability | 25% | 9.2/10 | Deterministic 40-minute offline notebook, visible artifacts, complete instructor route; no timed learner rehearsal. |
| Technical correctness and safety | 20% | 9.7/10 | 330 passing tests, real local MCP lifecycle, capability-specific fact provenance, exact returned-hit citation pairing, typed failure/replan/evidence/provenance gates, repository validation. |
| Conceptual progression | 20% | 9.5/10 | Deck, chapter, and notebook move from proposal to host control, retained failure, evidence gate, and Lesson 12 handoff. |
| Live delivery | 15% | 4.0/10 | Offline delivery is observed; neither live provider was available/configured, so no live-provider pass is awarded. |
| Visuals | 10% | 9.7/10 | Six current notebook figures and the nine-slide deck pass full-size review, overflow, template-plan, fidelity, placeholder, and theme checks. |
| Repository quality | 10% | 9.7/10 | Full tests, Ruff, notebook validator, repository validator, whitespace, deterministic build, and unchanged certified deck artifacts. |

Weighted readiness score: `(9.2 × 0.25) + (9.7 × 0.20) + (9.5 × 0.20) + (4.0 × 0.15) + (9.7 × 0.10) + (9.7 × 0.10) =` **8.68/10**.

## Decision

**Conditionally ready for an offline instructor-led test class.** The maintained
offline route and full repository are verified. It is not fully certified for
live-provider delivery because neither optional provider route was available for
this review.
