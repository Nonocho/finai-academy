# Task 6 report

- **RED:** Added chapter and index contract tests. The focused run failed as expected because the Lesson 11 chapter and discoverable links did not yet exist.
- **GREEN:** Added the instructor chapter and indexes. The focused chapter/index contracts passed, then the full Lesson 11 asset suite passed with a permitted local Jupyter kernel.
- **Self-review:** Confirmed the exact 13:30-14:30 split, stable notebook-cell mapping, expected failure and tail-only replan route, offline/Ollama/OpenAI commands, recovery and no-network fallback, skip-if-late route, knowledge-check answer key, read-only and no-advice boundary, and Lesson 12 fields. Confirmed the deck is named only as planned and has no active link.
- **Commit:** `docs: add lesson 11 instructor route`
- **Concerns:** Repository-wide Ruff reports four pre-existing unused imports in `notebooks/11_plan_and_execute_analyst.ipynb`, stable cell `lesson11-003`: `RECORDED_REPLACEMENT_STEPS`, `AnalystBriefing`, `ResearchObservation`, and `ResearchPlan` (reported as notebook cell 4). This is a deferred whole-lesson issue; Task 6 does not edit the Task 5 notebook or builder. The changed Task 6 test file passes Ruff. The planned Lesson 11 deck is intentionally absent and unlinked until Task 7.
