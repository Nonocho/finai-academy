# Task 8 report: end-to-end capstone certification

## Status

Implemented and committed Task 8 at `dad9282` (`test: certify Financial Analyst Copilot capstone`). The deterministic offline certification gate passes. A real browser screenshot was captured after executing the recorded reference mission at 1440×1000 and is committed as `artifacts/capstone/reference-mission.png`.

Certification exposed two real desktop presentation defects in the Task 5 Streamlit UI: oversized status metrics clipped provider/data/evidence labels, and wide citation/trace dataframes clipped factual claim/source pairs. Focused AppTest regressions were observed failing before the UI was changed to compact wrapping status text, wrapped citation/evidence rows, and wrapped trace-event cards.

## Scoped files

- Added `scripts/certify_capstone.py`.
- Added `tests/test_capstone_certification.py`.
- Added sanitized `artifacts/capstone/certification.json`, `readiness.md`, `visual-inspection.json`, and the real 1440×1000 `reference-mission.png`.
- Modified `src/finai_academy/capstone/streamlit_ui.py` and `tests/test_capstone_streamlit.py` only for certification-discovered clipping/readability defects.
- Kept `artifacts/capstone/mlflow/` runtime database and caches ignored.

## Mandatory certification evidence

- Recorded provider and certified snapshots completed the fixed NVIDIA/Schneider Electric mission.
- One bounded replan and five observations were validated within the 1-replan/6-step budgets.
- Both companies have source-addressable evidence; every fact/source pair was validated against collected observations, and document facts were matched by company, evidence ID, and source.
- All five deterministic metrics are present in fixed order at `1.0`; citation integrity is `1.0`; release passed.
- The reference AppTest journey renders route labels, plan/tool/replan state, briefing, readable citation pairs, trace, deterministic release, separate optional judge, and exact footer.
- The committed student starter launches and reports exactly four intended incomplete groups with no marker.
- A temporary solved copy exits zero and prints exactly one standalone `CAPSTONE_PASS`.
- MLflow persisted a run, linked trace, all five metrics, release decision, and the three sanitized public evidence artifacts.
- OpenAI, Ollama, Tavily, and timed classroom rehearsal are truthfully recorded as `NOT RUN` and do not affect offline release.

## Visual evidence

The in-app browser executed the real local Streamlit mission at an explicit 1440×1000 viewport. The final screenshot is a valid 1440×1000 PNG. Scrolled inspection observed:

- readable hierarchy and corrected unclipped status labels;
- recorded provider and certified data labels;
- readable plan, tool error, replan information, and success state;
- wrapped briefing and claim/source/evidence-ID pairs;
- expanded, readable trace-event cards;
- distinct warning, replan/info, success, unavailable, and typed evidence-stop treatments;
- deterministic release separated from the optional judge;
- exact `First Finance - Arnaud Demes` footer.

A temporary real Streamlit harness exercised the evidence-stop state in the same renderer; it was not committed and did not alter certification output.

## Test and gate evidence

- TDD RED: missing `scripts/certify_capstone.py` produced the expected certification-test error.
- TDD RED: the new wrapping-status AppTest failed on the existing eight `st.metric` elements before the UI fix.
- TDD RED: the new citation/trace wrapping AppTest failed on the existing wide dataframes before the UI fix.
- `.venv/bin/python scripts/certify_capstone.py`: `CAPSTONE_CERTIFICATION_PASS`.
- `.venv/bin/python -m pytest tests/test_capstone_certification.py -q`: `5 passed in 85.71s`.
- `.venv/bin/python -m pytest tests/test_capstone_streamlit.py -q`: `9 passed in 1.89s`.
- `.venv/bin/python -m pytest tests/test_capstone_certification.py::test_committed_certification_artifacts_match_the_contract -q`: `1 passed in 0.03s` after the final PNG assertion was added.
- `.venv/bin/ruff check src tests scripts final-project`: passed.
- `.venv/bin/python scripts/validate_repo.py`: `FinAI Academy repository structure is valid.`
- Public certification text and PNG metadata/string scans: passed.
- `git diff --check`: passed before commit.

## Full baseline concerns

The permission-correct full suite did not pass fully: `.venv/bin/python -m pytest -q` finished with `564 passed, 11 failed in 397.61s`.

- Ten unchanged notebook visual checks executed their notebooks successfully but observed zero `image/png` outputs. The failures cover Lessons 03–12 visual-output count assertions and match the known host notebook-PNG limitation from the brief.
- One unchanged MLflow safety test found the host path in auto-generated MLflow trace metadata: `mlflow.source.name` points to the environment's `pytest/__main__.py` under `/Users/...`.
- `git diff dcd2446` is empty for the notebook sources/executor and for `src/finai_academy/mlflow_agent_evaluation.py` plus its failing test. No full-green claim is made.
- An initial sandboxed full run had 15 additional Jupyter socket-bind denials; rerunning with local-kernel permission removed those environment-only launch failures and yielded the authoritative 11-failure result above.

No live provider or timed rehearsal was run or claimed.

## Review round 1 remediation

The MLflow privacy finding was reproduced against the capstone SQLite store before remediation: the default and capstone experiment locations, run artifact URIs, MLflow-generated trace user, and trace artifact-location tag contained host identity or absolute paths. The persistence boundary now uses a neutral `local-capstone-user`, rewrites all experiment, run, run-tag, trace-request, and trace-tag host metadata to relative/path-free values after MLflow flushes, and supports safely reopening the same store for later runs. Lesson 12 persistence was not changed.

Certification now enumerates every SQLite table and every persisted string value, including experiments, runs, tags, trace info, trace request metadata, trace tags, spans, and MLflow schema tables. It also scans relative artifact names and all UTF-8 artifact contents. The fail-closed patterns cover the local username, POSIX and Windows user paths, `file:///Users`, authorization headers, API-key assignments and token shapes, client secrets, and private keys. Repository privacy booleans and public-artifact validation are derived from executed scans rather than constants. Regression injections independently failed in the experiment, run, tag, trace-info, trace-request, trace-tag, span, and artifact domains before the sanitization/audit implementation was completed.

The result-view regression is now exact rather than sentinel-based: it checks six citation cards and every claim/company/source/evidence-ID tuple, plus all ten trace events in order with phase/status, capability, error, and revision. The visual manifest contract was upgraded to schema 2: a PASS requires at least four hash-bound 1440×1000 PNG captures, full Pillow verify-and-load decoding, per-image browser/route/state/timestamp provenance, safe PNG metadata, and union coverage of every required acceptance element. Tests exercise a copied-image PASS fixture and reject a changed hash, a re-hashed but truncated PNG, and missing acceptance coverage.

The requested four-state browser recapture could not be completed in this review turn. The local Streamlit server started successfully, but browser discovery returned no controllable browser instances. In accordance with the brief, the earlier single top viewport was not relabeled or duplicated as below-fold proof. `visual-inspection.json` and the generated certification/readiness artifacts now truthfully record visual evidence as `NOT RUN`; the unbound earlier PNG remains only as historical evidence. This is the only unmet review item and is explicitly not presented as a visual PASS.

Review-round verification observed:

- `.venv/bin/pytest -q tests/test_capstone_persistence.py`: `14 passed`.
- `.venv/bin/pytest -q tests/test_capstone_streamlit.py tests/test_capstone_persistence.py`: `25 passed`.
- `.venv/bin/python scripts/certify_capstone.py`: `CAPSTONE_CERTIFICATION_PASS` with visual status `NOT RUN`.
- Direct SQLite counts after certification: zero experiment locations, run artifact URIs, trace users, or trace artifact tags containing the tested personal host values.
- Recursive capstone MLflow/certification safe scan: no matching personal paths, username, credential headers, client/private-key labels, or secret-shaped tokens.
- `.venv/bin/pytest -q tests/test_capstone_*.py`: `166 passed in 266.17s`.
- `.venv/bin/ruff check src tests scripts final-project`: passed.
- `.venv/bin/python scripts/validate_repo.py`: repository structure valid.
- `git diff --check`: passed.

The original full-suite baseline diagnosis remains unchanged and separate: ten notebook PNG-output failures and one Lesson 12 auto-generated MLflow source host-path failure. No full-green claim is made.
