# Financial Analyst Copilot student challenge

Complete four bounded integration seams in a certified offline-first analyst workflow. The starter launches before the seams are complete and reports named, conceptual diagnostics without requiring API credentials, network access, Tavily, or Ollama.

## Start and verify

From the repository root:

```bash
uv sync --extra capstone --extra ai
```

Use Terminal 2 for the student application:

```bash
# Terminal 2: student application
uv run streamlit run final-project/student/streamlit_app.py
```

The Streamlit server in Terminal 2 stays running. Run the verifier in Terminal 3, or stop the server with `Ctrl+C` before reusing Terminal 2:

```bash
# Terminal 3: public verifier
uv run python final-project/student/verify.py
```

The starter exits nonzero with one diagnostic per unfinished seam plus the deliberate evidence-routing regression. A complete integration and corrected diagnostic print diagnostics followed by one standalone `CAPSTONE_PASS` line. Do not print that marker from your function bodies.

## Scope

For the implementation phase, edit only the four function bodies in `integration.py`:

1. `wire_retriever`
2. `register_analyst_capabilities`
3. `evaluate_student_evidence_gate`
4. `assemble_public_briefing_view`

After those seams pass, the diagnostic phase asks you to edit only `diagnostic_case.json`. Run:

```bash
uv run python final-project/student/diagnose.py run
uv run python final-project/student/diagnose.py inspect
```

Inspect the persisted trace owner, set the deliberate `drop_company` regression to JSON `null`, rerun the diagnostic, then rerun `uv run python final-project/student/verify.py`.

The recorded mission uses repository fixtures only. Keep output path-free and credential-free. Do not add shell commands, provider fallback, trading actions, or learner-controlled process execution.

Read [../STUDENT_BRIEF.md](../STUDENT_BRIEF.md) for the mission, timetable, public pass rule, pair guidance, and demonstration requirements. Use [CHECKLIST.md](CHECKLIST.md) to keep the challenge within 60 minutes.

This application provides research support, not investment advice.

First Finance - Arnaud Demes
