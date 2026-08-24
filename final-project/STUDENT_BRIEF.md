# Financial Analyst Copilot student brief

## Mission

> Compare NVIDIA and Schneider Electric using official documents and selected financial metrics. Identify the main operating-growth evidence, explain why direct comparison is limited, and cite every factual claim.

Complete this 60-minute challenge individually or in a pair, followed by a 30-minute demonstration and architecture review. In a pair, both people remain responsible for the full solution. A driver switch after two seams is optional guidance, not an alternative assessment.

## Deliverables

1. Complete the four function bodies in `final-project/student/integration.py`.
2. Show the student interface with all four seams ready.
3. Run the public verifier and show its single standalone `CAPSTONE_PASS` line.
4. Prepare a short explanation of one seam, the evidence gate, and the public view boundary.

## The four seams

- `wire_retriever`: connect a company-scoped request to certified document evidence.
- `register_analyst_capabilities`: retain only discovered, approved read capabilities.
- `evaluate_student_evidence_gate`: require document evidence for NVIDIA and Schneider Electric before release.
- `assemble_public_briefing_view`: transform a completed result through the safe public presentation boundary.

Work only inside the four function bodies. The starter, retrieval, policy, recorded mission, evaluation, persistence, and view components already exist. Do not rebuild them.

## Start and verify

From the repository root:

```bash
uv sync --extra capstone --extra ai
```

Use separate terminals so each Streamlit server stays running while the next route is started:

```bash
# Terminal 1: reference application
uv run streamlit run final-project/reference/streamlit_app.py
```

```bash
# Terminal 2: student application
uv run streamlit run final-project/student/streamlit_app.py
```

```bash
# Terminal 3: public verifier
uv run python final-project/student/verify.py
```

The Streamlit servers in Terminal 1 and Terminal 2 stay running. Start the next command in its assigned terminal, or stop the active server with `Ctrl+C` before reusing a terminal. Use the reference application only to understand the target route. In the student application, each incomplete seam has a named diagnostic. Re-run the verifier after each seam. A public pass requires the four seam contracts, the recorded reference mission, citation integrity, five deterministic release metrics, and local temporary persistence. `CAPSTONE_PASS` comes from `verify.py`; do not print it from a solution function.

## Safety constraints

The recorded route is the certified offline fallback. It needs no API key, network, Tavily, or Ollama. Keep the two-company boundary. Do not add shell commands, credentials, personal paths, provider fallback, trading actions, or process execution to the student module. Keep output path-free and credential-free.

This application provides research support, not investment advice.

## Timetable

| Time | Activity |
| --- | --- |
| 15:30–15:40 | Understand mission |
| 15:40–16:10 | Complete four seams |
| 16:10–16:25 | Evaluate and diagnose |
| 16:25–16:30 | Prepare demo |
| 16:30–17:00 | Demonstration and architecture review |

First Finance - Arnaud Demes
