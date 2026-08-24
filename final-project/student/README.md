# Financial Analyst Copilot student challenge

Complete four integration seams in a certified, offline-first analyst workflow. The
retrieval, tool policy, recorded mission, release evaluation, persistence, and public
view components already exist. Your task is to connect them, not rebuild them.

The starter is intentionally launchable. It reports four conceptual incomplete
statuses without import errors, API credentials, network access, Tavily, or Ollama.

## Start here

From the repository root, launch the student workspace:

```bash
.venv/bin/streamlit run final-project/student/streamlit_app.py
```

In a second terminal, run the verifier:

```bash
.venv/bin/python final-project/student/verify.py
```

The starter exits nonzero with one named diagnostic for each unfinished seam. A
complete integration prints diagnostics followed by exactly one standalone
`CAPSTONE_PASS` line. Do not print that marker from your own function bodies.

## Your four seams

Edit only these function bodies in `integration.py`:

1. `wire_retriever` — route a company-scoped query through the certified document
   retriever and keep its typed evidence hits.
2. `register_analyst_capabilities` — intersect runtime discovery with the approved
   analyst read-tool policy.
3. `evaluate_student_evidence_gate` — release only when document evidence covers
   NVIDIA and Schneider Electric, preserving the collected hits.
4. `assemble_public_briefing_view` — cross the existing display-safe public view
   boundary instead of exposing domain internals.

Each diagnostic is independent. Re-run the verifier after completing a seam so that
the remaining list stays short and attributable.

## Fixed mission and acceptance boundary

The verifier runs the recorded NVIDIA and Schneider Electric reference mission. It
then checks citation integrity, the five deterministic release metrics, and local
temporary persistence. The recorded route uses only versioned repository fixtures.
It does not silently select OpenAI, Ollama, live news, or another network service.

Keep all outputs path-free and credential-free. Do not add shell commands, API keys,
personal paths, trading capabilities, or learner-controlled process execution to the
integration module or verifier.

## Working alone or in a pair

If working alone, implement the seams in verifier order. In a pair, one person can
take retrieval plus capability policy while the other takes the evidence gate plus
public view; swap for a quick review before the final verifier run.

Use [CHECKLIST.md](CHECKLIST.md) to keep the exercise within 60 minutes.

This application supports financial research. It does not provide investment advice.
