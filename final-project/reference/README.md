# Financial Analyst Copilot reference application

This Streamlit application presents the complete, bounded capstone journey for
NVIDIA and Schneider Electric. The default **Recorded demo** route uses tracked
certified snapshots and needs no network connection, API key, OpenAI account, or
running Ollama service.

## Run locally

From the repository root:

```bash
uv sync --extra capstone
uv run streamlit run final-project/reference/streamlit_app.py
```

Keep **Recorded demo** and **Certified snapshots** selected, then choose
**Run reference mission**. The application displays the validated plan, tool
activity, typed recovery, evidence gate, five briefing sections, citations,
public execution trace, exactly five deterministic release metrics, and the
separate optional judge result.

The **Ask the analyst** tab accepts questions of up to 240 characters. Every
request remains limited to the NVIDIA and Schneider Electric research universe.

## Provider and data labels

- **Recorded demo** is the certified offline route.
- **Ollama** and **OpenAI** are explicit optional provider routes. An unavailable
  selection is reported as unavailable; the application does not silently change
  providers.
- **Certified snapshots** use the tracked classroom evidence.
- **Optional live enrichment** is visibly separate and reports Tavily readiness.

The interface stores only JSON-compatible public selections, chat messages, and
presentation views in session state. It does not store provider clients, model
objects, prompts, credentials, exceptions, private reasoning, or personal paths.

This application supports financial research. It does not provide investment
advice or recommendations.

First Finance - Arnaud Demes
