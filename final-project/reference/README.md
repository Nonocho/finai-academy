# Financial Analyst Copilot reference application

The reference application is the complete correction and classroom demonstration route. It compares NVIDIA and Schneider Electric using a bounded, evidence-backed workflow.

## Run the certified route

From the repository root:

```bash
uv sync --extra capstone --extra ai
uv run streamlit run final-project/reference/streamlit_app.py
```

Select **Recorded demo**, **Certified snapshots**, and **Reference mission**, then choose **Run reference mission**. This certified offline fallback requires no network, API key, Tavily, or Ollama service.

The route displays the plan, tool activity, typed recovery and replan, evidence gate, five briefing sections, citations, execution trace, deterministic release evaluation, and separate optional judge result. **Ask the analyst** accepts a bounded question in the same two-company research universe after the reference route is understood.

## Provider and data labels

- **Recorded demo** is the certified classroom route.
- **Ollama** is an explicit local option. Use `qwen3:4b` with `FINAI_MODEL_PROVIDER=ollama` when local Ollama is available.
- **OpenAI** is an explicit hosted option and requires `OPENAI_API_KEY` in `.env`.
- **Certified snapshots** are tracked classroom evidence.
- **Optional live enrichment** can use `TAVILY_API_KEY` when configured. It is not required for classroom success.

The application stores only JSON-compatible public presentation data in session state. It does not retain provider clients, credentials, private reasoning, exceptions, or personal paths.

This application provides research support, not investment advice.

First Finance - Arnaud Demes
