# Financial Analyst Copilot reference application

The reference application is the complete correction and classroom demonstration route. It compares NVIDIA and Schneider Electric using a bounded, evidence-backed workflow.

## Run the certified route

From the repository root:

```bash
uv sync --extra capstone --extra ai
```

Use the classroom terminal allocation:

```bash
# Terminal 1: reference application
uv run streamlit run final-project/reference/streamlit_app.py
```

Select **Recorded demo**, **Certified snapshots**, and **Reference mission**, then choose **Run reference mission**. This certified offline fallback requires no network, API key, Tavily, or Ollama service.

The Streamlit server in Terminal 1 stays running. To begin the challenge without stopping the demonstration, use Terminal 2 for `uv run streamlit run final-project/student/streamlit_app.py` and Terminal 3 for `uv run python final-project/student/verify.py`. Otherwise, stop the server with `Ctrl+C` before reusing Terminal 1.

The route displays the plan, tool activity, typed recovery and replan, evidence gate, five briefing sections, citations, execution trace, deterministic release evaluation, and separate optional judge result. **Ask the analyst** accepts bounded operating-growth, valuation, and revenue-growth questions in the same two-company universe. Other questions return a typed planner stop before tools run.

## Provider and data labels

- **Recorded demo** is the certified classroom route.
- **Ollama** is an explicit local option. Use `qwen3:4b` with `FINAI_MODEL_PROVIDER=ollama` when local Ollama is available.
- **OpenAI** is an explicit hosted option and requires `OPENAI_API_KEY` in `.env`.
- **Certified snapshots** are tracked classroom evidence.
- The shipped app has no live-news route. The standalone Tavily adapter used in tool lessons is not composed into this capstone UI.

The application stores only JSON-compatible public presentation data in session state. It does not retain provider clients, credentials, private reasoning, exceptions, or personal paths.

This application provides research support, not investment advice.

First Finance - Arnaud Demes
