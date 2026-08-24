# Financial Analyst Copilot capstone

## Learning outcome

Complete and explain a bounded, evidence-backed financial research workflow that compares NVIDIA and Schneider Electric without exposing credentials, private paths, or investment recommendations.

## Shortest recorded-mode start

From the repository root:

```bash
uv sync --extra capstone --extra ai
```

Use separate terminals for the classroom workflow:

```bash
# Terminal 1: reference application
uv run streamlit run final-project/reference/streamlit_app.py
```

Keep **Recorded demo** and **Certified snapshots** selected, open **Reference mission**, then select **Run reference mission**. This certified offline fallback uses tracked classroom fixtures and requires no API key, Ollama service, or network connection.

The result shows the validated plan, tool activity, typed recovery and replan, evidence gate, five briefing sections, citations, execution trace, and deterministic release evaluation. The optional judge is separate from the release decision.

## Folders and routes

- [`reference/`](reference/README.md) is the complete application and correction route for demonstration and review.
- [`student/`](student/README.md) is the launchable challenge scaffold. Learners edit only the four function bodies in `student/integration.py`.
- [STUDENT_BRIEF.md](STUDENT_BRIEF.md) is the learner handout.
- [INSTRUCTOR_GUIDE.md](INSTRUCTOR_GUIDE.md) is the facilitation, correction, and recovery runbook.

The reference interface has two modes: **Reference mission** for the fixed, certified comparison and **Ask the analyst** for a bounded question in the NVIDIA and Schneider Electric research universe. **Certified snapshots** are versioned course data. **Optional live enrichment** is visibly separate and may use Tavily when configured; it never determines classroom success.

The Streamlit server in Terminal 1 stays running. Start the student route in a new terminal, or stop the server with `Ctrl+C` before reusing that terminal:

```bash
# Terminal 2: student application
uv run streamlit run final-project/student/streamlit_app.py
```

The Streamlit server in Terminal 2 stays running. Run the verifier in another terminal, or stop the server with `Ctrl+C` before reusing that terminal:

```bash
# Terminal 3: public verifier
uv run python final-project/student/verify.py
```

The verifier prints one standalone `CAPSTONE_PASS` line only after all four seams, the recorded mission, citations, deterministic metrics, and local persistence pass.

## Provider choices and setup

Recorded demo is the classroom default. Ollama and OpenAI are explicit alternatives; an unavailable selected provider is reported as unavailable and is not replaced silently.

### macOS

Install [uv](https://docs.astral.sh/uv/) if needed, then create the local environment file and install the capstone dependencies:

```bash
cp .env.example .env
uv sync --extra capstone --extra ai
```

For Ollama, install and start Ollama, pull the course model, and set the provider in `.env`:

```bash
ollama pull qwen3:4b
FINAI_MODEL_PROVIDER=ollama
FINAI_CHAT_MODEL=qwen3:4b
```

### Windows PowerShell

Install [uv](https://docs.astral.sh/uv/) if needed, then run:

```powershell
Copy-Item .env.example .env
uv sync --extra capstone --extra ai
```

For an Ollama route, install Ollama, run `ollama pull qwen3:4b`, then set `FINAI_MODEL_PROVIDER=ollama` and `FINAI_CHAT_MODEL=qwen3:4b` in `.env`.

### OpenAI and optional Tavily

For OpenAI, set `FINAI_MODEL_PROVIDER=openai` and add your own `OPENAI_API_KEY` to `.env`. Do not commit `.env` or paste keys into course files. Optional live enrichment checks `TAVILY_API_KEY`; leaving it unset keeps the certified route fully available. Never use a live provider or Tavily as a substitute for the recorded classroom route.

## Output boundary

The application is research support, not investment advice. It does not place trades, recommend securities, or make suitability decisions. Keep learner output within the fixed company boundary, evidence-backed, cited, path-free, and credential-free.

First Finance - Arnaud Demes
