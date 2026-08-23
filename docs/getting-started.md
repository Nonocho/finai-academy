# Getting started

This is the supported path from a fresh machine to Lesson 01. Use Ollama for the
free local course path. OpenAI and Docker are optional.

## 1. Install Git and uv

### macOS

Check Git first:

```bash
git --version
```

If Git is unavailable, install the Apple command-line tools when macOS prompts you,
or use the [official Git installer](https://git-scm.com/download/mac).

Install `uv` with the [official Astral installer](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart Terminal, then confirm:

```bash
uv --version
```

### Windows PowerShell

Install [Git for Windows](https://git-scm.com/download/win), open a new PowerShell
window, and confirm:

```powershell
git --version
```

Install `uv` with the official Astral installer:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart PowerShell, then confirm:

```powershell
uv --version
```

## 2. Clone and install the course

The same commands work in Terminal and PowerShell:

```bash
git clone https://github.com/Nonocho/finai-academy.git
cd finai-academy
uv python install 3.11
uv sync --extra ai --extra rag --extra evaluation --extra dev
```

Expected result: `uv` creates `.venv` and completes without a dependency error.

## 3. Create the environment file

On macOS:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

The committed defaults select Ollama. `.env` is ignored by Git. Never paste API
keys into a notebook, source file, screenshot, or commit.

## 4. Verify the offline environment

```bash
uv run python scripts/setup_check.py --offline
```

Expected final row:

```text
READY Course readiness — Environment is ready for Day 1.
```

Ollama, OpenAI, and Docker are allowed to show as `OPTIONAL` in this check.

## 5. Install Ollama and the course models

Install and start [Ollama](https://ollama.com/download). On macOS, open the Ollama
application once. On Windows, complete the installer and confirm Ollama is running
from the taskbar.

Pull the tested course pair:

```bash
ollama pull qwen3:8b
ollama pull qwen3-embedding:0.6b
```

The chat model generates text and structured outputs. The embedding model encodes
text for semantic chunking and retrieval; it does not generate answers.

| Profile | Indicative memory | Chat | Embeddings | Support |
|---|---:|---|---|---|
| Light | 8 GB | `qwen3:4b` | `qwen3-embedding:0.6b` | Best effort |
| Course default | 16 GB | `qwen3:8b` | `qwen3-embedding:0.6b` | Fully tested |
| Advanced | 32 GB+ | `qwen3:14b` | `qwen3-embedding:4b` | Optional |

Use the course default whenever the machine supports it. Gemma, Llama, and BGE-M3
are useful comparison families, but they are outside the supported setup path.

Verify the local live path:

```bash
uv run python scripts/setup_check.py --provider ollama
```

Expected: `PASS` for the Ollama service, chat model, and embedding model, followed
by `READY Course readiness`.

## 6. Start the course

```bash
uv run jupyter lab
```

Open `notebooks/01_model_gateway.ipynb` and run the cells from top to bottom.

## Optional: OpenAI

Create a key from the [OpenAI API dashboard](https://platform.openai.com/api-keys).
API access is separate from a ChatGPT subscription and may require account credit.

Edit `.env`:

```dotenv
FINAI_MODEL_PROVIDER=openai
FINAI_CHAT_MODEL=gpt-5-mini
FINAI_EMBEDDING_PROVIDER=openai
FINAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=your_key_here
```

Then run:

```bash
uv run python scripts/setup_check.py --provider openai
```

Do not commit `.env`.

## Lesson 12: local MLflow agent evaluation

Lesson 12 uses local SQLite and local artifacts for its deterministic core. Docker is not required for Lesson 12, and the browser UI is not required for Lesson 12. The notebook
renders the essential run comparison, per-case scorecard, trace, and failure evidence
inline.

On macOS or Linux, select and print a resolved local directory before executing the
notebook:

```bash
export FINAI_MLFLOW_DIR=/private/tmp/finai-lesson12-mlflow
uv run python -c "import os; from pathlib import Path; print(Path(os.environ['FINAI_MLFLOW_DIR']).resolve())"
uv run python scripts/execute_notebooks.py notebooks/12_evaluating_agentic_systems.ipynb \
  --mode offline --output-dir /private/tmp/finai-lesson12-offline
```

The notebook prints the resolved database path, local artifacts directory, and exact UI
command. Starting that UI is optional; when started, it is available at
`http://127.0.0.1:5000`.

The optional judge routes use explicit judge URIs. These examples document the accepted
forms without enabling either route:

```text
FINAI_EVAL_JUDGE_MODEL=openai:/<model>
FINAI_EVAL_JUDGE_MODEL=ollama_chat:/<model>
```

Select at most one explicit provider/model for an optional extension. If it is not
actually executed, report the judge as `NOT RUN`; installed clients, credentials, or
configuration text are not execution evidence.

## Optional: Docker

Docker is not needed for Day 1. Install [Docker Desktop](https://docs.docker.com/get-started/get-docker/)
before the service and deployment lessons if you want the containerized path.

```bash
docker --version
```

For a failed check, use the [troubleshooting guide](troubleshooting.md).
