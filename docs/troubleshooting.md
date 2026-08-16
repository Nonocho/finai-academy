# Troubleshooting

Run commands from the repository root. Start with:

```bash
uv run python scripts/setup_check.py --offline
```

| Symptom | Likely cause | Action |
|---|---|---|
| `uv: command not found` | The shell has not reloaded its PATH | Restart Terminal or PowerShell, then run `uv --version` |
| `pyproject.toml` not found | Wrong working directory | Run `cd finai-academy` |
| Dependency import fails | Environment is incomplete | Run `uv sync --extra ai --extra rag --extra evaluation --extra dev` |
| Notebook uses the wrong kernel | Jupyter was started outside `uv` | Stop it and run `uv run jupyter lab` |
| Ollama connection refused | Ollama is not running | Start the Ollama application, then rerun `uv run python scripts/setup_check.py --provider ollama` |
| Chat model is missing | Default chat model was not pulled | Run `ollama pull qwen3:8b` |
| Embedding model is missing | Retrieval model was not pulled | Run `ollama pull qwen3-embedding:0.6b` |
| Ollama is very slow or closes | The model exceeds available memory | Set `FINAI_CHAT_MODEL=qwen3:4b` in `.env`, then run `ollama pull qwen3:4b` |
| OpenAI reports a missing key | `.env` is absent or incomplete | Copy `.env.example` to `.env`, add `OPENAI_API_KEY`, then rerun the OpenAI check |
| OpenAI authentication fails | Key or account access is invalid | Create or verify the key in the [OpenAI dashboard](https://platform.openai.com/api-keys) |
| A notebook PASS gate fails | An earlier cell, provider, or artifact is inconsistent | Restart the kernel, run all cells in order, and read the first failing assertion |
| Docker is unavailable | Optional tooling is not installed | Continue Day 1 or install [Docker Desktop](https://docs.docker.com/get-started/get-docker/) before container lessons |

## Safe reset inside a notebook

1. Restart the kernel.
2. Confirm the active `.env` provider.
3. Run all cells from the top.
4. Do not edit expected answers or weaken assertions to force a PASS.

## Secrets

Never paste a key into an issue, notebook output, screenshot, or commit. If a key is
exposed, revoke it at the provider immediately and create a new one.

Return to [Getting started](getting-started.md) or the
[Day 1 student guide](day-1-student-guide.md).
