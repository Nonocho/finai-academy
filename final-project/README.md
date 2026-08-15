# Financial Analyst Copilot

The course capstone is a conversational, evidence-backed assistant for
asset-management research. It is developed incrementally across the program, not
assembled in a single final notebook.

See:

- [Product specification](PRODUCT_SPEC.md)
- [Canonical program blueprint](../docs/program-blueprint.md)

## Current vertical slice

The first implementation establishes the Module 00 contract:

- a provider-neutral structured-model interface;
- a typed `AnalystBrief` response;
- explicit evidence categories;
- a versioned prompt for analysing user-supplied text; and
- a command-line entry point that can use Ollama or OpenAI through LangChain.

It intentionally does not include retrieval, tools, agents, or MCP yet. Those
capabilities will be added only after their simpler predecessors are testable.

## Run the first slice

Install the optional AI dependencies and ensure the selected provider is ready.

```bash
uv sync --extra ai
cp .env.example .env
```

Create a local text file containing an earnings-release or filing excerpt, then:

```bash
uv run python final-project/app.py \
  --company NVIDIA \
  --period FY2026 \
  --input path/to/excerpt.txt
```

The local default is Ollama. To use OpenAI, set:

```bash
FINAI_MODEL_PROVIDER=openai
FINAI_CHAT_MODEL=<an-available-openai-model>
OPENAI_API_KEY=<your-key>
```

The command prints a validated JSON analyst brief. Source text remains
user-supplied in this first slice; official acquisition and citations arrive in
Module 01.

## Development rule

Notebooks demonstrate and experiment with concepts. Stable domain and application
code belongs in `src/finai_academy/capstone`, and the integrated interface belongs
here.
