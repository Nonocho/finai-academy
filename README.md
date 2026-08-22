# AI Engineering for Asset Management

**First Finance - Arnaud Demes**

A two-day, notebook-first technical course. Build a Financial Analyst Copilot that
turns financial documents and market information into cited, evaluated analysis.

## Day 1 outcome

By 17:00, you will have built and measured a financial RAG pipeline:

```text
financial documents → parsing → chunks → hybrid retrieval → cited answer → evaluation trace
```

The course uses NVIDIA and Schneider Electric evidence so that each engineering
choice is tested on a realistic analyst workflow.

## Prerequisites

- macOS or Windows with at least 8 GB RAM; 16 GB is recommended for the tested model;
- Git and an internet connection for the initial installation;
- no paid API is required;
- Python experience is useful, but the notebooks explain each course-specific step.

## Quick start

Run these commands from the repository root after completing the
[installation guide](docs/getting-started.md):

```bash
uv sync --extra ai --extra rag --extra evaluation --extra dev
uv run python scripts/setup_check.py --offline
uv run python scripts/setup_check.py --provider ollama
uv run jupyter lab
```

Then open `notebooks/01_model_gateway.ipynb`.

## Execution modes

| Mode | Use | Requirement |
|---|---|---|
| Offline | Deterministic checks and course verification | No external service |
| Ollama | Official free live path | Local Ollama models |
| OpenAI | Optional hosted comparison | API key and account credit |

Docker is optional and is not required for Day 1.

## Day 1 schedule

| Time | Lesson | Capstone increment |
|---|---|---|
| 09:00-09:30 | Course introduction and architecture | Shared evidence contract |
| 09:30-10:00 | 01 — Model gateway | Provider-neutral response |
| 10:00-10:30 | 02 — Prompts and structured outputs | Validated analyst brief |
| 10:30-10:45 | Break | |
| 10:45-11:30 | 03 — Context engineering and CAG | Complete-document answer |
| 11:30-12:00 | 04 — RAG from first principles | First retrieval-backed answer |
| 12:00-13:30 | Lunch | |
| 13:30-15:00 | 05 — Financial documents and chunking | Configurable ingestion pipeline |
| 15:00-15:15 | Break | |
| 15:15-16:00 | 06 — Embeddings and hybrid retrieval | Filtered retrieval and reranking |
| 16:00-16:45 | 07 — RAG evaluation and tracing | Evaluation suite and traces |
| 16:45-17:00 | Integration checkpoint | Complete Day 1 financial RAG pipeline |

## Course guides

- [Install and configure the course](docs/getting-started.md)
- [Follow the Day 1 student guide](docs/day-1-student-guide.md)
- [Resolve setup and notebook issues](docs/troubleshooting.md)
- [Understand the course architecture](docs/course-architecture.md)
- [Review the complete two-day blueprint](docs/program-blueprint.md)

## Repository map

```text
chapters/               Instructor learning contracts
decks/                  Introduction and technical micro-decks
notebooks/              Guided student labs
src/finai_academy/      Reusable application components
data/course/            Versioned teaching evidence
final-project/          Financial Analyst Copilot capstone
scripts/                Setup, execution, and validation commands
tests/                  Engineering and course contracts
```

Day 1 consists of the introduction plus canonical Lessons 01-07. Day 2 adds
workflows, tools, bounded agents, and MCP. The full Lesson 10 route is ready for an instructor-led test class.

## Copyright

Copyright © 2026 Arnaud Demes. All rights reserved.
