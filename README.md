# AI Engineering for Asset Management

**A FinAI Academy program by Arnaud Demes.**

Build trustworthy AI systems for financial research, from the first local model
call to a deployed analyst copilot.

FinAI Academy is an English-language learning path for finance professionals,
analysts, data practitioners, and software engineers who want to understand and
build modern AI applications with Python.

The course is designed as a reusable academy rather than a fixed two-day event.
Each chapter combines:

- a PowerPoint deck for concepts and architecture;
- a guided Jupyter notebook for experiments;
- checkpoint questions and practical exercises;
- optional challenges for advanced learners;
- one cumulative capstone: **Financial Analyst Copilot**.

Every public asset in this repository is original FinAI Academy material.

## Canonical program

The [program blueprint](docs/program-blueprint.md) is the source of truth for
learning outcomes, lesson order, module missions, and scope.

| Module | Theme | Product milestone |
|---:|---|---|
| 00 | First financial AI app: models, prompts, structured outputs, memory | Financial Brief v1 |
| 01 | Context engineering, chunking, retrieval, citations, and RAG evaluation | Filings Intelligence v2 |
| 02 | Typed financial tools, Tavily news, and deterministic workflows | Research Workflow v3 |
| 03 | Bounded agents, recovery, planning, MCP, and trajectory evaluation | Research Agent v4 |
| 04 | Datasets, evaluation, optimization, security, cost, and observability | Reliability v5 |
| 05 | API, UI, persistence, Docker, deployment, and monitoring | Deployed Copilot v6 |
| Capstone | NVIDIA and Schneider Electric comparative research | Financial Analyst Copilot |

The existing chapter briefs and notebooks are seed assets. They will be refactored
into this module sequence as lessons are produced.

## Repository map

```text
finai-academy/
├── assets/                 # Visual system and reusable non-logo assets
├── chapters/               # Chapter briefs and learning contracts
├── decks/                  # One PowerPoint deck per chapter
├── notebooks/              # One guided notebook per chapter
├── src/finai_academy/      # Reusable Python package
├── final-project/          # Financial Analyst Copilot capstone and product spec
├── docs/                   # Authoring, delivery, model, and content standards
├── scripts/                # Repository validation and authoring helpers
└── tests/                  # Tests for shared course code
```

## Delivery formats

The same content can be assembled into different products:

- **Two-day client workshop** - curated chapters, live demos, and a guided capstone;
- **Five-day technical bootcamp** - the complete build sequence;
- **Self-paced academy** - all chapters, solutions, challenges, and evaluation;
- **Executive track** - concepts, architecture, governance, and demonstrations.

The source curriculum is intentionally broader than any single delivery. A client
version is created by selecting chapters and exercises, not by rebuilding the
course from scratch.

## Model strategy

The learning path is provider-neutral:

- Ollama is the local-first teaching baseline;
- OpenAI and Gemini can be enabled through adapters;
- the final demo always supports a cloud fallback;
- notebooks label provider-specific behavior explicitly.

See [docs/model-strategy.md](docs/model-strategy.md).

## Visual identity

The decks use the ScaleNow color system without the ScaleNow logo. Every deck is
signed:

> FinAI Academy - Arnaud Demes

See [assets/brand/finai-academy-style.md](assets/brand/finai-academy-style.md).

## Current status

The canonical program and capstone specification are established. The first
capstone vertical slice defines a provider-neutral, Pydantic-validated analyst
brief and a CLI entry point. RAG, tools, agents, and MCP will be added in the same
order in which learners encounter them.

## Copyright

Copyright © 2026 Arnaud Demes. All rights reserved.

No redistribution or commercial reuse is granted unless stated otherwise in a
future license.
