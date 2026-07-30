# FinAI Academy

**Build trustworthy AI systems for financial research - from first principles to production.**

FinAI Academy is an English-language learning path for finance professionals,
analysts, data practitioners, and software engineers who want to understand and
build modern AI applications with Python.

The course is designed as a reusable academy rather than a fixed two-day event.
Each chapter combines:

- a PowerPoint deck for concepts and architecture;
- a guided Jupyter notebook for experiments;
- checkpoint questions and practical exercises;
- optional challenges for advanced learners;
- one cumulative capstone: **FinResearch Copilot**.

Every public asset in this repository is original FinAI Academy material.

## Learning path

| # | Chapter | Core outcome |
|---:|---|---|
| 00 | Product demo and system map | Understand the product we will build |
| 01 | AI and LLM foundations | Explain tokens, context, inference, limits, and model trade-offs |
| 02 | Prompting and structured outputs | Produce reliable, typed responses |
| 03 | Retrieval from first principles | Build search before introducing vector databases |
| 04 | Document ingestion and chunking | Turn financial documents into retrieval-ready knowledge |
| 05 | Embeddings and advanced retrieval | Combine semantic, lexical, hybrid, and reranked search |
| 06 | RAG with evidence | Generate grounded answers with claim-level citations |
| 07 | Tools and deterministic workflows | Connect models to safe, typed financial tools |
| 08 | LangGraph agents and self-correction | Build stateful agents with retries and stop conditions |
| 09 | Multi-agent financial research | Coordinate specialist agents and synthesize results |
| 10 | Evaluation, observability, and LLMOps | Measure quality, latency, cost, and reliability |
| Capstone | FinResearch Copilot | Assemble the complete application |

## Repository map

```text
finai-academy/
├── assets/                 # Visual system and reusable non-logo assets
├── chapters/               # Chapter briefs and learning contracts
├── decks/                  # One PowerPoint deck per chapter
├── notebooks/              # One guided notebook per chapter
├── src/finai_academy/      # Reusable Python package
├── final-project/          # FinResearch Copilot capstone
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

## Status

The repository structure and learning contracts are established. Chapter content,
decks, notebooks, exercises, and the capstone will be developed incrementally.

## Copyright

Copyright © 2026 Arnaud Demes. All rights reserved.

No redistribution or commercial reuse is granted unless stated otherwise in a
future license.
