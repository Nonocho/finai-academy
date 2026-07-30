# FinResearch Copilot

The capstone is a private financial research workspace that combines documents,
market data, tools, retrieval, agents, citations, and evaluation.

## Target experience

```text
Workspace
  ├── document library
  ├── portfolio
  └── conversations

Research chat
  ├── answer
  ├── claim-level evidence
  └── fact / calculation / interpretation labels

Execution trace
  ├── plan
  ├── retrieval
  ├── tools
  ├── retries
  └── timing and cost

Evaluation
  ├── retrieval relevance
  ├── groundedness
  ├── answer completeness
  └── tool-call correctness
```

## Architecture rule

The production application reuses code from `src/finai_academy`. It is not
implemented inside a notebook.

## Planned components

- FastAPI backend;
- web interface;
- background document ingestion;
- provider-neutral model adapter;
- hybrid retrieval and reranking;
- deterministic and agentic workflows;
- evaluation dataset and dashboard;
- local-first deployment with cloud fallback.
