# Financial Analyst Copilot - Product specification

## Product statement

Financial Analyst Copilot is a conversational, evidence-backed assistant that
helps an analyst investigate a public company across filings, structured facts,
market data, calculations, and current news.

The reference research universe contains:

- NVIDIA as the principal US company; and
- Schneider Electric as the principal European company.

The product supports research. It does not make investment decisions, place
orders, or present generated interpretation as reported fact.

## Primary user jobs

1. Ask a qualitative question about a filing and inspect the supporting passage.
2. Extract a structured analyst brief from a company disclosure.
3. Compare a metric or management theme across companies or periods.
4. Combine official disclosures with current, source-qualified news.
5. See which documents, tools, and calculations produced an answer.
6. Know when the available evidence is insufficient or contradictory.

## Evidence contract

Every material claim is labelled as one of:

- `reported_fact` - directly stated in an authoritative source;
- `calculation` - computed from displayed inputs and a deterministic formula;
- `management_claim` - attributed to company management;
- `external_fact` - reported by an external source;
- `interpretation` - analysis produced by the system; or
- `open_question` - unresolved because evidence is missing or conflicting.

Claims derived from source material retain a URL or document identifier and, when
available, a page, section, publication date, and retrieval timestamp.

## Target experience

```text
Research workspace
  ├── company and period filters
  ├── official document library
  ├── conversation
  └── saved research sessions

Answer
  ├── concise analyst response
  ├── structured brief
  ├── evidence and citations
  ├── calculations and assumptions
  └── uncertainty and open questions

Execution trace
  ├── route or plan
  ├── retrieval queries and results
  ├── tool calls and observations
  ├── retries and fallbacks
  └── latency, token, and cost metadata

Evaluation
  ├── retrieval quality
  ├── citation and groundedness quality
  ├── answer completeness
  └── tool-call and trajectory correctness
```

## Product architecture

```text
Web or CLI interface
        │
Application service
        ├── model gateway: Ollama | OpenAI
        ├── controlled document RAG
        ├── deterministic research workflow
        ├── bounded agent loop
        └── evaluation and tracing
                 │
Capability layer ├── filing search
                 ├── financial facts
                 ├── market prices
                 ├── calculator and FX
                 └── Tavily news search
```

At least one capability will also be exposed through an MCP server. MCP is added
after the underlying typed tool is tested independently.

## Progressive releases

| Version | Module | Added capability |
|---|---:|---|
| v1 | 00 | Prompted, typed analyst brief using Ollama or OpenAI |
| v2 | 01 | Filing ingestion, retrieval, citations, and RAG evaluation |
| v3 | 02 | Financial, market, calculation, and news tools in a workflow |
| v4 | 03 | Bounded research agent and MCP server/client |
| v5 | 04 | Evaluation dataset, tracing, security, cost, and reliability |
| v6 | 05 | API, UI, persistence, Docker, deployment, and monitoring |

## Core tools

The core product intentionally uses a small tool registry:

1. `search_financial_documents`
2. `get_financial_facts`
3. `get_market_prices`
4. `calculate_financial_metric`
5. `search_company_news`
6. optional `convert_currency`

## Initial acceptance criteria

The v1 structured-brief slice is complete when:

- the model provider is selected through environment configuration;
- the same application service supports Ollama and OpenAI adapters;
- output validates against the `AnalystBrief` Pydantic model;
- facts, management claims, interpretations, and open questions are separated;
- invalid provider configuration fails clearly;
- application behavior can be tested with a fake model and no network access; and
- the CLI accepts company, reporting period, and a local source-text file.

## Final acceptance criteria

The final capstone must:

- answer a maintained evaluation set for NVIDIA and Schneider Electric;
- cite official documents for filing-derived claims;
- display calculation inputs and formulas;
- retain URLs and dates for current-news claims;
- reject or qualify unsupported conclusions;
- constrain the agent by tool, iteration, time, and cost budgets;
- expose an inspectable trajectory;
- include retrieval, answer, citation, and agent-trajectory evaluations;
- run locally with Ollama; and
- document optional OpenAI configuration and expected costs.

## Explicit non-goals

- autonomous trading or order execution;
- portfolio recommendations presented as advice;
- unrestricted browsing or code execution;
- twenty loosely defined tools;
- a multi-agent system as the default architecture;
- claiming that any parser, model, or retrieval method is universally reliable.
