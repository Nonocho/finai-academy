# AI Engineering for Asset Management

## Canonical program blueprint

This document is the source of truth for the course. It captures the complete
zero-to-hero learning path before individual notebooks, decks, and exercises are
produced.

The program is inspired by a top-down engineering progression: build a useful
system early, observe its failures, learn the concepts required to improve it,
and integrate every improvement into one cumulative product.

The course does not reproduce another academy's lessons or code. It adapts that
effective learning structure to the needs of asset-management professionals and
uses original financial examples, exercises, datasets, and implementation.

## Audience and promise

The primary learner is a finance professional, analyst, data practitioner, or
technically curious developer who can use a laptop but is not yet an AI engineer.

By the end, the learner can design, build, evaluate, and deploy a trustworthy
Financial Analyst Copilot that:

- works with local or hosted language models;
- produces validated structured outputs;
- searches filings and annual reports with citations;
- uses typed financial, market-data, calculation, and news tools;
- executes deterministic workflows and bounded agent loops;
- exposes and consumes tools through MCP;
- distinguishes facts, calculations, management claims, and interpretations;
- can be evaluated for retrieval, answer, citation, and tool-use quality; and
- runs as a deployable application rather than only as a notebook.

NVIDIA is the principal US case study and Schneider Electric is the principal
European case study. Exercises use real public-company questions while avoiding
investment recommendations.

## Teaching philosophy

### Product first

Learners see the final product architecture before writing code. Each module adds
one observable capability to the same capstone.

### Build, observe, debug

Every major concept follows the same sequence:

1. build the simplest working version;
2. observe a concrete failure;
3. study the mechanism behind the failure;
4. improve the design;
5. verify the improvement with tests or evaluation data.

### Notebook first, application backed

Approximately 80% of the guided learning happens in notebooks. Reusable code is
then moved into `src/finai_academy`, and the integrated product lives in
`final-project`. This preserves accessibility without teaching learners to ship a
production application as one giant notebook.

### Workflows before agents

Learners first build deterministic functions and workflows. Agents are introduced
only when dynamic tool choice or replanning creates clear value. The core product
uses one bounded agent; multi-agent orchestration is an optional advanced topic.

### Evaluation throughout

Evaluation is not a final chapter added after the product is complete. Every
module creates evaluation cases for the capability it introduces.

## Standard lesson format

Each lesson contains:

1. **Why and architecture** - the business problem, final behavior, and system map;
2. **Guided notebook** - a minimal implementation built step by step;
3. **Failure lab** - a controlled example that exposes limitations;
4. **Verification** - tests, expected outputs, or an evaluation dataset;
5. **Challenge** - a scaffold with selected logic removed; and
6. **Product integration** - the reusable component added to the capstone.

Every module ends with an engineering mission. The mission provides a product
specification and scaffold, but not the complete solution.

## Program map

| Module | Theme | Lessons | Product milestone |
|---:|---|---:|---|
| 00 | Build your first financial AI app | 7 | Financial Brief v1 |
| 01 | Context engineering and financial RAG | 10 | Filings Intelligence v2 |
| 02 | Tools and deterministic workflows | 7 | Research Workflow v3 |
| 03 | Agents and MCP | 10 | Bounded Research Agent v4 |
| 04 | AI systems engineering | 8 | Reliability and Evaluation v5 |
| 05 | Production and deployment | 8 | Deployed Copilot v6 |
| Capstone | NVIDIA and Schneider research | — | Financial Analyst Copilot |

## Module 00 - Build your first financial AI app

**Outcome:** build a working analyst assistant before introducing RAG or agents.

### 00.1 System map and final product demo

- distinguish a model from an AI application;
- identify prompt, model, context, memory, tools, workflow, and evaluation layers;
- inspect the final Financial Analyst Copilot architecture;
- define appropriate and inappropriate asset-management use cases.

### 00.2 Practical Python, Git, and environment setup

- run notebooks and Python modules;
- create a project with `uv`;
- use Git for a safe individual workflow;
- manage `.env` files and API secrets;
- read tracebacks and debug common setup failures.

### 00.3 Run an LLM locally with Ollama

- install Ollama and download a small model;
- call the model from the command line and Python;
- understand messages, roles, tokens, context, sampling, and inference;
- measure local latency and resource constraints.

### 00.4 Add an OpenAI model provider

- configure an API key safely;
- use a provider-neutral model gateway;
- switch between Ollama and OpenAI without changing application logic;
- compare quality, latency, privacy, and cost.

### 00.5 Prompt engineering for financial analysis

- treat prompts as versioned application code;
- separate instructions, context, examples, constraints, and output criteria;
- use delimiters and few-shot examples;
- distinguish reported facts from management claims and interpretation;
- test prompts against adversarial and ambiguous inputs.

### 00.6 Structured outputs with Pydantic

- turn model text into typed application data;
- define fields, enumerations, optional values, and validation rules;
- distinguish valid JSON from a semantically correct response;
- retry or fail safely when validation fails;
- render the same structured result in a UI, API, or database.

### 00.7 Conversation, memory, and streaming

- manage conversation history explicitly;
- compare stateless, windowed, and summarized memory;
- stream responses through a minimal user interface;
- measure token usage and observe context-window degradation.

### Module 00 mission - Financial Brief v1

Build an assistant that accepts an earnings-release excerpt and produces a
validated analyst brief containing key results, catalysts, risks, management
claims, and open questions. It must run with Ollama and optionally OpenAI.

## Module 01 - Context engineering and financial RAG

**Outcome:** answer questions over official financial documents with traceable
evidence.

### 01.1 Long context and cache-augmented generation

- load a small complete document into the context;
- understand when full-context prompting is sufficient;
- observe context-window, latency, cost, and lost-in-the-middle limitations;
- choose between direct context, cache-augmented generation, and retrieval.

### 01.2 RAG from first principles

- split text with plain Python;
- build TF-IDF retrieval with cosine similarity;
- inject retrieved passages into a grounded prompt;
- explain RAG as retrieval plus controlled context construction.

### 01.3 Financial document ingestion

- use SEC HTML and XBRL for NVIDIA where available;
- use PDF, XHTML, and ESEF sources for Schneider Electric;
- extract machine-generated PDFs with `pdfplumber`;
- preserve pages, headings, tables, footnotes, order, and provenance;
- normalize sources into a canonical `DocumentBlock` model.

### 01.4 Chunking strategy laboratory

- implement fixed, recursive, structural, semantic, hierarchical, and LLM-based
  chunking;
- test parent-child, contextual, and proposition-based representations;
- keep table and section integrity;
- compare strategies using the same corpus, queries, and evaluator.

### 01.5 Embeddings and vector indexes

- explain embeddings and similarity without unnecessary mathematics;
- compare local and hosted embedding models;
- persist vectors and metadata;
- understand why model changes require index versioning.

### 01.6 Metadata and filtered retrieval

- filter by company, period, document type, section, currency, and language;
- avoid cross-company and cross-period evidence leakage;
- design stable identifiers and provenance metadata.

### 01.7 Hybrid retrieval

- compare lexical and dense retrieval;
- understand exact-term and accounting-language failures;
- combine results with reciprocal-rank fusion;
- select an evidence-oriented top-k policy.

### 01.8 Advanced and hierarchical retrieval

- rerank results;
- rewrite or decompose queries;
- expand neighbors and recover parent sections;
- navigate document trees and compare vectorless retrieval;
- budget final context and remove duplicates.

### 01.9 Evidence-backed generation

- create page- and URL-level citations;
- classify reported fact, calculation, management claim, external fact,
  interpretation, and open question;
- reconcile conflicting evidence;
- abstain when the corpus cannot support an answer.

### 01.10 RAG tracing and evaluation

- create a gold set of analyst questions;
- measure retrieval recall, ranking, groundedness, citation correctness, and
  answer completeness;
- trace ingestion, retrieval, reranking, and generation;
- compare chunking and retrieval strategies empirically.

### Module 01 mission - Filings Intelligence v2

Index selected NVIDIA and Schneider Electric reports, answer questions with
citations, and submit a comparison of at least three chunking or retrieval
strategies.

## Module 02 - Tools and deterministic workflows

**Outcome:** combine documents with structured and current external information.

### 02.1 Model, function, workflow, or agent?

- classify tasks by required determinism and autonomy;
- identify when normal Python is safer than an LLM;
- choose sequential, routing, parallel, or agentic execution.

### 02.2 Typed tool calling

- define tool names, descriptions, argument schemas, and return contracts;
- validate arguments and observations;
- handle timeouts, retries, empty results, and unavailable tools;
- prevent the model from inventing tool results.

### 02.3 Official filing and financial-fact tools

- retrieve SEC filings and company disclosures;
- query XBRL facts;
- maintain source dates, periods, currencies, and units;
- distinguish reported from calculated values.

### 02.4 Market-data and calculation tools

- retrieve educational market-price data;
- calculate returns, growth, margins, and simple ratios deterministically;
- add an optional ECB foreign-exchange tool;
- display calculation inputs and formulas.

### 02.5 Current news with Tavily

- search by company, topic, and date;
- deduplicate syndicated articles;
- classify primary, major-media, specialist, and unknown sources;
- retrieve original URLs and treat web content as untrusted input.

### 02.6 Financial research workflows

- route questions to documents, structured facts, prices, news, or calculators;
- combine deterministic steps into an inspectable research workflow;
- return partial results when one source fails.

### 02.7 Reliability and human control

- implement caching, rate limits, idempotency, and fallback providers;
- require confirmation for consequential actions;
- log inputs, outputs, timing, and errors without exposing secrets.

### Module 02 mission - Research Workflow v3

Produce a sourced company update that combines official documents, structured
facts, prices, calculations, and current news without using an autonomous agent.

## Module 03 - Agents and MCP

**Outcome:** add controlled autonomy and interoperable external tools.

### 03.1 Workflows versus agents

- identify dynamic decisions that justify an agent;
- compare inspectability, flexibility, latency, and failure modes;
- convert one workflow step into a bounded agent decision.

### 03.2 Build an agent loop from scratch

- implement reason, select, act, observe, and stop;
- set iteration, token, time, and tool budgets;
- preserve an inspectable trajectory.

### 03.3 ReAct and tool selection

- connect reasoning to typed tool execution;
- improve descriptions and argument contracts;
- prevent redundant calls and premature completion.

### 03.4 Agent state and memory

- separate conversation, workflow, and research state;
- checkpoint and resume execution;
- summarize history without losing key evidence.

### 03.5 Self-correction and recovery

- detect invalid observations and failed calls;
- revise queries or arguments;
- use explicit stop and escalation conditions;
- avoid uncontrolled reflection loops.

### 03.6 Plan, execute, and replan

- represent a research plan as structured data;
- execute steps through a tool registry;
- update the plan when findings change;
- report progress and unresolved work.

### 03.7 MCP fundamentals

- explain MCP host, client, server, tools, resources, and prompts;
- compare MCP with application-specific function calling;
- discover capabilities and understand trust boundaries.

### 03.8 Build a financial MCP server

- expose filings, facts, prices, calculations, and document search;
- define narrow inputs and stable results;
- add authentication, permissions, and audit metadata.

### 03.9 Consume MCP and build agentic RAG

- connect an MCP client;
- let a bounded agent choose between RAG, structured facts, and news;
- synthesize multi-source evidence without hiding provenance.

### 03.10 Evaluate agent trajectories

- score required, correct, redundant, and failed tool calls;
- assess both trajectory and final answer;
- build regression cases for routing, recovery, and stopping behavior.

### Module 03 mission - Bounded Research Agent v4

Build a single agent that plans and executes a comparative research question
using the course tools and MCP server. Multi-agent research is an optional
extension, not a core requirement.

## Module 04 - AI systems engineering

**Outcome:** replace ad hoc experimentation with measurable, maintainable
components.

### 04.1 Model gateways and capability differences

- normalize Ollama and OpenAI access;
- detect structured-output, tool-use, and context capabilities;
- select models by task, privacy, quality, latency, and cost.

### 04.2 Prompt, schema, and data versioning

- version prompts and structured contracts;
- record model and configuration metadata;
- make experiments reproducible.

### 04.3 Build an evaluation dataset

- collect representative questions and expected evidence;
- create difficult, adversarial, and insufficient-evidence cases;
- separate development and holdout sets.

### 04.4 Evaluate models and applications

- combine deterministic checks, domain review, and LLM judges;
- calibrate judges against human decisions;
- compare models without relying on generic leaderboards.

### 04.5 Prompt optimization

- establish a baseline;
- change one variable at a time;
- explore automated prompt optimization as an advanced technique;
- verify improvement against holdout cases.

### 04.6 Performance and cost engineering

- cache model, embedding, retrieval, and tool results;
- batch work and apply concurrency safely;
- define latency and cost budgets.

### 04.7 Security and governance

- defend against prompt injection in documents and web pages;
- isolate secrets and sensitive portfolios;
- apply permissions, audit logs, retention, and human approval;
- communicate that the system supports research rather than giving advice.

### 04.8 Observability and regression testing

- trace model, retrieval, workflow, and agent steps;
- monitor failures, latency, tokens, and cost;
- run capability-specific regression suites before releases.

### Module 04 mission - Reliability and Evaluation v5

Create an evaluation report and reliability dashboard for the capstone, then fix
one measured retrieval, prompting, or tool-use failure.

## Module 05 - Production and deployment

**Outcome:** ship the copilot as an operable application.

### 05.1 From notebooks to a Python package

- separate domain, application, infrastructure, and interface code;
- move stable components into `src/finai_academy`;
- keep notebooks as experiments and teaching assets.

### 05.2 FastAPI application layer

- expose chat, ingestion, research, and health endpoints;
- validate requests and responses;
- stream long-running research responses.

### 05.3 User interface

- build a simple research chat;
- display sources, evidence types, calculations, and execution traces;
- support company, period, and document filters.

### 05.4 Persistent storage

- store documents, chunks, embeddings, conversations, traces, and evaluations;
- introduce PostgreSQL and vector storage when justified;
- define migrations and retention policies.

### 05.5 Background ingestion

- queue document acquisition, parsing, chunking, and indexing;
- make jobs resumable and idempotent;
- display ingestion status and failures.

### 05.6 Docker and reproducible environments

- package the API, UI, and supporting services;
- configure local development with containers;
- add health checks and explicit versions.

### 05.7 Deployment, monitoring, and operations

- deploy a small demonstration environment;
- configure secrets and provider credentials;
- monitor availability, latency, costs, and quality signals;
- define rollback and incident procedures.

### 05.8 Continuous evaluation

- run smoke and regression evaluations during delivery;
- monitor data and retrieval drift;
- review failed or low-confidence answers.

### Module 05 mission - Deployed Copilot v6

Deploy the Financial Analyst Copilot with a documented architecture, evaluation
report, sample research session, and operating guide.

## Final capstone - Financial Analyst Copilot

The final mission is a comparative NVIDIA and Schneider Electric research task.
The exact question can change by cohort, but the product must:

1. answer over official filings with citations;
2. retrieve structured financial facts and market data;
3. perform deterministic calculations;
4. find and qualify recent news;
5. distinguish evidence from interpretation;
6. use a bounded research workflow or agent;
7. expose at least one capability through MCP;
8. show its execution trace;
9. pass a documented evaluation suite; and
10. run locally with Ollama and optionally with OpenAI.

The submitted portfolio contains the application repository, architecture diagram,
evaluation report, sample outputs, limitations, and a short product demonstration.

## Scope controls

The following topics are electives rather than core requirements:

- multi-agent teams;
- fine-tuning an LLM;
- visual-language-model document extraction;
- cloud GPU serving;
- advanced frontend engineering;
- automated trading or order execution.

Keeping these optional protects the zero-to-hero path while leaving room for an
advanced academy track.
