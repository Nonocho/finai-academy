# Two-Day AI Engineering for Asset Management Class

## Design status

Approved direction: **Option A - balanced AI Engineering bootcamp**.

This specification defines the two-day technical course to be validated with
Antoine. It is the source of truth for the external proposal, lesson sequence,
slides, notebooks, and capstone scope.

## Course identity

**Title:** AI Engineering for Asset Management

**Subtitle:** Build a Financial Analyst Copilot in Two Days

**Audience:** a software developer who can read, run, and modify Python code and
Jupyter notebooks. The course assumes development experience but does not assume
prior professional experience with LLM applications, RAG, agents, or MCP.

**Format:** two in-person days, 09:00-17:00, with a 90-minute lunch break and
short morning and afternoon breaks.

**Teaching mix:** approximately 20% architecture and concepts, 65% guided
technical practice, and 15% debugging, evaluation, and review.

## Learning promise

By the end of the course, the learner can explain, build, and evaluate the main
layers of a modern financial AI application:

- local and hosted model access;
- prompt engineering and validated structured outputs;
- context engineering, CAG, and RAG;
- financial-document parsing and chunking;
- embeddings, hybrid retrieval, reranking, and citations;
- typed financial tools and deterministic workflows;
- a bounded tool-using agent;
- an MCP server and client;
- retrieval, answer, citation, and trajectory evaluation; and
- integration of these capabilities into a Financial Analyst Copilot.

The course teaches one causal progression. Every new component is introduced to
solve a failure observed in the previous version of the product.

```text
Model call
  -> engineered prompt
  -> structured output
  -> complete document context
  -> retrieval
  -> document-aware chunking
  -> hybrid search and reranking
  -> evaluated RAG workflow
  -> financial tools
  -> bounded agent
  -> MCP
  -> integrated Financial Analyst Copilot
```

## Scope allocation

- Context engineering and RAG: 45%
- Tools, workflows, agents, and MCP: 30%
- Prompting and structured outputs: 15%
- Architecture, evaluation, and production considerations: 10%

The core course uses one bounded agent. Plan-and-execute and multi-agent
architectures are extensions, not required live builds.

## Financial case study

The course uses a stable, versioned research corpus with:

- NVIDIA as the principal US company;
- Schneider Electric as the principal European company;
- selected official filings, annual reports, earnings releases, and investor
  materials;
- a small maintained market-data snapshot for deterministic exercises; and
- optional current news retrieved through Tavily during the live application.

The final research mission compares NVIDIA and Schneider Electric on the theme
of data-centre demand, supporting evidence, financial implications, risks, and
recent developments.

The product supports financial research. It does not make investment decisions,
execute trades, or present interpretations as reported facts.

## Mandatory pre-work

Estimated time: 60-90 minutes.

The learner must complete the following before the first day:

1. install Git;
2. install `uv`;
3. install Docker;
4. install Ollama;
5. clone the course repository;
6. download the configured chat and embedding models;
7. run `uv sync`;
8. run `setup_check.py`; and
9. optionally configure OpenAI and Tavily API keys.

The repository includes a short setup guide and one verification command. The
first live session does not repeat general Python, Git, Docker, or shell training.

## Teaching unit

Each technical block follows the same lesson sandwich:

1. **Why and architecture** - 5 to 15 minutes of slides;
2. **Minimal build** - create the simplest working implementation;
3. **Failure lab** - make a limitation visible;
4. **Improvement** - introduce the relevant engineering pattern;
5. **Verification** - run a test or evaluation case; and
6. **Capstone integration** - move stable code into the application package.

Slides create the mental model. Notebooks contain the real code. The capstone
repository contains the integrated application.

## Day 1 - From LLM to Financial RAG

### 09:00-09:30 - Product demo and system architecture

**Format:** 15 minutes slides, 15 minutes demonstration.

- demonstrate the final Financial Analyst Copilot;
- distinguish model, prompt, context, retrieval, tools, workflow, agent, and MCP;
- show the repository and cumulative build sequence;
- introduce the NVIDIA and Schneider corpus;
- define the evidence and evaluation contract.

**Output:** shared mental model of the product to build.

### 09:30-10:00 - Local and hosted model gateway

**Format:** 10 minutes slides, 20 minutes notebook.

- call a local Ollama model from Python;
- understand messages, tokens, temperature, and context limits;
- switch between Ollama and OpenAI through one application boundary;
- compare privacy, quality, latency, and cost.

**Notebook:** `01_model_gateway.ipynb`

**Capstone increment:** first model-backed response.

### 10:00-10:30 - Prompt engineering and structured outputs

**Format:** 10 minutes slides, 20 minutes notebook.

- treat prompts as versioned code;
- separate instructions, source context, constraints, and examples;
- apply delimiters and few-shot prompting;
- define an `AnalystBrief` with Pydantic;
- validate model output and handle invalid responses;
- distinguish syntactic JSON validity from financial correctness.

**Notebook:** `02_prompts_and_structured_outputs.ipynb`

**Capstone increment:** validated results, catalysts, risks, management claims,
interpretations, and open questions.

### 10:30-10:45 - Break

### 10:45-11:30 - Context engineering and CAG

**Format:** 15 minutes slides, 30 minutes notebook.

- inject a complete source document into the context;
- build a minimal Cache-Augmented Generation application;
- understand when complete-context prompting is sufficient;
- measure context size, latency, and relevance dilution;
- observe a failure on a longer financial document.

**Notebook:** `03_cag_financial_document.ipynb`

**Capstone increment:** document-grounded answer without retrieval.

### 11:30-12:00 - RAG from first principles

**Format:** 10 minutes slides, 20 minutes notebook.

- split text with simple Python;
- build TF-IDF retrieval and cosine similarity;
- retrieve top-k passages;
- construct a grounded prompt;
- abstain when the answer is absent.

**Notebook:** `04_rag_from_scratch.ipynb`

**Capstone increment:** first retrieval-backed answer.

### 12:00-13:30 - Lunch

### 13:30-15:00 - Financial documents and chunking laboratory

**Format:** 20 minutes slides, 70 minutes notebook.

Document engineering:

- use SEC HTML and XBRL where appropriate;
- extract a machine-generated report with `pdfplumber`;
- preserve headings, pages, tables, order, and provenance;
- normalize content into a canonical `DocumentBlock` structure.

Chunking experiment:

- fixed-size;
- recursive;
- structure-aware;
- semantic;
- hierarchical parent-child;
- contextual enrichment; and
- LLM-based boundaries or propositions.

All strategies use the same corpus, questions, embedding model, and evaluator.

**Notebook:** `05_document_and_chunking_lab.ipynb`

**Capstone increment:** configurable ingestion and chunking pipeline.

### 15:00-15:15 - Break

### 15:15-16:00 - Embeddings and hybrid retrieval

**Format:** 15 minutes slides, 30 minutes notebook.

- build intuition for embeddings and cosine similarity;
- persist chunks and metadata in the prepared store;
- compare dense and keyword retrieval;
- expose exact-term failures on tickers and accounting language;
- combine results with reciprocal-rank fusion;
- rerank final candidates;
- filter by company, period, and document type.

**Notebook:** `06_hybrid_retrieval.ipynb`

**Capstone increment:** metadata-filtered hybrid retrieval and reranking.

### 16:00-16:45 - RAG evaluation and tracing

**Format:** 15 minutes slides, 30 minutes notebook.

- create a small gold dataset;
- separate retrieval failures from generation failures;
- measure retrieval relevance, groundedness, citation correctness, answer
  completeness, and abstention;
- trace retrieve, rerank, and generate operations;
- compare chunking strategies using evaluation evidence.

**Notebook:** `07_rag_evaluation.ipynb`

**Capstone increment:** first evaluation suite and traces.

### 16:45-17:00 - Integration checkpoint

- run the complete Day 1 pipeline;
- review one successful and one failed query;
- move stable code from notebooks into the package;
- commit the Day 1 milestone.

**Day 1 result:** parsed documents, configurable chunks, hybrid retrieval,
reranking, cited generation, and an evaluation report.

## Day 2 - From RAG to Agentic Application

### 09:00-09:30 - Debugging review

- inspect Day 1 traces;
- compare chunking outcomes;
- correct one measured retrieval failure;
- introduce the Day 2 application architecture.

### 09:30-10:15 - Stateful RAG workflow with LangGraph

**Format:** 10 minutes slides, 35 minutes notebook.

- define typed graph state;
- create retrieve, rerank, and generate nodes;
- connect deterministic edges;
- stream output;
- maintain conversation memory and checkpoints;
- emit typed citations and completion events.

**Notebook:** `08_langgraph_rag_workflow.ipynb`

**Capstone increment:** stateful streaming RAG workflow.

### 10:15-10:30 - Break

### 10:30-11:15 - Financial tools and deterministic workflows

**Format:** 10 minutes slides, 35 minutes notebook.

Implement a small typed tool registry:

1. `search_financial_documents`;
2. `get_financial_facts`;
3. `get_market_prices`;
4. `calculate_financial_metric`;
5. `search_company_news`; and
6. optional `convert_currency`.

Route known question types through explicit application logic. Validate tool
arguments and return structured errors.

**Notebook:** `09_tools_and_workflows.ipynb`

**Capstone increment:** deterministic multi-source research workflow.

### 11:15-12:00 - Workflow versus agent

**Format:** 15 minutes slides, 30 minutes notebook.

- compare workflows, agents, and multi-agent systems;
- build one fixed tool-use workflow;
- demonstrate its failure on a dynamic multi-step question;
- replace the fixed path with a bounded reason-act-observe loop;
- add maximum steps and explicit stop behavior;
- handle invalid arguments and unavailable data.

**Notebook:** `10_workflow_vs_agent.ipynb`

**Capstone increment:** minimal bounded agent loop.

### 12:00-13:30 - Lunch

### 13:30-14:30 - Reliable agent and trajectory evaluation

**Format:** 15 minutes slides, 45 minutes notebook.

- maintain agent state;
- feed tool errors back as observations;
- retry safely;
- apply tool, iteration, time, token, and cost budgets;
- evaluate tool choice, arguments, ordering, efficiency, and final answer;
- inspect successful and unsuccessful trajectories.

**Notebook:** `11_reliable_agent.ipynb`

**Capstone increment:** recovery, guardrails, and trajectory evaluation.

### 14:30-15:30 - Model Context Protocol

**Format:** 15 minutes slides, 45 minutes notebook.

- explain host, client, server, resources, tools, and prompts;
- compare MCP with direct application tool calling;
- expose existing tested capabilities through a local MCP server;
- connect a client over `stdio`;
- discover capabilities dynamically;
- discuss permissions and trust boundaries.

The live MCP server exposes document search, market data, and financial
calculation capabilities. It does not execute trades.

**Notebook:** `12_financial_mcp.ipynb`

**Capstone increment:** MCP server and client integration.

### 15:30-16:30 - Capstone integration challenge

The learner receives a prepared application scaffold containing:

- repository structure;
- Streamlit interface;
- FastAPI application;
- Docker Compose configuration;
- database schema and migrations;
- API schemas and SSE contracts;
- starter tests; and
- selected public financial documents.

The learner completes or connects the intelligence layer:

1. analyst brief schema and prompt;
2. ingestion and chunking;
3. hybrid retrieval and reranking;
4. LangGraph RAG workflow;
5. financial tool registry;
6. bounded agent;
7. MCP server; and
8. evaluation cases.

**Final mission:** answer a comparative NVIDIA-Schneider research question using
official documents, structured financial data, calculations, and recent news,
while preserving citations and evidence categories.

### 16:30-17:00 - Demonstration and review

- demonstrate the integrated product;
- inspect citations and evidence types;
- inspect the execution trajectory;
- run the evaluation suite;
- review architecture and production limitations;
- identify optional next steps.

## Capstone architecture

```text
Streamlit interface
        |
FastAPI application
        |
LangGraph orchestration
        +-- evaluated document RAG
        +-- typed financial tools
        +-- Tavily news search
        +-- deterministic calculator
        +-- bounded agent
        |
MCP server and client
        |
Traces and evaluation reports
```

The application must label material statements as reported facts, calculations,
management claims, external facts, interpretations, or open questions.

## Capstone implementation boundary

The two-day course does not build infrastructure from a blank folder. The API,
UI, containers, database migrations, schemas, and basic streaming plumbing are
provided. Live work focuses on the AI engineering decisions that the course is
intended to teach.

The final application must:

- run locally with Ollama;
- optionally support OpenAI through configuration;
- cite filing-derived claims;
- display calculation inputs;
- retain URLs and publication dates for news claims;
- abstain when evidence is insufficient;
- constrain agent execution;
- expose an inspectable trace; and
- pass the maintained evaluation cases.

## Slide and lesson system

### Three complementary surfaces

#### Live micro-decks

Each technical block has a five- to seven-slide micro-deck:

1. business or engineering problem;
2. mental model;
3. architecture;
4. failure mode;
5. trade-offs;
6. notebook mission; and
7. debrief where required.

The two-day course uses approximately 55-65 live slides. Slides do not contain
large implementation listings; code lives in notebooks.

#### Lesson website

The course website provides the long-form learning experience:

```text
Lesson
  -> Why this matters
  -> Architecture
  -> Concepts
  -> Guided notebook
  -> Failure lab
  -> Common pitfalls
  -> Verification
  -> Challenge
  -> References
```

Quarto is the recommended source system because it can produce a navigable lesson
website, integrate notebooks and Mermaid diagrams, and render Reveal.js slides
from version-controlled Markdown.

#### Technical repository

```text
ai-engineering-asset-management/
  course/
    day-1/
    day-2/
  notebooks/
    guided/
    solutions/
  slides/
  site/
  capstone/
    backend/
    frontend/
    mcp/
    evals/
  setup_check.py
```

### Visual direction

The visual system takes inspiration from the clarity and engineering character
of MLE Academy without copying its brand:

- dark navy or near-black backgrounds;
- cyan for system components and data flow;
- orange for warnings and failure modes;
- restrained technical typography;
- terminal and code-window components;
- simple architecture diagrams;
- generous whitespace;
- one message per slide; and
- a visible course-progress indicator.

The website contains explanation. Slides control the live narrative. Notebooks
contain the implementation.

## Participant deliverables

The learner leaves with:

1. twelve guided notebooks;
2. completed reusable Python modules;
3. a working Financial Analyst Copilot;
4. an MCP server and client example;
5. a RAG and agent evaluation report;
6. architecture diagrams;
7. the lesson website; and
8. the complete course repository.

## Explicit non-goals

- generic tours of many consumer AI products;
- a model-arena comparison exercise;
- NotebookLM or Perplexity as independent modules;
- no-code automation with n8n;
- autonomous trading;
- a mandatory multi-agent architecture;
- fine-tuning;
- building production infrastructure from scratch during the two live days; or
- claiming full production readiness after a two-day course.

## Success criteria

The design is successful when:

- every live technical block modifies the same capstone;
- the learner writes or completes meaningful code in every notebook;
- RAG is evaluated rather than judged by demonstration quality;
- the distinction between workflow, agent, and MCP is observable in code;
- the final application can answer the maintained NVIDIA-Schneider cases with
  traceable evidence; and
- the entire course fits within the agreed 09:00-17:00 schedule without relying
  on live dependency installation.
