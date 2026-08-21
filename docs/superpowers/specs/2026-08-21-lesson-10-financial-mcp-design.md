# Lesson 10 Financial MCP Design

## Status

Approved in-chat direction from Arnaud Demes on 21 August 2026. This design
implements the approved Day 2 slot from 11:15 to 12:00 and preserves the
course rule: add the smallest useful system, make its value observable, and
keep the finance example read-only.

The lesson follows the useful progression of the MLExpert Academy MCP lesson
without reproducing its code, copy, portfolio database, or trading scenario.
The original course implementation uses NVIDIA and Schneider Electric,
existing tested financial capabilities, typed results, controlled fixtures,
and original diagrams and exercises.

## Purpose

Lesson 09 imports a Python metric registry directly into the agent
application. Lesson 10 answers the next engineering question:

> If financial capabilities live behind another process, how can an analyst
> application discover their schemas, call them safely, and retain evidence?

The observable result is a local MCP client that imports no server business
function, discovers one resource, two tools, and one prompt at runtime, then
uses them over `stdio`.

## Classroom boundary

- Duration: 45 minutes.
- Format: 10-minute concept deck, 30-minute guided notebook, 5-minute
  verification and debrief.
- The core demonstrates one server and one client.
- The server is read-only: no orders, transactions, rebalancing, or portfolio
  mutation.
- The core transport is local `stdio`; Streamable HTTP is named only as the
  production extension.
- No SQLite database, authentication implementation, deployment, or
  multi-server orchestration appears in the core.
- Protocol operations work without an LLM. A final provider-neutral extension
  lets Ollama or OpenAI select from the discovered tool names.
- The source notebook remains output-free and contains no secrets.

## Learning objectives

By the end of the lesson, a learner can:

1. distinguish an MCP host, client, server, and transport;
2. explain who controls resources, tools, and prompts;
3. register one example of each primitive with the official Python SDK;
4. connect through `stdio` and complete the MCP initialization lifecycle;
5. discover capabilities instead of importing server functions;
6. read a resource, call a tool, and render a prompt;
7. inspect a typed validation error crossing the protocol boundary; and
8. state the permissions and trust checks required before exposing a tool to a
   model.

## Design alternatives

### Selected: one resource, two read-only tools, one prompt

This is the smallest design that teaches all three primitives and still feels
like a real Financial Analyst Copilot boundary. It reuses the course metric and
document evidence contracts and keeps the notebook inside 30 minutes.

### Rejected: portfolio ledger with SQLite and trade tools

This is a useful standalone MCP application but adds state mutation, accounting
rules, database setup, transaction safety, and financial-action risk. Those
concerns would obscure protocol discovery and conflict with the course's
read-only research boundary.

### Rejected: tools only

A tools-only server is shorter, but students would leave without understanding
the distinct control models of resources, tools, and prompts. The three-way
comparison is the main conceptual value of this lesson.

## Architecture

```text
Financial Analyst host
        |
        | owns lifecycle, permissions, context and model access
        v
MCP client
        |
        | initialization + discovery + stdio messages
        v
Financial MCP server
        |
        +-- resource: finance://coverage
        +-- tool: get_company_metric
        +-- tool: search_financial_documents
        +-- prompt: compare_companies
        |
        v
Tested course registries and versioned evidence
```

The protocol adapter does not duplicate financial logic. Pure Python capability
functions own validation, evidence identifiers, dates, sources, and stable
result schemas. The MCP server decorates those functions. The client knows only
protocol names and schemas.

## MCP primitives

### Resource: `finance://coverage`

Control: application-controlled.

The resource returns a JSON object containing:

```text
dataset_id, as_of, companies, tickers, supported_metrics,
document_ids, source_notice
```

It tells the host what the server can ground before a model is asked to choose
a tool. It contains no credentials, local filesystem paths, or hidden
instructions.

### Tool: `get_company_metric`

Control: model-controlled, subject to host approval and allowlisting.

Input:

```text
ticker, metric
```

Success result:

```text
status, ticker, company, metric, value, unit, as_of, source
```

Validation failure:

```text
status=error, error_code, rejected_value, valid_values, retryable
```

The maintained failure uses `metric="PE"`. The error must preserve the valid
name `P/E` so the student can connect MCP error handling to Lesson 09.

### Tool: `search_financial_documents`

Control: model-controlled, subject to host approval and allowlisting.

Input:

```text
company, query, top_k
```

Result:

```text
status, query, company, hits[{evidence_id, text, document_id,
section, period, source}], trace_id
```

The tool searches a small versioned catalog derived from the tracked NVIDIA and
Schneider Electric course documents. It remains deterministic for classroom
recovery and does not make a network request.

### Prompt: `compare_companies`

Control: user-controlled.

Arguments:

```text
metric, question
```

The rendered user message asks for a source-backed comparison, requires both
companies, names the available protocol capabilities, and forbids investment
recommendations. It is a reusable message template, not an automatic model
call.

## Client lifecycle and data flow

The guided run makes each protocol phase visible:

1. the host starts the server subprocess;
2. the client opens the `stdio` transport;
3. client and server initialize and negotiate capabilities;
4. the client lists resources, tools, and prompts;
5. the host reads `finance://coverage` into application context;
6. the client calls both read-only tools;
7. the user selects and renders `compare_companies`;
8. the client exits its async context and closes the subprocess.

The notebook records one trace row per operation with:

```text
sequence, primitive, operation, capability, status, duration_ms,
evidence_count, error_code
```

The core run never depends on model output. In live mode, the shared Ollama or
OpenAI gateway receives the discovered tool catalog and returns one structured
tool selection. Python validates the selected name against the discovered and
allowlisted names before the MCP client executes it.

## Error handling and trust boundary

- Server validation failures are returned as typed tool errors, not notebook
  crashes.
- `top_k` has a small enforced range.
- Company, ticker, metric, and tool-name allowlists are explicit.
- Tool descriptions and server-returned content are treated as untrusted data.
- The client refuses unknown or non-allowlisted tools even if a model requests
  one.
- Every successful result retains evidence, source, and date metadata.
- The notebook does not send local files, credentials, or personal data.
- `stdio` is presented as a local transport, not as an authentication boundary.
- Streamable HTTP requires separate authentication, authorization, rate
  limits, audit logging, and deployment controls.

## Notebook design

The notebook is `notebooks/10_financial_mcp.ipynb` and follows this sequence:

1. learning objectives and Lesson 09 handoff;
2. provider and deterministic fallback setup;
3. visual comparison of direct import and MCP discovery;
4. inspect the pure capability contracts;
5. inspect the four FastMCP registrations;
6. start the `stdio` client and visualize initialization;
7. display the discovered capability catalog;
8. read `finance://coverage`;
9. call the metric and document-search tools;
10. render `compare_companies`;
11. failure lab with `metric="PE"`;
12. optional Ollama or OpenAI structured tool selection;
13. verification, challenge, capstone integration, and recap.

At least five code-generated visuals are required:

- direct import versus MCP architecture;
- host-client-server sequence;
- resources/tools/prompts control table;
- discovered capability catalog;
- successful and failed protocol traces.

The exact completion marker is `LESSON_10_PASS`.

## Concept deck design

The deck is `decks/10-financial-mcp.pptx`, contains nine slides, uses the
existing First Finance visual system, and includes a source block in every
speaker-notes section.

1. Build a Financial MCP.
2. Direct import creates tight coupling.
3. Host, client, server, and transport.
4. Resources versus tools versus prompts recap table.
5. One financial server, four capabilities.
6. Discovery happens at runtime.
7. The `stdio` request sequence.
8. Discovery is not permission.
9. Production rule and transition to Lesson 11.

Slide copy must use short English sentences, avoid em dashes, and keep one
dominant teaching point per slide.

## Instructor chapter

`chapters/10-financial-mcp.md` provides:

- the exact 11:15-12:00 timing;
- a 10-minute slide script;
- a 30-minute notebook route;
- expected output after each protocol step;
- common setup and subprocess failures;
- a no-network recovery path;
- a skip-if-late path that retains discovery, one resource, one tool, one
  prompt, and the trust-boundary debrief;
- debrief questions and the Lesson 11 handoff.

## Implementation boundaries

The implementation adds focused units:

- a pure financial capability registry;
- an MCP server adapter;
- an async MCP client and trace helper;
- a versioned Lesson 10 evidence catalog;
- the notebook builder and source notebook;
- the chapter and nine-slide deck;
- tests and a delivery-readiness report.

The implementation adds the official MCP Python SDK as an explicit project
dependency. The exact supported version range is chosen after verifying the
current SDK API and is locked in `uv.lock`.

## Verification strategy

### Pure capability tests

- valid NVIDIA and Schneider metric calls retain provenance;
- invalid `PE` produces the required typed error;
- document search returns only the requested company;
- invalid `top_k` and unknown companies fail safely.

### MCP server tests

- an in-process client discovers exactly the intended primitives;
- the resource is readable;
- both tools have stable input schemas;
- the prompt renders one user-controlled message;
- server errors cross the protocol as errors.

### Transport test

- a real `stdio` subprocess initializes, lists capabilities, performs one read
  and one tool call, then exits cleanly.

### Notebook and deck tests

- the checked-in notebook is output-free and has unique cell IDs;
- offline execution produces at least five PNG outputs and
  `LESSON_10_PASS`;
- Ollama and OpenAI use the same structured-selection contract;
- the deck has nine slides, the required footer, simple English, and complete
  source notes;
- rendered slides pass visual inspection and overflow checks.

### Course regression

- the full repository test suite passes;
- Ruff passes;
- all indexes and the course manifest expose Lesson 10;
- the readiness report records offline, Ollama, OpenAI, deck, and timing status
  without claiming an unavailable provider run.

## Acceptance criteria

Lesson 10 is ready when a fresh clone can:

1. install the locked MCP dependency with the existing `uv` workflow;
2. run the notebook offline without network access;
3. connect to the server over `stdio`;
4. discover one resource, two tools, and one prompt;
5. complete one resource read, two successful tool calls, one prompt render,
   and one typed failure;
6. show at least five readable visuals;
7. pass the targeted and full test suites; and
8. transition cleanly to the Lesson 11 plan-and-execute analyst.

## Sources

- MLExpert Academy, “Build an MCP Agent”:
  https://www.mlexpert.io/academy/v1/ai-agents/build-mcp-agent
- Official MCP Python SDK:
  https://github.com/modelcontextprotocol/python-sdk
- Official MCP Python SDK first steps:
  https://github.com/modelcontextprotocol/python-sdk/blob/main/docs/get-started/first-steps.md
- Model Context Protocol documentation:
  https://modelcontextprotocol.io/
- Approved Day 2 progression:
  `docs/superpowers/specs/2026-08-21-day-two-agent-progression-and-lesson-08-design.md`
