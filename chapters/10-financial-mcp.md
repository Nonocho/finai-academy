# Lesson 10 — Financial MCP

**First Finance - Arnaud Demes**
**Day 2 · 11:15–12:00 · 10-minute concept deck + 30-minute notebook + 5-minute verification and debrief**

## Instructor outcome

Students connect a local financial application to one MCP server over `stdio`, discover its declared capabilities at runtime, and use them within a host-owned permission boundary. The NVIDIA (`NVDA`) and Schneider Electric (`SU.PA`) examples are read-only, deterministic, and grounded in controlled course fixtures. This lesson does not demonstrate trading, portfolio changes, or an investment recommendation.

The instructor chapter and notebook route are ready. The full 45-minute delivery route is pending Task 6, which creates and certifies the nine-slide deck; do not present the lesson as fully deliverable until that deck is available.

```text
host starts local server → client discovers capabilities → host reads context
→ client calls read-only tools → user reviews prompt → host verifies trace
```

## Before class

Run these commands from the repository root:

```bash
uv sync --frozen --extra ai --extra rag --extra finance --extra evaluation --extra dev
uv run python -c "from mcp.server import MCPServer; print(MCPServer.__name__)"
uv run python scripts/build_lesson10_notebook.py
uv run python scripts/validate_notebooks.py notebooks/10_financial_mcp.ipynb
uv run jupyter lab
```

Open `notebooks/10_financial_mcp.ipynb`. Offline mode is the core lesson: it uses the local server and requires neither network nor a model. For optional Ollama, prepare it before class:

```bash
ollama pull qwen3:8b
FINAI_LIVE_MODE=1 FINAI_MODEL_PROVIDER=ollama FINAI_CHAT_MODEL=qwen3:8b \
  uv run python scripts/execute_notebooks.py notebooks/10_financial_mcp.ipynb \
  --mode live --provider ollama --output-dir /private/tmp/finai-lesson10-ollama
```

For an optional OpenAI comparison, configure the key outside the notebook and repository:

```bash
export OPENAI_API_KEY="..."
FINAI_LIVE_MODE=1 FINAI_MODEL_PROVIDER=openai FINAI_CHAT_MODEL=gpt-5-mini \
  uv run python scripts/execute_notebooks.py notebooks/10_financial_mcp.ipynb \
  --mode live --provider openai --output-dir /private/tmp/finai-lesson10-openai
```

Never display, print, trace, or commit the key. A live provider is optional; the shared gateway gives Ollama and OpenAI the same discovered-tool contract.

## Static recovery catalog

Use this table only when local discovery cannot run. It is a teaching fallback, not a substitute for runtime discovery or host policy. The controlled evidence fixture is `assets/course-data/mcp/lesson10_evidence_catalog_v1.json`.

| Primitive | Controller | Expected capability | Classroom purpose |
|---|---|---|---|
| Resource | Application | `finance://coverage` | Read the controlled coverage boundary before tool selection. |
| Tool | Model + host approval | `get_company_metric` | Read one dated, source-bearing metric. |
| Tool | Model + host approval | `search_financial_documents` | Find controlled evidence with document and evidence IDs. |
| Prompt | User | `compare_companies` | Render a reusable comparison request for review. |

The expected discovery result is exactly one resource, two tools, and one prompt. The host must still allowlist a tool name, validate its arguments, and apply permissions.

## 10-minute concept deck

Task 6 creates and certifies the nine-slide `decks/10-financial-mcp.pptx` deck. Until Task 6 is complete, this table is the planned slide route, not an available deck or a complete delivery substitute. Once available, use the deck to frame the notebook; do not read the slides aloud.

| Time | Slide | Instructor job |
|---:|---:|---|
| 0:00–1:00 | 1 | State the question: discover capabilities instead of importing server functions. |
| 1:00–2:00 | 2 | Contrast a direct Python import with a declared process boundary. |
| 2:00–3:15 | 3 | Name host, client, server, and transport; the host owns lifecycle and permissions. |
| 3:15–4:30 | 4 | Separate resources, tools, and prompts by their controller. |
| 4:30–5:30 | 5 | Preview one resource, two read-only tools, and one prompt. |
| 5:30–6:45 | 6 | Explain runtime discovery: list, inspect schema, decide, call. |
| 6:45–7:45 | 7 | Show local `stdio`: the host starts and closes a subprocess transport. |
| 7:45–9:00 | 8 | State the rule: discovery is not permission; allowlist and validate. |
| 9:00–10:00 | 9 | Set the production boundary and preview the Lesson 11 handoff. |

The rows total 10 minutes.

## Version and transport note

This course uses the official MCP Python SDK v2 name `MCPServer`. Older v1 tutorials, including the reviewed MLExpert inspiration, may use `FastMCP`. The decorator pattern remains familiar: `@mcp.resource()`, `@mcp.tool()`, and `@mcp.prompt()` register declared capabilities. Do not change the notebook to the earlier import name.

The classroom core uses local `stdio`: the host starts one subprocess and the client carries protocol messages on standard streams. `stdio` is local transport, not authentication. Streamable HTTP is a production extension only; it needs separate authentication, authorization, rate limits, audit logging, and deployment controls.

## 30-minute notebook route

The route maps to checked-in stable cell IDs and visible outputs. Do not create a second server or replace the controlled fixture with live market data.

| Time | Cells | Instructor action | Expected visible output |
|---:|---|---|---|
| 0:00–3:00 | `lesson10-000`–`lesson10-003` | Connect Lesson 09 direct imports to MCP discovery, then run setup. | Lesson context, `offline fixture · deterministic course run`, `Transport: local stdio`, read-only boundary. |
| 3:00–6:00 | `lesson10-004`–`lesson10-005` | Compare direct imports with the host-client-server boundary. | Figure 1: direct import versus MCP discovery. |
| 6:00–9:00 | `lesson10-006`–`lesson10-008` | Inspect contracts and four `MCPServer` registrations; identify controllers. | Two contract tables and Figure 2 control matrix. |
| 9:00–13:00 | `lesson10-009`–`lesson10-011` | Start the real lifecycle; name each phase before running it. | `First Finance Research`; one resource, two tools, one prompt; Figure 3 `stdio` sequence. |
| 13:00–16:00 | `lesson10-012`–`lesson10-014` | Read discovery as input to host policy, not permission. | Capability table plus Figure 4 with 1 resource, 2 tools, 1 prompt. |
| 16:00–20:00 | `lesson10-015`–`lesson10-016` | Read `finance://coverage`, inspect a metric, search documents, render `compare_companies`. | Coverage, dated metric with source, document hits with evidence IDs, rendered user prompt. |
| 20:00–24:00 | `lesson10-017`–`lesson10-018` | Run the maintained invalid alias and discuss the typed result. | `unsupported_metric`, valid `P/E`, retryability, trace table, Figure 5. |
| 24:00–26:00 | `lesson10-019`–`lesson10-020` | Show live selection only if ready; otherwise keep offline selection. | Offline allowlisted choice, or Ollama/OpenAI structured selection validated by Python. |
| 26:00–28:00 | `lesson10-021`–`lesson10-022` | Verify each observable contract. | `LESSON_10_PASS`. |
| 28:00–30:00 | `lesson10-023`–`lesson10-025` | Knowledge check, capstone increment, and recap. | Challenge policy, capstone boundary, Lesson 11 handoff. |

The rows total 30 minutes. The core runs without a model. In live mode, the gateway receives only discovered tool names, descriptions, and input schemas; Python checks the returned name against runtime discovery and the allowlist.

## 5-minute verification and debrief

| Time | Instructor action | Evidence |
|---:|---|---|
| 0:00–2:00 | Confirm final assertions and ask a learner to read the catalog aloud. | Exactly `finance://coverage`, `get_company_metric`, `search_financial_documents`, and `compare_companies`; `LESSON_10_PASS`. |
| 2:00–3:30 | Ask who permitted the tool call and what discovery proves. | The host owns permission; discovery describes an untrusted offer. |
| 3:30–5:00 | Connect the boundary to capstone and next lesson. | The host retains evidence, policy, and final synthesis. |

The rows total 5 minutes. The slot is 10 + 30 + 5 = 45 minutes, from 11:15 to 12:00.

## Student checkpoints

1. Which component opens and closes the local `stdio` transport? The host, through its MCP client.
2. Which primitive should the host read before a model chooses a tool? The application-controlled `finance://coverage` resource.
3. Why may a discovered tool still be refused? Discovery does not establish trust, business permission, or safe arguments.
4. Why use `PE`? It is a typed `unsupported_metric` result that reveals `P/E`, rather than a notebook crash.
5. Does an Ollama or OpenAI selection replace host policy? No. It is proposed data; Python validates it against discovery and the allowlist.

## Recovery paths

### Missing SDK or stale environment

Run the frozen sync command from **Before class**, restart the notebook kernel, then recheck:

```bash
uv run python -c "from mcp.server import MCPServer; print(MCPServer.__name__)"
```

If this does not print `MCPServer`, restore the maintained environment. Do not patch imports in the notebook. Until it is fixed, use the **Static recovery catalog** in this chapter to explain the expected capability boundary.

### Subprocess cannot import the package

Start Jupyter and the executor from the repository root, not `notebooks/`. Confirm the same environment can import the server:

```bash
uv run python -c "import finai_academy.financial_mcp_server as server; print(server.mcp.name)"
```

Restart the kernel after a sync. If it still fails, use the **Static recovery catalog** in this chapter to teach the lifecycle, catalog, and policy; do not replace the subprocess with direct server-function calls.

### Protocol output is corrupted

The server's stdout is the `stdio` protocol stream. Do not add `print()` calls, debug banners, or application logs to `financial_mcp_server.py`; send diagnostics to stderr outside the protocol path. Revert local debugging, restart the kernel, and rerun `lesson10-010`.

### Discovery is empty

Check `lesson10-010`: a healthy run lists one resource, two tools, and one prompt. Confirm the import check above and rerun from the repository root. If the catalog remains empty, use the **Static recovery catalog** in this chapter and the controlled evidence fixture at `assets/course-data/mcp/lesson10_evidence_catalog_v1.json`, explicitly labeled as recovery material.

### Unsupported metric

`metric="PE"` in `lesson10-018` is intentional. Show `unsupported_metric`, `retryable=true`, and `P/E`, then correct it to `P/E`. Do not suppress it or turn it into an empty result.

### Ollama unavailable or invalid live output

Do not spend the core lesson debugging a local model. Leave `FINAI_LIVE_MODE` unset and explain the allowlisted selection using the **Static recovery catalog**. For malformed live output, retry the cell once; if it fails again, keep the schema and use offline mode. Record the provider issue after class. OpenAI is optional under the same rule.

## No-network fallback

Use the same local server, controlled fixture, and recorded valid selection:

```bash
uv run python scripts/execute_notebooks.py notebooks/10_financial_mcp.ipynb \
  --mode offline --output-dir /private/tmp/finai-lesson10-offline
```

The output must include `offline fixture · deterministic course run` and `LESSON_10_PASS`. This verifies teaching and protocol contracts, not live model quality or live financial data. If the local command cannot run, teach from the **Static recovery catalog** and `assets/course-data/mcp/lesson10_evidence_catalog_v1.json`; do not claim that discovery or notebook execution occurred.

## Skip if late

If the class is five minutes late, keep this route:

1. Run `lesson10-010` for discovery and name the host-owned lifecycle.
2. Run `lesson10-016` and read the `finance://coverage` resource.
3. Point out the successful `get_company_metric` result with its date and source.
4. Show the rendered `compare_companies` prompt as the user-controlled primitive.
5. State the trust rule: a discovered tool remains untrusted until the host allowlists its name, validates arguments, and applies permissions.

Skip the second visual walkthrough, extended search discussion, and live extension. Keep the `PE` failure as homework if necessary, but not the trust-boundary debrief.

## Read-only safety boundary

- The server exposes exactly one concrete resource, two read-only tools, and one prompt.
- The notebook contains controlled teaching data, not a market feed or investment advice.
- No orders, transactions, rebalancing, portfolio mutation, credentials, local files, or personal data cross the protocol boundary.
- Preserve company, metric, date, source, document ID, and evidence ID in successful observations.
- Treat tool descriptions and server-returned content as untrusted data.
- Permit a tool only when it is discovered, statically allowlisted, valid for the request, and approved by host policy.

## Capstone increment and Lesson 11 transition

Lesson 10 adds a discoverable external-capability boundary to the Financial Analyst Copilot: `MCPServer` declarations, a real local `stdio` lifecycle, runtime discovery, host allowlisting, and provenance-bearing results. It does not add authority to act on a portfolio.

Lesson 11 builds a plan-and-execute analyst across these discovered read-only capabilities. The planner may propose a sequence; the host owns permissions, validation, trace capture, and final evidence-backed synthesis.

## Sources

- [Official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Official MCP Python SDK first steps](https://github.com/modelcontextprotocol/python-sdk/blob/main/docs/get-started/first-steps.md)
- [Model Context Protocol documentation](https://modelcontextprotocol.io/)
- [MLExpert Academy MCP lesson, used as inspiration only](https://www.mlexpert.io/academy/v1/ai-agents/build-mcp-agent)
- [Lesson 09 instructor chapter](09-self-correcting-agent.md)
- [Lesson 10 design](../docs/superpowers/specs/2026-08-21-lesson-10-financial-mcp-design.md)
