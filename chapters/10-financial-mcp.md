# Lesson 10 — Connect Financial Tools with MCP

**First Finance - Arnaud Demes**
**Day 2 · 11:15–12:00 · 10-minute concept deck + 30-minute notebook + 5-minute verification and debrief**

## Instructor outcome

Students connect a financial application to one real local MCP server over `stdio`, discover the server's declared resources, tools, and prompts, and call read-only capabilities inside a host-owned trust boundary.

The full Lesson 10 route is ready for an instructor-led test class. Use the **eleven-slide deck** to explain the boundary, then let the notebook prove it:

```text
host starts server → client lists capabilities → host inspects schemas
→ host applies policy → client reads or calls → host records evidence
```

The controlled NVIDIA (`NVDA`) and Schneider Electric (`SU.PA`) fixtures keep classroom output reproducible. They are derived from tracked course documents, not live market feeds, trading systems, or investment recommendations.

## The lesson in one sentence

**MCP standardizes how capabilities are discovered and invoked; it does not decide whether they are trusted or permitted.**

## Why MCP exists

A direct Python import couples the host to one language, module path, implementation, and release cycle. MCP moves the boundary to a declared protocol contract:

- the **host** owns the user experience, model access, lifecycle, permission, and final answer;
- the **client** carries MCP requests and responses for one server connection;
- the **server** declares focused resources, tools, and prompts;
- the **transport** carries protocol messages locally or over a network.

The benefit is interoperability, not automatic safety. A discovered capability remains untrusted input until the host applies its own policy.

## Current SDK and protocol note

The course pins the official MCP Python SDK v2 line with `mcp[cli]>=2,<3`. In v2, the high-level server class is:

```python
from mcp.server import MCPServer
```

Older v1 tutorials use `FastMCP`. The familiar decorators remain:

```python
@server.resource("finance://coverage")
@server.tool()
@server.prompt()
```

The current MCP `2026-07-28` specification has a stateless protocol core and a discovery RPC. The course SDK handles those protocol details. Students focus on the durable application boundary: discover names and schemas, apply host policy, call, and preserve evidence.

## Resources, tools, and prompts have different controllers

| Primitive | Typical controller | Lesson capability | Purpose |
|---|---|---|---|
| Resource | Application | `finance://coverage` | Read the controlled data boundary before choosing a tool. |
| Tool | Model + host approval | `get_company_metric` | Return one dated, source-bearing metric. |
| Tool | Model + host approval | `search_financial_documents` | Search versioned evidence and retain document and evidence IDs. |
| Prompt | User | `compare_companies` | Render a reusable comparison request for review. |

The expected catalog is exactly one resource, two tools, and one prompt.

## Discovery is not permission

The host must still enforce five controls:

1. **Approval** — require user confirmation for consequential actions.
2. **Authentication** — establish the identity allowed to access a remote server.
3. **Allowlist** — permit only capabilities relevant to the application.
4. **Argument validation** — validate names and arguments against discovered schemas and business rules.
5. **Evidence and audit** — retain provenance, results, refusals, and policy decisions.

Tool descriptions and server-returned content are data, not trusted instructions. A malicious or irrelevant server can still advertise a tool.

## Local stdio and remote MCP

| Boundary | Local `stdio` | Remote MCP |
|---|---|---|
| Entry point | Host starts a subprocess | Client or OpenAI receives a `server_url` |
| Location | Same machine | Network service |
| Transport concern | Process lifecycle and clean protocol streams | TLS, availability, latency, rate limits, and deployment |
| Identity | Local process identity is not authentication | OAuth or another supported authorization design may be required |
| Permission | Host policy | Approval policy plus host and service authorization |

`stdio` is a local transport, not authentication. Streamable HTTP is the production extension discussed in class; it requires a separate authentication, authorization, approval, monitoring, and audit design.

OpenAI's Responses API can use connectors and remote MCP servers through the `mcp` tool type. A remote server provides a `server_url`; the developer chooses whether tool calls require approval. This is an extension of the lesson's local boundary, not a replacement for the host policy.

## Before class

Run from the repository root:

```bash
uv sync --frozen --extra ai --extra rag --extra finance --extra evaluation --extra dev
uv run python -c "from mcp.server import MCPServer; print(MCPServer.__name__)"
uv run python scripts/build_lesson10_notebook.py
uv run python scripts/validate_notebooks.py notebooks/10_financial_mcp.ipynb
uv run jupyter lab
```

Open `notebooks/10_financial_mcp.ipynb` and run it from the repository root.

## Live model extension

The core lesson requires neither network access nor a model. Live mode lets a model propose one selection from the discovered tool catalog; Python still enforces discovery and the allowlist.

For OpenAI, keep the key in the project `.env`. The maintained default is `gpt-5.6-luna`:

```bash
uv run python scripts/execute_notebooks.py \
  notebooks/10_financial_mcp.ipynb \
  --mode live \
  --provider openai \
  --output-dir /private/tmp/finai-lesson10-openai
```

For Ollama:

```bash
ollama pull qwen3:8b
uv run python scripts/execute_notebooks.py \
  notebooks/10_financial_mcp.ipynb \
  --mode live \
  --provider ollama \
  --output-dir /private/tmp/finai-lesson10-ollama
```

Never print, trace, or commit an API key.

## 10-minute concept deck

The certified eleven-slide `decks/10-financial-mcp.pptx` deck follows one cumulative argument:

| Time | Slide | Teaching job |
|---:|---:|---|
| 0:00–0:45 | 1 | State the outcome: connect financial tools through a discoverable boundary. |
| 0:45–1:40 | 2 | Explain why the host should discover capabilities instead of importing implementation. |
| 1:40–2:35 | 3 | Identify host, client, server, and the capability boundary. |
| 2:35–3:30 | 4 | Separate resources, tools, and prompts by controller. |
| 3:30–4:30 | 5 | Use an official MCP visual to show the real ecosystem and development surface. |
| 4:30–5:30 | 6 | Follow `list → inspect schema → apply policy → call → record`. |
| 5:30–6:30 | 7 | Show the real financial catalog and provenance-bearing call. |
| 6:30–7:30 | 8 | State the security rule: discovery returns schemas, not trust. |
| 7:30–8:30 | 9 | Compare local `stdio` with OpenAI connectors and remote MCP servers. |
| 8:30–9:15 | 10 | Quiz. |
| 9:15–10:00 | 11 | Correct the quiz and transition to Lesson 11. |

## 30-minute notebook route

| Time | Cells | Visible result |
|---:|---|---|
| 0:00–4:00 | `lesson10-000`–`lesson10-002` | Learning contract, runtime, local `stdio`, read-only boundary. |
| 4:00–8:00 | `lesson10-003`–`lesson10-004` | Real `MCPServer` declarations and direct import versus MCP visual. |
| 8:00–15:00 | `lesson10-005`–`lesson10-006` | Visible `Client`, `list_tools`, `list_resources`, `list_prompts`, `read_resource`, `call_tool`, and `get_prompt` code. |
| 15:00–19:00 | `lesson10-007` | Discovered names, schemas, and control model. |
| 19:00–22:00 | `lesson10-008`–`lesson10-009` | Coverage, metric, evidence hits, and rendered prompt with provenance. |
| 22:00–26:00 | `lesson10-010`–`lesson10-011` | Allowlist refusal, typed `unsupported_metric`, protocol trace, local versus remote boundary. |
| 26:00–28:00 | `lesson10-012`–`lesson10-013` | Optional Ollama or OpenAI `gpt-5.6-luna` tool selection. |
| 28:00–30:00 | `lesson10-014`–`lesson10-015` | `LESSON_10_PASS`, knowledge check, and Lesson 11 handoff. |

## Static recovery catalog

Use this only if the local server cannot start. It is a teaching fallback, not evidence that discovery occurred.

| Primitive | Name |
|---|---|
| Resource | `finance://coverage` |
| Tool | `get_company_metric` |
| Tool | `search_financial_documents` |
| Prompt | `compare_companies` |

The controlled fallback fixture is `assets/course-data/mcp/lesson10_evidence_catalog_v1.json`.

## No-network fallback

```bash
uv run python scripts/execute_notebooks.py \
  notebooks/10_financial_mcp.ipynb \
  --mode offline \
  --output-dir /private/tmp/finai-lesson10-offline
```

The output must include:

```text
catalog=1 resource | 2 tools | 1 prompt
allowlist_refusal=blocked
LESSON_10_PASS
```

This proves the local protocol and teaching contract. It does not prove live model quality or live financial data.

## Recovery paths

### Missing SDK or stale environment

Run the frozen sync command from **Before class**, restart the kernel, and verify that `MCPServer` imports. Do not switch the notebook back to `FastMCP`.

### Subprocess import fails

Start Jupyter and the notebook executor from the repository root. Confirm:

```bash
uv run python -c "import finai_academy.financial_mcp_server as server; print(server.mcp.name)"
```

### Protocol output is corrupted

The server's stdout carries MCP messages. Do not add debug `print()` calls to `financial_mcp_server.py`; send diagnostics to stderr.

### Discovery is empty

Verify that the server module imports, restart the kernel, and rerun `lesson10-006`. If necessary, teach from the **Static recovery catalog** and label it explicitly as fallback material.

### Unsupported metric

`metric="PE"` is intentional. The expected typed result is `unsupported_metric`, with `P/E` visible and `retryable=true`.

### Live provider is unavailable

Do not debug a provider during the core lesson. Leave live mode disabled and use the deterministic offline selection. OpenAI and Ollama are optional extensions.

## Skip if late

Keep four moments:

1. Show the four `MCPServer` declarations in `lesson10-003`.
2. Run the real discovery and calls in `lesson10-006`.
3. Run the refusal and typed error in `lesson10-011`.
4. State the rule: discovery returns schemas, not trust.

Skip the extended evidence tables and live provider comparison.

## Capstone increment and Lesson 11 transition

Lesson 10 adds a discoverable read-only capability boundary to the Financial Analyst Copilot. The server exposes evidence; the host owns permission, validation, trace capture, and final synthesis.

Lesson 11 plans across these discovered capabilities. A planner may propose a sequence, but it never inherits authority merely because a tool was discovered.

## Sources

- [Official MCP Python SDK v2](https://github.com/modelcontextprotocol/python-sdk/blob/main/docs/index.md)
- [Official MCP Python SDK first steps](https://github.com/modelcontextprotocol/python-sdk/blob/main/docs/get-started/first-steps.md)
- [MCP 2026-07-28 specification release](https://blog.modelcontextprotocol.io/posts/2026-07-28/)
- [Model Context Protocol primitives](https://modelcontextprotocol.io/specification/2026-07-28/server)
- [OpenAI MCP and Connectors](https://developers.openai.com/api/docs/guides/tools-connectors-mcp)
- [OpenAI GPT-5.6 Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna)
- [Lesson 09 instructor chapter](09-self-correcting-agent.md)
