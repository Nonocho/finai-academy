# Lesson 10 Financial MCP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and certify a 45-minute Lesson 10 in which a local client discovers and uses read-only NVIDIA and Schneider Electric capabilities from an MCP server over `stdio`.

**Architecture:** Keep financial validation and evidence search in one pure Python registry, wrap it with the official MCP Python SDK v2 `MCPServer`, and access it only through a small async client. Tests use the SDK's in-memory client for fast protocol checks and a real subprocess for one `stdio` conformance run; the notebook visualizes the same client trace and adds an optional provider-neutral tool-selection cell.

**Tech Stack:** Python 3.11+, Pydantic 2, MCP Python SDK v2 (`mcp[cli]>=2,<3`), `uv`, Jupyter/nbclient, Matplotlib, Pandas, Ollama or OpenAI through the existing model gateway, pytest, Ruff, PowerPoint via `@oai/artifact-tool`.

**Spec:** `docs/superpowers/specs/2026-08-21-lesson-10-financial-mcp-design.md`

## Global Constraints

- Preserve the canonical Day 2 slot: 11:15-12:00.
- Core format: 10-minute deck, 30-minute notebook, 5-minute verification and debrief.
- Expose exactly one concrete resource, two read-only tools, and one prompt.
- Use `MCPServer`, not the v1 `FastMCP` import; explain the rename once in teaching material.
- Use local `stdio` in the classroom core and name Streamable HTTP only as a production extension.
- Do not add portfolio mutation, trading, SQLite, authentication implementation, deployment, or multi-server orchestration.
- Reuse NVIDIA (`NVDA`) and Schneider Electric (`SU.PA`) and retain source, date, document, and evidence identifiers.
- Keep protocol functionality independent of an LLM; live selection must work with Ollama or OpenAI through the shared gateway.
- Keep the checked-in notebook output-free, deterministic offline, and free of secrets.
- Use short, factual English in slides; do not use em dashes.
- Use the footer `First Finance - Arnaud Demes` on every slide.
- Add a `[Sources]` block containing source lines and a closing `[/Sources]` tag to every slide's speaker notes.
- Do not reproduce MLExpert Academy code or lesson copy.

## File Map

| File | Responsibility |
| --- | --- |
| `assets/course-data/mcp/lesson10_evidence_catalog_v1.json` | Versioned NVIDIA and Schneider document passages used by the MCP search tool. |
| `assets/course-data/manifest.json` | Provenance and checksum entry for the new catalog. |
| `src/finai_academy/financial_mcp_capabilities.py` | Pure Pydantic contracts, validation, metric lookup, coverage resource data, and deterministic document search. |
| `src/finai_academy/financial_mcp_server.py` | `MCPServer` registrations and executable `stdio` entry point. |
| `src/finai_academy/financial_mcp_client.py` | Capability discovery, protocol calls, allowlisting, content parsing, and trace records. |
| `tests/test_financial_mcp_capabilities.py` | Pure registry contract and error tests. |
| `tests/test_financial_mcp_server.py` | In-memory MCP discovery, resource, tool, prompt, and error tests. |
| `tests/test_financial_mcp_client.py` | Real `stdio` subprocess and allowlist tests. |
| `scripts/build_lesson10_notebook.py` | Deterministic notebook source generator. |
| `notebooks/10_financial_mcp.ipynb` | Student-facing 30-minute guided implementation. |
| `chapters/10-financial-mcp.md` | Instructor timing, expected outputs, recovery path, and debrief. |
| `decks/10-financial-mcp.pptx` | Nine-slide concept deck. |
| `tests/test_lesson10_assets.py` | Notebook, chapter, deck, index, and execution contract. |
| `docs/reviews/lesson-10-readiness.md` | Evidence-based delivery score and provider status. |

---

### Task 1: Add the MCP dependency and pure financial capability registry

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `assets/course-data/mcp/lesson10_evidence_catalog_v1.json`
- Modify: `assets/course-data/manifest.json`
- Create: `src/finai_academy/financial_mcp_capabilities.py`
- Create: `tests/test_financial_mcp_capabilities.py`

**Interfaces:**
- Consumes: `MetricRegistry`, `MetricRequest`, and `build_metric_registry` from `finai_academy.self_correcting_agent`; `assets/course-data/market/lesson09_metrics_snapshot_v1.json`.
- Produces: `CapabilityError`, `CapabilityValidationError`, `CoverageSnapshot`, `MetricResult`, `DocumentHit`, `DocumentSearchResult`, `FinancialCapabilityRegistry`, and `build_financial_capability_registry()`.

- [ ] **Step 1: Write failing pure-capability tests**

Create tests with these exact behaviors:

```python
def test_coverage_exposes_only_the_two_course_companies(registry):
    coverage = registry.coverage()
    assert coverage.dataset_id == "lesson10-financial-mcp-v1"
    assert {item.ticker for item in coverage.companies} == {"NVDA", "SU.PA"}
    assert coverage.supported_metrics == ("EPS", "P/E")


def test_metric_result_preserves_provenance(registry):
    result = registry.get_company_metric("NVDA", "P/E")
    assert result.status == "ok"
    assert result.value == 52.4
    assert result.as_of == "2026-08-20"
    assert result.source


def test_invalid_metric_is_a_typed_retryable_error(registry):
    with pytest.raises(CapabilityValidationError) as caught:
        registry.get_company_metric("NVDA", "PE")
    assert caught.value.error.error_code == "unsupported_metric"
    assert caught.value.error.rejected_value == "PE"
    assert "P/E" in caught.value.error.valid_values
    assert caught.value.error.retryable is True


def test_document_search_filters_company_and_keeps_evidence_ids(registry):
    result = registry.search_financial_documents("Schneider Electric", "energy management", 2)
    assert result.status == "ok"
    assert result.company == "Schneider Electric"
    assert 1 <= len(result.hits) <= 2
    assert all(hit.evidence_id.startswith("SU-") for hit in result.hits)
    assert all(hit.source for hit in result.hits)


@pytest.mark.parametrize("top_k", [0, 4])
def test_document_search_rejects_out_of_range_top_k(registry, top_k):
    with pytest.raises(CapabilityValidationError) as caught:
        registry.search_financial_documents("NVIDIA", "data center", top_k)
    assert caught.value.error.error_code == "invalid_top_k"
```

- [ ] **Step 2: Run the tests and confirm RED**

Run:

```bash
.venv/bin/pytest -q tests/test_financial_mcp_capabilities.py
```

Expected: collection fails because `finai_academy.financial_mcp_capabilities` does not exist.

- [ ] **Step 3: Add the locked SDK dependency**

Add this entry to `[project].dependencies` in `pyproject.toml`:

```toml
"mcp[cli]>=2,<3",
```

Then run:

```bash
uv lock
uv sync --extra ai --extra dev
```

Expected: `uv.lock` resolves an MCP 2.x release and the existing `.venv` can import `MCPServer` and `Client`.

- [ ] **Step 4: Create the evidence catalog and manifest entry**

Create `lesson10_evidence_catalog_v1.json` with this schema and at least two passages per company:

```json
{
  "dataset_id": "lesson10-financial-mcp-v1",
  "as_of": "2026-08-20",
  "notice": "Controlled classroom evidence derived from tracked course fixtures; not live market data.",
  "documents": [
    {
      "evidence_id": "NVDA-FY2026-DATA-CENTER-001",
      "company": "NVIDIA",
      "ticker": "NVDA",
      "document_id": "NVDA-FY2026-EXCERPT",
      "period": "FY2026",
      "section": "Revenue",
      "text": "NVIDIA reported fiscal 2026 total revenue of $215.9 billion, including $193.7 billion from Data Center.",
      "source": "assets/course-data/fixtures/nvidia_fy2026_excerpt.html"
    },
    {
      "evidence_id": "SU-FY2025-ENERGY-MANAGEMENT-001",
      "company": "Schneider Electric",
      "ticker": "SU.PA",
      "document_id": "SU-FY2025-EXCERPT",
      "period": "FY2025",
      "section": "Energy Management",
      "text": "Schneider Electric reported FY2025 revenue of EUR 40.2 billion and an adjusted EBITA margin of 18.7%.",
      "source": "assets/course-data/fixtures/schneider_fy2025_excerpt.pdf"
    }
  ]
}
```

Add two more grounded passages: NVIDIA Gaming revenue grew 41% in fiscal 2026, and Schneider Electric's FY2025 extract reports the Energy Management business context. Keep every passage a short factual paraphrase and add the file checksum, source paths, date, and controlled-fixture notice to `assets/course-data/manifest.json`.

- [ ] **Step 5: Implement the minimal pure registry**

Use these public contracts:

```python
class CapabilityError(BaseModel):
    status: Literal["error"] = "error"
    error_code: str
    message: str
    rejected_value: str | int | None = None
    valid_values: tuple[str, ...] = ()
    retryable: bool = False


class CapabilityValidationError(ValueError):
    def __init__(self, error: CapabilityError) -> None:
        self.error = error
        super().__init__(error.model_dump_json())


def build_financial_capability_registry(
    *,
    metric_snapshot_path: Path | None = None,
    evidence_catalog_path: Path | None = None,
) -> FinancialCapabilityRegistry:
    metric_payload = json.loads((metric_snapshot_path or DEFAULT_METRICS).read_text())
    evidence_payload = json.loads((evidence_catalog_path or DEFAULT_EVIDENCE).read_text())
    return FinancialCapabilityRegistry(metric_payload, evidence_payload)
```

`FinancialCapabilityRegistry` exposes exactly these methods:

- `coverage(self) -> CoverageSnapshot`
- `get_company_metric(self, ticker: str, metric: str) -> MetricResult`
- `search_financial_documents(self, company: str, query: str, top_k: int = 2) -> DocumentSearchResult`

Normalize company aliases to exactly `NVIDIA` and `Schneider Electric`. Rank evidence with deterministic token overlap, then break score ties by `evidence_id`. Reject blank queries, unknown companies, and `top_k` values outside `1..3` using `CapabilityValidationError`.

- [ ] **Step 6: Run focused tests and lint**

Run:

```bash
.venv/bin/pytest -q tests/test_financial_mcp_capabilities.py
.venv/bin/ruff check src/finai_academy/financial_mcp_capabilities.py tests/test_financial_mcp_capabilities.py
```

Expected: all capability tests pass and Ruff reports no issues.

- [ ] **Step 7: Commit the independently usable registry**

```bash
git add pyproject.toml uv.lock assets/course-data/manifest.json assets/course-data/mcp/lesson10_evidence_catalog_v1.json src/finai_academy/financial_mcp_capabilities.py tests/test_financial_mcp_capabilities.py
git commit -m "feat: add financial MCP capability registry"
```

---

### Task 2: Expose the registry through an MCPServer

**Files:**
- Create: `src/finai_academy/financial_mcp_server.py`
- Create: `tests/test_financial_mcp_server.py`

**Interfaces:**
- Consumes: `FinancialCapabilityRegistry` and `build_financial_capability_registry()` from Task 1.
- Produces: `build_financial_mcp_server(registry: FinancialCapabilityRegistry | None = None) -> MCPServer`, module-level `mcp`, and `main() -> None`.

- [ ] **Step 1: Write failing in-memory protocol tests**

Use the official v2 `Client(server, raise_exceptions=True)` test harness. Assert:

```python
async with Client(server, raise_exceptions=True) as client:
    tools = await client.list_tools()
    resources = await client.list_resources()
    prompts = await client.list_prompts()
    assert [item.name for item in tools.tools] == [
        "get_company_metric",
        "search_financial_documents",
    ]
    assert [str(item.uri) for item in resources.resources] == ["finance://coverage"]
    assert [item.name for item in prompts.prompts] == ["compare_companies"]
```

Add separate tests that:

- read `finance://coverage` and parse its JSON;
- call `get_company_metric` successfully and inspect `structured_content`;
- call `search_financial_documents` and retain `evidence_id`;
- get the `compare_companies` prompt and assert one user message;
- call `get_company_metric` with `PE` and assert `result.is_error is True`, `unsupported_metric` in its text content, and `P/E` in its text content.

- [ ] **Step 2: Run the server tests and confirm RED**

```bash
.venv/bin/pytest -q tests/test_financial_mcp_server.py
```

Expected: collection fails because `financial_mcp_server.py` does not exist.

- [ ] **Step 3: Implement the MCP adapter**

Build the server with the current SDK:

```python
from mcp.server import MCPServer


def build_financial_mcp_server(
    registry: FinancialCapabilityRegistry | None = None,
) -> MCPServer:
    active_registry = registry or build_financial_capability_registry()
    server = MCPServer(
        "First Finance Research",
        instructions=(
            "Read-only financial research capabilities for NVIDIA and "
            "Schneider Electric. Preserve dates, sources, and evidence IDs."
        ),
    )

    @server.resource("finance://coverage", mime_type="application/json")
    def coverage() -> dict[str, object]:
        return active_registry.coverage().model_dump(mode="json")

    @server.tool()
    def get_company_metric(ticker: str, metric: str) -> dict[str, object]:
        return active_registry.get_company_metric(ticker, metric).model_dump(mode="json")

    @server.tool()
    def search_financial_documents(
        company: str, query: str, top_k: Annotated[int, Field(ge=1, le=3)] = 2
    ) -> dict[str, object]:
        return active_registry.search_financial_documents(
            company, query, top_k
        ).model_dump(mode="json")

    @server.prompt()
    def compare_companies(metric: str, question: str) -> str:
        return (
            f"Compare NVIDIA and Schneider Electric using the metric {metric}.\n"
            f"Question: {question}\n"
            "Use only MCP resource and tool results. Cite every evidence ID, date, "
            "and source. State missing evidence. Do not make an investment recommendation."
        )

    return server
```

For validation failures, let `CapabilityValidationError` escape the tool body so the SDK converts it into `is_error=True` content for the model. Keep resource and prompt content free of local paths and credentials.

At module level:

```python
mcp = build_financial_mcp_server()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
```

Do not print to stdout before or during `mcp.run()` because stdout is the protocol stream.

- [ ] **Step 4: Run protocol tests and inspect schemas**

```bash
.venv/bin/pytest -q tests/test_financial_mcp_server.py
.venv/bin/python -c "from finai_academy.financial_mcp_server import mcp; print(mcp.name)"
```

Expected: protocol tests pass; the import check prints `First Finance Research`; no duplicate-capability warning appears.

- [ ] **Step 5: Commit the server adapter**

```bash
git add src/finai_academy/financial_mcp_server.py tests/test_financial_mcp_server.py
git commit -m "feat: expose financial capabilities through MCP"
```

---

### Task 3: Build the allowlisted stdio client and visible trace

**Files:**
- Create: `src/finai_academy/financial_mcp_client.py`
- Create: `tests/test_financial_mcp_client.py`

**Interfaces:**
- Consumes: the module entry point `python -m finai_academy.financial_mcp_server` from Task 2.
- Produces: `DiscoveredCapability`, `McpOperationEvent`, `FinancialMcpRun`, `discover_and_run_financial_mcp()`, and `call_allowlisted_tool()`.

- [ ] **Step 1: Write failing client tests**

Assert these exact outcomes from a real subprocess:

```python
run = asyncio.run(discover_and_run_financial_mcp())
assert run.server_name == "First Finance Research"
assert run.resource_names == ("finance://coverage",)
assert run.tool_names == ("get_company_metric", "search_financial_documents")
assert run.prompt_names == ("compare_companies",)
assert run.coverage.dataset_id == "lesson10-financial-mcp-v1"
assert run.metric.status == "ok"
assert run.search.hits
assert run.rendered_prompt
assert run.failure.error_code == "unsupported_metric"
assert [event.sequence for event in run.trace] == list(range(1, len(run.trace) + 1))
```

Add an allowlist test:

```python
with pytest.raises(ValueError, match="not allowlisted"):
    asyncio.run(call_allowlisted_tool("delete_portfolio", {}))
```

- [ ] **Step 2: Run the client tests and confirm RED**

```bash
.venv/bin/pytest -q tests/test_financial_mcp_client.py
```

Expected: collection fails because `financial_mcp_client.py` does not exist.

- [ ] **Step 3: Implement the stdio lifecycle**

Use the v2 client API:

```python
from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client


ALLOWED_TOOLS = frozenset({"get_company_metric", "search_financial_documents"})


def financial_stdio_transport():
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "finai_academy.financial_mcp_server"],
    )
    return stdio_client(parameters)


async def discover_and_run_financial_mcp() -> FinancialMcpRun:
    async with Client(financial_stdio_transport()) as client:
        tools = await client.list_tools()
        resources = await client.list_resources()
        prompts = await client.list_prompts()
        coverage = await client.read_resource("finance://coverage")
        metric = await client.call_tool(
            "get_company_metric", {"ticker": "NVDA", "metric": "P/E"}
        )
        search = await client.call_tool(
            "search_financial_documents",
            {"company": "Schneider Electric", "query": "energy management", "top_k": 2},
        )
        prompt = await client.get_prompt(
            "compare_companies",
            {"metric": "P/E", "question": "Compare valuation and operating evidence."},
        )
        failure = await client.call_tool(
            "get_company_metric", {"ticker": "NVDA", "metric": "PE"}
        )
```

Parse text and structured content with explicit `isinstance(block, TextContent)` checks. Convert every operation to a sequential `McpOperationEvent` containing duration, status, evidence count, and error code. Do not expose subprocess environment values in the trace.

`call_allowlisted_tool(name, arguments)` must reject a name unless it is both in `ALLOWED_TOOLS` and present in the server's discovered tools before calling it.

- [ ] **Step 4: Run client, server, and capability tests**

```bash
.venv/bin/pytest -q tests/test_financial_mcp_capabilities.py tests/test_financial_mcp_server.py tests/test_financial_mcp_client.py
.venv/bin/ruff check src/finai_academy/financial_mcp_capabilities.py src/finai_academy/financial_mcp_server.py src/finai_academy/financial_mcp_client.py tests/test_financial_mcp_capabilities.py tests/test_financial_mcp_server.py tests/test_financial_mcp_client.py
```

Expected: all focused tests pass, the subprocess exits cleanly, and Ruff passes.

- [ ] **Step 5: Commit the client boundary**

```bash
git add src/finai_academy/financial_mcp_client.py tests/test_financial_mcp_client.py
git commit -m "feat: add discoverable financial MCP client"
```

---

### Task 4: Build and execute the visual Lesson 10 notebook

**Files:**
- Create: `tests/test_lesson10_assets.py`
- Create: `scripts/build_lesson10_notebook.py`
- Create: `notebooks/10_financial_mcp.ipynb`
- Modify: `tests/test_course_manifest.py`

**Interfaces:**
- Consumes: `discover_and_run_financial_mcp()`, `call_allowlisted_tool()`, the existing `Settings`, and `create_chat_model()`.
- Produces: a 22-to-26-cell output-free notebook with at least five PNG visuals and exact marker `LESSON_10_PASS`.

- [ ] **Step 1: Write failing notebook asset tests**

Require:

```python
assert notebook.metadata["finai"]["expected_runtime_minutes"] == 30
assert all(not cell.get("outputs") for cell in notebook.cells if cell.cell_type == "code")
assert len({cell.id for cell in notebook.cells}) == len(notebook.cells)
```

Require these headings and markers in source:

```text
## Learning objectives
## Where this fits
## Failure lab
## Verification
## Challenge
## Capstone integration
## Recap
MCPServer
finance://coverage
get_company_metric
search_financial_documents
compare_companies
stdio
FINAI_LIVE_MODE
create_chat_model
Ollama
OpenAI
LESSON_10_PASS
```

The offline execution test must run `scripts/execute_notebooks.py`, require at least five PNG outputs, and find `LESSON_10_PASS` in stream output.

- [ ] **Step 2: Run the notebook tests and confirm RED**

```bash
.venv/bin/pytest -q tests/test_lesson10_assets.py -k notebook
```

Expected: failure because the notebook and builder are absent.

- [ ] **Step 3: Implement the notebook builder**

Follow the established stable-cell-ID pattern from `scripts/build_lesson09_notebook.py`. Build this exact learning sequence:

1. title, time, prerequisite, and run instructions;
2. objectives and Lesson 09 direct-import handoff;
3. provider-neutral setup with deterministic offline mode;
4. Figure 1: direct import versus host-client-server architecture;
5. a concise source view of the pure registry and four `MCPServer` decorators;
6. Figure 2: resource/tool/prompt control matrix;
7. execute `mcp_run = await discover_and_run_financial_mcp()`;
8. Figure 3: `stdio` discovery and call sequence;
9. display the discovered resource, tool, and prompt catalog;
10. Figure 4: capability catalog by primitive and controller;
11. display coverage, metric result, document hits, and rendered prompt;
12. display the maintained `PE` failure without raising in the notebook;
13. Figure 5: successful calls versus typed error trace;
14. optional live structured selection through Ollama or OpenAI;
15. verification assertions and `LESSON_10_PASS`;
16. knowledge check, challenge, capstone integration, and recap.

In offline mode, use a recorded valid selection. In live mode, define:

```python
class DiscoveredToolChoice(BaseModel):
    tool_name: str
    arguments: dict[str, str | int | float]
    reason: str
```

Pass only the discovered tool names, descriptions, and input schemas to `create_chat_model(settings).with_structured_output(DiscoveredToolChoice)`. Validate the returned name through `call_allowlisted_tool()` before execution.

- [ ] **Step 4: Generate, validate, and execute offline**

```bash
.venv/bin/python scripts/build_lesson10_notebook.py
.venv/bin/python scripts/validate_notebooks.py notebooks/10_financial_mcp.ipynb
.venv/bin/python scripts/execute_notebooks.py notebooks/10_financial_mcp.ipynb --mode offline --output-dir /private/tmp/finai-lesson10-offline
```

Expected: notebook source validation passes; execution produces at least five PNGs and `LESSON_10_PASS`; the checked-in notebook remains output-free.

- [ ] **Step 5: Run notebook and protocol regression tests**

```bash
.venv/bin/pytest -q tests/test_lesson10_assets.py -k notebook tests/test_financial_mcp_capabilities.py tests/test_financial_mcp_server.py tests/test_financial_mcp_client.py
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit the notebook increment**

```bash
git add scripts/build_lesson10_notebook.py notebooks/10_financial_mcp.ipynb tests/test_lesson10_assets.py tests/test_course_manifest.py
git commit -m "lesson: add visual financial MCP notebook"
```

---

### Task 5: Write the instructor chapter and expose Lesson 10 in indexes

**Files:**
- Create: `chapters/10-financial-mcp.md`
- Modify: `chapters/README.md`
- Modify: `notebooks/README.md`
- Modify: `decks/README.md`
- Modify: `README.md` if the Day 2 quick-start table requires the new direct links.
- Modify: `tests/test_lesson10_assets.py`

**Interfaces:**
- Consumes: the exact notebook cell order and output contracts from Task 4.
- Produces: the instructor's complete 11:15-12:00 delivery route and discoverable course links.

- [ ] **Step 1: Extend the failing chapter/index tests**

Require the chapter to contain:

```text
11:15–12:00
10-minute concept deck
30-minute notebook
finance://coverage
get_company_metric
search_financial_documents
compare_companies
MCPServer
FastMCP
stdio
Streamable HTTP
Ollama
OpenAI
No-network fallback
Skip if late
Lesson 11
```

Require exact Lesson 10 paths in the notebook, chapter, and deck indexes.

- [ ] **Step 2: Run the chapter/index tests and confirm RED**

```bash
.venv/bin/pytest -q tests/test_lesson10_assets.py -k "chapter or discoverable"
```

Expected: failure because the chapter and links are absent.

- [ ] **Step 3: Write the complete instructor chapter**

Include:

- a one-paragraph lesson purpose;
- prerequisites and exact startup commands;
- a minute-by-minute 10-minute slide script;
- a minute-by-minute 30-minute notebook script with expected visible output;
- the five-minute verification and debrief;
- the v1 `FastMCP` to v2 `MCPServer` note;
- common failures: missing SDK, subprocess cannot import package, stdout protocol corruption, empty discovery list, unsupported metric, local Ollama unavailable;
- no-network fallback using the same local server and recorded LLM selection;
- skip-if-late route: discovery, coverage resource, one successful tool, one prompt, trust boundary;
- student checkpoint questions;
- read-only safety boundary;
- capstone increment and Lesson 11 transition;
- source links to the official SDK, protocol docs, MLExpert inspiration, and course modules.

- [ ] **Step 4: Update all indexes and run tests**

```bash
.venv/bin/pytest -q tests/test_lesson10_assets.py -k "chapter or discoverable"
.venv/bin/ruff check .
```

Expected: chapter/index tests pass and Ruff passes.

- [ ] **Step 5: Commit the instructor materials**

```bash
git add chapters/10-financial-mcp.md chapters/README.md notebooks/README.md decks/README.md README.md tests/test_lesson10_assets.py
git commit -m "docs: add lesson 10 instructor route"
```

---

### Task 6: Create and visually certify the nine-slide deck

**Files:**
- Create: `decks/10-financial-mcp.pptx`
- Modify: `tests/test_lesson10_assets.py`
- Use ignored QA workspace: `.artifacts/lesson10-deck/`

**Interfaces:**
- Consumes: Lesson 10 chapter language and the visual system from `decks/09-self-correcting-agent.pptx`.
- Produces: exactly nine sourced slides with the required footer and no text overflow.

- [ ] **Step 1: Add failing deck contract tests**

Require exactly nine slide XML parts and nine notes parts. Require the footer and source blocks on every slide. Require case-insensitive markers:

```text
Build a Financial MCP
Direct import
HOST
CLIENT
SERVER
RESOURCES
TOOLS
PROMPTS
finance://coverage
DISCOVERY
stdio
DISCOVERY IS NOT PERMISSION
LESSON 11
```

Reject the em dash character in visible slide text.

- [ ] **Step 2: Run the deck test and confirm RED**

```bash
.venv/bin/pytest -q tests/test_lesson10_assets.py -k deck
```

Expected: failure because `decks/10-financial-mcp.pptx` is absent.

- [ ] **Step 3: Build the deck from the Lesson 09 template**

Use the presentation skill's template-following workflow and exact nine-slide narrative:

1. `Build a Financial MCP` - discover capabilities instead of importing functions.
2. `Direct import creates tight coupling` - application and function implementation are coupled.
3. `Host, client, server, transport` - one labeled architecture diagram.
4. `Resources, tools, prompts` - recap table with application/model/user control.
5. `One server, four capabilities` - resource URI, two tools, one prompt.
6. `Discovery happens at runtime` - list, inspect schema, select, call.
7. `stdio carries local protocol messages` - subprocess sequence diagram.
8. `Discovery is not permission` - allowlist, validate, preserve evidence, stop.
9. `Production rule` - MCP standardizes access; Lesson 11 plans across capabilities.

Every notes block must cite the course chapter plus the relevant official MCP SDK or protocol source. Use original copy and diagrams.

- [ ] **Step 4: Render and inspect every slide**

Run the presentation renderer, create a montage, inspect the montage, and inspect all nine full-size slide PNGs. Correct any collision, clipping, low contrast, awkward wrap, unexplained acronym, or inconsistent alignment.

- [ ] **Step 5: Run automated deck QA**

Run:

```bash
/Users/arnauddemes/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 /Users/arnauddemes/.codex/plugins/cache/openai-primary-runtime/presentations/26.819.11345/skills/presentations/container_tools/slides_test.py decks/10-financial-mcp.pptx
.venv/bin/pytest -q tests/test_lesson10_assets.py -k deck
```

Also run the template plan and fidelity checks against the Lesson 09 source deck. Expected: no overflow, nine slides, all markers, all notes, and zero fidelity issues.

- [ ] **Step 6: Commit the certified deck**

```bash
git add decks/10-financial-mcp.pptx tests/test_lesson10_assets.py
git commit -m "docs: add lesson 10 financial MCP deck"
```

---

### Task 7: Run live providers, full regression, and record readiness

**Files:**
- Create: `docs/reviews/lesson-10-readiness.md`
- Modify: only implementation files required to fix failures found by verification.

**Interfaces:**
- Consumes: all Lesson 10 assets and tests from Tasks 1-6.
- Produces: evidence-based readiness decision, weighted score, clean branch, and final Lesson 10 commits.

- [ ] **Step 1: Execute the complete targeted package**

```bash
.venv/bin/pytest -q tests/test_financial_mcp_capabilities.py tests/test_financial_mcp_server.py tests/test_financial_mcp_client.py tests/test_lesson10_assets.py tests/test_course_manifest.py
.venv/bin/ruff check .
.venv/bin/python scripts/validate_notebooks.py notebooks/10_financial_mcp.ipynb
```

Expected: all targeted tests, Ruff, and notebook validation pass.

- [ ] **Step 2: Run the Ollama classroom path**

Confirm `qwen3:8b` is installed, then run:

```bash
FINAI_LIVE_MODE=1 FINAI_MODEL_PROVIDER=ollama FINAI_CHAT_MODEL=qwen3:8b .venv/bin/python scripts/execute_notebooks.py notebooks/10_financial_mcp.ipynb --mode live --provider ollama --output-dir /private/tmp/finai-lesson10-ollama
```

Inspect the executed notebook and record the provider label, figure count, discovered tools, selected allowlisted tool, final marker, and elapsed time.

- [ ] **Step 3: Run OpenAI only when configured**

If `OPENAI_API_KEY` is present, run:

```bash
FINAI_LIVE_MODE=1 FINAI_MODEL_PROVIDER=openai FINAI_CHAT_MODEL=gpt-5-mini .venv/bin/python scripts/execute_notebooks.py notebooks/10_financial_mcp.ipynb --mode live --provider openai --output-dir /private/tmp/finai-lesson10-openai
```

If the key is absent, record `NOT CONFIGURED`; do not claim an OpenAI pass.

- [ ] **Step 4: Run the full course regression suite**

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
git diff --check
```

Expected: zero failures, zero lint issues, and no whitespace errors.

- [ ] **Step 5: Write the readiness report**

Record exact evidence for:

- full and targeted test counts;
- source notebook validation;
- offline figure count and final marker;
- real `stdio` subprocess result;
- in-memory MCP discovery and error result;
- Ollama and OpenAI status;
- all nine slides inspected;
- overflow and template-fidelity status;
- instructor timing and skip-if-late route;
- weighted score across clarity, notebook visuals, technical correctness, provider neutrality, deck quality, and timing.

Do not assign 10/10 unless both live providers and a real timed learner rehearsal pass.

- [ ] **Step 6: Commit readiness and any verified corrections**

```bash
git add docs/reviews/lesson-10-readiness.md
git commit -m "docs: certify lesson 10 readiness"
```

- [ ] **Step 7: Perform the post-commit completion gate**

```bash
git status --porcelain
git log -7 --oneline
.venv/bin/pytest -q tests/test_financial_mcp_capabilities.py tests/test_financial_mcp_server.py tests/test_financial_mcp_client.py tests/test_lesson10_assets.py
```

Expected: empty Git status, the Lesson 10 commit sequence is visible, and all targeted tests pass after commit.
