"""Build the compact Lesson 10 financial MCP notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/10_financial_mcp.ipynb"


def _markdown(cell_id: str, source: str):
    cell = new_markdown_cell(source.strip())
    cell["id"] = cell_id
    return cell


def _code(cell_id: str, source: str):
    cell = new_code_cell(source.strip())
    cell["id"] = cell_id
    return cell


def build_notebook():
    cells = [
        _markdown(
            "lesson10-000",
            """
# 10 — Connect financial tools with MCP

**First Finance - Arnaud Demes**
**Day 2 · 11:15–12:00 · 10 minutes concepts + 30 minutes notebook + 5 minutes debrief**

**Engineering question:** how can a financial application discover capabilities from another process without importing that process's business functions?

This lesson uses a real local MCP server over `stdio`, plus controlled NVIDIA (`NVDA`) and Schneider Electric (`SU.PA`) evidence. It is not live market data or investment advice. Offline mode is the classroom default.
""",
        ),
        _markdown(
            "lesson10-001",
            """
## Learning objectives

By the end, you can:

1. identify the **host**, **client**, **server**, and **transport**;
2. distinguish application-controlled **resources**, model-requested **tools**, and user-controlled **prompts**;
3. run the real `list → inspect → call → record` lifecycle;
4. keep sources, dates, document IDs, and evidence IDs across the protocol boundary; and
5. explain why discovery describes an offer but never grants permission.

The maintained OpenAI extension uses the project `.env` and defaults to `gpt-5.6-luna`. The core run needs no model or network.
""",
        ),
        _code(
            "lesson10-002",
            """
import json
import os
import re
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
from mcp import Client
from mcp.types import TextContent
from pydantic import BaseModel, ConfigDict

from finai_academy.financial_mcp_capabilities import (
    CapabilityError,
    CoverageSnapshot,
    DocumentSearchResult,
    MetricResult,
)
from finai_academy.financial_mcp_client import (
    ALLOWED_TOOLS,
    DiscoveredToolSpec,
    call_allowlisted_tool,
    financial_stdio_transport,
)
from finai_academy.providers import create_chat_model, provider_summary
from finai_academy.settings import Settings

LIVE_MODE = os.getenv("FINAI_LIVE_MODE", "0") == "1"
settings = Settings.from_environment()
runtime_label = provider_summary(settings) if LIVE_MODE else "offline fixture · deterministic course run"

print(f"runtime={runtime_label}")
print("transport=local stdio")
print("boundary=read-only research")
""",
        ),
        _markdown(
            "lesson10-003",
            """
## The server declares a contract

The server lives in a real Python module because `stdio` starts it as a separate process. Four decorators expose the boundary without duplicating the financial registry:

```python
server = MCPServer("First Finance Research")

@server.resource("finance://coverage")
def coverage(): ...

@server.tool()
def get_company_metric(ticker: str, metric: str): ...

@server.tool()
def search_financial_documents(company: str, query: str, top_k: int = 2): ...

@server.prompt()
def compare_companies(metric: str, question: str): ...
```

The registry still owns business validation. MCP standardizes how another application discovers and invokes that contract.
""",
        ),
        _code(
            "lesson10-004",
            """
fig, ax = plt.subplots(figsize=(12, 4.4))
ax.axis("off")

ax.text(0.20, 0.90, "DIRECT IMPORT", ha="center", weight="bold", color="#F07D00", fontsize=13)
ax.text(0.76, 0.90, "MCP BOUNDARY", ha="center", weight="bold", color="#1F40CB", fontsize=13)

for y, label in zip((0.66, 0.44, 0.22), ("Host knows module", "Imports implementation", "Calls Python function")):
    ax.text(0.20, y, label, ha="center", va="center", fontsize=11,
            bbox={"boxstyle": "round,pad=0.6", "fc": "#FFF2E5", "ec": "#F07D00"})

for x, label in zip((0.55, 0.75, 0.95), ("HOST", "CLIENT", "SERVER")):
    ax.text(x, 0.53, label, ha="center", va="center", weight="bold", fontsize=12,
            bbox={"boxstyle": "round,pad=0.8", "fc": "#EAF7FD", "ec": "#00A2EB"})
ax.annotate("", xy=(0.68, 0.53), xytext=(0.62, 0.53), arrowprops={"arrowstyle": "->", "color": "#4B6070", "lw": 2})
ax.annotate("", xy=(0.88, 0.53), xytext=(0.82, 0.53), arrowprops={"arrowstyle": "->", "color": "#4B6070", "lw": 2})
ax.text(0.75, 0.25, "discover names + schemas → apply policy → call", ha="center", color="#1F40CB", weight="bold")
ax.text(0.50, 0.05, "The host depends on a protocol contract, not the server's implementation.", ha="center", weight="bold")
ax.set_title("Figure 1. MCP replaces implementation coupling with a discoverable boundary", loc="left", weight="bold")
plt.tight_layout()
plt.show()
""",
        ),
        _markdown(
            "lesson10-005",
            """
## Run the real MCP lifecycle

The next cell shows the protocol calls directly. `Client(financial_stdio_transport())` starts the local server process, performs discovery and read-only calls, then closes the transport when the context exits.

The host checks both the static allowlist and the names discovered at runtime before each tool call.
""",
        ),
        _code(
            "lesson10-006",
            """
trace: list[dict[str, object]] = []

async with Client(financial_stdio_transport()) as client:
    server_name = client.server_info.name

    tools_result = await client.list_tools()
    resources_result = await client.list_resources()
    prompts_result = await client.list_prompts()

    tool_specs = tuple(
        DiscoveredToolSpec(
            name=tool.name,
            description=tool.description or "",
            input_schema=dict(tool.input_schema),
        )
        for tool in tools_result.tools
    )
    tool_names = tuple(tool.name for tool in tool_specs)
    resource_names = tuple(str(resource.uri) for resource in resources_result.resources)
    prompt_names = tuple(prompt.name for prompt in prompts_result.prompts)
    trace.extend([
        {"operation": "list_tools", "status": "ok", "result": len(tool_names)},
        {"operation": "list_resources", "status": "ok", "result": len(resource_names)},
        {"operation": "list_prompts", "status": "ok", "result": len(prompt_names)},
    ])

    coverage_raw = await client.read_resource("finance://coverage")
    coverage = CoverageSnapshot.model_validate_json(coverage_raw.contents[0].text)
    trace.append({"operation": "read_resource", "status": "ok", "result": "coverage"})

    for requested_tool in ("get_company_metric", "search_financial_documents"):
        assert requested_tool in ALLOWED_TOOLS and requested_tool in tool_names

    metric_raw = await client.call_tool(
        "get_company_metric", {"ticker": "NVDA", "metric": "P/E"}
    )
    metric = MetricResult.model_validate(metric_raw.structured_content)
    trace.append({"operation": "call_tool", "status": "ok", "result": "metric"})

    search_raw = await client.call_tool(
        "search_financial_documents",
        {"company": "Schneider Electric", "query": "energy management", "top_k": 2},
    )
    search = DocumentSearchResult.model_validate(search_raw.structured_content)
    trace.append({"operation": "call_tool", "status": "ok", "result": f"{len(search.hits)} hits"})

    prompt_raw = await client.get_prompt(
        "compare_companies",
        {"metric": "P/E", "question": "Compare valuation and operating evidence."},
    )
    rendered_prompt = prompt_raw.messages[0].content.text
    trace.append({"operation": "get_prompt", "status": "ok", "result": "user prompt"})

    failure_raw = await client.call_tool(
        "get_company_metric", {"ticker": "NVDA", "metric": "PE"}
    )
    failure_text = "\\n".join(
        block.text for block in failure_raw.content if isinstance(block, TextContent)
    )
    failure_json = re.search(r"[{].*[}]", failure_text, flags=re.DOTALL).group()
    failure = CapabilityError.model_validate_json(failure_json)
    trace.append({"operation": "call_tool", "status": "error", "result": failure.error_code})

print(f"server={server_name}")
print(f"catalog={len(resource_names)} resource | {len(tool_names)} tools | {len(prompt_names)} prompt")
print("tools=" + ", ".join(tool_names))
""",
        ),
        _code(
            "lesson10-007",
            """
capability_frame = pd.DataFrame(
    [
        ("RESOURCE", "Application", resource_names[0], "context"),
        ("TOOL", "Model proposes · host approves", tool_names[0], "read-only call"),
        ("TOOL", "Model proposes · host approves", tool_names[1], "evidence search"),
        ("PROMPT", "User", prompt_names[0], "reusable request"),
    ],
    columns=["primitive", "controller", "discovered name", "role"],
)
display(capability_frame)

fig, ax = plt.subplots(figsize=(11.5, 4.0))
ax.axis("off")
table = ax.table(
    cellText=capability_frame.values,
    colLabels=capability_frame.columns,
    cellLoc="left",
    colLoc="left",
    loc="center",
    colWidths=[0.15, 0.29, 0.34, 0.22],
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.8)
for column in range(len(capability_frame.columns)):
    table[(0, column)].set_facecolor("#051C2A")
    table[(0, column)].get_text().set_color("white")
    table[(0, column)].get_text().set_weight("bold")
for row, color in enumerate(("#EAF7FD", "#F4F7FF", "#F4F7FF", "#FFF2E5"), start=1):
    for column in range(len(capability_frame.columns)):
        table[(row, column)].set_facecolor(color)
ax.set_title("Figure 2. Discovery returns names and schemas with different controllers", loc="left", weight="bold")
plt.tight_layout()
plt.show()
""",
        ),
        _markdown(
            "lesson10-008",
            """
## Read the returned evidence

The protocol boundary must not erase provenance. The resource states coverage; the metric keeps its date and source; document search keeps document and evidence IDs; the prompt is rendered for user review without making a model call.
""",
        ),
        _code(
            "lesson10-009",
            """
display(pd.DataFrame([coverage.model_dump(mode="json")]))
display(pd.DataFrame([metric.model_dump(mode="json")]))
display(pd.DataFrame([hit.model_dump(mode="json") for hit in search.hits]))
print("Rendered compare_companies prompt:\\n")
print(rendered_prompt)
""",
        ),
        _markdown(
            "lesson10-010",
            """
## Discovery is not permission

A server can advertise a capability that is irrelevant, unsafe, or outside the user's request. The host must still apply:

- **approval** for consequential calls;
- **authentication** for remote servers;
- a static **allowlist** of permitted capability names;
- **argument validation** against the discovered schema; and
- evidence and audit requirements after execution.

`stdio` is a local transport, not an authentication mechanism. Remote MCP uses a URL and requires a separate production security design.
""",
        ),
        _code(
            "lesson10-011",
            """
try:
    await call_allowlisted_tool("delete_portfolio", {"ticker": "NVDA"})
except ValueError as error:
    allowlist_refusal = "blocked"
    print(f"allowlist_refusal={allowlist_refusal}")
    print(str(error))

print(
    "typed_error="
    f"{failure.error_code} | rejected={failure.rejected_value} | "
    f"valid={', '.join(failure.valid_values)} | retryable={failure.retryable}"
)

trace_frame = pd.DataFrame(trace)
display(trace_frame)

fig, ax = plt.subplots(figsize=(11.5, 3.8))
x = range(1, len(trace_frame) + 1)
colors = ["#F07D00" if status == "error" else "#2E8B57" for status in trace_frame["status"]]
ax.plot(list(x), [0] * len(trace_frame), color="#A0A7AE", linewidth=2, zorder=1)
ax.scatter(list(x), [0] * len(trace_frame), s=230, color=colors, zorder=3)
for index, row in trace_frame.iterrows():
    ax.annotate(
        row["operation"].replace("_", "\\n"),
        (index + 1, 0),
        xytext=(0, 24 if index % 2 == 0 else -42),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        weight="bold",
    )
ax.set(xlim=(0.5, len(trace_frame) + 0.5), ylim=(-0.65, 0.65), xlabel="Observed protocol operation")
ax.set_yticks([])
ax.set_title("Figure 3. Successful calls and one typed error remain visible in one trace", loc="left", weight="bold")
ax.grid(axis="x", alpha=0.12)
plt.tight_layout()
plt.show()

transport_frame = pd.DataFrame(
    [
        ("Local stdio", "host starts subprocess", "local machine", "host policy"),
        ("Remote MCP", "server_url", "network service", "auth + approval + host policy"),
    ],
    columns=["connection", "entry point", "server location", "required boundary"],
)
fig, ax = plt.subplots(figsize=(11.5, 2.8))
ax.axis("off")
transport_table = ax.table(
    cellText=transport_frame.values,
    colLabels=transport_frame.columns,
    cellLoc="left",
    colLoc="left",
    loc="center",
    colWidths=[0.17, 0.25, 0.22, 0.36],
)
transport_table.auto_set_font_size(False)
transport_table.set_fontsize(10)
transport_table.scale(1, 1.9)
for column in range(len(transport_frame.columns)):
    transport_table[(0, column)].set_facecolor("#051C2A")
    transport_table[(0, column)].get_text().set_color("white")
    transport_table[(0, column)].get_text().set_weight("bold")
for column in range(len(transport_frame.columns)):
    transport_table[(1, column)].set_facecolor("#EAF7FD")
    transport_table[(2, column)].set_facecolor("#FFF2E5")
ax.set_title("Figure 4. The protocol contract travels; the deployment boundary changes", loc="left", weight="bold")
plt.tight_layout()
plt.show()
""",
        ),
        _markdown(
            "lesson10-012",
            """
## Optional live selection through the shared gateway

The real MCP core already ran without a model. In live mode, the model receives only the discovered tool names, descriptions, and input schemas. It proposes a typed selection; Python checks discovery and the allowlist before execution.

OpenAI uses the project `.env` and `gpt-5.6-luna`. Ollama uses the same contract.
""",
        ),
        _code(
            "lesson10-013",
            """
class DiscoveredToolChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: Literal["get_company_metric"]
    ticker: str
    metric: Literal["EPS", "P/E"]
    reason: str


tool_catalog = tuple(tool.model_dump(mode="json") for tool in tool_specs)

if LIVE_MODE:
    selection_model = create_chat_model(settings).with_structured_output(DiscoveredToolChoice)
    selection_prompt = f'''Choose one read-only tool for this request: retrieve Schneider Electric EPS.
Use only this discovered catalog: {json.dumps(tool_catalog)}
Return the tool name, ticker, and metric. Do not request an investment action.'''
    tool_choice = selection_model.invoke([("human", selection_prompt)])
    live_arguments = {"ticker": tool_choice.ticker, "metric": tool_choice.metric}
    live_tool_result = await call_allowlisted_tool(tool_choice.tool_name, live_arguments)
    print("live_provider=" + json.dumps(provider_summary(settings), sort_keys=True))
    print("selected=" + tool_choice.model_dump_json())
    print(f"protocol_error={live_tool_result.is_error}")
else:
    tool_choice = DiscoveredToolChoice(
        tool_name="get_company_metric",
        ticker="SU.PA",
        metric="EPS",
        reason="Recorded valid selection for offline verification.",
    )
    print("offline_selection=" + tool_choice.model_dump_json())
""",
        ),
        _code(
            "lesson10-014",
            """
assert server_name == "First Finance Research"
assert resource_names == ("finance://coverage",)
assert tool_names == ("get_company_metric", "search_financial_documents")
assert prompt_names == ("compare_companies",)
assert coverage.dataset_id == "lesson10-financial-mcp-v1"
assert metric.as_of and metric.source
assert search.hits and all(hit.evidence_id and hit.source for hit in search.hits)
assert failure.error_code == "unsupported_metric" and "P/E" in failure.valid_values
assert allowlist_refusal == "blocked"
assert tool_choice.tool_name in tool_names and tool_choice.tool_name in ALLOWED_TOOLS
print("LESSON_10_PASS")
""",
        ),
        _markdown(
            "lesson10-015",
            """
## Knowledge check and capstone handoff

1. Who opens and closes the local `stdio` connection?
   **The host, through its MCP client.**
2. What does discovery prove?
   **Only what the server declares—not trust, permission, or safe arguments.**
3. Which primitive packages the reusable comparison request?
   **The user-controlled `compare_companies` prompt.**
4. What changes when the server becomes remote?
   **The same MCP contract gains network authentication, authorization, approval, rate-limit, and audit requirements.**

### Recap

- MCP separates capability contracts from application implementation.
- The notebook executed the real `list → inspect → call → record` route.
- Resources, tools, and prompts have different controllers.
- The host keeps policy and evidence even when a model proposes a tool.
- Lesson 11 plans across these discovered read-only capabilities.
""",
        ),
    ]

    notebook = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "finai": {
                "lesson": 10,
                "title": "Connect financial tools with MCP",
                "expected_runtime_minutes": 30,
                "offline_default": True,
            },
        },
    )
    notebook["nbformat"] = 4
    notebook["nbformat_minor"] = 5
    return notebook


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(build_notebook(), OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
