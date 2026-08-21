"""Build the canonical output-free Lesson 10 financial MCP notebook."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/10_financial_mcp.ipynb"


def _markdown(cell_id: str, source: str):
    cell = nbformat.v4.new_markdown_cell(dedent(source).strip())
    cell.id = cell_id
    return cell


def _code(cell_id: str, source: str):
    cell = nbformat.v4.new_code_cell(dedent(source).strip())
    cell.id = cell_id
    return cell


def build_notebook():
    """Return the deterministic 30-minute Lesson 10 notebook."""

    notebook = nbformat.v4.new_notebook()
    notebook.metadata.kernelspec = {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata.language_info = {"name": "python", "version": "3.11"}
    notebook.metadata.finai = {"expected_runtime_minutes": 30, "lesson": "10"}
    notebook.cells = [
        _markdown(
            "lesson10-000",
            """
            # 10 — Build a financial MCP

            **First Finance - Arnaud Demes**  
            **Day 2 · 11:15–12:00 · 10 minutes concepts + 30 minutes notebook + 5 minutes debrief**

            **Engineering question:** how can an analyst application discover read-only financial capabilities from another local process instead of importing its business functions?

            This lesson uses controlled NVIDIA (`NVDA`) and Schneider Electric (`SU.PA`) fixtures. It is not live market data or investment advice. Run all cells in order. Offline mode is the classroom default.
            """,
        ),
        _markdown(
            "lesson10-001",
            """
            ## Learning objectives

            By the end, you can:

            1. name the MCP host, client, server, and `stdio` transport;
            2. distinguish application-controlled resources, model-requested tools, and user-controlled prompts;
            3. discover one resource, two tools, and one prompt at runtime;
            4. read a resource, call read-only tools, and render a prompt;
            5. retain dates, sources, and evidence IDs across the protocol boundary; and
            6. reject a tool name that is not discovered and allowlisted.

            **Expected visible result:** a local client completes the real MCP lifecycle, including one maintained typed validation error for `PE`.
            """,
        ),
        _markdown(
            "lesson10-002",
            """
            ## Where this fits

            Lesson 09 imports a tested metric registry directly into a bounded agent. Lesson 10 moves the same read-only capability behind a process boundary:

            ```text
            direct import → capability discovery → host approval → protocol call
            ```

            The current SDK uses `MCPServer`. Older v1 tutorials may show `FastMCP`; the decorator pattern is familiar, but this lesson uses the v2 name.
            """,
        ),
        _code(
            "lesson10-003",
            """
            import json
            import os
            from pathlib import Path

            import matplotlib.pyplot as plt
            import pandas as pd
            from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
            from pydantic import BaseModel

            from finai_academy.financial_mcp_client import (
                DiscoveredToolSpec,
                call_allowlisted_tool,
                discover_and_run_financial_mcp,
            )
            from finai_academy.providers import create_chat_model, provider_summary
            from finai_academy.settings import Settings

            PROJECT_ROOT = Path.cwd().resolve()
            if PROJECT_ROOT.name == "notebooks":
                PROJECT_ROOT = PROJECT_ROOT.parent

            LIVE_MODE = os.getenv("FINAI_LIVE_MODE", "0") == "1"
            settings = Settings.from_environment()
            runtime_label = (
                provider_summary(settings)
                if LIVE_MODE
                else "offline fixture · deterministic course run"
            )
            print(f"Runtime: {runtime_label}")
            print("Transport: local stdio")
            print("Boundary: read-only research capabilities")
            """,
        ),
        _markdown(
            "lesson10-004",
            """
            ### Direct imports couple the host to implementation

            With a direct import, the host knows Python function names and data shapes. With MCP, the host discovers declared capabilities, chooses whether they are permitted, and then sends protocol requests. MCP discovery does not grant permission by itself.
            """,
        ),
        _code(
            "lesson10-005",
            """
            fig, ax = plt.subplots(figsize=(12, 4.8))
            ax.axis("off")
            panels = [
                (0.05, "Direct import", ["Host", "import registry", "call Python function"], "#FFF2E5", "#F07D00"),
                (0.54, "MCP discovery", ["Host", "MCP client", "MCPServer over stdio", "versioned fixtures"], "#EAF7FD", "#1F40CB"),
            ]
            for x, title, boxes, fill, edge in panels:
                ax.text(x + 0.19, 0.91, title, ha="center", weight="bold", fontsize=14, color=edge)
                for index, label in enumerate(boxes):
                    y = 0.72 - index * 0.18
                    ax.add_patch(FancyBboxPatch((x, y), 0.38, 0.11, boxstyle="round,pad=0.02", facecolor=fill, edgecolor=edge, linewidth=1.8))
                    ax.text(x + 0.19, y + 0.055, label, ha="center", va="center", fontsize=10)
                    if index < len(boxes) - 1:
                        ax.add_patch(FancyArrowPatch((x + 0.19, y), (x + 0.19, y - 0.065), arrowstyle="-|>", mutation_scale=13, color="#4B6070"))
            ax.text(0.5, 0.06, "The host owns lifecycle and permissions. The server exposes declarations and results.", ha="center", color="#051C2A", weight="bold")
            ax.set_title("Figure 1. MCP replaces implementation coupling with a discoverable boundary", loc="left", weight="bold")
            plt.tight_layout()
            plt.show()
            """,
        ),
        _markdown(
            "lesson10-006",
            """
            ### Inspect the tracked contracts and registrations

            The `FinancialCapabilityRegistry` owns validation and evidence. The adapter registers one `finance://coverage` resource, two read-only tools, and one prompt. This compact read-only view extracts the real method contracts and decorators from the tracked course modules without starting a second server or importing server business functions into the core run.
            """,
        ),
        _code(
            "lesson10-007",
            """
            registry_source = (
                PROJECT_ROOT / "src/finai_academy/financial_mcp_capabilities.py"
            ).read_text(encoding="utf-8")
            server_source = (
                PROJECT_ROOT / "src/finai_academy/financial_mcp_server.py"
            ).read_text(encoding="utf-8")
            registry_contracts = [
                line.strip()
                for line in registry_source.splitlines()
                if line.strip().startswith("def ")
                and any(
                    name in line
                    for name in (
                        "coverage(",
                        "get_company_metric(",
                        "search_financial_documents(",
                    )
                )
            ]
            registrations = [
                line.strip()
                for line in server_source.splitlines()
                if line.strip().startswith("@server.")
            ]
            display(pd.DataFrame({"registry method contract": registry_contracts}))
            display(pd.DataFrame({"MCPServer registration": registrations}))
            """,
        ),
        _code(
            "lesson10-008",
            """
            control_frame = pd.DataFrame(
                [
                    ("Resource", "Application", "finance://coverage", "Read context before a model chooses"),
                    ("Tool", "Model + host approval", "get_company_metric", "Run a bounded read-only action"),
                    ("Tool", "Model + host approval", "search_financial_documents", "Find versioned evidence"),
                    ("Prompt", "User", "compare_companies", "Render a reusable request"),
                ],
                columns=["primitive", "controller", "example", "purpose"],
            )
            colors = {"Resource": "#1F40CB", "Tool": "#00A2EB", "Prompt": "#F07D00"}
            fig, ax = plt.subplots(figsize=(11.5, 3.7))
            ax.axis("off")
            for row_index, row in control_frame.iterrows():
                y = 0.76 - row_index * 0.18
                ax.add_patch(FancyBboxPatch((0.03, y), 0.18, 0.11, boxstyle="round,pad=0.01", facecolor="#F5F5F5", edgecolor=colors[row["primitive"]], linewidth=2))
                ax.text(0.12, y + 0.055, row["primitive"], ha="center", va="center", weight="bold")
                ax.text(0.25, y + 0.055, row["controller"], va="center", fontsize=10, weight="bold")
                ax.text(0.55, y + 0.055, row["example"], va="center", fontsize=10, family="monospace")
                ax.text(0.78, y + 0.055, row["purpose"], va="center", fontsize=9)
            ax.text(0.03, 0.95, "Primitive", weight="bold")
            ax.text(0.25, 0.95, "Controller", weight="bold")
            ax.text(0.55, 0.95, "Concrete capability", weight="bold")
            ax.text(0.78, 0.95, "Why it exists", weight="bold")
            ax.set_title("Figure 2. Resources, tools, and prompts have different control models", loc="left", weight="bold")
            plt.tight_layout()
            plt.show()
            """,
        ),
        _markdown(
            "lesson10-009",
            """
            ### Start the real local lifecycle

            The next cell starts the server subprocess, opens the client context, discovers capabilities, performs the reads and calls, renders the prompt, then closes the `stdio` transport. No model is needed for this core run.
            """,
        ),
        _code(
            "lesson10-010",
            """
            mcp_run = await discover_and_run_financial_mcp()
            print(f"Server: {mcp_run.server_name}")
            print(f"Resources: {', '.join(mcp_run.resource_names)}")
            print(f"Tools: {', '.join(mcp_run.tool_names)}")
            print(f"Prompts: {', '.join(mcp_run.prompt_names)}")
            """,
        ),
        _code(
            "lesson10-011",
            """
            sequence_steps = [
                ("host", "start server", "#1F40CB"),
                ("client", "open stdio", "#00A2EB"),
                ("server", "list capabilities", "#2E8B57"),
                ("client", "read and call", "#00A2EB"),
                ("client", "get prompt", "#00A2EB"),
                ("client", "close stdio", "#00A2EB"),
            ]
            lanes = {"host": 2, "client": 1, "server": 0}
            fig, ax = plt.subplots(figsize=(12, 4.6))
            for name, lane in lanes.items():
                ax.hlines(lane, 0.4, 6.6, color="#D7DEE3", linewidth=2)
                ax.text(0.05, lane, name.title(), va="center", weight="bold", color="#4B6070")
            for index, (actor, label, color) in enumerate(sequence_steps, start=1):
                lane = lanes[actor]
                ax.scatter(index, lane, s=260, color=color, zorder=3)
                ax.annotate(label, (index, lane), xytext=(0, 18 if index % 2 else -32), textcoords="offset points", ha="center", fontsize=9, weight="bold")
                if index < len(sequence_steps):
                    next_lane = lanes[sequence_steps[index][0]]
                    ax.add_patch(FancyArrowPatch((index + 0.08, lane), (index + 0.92, next_lane), arrowstyle="-|>", mutation_scale=13, color="#4B6070", alpha=0.75))
            ax.text(5, 2.32, "Host intent: user selects compare_companies", ha="center", fontsize=9, color="#1F40CB", weight="bold")
            ax.set(xlim=(0, 7), ylim=(-0.55, 2.55), xticks=range(1, 7), xlabel="Protocol phase")
            ax.set_yticks([])
            ax.set_title("Figure 3. The host owns the stdio lifecycle and the client carries requests", loc="left", weight="bold")
            ax.grid(axis="x", alpha=0.15)
            plt.tight_layout()
            plt.show()
            """,
        ),
        _markdown(
            "lesson10-012",
            """
            ### Inspect discovery before using a capability

            The discovered names, not local Python imports, are the source of truth for this run. The host can now apply its own static allowlist and runtime discovery checks before it lets any caller invoke a tool.
            """,
        ),
        _code(
            "lesson10-013",
            """
            capability_frame = pd.DataFrame(
                [item.model_dump() for item in mcp_run.capabilities]
            ).rename(columns={"name": "discovered_name"})
            capability_frame["controller"] = capability_frame["primitive"].map(
                {"resource": "application", "tool": "model + host approval", "prompt": "user"}
            )
            display(capability_frame)
            """,
        ),
        _code(
            "lesson10-014",
            """
            primitive_order = ["resource", "tool", "prompt"]
            catalog_counts = capability_frame["primitive"].value_counts().reindex(primitive_order, fill_value=0)
            fig, ax = plt.subplots(figsize=(8.8, 4.2))
            catalog_counts.plot.barh(ax=ax, color=["#1F40CB", "#00A2EB", "#F07D00"])
            ax.set(xlabel="Discovered capabilities", ylabel="Primitive")
            ax.set_title("Figure 4. Discovery returns the compact course capability catalog", loc="left", weight="bold")
            for index, value in enumerate(catalog_counts):
                ax.text(value + 0.03, index, str(value), va="center", weight="bold")
            ax.set_xlim(0, max(catalog_counts) + 0.8)
            plt.tight_layout()
            plt.show()
            """,
        ),
        _markdown(
            "lesson10-015",
            """
            ### Read evidence and render the user prompt

            `finance://coverage` states the controlled dataset boundary. The metric retains its date and source. Document search retains evidence identifiers. The prompt is rendered for the user to review; it does not make a model call.
            """,
        ),
        _code(
            "lesson10-016",
            """
            display(pd.DataFrame([mcp_run.coverage.model_dump(mode="json")]))
            display(pd.DataFrame([mcp_run.metric.model_dump(mode="json")]))
            document_frame = pd.DataFrame(
                [hit.model_dump(mode="json") for hit in mcp_run.search.hits]
            )
            display(document_frame)
            print("Rendered compare_companies prompt:\\n")
            print(mcp_run.rendered_prompt)
            """,
        ),
        _markdown(
            "lesson10-017",
            """
            ## Failure lab

            The maintained invalid alias is `PE`. The protocol returns a typed error with `unsupported_metric`, valid values including `P/E`, and retryability. It is visible data for the host, not an unhandled notebook exception.
            """,
        ),
        _code(
            "lesson10-018",
            """
            failure_frame = pd.DataFrame([mcp_run.failure.model_dump(mode="json")])
            display(failure_frame)
            trace_frame = pd.DataFrame([event.model_dump(mode="json") for event in mcp_run.trace])
            display(trace_frame)

            milestones = trace_frame.loc[
                (trace_frame["operation"].isin({"open_transport", "list_tools", "read_resource", "get_prompt", "close_transport"}))
                | (trace_frame["status"] == "error")
            ].copy()
            milestones["label"] = ["Open", "Discover", "Coverage", "Prompt", "PE error", "Close"]
            milestone_colors = milestones["status"].map({"ok": "#2E8B57", "error": "#F07D00"})
            fig, ax = plt.subplots(figsize=(10.5, 4.2))
            ax.plot(milestones["sequence"], [0] * len(milestones), color="#A0A7AE", linewidth=2, zorder=1)
            ax.scatter(milestones["sequence"], [0] * len(milestones), s=260, color=milestone_colors, zorder=3)
            for _, row in milestones.iterrows():
                ax.annotate(row["label"], (row["sequence"], 0), xytext=(0, 24 if row["sequence"] % 2 else -35), textcoords="offset points", ha="center", fontsize=10, weight="bold")
            ax.scatter([], [], color="#2E8B57", label="successful lifecycle or call")
            ax.scatter([], [], color="#F07D00", label="typed tool error")
            ax.legend(loc="upper right", frameon=False)
            ax.set(xlim=(0.5, len(trace_frame) + 0.5), ylim=(-0.65, 0.65), xlabel="Recorded operation sequence")
            ax.set_yticks([])
            ax.set_title("Figure 5. Successful protocol calls and one typed validation error share one trace", loc="left", weight="bold")
            ax.grid(axis="x", alpha=0.15)
            plt.tight_layout()
            plt.show()
            """,
        ),
        _markdown(
            "lesson10-019",
            """
            ### Optional live selection through the shared gateway

            The core already worked without a model. Set `FINAI_LIVE_MODE=1` through the executor only when Ollama or OpenAI is configured. The gateway receives only discovered tool names, descriptions, and input schemas. Python validates the returned tool name through `call_allowlisted_tool()` before executing it.
            """,
        ),
        _code(
            "lesson10-020",
            """
            class DiscoveredToolChoice(BaseModel):
                tool_name: str
                arguments: dict[str, str | int | float]
                reason: str


            discovered_tool_catalog: tuple[DiscoveredToolSpec, ...] = mcp_run.tool_specs
            tool_catalog = tuple(
                tool.model_dump(mode="json") for tool in discovered_tool_catalog
            )
            discovered_tool_names = set(mcp_run.tool_names)
            assert discovered_tool_names == {item["name"] for item in tool_catalog}

            if LIVE_MODE:
                selection_model = create_chat_model(settings).with_structured_output(DiscoveredToolChoice)
                selection_prompt = f'''Choose one tool for a read-only financial research request.
            Use only this discovered catalog: {json.dumps(tool_catalog)}
            Return arguments that match the selected schema. Do not request an investment action.'''
                tool_choice = selection_model.invoke([("human", selection_prompt)])
                live_tool_result = await call_allowlisted_tool(
                    tool_choice.tool_name, tool_choice.arguments
                )
                print("Live provider:", provider_summary(settings))
                print("Selected:", tool_choice.model_dump(mode="json"))
                print("Protocol error:", live_tool_result.is_error)
            else:
                tool_choice = DiscoveredToolChoice(
                    tool_name="get_company_metric",
                    arguments={"ticker": "SU.PA", "metric": "EPS"},
                    reason="Recorded valid selection for offline verification.",
                )
                print("Offline recorded selection:", tool_choice.model_dump(mode="json"))
            """,
        ),
        _markdown(
            "lesson10-021",
            """
            ## Verification

            Check observable behavior:

            - exactly one resource, two tools, and one prompt are discovered;
            - the core run uses a real local `stdio` client lifecycle;
            - coverage, metrics, and document hits retain provenance;
            - `PE` remains a typed retryable failure with `P/E` visible; and
            - a live model, if enabled, can select only a discovered allowlisted tool.
            """,
        ),
        _code(
            "lesson10-022",
            """
            assert mcp_run.resource_names == ("finance://coverage",)
            assert mcp_run.tool_names == ("get_company_metric", "search_financial_documents")
            assert mcp_run.prompt_names == ("compare_companies",)
            assert mcp_run.coverage.dataset_id == "lesson10-financial-mcp-v1"
            assert mcp_run.metric.as_of and mcp_run.metric.source
            assert mcp_run.search.hits and all(hit.evidence_id and hit.source for hit in mcp_run.search.hits)
            assert mcp_run.failure.error_code == "unsupported_metric"
            assert "P/E" in mcp_run.failure.valid_values
            assert mcp_run.failure.retryable is True
            assert {"open_transport", "close_transport"} <= set(trace_frame["operation"])
            print("LESSON_10_PASS")
            """,
        ),
        _markdown(
            "lesson10-023",
            """
            ## Challenge and knowledge check

            ### Knowledge check

            1. Which component opens and closes the local `stdio` lifecycle?
            2. Why must a discovered tool still pass an allowlist check?
            3. Which primitive renders a reusable user-controlled comparison request?

            **Answers:** the host through its MCP client; discovery is not permission or trust; the `compare_companies` prompt.

            Add a host policy that requires a user confirmation before any tool call, even a read-only call. Record the decision in the trace without sending credentials, local files, or personal data to the server.

            Advanced option: make the policy reject tool descriptions that request unrelated data. Explain why a discovered capability is still untrusted input.
            """,
        ),
        _markdown(
            "lesson10-024",
            """
            ## Capstone integration

            Lesson 10 contributes a clean external-capability boundary:

            - `MCPServer` declarations for one resource, two tools, and one prompt;
            - a real local `stdio` lifecycle;
            - runtime discovery instead of server-function imports;
            - host allowlisting before tool execution; and
            - provenance carried through metric and evidence results.

            Lesson 11 can plan across these discovered read-only capabilities while keeping the host responsible for permissions and final synthesis.
            """,
        ),
        _markdown(
            "lesson10-025",
            """
            ## Recap

            - MCP separates a capability contract from the host implementation.
            - Resources, tools, and prompts have distinct controllers.
            - Discovery identifies what the server offers; it does not grant trust.
            - The local `stdio` core is deterministic and needs no model.
            - Ollama and OpenAI use the same shared gateway and structured selection contract.
            - Streamable HTTP is a production extension that needs separate authentication, authorization, and audit controls.
            """,
        ),
    ]
    return notebook


def main() -> None:
    notebook = build_notebook()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, OUTPUT)
    print(f"Wrote {OUTPUT.relative_to(ROOT)} with {len(notebook.cells)} cells")


if __name__ == "__main__":
    main()
