"""Build the canonical output-free Lesson 11 plan-and-execute notebook."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/11_plan_and_execute_analyst.ipynb"


def _markdown(cell_id: str, source: str):
    cell = nbformat.v4.new_markdown_cell(dedent(source).strip())
    cell.id = cell_id
    return cell


def _code(cell_id: str, source: str):
    cell = nbformat.v4.new_code_cell(dedent(source).strip())
    cell.id = cell_id
    return cell


def build_notebook():
    """Return the deterministic 40-minute Lesson 11 notebook."""

    notebook = nbformat.v4.new_notebook()
    notebook.metadata.kernelspec = {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata.language_info = {"name": "python", "version": "3.11"}
    notebook.metadata.finai = {"expected_runtime_minutes": 40, "lesson": "11"}
    notebook.cells = [
        _markdown(
            "lesson11-000",
            """
            # 11 - Plan-and-execute financial analyst

            **First Finance - Arnaud Demes**
            **Day 2 · 13:30-14:30 · 12 minutes deck + 40 minutes notebook + 8 minutes debrief**

            **Outcome:** plan, execute, revise, and synthesize a cited NVIDIA and Schneider Electric research mission through one read-only MCP lifecycle.

            **Prerequisite:** Lessons 08-10, especially bounded LangGraph recovery and financial MCP discovery. The controlled evidence is not live market data or investment advice.
            """,
        ),
        _markdown(
            "lesson11-001",
            """
            ## Learning objectives

            By the end, you can:

            1. distinguish a next-action loop from a validated research plan;
            2. inspect `ResearchPlan`, `ReplanDecision`, observations, and a briefing;
            3. keep model proposals separate from host validation and MCP execution;
            4. replace only unfinished work after a typed tool failure;
            5. require metric and document evidence for both companies before reporting; and
            6. retain a trajectory that Lesson 12 can evaluate.
            """,
        ),
        _markdown(
            "lesson11-002",
            """
            ## Where this fits

            Lesson 08 made workflow and agent control visible. Lesson 09 corrected one tool request. Lesson 10 discovered permitted tools through MCP. Lesson 11 coordinates several evidence steps while the host keeps the execution and evidence boundaries.

            ```text
            mission -> discovered catalog -> validated plan -> observations -> revised tail -> evidence gate -> cited briefing
            ```
            """,
        ),
        _code(
            "lesson11-003",
            """
            import os
            from pathlib import Path

            import matplotlib.pyplot as plt
            import pandas as pd
            from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

            from finai_academy.plan_execute_graph import PlanExecuteResult, run_plan_execute
            from finai_academy.plan_execute_policies import (
                INITIAL_RECORDED_STEPS,
                MISSION,
                build_live_plan_execute_policies,
                recorded_planner,
                recorded_replanner,
                recorded_report_writer,
            )
            from finai_academy.planning_mcp_executor import FinancialMcpPlanningExecutor
            from finai_academy.research_planning import (
                ReplanDecision,
                validate_plan,
            )
            from finai_academy.settings import Settings

            PROJECT_ROOT = Path.cwd().resolve()
            if PROJECT_ROOT.name == "notebooks":
                PROJECT_ROOT = PROJECT_ROOT.parent

            LIVE_MODE = os.getenv("FINAI_LIVE_MODE", "0") == "1"
            run_details: dict[str, object] = {}


            async def run_lesson11(*, live_mode: bool) -> PlanExecuteResult:
                settings = Settings.from_environment()
                if live_mode:
                    planner, replanner, report_writer = build_live_plan_execute_policies(settings)
                else:
                    planner = recorded_planner
                    replanner = recorded_replanner
                    report_writer = recorded_report_writer
                async with FinancialMcpPlanningExecutor() as executor:
                    run_details["server_name"] = executor.server_name
                    run_details["permitted_tools"] = ", ".join(tool.name for tool in executor.catalog)
                    run_details["catalog"] = executor.catalog
                    return await run_plan_execute(
                        question=MISSION,
                        executor=executor,
                        planner=planner,
                        replanner=replanner,
                        report_writer=report_writer,
                    )


            runtime_label = (
                f"live {Settings.from_environment().provider} route through the shared graph"
                if LIVE_MODE
                else "offline fixture · deterministic planner and replanner · real local MCP execution"
            )
            print(f"Runtime: {runtime_label}")
            print("Boundary: discovered, allowlisted, read-only research tools")
            """,
        ),
        _markdown(
            "lesson11-004",
            """
            ### Figure 1 supports a control decision

            A workflow fixes the path. ReAct selects one next action at a time. Plan-and-execute validates a coordinated plan, then permits only one controlled action at a time.
            """,
        ),
        _code(
            "lesson11-005",
            """
            fig, ax = plt.subplots(figsize=(12, 4.8))
            ax.axis("off")
            patterns = [
                ("Workflow", ["fixed steps", "known path"], "#EAF7FD", "#1F40CB"),
                ("ReAct", ["observe", "choose next action", "repeat"], "#FFF2E5", "#F07D00"),
                ("Plan-and-execute", ["validated plan", "execute one step", "revise tail"], "#EAF8EE", "#2E8B57"),
            ]
            for column, (title, steps, fill, edge) in enumerate(patterns):
                x = 0.04 + column * 0.32
                ax.text(x + 0.13, 0.92, title, ha="center", weight="bold", color=edge, fontsize=13)
                for index, step in enumerate(steps):
                    y = 0.71 - index * 0.18
                    ax.add_patch(FancyBboxPatch((x, y), 0.26, 0.10, boxstyle="round,pad=0.02", facecolor=fill, edgecolor=edge, linewidth=1.8))
                    ax.text(x + 0.13, y + 0.05, step, ha="center", va="center", fontsize=9)
                    if index < len(steps) - 1:
                        ax.add_patch(FancyArrowPatch((x + 0.13, y), (x + 0.13, y - 0.07), arrowstyle="-|>", mutation_scale=12, color="#4B6070"))
            ax.text(0.5, 0.06, "Lesson 11 uses a plan only when the mission benefits from coordinated evidence collection.", ha="center", weight="bold", color="#051C2A")
            ax.set_title("Figure 1. Control patterns make different planning commitments", loc="left", weight="bold")
            plt.tight_layout()
            plt.show()
            """,
        ),
        _markdown(
            "lesson11-006",
            """
            ### Discover the permitted catalog and inspect the contracts

            One `FinancialMcpPlanningExecutor` stays open for the complete run. The planner sees only discovered, allowlisted tool metadata. The notebook then inspects four typed contracts without exposing client or process internals.
            """,
        ),
        _code(
            "lesson11-007",
            """
            result = await run_lesson11(live_mode=LIVE_MODE)
            print(f"Real MCP server: {run_details['server_name']}")
            print(f"Permitted tools: {run_details['permitted_tools']}")
            display(pd.DataFrame([
                ("ResearchPlan", "goal and ordered validated steps"),
                ("ReplanDecision", "continue, replace tail, finish, or stop"),
                ("ResearchObservation", "append-only tool result or typed error"),
                ("AnalystBriefing", "facts, comparison, interpretation, limits, sources"),
            ], columns=["contract", "teaching role"]))
            """,
        ),
        _code(
            "lesson11-008",
            """
            fig, ax = plt.subplots(figsize=(12, 4.6))
            ax.axis("off")
            positions = {1: (0.04, 0.67), 2: (0.34, 0.67), 3: (0.04, 0.23), 4: (0.34, 0.23)}
            for step in INITIAL_RECORDED_STEPS:
                x, y = positions[step.step_id]
                color = "#F07D00" if step.step_id == 3 else "#1F40CB"
                labels = {
                    1: "1. Metric lookup\\nNVIDIA P/E",
                    2: "2. Metric lookup\\nSchneider P/E",
                    3: "3. Metric lookup\\nNVIDIA Revenue",
                    4: "4. Document search\\nSchneider revenue",
                }
                ax.add_patch(FancyBboxPatch((x, y), 0.24, 0.18, boxstyle="round,pad=0.02", facecolor="#F5F5F5", edgecolor=color, linewidth=2))
                ax.text(x + 0.12, y + 0.09, labels[step.step_id], ha="center", va="center", fontsize=9)
                for dependency in step.depends_on:
                    start_x, start_y = positions[dependency]
                    ax.add_patch(FancyArrowPatch((start_x + 0.12, start_y - 0.01), (x + 0.12, y + 0.18), arrowstyle="-|>", mutation_scale=14, color="#4B6070"))
            ax.text(0.70, 0.62, "Step 3 is schema-valid input,\\nbut a domain-invalid metric request.", fontsize=9, va="center", bbox={"boxstyle": "round,pad=0.4", "facecolor": "#FFF2E5", "edgecolor": "#F07D00"})
            ax.text(0.70, 0.30, "Step 4 is unfinished work.\\nIt may be replaced after failure.", fontsize=9, va="center", bbox={"boxstyle": "round,pad=0.4", "facecolor": "#EAF7FD", "edgecolor": "#1F40CB"})
            ax.set_title("Figure 2. Initial plan dependencies show the strategy before execution", loc="left", weight="bold")
            plt.tight_layout()
            plt.show()
            """,
        ),
        _markdown(
            "lesson11-009",
            """
            ### Validate the plan before it can call a tool

            Model-owned roles propose a plan, a replan decision, and a report. The host owns capability validation, argument checks, step and revision limits, execution, and the evidence gate.
            """,
        ),
        _code(
            "lesson11-010",
            """
            validated_plan = validate_plan(
                result.initial_plan, run_details["catalog"], max_steps=6
            )
            print(f"Initial plan steps: {[step.step_id for step in result.initial_plan.steps]}")
            print(f"Final retained steps: {[step.step_id for step in result.final_steps]}")
            print(f"Validated plan steps: {[step.step_id for step in validated_plan.steps]}")
            print(f"Host policy result: {result.trajectory[1].summary}")
            assert tuple(step.capability for step in result.initial_plan.steps) == (
                "get_company_metric", "get_company_metric", "get_company_metric", "search_financial_documents"
            )
            """,
        ),
        _code(
            "lesson11-011",
            """
            fig, ax = plt.subplots(figsize=(12, 4.8))
            ax.axis("off")
            nodes = [
                ("Planner", 0.04, "model", "#EAF7FD", "#1F40CB"),
                ("Plan gate", 0.20, "host", "#FFF2E5", "#F07D00"),
                ("Executor", 0.36, "host + MCP", "#EAF8EE", "#2E8B57"),
                ("Replanner", 0.52, "model", "#EAF7FD", "#1F40CB"),
                ("Evidence gate", 0.68, "host", "#FFF2E5", "#F07D00"),
                ("Report", 0.84, "model", "#EAF7FD", "#1F40CB"),
            ]
            for index, (label, x, owner, fill, edge) in enumerate(nodes):
                ax.add_patch(FancyBboxPatch((x, 0.49), 0.12, 0.18, boxstyle="round,pad=0.02", facecolor=fill, edgecolor=edge, linewidth=2))
                ax.text(x + 0.06, 0.59, label, ha="center", va="center", fontsize=9, weight="bold")
                ax.text(x + 0.06, 0.41, owner, ha="center", fontsize=8, color=edge)
                if index < len(nodes) - 1:
                    ax.add_patch(FancyArrowPatch((x + 0.12, 0.58), (nodes[index + 1][1], 0.58), arrowstyle="-|>", mutation_scale=13, color="#4B6070"))
            ax.add_patch(FancyArrowPatch((0.58, 0.49), (0.42, 0.49), arrowstyle="-|>", mutation_scale=13, color="#4B6070", connectionstyle="arc3,rad=-0.45"))
            ax.text(0.50, 0.18, "A typed failure can revise only the unfinished tail. The host still validates the replacement.", ha="center", weight="bold", color="#051C2A")
            ax.set_title("Figure 3. One six-node graph separates proposal from control", loc="left", weight="bold")
            plt.tight_layout()
            plt.show()
            """,
        ),
        _markdown(
            "lesson11-012",
            """
            ## Failure lab

            The maintained failure is `get_company_metric(ticker="NVDA", metric="Revenue")`. `Revenue` is not a controlled metric, so the real MCP server returns `unsupported_metric`. This is a strategy revision, not a retry with a spelling correction.
            """,
        ),
        _code(
            "lesson11-013",
            """
            first_three = result.observations[:3]
            display(pd.DataFrame([
                {
                    "attempt": item.attempt_id,
                    "step": item.step_id,
                    "capability": item.capability,
                    "status": item.status,
                    "error": item.error_code or "",
                    "evidence count": len(item.evidence_ids),
                }
                for item in first_three
            ]))
            assert first_three[2].error_code == "unsupported_metric"
            print("The third attempt is retained as a typed error in the trajectory.")
            """,
        ),
        _code(
            "lesson11-014",
            """
            status_colors = {"ok": "#2E8B57", "error": "#F07D00", "blocked": "#C83737"}
            fig, ax = plt.subplots(figsize=(12, 4.8))
            observations = result.observations
            durations = [max(item.duration_ms, 0.1) for item in observations]
            bars = ax.bar([str(item.attempt_id) for item in observations], durations, color=[status_colors[item.status] for item in observations])
            for bar, item in zip(bars, observations, strict=True):
                ax.text(bar.get_x() + bar.get_width() / 2, max(item.duration_ms, 0.1) * 1.35, f"step {item.step_id}\\n{len(item.evidence_ids)} evidence", ha="center", va="bottom", fontsize=8)
            ax.set_xlabel("Execution attempt")
            ax.set_ylabel("Measured duration (ms, log scale)")
            ax.set_yscale("log")
            ax.set_ylim(min(durations) / 2, max(durations) * 3)
            ax.set_title("Figure 4. Timeline keeps success, failure, duration, and evidence visible", loc="left", weight="bold")
            ax.grid(axis="y", alpha=0.25)
            plt.tight_layout()
            plt.show()
            """,
        ),
        _markdown(
            "lesson11-015",
            """
            ### Replan the tail only

            The two successful valuation observations remain immutable. The rejected third step and original unfinished fourth step are superseded by newly numbered document searches. The retained error explains why the plan changed.
            """,
        ),
        _code(
            "lesson11-016",
            """
            replan_decision = await recorded_replanner({
                "observations": result.observations[:3],
                "active_steps": INITIAL_RECORDED_STEPS,
                "current_index": 3,
            })
            display(pd.DataFrame([
                (replan_decision.action, replan_decision.reasoning, [step.step_id for step in replan_decision.replacement_steps]),
            ], columns=["action", "reasoning", "replacement step IDs"]))
            assert replan_decision.action == "replace_remaining"
            assert isinstance(replan_decision, ReplanDecision)
            """,
        ),
        _code(
            "lesson11-017",
            """
            fig, ax = plt.subplots(figsize=(12, 4.6))
            ax.axis("off")
            groups = [
                ("Executed prefix", ["1 NVIDIA P/E", "2 Schneider P/E"], 0.04, "#EAF8EE", "#2E8B57"),
                ("Rejected and superseded", ["3 Revenue metric", "4 original document tail"], 0.37, "#FFF2E5", "#F07D00"),
                ("Replacement tail", ["5 NVIDIA document", "6 Schneider document"], 0.70, "#EAF7FD", "#1F40CB"),
            ]
            for title, labels, x, fill, edge in groups:
                ax.text(x + 0.12, 0.91, title, ha="center", weight="bold", color=edge)
                for index, label in enumerate(labels):
                    y = 0.67 - index * 0.25
                    ax.add_patch(FancyBboxPatch((x, y), 0.24, 0.13, boxstyle="round,pad=0.02", facecolor=fill, edgecolor=edge, linewidth=2))
                    ax.text(x + 0.12, y + 0.065, label, ha="center", va="center", fontsize=9)
            ax.add_patch(FancyArrowPatch((0.61, 0.61), (0.70, 0.61), arrowstyle="-|>", mutation_scale=16, color="#4B6070"))
            ax.text(0.655, 0.70, "new IDs", ha="center", fontsize=9, color="#4B6070")
            ax.set_title("Figure 5. Replanning replaces unfinished work without deleting history", loc="left", weight="bold")
            plt.tight_layout()
            plt.show()
            """,
        ),
        _markdown(
            "lesson11-018",
            """
            ### Complete the corrected plan and check coverage

            The corrected route makes five attempts for step IDs 1, 2, 3, 5, and 6. It does not repeat the two successful metric calls or execute superseded step 4.
            """,
        ),
        _code(
            "lesson11-019",
            """
            display(pd.DataFrame([
                {
                    "attempt": item.attempt_id,
                    "step": item.step_id,
                    "revision": item.plan_revision,
                    "status": item.status,
                    "sources": len(item.source_references),
                }
                for item in result.observations
            ]))
            assert [item.step_id for item in result.observations] == [1, 2, 3, 5, 6]
            assert result.replan_count == 1
            print("No successful call was duplicated after replanning.")
            """,
        ),
        _code(
            "lesson11-020",
            """
            companies = ["NVIDIA", "Schneider Electric"]
            evidence_types = ["metric", "document"]
            coverage = result.evidence_gate.coverage
            matrix = [[1 if evidence_type in coverage.get(company, ()) else 0 for evidence_type in evidence_types] for company in companies]
            fig, ax = plt.subplots(figsize=(9, 4.6))
            image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
            for row, company in enumerate(companies):
                for column, evidence_type in enumerate(evidence_types):
                    present = bool(matrix[row][column])
                    ax.text(column, row, "present" if present else "missing", ha="center", va="center", color="#FFFFFF" if present else "#051C2A", weight="bold")
            ax.set_xticks(range(len(evidence_types)), evidence_types)
            ax.set_yticks(range(len(companies)), companies)
            ax.set_title("Figure 6. Evidence coverage matrix decides whether reporting may start", loc="left", weight="bold")
            fig.colorbar(image, ax=ax, ticks=[0, 1], label="Evidence present")
            plt.tight_layout()
            plt.show()
            """,
        ),
        _markdown(
            "lesson11-021",
            """
            ### Evidence gate and cited briefing

            The evidence gate requires one successful metric observation and one document evidence hit for each company. A fluent report cannot bypass this check.
            """,
        ),
        _code(
            "lesson11-022",
            """
            print(f"Evidence gate passed: {result.evidence_gate.passed}")
            display(pd.DataFrame([
                (company, ", ".join(evidence))
                for company, evidence in result.evidence_gate.coverage.items()
            ], columns=["company", "verified evidence"]))
            assert result.briefing is not None
            display(pd.DataFrame([
                ("reported facts", len(result.briefing.reported_facts)),
                ("comparison observations", len(result.briefing.cross_company_observations)),
                ("limitations", len(result.briefing.limitations)),
                ("source references", len(result.briefing.source_references)),
            ], columns=["briefing field", "count"]))
            """,
        ),
        _markdown(
            "lesson11-023",
            """
            ### Optional live route and Lesson 12 handoff

            Set `FINAI_LIVE_MODE=1` and `FINAI_MODEL_PROVIDER=ollama` for Ollama, or `FINAI_MODEL_PROVIDER=openai` with an available key for OpenAI. Both use the same graph, validation, MCP lifecycle, evidence gate, and trajectory shape as offline mode. Offline mode remains the maintained classroom route.
            """,
        ),
        _code(
            "lesson11-024",
            """
            trajectory_frame = pd.DataFrame([
                {
                    "index": event.index,
                    "phase": event.phase,
                    "status": event.status,
                    "step": event.step_id or "",
                    "attempt": event.attempt_id or "",
                }
                for event in result.trajectory
            ])
            display(trajectory_frame)
            print("Lesson 12 receives the mission, initial and final plans, observations, errors, evidence IDs, sources, and stage timing.")
            print("Ollama and OpenAI are optional live policies. Offline execution remains deterministic.")
            """,
        ),
        _markdown(
            "lesson11-025",
            """
            ## Verification

            The maintained offline run must preserve the controlled failure, make exactly one revision, pass the evidence gate, and emit one explicit success marker.
            """,
        ),
        _code(
            "lesson11-026",
            """
            if LIVE_MODE:
                print("Live route uses the same graph. Review its model-produced plan before accepting it.")
            else:
                assert result.status == "completed"
                assert result.replan_count == 1
                assert result.evidence_gate.passed is True
                assert sum(item.error_code == "unsupported_metric" for item in result.observations) == 1
                assert [item.step_id for item in result.observations] == [1, 2, 3, 5, 6]
                assert run_details["server_name"] == "First Finance Research"
                print("Plan revisions: 1")
                print("LESSON_11_PASS")
            """,
        ),
        _markdown(
            "lesson11-027",
            """
            ## Knowledge check

            1. Why is the failed step retained? It explains the observed strategy change.
            2. Who can execute a tool? The host, after validation and allowlisting.
            3. What blocks the report? Missing metric or document evidence for either company.

            ## Challenge

            Add an allowed document query only if it fits the six-step budget, preserves successful calls, and leaves the evidence gate rules unchanged.

            ## Capstone integration

            The Financial Analyst Copilot now produces a research trajectory: mission, catalog, plan, observations, replan, evidence gate, and cited briefing.

            ## Recap

            A plan makes multi-step research inspectable. Typed failures revise only unfinished work. Evidence gates prevent incomplete reporting. Lesson 12 evaluates both the answer and the path.
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
