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
    """Return the progressive 40-minute Lesson 11 notebook."""

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

            **Outcome:** approve a research plan before execution, observe one typed failure, revise only unfinished work, and permit a cited briefing only when the evidence is complete.

            The mission compares NVIDIA and Schneider Electric through one read-only MCP lifecycle. Controlled evidence is not live market data or investment advice.
            """,
        ),
        _markdown(
            "lesson11-001",
            """
            ## Learning objectives

            By the end, you can:

            1. explain when a multi-step mission benefits from a plan;
            2. distinguish a model proposal from host approval;
            3. inspect `ResearchPlan`, `ReplanDecision`, and typed observations;
            4. preserve successful work while revising an unfinished tail; and
            5. require metric and document evidence before writing a briefing.
            """,
        ),
        _markdown(
            "lesson11-002",
            """
            ## Where this fits

            Lesson 09 corrected one tool request. Lesson 10 discovered read-only tools through MCP. Lesson 11 coordinates several evidence needs while the host keeps control.

            ```text
            mission -> discover -> propose -> approve -> execute -> replan -> evidence gate -> briefing
            ```

            **The practical rule:** the model may propose the route; only the host may approve and execute it.
            """,
        ),
        _code(
            "lesson11-003",
            """
            import os
            import matplotlib.pyplot as plt
            import pandas as pd

            from finai_academy.plan_execute_graph import run_plan_execute
            from finai_academy.plan_execute_policies import (
                MAX_RESEARCH_STEPS,
                MISSION,
                build_live_plan_execute_policies,
                recorded_planner,
                recorded_replanner,
                recorded_report_writer,
            )
            from finai_academy.planning_mcp_executor import FinancialMcpPlanningExecutor
            from finai_academy.research_planning import ReplanDecision, ResearchPlan, validate_plan
            from finai_academy.settings import Settings

            LIVE_MODE = os.getenv("FINAI_LIVE_MODE", "0") == "1"
            settings = Settings.from_environment()
            if LIVE_MODE:
                planner, replanner, report_writer = build_live_plan_execute_policies(settings)
            else:
                planner, replanner, report_writer = (
                    recorded_planner,
                    recorded_replanner,
                    recorded_report_writer,
                )

            runtime_label = (
                f"live {settings.provider} route through the shared graph"
                if LIVE_MODE
                else "offline fixture · deterministic planner and replanner · real local MCP execution"
            )
            print(f"Runtime: {runtime_label}")
            """,
        ),
        _markdown(
            "lesson11-004",
            """
            ## 1. See the mission and tools before planning

            The planner receives the mission and only the catalog discovered from the running MCP server. Discovery supplies candidates; host policy decides what may execute.
            """,
        ),
        _code(
            "lesson11-005",
            """
            async with FinancialMcpPlanningExecutor() as preview_executor:
                catalog = tuple(preview_executor.catalog)
                server_name = preview_executor.server_name
                initial_plan: ResearchPlan = await planner(MISSION, catalog)
                approved_plan = validate_plan(initial_plan, catalog, max_steps=MAX_RESEARCH_STEPS)

            print(f"Real MCP server: {server_name}")
            display(pd.DataFrame([
                {
                    "tool": tool.name,
                    "purpose": tool.description,
                    "host decision": "read-only candidate",
                }
                for tool in catalog
            ]))
            plan_frame = pd.DataFrame([
                {
                    "step": step.step_id,
                    "capability": step.capability,
                    "arguments": step.arguments,
                    "depends on": list(step.depends_on),
                    "evidence expected": ", ".join(step.expected_evidence),
                }
                for step in approved_plan.steps
            ])
            display(plan_frame)
            print("Plan approved before execution: True")
            print("No tool has run yet.")
            """,
        ),
        _code(
            "lesson11-006",
            """
            colors = ["#2447D8", "#00A6E8", "#F47C00", "#8291A6"]
            labels = [f"{step.step_id}  {step.capability}" for step in approved_plan.steps]
            fig, ax = plt.subplots(figsize=(12, 4.2))
            ax.barh(range(len(labels)), [1] * len(labels), color=colors[: len(labels)])
            ax.set_yticks(range(len(labels)), labels)
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_title("Figure 1. The approved plan is visible before execution", loc="left", weight="bold")
            ax.spines[:].set_visible(False)
            for index, step in enumerate(approved_plan.steps):
                ax.text(0.03, index, f"expects: {', '.join(step.expected_evidence)}", va="center", color="white", weight="bold")
            plt.tight_layout()
            plt.show()
            """,
        ),
        _markdown(
            "lesson11-007",
            """
            ## Decision checkpoint

            Inspect step 3 before continuing. Its structure is valid, but `Revenue` is not one of the controlled metric names.

            **Choose:** should the system silently repair it, retry it unchanged, or preserve the failure and revise the remaining strategy? Write your answer before running the next cell.
            """,
        ),
        _code(
            "lesson11-008",
            """
            async def approved_planner(_question, _catalog):
                return approved_plan


            async with FinancialMcpPlanningExecutor() as executor:
                result = await run_plan_execute(
                    question=MISSION,
                    executor=executor,
                    planner=approved_planner,
                    replanner=replanner,
                    report_writer=report_writer,
                )
            trajectory_frame = pd.DataFrame([
                {
                    "event": event.index,
                    "phase": event.phase,
                    "status": event.status,
                    "step": event.step_id or "",
                    "attempt": event.attempt_id or "",
                    "summary": event.summary,
                }
                for event in result.trajectory
            ])
            display(trajectory_frame)
            print(f"Observed attempts: {[item.step_id for item in result.observations]}")
            """,
        ),
        _markdown(
            "lesson11-009",
            """
            ## Failure lab

            The server returns `unsupported_metric` for step 3. This is evidence about the strategy, not something to erase. The successful metric observations stay immutable, and only unfinished work may change.
            """,
        ),
        _code(
            "lesson11-010",
            """
            failure = next(item for item in result.observations if item.status == "error")
            display(pd.DataFrame([{
                "attempt": failure.attempt_id,
                "step": failure.step_id,
                "capability": failure.capability,
                "status": failure.status,
                "typed error": failure.error_code,
            }]))

            decision = ReplanDecision(
                action="replace_remaining",
                reasoning="Use document search because Revenue is not a supported metric.",
                replacement_steps=tuple(result.final_steps[-2:]),
            )
            print("Learner decision: replan the unfinished tail")
            print(f"System decision: {decision.action}")
            print("Successful calls repeated: 0")
            """,
        ),
        _code(
            "lesson11-011",
            """
            story = pd.DataFrame([
                ("1", "NVIDIA P/E", "kept"),
                ("2", "Schneider P/E", "kept"),
                ("3", "Revenue metric", "failed + retained"),
                ("4", "Original document tail", "superseded"),
                ("5", "NVIDIA document", "replacement"),
                ("6", "Schneider document", "replacement"),
            ], columns=["step", "work", "outcome"])
            palette = {"kept": "#2E8B57", "failed + retained": "#F47C00", "superseded": "#A9B1BC", "replacement": "#2447D8"}
            fig, ax = plt.subplots(figsize=(12, 4.5))
            ax.scatter(story["step"], [1.08] * len(story), s=1800, c=[palette[value] for value in story["outcome"]])
            for _, row in story.iterrows():
                ax.text(row["step"], 1.08, row["step"], ha="center", va="center", color="white", weight="bold", fontsize=12)
                ax.text(row["step"], 0.80, row["outcome"], ha="center", va="center", color="#051C2A", weight="bold", fontsize=9)
            ax.set_ylim(0.60, 1.40)
            ax.set_yticks([])
            ax.set_title("Figure 2. Replanning keeps completed work and replaces only the tail", loc="left", weight="bold")
            ax.spines[:].set_visible(False)
            plt.tight_layout()
            plt.show()
            """,
        ),
        _markdown(
            "lesson11-012",
            """
            ## Evidence gate and cited briefing

            Reporting is permitted only when both companies have sourced metric evidence and sourced document evidence. A fluent paragraph cannot compensate for a missing row.
            """,
        ),
        _code(
            "lesson11-013",
            """
            print(f"Evidence gate passed: {result.evidence_gate.passed}")
            display(pd.DataFrame([
                (company, ", ".join(evidence))
                for company, evidence in result.evidence_gate.coverage.items()
            ], columns=["company", "verified evidence"]))

            assert result.briefing is not None
            print("\\nCITED FACTS")
            for index, fact in enumerate(result.briefing.reported_facts, start=1):
                print(f"{index}. {fact.claim}")
                print(f"   Kind: {fact.provenance_kind}")
                print(f"   Sources: {', '.join(fact.source_references)}")
                print(f"   Evidence IDs: {', '.join(fact.evidence_ids) or 'none (metric observation)'}")

            for heading, items in (
                ("CROSS-COMPANY OBSERVATIONS", result.briefing.cross_company_observations),
                ("INTERPRETATION", result.briefing.interpretation),
                ("LIMITATIONS", result.briefing.limitations),
            ):
                print(f"\\n{heading}")
                for item in items:
                    print(f"- {item}")

            print("\\nAggregate sources:")
            for source in result.briefing.source_references:
                print(f"- {source}")
            """,
        ),
        _code(
            "lesson11-014",
            """
            companies = ["NVIDIA", "Schneider Electric"]
            evidence_types = ["metric", "document"]
            coverage = result.evidence_gate.coverage
            matrix = [[int(kind in coverage.get(company, ())) for kind in evidence_types] for company in companies]
            fig, ax = plt.subplots(figsize=(8.5, 4.3))
            image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
            for row, company in enumerate(companies):
                for column, kind in enumerate(evidence_types):
                    ax.text(column, row, "READY" if matrix[row][column] else "MISSING", ha="center", va="center", color="white" if matrix[row][column] else "#051C2A", weight="bold")
            ax.set_xticks(range(2), evidence_types)
            ax.set_yticks(range(2), companies)
            ax.set_title("Figure 3. The evidence matrix decides whether writing may start", loc="left", weight="bold")
            fig.colorbar(image, ax=ax, ticks=[0, 1], label="evidence present")
            plt.tight_layout()
            plt.show()
            """,
        ),
        _markdown(
            "lesson11-015",
            """
            ## Optional live route and Lesson 12 handoff

            Set `FINAI_LIVE_MODE=1` with `FINAI_MODEL_PROVIDER=openai` or `FINAI_MODEL_PROVIDER=ollama`. OpenAI and Ollama may propose the plan, replan, and briefing, but the same Python validation, MCP lifecycle, budgets, and evidence gate remain in force.

            Lesson 12 evaluates answer quality and trajectory quality separately.
            """,
        ),
        _code(
            "lesson11-016",
            """
            ## Verification
            if not LIVE_MODE:
                assert result.status == "completed"
                assert result.replan_count == 1
                assert result.evidence_gate.passed is True
                assert [item.step_id for item in result.observations] == [1, 2, 3, 5, 6]
                assert sum(item.error_code == "unsupported_metric" for item in result.observations) == 1
                print("Plan revisions: 1")
                print("LESSON_11_PASS")
            else:
                print("Live route complete. Review the model-produced plan and briefing before acceptance.")

            """,
        ),
        _markdown(
            "lesson11-017",
            """
            ## Verification

            The offline route must retain one typed failure, make one revision, pass the evidence gate, and print `LESSON_11_PASS`.

            ## Knowledge check

            1. Who approves a plan? **The host.**
            2. What does replanning replace? **Only unfinished work.**
            3. What blocks an incomplete briefing? **The evidence gate.**

            ## Challenge

            Change one allowed document query. Predict which evidence row changes, then rerun from the approved plan. Keep the six-step and one-replan budgets.

            ## Capstone integration

            The Financial Analyst Copilot now produces an inspectable trajectory: mission, catalog, approved plan, observations, replan, evidence gate, and cited briefing.

            ## Recap

            Plan before acting. Preserve failures. Revise only unfinished work. Write only from verified evidence.
            """,
        ),
    ]
    return notebook


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(build_notebook(), OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
