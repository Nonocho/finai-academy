"""Build the canonical output-free Lesson 08 notebook."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/08_workflows_vs_agents.ipynb"


def _markdown(cell_id: str, source: str):
    cell = nbformat.v4.new_markdown_cell(dedent(source).strip())
    cell.id = cell_id
    return cell


def _code(cell_id: str, source: str):
    cell = nbformat.v4.new_code_cell(dedent(source).strip())
    cell.id = cell_id
    return cell


def build_notebook():
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.kernelspec = {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata.language_info = {"name": "python", "version": "3.11"}
    notebook.metadata.finai = {"expected_runtime_minutes": 30, "lesson": "08"}
    notebook.cells = [
        _markdown(
            "lesson08-000",
            """
            # 08 — Workflows versus agents

            **First Finance - Arnaud Demes**  
            **Day 2 · 09:30–10:15 · 10 minutes concepts + 30 minutes notebook + 5 minutes debrief**

            **Engineering question:** when is a deterministic workflow sufficient, and when must the next action depend on an intermediate observation?

            This notebook uses NVIDIA (`NVDA`) and Schneider Electric (`SU.PA`) as educational examples. Market observations are versioned course snapshots, not live quotes or investment advice.
            """,
        ),
        _markdown(
            "lesson08-001",
            """
            ## Learning objectives

            By the end, you can:

            1. distinguish a function, workflow, bounded agent, and multi-agent system;
            2. inspect typed tool requests and observations;
            3. expose a dependency a one-pass workflow cannot resolve;
            4. run a visible reason–act–observe–stop loop with `MAX_STEPS`;
            5. compare trajectory, latency, and failure surface; and
            6. choose the lowest useful autonomy for an analyst task.

            **Expected visible result:** the one-pass workflow succeeds on a direct price request, returns `unsupported_dependency` for a price-to-EUR request, and the bounded agent completes the same compound request by calling `get_market_price` before `convert_currency`.
            """,
        ),
        _markdown(
            "lesson08-002",
            """
            ## Where this fits

            Day 1 produced an evaluated financial RAG pipeline. Day 2 now adds controlled autonomy:

            ```text
            deterministic workflow → bounded agent → self-correction → MCP → planning → evaluation
            ```

            Lesson 08 keeps the loop in plain Python. Lesson 09 will introduce LangGraph only when explicit state and recovery routing earn their complexity.

            Set `FINAI_LIVE_MODE=1` through the course executor to run the same lab with Ollama or OpenAI. The default offline fixture is only the deterministic test and classroom-recovery path.
            """,
        ),
        _code(
            "lesson08-003",
            """
            import json
            import os
            from pathlib import Path

            import matplotlib.pyplot as plt
            import pandas as pd
            from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

            from finai_academy.agent_workflows import (
                AgentDecision,
                ToolRequest,
                WorkflowPlan,
                build_course_tool_registry,
                load_course_market_snapshot,
                run_bounded_agent,
                run_one_pass_workflow,
            )
            from finai_academy.lesson_support import RecordedLesson08Model
            from finai_academy.providers import create_chat_model, provider_summary
            from finai_academy.settings import Settings

            PROJECT_ROOT = Path.cwd().resolve()
            if PROJECT_ROOT.name == "notebooks":
                PROJECT_ROOT = PROJECT_ROOT.parent

            LIVE_MODE = os.getenv("FINAI_LIVE_MODE", "0") == "1"
            settings = Settings.from_environment()
            runtime_label = (
                f"{settings.provider} · {settings.chat_model}"
                if LIVE_MODE
                else "offline fixture · deterministic course run"
            )
            snapshot_path = (
                PROJECT_ROOT
                / "assets/course-data/market/lesson08_market_snapshot_v1.json"
            )
            snapshot = load_course_market_snapshot(snapshot_path)
            registry = build_course_tool_registry(snapshot)
            print(f"Runtime: {runtime_label}")
            print(f"Dataset: {snapshot['dataset_id']}")
            print(f"Tools: {', '.join(registry.names)}")
            """,
        ),
        _markdown(
            "lesson08-004",
            """
            ### The autonomy spectrum is a design choice

            Autonomy is not a maturity score. Moving right adds flexibility, but also latency, nondeterminism, testing work, and a larger failure surface.
            """,
        ),
        _code(
            "lesson08-005",
            """
            labels = ["Function", "Workflow", "Bounded agent", "Multi-agent"]
            autonomy = [0.08, 0.32, 0.70, 0.94]
            predictability = [0.96, 0.82, 0.48, 0.24]

            fig, ax = plt.subplots(figsize=(10, 4.6))
            ax.plot(autonomy, predictability, color="#1F40CB", linewidth=3)
            for index, label in enumerate(labels):
                ax.scatter(autonomy[index], predictability[index], s=180, color="#00A2EB", zorder=3)
                ax.annotate(label, (autonomy[index], predictability[index]), xytext=(0, 14), textcoords="offset points", ha="center", fontsize=11, weight="bold")
            ax.set(xlim=(0, 1), ylim=(0, 1.08), xlabel="Autonomy and dynamic choice →", ylabel="Predictability and fixed control →")
            ax.set_title(f"Choose the lowest useful autonomy · {runtime_label}", loc="left", weight="bold")
            ax.grid(alpha=0.2)
            plt.tight_layout()
            plt.show()
            """,
        ),
        _markdown(
            "lesson08-006",
            """
            ### Typed tools are application boundaries

            Both architectures use the same tools:

            - `get_market_price(ticker)` returns company, price, currency, date, and source;
            - `convert_currency(amount, from_currency, to_currency)` returns the calculation inputs, rate, date, and source.

            The model may request a tool. Only deterministic Python executes it and creates a market-data observation.
            """,
        ),
        _code(
            "lesson08-007",
            """
            snapshot_rows = [
                {
                    "instrument": ticker,
                    "company": record["company"],
                    "value": record["price"],
                    "currency": record["currency"],
                    "as_of": record["as_of"],
                    "provenance": "checked-in Yahoo Finance snapshot",
                }
                for ticker, record in snapshot["prices"].items()
            ]
            snapshot_rows.append(
                {
                    "instrument": "USD_EUR",
                    "company": "FX reference",
                    "value": snapshot["fx"]["USD_EUR"]["rate"],
                    "currency": "EUR per USD",
                    "as_of": snapshot["fx"]["USD_EUR"]["as_of"],
                    "provenance": "checked-in Yahoo Finance snapshot",
                }
            )
            pd.DataFrame(snapshot_rows)
            """,
        ),
        _markdown(
            "lesson08-008",
            """
            ### One-pass workflow

            The workflow chooses its route before it has any observation. This is appropriate for a stable direct lookup. A new dependency shape needs a new coded branch.
            """,
        ),
        _code(
            "lesson08-009",
            """
            fig, ax = plt.subplots(figsize=(10, 3.4))
            ax.axis("off")
            nodes = [(0.04, "Question"), (0.29, "Plan once"), (0.54, "One tool"), (0.79, "Final answer")]
            for x, text in nodes:
                box = FancyBboxPatch((x, 0.36), 0.17, 0.28, boxstyle="round,pad=0.02", facecolor="#F5F5F5", edgecolor="#1F40CB", linewidth=2)
                ax.add_patch(box)
                ax.text(x + 0.085, 0.50, text, ha="center", va="center", weight="bold")
            for index in range(len(nodes) - 1):
                left, right = nodes[index], nodes[index + 1]
                ax.add_patch(FancyArrowPatch((left[0] + 0.17, 0.50), (right[0], 0.50), arrowstyle="-|>", mutation_scale=16, color="#00A2EB", linewidth=2))
            ax.text(0.5, 0.12, "No edge returns from the observation to planning", ha="center", color="#F07D00", weight="bold")
            ax.set_title(f"The workflow fixes its path before tool output · {runtime_label}", loc="left", weight="bold")
            plt.show()
            """,
        ),
        _code(
            "lesson08-010",
            """
            class LiveLesson08Model:
                # Visible structured actions only; no hidden chain-of-thought is requested.

                mode = "live model"

                def __init__(self, chat_model):
                    self.chat_model = chat_model
                    self.workflow_planner = chat_model.with_structured_output(WorkflowPlan)
                    self.agent_policy = chat_model.with_structured_output(AgentDecision)

                def plan_workflow(self, question):
                    prompt = f'''You route a one-pass financial workflow.
            It may execute at most one tool before a non-tool answer writer runs.
            Valid tools:
            - get_market_price(ticker)
            - convert_currency(amount, from_currency, to_currency)
            If a requested conversion amount depends on an unseen price result, return
            route='unsupported_dependency'. Never invent the amount.
            Question: {question}'''
                    return self.workflow_planner.invoke([("human", prompt)])

                def write_workflow_answer(self, question, observations):
                    prompt = f'''Answer using only this typed tool observation.
            Include value, currency, date and source. Do not add investment advice.
            Question: {question}
            Observation: {json.dumps([item.model_dump(mode='json') for item in observations])}'''
                    response = self.chat_model.invoke([("human", prompt)])
                    return response.content if isinstance(response.content, str) else str(response.content)

                def decide_agent(self, question, trajectory):
                    remaining = MAX_STEPS - sum(step.phase == "plan" for step in trajectory)
                    visible_trace = [step.model_dump(mode="json") for step in trajectory]
                    prompt = f'''Return only the next typed action for a bounded financial agent.
            Valid tools:
            - get_market_price(ticker)
            - convert_currency(amount, from_currency, to_currency)
            Use only values from successful observations. For a price converted to EUR:
            first call get_market_price; then pass its observed price and currency to
            convert_currency; then finish with value, rate, dates and sources.
            Remaining model steps: {remaining}
            Question: {question}
            Visible trajectory: {json.dumps(visible_trace)}'''
                    return self.agent_policy.invoke([("human", prompt)])


            MAX_STEPS = 4
            if LIVE_MODE:
                chat_model = create_chat_model(settings)
                lesson_model = LiveLesson08Model(chat_model)
                print("Live provider:", provider_summary(settings))
            else:
                lesson_model = RecordedLesson08Model()
                print("Policy: offline fixture (labelled deterministic fallback)")
            """,
        ),
        _code(
            "lesson08-011",
            """
            direct_question = "What is NVIDIA's latest available share price?"
            workflow_direct = run_one_pass_workflow(
                direct_question,
                planner=lesson_model.plan_workflow,
                answer_writer=lesson_model.write_workflow_answer,
                registry=registry,
            )
            print(f"workflow_direct_status={workflow_direct.status}")
            print(workflow_direct.answer)
            """,
        ),
        _markdown(
            "lesson08-012",
            """
            ## Failure lab

            Ask for NVIDIA's price in euros. The conversion amount does not exist until the price tool returns. The one-pass workflow must expose `unsupported_dependency`; a fabricated EUR value would be a grounding failure.

            A developer could add an explicit two-step conversion branch. The lesson does **not** claim workflows cannot chain operations. It shows that each new dependency needs another predefined route.
            """,
        ),
        _code(
            "lesson08-013",
            """
            compound_question = "What is NVIDIA's latest available share price converted to euros?"
            workflow_dependency = run_one_pass_workflow(
                compound_question,
                planner=lesson_model.plan_workflow,
                answer_writer=lesson_model.write_workflow_answer,
                registry=registry,
            )
            print(f"workflow_dependency_status={workflow_dependency.status}")
            print(workflow_dependency.trajectory[0].summary)

            fig, ax = plt.subplots(figsize=(10, 3.4))
            ax.axis("off")
            ax.add_patch(FancyBboxPatch((0.05, 0.38), 0.24, 0.26, boxstyle="round,pad=0.02", facecolor="#F5F5F5", edgecolor="#1F40CB", linewidth=2))
            ax.text(0.17, 0.51, "Need NVDA price", ha="center", va="center", weight="bold")
            ax.add_patch(FancyArrowPatch((0.29, 0.51), (0.44, 0.51), arrowstyle="-|>", mutation_scale=16, color="#00A2EB", linewidth=2))
            ax.add_patch(FancyBboxPatch((0.44, 0.38), 0.22, 0.26, boxstyle="round,pad=0.02", facecolor="#FFF2E5", edgecolor="#F07D00", linewidth=2))
            ax.text(0.55, 0.51, "Amount unknown", ha="center", va="center", weight="bold")
            ax.add_patch(FancyArrowPatch((0.66, 0.51), (0.79, 0.51), arrowstyle="-|>", mutation_scale=16, color="#F07D00", linewidth=2))
            ax.text(0.87, 0.51, "STOP", ha="center", va="center", color="#F07D00", fontsize=16, weight="bold")
            ax.text(0.5, 0.17, workflow_dependency.status, ha="center", color="#F07D00", weight="bold")
            ax.set_title(f"Expose the dependency; never invent the conversion input · {runtime_label}", loc="left", weight="bold")
            plt.show()
            """,
        ),
        _markdown(
            "lesson08-014",
            """
            ### Bounded agent loop

            The agent may choose another tool after observing the previous result. Application code still owns validation, execution, trace recording, and the stop budget.

            ```text
            visible state → typed action → validated tool → typed observation → visible state
                                  ↘ finish or MAX_STEPS stop ↗
            ```
            """,
        ),
        _code(
            "lesson08-015",
            """
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.axis("off")
            positions = {
                "Visible state": (0.08, 0.63),
                "Typed action": (0.39, 0.63),
                "Tool observation": (0.70, 0.63),
                "Finish / guardrail": (0.39, 0.18),
            }
            for label, (x, y) in positions.items():
                edge = "#F07D00" if "guardrail" in label else "#1F40CB"
                face = "#FFF2E5" if "guardrail" in label else "#F5F5F5"
                ax.add_patch(FancyBboxPatch((x, y), 0.22, 0.18, boxstyle="round,pad=0.02", facecolor=face, edgecolor=edge, linewidth=2))
                ax.text(x + 0.11, y + 0.09, label, ha="center", va="center", weight="bold")
            forward_arrows = [
                ((0.30, 0.72), (0.39, 0.72)),
                ((0.61, 0.72), (0.70, 0.72)),
                ((0.50, 0.63), (0.50, 0.36)),
            ]
            for start, end in forward_arrows:
                ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, color="#00A2EB", linewidth=2))
            ax.add_patch(
                FancyArrowPatch(
                    (0.81, 0.83),
                    (0.19, 0.83),
                    arrowstyle="-|>",
                    mutation_scale=16,
                    color="#00A2EB",
                    linewidth=2,
                    connectionstyle="arc3,rad=0.30",
                )
            )
            ax.text(0.50, 0.08, f"Hard boundary: MAX_STEPS={MAX_STEPS}", ha="center", color="#F07D00", weight="bold")
            ax.set_title(f"Autonomy stays inside an inspectable budget · {runtime_label}", loc="left", weight="bold")
            plt.show()
            """,
        ),
        _code(
            "lesson08-016",
            """
            agent_result = run_bounded_agent(
                compound_question,
                policy=lesson_model.decide_agent,
                registry=registry,
                max_steps=MAX_STEPS,
            )
            agent_tool_order = [
                step.tool_name for step in agent_result.trajectory if step.phase == "tool"
            ]
            print(f"agent_status={agent_result.status}")
            print("agent_tool_order=" + " -> ".join(agent_tool_order))
            print(agent_result.answer)
            """,
        ),
        _code(
            "lesson08-017",
            """
            trace_rows = []
            for architecture, result in [
                ("one-pass workflow", workflow_dependency),
                ("bounded agent", agent_result),
            ]:
                for step in result.trajectory:
                    trace_rows.append(
                        {
                            "architecture": architecture,
                            "event": step.index,
                            "phase": step.phase,
                            "tool": step.tool_name or "—",
                            "status": step.observation.status if step.observation else result.status,
                            "summary": step.summary,
                        }
                    )
            trace_frame = pd.DataFrame(trace_rows)
            display(trace_frame)

            phase_colors = {"plan": "#1F40CB", "tool": "#00A2EB", "finish": "#2E8B57", "guardrail": "#F07D00"}
            fig, ax = plt.subplots(figsize=(11, 4.5))
            y_positions = {"one-pass workflow": 1, "bounded agent": 0}
            for _, row in trace_frame.iterrows():
                y = y_positions[row["architecture"]]
                ax.scatter(row["event"], y, s=240, color=phase_colors[row["phase"]], zorder=3)
                label = row["tool"] if row["tool"] != "—" else row["phase"]
                ax.annotate(
                    label,
                    (row["event"], y),
                    xytext=(0, 20 if y else -28),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if y else "top",
                    fontsize=8.5,
                    rotation=0 if y else 18,
                )
            ax.set_yticks([0, 1], ["bounded agent", "one-pass workflow"])
            ax.set_ylim(-0.45, 1.45)
            ax.set_xticks(range(1, int(trace_frame["event"].max()) + 1))
            ax.set_xlabel("Recorded event order")
            ax.set_title(f"The trajectory shows why the agent succeeds · {runtime_label}", loc="left", weight="bold")
            ax.grid(axis="x", alpha=0.2)
            plt.tight_layout()
            plt.show()
            """,
        ),
        _code(
            "lesson08-018",
            """
            comparison = pd.DataFrame(
                [
                    {
                        "architecture": "one-pass workflow",
                        "status": workflow_dependency.status,
                        "tool_calls": sum(step.phase == "tool" for step in workflow_dependency.trajectory),
                        "events": len(workflow_dependency.trajectory),
                        "latency_ms": round(workflow_dependency.latency_ms, 2),
                        "grounded_compound_answer": bool(workflow_dependency.answer),
                    },
                    {
                        "architecture": "bounded agent",
                        "status": agent_result.status,
                        "tool_calls": sum(step.phase == "tool" for step in agent_result.trajectory),
                        "events": len(agent_result.trajectory),
                        "latency_ms": round(agent_result.latency_ms, 2),
                        "grounded_compound_answer": agent_result.status == "completed",
                    },
                ]
            )
            comparison
            """,
        ),
        _markdown(
            "lesson08-019",
            """
            ### Failure lab — a budget is behavior, not configuration decoration

            A policy that always asks for another lookup must stop. This controlled failure proves that the application, not the model, owns termination.
            """,
        ),
        _code(
            "lesson08-020",
            """
            def looping_policy(_question, _trajectory):
                return AgentDecision(
                    action="tool",
                    request=ToolRequest(
                        name="get_market_price", arguments={"ticker": "NVDA"}
                    ),
                )

            budget_result = run_bounded_agent(
                "Repeat the lookup indefinitely.",
                policy=looping_policy,
                registry=registry,
                max_steps=2,
            )
            budget_counts = pd.Series(
                [step.phase for step in budget_result.trajectory]
            ).value_counts()
            fig, ax = plt.subplots(figsize=(8, 3.8))
            budget_counts.reindex(["plan", "tool", "guardrail"], fill_value=0).plot(
                kind="bar", ax=ax, color=["#1F40CB", "#00A2EB", "#F07D00"]
            )
            ax.set_title(f"The application stops the loop · {budget_result.status}", loc="left", weight="bold")
            ax.set(xlabel="Recorded phase", ylabel="Event count")
            ax.tick_params(axis="x", rotation=0)
            plt.tight_layout()
            plt.show()
            print(budget_result.trajectory[-1].summary)
            """,
        ),
        _markdown(
            "lesson08-021",
            """
            ## Verification

            A successful lesson run must prove architecture behavior, not merely display a plausible sentence:

            - direct workflow lookup completes;
            - compound workflow stops at `unsupported_dependency`;
            - agent calls price before conversion;
            - conversion inputs equal observed price metadata;
            - loop guardrail stops at `MAX_STEPS`; and
            - all numeric observations retain date, currency, and source.
            """,
        ),
        _code(
            "lesson08-022",
            """
            assert workflow_direct.status == "completed"
            assert workflow_dependency.status == "unsupported_dependency"
            assert workflow_dependency.answer is None
            assert agent_result.status == "completed"
            assert agent_tool_order == ["get_market_price", "convert_currency"]
            price_observation = next(
                step.observation
                for step in agent_result.trajectory
                if step.tool_name == "get_market_price"
            )
            conversion_observation = next(
                step.observation
                for step in agent_result.trajectory
                if step.tool_name == "convert_currency"
            )
            assert price_observation is not None and price_observation.status == "ok"
            assert conversion_observation is not None and conversion_observation.status == "ok"
            assert conversion_observation.payload["input_amount"] == price_observation.payload["price"]
            assert budget_result.status == "step_budget_exhausted"
            assert budget_result.trajectory[-1].phase == "guardrail"
            print("LESSON_08_PASS")
            """,
        ),
        _markdown(
            "lesson08-023",
            """
            ### Knowledge check

            1. Why is the one-pass workflow safer than inventing the missing EUR amount?
            2. When would adding a deterministic conversion branch be better than retaining the agent?
            3. Which component validates and executes a tool request?
            4. What evidence proves the converted answer is grounded?
            5. What does `MAX_STEPS` protect against?

            Answers: expose unsupported dependencies; prefer a workflow for stable known routes; Python owns execution; require successful price and conversion observations; prevent unbounded latency, cost, and looping.
            """,
        ),
        _markdown(
            "lesson08-024",
            """
            ## Challenge

            Choose one bounded change:

            1. add a deterministic two-step currency-conversion branch and compare it with the agent; or
            2. add `calculate_return` and make the agent compare one maintained NVIDIA or Schneider Electric period.

            Record which implementation is easier to test and whether the added autonomy earns its latency and failure surface.
            """,
        ),
        _markdown(
            "lesson08-025",
            """
            ## Capstone integration

            Lesson 08 contributes three reusable pieces to the Financial Analyst Copilot:

            - typed tool requests and observations;
            - one normalized trajectory record; and
            - a minimal bounded agent loop.

            Lesson 09 keeps these contracts, adds LangGraph state, feeds structured tool errors back to the model, and demonstrates controlled self-correction.
            """,
        ),
        _markdown(
            "lesson08-026",
            """
            ## Recap

            - Workflows are preferred when routes and dependencies are known and stable.
            - Agents earn their complexity when observations determine an open-ended next action.
            - Tool execution remains deterministic and typed.
            - A professional agent is bounded, inspectable, and grounded in observations.
            - When an agent pattern stabilizes, consider replacing it with a workflow.
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
