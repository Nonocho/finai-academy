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
            # 08 — Who chooses the next step?

            **First Finance - Arnaud Demes**  
            **Day 2 · 09:30–10:15 · 10 minutes concepts + 30 minutes notebook + 5 minutes debrief**

            ## Learning objectives

            By the end, you can:

            1. distinguish a workflow from an agent by **who controls execution**;
            2. build direct and deterministic two-step workflows with typed tools;
            3. inspect a bounded model-directed loop;
            4. compare both designs on the same task; and
            5. use an agent only when model-directed control flow creates measurable value.

            ## Where this fits

            Day 1 built an evaluated RAG pipeline. Lesson 08 adds controlled autonomy in plain Python. Lesson 09 adds explicit state and self-correction. Run with OpenAI or Ollama in live mode; the labelled offline fixture is the classroom recovery path. The market data below is a versioned course snapshot, not a live quote or investment advice.
            """,
        ),
        _code(
            "lesson08-001",
            """
            import json
            import os
            from pathlib import Path

            import matplotlib.pyplot as plt
            import pandas as pd
            from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

            from finai_academy.agent_workflows import (
                AgentDecision,
                ModelAgentDecision,
                ModelWorkflowDecision,
                ToolRequest,
                build_course_tool_registry,
                load_course_market_snapshot,
                run_bounded_agent,
                run_one_pass_workflow,
                run_price_to_currency_workflow,
            )
            from finai_academy.lesson_support import RecordedLesson08Model
            from finai_academy.providers import create_chat_model, provider_summary
            from finai_academy.settings import Settings

            PROJECT_ROOT = Path.cwd().resolve()
            if PROJECT_ROOT.name == "notebooks":
                PROJECT_ROOT = PROJECT_ROOT.parent
            LIVE_MODE = os.getenv("FINAI_LIVE_MODE", "0") == "1"
            settings = Settings.from_environment()
            runtime_label = f"{settings.provider} · {settings.chat_model}" if LIVE_MODE else "offline fixture"
            snapshot = load_course_market_snapshot(
                PROJECT_ROOT / "assets/course-data/market/lesson08_market_snapshot_v1.json"
            )
            registry = build_course_tool_registry(snapshot)
            MAX_STEPS = 4
            print(f"Runtime: {runtime_label}")
            print(f"Dataset: {snapshot['dataset_id']}")
            print(f"Tools: {', '.join(registry.names)}")
            """,
        ),
        _markdown(
            "lesson08-002",
            """
            ## The distinction is control flow, not tool count

            Anthropic defines workflows as systems where LLMs and tools follow predefined code paths, while agents dynamically direct their own process and tool use. LangGraph makes the same distinction: workflows can chain, branch, and loop; an agent uses the model to decide what to do next.

            | Design | Who chooses the next action? | Best fit |
            |---|---|---|
            | Function | Python | One deterministic operation |
            | Workflow | Python | Known sequence, branch, or loop |
            | Agent | Model inside application limits | Path cannot be fully specified in advance |

            **Critical correction:** a tool result may feed the next workflow step. That does not make the system an agent.
            """,
        ),
        _code(
            "lesson08-003",
            """
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
            for ax in axes:
                ax.axis("off")

            for x, label in [(0.03, "Code"), (0.38, "Tool A"), (0.73, "Tool B")]:
                axes[0].add_patch(FancyBboxPatch((x, .43), .23, .22, boxstyle="round,pad=.02", facecolor="#F4F7FF", edgecolor="#1F40CB", linewidth=2))
                axes[0].text(x + .115, .54, label, ha="center", va="center", weight="bold")
            for x in (.26, .61):
                axes[0].add_patch(FancyArrowPatch((x, .54), (x + .11, .54), arrowstyle="-|>", mutation_scale=16, color="#00A2EB", linewidth=2))
            axes[0].set_title("WORKFLOW · code owns the route", loc="left", weight="bold")

            for x, y, label in [(0.08, .57, "Model\\nchooses"), (.62, .57, "Tool"), (.35, .12, "Finish / stop")]:
                edge = "#F07D00" if "stop" in label else "#1F40CB"
                axes[1].add_patch(FancyBboxPatch((x, y), .27, .20, boxstyle="round,pad=.02", facecolor="#F4F7FF", edgecolor=edge, linewidth=2))
                axes[1].text(x + .135, y + .10, label, ha="center", va="center", weight="bold")
            axes[1].add_patch(FancyArrowPatch((.35, .67), (.61, .67), arrowstyle="-|>", mutation_scale=16, color="#00A2EB", linewidth=2))
            axes[1].add_patch(FancyArrowPatch((.75, .78), (.22, .78), connectionstyle="arc3,rad=.3", arrowstyle="-|>", mutation_scale=16, color="#00A2EB", linewidth=2))
            axes[1].add_patch(FancyArrowPatch((.35, .58), (.47, .34), arrowstyle="-|>", mutation_scale=16, color="#F07D00", linewidth=2))
            axes[1].set_title("AGENT · model owns the next choice", loc="left", weight="bold")
            fig.suptitle("Same tools. Different controller.", x=.04, ha="left", fontsize=16, weight="bold")
            plt.tight_layout()
            plt.show()
            """,
        ),
        _markdown(
            "lesson08-004",
            """
            ## Typed tools keep both designs grounded

            `get_market_price(ticker)` returns price, currency, date, and source. `convert_currency(amount, from_currency, to_currency)` returns the calculation inputs, rate, date, and source. The model may request an action; only deterministic Python validates and executes it.
            """,
        ),
        _code(
            "lesson08-005",
            """
            rows = [
                {"instrument": ticker, "value": item["price"], "unit": item["currency"], "as_of": item["as_of"]}
                for ticker, item in snapshot["prices"].items()
            ]
            rows.append({"instrument": "USD_EUR", "value": snapshot["fx"]["USD_EUR"]["rate"], "unit": "EUR per USD", "as_of": snapshot["fx"]["USD_EUR"]["as_of"]})
            display(pd.DataFrame(rows))
            """,
        ),
        _markdown(
            "lesson08-006",
            """
            ## 1 · Direct lookup: one-step workflow wins

            The route is known: parse the request, call one typed tool, format the observed value. An agent loop would add model calls without adding a useful decision.
            """,
        ),
        _code(
            "lesson08-007",
            """
            class LiveLesson08Model:
                mode = "live model"

                def __init__(self, chat_model):
                    self.workflow_planner = chat_model.with_structured_output(ModelWorkflowDecision)
                    self.agent_policy = chat_model.with_structured_output(ModelAgentDecision)

                def plan_workflow(self, question):
                    prompt = f'''Route this direct financial lookup. Select get_market_price for NVIDIA/NVDA.
            Every schema field is required: use null for fields that do not apply. Never invent values.
            Question: {question}'''
                    return self.workflow_planner.invoke([("human", prompt)]).to_workflow_plan()

                def decide_agent(self, question, trajectory):
                    visible = [step.model_dump(mode="json") for step in trajectory]
                    prompt = f'''Choose one next action for a bounded financial agent.
            Tools: get_market_price(ticker); convert_currency(amount, from_currency, to_currency).
            For price in EUR: observe price first, pass that exact price and currency to conversion, then finish.
            Use only successful observations. Every schema field is required; use null when irrelevant.
            Question: {question}\\nVisible trajectory: {json.dumps(visible)}'''
                    return self.agent_policy.invoke([("human", prompt)]).to_agent_decision()

            lesson_model = LiveLesson08Model(create_chat_model(settings)) if LIVE_MODE else RecordedLesson08Model()
            print("Live provider:", provider_summary(settings)) if LIVE_MODE else print("Policy: offline fixture (deterministic classroom fallback)")

            def write_price_answer(_question, observations):
                item = observations[0].payload
                return f"{item['company']}: {item['price']:.2f} {item['currency']} as of {item['as_of']} [{item['source']}]."

            direct_question = "What is NVIDIA's latest available share price?"
            workflow_direct = run_one_pass_workflow(direct_question, planner=lesson_model.plan_workflow, answer_writer=write_price_answer, registry=registry)
            print(f"workflow_direct_status={workflow_direct.status}")
            print(workflow_direct.answer)

            fig, ax = plt.subplots(figsize=(10, 2.8)); ax.axis("off")
            for x, label in [(0.04, "Question"), (.38, "get_market_price"), (.72, "Grounded answer")]:
                ax.add_patch(FancyBboxPatch((x, .35), .24, .28, boxstyle="round,pad=.02", facecolor="#F4F7FF", edgecolor="#1F40CB", linewidth=2)); ax.text(x+.12, .49, label, ha="center", va="center", weight="bold")
            for x in (.28, .62):
                ax.add_patch(FancyArrowPatch((x, .49), (x+.09, .49), arrowstyle="-|>", mutation_scale=16, color="#00A2EB", linewidth=2))
            ax.set_title("Known path: keep control in code", loc="left", weight="bold")
            plt.show()
            """,
        ),
        _markdown(
            "lesson08-008",
            """
            ## 2 · Price → FX: a two-step workflow still wins

            The second call depends on the first result, but the dependency is known before execution: `price → conversion`. Python passes the observed price and currency forward. This is a deterministic two-step workflow—not an agent.
            """,
        ),
        _code(
            "lesson08-009",
            """
            compound_question = "What is NVIDIA's latest available share price converted to euros?"

            def write_conversion_answer(_question, observations):
                price, conversion = (item.payload for item in observations)
                return (f"{price['company']}: EUR {conversion['output_amount']:.2f}; "
                        f"source price {price['price']:.2f} {price['currency']} as of {price['as_of']}; "
                        f"FX {conversion['rate']:.4f} as of {conversion['rate_as_of']} [{conversion['source']}].")

            workflow_compound = run_price_to_currency_workflow(compound_question, ticker="NVDA", target_currency="EUR", answer_writer=write_conversion_answer, registry=registry)
            workflow_tools = [step.tool_name for step in workflow_compound.trajectory if step.phase == "tool"]
            print(f"workflow_compound_status={workflow_compound.status}")
            print("workflow_tool_order=" + " -> ".join(workflow_tools))
            print(workflow_compound.answer)

            fig, ax = plt.subplots(figsize=(11, 3)); ax.axis("off")
            for x, label in [(0.02, "Code route"), (.27, "Price\\n180 USD"), (.52, "Pass observed\\n180 USD"), (.77, "FX\\n154.80 EUR")]:
                ax.add_patch(FancyBboxPatch((x, .35), .19, .30, boxstyle="round,pad=.02", facecolor="#F4F7FF", edgecolor="#1F40CB", linewidth=2)); ax.text(x+.095, .50, label, ha="center", va="center", weight="bold")
            for x in (.21, .46, .71):
                ax.add_patch(FancyArrowPatch((x, .50), (x+.05, .50), arrowstyle="-|>", mutation_scale=16, color="#00A2EB", linewidth=2))
            ax.set_title("Workflows can use tool results", loc="left", weight="bold")
            ax.text(.5, .12, "The route is fixed; the data moves between steps.", ha="center", color="#1F40CB", weight="bold")
            plt.show()
            """,
        ),
        _markdown(
            "lesson08-010",
            """
            ## 3 · Bounded agent: the model chooses after each observation

            The loop is useful when the next action cannot be fully specified in advance—for example, investigating a reconciliation exception where each observation changes which evidence should be checked next. Application code still owns schemas, tool execution, grounding checks, and `MAX_STEPS`.

            Here we deliberately run an agent on the same fixed price-to-EUR task. It should succeed, but success alone does not justify its extra model decisions.
            """,
        ),
        _code(
            "lesson08-011",
            """
            agent_result = run_bounded_agent(compound_question, policy=lesson_model.decide_agent, registry=registry, max_steps=MAX_STEPS)
            agent_tools = [step.tool_name for step in agent_result.trajectory if step.phase == "tool"]
            print(f"agent_status={agent_result.status}")
            print("agent_tool_order=" + " -> ".join(agent_tools))
            print(agent_result.answer)

            fig, ax = plt.subplots(figsize=(9, 4)); ax.axis("off")
            for x, y, label in [(0.05, .57, "Visible state"), (.39, .57, "Model choice"), (.73, .57, "Typed tool"), (.39, .12, f"Finish or\\nMAX_STEPS={MAX_STEPS}")]:
                edge = "#F07D00" if "Finish" in label else "#1F40CB"
                ax.add_patch(FancyBboxPatch((x, y), .22, .20, boxstyle="round,pad=.02", facecolor="#F4F7FF", edgecolor=edge, linewidth=2)); ax.text(x+.11, y+.10, label, ha="center", va="center", weight="bold")
            ax.add_patch(FancyArrowPatch((.27,.67),(.38,.67),arrowstyle="-|>",mutation_scale=16,color="#00A2EB",linewidth=2))
            ax.add_patch(FancyArrowPatch((.61,.67),(.72,.67),arrowstyle="-|>",mutation_scale=16,color="#00A2EB",linewidth=2))
            ax.add_patch(FancyArrowPatch((.84,.79),(.16,.79),connectionstyle="arc3,rad=.28",arrowstyle="-|>",mutation_scale=16,color="#00A2EB",linewidth=2))
            ax.add_patch(FancyArrowPatch((.50,.56),(.50,.34),arrowstyle="-|>",mutation_scale=16,color="#F07D00",linewidth=2))
            ax.set_title("An agent is a model-directed loop inside code-owned limits", loc="left", weight="bold")
            plt.show()
            """,
        ),
        _markdown(
            "lesson08-012",
            """
            ## Same result, different control cost

            For this fixed task, both designs use the same two tools. The workflow requires no model decision inside the route; the agent repeatedly asks the model what to do next. Prefer the simpler architecture unless measured task quality improves.
            """,
        ),
        _code(
            "lesson08-013",
            """
            comparison = pd.DataFrame([
                {"architecture": "deterministic workflow", "tool calls": len(workflow_tools), "model route decisions": 0, "status": workflow_compound.status},
                {"architecture": "bounded agent", "tool calls": len(agent_tools), "model route decisions": sum(step.phase == "plan" for step in agent_result.trajectory), "status": agent_result.status},
            ])
            display(comparison)
            fig, ax = plt.subplots(figsize=(8.5, 4))
            comparison.set_index("architecture")[["tool calls", "model route decisions"]].plot(kind="bar", ax=ax, color=["#00A2EB", "#F07D00"])
            ax.set_title("The agent adds decisions but no value on a fixed route", loc="left", weight="bold")
            ax.set(xlabel="", ylabel="Count"); ax.tick_params(axis="x", rotation=0); ax.legend(frameon=False)
            plt.tight_layout(); plt.show()
            print("preferred_architecture=workflow")
            """,
        ),
        _markdown(
            "lesson08-014",
            """
            ## Failure lab

            A model-selected conversion is rejected unless its amount and source currency exactly match a successful price observation. A policy that keeps requesting tools is stopped by `MAX_STEPS`. These are application guarantees, not prompt suggestions.

            ## Verification

            The checks below prove: direct and two-step workflows complete; both pass the same observed value between tools; the agent completes in the correct order; and the step budget stops a looping policy.

            ## Challenge

            Add a predefined branch for Schneider Electric. Then propose one genuinely open-ended finance investigation and name the measurable quality gain that would justify model-directed control flow.

            ## Capstone integration

            Reuse the typed requests, observations, normalized trace, and hard stop in the Financial Analyst Copilot. Lesson 09 keeps these contracts and adds explicit recovery state.

            ## Recap

            - Workflows can chain, branch, and loop using tool results.
            - Agents differ because the model chooses the next action.
            - Known routes belong in deterministic code.
            - Use an agent only when model-directed control flow creates measurable value.
            """,
        ),
        _code(
            "lesson08-015",
            """
            def looping_policy(_question, _trajectory):
                return AgentDecision(action="tool", request=ToolRequest(name="get_market_price", arguments={"ticker": "NVDA"}))

            budget_result = run_bounded_agent("Repeat forever", policy=looping_policy, registry=registry, max_steps=2)
            counts = pd.Series([step.phase for step in budget_result.trajectory]).value_counts().reindex(["plan", "tool", "guardrail"], fill_value=0)
            fig, ax = plt.subplots(figsize=(7.5, 3.5)); counts.plot(kind="bar", ax=ax, color=["#1F40CB", "#00A2EB", "#F07D00"])
            ax.set_title("Code—not the model—owns termination", loc="left", weight="bold"); ax.set(xlabel="Recorded event", ylabel="Count"); ax.tick_params(axis="x", rotation=0)
            plt.tight_layout(); plt.show()

            assert workflow_direct.status == "completed"
            assert workflow_compound.status == "completed"
            assert workflow_tools == ["get_market_price", "convert_currency"]
            assert agent_result.status == "completed"
            assert agent_tools == ["get_market_price", "convert_currency"]
            workflow_price = workflow_compound.trajectory[1].observation
            workflow_fx = workflow_compound.trajectory[2].observation
            assert workflow_price is not None and workflow_fx is not None
            assert workflow_fx.payload["input_amount"] == workflow_price.payload["price"]
            assert budget_result.status == "step_budget_exhausted"
            assert budget_result.trajectory[-1].phase == "guardrail"
            print(budget_result.trajectory[-1].summary)
            print("LESSON_08_PASS")
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
