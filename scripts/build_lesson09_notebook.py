"""Build the compact, output-free Lesson 09 notebook."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/09_self_correcting_agent.ipynb"


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
    notebook.metadata.finai = {"expected_runtime_minutes": 30, "lesson": "09"}
    notebook.cells = [
        _markdown(
            "lesson09-000",
            """
            # 09 — A tool error can become the next input

            **Engineering question:** how can an agent use precise external feedback to correct an observable action without entering an unlimited loop?

            This lesson uses NVIDIA (`NVDA`) and Schneider Electric (`SU.PA`) with a controlled classroom fixture. Values are not live market data or investment advice.

            ## Learning objectives

            By the end, you can classify an error, return model-correctable feedback through LangGraph state, inspect the correction trace, and enforce retry and tool-call budgets.

            ## Where this fits

            Lesson 08 introduced a bounded agent loop. Lesson 09 makes its state, recovery route, and stopping conditions explicit with `StateGraph`.
            """,
        ),
        _code(
            "lesson09-001",
            """
            import json
            import os
            from pathlib import Path

            import matplotlib.pyplot as plt
            import pandas as pd
            from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

            from finai_academy.providers import create_chat_model, provider_summary
            from finai_academy.self_correcting_agent import (
                AgentAction,
                MetricRequest,
                ModelAgentAction,
                build_metric_registry,
                build_self_correcting_graph,
                recorded_correction_policy,
                run_self_correcting_agent,
            )
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
            snapshot = json.loads(
                (PROJECT_ROOT / "assets/course-data/market/lesson09_metrics_snapshot_v1.json")
                .read_text(encoding="utf-8")
            )
            registry = build_metric_registry(snapshot)
            print(f"Runtime: {runtime_label}")
            print(f"Dataset: {snapshot['dataset_id']}")
            print(f"Valid tickers: {', '.join(registry.tickers)}")
            print(f"Valid metrics: {', '.join(registry.metric_names)}")
            """,
        ),
        _markdown(
            "lesson09-002",
            """
            ## First classify the error

            “Retry” is not one universal action. A **model-correctable** validation error should return precise context to the model. A **transient** timeout needs system retry and backoff. A user-fixable omission should pause for input. An unexpected bug should surface for debugging.
            """,
        ),
        _code(
            "lesson09-003",
            """
            strategies = pd.DataFrame(
                [
                    ("MODEL-CORRECTABLE", "Unsupported metric", "Return typed feedback to agent", "#F07D00"),
                    ("TRANSIENT", "Timeout or rate limit", "Retry with backoff", "#00A2EB"),
                    ("USER-FIXABLE", "Missing ticker", "Pause and ask", "#2E8B57"),
                    ("UNEXPECTED", "Unknown application bug", "Raise, log, investigate", "#8A2C2C"),
                ],
                columns=["error_type", "example", "owner_action", "color"],
            )
            display(strategies.drop(columns="color"))

            fig, ax = plt.subplots(figsize=(11, 4.8))
            ax.axis("off")
            for row, item in strategies.iterrows():
                y = 0.80 - row * 0.20
                ax.add_patch(FancyBboxPatch((0.02, y), 0.25, 0.12, boxstyle="round,pad=0.015", facecolor=item.color, edgecolor="none"))
                ax.text(0.145, y + 0.06, item.error_type, ha="center", va="center", color="white", weight="bold")
                ax.text(0.32, y + 0.06, item.example, va="center", weight="bold", color="#0B2230")
                ax.text(0.66, y + 0.06, item.owner_action, va="center", color="#334E5F")
            ax.set_title("Errors need different recovery strategies", loc="left", weight="bold", fontsize=16)
            plt.show()
            """,
        ),
        _markdown(
            "lesson09-004",
            """
            ## A precise tool error becomes context

            The financial tool rejects `metric="PE"` without crashing. Its structured `unsupported_metric` observation includes valid alternatives and explicitly marks whether a corrected attempt is allowed.
            """,
        ),
        _code(
            "lesson09-005",
            """
            wrong_request = MetricRequest(ticker="NVDA", metric="PE")
            error_observation = registry.invoke(wrong_request)
            display(pd.DataFrame([error_observation.model_dump(mode="json")]))

            fig, ax = plt.subplots(figsize=(11, 4.2))
            ax.axis("off")
            steps = [
                (0.04, "INVALID REQUEST", "NVDA · PE", "#1F40CB"),
                (0.37, "TYPED FEEDBACK", "unsupported_metric\\nValid: EPS, P/E", "#F07D00"),
                (0.72, "CORRECTED REQUEST", "NVDA · P/E", "#2E8B57"),
            ]
            for index, (x, title, body, color) in enumerate(steps):
                ax.add_patch(FancyBboxPatch((x, 0.28), 0.24, 0.38, boxstyle="round,pad=0.02", facecolor="#F7F9FA", edgecolor=color, linewidth=2.5))
                ax.text(x + 0.12, 0.57, title, ha="center", weight="bold", color=color)
                ax.text(x + 0.12, 0.42, body, ha="center", va="center", fontsize=12)
                if index < 2:
                    ax.add_patch(FancyArrowPatch((x + 0.24, 0.47), (steps[index + 1][0], 0.47), arrowstyle="-|>", mutation_scale=18, color="#00A2EB", linewidth=2.5))
            ax.set_title("External feedback changes the next observable action", loc="left", weight="bold", fontsize=16)
            plt.show()
            """,
        ),
        _markdown(
            "lesson09-006",
            """
            ## The graph owns the recovery route

            The model proposes a typed action. Python validates and executes it. LangGraph routes the resulting state to another action, a successful finish, or a guardrail.

            Offline mode uses a recorded policy. Live Ollama or OpenAI mode uses `with_structured_output(ModelAgentAction)` and converts that strict wire schema into the internal `AgentAction`.
            """,
        ),
        _code(
            "lesson09-007",
            """
            class LiveCorrectionPolicy:
                def __init__(self, chat_model):
                    self.action_model = chat_model.with_structured_output(ModelAgentAction)

                def __call__(self, state):
                    tool_events = [
                        event for event in state.get("trace", ())
                        if event.phase in {"tool_error", "tool_ok"}
                    ]
                    if not tool_events:
                        return AgentAction(
                            action="tool",
                            request=MetricRequest(ticker="NVDA", metric="PE"),
                            reason="Inject one invalid alias so recovery remains visible.",
                        )
                    visible_trace = [event.model_dump(mode="json") for event in state.get("trace", ())]
                    prompt = f'''Choose the next action for this bounded financial agent.
            Tool: get_metric(ticker, metric).
            Valid tickers: NVDA, SU.PA.
            Use structured tool feedback to correct invalid metric names.
            For comparisons, collect successful P/E observations for both companies.
            Finish only from successful observations and include value, date and source.
            Question: {state['question']}
            Visible trace: {json.dumps(visible_trace)}'''
                    wire_action = self.action_model.invoke([("human", prompt)])
                    return wire_action.to_agent_action()


            MAX_RETRIES = 1
            MAX_TOOL_CALLS = 4
            correction_policy = (
                LiveCorrectionPolicy(create_chat_model(settings))
                if LIVE_MODE
                else recorded_correction_policy
            )
            if LIVE_MODE:
                print("Live provider:", provider_summary(settings))
            else:
                print("Policy: offline recorded correction")

            graph = build_self_correcting_graph(
                registry=registry,
                policy=correction_policy,
                max_retries=MAX_RETRIES,
                max_tool_calls=MAX_TOOL_CALLS,
            )

            fig, ax = plt.subplots(figsize=(11, 4.8))
            ax.axis("off")
            nodes = {
                "agent": (0.05, 0.55, "#1F40CB"),
                "tools": (0.35, 0.55, "#00A2EB"),
                "finish": (0.70, 0.55, "#2E8B57"),
                "guardrails": (0.35, 0.12, "#F07D00"),
            }
            arrows = [
                ((0.25, 0.66), (0.35, 0.66), "request"),
                ((0.55, 0.66), (0.70, 0.66), "evidence"),
                ((0.44, 0.55), (0.15, 0.55), "result or error"),
                ((0.45, 0.55), (0.45, 0.34), "budget reached"),
            ]
            for start, end, label in arrows:
                ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=17, color="#7F8C94", linewidth=2, connectionstyle="arc3,rad=0.12" if "error" in label else "arc3"))
                ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.05, label, ha="center", color="#4B6070")
            for label, (x, y, color) in nodes.items():
                ax.add_patch(FancyBboxPatch((x, y), 0.20, 0.22, boxstyle="round,pad=0.02", facecolor="#F7F9FA", edgecolor=color, linewidth=2.5))
                ax.text(x + 0.10, y + 0.11, label, ha="center", va="center", weight="bold", fontsize=13, color=color)
            ax.set_title("LangGraph makes the recovery route explicit", loc="left", weight="bold", fontsize=16)
            ax.text(0.5, 0.02, "State = question · decision · observation · error_count · tool_calls · trace", ha="center", color="#1F40CB", weight="bold")
            plt.show()
            """,
        ),
        _markdown(
            "lesson09-008",
            """
            ## Follow the successful correction

            Judge the sequence, not only the final sentence: failed request, actionable feedback, corrected request, second company observation, grounded finish.
            """,
        ),
        _code(
            "lesson09-009",
            """
            recovery_result = graph.invoke({"question": "Compare NVIDIA's P/E with Schneider Electric."})
            print(f"success_path={recovery_result['status']}")
            print(f"errors={recovery_result['error_count']} · tool_calls={recovery_result['tool_calls']}")
            print(recovery_result["answer"])

            trace_frame = pd.DataFrame(
                {
                    "event": event.index,
                    "phase": event.phase,
                    "request": (
                        f"{event.request.ticker} {event.request.metric}"
                        if event.request else ""
                    ),
                    "summary": event.summary,
                }
                for event in recovery_result["trace"]
            )
            display(trace_frame)

            phase_colors = {
                "agent": "#1F40CB",
                "tool_error": "#F07D00",
                "tool_ok": "#00A2EB",
                "finish": "#2E8B57",
                "guardrail": "#8A2C2C",
            }
            fig, ax = plt.subplots(figsize=(12, 4.5))
            for _, row in trace_frame.iterrows():
                ax.scatter(row.event, 0, s=340, color=phase_colors[row.phase], zorder=3)
                label = row.phase.replace("_", " ")
                if row.request:
                    label += f"\\n{row.request}"
                ax.annotate(label, (row.event, 0), xytext=(0, 30 if row.event % 2 else -52), textcoords="offset points", ha="center", weight="bold", fontsize=9)
            ax.plot(trace_frame.event, [0] * len(trace_frame), color="#A0A7AE", linewidth=2)
            ax.set(xlim=(0.5, len(trace_frame) + 0.5), ylim=(-0.75, 0.75), xlabel="Recorded event order")
            ax.set_yticks([])
            ax.set_title("The trace proves that external feedback changed the action", loc="left", weight="bold", fontsize=16)
            ax.grid(axis="x", alpha=0.15)
            plt.tight_layout()
            plt.show()
            """,
        ),
        _markdown(
            "lesson09-010",
            """
            ## Failure lab

            Now force the policy to repeat `PE`. `MAX_RETRIES = 1` allows one correction opportunity; the second validation failure must route to `retry_budget_exhausted` before a third tool execution.
            """,
        ),
        _code(
            "lesson09-011",
            """
            def always_wrong(_state):
                return AgentAction(
                    action="tool",
                    request=MetricRequest(ticker="NVDA", metric="PE"),
                    reason="Repeat the same unsupported alias.",
                )


            failed_result = run_self_correcting_agent(
                "Return NVIDIA's P/E.",
                registry=registry,
                policy=always_wrong,
                max_retries=MAX_RETRIES,
                max_tool_calls=MAX_TOOL_CALLS,
            )
            print(f"failure_path={failed_result.status}")
            print(f"errors={failed_result.error_count} · tool_calls={failed_result.tool_calls}")

            def plot_path(ax, events, title):
                phases = [event.phase for event in events]
                xs = list(range(1, len(phases) + 1))
                ax.plot(xs, [0] * len(xs), color="#A0A7AE", linewidth=2)
                for x, phase in zip(xs, phases, strict=True):
                    ax.scatter(x, 0, s=260, color=phase_colors[phase], zorder=3)
                    ax.annotate(phase.replace("_", "\\n"), (x, 0), xytext=(0, 24 if x % 2 else -42), textcoords="offset points", ha="center", fontsize=8, weight="bold")
                ax.set(xlim=(0.5, len(xs) + 0.5), ylim=(-0.65, 0.65), title=title)
                ax.set_yticks([])
                ax.set_xticks(xs)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
            plot_path(axes[0], recovery_result["trace"], "Corrected: evidence then finish")
            plot_path(axes[1], failed_result.trace, "Repeated failure: budget stops the loop")
            fig.suptitle("The same graph can complete safely or stop safely", weight="bold", fontsize=16)
            plt.tight_layout()
            plt.show()
            """,
        ),
        _markdown(
            "lesson09-012",
            """
            ## Verification

            Verify behavior: `PE` fails, `P/E` succeeds, both companies provide evidence, one error is recorded on the successful path, and the repeated failure stops after two executed calls.

            ## Challenge

            Add a `failed_calls` set and prevent an identical failed request from reaching the tool twice. Then decide which failures belong to model correction, infrastructure retry, human input, or immediate stop.
            """,
        ),
        _code(
            "lesson09-013",
            """
            executed = [
                event for event in recovery_result["trace"]
                if event.phase in {"tool_error", "tool_ok"}
            ]
            successful = [
                event.observation.payload for event in recovery_result["trace"]
                if event.phase == "tool_ok" and event.observation is not None
            ]
            assert recovery_result["status"] == "completed"
            assert recovery_result["error_count"] == 1
            assert recovery_result["tool_calls"] == 3
            assert [event.request.metric for event in executed] == ["PE", "P/E", "P/E"]
            assert {item["ticker"] for item in successful} == {"NVDA", "SU.PA"}
            assert all(item["as_of"] and item["source"] for item in successful)
            assert failed_result.status == "retry_budget_exhausted"
            assert failed_result.tool_calls == 2
            assert failed_result.trace[-1].phase == "guardrail"
            print("LESSON_09_PASS")
            """,
        ),
        _markdown(
            "lesson09-014",
            """
            ## Capstone integration

            Lesson 09 contributes explicit state, typed tool feedback, conditional recovery routes, evidence-aware finishing, and bounded retry budgets. Lesson 10 exposes the same financial contracts through MCP.

            ## Recap

            - External feedback can change the next observable action.
            - Self-correction does not guarantee truth or better hidden reasoning.
            - Different failures need different owners and recovery strategies.
            - Application code owns validation, budgets, evidence checks, and stopping.
            """,
        ),
        _markdown(
            "lesson09-015",
            """
            ### Knowledge check

            1. Why is `unsupported_metric` better than a generic exception?
            2. Which errors should loop back to the model?
            3. Why should timeouts use a separate policy?
            4. What do `MAX_RETRIES` and `MAX_TOOL_CALLS` guarantee?
            5. Which trace events support the final comparison?

            **Answers:** it carries corrective context; only model-correctable errors; transient failures need backoff; budgets bound behavior rather than quality; require successful observations for both companies before finish.
            """,
        ),
    ]
    return notebook


def main() -> None:
    nbformat.write(build_notebook(), OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
