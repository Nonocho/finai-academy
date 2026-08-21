"""Build the canonical output-free Lesson 09 notebook."""

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
            # 09 — Self-correcting financial agent

            **First Finance - Arnaud Demes**  
            **Day 2 · 10:30–11:15 · 10 minutes concepts + 30 minutes notebook + 5 minutes debrief**

            **Engineering question:** how can a financial agent use a typed tool error to correct its next action without entering an unlimited loop?

            This notebook uses NVIDIA (`NVDA`) and Schneider Electric (`SU.PA`) with a controlled classroom fixture. The values are not live market data or investment advice.
            """,
        ),
        _markdown(
            "lesson09-001",
            """
            ## Learning objectives

            By the end, you can:

            1. define explicit state for a LangGraph agent;
            2. separate an agent node from a deterministic tool node;
            3. convert validation failures into structured observations;
            4. route errors back to the model as useful context;
            5. enforce `MAX_RETRIES` and `MAX_TOOL_CALLS`; and
            6. verify both the answer and the correction path.

            **Expected visible result:** the first request uses invalid metric `PE`, the tool returns `unsupported_metric`, the agent retries with `P/E`, then compares NVIDIA and Schneider Electric from successful observations.
            """,
        ),
        _markdown(
            "lesson09-002",
            """
            ## Where this fits

            Lesson 08 exposed the agent loop in plain Python. Lesson 09 earns LangGraph by making state, error routing, and stop conditions explicit:

            ```text
            bounded agent → structured error → corrected action → bounded completion
            ```

            Set `FINAI_LIVE_MODE=1` through the course executor to use Ollama or OpenAI. Offline mode runs the same graph with a deterministic recorded policy.
            """,
        ),
        _code(
            "lesson09-003",
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
                build_metric_registry,
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
            snapshot_path = (
                PROJECT_ROOT
                / "assets/course-data/market/lesson09_metrics_snapshot_v1.json"
            )
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            registry = build_metric_registry(snapshot)
            print(f"Runtime: {runtime_label}")
            print(f"Dataset: {snapshot['dataset_id']}")
            print(f"Tickers: {', '.join(registry.tickers)}")
            print(f"Valid metrics: {', '.join(registry.metric_names)}")
            """,
        ),
        _markdown(
            "lesson09-004",
            """
            ### The graph makes control visible

            The model chooses an action. Python validates and executes the tool. Conditional edges decide whether to continue, finish, or stop at a guardrail.

            The course module compiles this contract with `workflow = StateGraph(RecoveryState)`. Keeping the graph in a tested Python module lets the notebook focus on state changes and visible evidence while students can still inspect the production implementation.
            """,
        ),
        _code(
            "lesson09-005",
            """
            fig, ax = plt.subplots(figsize=(10, 5.2))
            ax.axis("off")
            positions = {
                "agent": (0.08, 0.58),
                "tools": (0.40, 0.58),
                "finish": (0.73, 0.58),
                "guardrails": (0.40, 0.16),
            }
            arrows = [
                ((0.26, 0.69), (0.40, 0.69), "tool request"),
                ((0.58, 0.69), (0.73, 0.69), "valid evidence"),
                ((0.49, 0.58), (0.17, 0.58), "result or error"),
                ((0.49, 0.58), (0.49, 0.35), "budget reached"),
            ]
            for start, end, label in arrows:
                ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, color="#00A2EB", linewidth=2, connectionstyle="arc3,rad=0.12" if "error" in label else "arc3"))
                ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.05, label, ha="center", fontsize=9, color="#4B6070")
            for label, (x, y) in positions.items():
                is_guard = label == "guardrails"
                ax.add_patch(FancyBboxPatch((x, y), 0.18, 0.20, boxstyle="round,pad=0.02", facecolor="#FFF2E5" if is_guard else "#F5F5F5", edgecolor="#F07D00" if is_guard else "#1F40CB", linewidth=2))
                ax.text(x + 0.09, y + 0.10, label, ha="center", va="center", weight="bold", fontsize=13)
            ax.text(0.5, 0.04, "State: question · decision · observation · error_count · tool_calls · trace", ha="center", color="#1F40CB", weight="bold")
            ax.set_title(f"LangGraph turns the loop into explicit state and routes · {runtime_label}", loc="left", weight="bold")
            plt.show()
            """,
        ),
        _markdown(
            "lesson09-006",
            """
            ### Tool errors are context, not crashes

            A useful error contains the rejected value, a stable error code, valid alternatives, and whether correction is allowed. The model receives this observation in the same trace as a successful result.
            """,
        ),
        _code(
            "lesson09-007",
            """
            wrong_request = MetricRequest(ticker="NVDA", metric="PE")
            error_observation = registry.invoke(wrong_request)
            display(pd.DataFrame([error_observation.model_dump(mode="json")]))
            print(error_observation.message)

            fig, ax = plt.subplots(figsize=(10, 3.8))
            ax.axis("off")
            labels = [
                (0.04, "Request", "metric = PE", "#1F40CB"),
                (0.36, "Tool feedback", "unsupported_metric\\nValid: EPS, P/E", "#F07D00"),
                (0.71, "Corrected request", "metric = P/E", "#2E8B57"),
            ]
            for index, (x, title, body, color) in enumerate(labels):
                ax.add_patch(FancyBboxPatch((x, 0.30), 0.24, 0.38, boxstyle="round,pad=0.02", facecolor="#F5F5F5", edgecolor=color, linewidth=2))
                ax.text(x + 0.12, 0.55, title, ha="center", weight="bold", color=color)
                ax.text(x + 0.12, 0.40, body, ha="center", va="center")
                if index < len(labels) - 1:
                    ax.add_patch(FancyArrowPatch((x + 0.24, 0.49), (labels[index + 1][0], 0.49), arrowstyle="-|>", mutation_scale=16, color="#00A2EB", linewidth=2))
            ax.set_title("A precise error is a prompt for the next action", loc="left", weight="bold")
            plt.show()
            """,
        ),
        _markdown(
            "lesson09-008",
            """
            ### Provider-neutral action policy

            Offline mode uses a recorded policy for deterministic verification. Live mode injects the same first invalid request, then asks the configured Ollama or OpenAI model for a structured `AgentAction` from the visible error and results. This is fault injection for teaching, not a claim that every model always makes this exact mistake.
            """,
        ),
        _code(
            "lesson09-009",
            """
            class LiveCorrectionPolicy:
                def __init__(self, chat_model):
                    self.action_model = chat_model.with_structured_output(AgentAction)

                def __call__(self, state):
                    tool_events = [
                        event
                        for event in state.get("trace", ())
                        if event.phase in {"tool_error", "tool_ok"}
                    ]
                    if not tool_events:
                        return AgentAction(
                            action="tool",
                            request=MetricRequest(ticker="NVDA", metric="PE"),
                            reason="Injected invalid alias for the recovery demonstration.",
                        )
                    visible_trace = [
                        event.model_dump(mode="json")
                        for event in state.get("trace", ())
                    ]
                    prompt = f'''Return the next typed action for a bounded financial agent.
            Tool: get_metric(ticker, metric).
            Valid tickers: NVDA, SU.PA.
            Do not guess valid metric names. Read structured tool feedback.
            For comparison questions, collect successful P/E observations for both companies.
            Finish only from successful observations and include value, date and source.
            Question: {state['question']}
            Visible trace: {json.dumps(visible_trace)}'''
                    return self.action_model.invoke([("human", prompt)])


            MAX_RETRIES = 1
            MAX_TOOL_CALLS = 4
            if LIVE_MODE:
                chat_model = create_chat_model(settings)
                correction_policy = LiveCorrectionPolicy(chat_model)
                print("Live provider:", provider_summary(settings))
            else:
                correction_policy = recorded_correction_policy
                print("Policy: offline recorded correction")
            """,
        ),
        _markdown(
            "lesson09-010",
            """
            ### Run the self-correcting graph

            Watch the sequence, not only the final sentence. A professional result must show the failed call, the error feedback, the corrected call, the second company lookup, and a grounded finish.
            """,
        ),
        _code(
            "lesson09-011",
            """
            question = "Compare NVIDIA's P/E with Schneider Electric."
            recovery_result = run_self_correcting_agent(
                question,
                registry=registry,
                policy=correction_policy,
                max_retries=MAX_RETRIES,
                max_tool_calls=MAX_TOOL_CALLS,
            )
            print(f"status={recovery_result.status}")
            print(f"errors={recovery_result.error_count}")
            print(f"tool_calls={recovery_result.tool_calls}")
            print(recovery_result.answer)
            """,
        ),
        _code(
            "lesson09-012",
            """
            trace_rows = []
            for event in recovery_result.trace:
                trace_rows.append(
                    {
                        "event": event.index,
                        "phase": event.phase,
                        "ticker": event.request.ticker if event.request else "",
                        "metric": event.request.metric if event.request else "",
                        "status": event.observation.status if event.observation else "",
                        "summary": event.summary,
                    }
                )
            trace_frame = pd.DataFrame(trace_rows)
            display(trace_frame)

            colors = {
                "agent": "#1F40CB",
                "tool_error": "#F07D00",
                "tool_ok": "#00A2EB",
                "finish": "#2E8B57",
                "guardrail": "#8A2C2C",
            }
            fig, ax = plt.subplots(figsize=(11, 4.4))
            for _, row in trace_frame.iterrows():
                ax.scatter(row["event"], 0, s=300, color=colors[row["phase"]], zorder=3)
                label = row["phase"]
                if row["metric"]:
                    label += f"\\n{row['ticker']} {row['metric']}"
                ax.annotate(label, (row["event"], 0), xytext=(0, 28 if row["event"] % 2 else -48), textcoords="offset points", ha="center", fontsize=9, weight="bold")
            ax.plot(trace_frame["event"], [0] * len(trace_frame), color="#A0A7AE", linewidth=2, zorder=1)
            ax.set(xlim=(0.5, len(trace_frame) + 0.5), ylim=(-0.7, 0.7), xlabel="Recorded event order")
            ax.set_yticks([])
            ax.set_title(f"The trace proves the correction path · {runtime_label}", loc="left", weight="bold")
            ax.grid(axis="x", alpha=0.15)
            plt.tight_layout()
            plt.show()
            """,
        ),
        _code(
            "lesson09-013",
            """
            successful = [
                event.observation.payload
                for event in recovery_result.trace
                if event.phase == "tool_ok" and event.observation is not None
            ]
            metric_frame = pd.DataFrame(successful)
            display(metric_frame)

            fig, ax = plt.subplots(figsize=(8.5, 4.2))
            ax.bar(metric_frame["company"], metric_frame["value"], color=["#1F40CB", "#00A2EB"])
            ax.set(title="Controlled classroom comparison", ylabel="P/E value")
            for index, value in enumerate(metric_frame["value"]):
                ax.text(index, value + 1, f"{value:.1f}", ha="center", weight="bold")
            ax.text(0.5, -0.20, "Controlled fixture · not live data or investment advice", transform=ax.transAxes, ha="center", color="#F07D00")
            plt.tight_layout()
            plt.show()
            """,
        ),
        _markdown(
            "lesson09-014",
            """
            ## Failure lab

            A model may ignore the feedback and repeat `PE`. The application must permit only one retry, record the second error, and stop before a third tool call.
            """,
        ),
        _code(
            "lesson09-015",
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
            phase_counts = pd.Series(event.phase for event in failed_result.trace).value_counts()
            fig, ax = plt.subplots(figsize=(8, 4))
            phase_counts.reindex(["agent", "tool_error", "guardrail"], fill_value=0).plot(kind="bar", ax=ax, color=["#1F40CB", "#F07D00", "#8A2C2C"])
            ax.set(title=f"The retry budget stops repeated failures · {failed_result.status}", xlabel="Recorded phase", ylabel="Event count")
            ax.tick_params(axis="x", rotation=0)
            plt.tight_layout()
            plt.show()
            print(failed_result.trace[-1].summary)
            """,
        ),
        _markdown(
            "lesson09-016",
            """
            ## Verification

            Verify behavior, not appearance:

            - first executed metric is `PE` and fails with `unsupported_metric`;
            - second executed metric is `P/E` and succeeds;
            - both companies have successful observations before the answer;
            - exactly one error is recorded in the successful run;
            - repeated invalid calls stop after the allowed retry; and
            - every successful value retains date and source.
            """,
        ),
        _code(
            "lesson09-017",
            """
            executed_events = [
                event
                for event in recovery_result.trace
                if event.phase in {"tool_error", "tool_ok"}
            ]
            assert recovery_result.status == "completed"
            assert recovery_result.error_count == 1
            assert recovery_result.tool_calls == 3
            assert [event.request.metric for event in executed_events] == ["PE", "P/E", "P/E"]
            assert executed_events[0].observation.error_code == "unsupported_metric"
            assert {item["ticker"] for item in successful} == {"NVDA", "SU.PA"}
            assert all(item["as_of"] and item["source"] for item in successful)
            assert failed_result.status == "retry_budget_exhausted"
            assert failed_result.tool_calls == 2
            assert failed_result.trace[-1].phase == "guardrail"
            print("LESSON_09_PASS")
            """,
        ),
        _markdown(
            "lesson09-018",
            """
            ### Knowledge check

            1. Why is `unsupported_metric` more useful than a generic exception?
            2. Which component executes the financial tool?
            3. Why count model-caused validation failures separately from infrastructure errors?
            4. What does `MAX_RETRIES` guarantee, and what does it not guarantee?
            5. Which trace events prove the final comparison is supported?

            Answers: it contains corrective context; Python executes tools; the recovery strategy differs; it bounds retries but not quality; require both successful company observations before `finish`.
            """,
        ),
        _markdown(
            "lesson09-019",
            """
            ## Challenge

            Add `failed_calls` to the state. Before executing a request, detect whether the same tool and arguments have already failed. Return a structured observation that tells the model not to repeat it.

            Advanced option: classify `validation_error`, `timeout`, and `rate_limit`. Decide which errors should consume the model-correction budget and which need an infrastructure retry policy.
            """,
        ),
        _markdown(
            "lesson09-020",
            """
            ## Capstone integration

            Lesson 09 contributes:

            - explicit LangGraph state;
            - agent and deterministic tool nodes;
            - structured error feedback;
            - conditional recovery routes; and
            - bounded retry and tool-call budgets.

            Lesson 10 exposes financial resources and tools through MCP so the application can discover capabilities instead of importing every function directly.
            """,
        ),
        _markdown(
            "lesson09-021",
            """
            ## Recap

            - Prompt instructions reduce tool errors but do not eliminate them.
            - Precise error observations help a model correct its next action.
            - LangGraph makes state, routes, and stop conditions explicit.
            - Application Python validates and executes every financial tool.
            - Self-correction is useful only when retries remain bounded and traces remain inspectable.
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
