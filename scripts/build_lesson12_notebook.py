"""Build the canonical output-free Lesson 12 agent-evaluation notebook."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/12_evaluating_agentic_systems.ipynb"


def _markdown(cell_id: str, source: str):
    cell = nbformat.v4.new_markdown_cell(dedent(source).strip())
    cell.id = cell_id
    return cell


def _code(cell_id: str, source: str):
    cell = nbformat.v4.new_code_cell(dedent(source).strip())
    cell.id = cell_id
    return cell


def build_notebook():
    """Return the deterministic 40-minute Lesson 12 notebook."""

    notebook = nbformat.v4.new_notebook()
    notebook.metadata.kernelspec = {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata.language_info = {"name": "python", "version": "3.11"}
    notebook.metadata.finai = {"expected_runtime_minutes": 40, "lesson": "12"}
    notebook.cells = [
        _markdown(
            "lesson12-000",
            """
            # 12 - Evaluating agentic systems with MLflow

            **First Finance - Arnaud Demes**
            **Day 2 · 14:30-15:30 · 12 minutes deck + 40 minutes notebook + 8 minutes debrief**

            **Outcome:** evaluate both the public trajectory and the cited answer of the NVIDIA and Schneider Electric analyst, then persist an aligned six-case comparison in local MLflow.

            **Prerequisites:** Lessons 07 and 11 plus the evaluation extra. The core route is offline, read-only, and based on controlled classroom evidence, not live market data or investment advice.

            **Visible outputs:** six cases, one real Lesson 11 reference run, five deterministic metrics, two MLflow run IDs, twelve trace IDs, a per-case scorecard, failure ownership, and six figures.

            **Public-state boundary:** compare plans, validated call signatures, typed observations, evidence-gate results, cited facts, sources, and public trajectory events. Do not compare hidden reasoning, private runtime objects, or wall-clock latency for fixture identity.
            """,
        ),
        _markdown(
            "lesson12-001",
            """
            ## Learning objectives

            By the end, you can:

            1. separate trajectory quality from answer quality;
            2. verify a versioned evaluation dataset by exact file hash;
            3. interpret `tool_call_correctness`, `tool_call_efficiency`, `answer_relevance`, `answer_completeness`, and `citation_integrity`;
            4. inspect one MLflow run per configuration and one trace per case; and
            5. assign a failed release check to the earliest public owner.

            ## Where this fits

            Lesson 11 produced a bounded research trajectory and cited briefing. Lesson 12 adds the final Financial Analyst Copilot increment: a **versioned agent trajectory and answer evaluation suite**. It evaluates the maintained system; it does not select the eventual capstone interface or architecture.
            """,
        ),
        _code(
            "lesson12-002",
            """
            import json
            import os
            from hashlib import sha256
            from pathlib import Path

            import matplotlib.pyplot as plt
            import mlflow
            import numpy as np
            import pandas as pd
            from IPython.display import display
            from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

            from finai_academy.agent_evaluation import (
                METRIC_NAMES,
                canonical_call_signature,
                load_agent_evaluation_dataset,
                load_recorded_agent_runs,
                prediction_from_plan_execute_result,
                score_agent_case,
            )
            from finai_academy.mlflow_agent_evaluation import (
                AgentEvaluationConfiguration,
                compare_agent_configurations,
                initialize_local_mlflow,
                run_mlflow_agent_evaluation,
            )
            from finai_academy.plan_execute_graph import run_plan_execute
            from finai_academy.plan_execute_policies import (
                MISSION,
                recorded_planner,
                recorded_replanner,
                recorded_report_writer,
            )
            from finai_academy.planning_mcp_executor import FinancialMcpPlanningExecutor

            PROJECT_ROOT = Path.cwd().resolve()
            if PROJECT_ROOT.name == "notebooks":
                PROJECT_ROOT = PROJECT_ROOT.parent

            JUDGE_MODEL = os.getenv("FINAI_EVAL_JUDGE_MODEL")
            OPENAI_JUDGE_URI = "openai:/<model>"
            OLLAMA_JUDGE_URI = "ollama_chat:/<model>"
            UI_COMMAND_PREFIX = "mlflow ui --backend-store-uri sqlite:///"

            print("Core route: offline deterministic scoring always runs.")
            print("Optional judge selection is explicit through FINAI_EVAL_JUDGE_MODEL.")
            print(f"Configured optional judge: {JUDGE_MODEL or 'none'}")
            """,
        ),
        _code(
            "lesson12-003",
            """
            manifest_path = PROJECT_ROOT / "assets/course-data/manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            case_entry = next(
                item for item in manifest["evaluation_datasets"]
                if item["dataset_version"] == "agent-cases-v1"
            )
            run_entry = next(
                item for item in manifest["evaluation_run_fixtures"]
                if item["fixture_version"] == "agent-runs-v1"
            )
            cases_path = PROJECT_ROOT / case_entry["path"]
            runs_path = PROJECT_ROOT / run_entry["path"]
            assert sha256(cases_path.read_bytes()).hexdigest() == case_entry["sha256"]
            assert sha256(runs_path.read_bytes()).hexdigest() == run_entry["sha256"]

            dataset = load_agent_evaluation_dataset(
                cases_path, expected_sha256=case_entry["sha256"]
            )
            recorded_runs = load_recorded_agent_runs(
                runs_path,
                cases=dataset,
                expected_sha256=run_entry["sha256"],
            )
            recorded_by_id = {
                item.configuration_id: item for item in recorded_runs.configurations
            }
            store = initialize_local_mlflow()

            assert dataset.dataset_version == "agent-cases-v1"
            assert tuple(recorded_by_id) == ("bounded-agent-v1", "regressed-agent-v0")
            print(f"Dataset: {dataset.dataset_version}")
            print(f"Dataset SHA-256: {dataset.dataset_sha256}")
            print(f"Cases: {len(dataset.cases)}")
            print("Configurations: bounded-agent-v1, regressed-agent-v0")
            print(f"Local tracking URI initialized: {store.tracking_uri}")
            """,
        ),
        _code(
            "lesson12-004",
            """
            fig, ax = plt.subplots(figsize=(13, 5.6))
            ax.axis("off")
            columns = [
                ("VERSIONED CASE", ["mission", "expected calls", "evidence + limits"], 0.03, "#EAF7FD", "#1F40CB"),
                ("PUBLIC TRAJECTORY", ["plan + dependencies", "typed tool outcomes", "replan + gate"], 0.36, "#FFF2E5", "#F07D00"),
                ("CITED ANSWER", ["companies + facts", "limitations", "source/evidence pairs"], 0.69, "#EAF8EE", "#2E8B57"),
            ]
            for title, lines, x, fill, edge in columns:
                ax.add_patch(FancyBboxPatch((x, 0.27), 0.27, 0.48, boxstyle="round,pad=0.025", facecolor=fill, edgecolor=edge, linewidth=2.2))
                ax.text(x + 0.135, 0.67, title, ha="center", va="center", weight="bold", color=edge, fontsize=12)
                for index, line in enumerate(lines):
                    ax.text(x + 0.135, 0.54 - index * 0.10, line, ha="center", va="center", fontsize=10, color="#051C2A")
            for start, end in ((0.30, 0.36), (0.63, 0.69)):
                ax.add_patch(FancyArrowPatch((start, 0.51), (end, 0.51), arrowstyle="-|>", mutation_scale=18, color="#4B6070", linewidth=1.8))
            ax.text(0.50, 0.12, "A release decision needs both path evidence and answer evidence.", ha="center", weight="bold", color="#051C2A", fontsize=11)
            ax.set_title("Figure 1. Versioned expectations evaluate trajectory and answer separately", loc="left", weight="bold", fontsize=14)
            plt.tight_layout()
            plt.show()
            """,
        ),
        _code(
            "lesson12-005",
            """
            case_table = pd.DataFrame([
                {
                    "case_id": case.case_id,
                    "expected_status": case.expected_final_status,
                    "expected_calls": len(case.expected_tool_calls),
                    "max_calls": case.max_tool_calls,
                    "briefing_allowed": case.allow_briefing,
                }
                for case in dataset.cases
            ])
            display(case_table)
            reference_case = next(case for case in dataset.cases if case.case_id == "reference_completed")
            complete_expectation = {
                **reference_case.model_dump(mode="json"),
                "expected_tool_calls": [
                    {
                        **call.model_dump(mode="json"),
                        "signature": canonical_call_signature(call.capability, call.arguments),
                    }
                    for call in reference_case.expected_tool_calls
                ],
            }
            print(f"Version/hash: {dataset.dataset_version} / {dataset.dataset_sha256}")
            print("One complete expectation row:")
            print(json.dumps(complete_expectation, indent=2))
            """,
        ),
        _code(
            "lesson12-006",
            """
            async def run_real_lesson11_reference():
                async with FinancialMcpPlanningExecutor() as executor:
                    server_name = executor.server_name
                    result = await run_plan_execute(
                        question=MISSION,
                        executor=executor,
                        planner=recorded_planner,
                        replanner=recorded_replanner,
                        report_writer=recorded_report_writer,
                    )
                return server_name, result


            reference_server_name, real_result = await run_real_lesson11_reference()
            print(f"Real Lesson 11 MCP server: {reference_server_name}")
            print(f"Real Lesson 11 final status: {real_result.status}")
            print(f"Real Lesson 11 attempts: {len(real_result.observations)}")
            print(f"Real Lesson 11 replans: {real_result.replan_count}")
            """,
        ),
        _code(
            "lesson12-007",
            """
            bounded_record = recorded_by_id["bounded-agent-v1"]
            recorded_reference = next(
                item for item in bounded_record.predictions
                if item.case_id == "reference_completed"
            )
            live_reference = prediction_from_plan_execute_result(
                real_result,
                case_id="reference_completed",
                dataset_version=dataset.dataset_version,
                dataset_sha256=dataset.dataset_sha256,
                configuration_id=bounded_record.configuration_id,
                agent_version=bounded_record.agent_version,
                provider=bounded_record.provider,
                agent_model=bounded_record.agent_model,
                prompt_version=bounded_record.prompt_version,
                max_steps=bounded_record.max_steps,
                max_replans=bounded_record.max_replans,
            )


            def public_signature(prediction):
                briefing = prediction.briefing
                return {
                    "initial_calls": tuple(
                        canonical_call_signature(step.capability, step.arguments)
                        for step in prediction.initial_plan.steps
                    ),
                    "final_calls": tuple(
                        canonical_call_signature(step.capability, step.arguments)
                        for step in prediction.final_steps
                    ),
                    "status": prediction.status,
                    "evidence_ids": tuple(
                        evidence_id
                        for observation in prediction.observations
                        for evidence_id in observation.evidence_ids
                    ),
                    "facts": tuple(
                        (
                            fact.claim,
                            fact.provenance_kind,
                            fact.source_references,
                            fact.evidence_ids,
                        )
                        for fact in (briefing.reported_facts if briefing else ())
                    ),
                    "aggregate_sources": briefing.source_references if briefing else (),
                    "replan_count": prediction.replan_count,
                }


            live_signature = public_signature(live_reference)
            fixture_signature = public_signature(recorded_reference)
            assert live_signature == fixture_signature
            print("Reference public signature: MATCH")
            print(f"Initial call signatures: {live_signature['initial_calls']}")
            print(f"Final call signatures: {live_signature['final_calls']}")
            print(f"Final status: {live_signature['status']}")
            print(f"Evidence IDs: {live_signature['evidence_ids']}")
            print(f"Facts: {live_signature['facts']}")
            print(f"Provenance sources: {live_signature['aggregate_sources']}")
            print(f"Replan count: {live_signature['replan_count']}")
            """,
        ),
        _code(
            "lesson12-008",
            """
            display(pd.DataFrame([
                {
                    "step_id": step.step_id,
                    "capability": step.capability,
                    "arguments": json.dumps(step.arguments, sort_keys=True),
                    "depends_on": list(step.depends_on),
                }
                for step in real_result.final_steps
            ]))
            display(pd.DataFrame([
                {
                    "index": event.index,
                    "phase": event.phase,
                    "attempt": event.attempt_id,
                    "status": event.status,
                    "summary": event.summary,
                    "latency_ms": event.duration_ms,
                }
                for event in real_result.trajectory
            ]))
            typed_errors = [
                {
                    "attempt": item.attempt_id,
                    "capability": item.capability,
                    "status": item.status,
                    "error_code": item.error_code,
                }
                for item in real_result.observations if item.error_code
            ]
            display(pd.DataFrame(typed_errors))
            print(f"Evidence gate: {real_result.evidence_gate.model_dump(mode='json')}")
            briefing_rows = [
                {
                    "claim": fact.claim,
                    "kind": fact.provenance_kind,
                    "sources": list(fact.source_references),
                    "evidence_ids": list(fact.evidence_ids),
                }
                for fact in real_result.briefing.reported_facts
            ]
            display(pd.DataFrame(briefing_rows))
            print(f"Limitations: {real_result.briefing.limitations}")
            print(f"Aggregate sources: {real_result.briefing.source_references}")
            """,
        ),
        _code(
            "lesson12-009",
            """
            expected_rows = []
            observed_by_signature = {
                canonical_call_signature(item.capability, item.arguments): item
                for item in live_reference.observations
            }
            for expected_call in reference_case.expected_tool_calls:
                signature = canonical_call_signature(
                    expected_call.capability, expected_call.arguments
                )
                observed = observed_by_signature.get(signature)
                expected_rows.append({
                    "call_id": expected_call.call_id,
                    "expected signature": signature,
                    "observed signature": (
                        canonical_call_signature(observed.capability, observed.arguments)
                        if observed else "missing"
                    ),
                    "dependencies": ", ".join(expected_call.prerequisite_call_ids) or "none",
                    "status": observed.status if observed else "missing",
                })
            call_alignment = pd.DataFrame(expected_rows)
            display(call_alignment)

            fig, ax = plt.subplots(figsize=(14, 6.2))
            ax.axis("off")
            for index, row in enumerate(expected_rows):
                x = 0.03 + index * 0.19
                edge = "#2E8B57" if row["status"] == "ok" else "#F07D00"
                ax.add_patch(FancyBboxPatch((x, 0.50), 0.16, 0.24, boxstyle="round,pad=0.018", facecolor="#F5F5F5", edgecolor=edge, linewidth=2))
                ax.text(x + 0.08, 0.68, row["call_id"].replace("-", "\\n", 1), ha="center", va="center", fontsize=9, weight="bold", color=edge)
                ax.text(x + 0.08, 0.55, row["status"], ha="center", va="center", fontsize=9)
                if index:
                    ax.add_patch(FancyArrowPatch((x - 0.03, 0.62), (x, 0.62), arrowstyle="-|>", mutation_scale=13, color="#4B6070"))
            dependency_pairs = [(0, 3), (1, 4), (3, 4)]
            for start, end in dependency_pairs:
                x1 = 0.11 + start * 0.19
                x2 = 0.11 + end * 0.19
                ax.add_patch(FancyArrowPatch((x1, 0.49), (x2, 0.49), arrowstyle="-|>", mutation_scale=11, connectionstyle="arc3,rad=0.18", color="#1F40CB", linewidth=1.3))
            ax.text(0.50, 0.24, "Orange retains the expected typed error; blue arcs are declared prerequisites, not inferred array order.", ha="center", color="#051C2A", weight="bold", fontsize=10)
            ax.set_title("Figure 2. Expected and observed dependency-aware call signatures align", loc="left", weight="bold", fontsize=14)
            plt.tight_layout()
            plt.show()
            """,
        ),
        _code(
            "lesson12-010",
            """
            reference_scores = score_agent_case(reference_case, live_reference)
            correctness = reference_scores.tool_call_correctness
            display(pd.DataFrame([{
                "metric": "tool_call_correctness",
                "score": correctness.value,
                "rationale": correctness.rationale,
            }]))
            print("Correctness checks capability + canonical arguments, declared dependencies, the typed unsupported_metric error, and the expected replan transition.")
            """,
        ),
        _code(
            "lesson12-011",
            """
            efficiency = reference_scores.tool_call_efficiency
            display(pd.DataFrame([{
                "metric": "tool_call_efficiency",
                "score": efficiency.value,
                "tool_calls": reference_scores.total_tool_calls,
                "redundant_calls": reference_scores.redundant_tool_calls,
                "budget": reference_case.max_tool_calls,
                "rationale": efficiency.rationale,
            }]))
            print("Efficiency penalizes repeated successful calls, budget overruns, post-terminal execution, and excess replans. A polished briefing cannot erase a wasteful path.")
            """,
        ),
        _code(
            "lesson12-012",
            """
            revision = 0
            trace_rows = []
            observations_by_attempt = {
                item.attempt_id: item for item in real_result.observations
            }
            for event in real_result.trajectory:
                if event.phase == "execution" and event.attempt_id in observations_by_attempt:
                    event_revision = observations_by_attempt[event.attempt_id].plan_revision
                    revision = max(revision, event_revision)
                elif event.phase == "replanning" and "replace_remaining" in event.summary:
                    revision = min(revision + 1, real_result.replan_count)
                    event_revision = revision
                else:
                    event_revision = revision
                trace_rows.append({
                    "phase": event.phase,
                    "attempt": event.attempt_id,
                    "revision": event_revision,
                    "status": event.status,
                    "latency_ms": event.duration_ms,
                    "summary": event.summary,
                })
            trace_table = pd.DataFrame(trace_rows)
            display(trace_table)
            execution_revisions = [
                int(row["revision"]) for row in trace_rows if row["phase"] == "execution"
            ]
            print(f"Execution revisions: {execution_revisions}")

            phase_colors = {
                "planning": "#1F40CB", "policy": "#F07D00", "execution": "#00A2EB",
                "replanning": "#7B61A8", "evidence_gate": "#2E8B57", "report": "#051C2A",
                "guardrail": "#C83737",
            }
            fig, ax = plt.subplots(figsize=(14, 7.0))
            y = np.arange(len(trace_rows))
            widths = [max(float(row["latency_ms"]), 0.15) for row in trace_rows]
            ax.barh(y, widths, color=[phase_colors[row["phase"]] for row in trace_rows], alpha=0.88)
            ax.set_yticks(y, [f"{index + 1:02d}  {row['phase']}" for index, row in enumerate(trace_rows)], fontsize=9)
            ax.invert_yaxis()
            ax.set_xscale("log")
            ax.set_xlim(0.1, max(widths) * 3.0)
            ax.set_xlabel("Public event latency (ms, log scale; 0 shown as 0.15)")
            ax.set_title("Figure 3. One public trace retains phase, attempt, revision, status, and latency", loc="left", weight="bold", fontsize=14)
            for index, (width, row) in enumerate(zip(widths, trace_rows, strict=True)):
                label = f"{row['status']} | attempt {row['attempt'] or '-'} | rev {row['revision']} | {row['latency_ms']:.1f} ms"
                ax.text(width * 1.12, index, label, va="center", fontsize=8, color="#051C2A")
            ax.grid(axis="x", alpha=0.22)
            plt.tight_layout()
            plt.show()
            """,
        ),
        _code(
            "lesson12-013",
            """
            display(pd.DataFrame([
                {
                    "metric": "answer_relevance",
                    "score": reference_scores.answer_relevance.value,
                    "rationale": reference_scores.answer_relevance.rationale,
                },
                {
                    "metric": "answer_completeness",
                    "score": reference_scores.answer_completeness.value,
                    "rationale": reference_scores.answer_completeness.rationale,
                },
            ]))
            print("Relevance asks whether the maintained mission dimensions and companies are addressed.")
            print("Completeness checks required evidence IDs, fact kinds, comparison content, limitations, and expected status.")
            """,
        ),
        _code(
            "lesson12-014",
            """
            citation_rules = pd.DataFrame([
                {
                    "fact kind": "metric",
                    "required provenance": "one successful metric source",
                    "evidence ID rule": "none",
                },
                {
                    "fact kind": "document",
                    "required provenance": "one exact returned source",
                    "evidence ID rule": "one exact returned source/evidence pair",
                },
                {
                    "fact kind": "aggregate",
                    "required provenance": "ordered union of cited fact sources",
                    "evidence ID rule": "derived from individual document facts",
                },
            ])
            display(citation_rules)
            display(pd.DataFrame([{
                "metric": "citation_integrity",
                "score": reference_scores.citation_integrity.value,
                "rationale": reference_scores.citation_integrity.rationale,
            }]))
            assert reference_scores.citation_integrity.value == 1.0
            print("Citation integrity is a deterministic financial release gate. Plausible text with missing or cross-paired provenance receives zero.")
            """,
        ),
        _code(
            "lesson12-015",
            """
            def evaluation_configuration(recorded):
                return AgentEvaluationConfiguration(
                    configuration_id=recorded.configuration_id,
                    dataset_version=dataset.dataset_version,
                    dataset_sha256=dataset.dataset_sha256,
                    agent_version=recorded.agent_version,
                    provider=recorded.provider,
                    agent_model=recorded.agent_model,
                    prompt_version=recorded.prompt_version,
                    max_steps=recorded.max_steps,
                    max_replans=recorded.max_replans,
                )


            bounded_summary = run_mlflow_agent_evaluation(
                tracking_directory=store.root_directory,
                experiment_name="lesson-12-agent-evaluation",
                configuration=evaluation_configuration(bounded_record),
                cases=dataset.cases,
                predictions=bounded_record.predictions,
            )
            assert bounded_summary.trace_count == 6
            print(f"Logged bounded-agent-v1: run={bounded_summary.run_id}, traces={bounded_summary.trace_count}")
            """,
        ),
        _code(
            "lesson12-016",
            """
            regressed_record = recorded_by_id["regressed-agent-v0"]
            regressed_summary = run_mlflow_agent_evaluation(
                tracking_directory=store.root_directory,
                experiment_name="lesson-12-agent-evaluation",
                configuration=evaluation_configuration(regressed_record),
                cases=dataset.cases,
                predictions=regressed_record.predictions,
            )
            assert regressed_summary.trace_count == 6
            assert bounded_summary.parameters["dataset_sha256"] == regressed_summary.parameters["dataset_sha256"]
            print(f"Logged regressed-agent-v0: run={regressed_summary.run_id}, traces={regressed_summary.trace_count}")
            """,
        ),
        _code(
            "lesson12-017",
            """
            summaries = (bounded_summary, regressed_summary)
            comparison = compare_agent_configurations(summaries)
            mean_table = pd.DataFrame(comparison.metric_mean_rows).merge(
                pd.DataFrame(comparison.metric_pass_rows),
                on=["configuration_id", "metric"],
                validate="one_to_one",
            )
            case_score_table = pd.DataFrame(comparison.case_metric_rows)
            tool_table = pd.DataFrame(comparison.tool_call_rows)
            latency_table = pd.DataFrame(comparison.latency_rows)
            failure_table = pd.DataFrame(comparison.failure_rows)

            print("Cases per configuration: 6")
            print(f"Run ID (bounded-agent-v1): {bounded_summary.run_id}")
            print(f"Run ID (regressed-agent-v0): {regressed_summary.run_id}")
            for configuration_id, summary in zip(comparison.configuration_ids, summaries, strict=True):
                for case_id, trace_id in summary.trace_ids.items():
                    print(f"Trace ID ({configuration_id}/{case_id}): {trace_id}")
            print(f"Total traces: {sum(summary.trace_count for summary in summaries)}")
            display(mean_table[["configuration_id", "metric", "mean", "pass_count", "case_count"]])
            display(case_score_table)
            display(tool_table.groupby("configuration_id", as_index=False).agg(total_tool_calls=("total_tool_calls", "sum"), redundant_tool_calls=("redundant_tool_calls", "sum")))
            display(latency_table)
            display(failure_table if not failure_table.empty else pd.DataFrame([{"failure_stage": "none"}]))
            """,
        ),
        _code(
            "lesson12-018",
            """
            heatmap_labels = [
                f"{row.configuration_id}\\n{row.case_id}"
                for row in case_score_table.itertuples(index=False)
            ]
            heatmap_values = case_score_table[list(METRIC_NAMES)].to_numpy(dtype=float)
            fig, ax = plt.subplots(figsize=(14, 9.0))
            image = ax.imshow(heatmap_values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(METRIC_NAMES)), [name.replace("_", "\\n") for name in METRIC_NAMES], fontsize=10)
            ax.set_yticks(range(len(heatmap_labels)), heatmap_labels, fontsize=8)
            for row in range(heatmap_values.shape[0]):
                for column in range(heatmap_values.shape[1]):
                    value = heatmap_values[row, column]
                    ax.text(column, row, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.38 or value > 0.78 else "#051C2A", weight="bold", fontsize=8)
            ax.set_title("Figure 4. Per-case metrics reveal failures hidden by configuration means", loc="left", weight="bold", fontsize=14)
            fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="score")
            plt.tight_layout()
            plt.show()
            """,
        ),
        _code(
            "lesson12-019",
            """
            pivot_means = mean_table.pivot(index="metric", columns="configuration_id", values="mean").reindex(METRIC_NAMES)
            fig, ax = plt.subplots(figsize=(14, 6.6))
            x = np.arange(len(METRIC_NAMES))
            width = 0.36
            colors = {"bounded-agent-v1": "#1F40CB", "regressed-agent-v0": "#F07D00"}
            for offset, configuration_id in zip((-width / 2, width / 2), comparison.configuration_ids, strict=True):
                values = pivot_means[configuration_id].to_numpy(dtype=float)
                bars = ax.bar(x + offset, values, width, label=configuration_id, color=colors[configuration_id])
                ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=3, fontsize=9)
            ax.set_xticks(x, [name.replace("_", "\\n") for name in METRIC_NAMES], fontsize=9)
            ax.set_ylim(0, 1.13)
            ax.set_ylabel("Mean deterministic score")
            ax.legend(loc="upper right")
            ax.grid(axis="y", alpha=0.22)
            ax.set_title("Figure 5. Aligned configurations compare all five means on one dataset hash", loc="left", weight="bold", fontsize=14)
            plt.tight_layout()
            plt.show()
            """,
        ),
        _code(
            "lesson12-020",
            """
            # ## Failure lab
            answer_good_path_bad = case_score_table[
                (case_score_table["configuration_id"] == "regressed-agent-v0")
                & (case_score_table["case_id"] == "redundant_metric_call")
            ]
            path_good_answer_incomplete = case_score_table[
                (case_score_table["configuration_id"] == "regressed-agent-v0")
                & (case_score_table["case_id"] == "document_fact_without_evidence_id")
            ]
            display(pd.concat([answer_good_path_bad, path_good_answer_incomplete], ignore_index=True))

            diagnostic_rows = []
            for case_id in ("redundant_metric_call", "document_fact_without_evidence_id"):
                score = regressed_summary.case_scores_by_id[case_id]
                diagnostic_rows.append({
                    "case_id": case_id,
                    "path diagnosis": score.tool_call_efficiency.rationale,
                    "answer diagnosis": score.citation_integrity.rationale,
                    "owner": score.failure_stage,
                })
            display(pd.DataFrame(diagnostic_rows))
            print("Answer-good/path-bad: the redundant call keeps a useful answer but lowers trajectory efficiency.")
            print("Path-good/answer-incomplete: the missing evidence ID breaks citation integrity even when the call path is aligned.")

            selected_configuration_id = "bounded-agent-v1"
            selected_case_id = "unsupported_metric_not_recovered"
            persisted_traces = mlflow.search_traces(
                run_id=bounded_summary.run_id,
                locations=[bounded_summary.experiment_id],
                return_type="list",
                flush=True,
            )
            selected_trace = next(
                trace
                for trace in persisted_traces
                if next(
                    span for span in trace.data.spans if span.parent_id is None
                ).inputs["case_id"]
                == selected_case_id
            )
            selected_root = next(
                span for span in selected_trace.data.spans if span.parent_id is None
            )
            selected_trace_id = selected_trace.info.trace_id
            associated_run_id = selected_trace.info.trace_metadata["mlflow.sourceRun"]
            assert selected_root.inputs["configuration_id"] == selected_configuration_id
            assert associated_run_id == bounded_summary.run_id
            assert selected_trace_id == bounded_summary.trace_ids[selected_case_id]
            assert selected_root.outputs["failure_stage"] == "evidence_gate"

            selected_children = sorted(
                (
                    span
                    for span in selected_trace.data.spans
                    if span.parent_id == selected_root.span_id
                ),
                key=lambda span: span.start_time_ns,
            )
            child_order = tuple(span.name for span in selected_children)
            assert child_order == (
                "planning",
                "plan_gate",
                "execution:1",
                "replanning",
                "execution:2",
                "replanning",
                "execution:3",
                "evidence_gate",
                "report",
            )

            failed_trace_rows = []
            for order, span in enumerate((selected_root, *selected_children)):
                inputs = span.inputs if isinstance(span.inputs, dict) else {}
                outputs = span.outputs if isinstance(span.outputs, dict) else {}
                guardrail_evidence = []
                for event in span.events:
                    if event.name != "guardrail":
                        continue
                    guardrail_evidence.append(
                        f"{event.attributes['status']} | {event.attributes['summary']}"
                    )
                phase = (
                    "root"
                    if span.parent_id is None
                    else "execution"
                    if span.name.startswith("execution:")
                    else span.name
                )
                failed_trace_rows.append({
                    "order": order,
                    "span_name": span.name,
                    "span_type": span.span_type,
                    "phase": phase,
                    "public_status": outputs.get(
                        "observed_status", outputs.get("status", "not_recorded")
                    ),
                    "attempt_id": span.attributes.get(
                        "attempt_id", inputs.get("attempt_id")
                    ),
                    "plan_revision": span.attributes.get(
                        "plan_revision", inputs.get("plan_revision")
                    ),
                    "typed_error": outputs.get("error_code") or "none",
                    "guardrail_evidence": "; ".join(guardrail_evidence) or "none",
                })

            failed_trace_table = pd.DataFrame(failed_trace_rows)
            display(failed_trace_table)
            print(f"Selected failed trace configuration: {selected_configuration_id}")
            print(f"Selected failed trace case: {selected_case_id}")
            print(f"Associated run ID: {associated_run_id}")
            print(f"Trace ID: {selected_trace_id}")
            print(f"Root span ID: {selected_root.span_id}")
            print(f"Persisted child order: {' -> '.join(child_order)}")
            print("Typed error evidence: unsupported_metric")
            print("Guardrail evidence: blocked | Execution stopped after the unsupported metric was not recovered.")
            print(f"Failure owner: {selected_root.outputs['failure_stage']}")
            """,
        ),
        _code(
            "lesson12-021",
            """
            diagnosis_rows = [
                ("planner", "missing or wrong expected call", "repair the validated plan"),
                ("tool boundary", "typed error or invalid result", "inspect capability + arguments"),
                ("replanner", "no recovery or excess revision", "replace only unfinished work"),
                ("evidence gate", "briefing emitted without coverage", "block report and retain missing evidence"),
                ("report writer", "missing fact, limitation, or citation", "repair public briefing fields"),
                ("dataset", "version/hash/case mismatch", "stop before partial scoring"),
                ("judge", "provider unavailable or disagreement", "record NOT RUN or observed result"),
            ]
            diagnosis_table = pd.DataFrame(diagnosis_rows, columns=["owner", "public symptom", "next action"])
            display(diagnosis_table)

            fig, ax = plt.subplots(figsize=(14, 7.5))
            ax.axis("off")
            y_positions = np.linspace(0.83, 0.15, len(diagnosis_rows))
            owner_colors = ["#1F40CB", "#00A2EB", "#7B61A8", "#F07D00", "#2E8B57", "#C83737", "#4B6070"]
            for y, (owner, symptom, action), color in zip(y_positions, diagnosis_rows, owner_colors, strict=True):
                ax.add_patch(FancyBboxPatch((0.03, y - 0.045), 0.18, 0.09, boxstyle="round,pad=0.012", facecolor=color, edgecolor=color))
                ax.text(0.12, y, owner.upper(), ha="center", va="center", color="white", weight="bold", fontsize=9)
                ax.add_patch(FancyBboxPatch((0.25, y - 0.045), 0.33, 0.09, boxstyle="round,pad=0.012", facecolor="#F5F5F5", edgecolor=color, linewidth=1.5))
                ax.text(0.415, y, symptom, ha="center", va="center", color="#051C2A", fontsize=8.5)
                ax.add_patch(FancyArrowPatch((0.59, y), (0.64, y), arrowstyle="-|>", mutation_scale=13, color="#4B6070"))
                ax.add_patch(FancyBboxPatch((0.65, y - 0.045), 0.31, 0.09, boxstyle="round,pad=0.012", facecolor="#FFFFFF", edgecolor=color, linewidth=1.5))
                ax.text(0.805, y, action, ha="center", va="center", color="#051C2A", fontsize=8.5)
            ax.text(0.415, 0.94, "PUBLIC FAILURE SIGNAL", ha="center", weight="bold", color="#051C2A")
            ax.text(0.805, 0.94, "OWNED RECOVERY", ha="center", weight="bold", color="#051C2A")
            ax.set_title("Figure 6. Failure diagnosis assigns the earliest public owner", loc="left", weight="bold", fontsize=14)
            plt.tight_layout()
            plt.show()
            """,
        ),
        _code(
            "lesson12-022",
            """
            optional_judge_rows = pd.DataFrame([
                {"provider": "OpenAI", "FINAI_EVAL_JUDGE_MODEL": OPENAI_JUDGE_URI, "core status": "NOT RUN"},
                {"provider": "Ollama", "FINAI_EVAL_JUDGE_MODEL": OLLAMA_JUDGE_URI, "core status": "NOT RUN"},
            ])
            display(optional_judge_rows)
            print("Optional MLflow judges: NOT RUN in the deterministic core route.")
            print(f"OpenAI example: FINAI_EVAL_JUDGE_MODEL={OPENAI_JUDGE_URI}")
            print(f"Ollama example: FINAI_EVAL_JUDGE_MODEL={OLLAMA_JUDGE_URI}")
            print("Run one explicit provider/model only. Missing provider support remains NOT RUN; it never changes deterministic scores or release status.")
            """,
        ),
        _code(
            "lesson12-023",
            """
            # ## Verification
            print(f"Resolved database path: {store.database_path}")
            print(f"Artifact directory: {store.artifact_directory}")
            print("Expected UI URL: http://127.0.0.1:5000")
            print(f"Exact UI command: {store.ui_command}")
            assert store.ui_command.startswith(UI_COMMAND_PREFIX)
            """,
        ),
        _code(
            "lesson12-024",
            """
            assert public_signature(live_reference) == public_signature(recorded_reference)
            assert comparison.dataset_version == "agent-cases-v1"
            assert comparison.dataset_sha256 == dataset.dataset_sha256
            assert comparison.configuration_ids == ("bounded-agent-v1", "regressed-agent-v0")
            assert bounded_summary.run_id != regressed_summary.run_id
            assert all(summary.trace_count == 6 for summary in summaries)
            assert sum(summary.trace_count for summary in summaries) == 12
            assert len(case_score_table) == 12
            assert set(METRIC_NAMES) <= set(case_score_table.columns)
            assert len(mean_table) == 10
            assert not failure_table.empty
            print("LESSON_12_PASS")
            """,
        ),
        _markdown(
            "lesson12-025",
            """
            ## Knowledge check

            1. Why are tool-call correctness and answer relevance separate scores?
            2. Why must both configurations use the same dataset version and exact SHA-256?
            3. What does a public trace add to an aggregate score table?
            4. Why is citation integrity a deterministic release gate for this financial mission?
            5. When should an LLM judge be marked `NOT RUN` instead of being replaced silently?

            ## Challenge

            Add one custom deterministic scorer for maximum latency, forbidden write tools, evidence freshness, mandatory currency and period caveats, or maximum plan revisions. The challenge is complete only when it adds a versioned expectation, a failing regression case, a passing implementation, and a visible MLflow metric. It must not change the core 40-minute route.
            """,
        ),
        _markdown(
            "lesson12-026",
            """
            ## Recap

            Versioned cases make evaluation reproducible. Public traces expose path failures that a good final answer can hide. Separate trajectory and answer metrics make remediation specific. Citation integrity keeps unsupported financial claims out of a release.

            ## Capstone integration

            Lesson 12 contributes versioned regression cases, public agent traces, trajectory and answer scorecards, a citation release gate, aligned MLflow runs, and per-case failure ownership. The instructor and course owner still decide the capstone application surface, interaction model, tool set, document corpus, thresholds, and demonstration mission; no architecture is assumed here.
            """,
        ),
    ]

    original = {cell.id: cell for cell in notebook.cells}

    def combined_code(cell_id: str, *source_ids: str):
        return _code(
            cell_id,
            "\n\n".join(original[source_id].source for source_id in source_ids),
        )

    alignment_without_plot = original["lesson12-009"].source.split("\nfig, ax =", 1)[0]
    diagnosis_without_plot = original["lesson12-021"].source.split("\nfig, ax =", 1)[0]

    notebook.cells = [
        _markdown(
            "lesson12-000",
            f"""
            {original['lesson12-000'].source}

            {original['lesson12-001'].source}

            The practical sequence is **define the exam -> verify one reference -> compare configurations -> diagnose one trace -> decide release**.
            """,
        ),
        combined_code("lesson12-001", "lesson12-002", "lesson12-003"),
        _markdown(
            "lesson12-002",
            """
            ## 1. Define the exam before scoring the agent

            A versioned case states the expected calls, dependencies, evidence, final status, and budget. The exact dataset hash prevents an apparently fair comparison from using different exams.
            """,
        ),
        combined_code("lesson12-003", "lesson12-005", "lesson12-004"),
        _markdown(
            "lesson12-004",
            """
            ## 2. Verify one real reference before trusting fixtures

            Run the Lesson 11 mission once and compare only its public, serializable signature with `reference_completed`.

            **Learner decision:** Which public fields prove that the path and cited answer match without exposing hidden reasoning?
            """,
        ),
        combined_code("lesson12-005", "lesson12-006", "lesson12-007"),
        combined_code("lesson12-006", "lesson12-008"),
        _code(
            "lesson12-007",
            "\n\n".join(
                (
                    alignment_without_plot,
                    original["lesson12-010"].source,
                    original["lesson12-011"].source,
                    original["lesson12-012"].source,
                )
            ),
        ),
        _markdown(
            "lesson12-008",
            """
            ## 3. Compare configurations without hiding failed cases

            Score the answer and trajectory separately, then persist one MLflow run per configuration and one root trace per case.

            **Learner decision:** Which configuration is safer, and which case blocks release even if its aggregate mean looks acceptable?
            """,
        ),
        combined_code("lesson12-009", "lesson12-013", "lesson12-014"),
        combined_code("lesson12-010", "lesson12-015", "lesson12-016"),
        combined_code("lesson12-011", "lesson12-017"),
        combined_code("lesson12-012", "lesson12-018"),
        combined_code("lesson12-013", "lesson12-019"),
        _markdown(
            "lesson12-014",
            """
            ## 4. Open one failed trace and assign ownership

            Aggregates identify that something changed. A trace identifies where it changed: plan, tool boundary, replan, evidence gate, report, dataset, or judge.

            **Learner decision:** Which case blocks release? Who owns the earliest public failure, and what evidence in the trace proves it?
            """,
        ),
        combined_code("lesson12-015", "lesson12-020"),
        _code(
            "lesson12-016",
            "\n\n".join(
                (
                    diagnosis_without_plot,
                    original["lesson12-022"].source,
                    original["lesson12-023"].source,
                )
            ),
        ),
        combined_code("lesson12-017", "lesson12-024"),
        _markdown(
            "lesson12-018",
            f"""
            {original['lesson12-025'].source}

            {original['lesson12-026'].source}
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
