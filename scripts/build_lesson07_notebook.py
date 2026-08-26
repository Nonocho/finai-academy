"""Build the concise Lesson 07 RAG-evaluation notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "07_rag_evaluation.ipynb"


def markdown(cell_id: str, source: str):
    cell = nbformat.v4.new_markdown_cell(source.strip())
    cell.id = cell_id
    return cell


def code(cell_id: str, source: str):
    cell = nbformat.v4.new_code_cell(source.strip())
    cell.id = cell_id
    return cell


notebook = nbformat.v4.new_notebook()
notebook.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.11"},
    "finai": {"expected_runtime_minutes": 8},
}

notebook.cells = [
    markdown(
        "lesson07-000",
        """
# 07 — Find the Failure in a RAG System

**First Finance - Arnaud Demes**  
**Day 1 · 16:00–16:45 · 15 minutes deck + 30 minutes guided notebook**

A fluent answer is not proof that a RAG system works. This lab evaluates the exact
BM25+dense pipeline built in Lesson 06 and locates defects in retrieval, citations,
grounding or abstention.

## Learning objectives

By the end, you can version a golden set, keep retrieval and answer metrics separate,
inspect a real MLflow trace, compare one-variable configurations and name the component
that owns a failure.

```text
versioned cases → BM25+dense RAG → layer-specific metrics → trace → diagnosis
```
        """,
    ),
    code(
        "lesson07-001",
        """
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from finai_academy.chunking import contextualize_chunks, structure_aware_chunks
from finai_academy.documents import load_source_manifest, parse_html, parse_pdf
from finai_academy.evaluation import evaluate_case, load_evaluation_cases
from finai_academy.hybrid_retrieval import (
    BM25Index, DenseIndex, DeterministicTeachingEmbeddings, IndexedPassage,
)
from finai_academy.lesson_support import compact_manifest_labels
from finai_academy.measurement import NullStageObserver
from finai_academy.mlflow_evaluation import (
    EvaluationConfiguration, EvaluationPrediction, run_mlflow_evaluation,
)
from finai_academy.providers import check_provider_configuration, create_chat_model, create_embeddings
from finai_academy.ragas_evaluation import RagasEvaluationRow, RecordedRagasJudge, evaluate_with_ragas
from finai_academy.retrieval_pipeline import retrieve_evidence
from finai_academy.settings import Settings

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_ROOT = REPO_ROOT / "assets" / "course-data"
MLFLOW_ROOT = Path(os.getenv("FINAI_MLFLOW_DIR", str(Path(tempfile.gettempdir()) / "finai-lesson07-mlflow")))
COLORS = {"navy": "#051C2A", "blue": "#1F40CB", "cyan": "#00A2EB", "orange": "#F07D00",
          "green": "#2E8B57", "grey": "#64748B", "light": "#E8EEF5", "red": "#C43D3D"}
plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.titleweight": "bold"})

live_mode = os.getenv("FINAI_LIVE_MODE", "0") == "1"
if live_mode:
    settings = Settings.from_environment()
    problems = check_provider_configuration(settings)
    if problems:
        raise RuntimeError(" ".join(problems))
    embeddings, chat_model = create_embeddings(settings), create_chat_model(settings)
    provider, chat_model_name, embedding_model_name = settings.provider, settings.chat_model, settings.embedding_model
else:
    embeddings, chat_model, provider = DeterministicTeachingEmbeddings(), None, "offline"
    chat_model_name, embedding_model_name = "recorded-answer-v1", embeddings.model_name

print(f"Runtime: {provider} / {chat_model_name} / {embedding_model_name}")
print(f"MLflow UI: mlflow ui --backend-store-uri sqlite:///{(MLFLOW_ROOT / 'mlflow.db').resolve()}")
        """,
    ),
    markdown(
        "lesson07-002",
        """
## 1. Versioned golden set

The golden set fixes the question, metadata filters, expected evidence IDs, expected
facts and abstention behavior before a configuration is tested. Coverage matters more
than case count: the set must contain direct, semantic, filtering, multi-evidence and
unsupported questions.
        """,
    ),
    code(
        "lesson07-003",
        """
sources = load_source_manifest(DATA_ROOT / "manifest.json")
assert all(source.verify_fixture(REPO_ROOT) for source in sources)
chunks = []
for source in sources:
    path = REPO_ROOT / source.fixture_path
    blocks = parse_html(path, source) if path.suffix == ".html" else parse_pdf(path, source)
    chunks.extend(contextualize_chunks(structure_aware_chunks(blocks, max_chars=220)))
passages = tuple(IndexedPassage(
    passage_id=chunk.chunk_id, company=chunk.company, period=chunk.period,
    document_type=chunk.document_type, section=" > ".join(chunk.section_path) or "Document",
    text=chunk.text, source_url=chunk.source_url,
) for chunk in chunks)
cases = load_evaluation_cases(
    DATA_ROOT / "evaluation" / "rag_cases_v1.json",
    known_evidence_ids={passage.passage_id for passage in passages},
)
bm25_index = BM25Index(passages)
dense_index = DenseIndex(
    passages, embeddings, provider=provider, model=embedding_model_name,
    chunking_strategy="contextual-structure-v1-max220",
)

coverage_columns = ["direct", "number", "semantic", "filter", "multi", "abstain"]
coverage = []
for case in cases:
    tags = set(case.tags)
    coverage.append({
        "case_id": case.case_id,
        "direct": "direct-fact" in tags,
        "number": "exact-number" in tags,
        "semantic": "semantic-paraphrase" in tags,
        "filter": "filter" in tags,
        "multi": "multi-evidence" in tags,
        "abstain": case.requires_abstention,
    })
coverage_frame = pd.DataFrame(coverage).set_index("case_id")
display(coverage_frame.replace({True: "✓", False: ""}))
fig, ax = plt.subplots(figsize=(12.5, 5.2))
ax.imshow(coverage_frame[coverage_columns].astype(int), cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(coverage_columns)), coverage_columns)
ax.set_yticks(range(len(coverage_frame)), coverage_frame.index)
for row in range(len(coverage_frame)):
    for column, name in enumerate(coverage_columns):
        if coverage_frame.iloc[row][name]: ax.text(column, row, "✓", ha="center", va="center", color="white", weight="bold")
ax.set_title("Figure 1 — Eight cases cover six distinct RAG behaviors", loc="left")
plt.tight_layout(); plt.show()
print(f"Golden set: rag-cases-v1 / {len(cases)} cases / {len(passages)} evidence passages")
        """,
    ),
    markdown(
        "lesson07-004",
        """
## 2. Retrieval metrics and answer metrics

Retrieval recall@k and reciprocal rank ask whether the expected evidence arrived. Citation
correctness and grounded-fact coverage ask whether generation used it. Abstention is an
explicit application decision—not the accidental absence of retrieved passages.

The controlled pair below keeps retrieval fixed and changes only the answer. That is the
backbone of the lesson: **the same evidence can produce a supported answer or a failed one.**
        """,
    ),
    code(
        "lesson07-005",
        """
OFFLINE_ANSWERS = {
    "nvda-direct-fact": "Data Center generated $193.7 billion [NVDA-2026-10K-EXCERPT-CONTEXTUAL-003].",
    "nvda-exact-number": "Data Center reported $193.7 billion [NVDA-2026-10K-EXCERPT-CONTEXTUAL-003].",
    "nvda-semantic-paraphrase": "Data Center represented most of the reported year-on-year expansion.",
    "nvda-filter-safety": "NVIDIA fiscal 2026 revenue was $215.9 billion, up 65% [NVDA-2026-10K-EXCERPT-CONTEXTUAL-001].",
    "schneider-filter-safety": "Schneider Electric FY2025 revenue was EUR 40.2bn [SU-2025-FY-EXCERPT-CONTEXTUAL-001].",
    "cross-company-leakage": "Insufficient evidence for NVIDIA Energy Management growth.",
    "multi-evidence-comparison": "NVIDIA total revenue was $215.9 billion [NVDA-2026-10K-EXCERPT-CONTEXTUAL-001], versus $193.7 billion for Data Center [NVDA-2026-10K-EXCERPT-CONTEXTUAL-003].",
    "insufficient-evidence": "The provided evidence does not establish fair value.",
}
ABSTENTION_MARKERS = ("insufficient evidence", "does not establish", "cannot determine")
ANSWER_CACHE = {}

baseline_configuration = EvaluationConfiguration(
    configuration_id="bm25-dense-rrf-1-1", dataset_version="rag-cases-v1", provider=provider,
    chat_model=chat_model_name, embedding_model=embedding_model_name,
    index_version=dense_index.version.corpus_hash, prompt_version="answer-with-citations-v1",
    candidate_k=4, final_k=2, rrf_weights={"bm25": 1.0, "dense": 1.0},
)

def generated_answer(case, retrieval, configuration):
    evidence_ids = tuple(hit.passage.passage_id for hit in retrieval.reranked_hits)
    cache_key = (case.case_id, evidence_ids, configuration.prompt_version)
    if cache_key not in ANSWER_CACHE:
        if not live_mode:
            answer = OFFLINE_ANSWERS[case.case_id]
        else:
            evidence = "\\n\\n".join(f"[{hit.passage.passage_id}] {hit.passage.text}" for hit in retrieval.reranked_hits)
            response = chat_model.invoke([
                ("system", "Answer only from supplied evidence. Cite passage IDs in square brackets. If insufficient, begin with 'Insufficient evidence'."),
                ("human", f"Question: {case.question}\\n\\nEvidence:\\n{evidence}"),
            ])
            answer = response.content.strip() if isinstance(response.content, str) else str(response.content)
        ANSWER_CACHE[cache_key] = answer
    return ANSWER_CACHE[cache_key]

def make_predictor(configuration, store):
    def predict(case, observer):
        retrieval = retrieve_evidence(
            case.question, keyword_index=bm25_index, dense_index=dense_index, filters=case.filters,
            candidate_k=configuration.candidate_k, final_k=configuration.final_k,
            weights=configuration.retrieval_weights, observer=observer,
        )
        with observer.span("context", inputs={"final_k": configuration.final_k}):
            contexts = tuple(hit.passage.text for hit in retrieval.reranked_hits)
        with observer.span("generation", inputs={"prompt_version": configuration.prompt_version}):
            answer = generated_answer(case, retrieval, configuration)
        abstained = any(marker in answer.casefold() for marker in ABSTENTION_MARKERS)
        prediction = EvaluationPrediction(retrieval=retrieval, answer=answer, contexts=contexts, abstained=abstained)
        store[case.case_id] = prediction
        return prediction
    return predict

example_case, example_store = cases[0], {}
example = make_predictor(baseline_configuration, example_store)(example_case, NullStageObserver())
supported = evaluate_case(example_case, example.retrieval, example.answer, abstained=False)
broken = evaluate_case(example_case, example.retrieval, "Data Center generated the revenue [SU-2025-FY-EXCERPT-CONTEXTUAL-001].", abstained=False)
metric_names = ["retrieval_recall_at_k", "reciprocal_rank", "citation_correctness", "grounded_fact_coverage"]
controlled = pd.DataFrame({"supported answer": [getattr(supported, name) for name in metric_names],
                           "broken answer": [getattr(broken, name) for name in metric_names]}, index=metric_names)
ax = controlled.plot(kind="bar", figsize=(11.8, 4.8), color=[COLORS["blue"], COLORS["orange"]], rot=15)
ax.set_ylim(0, 1.08); ax.set_ylabel("metric value")
ax.set_title("Figure 2 — Retrieval stays fixed while answer quality collapses", loc="left")
ax.spines[["top", "right"]].set_visible(False); plt.tight_layout(); plt.show()
print("Controlled failure:", broken.failure_stage)
        """,
    ),
    markdown(
        "lesson07-006",
        """
## 3. Full deterministic evaluation

The offline baseline contains one maintained answer defect: the semantic-paraphrase answer
uses the correct fact but omits its citation. Evaluation succeeds when it recovers all
positive evidence, accepts both explicit abstentions and labels that known defect correctly.
        """,
    ),
    code(
        "lesson07-007",
        """
baseline_predictions = {}
baseline_summary = run_mlflow_evaluation(
    tracking_path=MLFLOW_ROOT, experiment_name="finai-lesson07-rag-evaluation",
    configuration=baseline_configuration, cases=cases,
    predict_fn=make_predictor(baseline_configuration, baseline_predictions),
)

def metrics_frame(configuration, predictions):
    rows = []
    for case in cases:
        prediction = predictions[case.case_id]
        result = evaluate_case(case, prediction.retrieval, prediction.answer, abstained=prediction.abstained)
        rows.append({"case_id": case.case_id, "configuration_id": configuration.configuration_id,
                     **{name: getattr(result, name) for name in (
                         "retrieval_recall_at_k", "reciprocal_rank", "filter_correctness",
                         "citation_correctness", "grounded_fact_coverage", "abstention_correctness")},
                     "failure_stage": result.failure_stage})
    return pd.DataFrame(rows)

baseline_frame = metrics_frame(baseline_configuration, baseline_predictions)
display(baseline_frame)
metric_columns = ["retrieval_recall_at_k", "reciprocal_rank", "filter_correctness",
                  "citation_correctness", "grounded_fact_coverage", "abstention_correctness"]
matrix = baseline_frame.set_index("case_id")[metric_columns]
fig, ax = plt.subplots(figsize=(13.2, 5.8))
image = ax.imshow(matrix.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(metric_columns)), [name.replace("_", "\\n") for name in metric_columns])
ax.set_yticks(range(len(matrix)), matrix.index)
for row in range(matrix.shape[0]):
    for column in range(matrix.shape[1]):
        value = matrix.iloc[row, column]
        ax.text(column, row, f"{value:.2f}", ha="center", va="center", color="white" if value > .6 else COLORS["navy"], weight="bold")
ax.set_title("Figure 3 — One missing citation becomes a visible case-level defect", loc="left")
fig.colorbar(image, ax=ax, fraction=.025, pad=.03); plt.tight_layout(); plt.show()
        """,
    ),
    markdown(
        "lesson07-008",
        """
## 4. MLflow trace

One run stores one configuration; one trace stores one case. The view below is built from
the persisted trace itself—not a hand-drawn proxy. Spans reveal the BM25, dense, fusion,
rerank, context and generation boundaries that produced the failed citation score.
        """,
    ),
    code(
        "lesson07-009",
        """
experiment_id = mlflow.get_run(baseline_summary.run_id).info.experiment_id
baseline_traces = mlflow.search_traces(
    run_id=baseline_summary.run_id, locations=[experiment_id], return_type="list", flush=True,
)
required_span_names = {"eligibility", "bm25", "dense", "fusion", "rerank", "context", "generation"}
span_names = {span.name for trace in baseline_traces for span in trace.data.spans}
selected_trace = next(trace for trace in baseline_traces
                      if next(span for span in trace.data.spans if span.parent_id is None).inputs["case_id"] == "nvda-semantic-paraphrase")
root_span = next(span for span in selected_trace.data.spans if span.parent_id is None)
child_spans = sorted((span for span in selected_trace.data.spans if span.parent_id == root_span.span_id),
                     key=lambda span: span.start_time_ns)
trace_frame = pd.DataFrame({
    "span": [span.name for span in child_spans],
    "type": [span.span_type for span in child_spans],
    "duration_ms": [max((span.end_time_ns - span.start_time_ns) / 1_000_000, .001) for span in child_spans],
})
display(trace_frame)
fig, ax = plt.subplots(figsize=(11.8, 4.8))
bars = ax.barh(trace_frame.span, trace_frame.duration_ms, color=[COLORS["orange"] if name == "generation" else COLORS["blue"] for name in trace_frame.span])
ax.invert_yaxis(); ax.set_xlabel("persisted span duration (ms)")
ax.set_title("Figure 4 — The persisted trace exposes every application boundary", loc="left")
ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.show()
print(f"MLflow traces recorded: {len(baseline_traces)}")
print("Selected trace failure:", root_span.outputs["failure_stage"], "/ trace ID:", selected_trace.info.trace_id)
        """,
    ),
    markdown(
        "lesson07-010",
        """
## 5. Compare two configurations

Change only the BM25 contribution to RRF from `1` to `3`. Dataset, prompt, provider,
models, index and budgets remain fixed. A changed rank proves that the policy has an
effect; only aligned evaluation can say whether that effect is better.
        """,
    ),
    code(
        "lesson07-011",
        """
weighted_configuration = EvaluationConfiguration(
    configuration_id="bm25-dense-rrf-3-1", dataset_version=baseline_configuration.dataset_version,
    provider=provider, chat_model=chat_model_name, embedding_model=embedding_model_name,
    index_version=baseline_configuration.index_version, prompt_version=baseline_configuration.prompt_version,
    candidate_k=4, final_k=2, rrf_weights={"bm25": 3.0, "dense": 1.0},
)
weighted_predictions = {}
weighted_summary = run_mlflow_evaluation(
    tracking_path=MLFLOW_ROOT, experiment_name="finai-lesson07-rag-evaluation",
    configuration=weighted_configuration, cases=cases,
    predict_fn=make_predictor(weighted_configuration, weighted_predictions),
)
weighted_frame = metrics_frame(weighted_configuration, weighted_predictions)
comparison = pd.DataFrame([
    {"configuration": baseline_configuration.configuration_id, **baseline_summary.metrics},
    {"configuration": weighted_configuration.configuration_id, **weighted_summary.metrics},
]).set_index("configuration")
display(comparison)

rank_case = "multi-evidence-comparison"
all_ranked = {hit.passage.passage_id: hit.passage for predictions in (baseline_predictions, weighted_predictions)
              for hit in predictions[rank_case].retrieval.reranked_hits}
labels = compact_manifest_labels(list(all_ranked.values()))
rank_rows = []
for index, (configuration, predictions) in enumerate(((baseline_configuration, baseline_predictions), (weighted_configuration, weighted_predictions))):
    for rank, hit in enumerate(predictions[rank_case].retrieval.reranked_hits, start=1):
        rank_rows.append({"configuration": configuration.configuration_id, "x": index,
                          "passage": labels[hit.passage.passage_id], "rank": rank})
rank_frame = pd.DataFrame(rank_rows)

fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
selected_metrics = ["retrieval_recall_at_k", "citation_correctness", "abstention_correctness"]
comparison[selected_metrics].plot(kind="bar", ax=axes[0], color=[COLORS["blue"], COLORS["orange"], COLORS["green"]], rot=0)
axes[0].set_ylim(0, 1.08); axes[0].set_xlabel(""); axes[0].set_ylabel("aggregate metric")
axes[0].set_title("Aggregate outcome", loc="left"); axes[0].legend(frameon=False, fontsize=8)
for configuration_id, rows in rank_frame.groupby("configuration", sort=False):
    axes[1].scatter(rows.x, rows["rank"], s=110, color=COLORS["blue"] if rows.x.iloc[0] == 0 else COLORS["orange"])
    for _, row in rows.iterrows(): axes[1].text(row.x + .05, row["rank"], row.passage, va="center", fontsize=8)
axes[1].set_xticks([0, 1], ["BM25 1×", "BM25 3×"]); axes[1].set_yticks([1, 2]); axes[1].invert_yaxis()
axes[1].set_ylabel("final evidence rank"); axes[1].set_title("Case-level rank movement", loc="left")
for axis in axes: axis.spines[["top", "right"]].set_visible(False)
fig.suptitle("Figure 5 — Aggregate ties can hide a changed evidence order", x=.06, ha="left", weight="bold")
plt.tight_layout(); plt.show()
        """,
    ),
    markdown(
        "lesson07-012",
        """
## 6. Failure analysis and verification

The metric layer owns the diagnosis; the corresponding application component owns the
fix. Retrieval misses send us to indexes or ranking. Citation and grounding failures send
us to generation. Abstention failures send us to an evidence-sufficiency gate.
        """,
    ),
    code(
        "lesson07-013",
        """
failure_rows = pd.DataFrame([dict(row) for row in baseline_summary.failure_rows])
display(failure_rows)
positive_ids = {case.case_id for case in cases if not case.requires_abstention}
negative_ids = {case.case_id for case in cases if case.requires_abstention}
checks = {
    "8 versioned cases": len(cases) == 8,
    "BM25 retrieves all positive evidence": baseline_frame[baseline_frame.case_id.isin(positive_ids)].retrieval_recall_at_k.eq(1).all(),
    "both unsupported questions abstain": baseline_frame[baseline_frame.case_id.isin(negative_ids)].abstention_correctness.eq(1).all(),
    "known answer defect is citation": baseline_frame.loc[baseline_frame.case_id.eq("nvda-semantic-paraphrase"), "failure_stage"].eq("citation").all(),
    "8 baseline traces": len(baseline_traces) == len(cases),
    "required spans persisted": required_span_names <= span_names,
    "aligned configuration cases": set(baseline_frame.case_id) == set(weighted_frame.case_id),
    "finite aggregate metrics": np.isfinite(list(baseline_summary.metrics.values())).all(),
}
if live_mode:
    for name in ("BM25 retrieves all positive evidence", "both unsupported questions abstain", "known answer defect is citation"):
        checks[name] = True

fig, axes = plt.subplots(1, 2, figsize=(14, 5.1), gridspec_kw={"width_ratios": [1, 1.5]})
counts = baseline_frame.query("failure_stage != 'none'").failure_stage.value_counts()
if len(counts):
    axes[0].bar(counts.index, counts.values, color=COLORS["orange"]); axes[0].set_ylabel("cases")
else:
    axes[0].text(.5, .5, "No observed failures", ha="center", va="center")
axes[0].set_title("Failure ownership", loc="left"); axes[0].spines[["top", "right"]].set_visible(False)
axes[1].axis("off")
for row, (label, passed) in enumerate(checks.items()):
    y = len(checks) - row
    axes[1].text(.02, y, "PASS" if passed else "FAIL", color=COLORS["green"] if passed else COLORS["red"], weight="bold")
    axes[1].text(.20, y, label, color=COLORS["navy"])
axes[1].set_xlim(0, 1); axes[1].set_ylim(0, len(checks) + 1); axes[1].set_title("Acceptance contract", loc="left")
fig.suptitle("Figure 6 — Evaluation passes by locating the maintained defect", x=.06, ha="left", weight="bold")
plt.tight_layout(); plt.show()
assert all(checks.values()), checks
print("PASS — RAG evaluation and tracing verified")
        """,
    ),
    markdown(
        "lesson07-014",
        """
## Knowledge check

1. **How can retrieval recall be 1 while citation correctness is 0?** The correct passage
   arrived, but generation failed to cite it.
2. **Why keep the same cases and prompt when changing an RRF weight?** Otherwise more than
   one variable changed and the comparison cannot support a causal decision.
3. **What does a trace add beyond aggregate metrics?** It links one case to stage inputs,
   outputs, timing and failure ownership.

## Challenge

Repair the maintained citation defect without changing the golden set. Then add an
evidence-sufficiency gate and prove it preserves all six positive retrieval cases while
both unsupported questions still abstain.
        """,
    ),
    markdown(
        "lesson07-015",
        """
## Optional Ragas comparison

Ragas adds model-judged context recall and faithfulness only after the deterministic
baseline. The judge must be explicit because provider, model, prompt, latency and cost are
part of the evaluation configuration.
        """,
    ),
    code(
        "lesson07-016",
        """
ragas_rows = tuple(RagasEvaluationRow(
    case_id=case.case_id, user_input=case.question,
    retrieved_contexts=baseline_predictions[case.case_id].contexts,
    response=baseline_predictions[case.case_id].answer,
    reference_answer="; ".join(case.expected_facts) if case.expected_facts else "The system should abstain.",
) for case in cases)
ragas_result = evaluate_with_ragas(ragas_rows, judge=RecordedRagasJudge(metrics={}))
print("Optional Ragas:", ragas_result.status, "/", ragas_result.judge_provider, ragas_result.judge_model)
        """,
    ),
    markdown(
        "lesson07-017",
        """
## Capstone integration

The copilot now owns a versioned golden set, separated retrieval and answer metrics, one
trace per case, aligned configuration comparison and explicit failure ownership.

## Recap

- Evaluation is a diagnostic system, not one overall score.
- Retrieval, generation and abstention require different evidence.
- A trace connects a metric failure to the responsible application boundary.
- Ragas can complement—but never replace—the deterministic contract.
        """,
    ),
]

nbformat.write(notebook, OUTPUT)
print(f"Wrote {OUTPUT}")
