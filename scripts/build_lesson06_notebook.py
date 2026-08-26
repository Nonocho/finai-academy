"""Build the concise Lesson 06 hybrid-retrieval notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "06_hybrid_retrieval.ipynb"


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
        "lesson06-000",
        """
# 06 — Embeddings and Hybrid Retrieval

**First Finance - Arnaud Demes**  
**Day 1 · 15:15–16:00 · 15 minutes deck + 30 minutes guided notebook**

## Learning objectives

By the end of this lab, you can:

1. explain why BM25 and dense retrieval cover different failure modes;
2. apply company and period filters before ranking;
3. combine channel ranks with reciprocal-rank fusion (RRF); and
4. inspect a transparent reranker without calling its score confidence.

The **deterministic offline laboratory success condition** is that the reranked hybrid pipeline
recovers all four maintained evidence tokens while the controlled dense and unfiltered
failures stay visible. **Live contract:** Live OpenAI and Ollama runs report observed recall
and verify only provider-invariant structural behavior.

## Where this fits

```text
Lesson 05                  Lesson 06                         Lesson 07
parse + chunk  →  BM25 + dense → filter → fuse → rerank  →  evaluate + trace
```

The corpus is not synthetic: this notebook rebuilds the seven provenance-preserving
chunks from the NVIDIA FY2026 10-K and Schneider Electric FY2025 results used in
Lesson 05. Only the offline embedding model is deterministic for reproducibility.
        """,
    ),
    code(
        "lesson06-001",
        """
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from finai_academy.chunking import contextualize_chunks, structure_aware_chunks
from finai_academy.documents import load_source_manifest, parse_html, parse_pdf
from finai_academy.hybrid_retrieval import (
    BM25Index, DenseIndex, DeterministicTeachingEmbeddings, IndexedPassage, RetrievalFilters,
)
from finai_academy.lesson_support import compact_manifest_labels
from finai_academy.providers import check_provider_configuration, create_embeddings
from finai_academy.reranking import RERANK_FEATURE_WEIGHTS, rerank_candidates
from finai_academy.retrieval_pipeline import retrieve_evidence, verify_retrieval_runs
from finai_academy.settings import Settings

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_ROOT = REPO_ROOT / "assets" / "course-data"
INDEX_ROOT = Path(os.getenv("FINAI_INDEX_DIR", tempfile.gettempdir())) / "finai-lesson06-index"
COLORS = {"navy": "#051C2A", "blue": "#1F40CB", "cyan": "#00A2EB", "orange": "#F07D00",
          "green": "#2E8B57", "grey": "#64748B", "red": "#C43D3D"}
plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.titleweight": "bold"})

def ranking_panel(axis, title, xlabel, items, expected_token=None, company_colors=None):
    labels = [PASSAGE_LABELS[p.passage_id] for p, _score in items]
    values = [score for _passage, score in items]
    if company_colors:
        colors = [company_colors[p.company] for p, _score in items]
    else:
        colors = [COLORS["green"] if expected_token and expected_token in p.text else COLORS["blue"]
                  for p, _score in items]
    bars = axis.barh(np.arange(len(items)), values, color=colors)
    axis.set_yticks(np.arange(len(items)), labels)
    axis.invert_yaxis(); axis.set_xlabel(xlabel); axis.set_title(title, loc="left")
    axis.spines[["top", "right"]].set_visible(False)
    axis.bar_label(bars, fmt="%.3f", padding=4, fontsize=8)

live_mode = os.getenv("FINAI_LIVE_MODE", "0") == "1"
if live_mode:
    settings = Settings.from_environment()
    problems = check_provider_configuration(settings)
    if problems:
        raise RuntimeError(" ".join(problems))
    embeddings = create_embeddings(settings)
    embedding_provider, embedding_model = settings.embedding_provider, settings.embedding_model
else:
    embeddings = DeterministicTeachingEmbeddings()
    embedding_provider, embedding_model = "offline", embeddings.model_name
print(f"Embedding runtime: {embedding_provider} / {embedding_model}")
        """,
    ),
    code(
        "lesson06-002",
        """
sources = load_source_manifest(DATA_ROOT / "manifest.json")
assert all(source.verify_fixture(REPO_ROOT) for source in sources)

chunks = []
for source in sources:
    fixture = REPO_ROOT / source.fixture_path
    blocks = parse_html(fixture, source) if fixture.suffix == ".html" else parse_pdf(fixture, source)
    chunks.extend(contextualize_chunks(structure_aware_chunks(blocks, max_chars=220)))
assert len(chunks) == 7

passages = tuple(IndexedPassage(
    passage_id=chunk.chunk_id, company=chunk.company, period=chunk.period,
    document_type=chunk.document_type, section=" > ".join(chunk.section_path) or "Document",
    text=chunk.text, source_url=chunk.source_url,
) for chunk in chunks)
PASSAGE_LABELS = compact_manifest_labels(passages)
bm25_index = BM25Index(passages)
dense_index = DenseIndex(
    passages, embeddings, provider=embedding_provider, model=embedding_model,
    chunking_strategy="contextual-structure-v1-max220",
)
dense_index.save(INDEX_ROOT)

EVIDENCE_SET_VERSION = "lesson05-maintained-v1"
EXPECTED_EVIDENCE = {
    "nvda-data-center": {"question": "Which NVIDIA business generated $193.7 billion?", "company": "NVIDIA", "period": "FY2026", "token": "$193.7 billion"},
    "nvda-gaming-growth": {"question": "How fast did NVIDIA Gaming revenue grow?", "company": "NVIDIA", "period": "FY2026", "token": "41%"},
    "se-revenue": {"question": "What was Schneider Electric FY2025 revenue?", "company": "Schneider Electric", "period": "FY2025", "token": "EUR 40.2bn"},
    "se-margin": {"question": "What margin did Schneider adjusted EBITA reach?", "company": "Schneider Electric", "period": "FY2025", "token": "18.7%"},
}
print(f"Rebuilt {len(passages)} passages from {len(sources)} official-source fixtures")
pd.DataFrame([{"passage": PASSAGE_LABELS[p.passage_id], "company": p.company, "period": p.period,
               "section": p.section} for p in passages])
        """,
    ),
    markdown(
        "lesson06-003",
        """
## 1. BM25 and dense retrieval solve different problems

**BM25** rewards exact lexical evidence while correcting for document length and repeated-term
saturation. Dense retrieval compares meaning in an embedding space. Their raw scores are not
comparable, so the pipeline will combine **rank positions**, not score magnitudes.

The first maintained finance question exposes the complementarity: BM25 finds the exact
`$193.7 billion`; the deterministic dense channel understands related business concepts but
ranks the literal-bearing passage lower.
        """,
    ),
    code(
        "lesson06-004",
        """
example_id = "nvda-data-center"
example = EXPECTED_EVIDENCE[example_id]
filters = RetrievalFilters(company=example["company"], period=example["period"])
bm25_ranking = bm25_index.bm25_scores(example["question"], filters)[:4]
dense_ranking = dense_index.cosine_scores(example["question"], filters)[:4]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
ranking_panel(axes[0], "BM25 · exact lexical evidence", "raw BM25 score", bm25_ranking, example["token"])
ranking_panel(axes[1], "Dense · semantic alignment", "raw cosine (not confidence)", dense_ranking, example["token"])
fig.suptitle(f"Figure 1 — {example['question']} · eligible: NVIDIA · FY2026", weight="bold")
fig.text(0.5, 0.01, "Green = passage contains the maintained evidence token", ha="center", color=COLORS["green"])
plt.tight_layout(rect=[0, 0.04, 1, 0.94]); plt.show()
        """,
    ),
    markdown(
        "lesson06-005",
        """
## Failure lab

### 2. Exact numbers can disappear in a semantic representation

The controlled embedding intentionally excludes numeric tokens. Querying only `18.7%`
therefore produces a dense tie, while BM25 preserves the literal. This is a controlled
failure—not a claim that every embedding model always misses every number.
        """,
    ),
    code(
        "lesson06-006",
        """
controlled_embeddings = DeterministicTeachingEmbeddings()
controlled_bm25_index = BM25Index(passages)
controlled_dense_index = DenseIndex(
    passages, controlled_embeddings, provider="controlled-offline",
    model=controlled_embeddings.model_name, chunking_strategy="contextual-structure-v1-max220",
)
exact_query = "18.7%"
exact_bm25 = controlled_bm25_index.bm25_scores(exact_query)[:4]
exact_dense = controlled_dense_index.cosine_scores(exact_query)[:4]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
ranking_panel(axes[0], "BM25", "raw BM25 score", exact_bm25, exact_query)
ranking_panel(axes[1], "Dense", "raw cosine similarity", exact_dense, exact_query)
fig.suptitle("Figure 2 — Exact-number query: '18.7%' · lexical signal versus dense tie", weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.93]); plt.show()

assert exact_query in exact_bm25[0][0].text
assert exact_query not in exact_dense[0][0].text
assert all(score == 0.0 for _passage, score in exact_dense)
print("BM25 exact-term recovery reproduced")
print("Dense exact-term failure reproduced")
        """,
    ),
    markdown(
        "lesson06-007",
        """
### 3. Eligibility must be enforced before ranking

Company, period, document type and section are structured constraints—not query hints.
Post-filtering a global top-k cannot recover eligible evidence that never entered the candidate
set. An empty eligible set should abstain rather than silently broaden the search.
        """,
    ),
    code(
        "lesson06-008",
        """
leakage_query = "energy management organic growth"
unfiltered = controlled_dense_index.cosine_scores(leakage_query)[:4]
nvidia_filter = RetrievalFilters(company="NVIDIA", period="FY2026")
filtered = controlled_dense_index.cosine_scores(leakage_query, nvidia_filter)[:4]
company_colors = {"NVIDIA": COLORS["blue"], "Schneider Electric": COLORS["orange"]}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.3))
ranking_panel(axes[0], "No eligibility filter", "raw cosine similarity", unfiltered, company_colors=company_colors)
ranking_panel(axes[1], "Pre-filter: NVIDIA · FY2026", "raw cosine similarity", filtered, company_colors=company_colors)
fig.suptitle(f"Figure 3 — Controlled cross-company query: {leakage_query!r}", weight="bold")
fig.text(0.5, 0.01, "Blue = NVIDIA · Orange = Schneider Electric", ha="center")
plt.tight_layout(rect=[0, 0.04, 1, 0.93]); plt.show()

assert unfiltered[0][0].company == "Schneider Electric"
assert {passage.company for passage, _score in filtered} == {"NVIDIA"}
print("Cross-company leakage blocked")
        """,
    ),
    markdown(
        "lesson06-009",
        """
## 4. Reciprocal-rank fusion combines positions—not incompatible scores

BM25 and cosine operate on different scales. RRF gives each eligible passage a contribution
`weight / (k + rank)` from each channel, deduplicates by stable passage ID and sums the
contributions. A weight is a policy to evaluate, not a probability.
        """,
    ),
    code(
        "lesson06-010",
        """
retrieval_runs = {
    question_id: retrieve_evidence(
        item["question"], keyword_index=bm25_index, dense_index=dense_index,
        filters=RetrievalFilters(company=item["company"], period=item["period"]),
        candidate_k=4, final_k=2,
    )
    for question_id, item in EXPECTED_EVIDENCE.items()
}
run = retrieval_runs[example_id]
fusion_hits = run.fused_hits[:4]
labels = [PASSAGE_LABELS[hit.passage.passage_id] for hit in fusion_hits]
keyword_parts, dense_parts = [], []
for hit in fusion_hits:
    ranks = dict(hit.channel_ranks)
    keyword_parts.append(1 / (60 + ranks["keyword"]) if "keyword" in ranks else 0)
    dense_parts.append(1 / (60 + ranks["dense"]) if "dense" in ranks else 0)

fig, ax = plt.subplots(figsize=(12.5, 5.6))
y = np.arange(len(labels))
ax.barh(y, keyword_parts, color=COLORS["orange"], label="BM25 rank contribution")
ax.barh(y, dense_parts, left=keyword_parts, color=COLORS["cyan"], label="dense rank contribution")
ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlabel("RRF score")
ax.set_title("Figure 4 — RRF makes each channel contribution inspectable", loc="left")
ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.show()
        """,
    ),
    markdown(
        "lesson06-011",
        """
## 5. Rerank a smaller eligible set and measure the outcome

The transparent teaching reranker combines lexical coverage, exact numeric coverage, section
overlap, metadata eligibility and the fusion signal. Numeric coverage carries the largest
weight in this finance lab. The weighted sum is a **rerank score**, not confidence.
        """,
    ),
    code(
        "lesson06-012",
        """
reranked = rerank_candidates(example["question"], run.fused_hits, top_k=len(run.fused_hits))
feature_names = list(RERANK_FEATURE_WEIGHTS)
feature_colors = [COLORS["blue"], COLORS["orange"], COLORS["cyan"], COLORS["green"], COLORS["grey"]]
labels = [PASSAGE_LABELS[hit.passage.passage_id] for hit in reranked]
fig, ax = plt.subplots(figsize=(13, 5.8)); left = np.zeros(len(reranked))
for name, color in zip(feature_names, feature_colors, strict=True):
    values = np.asarray([getattr(hit.features, name) * RERANK_FEATURE_WEIGHTS[name] for hit in reranked])
    ax.barh(np.arange(len(labels)), values, left=left, color=color,
            label=f"{name} × {RERANK_FEATURE_WEIGHTS[name]:.2f}")
    left += values
ax.set_yticks(np.arange(len(labels)), labels); ax.invert_yaxis(); ax.set_xlabel("rerank score")
ax.set_title("Figure 5 — Every rerank score is the sum of visible evidence features", loc="left")
ax.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
ax.spines[["top", "right"]].set_visible(False); plt.tight_layout(); plt.show()

def contains_expected(passage, expected):
    return expected["token"] in passage.text

stage_success = {"BM25": [], "Dense": [], "RRF fusion": [], "Reranked hybrid": []}
for question_id, expected in EXPECTED_EVIDENCE.items():
    result = retrieval_runs[question_id]
    for stage, hits in [("BM25", result.keyword_hits), ("Dense", result.dense_hits),
                        ("RRF fusion", result.fused_hits), ("Reranked hybrid", result.reranked_hits)]:
        stage_success[stage].append(contains_expected(hits[0].passage, expected))
score_matrix = np.asarray(list(stage_success.values()), dtype=float)
fig, ax = plt.subplots(figsize=(11.5, 5.2))
ax.imshow(score_matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(EXPECTED_EVIDENCE)), list(EXPECTED_EVIDENCE), rotation=18, ha="right")
ax.set_yticks(range(len(stage_success)), list(stage_success))
for row in range(score_matrix.shape[0]):
    for column in range(score_matrix.shape[1]):
        ax.text(column, row, "PASS" if score_matrix[row, column] else "MISS",
                ha="center", va="center", color="white" if score_matrix[row, column] else COLORS["red"], weight="bold")
ax.set_title("Figure 6 — Maintained evidence recovered at rank 1", loc="left")
plt.tight_layout(); plt.show()

maintained_recall = {stage: sum(values) / len(values) for stage, values in stage_success.items()}
if not live_mode:
    assert maintained_recall["Reranked hybrid"] > maintained_recall["Dense"]
    print("Hybrid retrieval improves maintained recall")
else:
    print("Live provider recall observed — no provider-specific ranking threshold asserted")
print(pd.Series(maintained_recall, name="recall@1").to_string())
        """,
    ),
    markdown(
        "lesson06-013",
        """
## Challenge

Increase only the BM25 RRF weight from `1.0` to `3.0`, keep dense at `1.0`, and observe
which maintained rankings move. A moved rank proves the policy has an effect—not that it is
better. Improvement requires a maintained evaluation set.

## Verification

The laboratory passes only when provenance, index identity, eligibility, controlled failures
and maintained evidence recovery remain observable. A high similarity score alone is never an
acceptance criterion.
        """,
    ),
    code(
        "lesson06-014",
        """
moved_rankings = []
for question_id, expected in EXPECTED_EVIDENCE.items():
    baseline = [hit.passage.passage_id for hit in retrieval_runs[question_id].fused_hits]
    weighted = retrieve_evidence(
        expected["question"], keyword_index=bm25_index, dense_index=dense_index,
        filters=RetrievalFilters(company=expected["company"], period=expected["period"]),
        candidate_k=4, final_k=2, weights={"keyword": 3.0, "dense": 1.0},
    )
    if [hit.passage.passage_id for hit in weighted.fused_hits] != baseline:
        moved_rankings.append(question_id)
print("Moved rankings:", ", ".join(moved_rankings) if moved_rankings else "none for this provider")

verification = verify_retrieval_runs(
    retrieval_runs,
    expected_evidence={question_id: item["token"] for question_id, item in EXPECTED_EVIDENCE.items()},
    require_expected_evidence=not live_mode,
    required_artifacts=(INDEX_ROOT / "manifest.json", INDEX_ROOT / "vectors.npy"),
)
checks = dict(verification.checks)
checks.update({
    "seven manifest-derived passages": len(passages) == 7,
    "provider vectors are finite": dense_index.dimension > 0 and np.isfinite(dense_index.document_matrix).all(),
    "provider filter blocks cross-company passages": {p.company for p, _ in dense_index.cosine_scores(leakage_query, nvidia_filter)} == {"NVIDIA"},
})
for label, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL'} — {label}")
assert all(checks.values())
print("PASS — hybrid retrieval laboratory verified")
        """,
    ),
    markdown(
        "lesson06-015",
        """
## Capstone integration

```text
versioned Lesson 05 chunks
  → BM25 + versioned embedding index
  → company/period pre-filter
  → reciprocal-rank fusion
  → transparent evidence reranker
  → final provenance-preserving passages
```

## Recap

- BM25 protects exact numbers, names and identifiers; dense retrieval protects semantic recall.
- Metadata filters define eligibility before either channel scores candidates.
- RRF combines rank positions without pretending raw scores share one scale.
- Reranking deliberately spends a smaller evidence budget and must remain inspectable.
- The offline model is reproducible; OpenAI and Ollama are live observations through the same gateway.

**Next — Lesson 07:** evaluate each retrieval stage and trace the answer path.
        """,
    ),
]

nbformat.write(notebook, OUTPUT)
print(f"Wrote {OUTPUT}")
