# Lesson 06 — Embeddings and Hybrid Retrieval

**Schedule:** Day 1, 15:15–16:00

**Format:** 15 minutes of concepts and diagrams, 30 minutes of guided notebook work

**Capstone increment:** filtered lexical+dense retrieval, reciprocal-rank fusion and
transparent reranking over a versioned local embedding index

## Teaching outcome

Students should leave with one operational principle:

> Retrieval quality is a staged, testable system property. A nearest neighbor is not yet
> eligible, complete or safe evidence.

The notebook rebuilds the seven contextual structure-aware chunks from Lesson 05's
versioned source manifest. It then embeds, pre-filters, retrieves, fuses and reranks those
same chunks. Nothing is silently replaced with the smaller unit-test corpus.

## Connection to Lessons 05 and 07

Lesson 05 established trustworthy evidence units. Lesson 06 holds those units constant and
changes their representation and ranking:

```text
Lesson 05                         Lesson 06                         Lesson 07
manifest → blocks → chunks  →  index → filter → rank → rerank  →  evaluate → trace
```

Lesson 07 will treat each maintained question and expected-evidence token as an evaluation
case. Its traces should expose the stages created here rather than only the final answer.

## Exact 15-minute deck pacing

| Time | Concept | Instructor move | Expected learner statement |
|---:|---|---|---|
| 0:00–1:30 | Lesson boundary | Point from trusted Lesson 05 chunks to the new retrieval stages. | “The corpus stays fixed; representation and ranking change.” |
| 1:30–3:30 | Embedding geometry | Define a vector and the dot product of normalized vectors. | “Cosine similarity measures alignment, not confidence.” |
| 3:30–5:00 | 2D projection caveat | Contrast a teaching projection with the full-dimensional score. | “The map is explanatory; retrieval uses the original dimensions.” |
| 5:00–7:00 | Metadata eligibility | Put company and period before both retrieval channels. | “Filtering after top-k can permanently lose eligible evidence.” |
| 7:00–9:00 | Lexical versus dense | Contrast exact-number matching with conceptual similarity. | “Each channel covers a different failure mode.” |
| 9:00–11:00 | Reciprocal-rank fusion | Write `w / (k + rank)` and add contributions by passage ID. | “RRF combines rank positions, not raw score scales.” |
| 11:00–13:00 | Transparent reranking | Name the five normalized features and the numeric weight. | “The rerank score is inspectable and is not confidence.” |
| 13:00–14:00 | Versioned index | Tie provider, model, dimension, corpus hash, policy and IDs together. | “Any identity mismatch requires an index rebuild.” |
| 14:00–15:00 | Failure forecast | Ask learners to predict the exact-number and leakage outcomes. | “Dense-only and unfiltered retrieval will fail in controlled ways.” |

Do not borrow notebook time to finish the deck. At 15:00, move to the executable pipeline.

## Exact 30-minute notebook pacing

| Time | Notebook action | Expected checkpoint |
|---:|---|---|
| 0:00–2:00 | Run imports and provider selection. | `Embedding runtime: offline / financial-concepts-v1` offline, or the selected live provider/model. |
| 2:00–5:00 | Inspect Figure 1 and rebuild source data. | Exactly seven passages; manifest schema v1, evidence-set version and corpus-hash prefix print. |
| 5:00–7:00 | Inspect Figure 2. | Seven passage points and four starred query points are visible; axes say “teaching view only.” |
| 7:00–10:00 | Inspect Figure 3. | Four-by-seven matrix with raw cosine values and a “not confidence” colorbar. |
| 10:00–13:00 | Run all maintained questions and inspect Figure 4. | Keyword, dense and RRF ladders expose their own score types; green identifies expected evidence. |
| 13:00–16:00 | Run the exact-number failure, Figure 5. | Keyword finds `18.7%`; dense raw cosine values tie at zero; `Dense exact-term failure reproduced` prints. |
| 16:00–19:00 | Run pre-filtering, Figure 6. | The unfiltered result is Schneider; every filtered candidate is NVIDIA; `Cross-company leakage blocked` prints. |
| 19:00–22:00 | Inspect RRF contributions, Figure 7. | Each bar is the visible sum of keyword and dense reciprocal-rank terms. |
| 22:00–25:00 | Inspect rerank features, Figure 8. | Weighted features sum to the displayed rerank score; exact numeric evidence leads. |
| 25:00–27:00 | Run Figure 9 scorecard. | Reranked hybrid reaches 4/4 and exceeds dense recall; `Hybrid retrieval improves maintained recall` prints. |
| 27:00–28:30 | Run verification. | Six checks print `PASS`; final line is `PASS — hybrid retrieval laboratory verified`. |
| 28:30–30:00 | Run the weight challenge and debrief. | At least one `Ranking moved — …` line and the moved question IDs print. |

## Checkpoints and markers

### Source and index checkpoint

Expected:

- both source fixture hashes verify against `assets/course-data/manifest.json`;
- `structure_aware_chunks(..., max_chars=220)` followed by `contextualize_chunks` yields
  seven chunks;
- every `IndexedPassage` retains company, period, document type, section, text, chunk ID
  and official URL; and
- the local index manifest records provider, model, dimension, corpus hash, strategy and
  ordered passage IDs.

If the chunk count is not seven, stop. Do not replace the manifest-derived chunks with a
handwritten fixture.

### `Dense exact-term failure reproduced`

The query is only `18.7%`. The deterministic embedding intentionally excludes numeric
tokens, so the raw query vector is zero and every raw cosine similarity is `0.000`.
Identifier tie-breaking places an NVIDIA passage first, while the lexical channel finds a
Schneider passage containing the exact number. The marker confirms the controlled failure,
not a successful dense lookup.

### `Cross-company leakage blocked`

The unfiltered query `energy management organic growth` ranks Schneider Electric evidence
first. With the application scope set to NVIDIA FY2026, the same dense search returns only
eligible NVIDIA passages. The marker confirms that pre-filtering blocks cross-company
candidates before ranking.

### `Hybrid retrieval improves maintained recall`

For the versioned four-question set, the expected offline scorecard is:

| Stage | Expected rank-1 recovery |
|---|---:|
| Keyword | 4/4 |
| Dense | 3/4 |
| RRF fusion | 3/4 |
| Reranked hybrid | 4/4 |

This is a diagnostic teaching set, not evidence that hybrid retrieval universally
dominates lexical retrieval. It proves that the staged policy repairs the controlled dense
miss while retaining the other expected evidence.

### `PASS — hybrid retrieval laboratory verified`

This final marker appears only after all six checks pass: seven manifest-derived passages,
complete provenance, two persisted index files, the cross-company barrier and 4/4 reranked
maintained recall.

## Core explanations

### Cosine similarity

For normalized vectors `q` and `d`, cosine similarity is their dot product. Values closer
to one indicate greater directional alignment in the embedding space. The value is not a
probability, calibrated relevance, confidence, truth or grounding score. Embedding model,
normalization and corpus changes can all change the ranking.

The 2D SVD figure is a projection for intuition. The similarity heatmap is the faithful
view because it uses the full embedding dimension.

### Pre-filtering

Pre-filtering determines which passages are eligible before either retrieval channel runs.
This matters because post-filtering a global top-k cannot recover a company passage that
was never retrieved. The public boundary accepts exact company, period, document type and
section filters. An empty eligible set produces an explicit abstention rather than a
broader, unsafe search.

### Reciprocal-rank fusion

Keyword TF-IDF scores and embedding cosine similarities have different meanings and are
not calibrated to each other. RRF uses rank positions:

```text
RRF(passage) = Σ channel_weight / (k + rank_in_channel)
```

The lesson uses `k=60`. Passage IDs deduplicate the channels. Weights express a policy and
must be evaluated; they do not turn RRF into probability.

### Reranking

The offline reranker exposes five normalized features:

- lexical query coverage: 0.25;
- exact numeric coverage: 0.45;
- query overlap with the section name: 0.10;
- metadata eligibility: 0.10; and
- normalized fusion signal: 0.10.

Candidates arrive only after pre-filtering, so metadata eligibility is one in this pipeline.
The fixed weighted sum is a rerank score. It is bounded and inspectable, but it remains a
ranking policy rather than confidence.

## Controlled failure debrief

For the exact-number failure, do not fix the deterministic embedder by adding `18.7%` to a
hidden vocabulary. The lesson's point is that concept embeddings may omit literal values
and lexical retrieval supplies a complementary signal.

For the cross-company failure, do not lower a threshold or append “NVIDIA” repeatedly to
the query. Company scope is structured eligibility data. Enforce it in the retrieval
boundary before scoring.

## Challenge solution

The challenge changes only the keyword weight from `1.0` to `3.0`; dense remains `1.0`.
In the deterministic offline run, at least the NVIDIA exact-number ranking moves because
the keyword-first passage receives a larger RRF contribution. Additional lower-rank moves
may print for the Schneider questions where channel order differs.

The correct conclusion is:

> A moved ranking proves that the weight has an effect. It does not prove the weight is an
> improvement. Compare expected evidence across a maintained evaluation set, then inspect
> latency, false positives and downstream grounding.

## Likely student mistakes

| Mistake | Diagnostic response |
|---|---|
| Calling cosine similarity “confidence” | Ask what event the number is a probability of; no calibrated event exists. |
| Comparing TF-IDF and dense raw scores directly | Return to the three score labels in Figure 4 and use ranks for fusion. |
| Filtering after global top-k | Show that an omitted eligible passage cannot be restored by post-filtering. |
| Rebuilding a four-passage fixture | Inspect the manifest loop and require the seven-chunk assertion to pass. |
| Dropping document type or URL during adaptation | Inspect the `IndexedPassage` constructor and fail the provenance check. |
| Treating the 2D map as retrieval truth | Compare Figure 2 with the full-dimensional heatmap in Figure 3. |
| Calling the rerank value confidence | Name it “rerank score” and inspect its weighted feature sum. |
| Tuning RRF on one question | Require the complete versioned evidence scorecard before choosing a weight. |
| Reusing vectors after a corpus or model change | Read the index manifest mismatch; rebuild instead of editing metadata. |

## Provider modes

The source manifest, parsers, chunking, filters, fusion, reranking and verification remain
the same in every mode. Only the embedding implementation changes through the shared
gateway.

### Offline

```bash
.venv/bin/python scripts/execute_notebooks.py \
  notebooks/06_hybrid_retrieval.ipynb \
  --mode offline \
  --output-dir /private/tmp/finai-lesson06-executed
```

### Ollama

```bash
FINAI_EMBEDDING_MODEL=qwen3-embedding:0.6b \
  .venv/bin/python scripts/execute_notebooks.py \
  notebooks/06_hybrid_retrieval.ipynb \
  --mode live --provider ollama
```

Ensure Ollama is running and the configured embedding model is available. The executor
sets `FINAI_LIVE_MODE=1` and selects the provider through shared settings.

### OpenAI

```bash
OPENAI_API_KEY=... FINAI_EMBEDDING_MODEL=text-embedding-3-small \
  .venv/bin/python scripts/execute_notebooks.py \
  notebooks/06_hybrid_retrieval.ipynb \
  --mode live --provider openai
```

Keep credentials outside the notebook. Live embeddings can change rank ties and therefore
challenge output; the safety and provenance checks must still hold.

## Index-version debugging

The notebook persists to:

```text
Path(os.getenv("FINAI_INDEX_DIR", tempfile.gettempdir())) / "finai-lesson06-index"
```

Inspect `manifest.json` and compare these fields before reusing vectors:

1. `schema_version` — unsupported schemas require migration or rebuild;
2. `provider` and `model` — vectors from different embedding spaces are incompatible;
3. `dimension` — query and document dimensions must match;
4. `corpus_hash` — any passage text or provenance change invalidates the corpus identity;
5. `chunking_strategy` — `contextual-structure-v1-max220` is part of the identity; and
6. `passage_ids` — order and membership must match the requested corpus.

Do not edit the manifest to make a mismatch disappear. Remove or redirect the temporary
artifact only through an intentional maintenance action, then rerun the notebook to build
a coherent index. `FINAI_INDEX_DIR` is useful when learners need an isolated clean index.

## Transition to Lesson 07

Lesson 06 ends with four maintained questions, literal expected evidence, visible channel
rankings, fusion contributions, rerank features and an index version. Lesson 07 should turn
each run into a trace with at least:

- query and exact metadata filter;
- index identity;
- eligible candidate count;
- keyword and dense ranks with their correctly named score types;
- RRF contributions;
- rerank feature values and final evidence IDs; and
- answer groundedness measured separately from retrieval recall.

End with: “We can now explain which evidence won. Next we measure whether it should have
won, and whether the generated answer actually used it.”

## Sources

- [NVIDIA fiscal 2026 Form 10-K](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm)
- [Schneider Electric 2025 full-year results](https://www.se.com/ww/en/assets/564/document/528237/release-fy-results-2025.pdf)
- Course source metadata and hashes: `assets/course-data/manifest.json`
- Reusable retrieval interfaces: `src/finai_academy/hybrid_retrieval.py`,
  `src/finai_academy/retrieval_pipeline.py` and `src/finai_academy/reranking.py`
