# Lesson 06 — Embeddings and Hybrid Retrieval

**Schedule:** Day 1, 15:15–16:00

**Format:** 15 minutes of concepts and diagrams, 30 minutes of guided notebook work

**Capstone increment:** filtered BM25+dense retrieval, reciprocal-rank fusion and
transparent reranking over a versioned local embedding index

## Teaching outcome

Students should leave with one operational principle:

> Retrieval quality is a staged, testable system property. A nearest neighbor is not yet
> eligible, complete or safe evidence.

The notebook rebuilds the seven contextual structure-aware chunks from Lesson 05's
versioned source manifest. It then uses BM25 and embeddings to retrieve, pre-filter, fuse
and rerank those same chunks. Nothing is silently replaced with the smaller unit-test
corpus. Timing, tracing and MLflow belong to Lesson 07.

## Connection to Lessons 05 and 07

Lesson 05 established trustworthy evidence units. Lesson 06 holds those units constant and
changes their representation and ranking:

```text
Lesson 05                         Lesson 06                              Lesson 07
manifest → blocks → chunks  →  BM25 + dense → filter → fuse → rerank  →  evaluate → trace
```

Lesson 07 will treat each maintained question and expected-evidence token as an evaluation
case. Its traces should expose the stages created here rather than only the final answer.

## Exact 15-minute deck pacing

| Time | Concept | Instructor move | Expected learner statement |
|---:|---|---|---|
| 0:00–1:30 | Lesson boundary | Point from trusted Lesson 05 chunks to the new retrieval stages. | “The corpus stays fixed; representation and ranking change.” |
| 1:30–3:30 | Why hybrid retrieval | Use the sourced industry diagram and benchmark to show the complementary channels. | “Exact evidence and semantic evidence fail differently.” |
| 3:30–5:00 | Embedding geometry | Define cosine as relative alignment, not confidence. | “A high cosine is not a probability of correctness.” |
| 5:00–7:00 | BM25 versus dense | Contrast exact-number matching with conceptual similarity. | “BM25 protects literals; dense protects semantic recall.” |
| 7:00–9:00 | Metadata eligibility | Put company and period before both retrieval channels. | “Filtering after top-k can permanently lose eligible evidence.” |
| 9:00–11:00 | Reciprocal-rank fusion | Write `w / (k + rank)` and add contributions by passage ID. | “RRF combines rank positions, not raw score scales.” |
| 11:00–13:00 | Transparent reranking | Name the five normalized features and the numeric weight. | “The rerank score is inspectable and is not confidence.” |
| 13:00–15:00 | Decision rules | Map each stage to the failure it repairs and forecast the notebook failures. | “Hybrid retrieval is a staged policy, not one magic score.” |

Do not borrow notebook time to finish the deck. At 15:00, move to the executable pipeline.

## Exact 30-minute notebook pacing

| Time | Notebook action | Expected checkpoint |
|---:|---|---|
| 0:00–4:00 | Select the provider and rebuild the seven real Lesson 05 chunks. | Source hashes pass; the passage table retains company, period, section and stable ID. |
| 4:00–8:00 | Inspect Figure 1: BM25 versus dense. | Green identifies the literal-bearing passage; raw score types stay distinct. |
| 8:00–12:00 | Run Figure 2: exact-number failure. | BM25 finds `18.7%`; the controlled dense query ties at zero. |
| 12:00–16:00 | Run Figure 3: filter safety. | The unfiltered winner is Schneider; every filtered candidate is NVIDIA. |
| 16:00–20:00 | Inspect Figure 4: RRF contributions. | Each bar is the visible sum of BM25 and dense reciprocal-rank terms. |
| 20:00–25:00 | Inspect Figures 5–6: rerank features and scorecard. | Weighted features stay visible; offline reranked hybrid reaches 4/4. |
| 25:00–28:00 | Change the BM25 RRF weight. | A moved ranking is an observation, not proof of improvement. |
| 28:00–30:00 | Run verification and debrief. | Every filter, ID, score, provenance field and index artifact passes. |

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

### `BM25 exact-term recovery reproduced`

The query is only `18.7%`. BM25 tokenization preserves the numeric literal and ranks the
Schneider Electric passage containing that figure first. The result demonstrates why a
lexical channel remains valuable for financial values, identifiers and names.

### `Dense exact-term failure reproduced`

This figure always uses a separate illustrative `DeterministicTeachingEmbeddings` index,
including during Ollama and OpenAI runs. The query is only `18.7%`. The deterministic
embedding intentionally excludes numeric
tokens, so the raw query vector is zero and every raw cosine similarity is `0.000`.
Identifier tie-breaking places an NVIDIA passage first, while the lexical channel finds a
Schneider passage containing the exact number. The marker confirms the controlled failure,
not a successful dense lookup.

### `Cross-company leakage blocked`

This is also a controlled deterministic illustration in every provider mode. The
unfiltered query `energy management organic growth` ranks Schneider Electric evidence
first. With the application scope set to NVIDIA FY2026, the same controlled dense search
returns only eligible NVIDIA passages. The marker confirms the mechanism. Verification
separately asserts that the active live provider's filtered result contains only NVIDIA
passages without requiring a particular rank order or similarity value.

### `Hybrid retrieval improves maintained recall`

This marker is an offline-only assertion. For the versioned four-question set, the expected
deterministic scorecard is:

| Stage | Expected rank-1 recovery |
|---|---:|
| BM25 | 4/4 |
| Dense | 3/4 |
| RRF fusion | 3/4 |
| Reranked hybrid | 4/4 |

This is a diagnostic teaching set, not evidence that hybrid retrieval universally
dominates lexical retrieval. It proves that the staged policy repairs the controlled dense
miss while retaining the other expected evidence.

### `PASS — hybrid retrieval laboratory verified`

This final marker is provider-neutral. It appears only after every maintained run proves
that all channel, fused and final hits satisfy its filters; fused IDs are unique; fused
ordering is exactly descending RRF score then passage ID; every score is finite;
provenance is complete; provider vectors are finite; and both index artifacts exist. Offline mode additionally
requires every maintained evidence token to appear within the configured `final_k=2` result.
Live mode does not require a particular ranking, tie, recall improvement or RRF weight
response.

## Core explanations

### Cosine similarity

For normalized vectors `q` and `d`, cosine similarity is their dot product. Values closer
to one indicate greater directional alignment in the embedding space. The value is not a
probability, calibrated relevance, confidence, truth or grounding score. Embedding model,
normalization and corpus changes can all change the ranking.

The deck supplies the geometric intuition. The notebook stays focused on observable
retrieval decisions rather than teaching projections.

### BM25

BM25 is the lexical channel. It rewards query terms that are rare across the corpus,
saturates repeated terms and normalizes for passage length. In this lesson it protects exact
financial figures such as `18.7%` and `$193.7 billion` that a semantic representation can
downweight or omit. Its raw score is a ranking signal within one index, not confidence.

### Pre-filtering

Pre-filtering determines which passages are eligible before either retrieval channel runs.
This matters because post-filtering a global top-k cannot recover a company passage that
was never retrieved. The public boundary accepts exact company, period, document type and
section filters. An empty eligible set produces an explicit abstention rather than a
broader, unsafe search.

### Reciprocal-rank fusion

BM25 scores and embedding cosine similarities have different meanings and are
not calibrated to each other. RRF uses rank positions:

```text
RRF(passage) = Σ channel_weight / (k + rank_in_channel)
```

The lesson uses `k=60`. Passage IDs deduplicate the channels. Weights express a policy and
must be evaluated; they do not turn RRF into probability.

### Reranking

The offline reranker exposes five normalized features:

- lexical query coverage, including ticker tokens such as `NVDA`: 0.25;
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
In the deterministic offline run, at least the NVIDIA exact-number ranking must move because
the keyword-first passage receives a larger RRF contribution. Additional lower-rank moves
may print for the Schneider questions where channel order differs.

Ollama and OpenAI rankings are provider observations. They may move or remain unchanged;
both are valid. The live challenge prints what happened and never fails solely because the
selected provider produced stable ranks.

The correct conclusion is:

> A moved ranking proves that the weight has an effect. It does not prove the weight is an
> improvement. Compare expected evidence across a maintained evaluation set, then inspect
> latency, false positives and downstream grounding.

## Likely student mistakes

| Mistake | Diagnostic response |
|---|---|
| Calling cosine similarity “confidence” | Ask what event the number is a probability of; no calibrated event exists. |
| Comparing BM25 and dense raw scores directly | Return to the channel labels in Figure 1 and use ranks for fusion. |
| Filtering after global top-k | Show that an omitted eligible passage cannot be restored by post-filtering. |
| Rebuilding a four-passage fixture | Inspect the manifest loop and require the seven-chunk assertion to pass. |
| Dropping document type or URL during adaptation | Inspect the `IndexedPassage` constructor and fail the provenance check. |
| Treating BM25 or cosine as confidence | Ask what calibrated event the number represents; neither score is a probability. |
| Calling the rerank value confidence | Name it “rerank score” and inspect its weighted feature sum. |
| Tuning RRF on one question | Require the complete versioned evidence scorecard before choosing a weight. |
| Reusing vectors after a corpus or model change | Read the index manifest mismatch; rebuild instead of editing metadata. |

## Provider modes

The source manifest, parsers, chunking, BM25, filters, fusion and reranking remain the same in
every mode. The active dense index uses the shared embedding gateway. Figures 2 and 3 deliberately
use a separate deterministic teaching index so their controlled failures remain reproducible
without imposing offline rank assumptions on the live provider.

The executor sets both `FINAI_MODEL_PROVIDER` and `FINAI_EMBEDDING_PROVIDER` from the
explicit live `--provider` argument, replacing stale provider values. Offline execution
selects `DeterministicTeachingEmbeddings` before constructing or validating `Settings`, so
unrelated provider environment values cannot break the deterministic path.

### Offline

```bash
.venv/bin/python scripts/execute_notebooks.py \
  notebooks/06_hybrid_retrieval.ipynb \
  --mode offline \
  --output-dir /private/tmp/finai-lesson06-executed
```

Expected: all five execution-contract markers print, the scorecard is 4/4 after reranking
and final verification passes. The weight challenge reports the observed moved rankings.

### Ollama

```bash
FINAI_EMBEDDING_MODEL=qwen3-embedding:0.6b \
  .venv/bin/python scripts/execute_notebooks.py \
  notebooks/06_hybrid_retrieval.ipynb \
  --mode live --provider ollama \
  --output-dir /private/tmp/finai-lesson06-ollama
```

Ensure Ollama is running and the configured embedding model is available. The executor
sets `FINAI_LIVE_MODE=1` and selects the provider through shared settings.

Expected: Figures 2 and 3 still print their clearly labelled controlled markers. The live
scorecard prints observed recall without asserting 4/4 or improvement; provider vectors,
dimensions, filtered companies, finite scores and provenance are verified.
The challenge may report moved rankings or no movement. Final verification must pass.

### OpenAI

```bash
OPENAI_API_KEY=... FINAI_EMBEDDING_MODEL=text-embedding-3-small \
  .venv/bin/python scripts/execute_notebooks.py \
  notebooks/06_hybrid_retrieval.ipynb \
  --mode live --provider openai \
  --output-dir /private/tmp/finai-lesson06-openai
```

Keep credentials outside the notebook. Live embeddings can change rank ties and therefore
scorecard and challenge output. The same provider-invariant structural checks described for
Ollama must pass; no fixed live ranking, cosine sign, recall value or RRF movement is
promised.

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
- BM25 and dense ranks with their correctly named score types;
- RRF contributions;
- rerank feature values and final evidence IDs; and
- answer groundedness measured separately from retrieval recall.

Lesson 07 attaches trace spans to these same decisions, then adds context assembly and
generation as later stages rather than hiding them inside retrieval.

End with: “We can now explain which evidence won. Next we measure whether it should have
won, and whether the generated answer actually used it.”

## Sources

- [NVIDIA fiscal 2026 Form 10-K](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm)
- [Schneider Electric 2025 full-year results](https://www.se.com/ww/en/assets/564/document/528237/release-fy-results-2025.pdf)
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)
- [Azure AI Search — Hybrid search overview](https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview)
- [Elastic — Hybrid search](https://www.elastic.co/what-is/hybrid-search)
- [OpenAI — text-embedding-3-small](https://developers.openai.com/api/docs/models/text-embedding-3-small)
- Course source metadata and hashes: `assets/course-data/manifest.json`
- Reusable retrieval interfaces: `src/finai_academy/hybrid_retrieval.py`,
  `src/finai_academy/retrieval_pipeline.py` and `src/finai_academy/reranking.py`
