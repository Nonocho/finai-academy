# Day 1 student guide

## Objective

Build the evidence layer of the Financial Analyst Copilot. By 17:00, the application
must retrieve the right financial evidence, cite it, and expose its failures through
evaluation traces.

Before class, complete [Getting started](getting-started.md) and require:

```text
READY Course readiness — Environment is ready for Day 1.
```

## Schedule and build sequence

| Time | Lesson and assets | What you build |
|---|---|---|
| 09:00-09:30 | Introduction — [slides](../decks/00-course-introduction.pptx) | Product architecture and evidence contract |
| 09:30-10:00 | 01 — Model gateway — [slides](../decks/01-model-gateway.pptx) · [notebook](../notebooks/01_model_gateway.ipynb) | One interface for Ollama and OpenAI |
| 10:00-10:30 | 02 — Structured outputs — [slides](../decks/02-prompts-and-structured-outputs.pptx) · [notebook](../notebooks/02_prompts_and_structured_outputs.ipynb) | A validated analyst brief |
| 10:30-10:45 | Break | |
| 10:45-11:30 | 03 — Context engineering and CAG — [slides](../decks/03-cag-financial-document.pptx) · [notebook](../notebooks/03_cag_financial_document.ipynb) | A complete-document grounded answer |
| 11:30-12:00 | 04 — RAG from first principles — [slides](../decks/04-rag-from-scratch.pptx) · [notebook](../notebooks/04_rag_from_scratch.ipynb) | A deliberately naive retrieval-backed answer |
| 12:00-13:30 | Lunch | |
| 13:30-15:00 | 05 — Documents and chunking — [slides](../decks/05-document-and-chunking-lab.pptx) · [notebook](../notebooks/05_document_and_chunking_lab.ipynb) | Structure-aware, semantic, and LLM-aware chunks |
| 15:00-15:15 | Break | |
| 15:15-16:00 | 06 — Hybrid retrieval — [slides](../decks/06-hybrid-retrieval.pptx) · [notebook](../notebooks/06_hybrid_retrieval.ipynb) | Filtered keyword+dense retrieval and reranking |
| 16:00-16:45 | 07 — Evaluation and tracing — [slides](../decks/07-rag-evaluation.pptx) · [notebook](../notebooks/07_rag_evaluation.ipynb) | Golden cases, retrieval metrics, MLflow traces, and a Ragas bridge |
| 16:45-17:00 | Integration checkpoint | One measured Day 1 pipeline |

## How to run each lesson

1. Use the slides for the engineering problem and system diagram.
2. Open the matching notebook and run cells in order.
3. Complete the failure lab before reading the improvement.
4. Require the notebook verification marker.
5. Record the capstone increment before moving to the next lesson.

Run Jupyter from the repository root:

```bash
uv run jupyter lab
```

## End-of-day checkpoint

You should be able to explain and demonstrate:

- why provider access belongs behind one boundary;
- why structured outputs need validation;
- when a full document fits the context window;
- why naive RAG fails before parsing and chunking improve it;
- how hierarchical, semantic, and LLM-aware chunking differ;
- why hybrid retrieval combines lexical precision and semantic recall;
- how a golden set, retrieval metrics, MLflow, and Ragas expose regressions.

```text
Day 1 complete = parsed evidence + configurable chunks + hybrid retrieval + cited answer + evaluation trace.
```

If a cell fails, use [Troubleshooting](troubleshooting.md) before changing notebook
code.
