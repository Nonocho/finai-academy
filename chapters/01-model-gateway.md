# 01 — Local and hosted model gateway

**Duration:** 30 minutes
**Format:** 10 minutes concepts, 20 minutes guided notebook
**Capstone increment:** first measured, provider-neutral model response

## Engineering question

How can one financial AI application use a local model or a hosted OpenAI model
without duplicating lesson and application code?

## Learning contract

The learner calls a chat model through one internal boundary, inspects the
message contract, measures latency and available tokens, observes streaming,
exposes the weakness of an underspecified financial question, grounds a second
answer in labelled NVIDIA filing evidence, and switches providers through
configuration only.

MLflow is not introduced here. Lessons 01–06 first build ordinary typed records
and observable stages. Lesson 07 will log those same records as experiments and
traces after the RAG pipeline exists.

## Teaching sequence

| Time | Instructor action | Learner output |
|---|---|---|
| 0–3 min | Confirm Ollama/OpenAI configuration and safe diagnostics | Provider and model, no secret |
| 3–7 min | Explain system/human messages and execute the first call | Non-empty ambiguous answer |
| 7–10 min | Read `ModelRun` latency and normalized token usage | Tokens or an explicit unavailable state |
| 10–13 min | Stream the same request | Partial delivery plus complete collected text |
| 13–17 min | Diagnose the ambiguous finance request | Five missing contract elements |
| 17–19 min | Run the NVIDIA evidence-card request | Evidence-bound answer and 4/4 rubric |
| 19–20 min | Run verification and knowledge check | One final PASS marker |

## Concepts to draw before coding

```text
Question → Settings → Provider adapter → Model response
                                      ↘ latency / tokens

structured output → context → RAG → evaluation → workflows → agents → MCP
```

Make two distinctions explicit:

- provider choice is configuration behind a stable application boundary;
- streaming changes response delivery, not evidence quality or factuality.

## Failure demonstrated

A technically successful model call is not necessarily a useful application
response. The failure lab sends an ambiguous finance question and makes the
missing company, period, source, evidence standard, and output contract visible.

Do not repair the vague prompt before students see the answer. The failure is
the evidence that a successful API call is not yet an analyst application.

## Real-company evidence

The guided answer uses a manually curated evidence card with four paraphrased
facts from NVIDIA's fiscal 2026 Form 10-K: total revenue, Data Center revenue,
Gaming revenue, and the H20-related charge affecting gross margin. The notebook
does not download or parse the filing. Facts are labelled `F1` to `F4` so the
learner can distinguish a sourced claim from model prose before document
loading begins in Lesson 03 and retrieval is automated in later RAG lessons.

Source: [NVIDIA fiscal 2026 Form 10-K](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm)

## Expected outputs

- Offline: `provider=offline`, a non-negative latency, and
  `Token usage: unavailable for this provider response`.
- Ollama/OpenAI: provider and model match configuration; the hosted course
  example uses `gpt-5.6-luna`; token counts are shown
  when the adapter returns all three normalized counts.
- Streaming: the output starts with `Streaming demo:` and the collected text is
  non-empty.
- Grounding: the deterministic fixture reaches `Grounding score: 4/4`.
- Final: exactly one `PASS — provider-neutral model gateway verified`.

Never convert missing token metadata to zero. Zero means measured zero;
`None` means the provider response did not expose a complete count.

## Knowledge-check answers

1. Streaming does not improve factual quality; it only changes delivery.
2. Token usage can be unavailable because provider response metadata is not
   uniform. The gateway keeps an honest missing state.
3. Ollama-to-OpenAI is a configuration change. Moving from free text to a
   validated schema is a new application contract and belongs in Lesson 02.

## Common failures

| Symptom | Diagnosis | Instructor action |
|---|---|---|
| Ollama connection refused | Local service is stopped | Start Ollama and rerun `scripts/setup_check.py` |
| Model not found | Configured local model is missing | Pull the exact class model and restart the kernel |
| OpenAI authentication error | Key absent in Jupyter shell | Export the key outside the notebook and restart |
| Token usage unavailable | Adapter returned incomplete metadata | Keep `None`; continue the lesson |
| Slow first local response | Model load/warm-up | Compare first and second call, do not change code |
| Fluent but unsupported answer | Request lacks evidence contract | Use the failure checklist before changing models |

## Verification

The notebook must complete in offline test mode and in separate live Ollama and
OpenAI runs. The resulting `ModelRun` retains provider, model, response text,
measured latency, and normalized token usage when it is supplied. The
deterministic offline answer must score 4/4 on the visible
criteria: company and period, evidence-bounded metrics with no invented numbers,
evidence citations, and an explicit limitation. Live answers target 4/4 but
produce `REVIEW` rather than crashing when model wording varies.

The source notebook remains output-free. Execution evidence is written to a
temporary output directory and inspected separately.

## Transition to Lesson 02

The provider boundary now returns observable text, but free text is not a
reliable interface. Lesson 02 progressively improves the prompt and binds the
final answer to a Pydantic financial schema.
