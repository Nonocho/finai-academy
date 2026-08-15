# 01 — Local and hosted model gateway

**Duration:** 30 minutes
**Format:** 10 minutes concepts, 20 minutes guided notebook
**Capstone increment:** first provider-neutral model response

## Engineering question

How can one financial AI application use a local model or a hosted OpenAI model
without duplicating lesson and application code?

## Learning contract

The learner calls a chat model through one internal boundary, inspects the
message contract, measures latency, exposes the weakness of an underspecified
financial question, grounds a second answer in labelled NVIDIA filing evidence,
and switches providers through configuration only.

## Failure demonstrated

A technically successful model call is not necessarily a useful application
response. The failure lab sends an ambiguous finance question and makes the
missing company, period, source, evidence standard, and output contract visible.

## Real-company evidence

The guided answer uses four paraphrased facts from NVIDIA's fiscal 2026 Form
10-K: total revenue, Data Center revenue, Gaming revenue, and the H20-related
charge affecting gross margin. Facts are labelled `F1` to `F4` so the learner
can distinguish a sourced claim from model prose before the RAG lessons.

Source: [NVIDIA fiscal 2026 Form 10-K](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm)

## Verification

The notebook must complete in offline test mode and in separate live Ollama and
OpenAI runs. The resulting `ModelRun` retains provider, model, response text, and
measured latency. The deterministic offline answer must score 4/4 on the visible
criteria: company and period, evidence-bounded metrics with no invented numbers,
evidence citations, and an explicit limitation. Live answers target 4/4 but
produce `REVIEW` rather than crashing when model wording varies.
