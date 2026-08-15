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
financial question, and switches providers through configuration only.

## Failure demonstrated

A technically successful model call is not necessarily a useful application
response. The failure lab sends an ambiguous finance question and makes the
missing company, period, source, evidence standard, and output contract visible.

## Verification

The notebook must complete in offline test mode and in separate live Ollama and
OpenAI runs. The resulting `ModelRun` retains provider, model, response text, and
measured latency.
