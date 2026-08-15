# 02 — Prompt Engineering and Structured Outputs

## Session contract

- **Time:** 10:00-10:30, Day 1.
- **Format:** 10 minutes of slides, 20 minutes of guided notebook work.
- **Case:** NVIDIA fiscal 2026 using the Lesson 1 SEC evidence card.
- **Capstone increment:** a Pydantic-validated `AnalystBrief` generated through
  the same Ollama/OpenAI boundary.

## Learning promise

Learners can turn an ambiguous financial request into a versioned prompt and a
typed response contract, then distinguish successful generation from an output
the application should accept.

## Instructor preparation

Before class:

1. run `uv run python scripts/setup_check.py --provider ollama` or the OpenAI
   equivalent;
2. execute the notebook once in offline mode;
3. confirm the selected live provider can generate a structured brief; and
4. keep the invalid candidate unchanged for the failure lab.

Do not start by explaining every Pydantic feature. The teaching problem is the
boundary between fluent text and accepted financial data.

## 10:00-10:10 — Concept deck

### Slide 1 — Structured output is the application boundary

Connect directly to Lesson 1: the gateway produced text, but an application
needs a stable object it can render, store, test, and evaluate.

### Slide 2 — A prompt is an explicit interface

Separate system instructions, trusted inputs, source data, and output contract.
Ask learners which layer should own the company and reporting period. The
answer is application code, not the model.

### Slide 3 — Prompt-only JSON remains fragile

Show four failure classes: invalid JSON, missing fields, unexpected categories,
and valid structure with unsupported claims. Emphasise that a parser only solves
the first failure.

### Slide 4 — `AnalystBrief` expresses the financial product

Introduce the finding category, evidence type, source excerpt, rationale,
caveats, and open questions. Keep the discussion on domain meaning rather than
Pydantic syntax.

### Slide 5 — Validation has three distinct layers

Use syntax, schema, and finance semantics. Structured generation helps the
schema layer. Pydantic validators and deterministic application checks enforce
the finance layer.

### Slide 6 — Notebook mission

State the observable sequence: parse valid JSON, watch finance validation fail,
bind the schema, generate a typed brief, and verify the object.

## 10:10-10:30 — Guided notebook

### Minutes 0-3 — Diagnose the vague request

Ask learners to identify what is missing from `Give me the main NVIDIA results
and risks`. Capture company, period, authorised source, evidence rules,
uncertainty, and output shape.

### Minutes 3-7 — Run the failure lab

The candidate parses as JSON and then fails Pydantic validation because a
reported fact has no `source_excerpt`.

Expected output includes:

```text
PASS — JSON syntax is valid
Validation caught the unsupported candidate
```

Pause on the result. The failure is not a model failure; it is the application
correctly refusing an unsupported product object.

### Minutes 7-11 — Read the contract

Inspect the JSON Schema and map each field back to an analyst requirement.
Explain why enums and `extra="forbid"` reduce ambiguity. Then show the two
finance validators:

- reported facts and management claims require source excerpts;
- interpretations require rationale.

### Minutes 11-16 — Generate the typed brief

Run the same `AnalystBriefService` with the configured provider. Offline mode
uses `RecordedStructuredModel`; live mode uses Ollama or OpenAI.

Point out that the service overwrites company and reporting period with trusted
application inputs after generation.

### Minutes 16-20 — Verify and debrief

The visible checks confirm the type, trusted inputs, findings, evidence rules,
and caveat. The target final line is:

```text
PASS — structured financial brief verified
```

If a live model fails, inspect the failed criterion. Do not weaken the contract
just to make the notebook green.

## Failure handling discussion

Use these distinctions:

| Failure | Appropriate application response |
|---|---|
| Transient provider error | Bounded retry with observability |
| Refusal | Preserve and present the refusal |
| Schema error | Log prompt version and validation details |
| Missing evidence | Return a bounded failure or request human review |

A retry can repair formatting. It cannot create evidence that is absent from
the source.

## Challenge answer guidance

The challenge adds `confidence_reason: str`, not a numeric confidence score.
An acceptable implementation requires a failing test first, updates the
Pydantic model and offline fixture, and preserves the dual-provider boundary.

The discussion point is whether the reason adds decision value. Avoid giving a
precise confidence percentage unless the course later defines a calibrated
measurement procedure.

## Provider commands

Offline regression run:

```bash
uv run python scripts/execute_notebooks.py \
  notebooks/02_prompts_and_structured_outputs.ipynb \
  --mode offline
```

Ollama live run:

```bash
FINAI_MODEL_PROVIDER=ollama FINAI_CHAT_MODEL=qwen3:8b \
uv run --extra ai jupyter lab
```

OpenAI live run:

```bash
FINAI_MODEL_PROVIDER=openai FINAI_CHAT_MODEL=gpt-5-mini \
uv run --extra ai jupyter lab
```

`OPENAI_API_KEY` must already exist in the shell environment.

## Checkpoint questions

1. What belongs in the prompt rather than the schema?
2. Why can valid JSON still be financially wrong?
3. Which inputs should the application preserve as trusted?
4. When is a retry appropriate, and when is it unsafe?
5. Which semantic constraints should remain deterministic Python code?

## Transition to Lesson 3

The output contract is now stable, but the evidence is still a four-line card.
Lesson 3 injects a complete financial document and studies the point at which
full-context prompting becomes sufficient, slow, or diluted.

## Sources

- [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
- [NVIDIA fiscal 2026 Form 10-K](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm)
