# Session 02 Structured Outputs Design

## Purpose

Lesson 2 turns the provider-neutral chat boundary from Lesson 1 into a typed
financial application boundary. In thirty minutes, learners move from a
plausible text answer to a validated `AnalystBrief` that the capstone can store,
render, test, and reuse.

## Delivery contract

- Time: 10:00-10:30 on Day 1.
- Format: 10 minutes of slides and 20 minutes of guided notebook work.
- Finance case: NVIDIA fiscal 2026, using the same evidence card as Lesson 1.
- Providers: the same notebook runs offline, with Ollama, or with OpenAI.
- Capstone increment: a typed analyst brief containing findings, evidence
  categories, source excerpts, caveats, and open questions.
- Brand footer: `First Finance - Arnaud Demes`.

## Teaching progression

1. Start with an underspecified analyst request and identify the missing
   company, period, evidence, constraints, and output contract.
2. Separate system instructions, trusted application inputs, source data, and
   output requirements.
3. Validate a syntactically correct JSON object that is financially invalid.
4. Introduce Pydantic as the application contract rather than a formatting
   convenience.
5. Generate the same `AnalystBrief` through the provider-neutral structured
   model interface.
6. Apply deterministic verification to the model result.

## Domain contract

`AnalystFinding` keeps these fields:

- `statement`: non-empty material statement;
- `category`: `key_result`, `catalyst`, or `risk`;
- `evidence_type`: reported fact, calculation, management claim, external fact,
  or interpretation;
- `source_excerpt`: required for reported facts and management claims;
- `rationale`: required for interpretations.

`AnalystBrief` keeps trusted company and reporting-period inputs outside the
model's authority. Extra fields are rejected. The application validates the
shape and the finance-specific rules after generation.

## Failure lab

The first candidate is valid JSON but contains a reported fact without a source
excerpt. Pydantic rejects it. A second example shows that an interpretation
without rationale is also invalid. The learner reads the validation errors and
then corrects the object.

## Offline and live execution

The offline fixture implements the same `StructuredModel` protocol as live
providers and returns a deterministic NVIDIA brief. Live mode calls
`create_structured_model(Settings.from_environment())`, which routes to Ollama
or OpenAI and binds the same Pydantic response model.

No API key is embedded in the notebook. OpenAI live mode uses `OPENAI_API_KEY`.
Ollama remains the local-first default.

## Deck narrative

Six slides form one learning progression:

1. structured outputs as the application boundary;
2. prompts as explicit interfaces;
3. why prompt-only JSON remains fragile;
4. the `AnalystBrief` contract;
5. syntax, schema, and finance validation as separate layers;
6. the notebook mission: fail, validate, bind, verify.

The visual system follows the existing Session 1 deck: off-white canvas, deep
navy typography, royal blue and cyan teaching accents, orange for failure and
risk, Calibri, and the First Finance footer.

## Acceptance criteria

- The notebook satisfies the repository teaching contract and contains no
  stored outputs, absolute user paths, or secrets.
- Offline execution prints a validation failure, a validated brief, and a clear
  PASS marker.
- Domain tests prove that source excerpts and interpretation rationales are
  enforced.
- Existing application tests remain green.
- The deck contains six legible slides with notes and source blocks.
- Every slide renders without clipping or unintended overlap.

## Sources

- OpenAI, Structured model outputs:
  https://developers.openai.com/api/docs/guides/structured-outputs
- NVIDIA, fiscal 2026 results evidence card already established in Lesson 1.
