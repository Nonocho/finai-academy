# Session 01 Production Readiness Design

## Objective

Raise Session 01 from pilot-ready to production-ready by aligning the technical
micro-deck, notebook, real financial evidence, assessment contract, provider
portability and First Finance branding.

## Scope

- Create the missing `decks/01-model-gateway.pptx` as a six-slide, ten-minute
  technical micro-deck.
- Preserve the factual editorial style of the introduction deck.
- Change the introduction title marker and shared brand documentation from
  `FinAI Academy` to `First Finance` where audience-facing identity is defined.
- Replace the anonymous notebook excerpt with a compact NVIDIA fiscal 2026
  evidence card grounded in the official Form 10-K filed with the SEC.
- Add a lightweight deterministic grounding rubric without pre-empting the
  structured-output and evaluation lessons.
- Move the offline recorded model out of the learner notebook into a small
  lesson-support module.
- Validate offline execution, live Ollama execution, the OpenAI configuration
  path, repository tests and slide rendering.

## Teaching design

The deck uses a cumulative learning progression:

1. a model gateway keeps application code stable;
2. messages form an inspectable contract;
3. a successful model call can still fail as analysis;
4. a real NVIDIA evidence card makes the request answerable;
5. a four-part rubric distinguishes execution from grounding;
6. the notebook challenge proves provider portability.

The notebook follows the same progression. It begins with an intentionally
underspecified question, then answers a bounded equity-research request from
four labelled facts. The learner can see why the second answer is more useful
without being told that it is investment-grade.

## Evidence contract

The NVIDIA evidence card contains only paraphrased facts from the official
fiscal 2026 Form 10-K:

- fiscal 2026 revenue of $215.9 billion, up 65% year on year;
- Data Center revenue of $193.7 billion, up 68%;
- Gaming revenue of $16.0 billion, up 41%;
- gross margin decreased and was affected by a $4.5 billion H20 charge.

Every fact receives a stable identifier (`F1` to `F4`) and the SEC URL is
visible in the notebook and included in the relevant slide speaker notes.

## Assessment contract

The grounded answer earns one point for each observable behaviour:

- identifies NVIDIA and fiscal 2026;
- uses at least two supplied metrics;
- cites at least two evidence identifiers;
- states one limitation or conclusion that the evidence cannot support.

A score of 4/4 is the guided target. The rubric remains intentionally simple;
semantic graders and regression datasets belong to later lessons.

## Visual contract

- 16:9, off-white background, deep navy text, royal blue and cyan emphasis,
  orange only for failure or risk.
- One primary claim per slide, six slides total.
- Minimum 50 pt deck title, 35 pt slide title, 16 pt body text.
- Flat editorial layouts rather than dashboard-like card grids.
- Footer: `First Finance - Arnaud Demes` on every slide except the title.
- External claims and URLs appear in `[Sources]` blocks in speaker notes.

## Acceptance criteria

- The clean notebook contains no stored output or secret and passes the course
  notebook contract.
- Offline and live Ollama runs finish with the gateway PASS and a grounding
  score of 4/4.
- The OpenAI adapter can be constructed from environment configuration without
  placing a key in the notebook; a live acceptance run is performed only when
  a valid user key is already present.
- All repository tests pass.
- Both presentation files render without overflow, clipping or inconsistent
  audience-facing branding.
- The six-slide micro-deck can be presented in ten minutes, leaving twenty
  minutes for the guided notebook.

