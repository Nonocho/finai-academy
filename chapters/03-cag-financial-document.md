# Lesson 03 — Context Engineering and Cache-Augmented Generation

**Schedule:** Day 1, 10:45–11:30

**Format:** 15 minutes of concepts and diagrams, 30 minutes of guided notebook work

**Capstone increment:** complete-document grounded answer with an explicit CAG/RAG route

## Teaching outcome

Students should leave with one operational distinction:

- prompt engineering defines the behaviour and output contract;
- context engineering defines the information made available to the model.

They build the simplest document-grounded path first: verify NVIDIA's complete
FY2026 Form 10-K, derive one bounded official context pack, keep that pack stable
across questions, and choose the route before generation.

They also leave with a vocabulary boundary that prevents common architecture errors:

| Concept | Engineering responsibility |
|---|---|
| Context | information supplied to the current model call |
| Cache | reuse of a stable prefix for efficiency |
| Memory | state retained across interactions |
| Grounding | evidence support for generated claims |
| RAG | selection of evidence units before generation |

These mechanisms can coexist. None is a synonym for another.

## Why this lesson comes before RAG

Retrieval is additional infrastructure. It should solve a demonstrated constraint,
not appear by default. A complete-document path is often sufficient for a short,
bounded source and gives students a clean baseline against which later retrieval can
be evaluated.

The lesson ends with a real constraint rather than synthetic filler. The bounded
official context pack fits the teaching window and routes to CAG; the complete
official filing does not and routes to RAG. The application makes both choices before
calling the model rather than truncating silently.

That choice is returned as a typed `ContextDecision`: route, component token
estimates, available input capacity, and a human-readable reason. The decision is
observable application state rather than a hidden boolean.

## Concept sequence

### 1. Context is a finite allocation

The input window is shared by system instructions, the complete document and the
question. The application must also reserve output capacity. A document does not fit
merely because its own token estimate is below the advertised model window.

The baseline estimator uses four characters per token. It is deterministic and clearly
labelled as approximate. A production implementation should use the tokenizer for the
selected model.

### 2. CAG keeps a long prefix stable

The prompt order is:

```text
stable instructions → stable source document → changing question
```

This order supports repeated questions over the same source. Some providers can reuse
an exact prompt prefix to reduce input-processing latency or cost. The course treats
that cache as an optional provider optimization. It never treats caching as evidence
grounding, memory, or a guarantee of faster responses.

### 3. Fit does not imply relevance

Even when a long document is technically accepted, useful information may be diluted
or placed where a model uses it less reliably. The notebook separates two claims:

- the budget gate is deterministic and tested;
- attention degradation is empirical and must be measured on representative questions.

The deck uses the published “Lost in the Middle” result to separate prompt fit from
evidence use. The notebook does not invent a model-accuracy score; it treats
long-context usefulness as an empirical evaluation question.

### 4. CAG and RAG solve different document shapes

Use direct context or CAG when the source is bounded, stable, within budget and useful
as a whole. Move to retrieval when the corpus grows, questions need selective evidence,
source updates are frequent, or full-context latency and cost are no longer acceptable.

## Notebook flow

1. Verify the downloaded Form 10-K against its manifest checksum and accession.
2. Derive a bounded context pack from two maintained official filing anchors.
3. Compare the complete filing and bounded pack on a logarithmic token chart.
4. Allocate an 8,192-token teaching window across instructions, evidence, question and output.
5. Inspect two explicit `ContextDecision` objects: pack → CAG, filing → RAG.
6. Build two prompts with the same exact evidence prefix and changing questions.
7. Run both through Ollama or OpenAI using the shared model boundary.
8. Inspect latency and cache telemetry without inferring a cache hit.
9. Validate the first answer against transparent grounding checks.
10. Record the real-document CAG/RAG boundary for the capstone.

## Guided notebook pacing

| Time | Activity | Expected evidence |
|---:|---|---|
| 0–5 min | Name Context, Cache, Memory, Grounding and RAG | five-row distinction table |
| 5–10 min | Verify the SEC filing and build the pack | accession, checksum and size |
| 10–14 min | Compare filing vs pack | logarithmic token figure |
| 14–18 min | Inspect both `ContextDecision` routes | pack → CAG; filing → RAG |
| 18–23 min | Ask two stable-prefix questions | measured calls, no invented cache claim |
| 23–27 min | Run grounding checks | explicit evidence checklist |
| 27–30 min | Record the architecture boundary | reasoned route in application state |

## Visual teaching contract

The notebook produces four executable figures:

- complete filing versus bounded pack;
- complete context-window allocation;
- observed latency for two repeated-prefix calls;
- the real-document CAG/RAG decision boundary.

The slide deck mirrors these mechanisms with editable explanatory diagrams. Students
should be able to connect each slide diagram to a notebook cell that calculates or
tests the same concept.

## Checkpoint questions

1. Why must reserved output tokens be part of the context decision?
2. Why is the changing question placed after the source document?
3. Does a lower second-call latency prove that prompt caching occurred?
4. What can fail even when the complete document fits the model window?
5. Which application signal should trigger the route from CAG to RAG?

Answers: reserve output so generation remains possible; put the changing question
after the reusable prefix; latency alone cannot prove a cache hit; relevant evidence
can be diluted even when the prompt fits; route to RAG when complete estimated input
exceeds available capacity or when maintained evaluation favours selective retrieval.

## Challenge answer

With a document estimate of 5,500 tokens, 650 tokens of prompt overhead and 1,500
reserved output tokens, the total allocation is 7,650 tokens. It fits inside an
8,192-token window.

With 2,200 reserved output tokens, the allocation becomes 8,350 tokens. The complete
document path must be rejected and routed to RAG or a larger verified context window.

## Instructor notes

- Do not present the four-characters-per-token estimate as a tokenizer.
- Ask students to predict the route before running each decision cell.
- Emphasize that the route depends on the artifact actually sent: the bounded pack
  and the complete filing legitimately produce different decisions.
- If a live provider omits a citation, use the failure to reinforce the difference
  between prompt fit and groundedness.
- Keep the maintained live question explicit: use only F1/F2, require bracketed
  citations, round disclosed monetary values to one decimal place, and forbid derived
  ratios. This stabilizes the evidence contract without forcing identical prose.
- Show provider cache telemetry only when it is actually returned.
- Keep PDF parsing and chunking out of this lesson; those are taught after the need for
  retrieval has been established.

## Sources

- [NVIDIA fiscal 2026 Form 10-K](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm)
- [OpenAI prompt caching guide](https://developers.openai.com/api/docs/guides/prompt-caching)
- [Liu et al., “Lost in the Middle: How Language Models Use Long Contexts”](https://arxiv.org/abs/2307.03172)
