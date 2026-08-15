# Lesson 03 — Context Engineering and Cache-Augmented Generation

**Schedule:** Day 1, 10:45–11:30

**Format:** 15 minutes of concepts and diagrams, 30 minutes of guided notebook work

**Capstone increment:** complete-document grounded answer with an explicit CAG/RAG route

## Teaching outcome

Students should leave with one operational distinction:

- prompt engineering defines the behaviour and output contract;
- context engineering defines the information made available to the model.

They build the simplest document-grounded path first: place one bounded NVIDIA
document pack in the prompt, keep it stable across questions, budget the complete
request, and stop when the document no longer fits safely.

## Why this lesson comes before RAG

Retrieval is additional infrastructure. It should solve a demonstrated constraint,
not appear by default. A complete-document path is often sufficient for a short,
bounded source and gives students a clean baseline against which later retrieval can
be evaluated.

The lesson ends with a controlled failure. A synthetic neutral appendix pushes the
same evidence pack beyond the application budget. The application chooses RAG before
calling the model rather than truncating silently.

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

The synthetic stress document visibly positions the real NVIDIA evidence in the middle,
but the chart does not invent a model-accuracy score.

### 4. CAG and RAG solve different document shapes

Use direct context or CAG when the source is bounded, stable, within budget and useful
as a whole. Move to retrieval when the corpus grows, questions need selective evidence,
source updates are frequent, or full-context latency and cost are no longer acceptable.

## Notebook flow

1. Load a curated teaching extract from NVIDIA's fiscal 2026 Form 10-K.
2. Visualize the estimated token contribution of each source section.
3. Allocate an 8,192-token teaching window across instructions, document, question and output.
4. Build two prompts with the same exact document prefix.
5. Run both through Ollama or OpenAI using the shared model boundary.
6. Inspect actual latency and any provider metadata without inferring a cache hit.
7. Validate the first answer against transparent grounding checks.
8. Place the evidence in the middle of a longer synthetic document.
9. Visualize the position and the CAG/RAG boundary.
10. Record the route selected by the application.

## Visual teaching contract

The notebook produces five executable figures:

- tokens by filing section;
- complete context-window allocation;
- observed latency for two repeated-prefix calls;
- evidence position inside the stress document;
- the point at which the application switches from CAG to RAG.

The slide deck mirrors these mechanisms with editable explanatory diagrams. Students
should be able to connect each slide diagram to a notebook cell that calculates or
tests the same concept.

## Checkpoint questions

1. Why must reserved output tokens be part of the context decision?
2. Why is the changing question placed after the source document?
3. Does a lower second-call latency prove that prompt caching occurred?
4. What can fail even when the complete document fits the model window?
5. Which application signal should trigger the route from CAG to RAG?

## Challenge answer

With a document estimate of 5,500 tokens, 650 tokens of prompt overhead and 1,500
reserved output tokens, the total allocation is 7,650 tokens. It fits inside an
8,192-token window.

With 2,200 reserved output tokens, the allocation becomes 8,350 tokens. The complete
document path must be rejected and routed to RAG or a larger verified context window.

## Instructor notes

- Do not present the four-characters-per-token estimate as a tokenizer.
- Ask students to predict the route before running each decision cell.
- Emphasize that the synthetic appendix contains no financial evidence.
- If a live provider omits a citation, use the failure to reinforce the difference
  between prompt fit and groundedness.
- Show provider cache telemetry only when it is actually returned.
- Keep PDF parsing and chunking out of this lesson; those are taught after the need for
  retrieval has been established.

## Sources

- [NVIDIA fiscal 2026 Form 10-K](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm)
- [OpenAI prompt caching guide](https://developers.openai.com/api/docs/guides/prompt-caching)
- [Liu et al., “Lost in the Middle: How Language Models Use Long Contexts”](https://arxiv.org/abs/2307.03172)
