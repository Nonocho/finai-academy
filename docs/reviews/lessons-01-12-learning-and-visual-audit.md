# Lessons 01–12 learning and visual audit

**Date:** 2026-08-25  
**Scope:** learning design, notebook clarity, visible learning, presentation narrative, and authoritative visual sourcing  
**Excluded:** code correctness, automated tests, live-provider behavior, and implementation changes

## Executive judgment

The course is currently **7.2/10** as a learning product.

Its strongest asset is the cumulative engineering sequence. Learners build one coherent financial-AI system through provider boundaries, structured outputs, context, retrieval, evaluation, workflows, agents, MCP, planning, and trajectory evaluation. The course also makes unusually good choices around provenance, bounded autonomy, typed tools, abstention, and failure ownership.

The learning experience does not yet express that strength. The notebooks often read like executable certification artifacts rather than guided learning experiences. The presentations are polished at the level of alignment, typography, color, and consistency, but they are visually and rhythmically repetitive. Across the twelve audited decks there are **122 slides and no embedded pictures**. Most slides use the same white canvas, dark-blue boxes, colored bars, five-step flows, and footer treatment. Twenty-four slides are the repeated quiz/answer pair, so nearly one fifth of the slide inventory uses an identical low-energy format.

The central redesign principle should be:

> Show the real financial or engineering object first, explain the mechanism second, and expose the implementation only after the learner understands what they are trying to observe.

## Audit method

The audit covered:

- all twelve notebooks, including every markdown and code cell;
- all twelve PowerPoint decks, rendered as 122 slides;
- notebook cell counts, markdown volume, code volume, plotting/display behavior, and committed output state;
- deck text density, layout repetition, external-image use, and source-note coverage;
- current authoritative visual candidates from primary documentation, original research, official filings, and high-quality benchmark providers.

The notebooks were not executed. This is intentionally a pedagogical and editorial review, not a code-readiness review.

## Scoring rubric

| Dimension | Question |
|---|---|
| Learning job | Does the lesson produce one clear capability and a reason to care? |
| Narrative | Does each step create the need for the next? |
| Notebook clarity | Can the learner understand the lab without decoding large implementation cells? |
| Visible learning | Are inputs, decisions, failures, and improvements observable? |
| Presentation quality | Does the deck tell a memorable story using varied, legible, meaningful visuals? |

## Course-wide findings

### What is already strong

1. **The sequence is coherent.** Each lesson creates a limitation that the next lesson addresses.
2. **Finance is not decorative.** NVIDIA and Schneider Electric evidence, periods, provenance, tables, and unsupported claims matter to the system design.
3. **Failure-first teaching is consistent.** The strongest moments are the vague request, invalid structured output, lost-in-the-middle boundary, lexical mismatch, table split, cross-company leakage, unsupported dependency, typed tool error, failed plan step, and good-answer/bad-path comparison.
4. **Safety and evaluation are present throughout.** The course does not wait until Lesson 12 to discuss grounding or verification.
5. **The slide titles generally make claims.** The decks avoid generic topic labels on most content slides.
6. **Speaker notes are sourced.** Every audited slide contains a source block, even though the visible slides do not yet use external imagery.

### What reduces learning quality

1. **The decks show abstractions instead of reality.** Students repeatedly see boxes representing a report, trace, schema, graph, or protocol, but rarely see the report page, trace UI, schema error, graph renderer, protocol inspector, or benchmark chart itself.
2. **Visual rhythm is almost identical across lessons.** The same design grammar makes different concepts feel interchangeable.
3. **The notebooks open with no rendered outputs.** Lessons 03–11 calculate meaningful figures, but the committed notebooks contain no output images. A learner browsing the material sees code rather than the visual insight promised by headings such as “Visual 1”.
4. **Several setup cells are too large for guided teaching.** Examples include 115 lines in Lesson 04, 87 lines in Lesson 05, 98 lines in Lesson 07, and 118 lines in Lesson 12.
5. **Verification language dominates the experience.** PASS markers, assertions, contracts, and implementation constraints are valuable, but often receive more attention than learner prediction, interpretation, and transfer.
6. **The final lessons become less explanatory.** Lesson 12 has only four markdown cells; most instructional transitions are embedded inside code cells.
7. **Quiz slides are over-allocated.** A quiz is valuable, but a separate answer slide in every short deck consumes narrative space and reinforces repetition. Answers belong in speaker notes or a shared appendix unless discussion requires the reveal.

## Required redesign standard

### Every notebook

Use the repeated learning rhythm:

1. **See:** show the real input, expected artifact, or decision.
2. **Predict:** ask the learner what should happen.
3. **Run:** execute a short, readable cell.
4. **Observe:** render a chart, table, trace, or structured object.
5. **Explain:** state what changed and why.
6. **Modify:** make one bounded change.
7. **Transfer:** connect the result to the capstone.

Visible learner-facing code cells should usually remain below 25–30 lines. Data construction, plotting helpers, fixtures, instrumentation, and service wiring should live in `src/finai_academy` when they do not teach the current mechanism.

For notebook visuals, choose one of these policies and apply it consistently:

- commit the deterministic offline outputs after sanitization; or
- keep outputs clear but include a small, sourced “reference result” image beside every major observation.

The current state—headings promising visuals with no visible output until execution—is the weakest option.

### Every presentation

Each lesson deck should contain:

- one opening real-world tension or financial artifact;
- one authoritative external visual or interface screenshot;
- one original explanatory diagram tied directly to notebook code;
- one actual notebook output or application-state screenshot;
- one failure/comparison slide;
- one synthesis or decision slide;
- one short knowledge check, with answers in notes or appendix.

External visuals must be cropped to the precise teaching point, not pasted as full webpages. Add a short source label on the slide and a complete `[Sources]` entry in speaker notes with URL, publisher, title, and access date. Prefer primary sources. Confirm redistribution terms before public or commercial distribution; educational use and attribution do not automatically grant broad republication rights.

## Lesson scorecard

| Lesson | Notebook | Deck | Visible learning | Overall | Priority |
|---:|---:|---:|---:|---:|---|
| 01 | 8.0 | 6.5 | 5.0 | **7.0** | High: establish the new standard |
| 02 | 8.0 | 6.5 | 5.5 | **7.0** | High |
| 03 | 8.5 | 6.5 | 7.5 | **7.5** | Medium |
| 04 | 8.5 | 7.5 | 8.0 | **8.0** | Preserve and polish |
| 05 | 7.5 | 7.0 | 8.0 | **7.5** | High: reduce overload |
| 06 | 7.5 | 7.0 | 8.0 | **7.5** | Medium |
| 07 | 7.5 | 6.5 | 7.5 | **7.0** | High |
| 08 | 8.0 | 7.0 | 7.5 | **7.5** | Medium |
| 09 | 8.0 | 7.0 | 7.5 | **7.5** | Medium |
| 10 | 8.0 | 6.5 | 7.5 | **7.5** | High: show the real protocol |
| 11 | 7.5 | 7.0 | 7.5 | **7.5** | Medium |
| 12 | 5.5 | 6.5 | 6.5 | **6.0** | Critical rewrite |

## Lesson-by-lesson audit

### Lesson 01 — Local and hosted model gateway

**Learning job:** Strong. It correctly establishes that the model is a component inside an application boundary, not the application itself.

**Notebook judgment:** The progression from vague request to grounded evidence is useful, and latency/metadata/streaming are relevant. The notebook is over-scaffolded at the end: troubleshooting, student checklist, verification, capstone integration, and recap repeat overlapping messages. Keep the challenge and recap; move most operational troubleshooting to the shared guide and shorten the checklist. The lesson needs one immediate visual comparison of provider, latency, privacy, cost, and model capability. Text output alone makes the first model call feel less substantial than it is.

**Cell actions:** Keep cells 12–28 as the conceptual core. Compress cells 3–6. Move most of cell 34 to shared troubleshooting. Merge cells 35–37 into one concise “What you can now explain and build” close.

**Deck judgment:** Clear but too abstract. The architecture, message contract, measurement, failure, evidence, and verification slides all use the same diagram grammar. The deck never shows Ollama, a provider response, a model benchmark, or a real run.

**Visual plan:** Replace one architecture slide with a current model intelligence-versus-cost or intelligence-versus-latency chart from [Artificial Analysis](https://artificialanalysis.ai/evaluations/artificial-analysis-intelligence-index). Add a tightly cropped screenshot of the selected `qwen3:8b` model page from the [official Ollama library](https://ollama.com/library/qwen3). Use the benchmark to teach that model selection is a multi-objective decision, not a leaderboard choice. Date the screenshot because model rankings change.

### Lesson 02 — Prompt engineering and structured outputs

**Learning job:** Strong. The distinction between parseable JSON, schema validity, and financial validity is one of the best ideas in the course.

**Notebook judgment:** The six-part prompt progression is well motivated, but code cell 21 contains 74 lines and hides the comparison mechanism. Split it into one cell that displays the six prompt stages and one cell that displays the validation result. Show the invalid candidate and the exact Pydantic error side by side before explaining the three validation layers. The current lesson says “structured outputs” more often than it lets the learner see the structural difference.

**Cell actions:** Keep cells 9–15 and 20–26. Split cell 21. Reduce provider/setup repetition in cells 3 and 16–19 by calling one shared lesson bootstrap. Make the knowledge-check answers an instructor reveal rather than expanded notebook text by default.

**Deck judgment:** The story is correct but entirely schematic. The audience needs to see a real malformed output, a validation error, and a schema-bound response.

**Visual plan:** Use a cropped example from the [OpenAI Structured Outputs guide](https://developers.openai.com/api/docs/guides/structured-outputs) to show JSON Schema as a provider contract. Pair it with a small `model_json_schema()` or validation-error screenshot grounded in the [Pydantic JSON Schema documentation](https://docs.pydantic.dev/latest/concepts/json_schema/). The slide should visually contrast “valid JSON” with “accepted financial object”.

### Lesson 03 — Context engineering and CAG

**Learning job:** Very strong. The notebook clearly separates context, cache, memory, grounding, and retrieval, then creates a deterministic CAG/RAG boundary.

**Notebook judgment:** The computed visuals are relevant and well sequenced. Code cell 8 contains 86 lines of document setup; move the teaching extract into a fixture and let the notebook inspect it. Keep the budget allocation, reusable-prefix, lost-in-the-middle, and route-boundary figures. Avoid implying a cache hit from latency; the notebook already handles this correctly.

**Cell actions:** Replace cell 8 with a short loader plus an immediately visible source table. Keep cells 10, 14, 22, 28, and 30. Shorten setup and repeated provider explanations.

**Deck judgment:** The concepts are present, but the slides do not show what context engineering looks like in a real product or provider response.

**Visual plan:** Use the “prompt engineering versus context engineering” visual from Anthropic’s [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents). Add a cropped token-usage or cached-input example from the [OpenAI prompt-caching documentation](https://developers.openai.com/api/docs/guides/prompt-caching). Keep the course’s original budget figure as the bridge to the notebook.

### Lesson 04 — Naive RAG from first principles

**Learning job:** Excellent. This is the clearest complete lesson because it isolates corpus, representation, ranking, top-k, context assembly, generation, and lexical failure.

**Notebook judgment:** The six visuals genuinely teach. The main weakness is code cell 8 at 115 lines: passage construction and source definitions obscure the first-principles retriever. Move corpus data to a fixture and leave a short, inspectable list of passages. Split cells 32–33 so the improvement map and verification are separate learner moments.

**Cell actions:** Keep the conceptual sequence in cells 9–31. Replace cell 8 with a compact corpus load/display. Preserve the top-k challenge; it is a good transfer task.

**Deck judgment:** This is the most visually successful current deck because the heatmap, ranking bars, prompt composition, and failure boundary vary the silhouette. It still lacks a connection to the original research and the actual financial evidence.

**Visual plan:** Add the architecture figure from the original NeurIPS paper, [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401), with a clear note that the classroom implementation is intentionally simpler. Pair it with a cropped NVIDIA filing passage from the [official SEC filing](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm) to show the object being retrieved.

### Lesson 05 — Financial documents and chunking

**Learning job:** Important but overloaded. It currently teaches parsing, canonical blocks, seven chunking strategies, semantic thresholds, hierarchy, contextual enrichment, token inflation, retrieval comparison, proposition chunking, and table failure in one session.

**Notebook judgment:** The material is relevant, but not all of it belongs in the 70-minute core path. Teach three core strategies in class: fixed, structure-aware, and hierarchical/contextual. Move recursive, semantic, LLM contextual, and proposition variants into comparison or extension sections. Code cells 12, 25, and 27 are too large for guided explanation. The strongest moment is the table-integrity failure; move it earlier so it becomes the reason for the strategies rather than a late certification step.

**Cell actions:** Open with cells 9–10 and 26–27. Then introduce the canonical block and three core strategies. Move cells 14–21 into an “advanced strategy gallery”. Keep provenance, raw-text preservation, and table integrity as non-negotiable constraints.

**Deck judgment:** The deck contains the right ideas but has the highest average text density in the course. It represents a financial table with drawn shapes instead of showing the real report page.

**Visual plan:** Use side-by-side crops from the [NVIDIA FY2026 SEC filing](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm) and the [Schneider Electric FY2025 results PDF](https://www.se.com/ww/en/assets/564/document/528237/release-fy-results-2025.pdf?p_File_Name=2025+FY+Results&p_enDocType=Financial+release). Annotate row, column, heading, footnote, and page provenance. A secondary implementation screenshot can come from the official [Docling table-export example](https://docling-project.github.io/docling/_generated/examples/export_tables/), clearly labelled as an alternative parser rather than the course baseline.

### Lesson 06 — Embeddings and hybrid retrieval

**Learning job:** Strong. Exact-term failure and cross-company leakage give hybrid search and pre-filtering real jobs.

**Notebook judgment:** The visuals are meaningful, but the 92-line bootstrap cell and plotting helpers make the lab look more complex than the retrieval logic. Move annotation and plotting helpers into the course package. The learner should directly manipulate the filter, RRF weights, and reranker—not debug chart layout. Explain earlier that the course uses RRF while vendor products may use different fusion semantics.

**Cell actions:** Shorten cell 4 substantially. Keep cells 9–24. Retain the moved-ranking challenge, but add a small prediction before the weight change.

**Deck judgment:** The deck covers cosine similarity, eligibility, fusion, reranking, storage, and measurement, but the visual language remains a sequence of bars and boxes. It needs one external industry implementation to show that these are real production choices.

**Visual plan:** Use the concrete fusion example from the [Weaviate hybrid-search concepts](https://docs.weaviate.io/weaviate/concepts/search/hybrid-search) to contrast rank fusion with score fusion. Use the architecture/trade-off table from the [Pinecone hybrid-search guide](https://docs.pinecone.io/guides/search/hybrid-search) only if the slide explicitly explains that the course’s two-channel-plus-RRF design is one valid pattern, not the only implementation.

### Lesson 07 — RAG evaluation and tracing

**Learning job:** Strong and necessary. The separation of retrieval, answer, abstention, and trace evidence is pedagogically sound.

**Notebook judgment:** The prose is clear, but too much implementation is concentrated in code cells 4, 6, 8, and 14. The learner should see a case card, a metric result, and a trace before seeing evaluation orchestration. Move predictor factories, offline answers, and plotting machinery behind small functions. The failure classification is one of the best cells and should be visually prominent.

**Cell actions:** Preserve the sequence and concepts. Reduce cell 8 from 98 lines to a visible call such as `evaluate_case(case, configuration)`. Reduce cell 14 from 72 lines to a comparison call plus a readable result frame. Keep the challenge adding an evidence-sufficiency span.

**Deck judgment:** The deck is accurate but presents MLflow as another five-box flow. Students never see the product they will use.

**Visual plan:** Add an actual trace-table/detail screenshot from the official [MLflow trace UI documentation](https://mlflow.org/docs/latest/genai/tracing/observe-with-traces/ui). Pair it with the retrieval-versus-generation framework from [MLflow’s RAG evaluation guide](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/rag/). The screenshot should highlight spans, inputs/outputs, timing, and failure location.

### Lesson 08 — Workflows versus agents

**Learning job:** Very strong. The EUR dependency is a concrete and honest reason to introduce observation-dependent control.

**Notebook judgment:** The direct workflow, unsupported dependency, agent completion, trace, and loop-budget failure form a coherent story. Move the 53-line live policy class out of the learner path. Replace some Matplotlib-drawn architecture figures with direct rendering of the trajectory table or an interactive graph/state view.

**Cell actions:** Keep cells 5–10 and 12–23. Collapse cell 11 to model selection plus a short policy factory. Add a learner decision before revealing whether the workflow or agent is appropriate.

**Deck judgment:** The deck communicates the autonomy ladder but visually treats every architecture as another bar sequence. It needs a respected external framing to anchor the vocabulary.

**Visual plan:** Use the workflow-versus-agent distinction and one pattern diagram from Anthropic’s [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents). Follow it immediately with the course’s NVIDIA-to-EUR example so the external framework does not replace the finance-specific teaching.

### Lesson 09 — Self-correcting financial agent

**Learning job:** Strong. The typed `PE` to `P/E` correction and bounded retry make self-correction concrete and safe.

**Notebook judgment:** The sequence is readable. The graph should be rendered from the actual compiled LangGraph rather than recreated manually with Matplotlib. The 40-line live correction policy can be hidden behind a factory after its input/output contract is explained. Emphasize that the first invalid action is injected for teaching; the notebook already states this, but it should also be visible on the output.

**Cell actions:** Keep cells 7–18. Replace cell 6 with actual graph rendering. Shorten cell 10. Keep the repeated-error failure lab and distinguish model correction budget from infrastructure retry policy.

**Deck judgment:** The slide story is good but remains visually indistinguishable from Lessons 08 and 11.

**Visual plan:** Use an official LangGraph graph/state visual from the [LangGraph Graph API overview](https://langchain-ai.github.io/langgraph/how-tos/visualization/) and pair it with a screenshot of this course’s rendered recovery graph. The external visual establishes nodes, edges, and state; the original graph teaches the actual finance correction path.

### Lesson 10 — Financial MCP

**Learning job:** Strong. Host, client, server, transport, resources, tools, prompts, discovery, and permission are correctly separated.

**Notebook judgment:** The real local lifecycle is valuable. The inspection of tracked decorators and source contracts is less compelling than inspecting the live discovered schemas. Prefer the protocol’s returned capability objects. The notebook should show the MCP Inspector or protocol messages, not only DataFrames and custom timelines.

**Cell actions:** Keep cells 10–23. Replace or reduce cells 8–9 with live discovery results. Add one screenshot or embedded view of the Inspector showing the resource, two tools, prompt, and typed error.

**Deck judgment:** The deck is conceptually disciplined but entirely abstract. This is the clearest missed screenshot opportunity in the course.

**Visual plan:** Use the official host/client/server model from the [MCP architecture specification](https://modelcontextprotocol.io/specification/2025-06-18/architecture). Add a screenshot from the official [MCP Inspector repository](https://github.com/modelcontextprotocol/inspector), cropped to the tools/resources/prompts interface. The source image in that repository is `mcp-inspector.png`; confirm its current license and version before embedding.

### Lesson 11 — Plan-and-execute financial analyst

**Learning job:** Strong but compressed. The core idea—retain successful work and replan only the unfinished tail—is valuable and differentiated from Lesson 09.

**Notebook judgment:** The notebook uses many generated figures, but the most useful artifact is the typed plan itself. Show the initial plan JSON, host validation, failed step, revised tail, evidence coverage, and cited brief as one progressive visual story. The implementation hidden inside `run_lesson11()` is appropriate, but the learner needs a clearer map of which role owns each decision.

**Cell actions:** Keep cells 7–27. Merge some separate plotting cells and spend the recovered space on learner interpretation. Make the plan diff the central visual: retained steps, rejected step, superseded step, and new steps.

**Deck judgment:** The deck is accurate and relatively concise, but it repeats the same control-bar vocabulary as Lessons 08–10.

**Visual plan:** Use the planning-agent comparison from LangChain’s [Plan-and-Execute Agents](https://blog.langchain.dev/planning-agents/) or its earlier [Plan-and-Execute overview](https://www.langchain.com/blog/plan-and-execute-agents). Pair it with an original before/after plan diff based on the NVIDIA/Schneider mission. Label older framework material by publication date and use it for the pattern, not as current API documentation.

### Lesson 12 — Evaluating agentic systems with MLflow

**Learning job:** Essential, but the notebook currently underdelivers pedagogically. It contains strong concepts—public state, trajectory versus answer, versioned cases, five metrics, failure ownership, and deterministic citation gates—but presents them as a long execution script.

**Notebook judgment:** This is the critical rewrite. There are only four markdown cells. Cells 3–25 run almost continuously, and the failure lab is a 118-line code cell whose heading is a code comment. Learners must infer the narrative from prints, assertions, and tables. Split the notebook into explicit markdown sections: evaluation anatomy, dataset, reference trajectory, trajectory metrics, answer metrics, aligned comparison, failure lab, MLflow inspection, optional judge, verification, and transfer. Hide persistence and table-building plumbing. Show the MLflow UI directly after the first trace is logged.

**Cell actions:** Retain the underlying examples but reorganize all cells 3–25. Split cell 8. Split cell 13. Replace cell 21 with at least four cells around one visible failure question. Turn the commented `# ## Failure lab` and `# ## Verification` markers into real markdown cells. Add prediction and interpretation prompts before each scorecard.

**Deck judgment:** The deck has some useful tables but still looks like the rest of the course and does not show MLflow. The lesson’s most important idea—same answer, different path—is visually present, but the professional evaluation surface is absent.

**Visual plan:** Use the multi-turn evaluation anatomy and grader taxonomy from Anthropic’s [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents). Add a real evaluation or trace-comparison screenshot from [MLflow Agent Evaluation](https://mlflow.org/genai/evaluations) and the [MLflow trace UI](https://mlflow.org/docs/latest/genai/tracing/observe-with-traces/ui). The redesigned deck should visibly separate dataset, transcript/trajectory, outcome, grader, and release decision.

## Recommended implementation order

### Phase 1 — Establish the quality bar

Redesign Lesson 01 first. It is the learner’s first executable experience and can establish the new slide rhythm, screenshot treatment, source-note format, notebook visual policy, and reduced boilerplate.

### Phase 2 — Repair the learning bottlenecks

1. Rewrite Lesson 12’s notebook narrative.
2. Reduce Lesson 05 to a core classroom path plus advanced strategy gallery.
3. Simplify Lesson 07’s visible code and add MLflow UI evidence.
4. Add real protocol/UI evidence to Lesson 10.

### Phase 3 — Apply the visual system across the remaining decks

Revise Lessons 02–04, 06, 08, 09, and 11 using the Lesson 01 standard while preserving their strongest original diagrams and failure labs.

## Definition of a polished lesson

A lesson is ready when:

- the learner can state the job of the lesson in one sentence;
- the opening uses a real problem or artifact rather than an abstract architecture;
- no essential concept is hidden inside a code cell longer than 30 lines;
- every major code section produces a visible, interpretable result;
- the learner predicts at least one failure before running it;
- every external screenshot has an on-slide source label and a complete notes citation;
- the deck contains meaningful silhouette changes without losing visual identity;
- the final slide asks for a decision, transfer, or application—not merely recall;
- the notebook and deck show the same central mechanism using the same terminology; and
- a learner can explain what changed, why it improved, and what remains unsafe.

## Final recommendation

Do not perform twelve independent cosmetic refreshes. First redesign Lesson 01 as the editorial and visual template, then perform the structural rewrites in Lessons 12, 05, 07, and 10. Once those patterns are proven, revise the remaining lessons in order. This protects the course’s strong engineering architecture while making the learning experience visibly worthy of it.
