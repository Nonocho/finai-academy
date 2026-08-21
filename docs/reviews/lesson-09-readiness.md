# Lesson 09 delivery readiness

Review date: 2026-08-21  
Lesson: Self-Correcting Financial Agent  
Environment: macOS 26.6, Apple Silicon, Python 3.13.9

## Decision

Lesson 09 is ready for an instructor-led test class. The lesson extends the bounded
agent from Lesson 08 with explicit LangGraph state, typed tool errors, conditional
routing and two application-owned budgets. The offline and Ollama paths passed. The
OpenAI path is implemented and documented but was not executed because
`OPENAI_API_KEY` was not configured on this machine.

## Verification evidence

| Gate | Result | Evidence |
| --- | --- | --- |
| Full test suite | PASS | `.venv/bin/pytest -q`: 226 passed in 43.49 seconds. |
| Lesson 09 package | PASS | 12 targeted tests cover the fixture, module, graph, notebook, chapter, indexes and deck. |
| Code quality | PASS | `.venv/bin/ruff check .`: no issues. |
| Notebook source contract | PASS | Lesson 09 passed `scripts/validate_notebooks.py`; the checked-in notebook is output-free. |
| Offline execution | PASS | Exact marker `LESSON_09_PASS`; five code-generated figures and all assertions passed. |
| Ollama live execution | PASS | `qwen3:8b`; five figures and the exact final marker were present. |
| Typed correction | PASS | The trace keeps `PE`, `unsupported_metric`, the valid alternatives and the corrected `P/E` request. |
| LangGraph implementation | PASS | The compiled `StateGraph` is exercised by unit tests and the notebook. |
| Retry budget | PASS | `MAX_RETRIES = 1` stops a second invalid model action. |
| Tool budget | PASS | `MAX_TOOL_CALLS = 4` bounds every execution, including errors. |
| Evidence guardrail | PASS | The agent refuses to complete without at least one successful observation. |
| OpenAI | NOT CONFIGURED | No key was present; no OpenAI success is claimed. |
| Deck render | PASS | Nine slides inspected as a montage and individually at full size. |
| Deck overflow | PASS | `slides_test.py` reported no overflow. |
| Template fidelity | PASS | Template fidelity check reported zero issues. |
| Speaker notes | PASS | All nine slides contain a `[Sources]` block. |

The local Ollama execution artifact is retained at
`/private/tmp/finai-lesson09-ollama/09_self_correcting_agent.ipynb`.

## Specification score

| Dimension | Weight | Score | Evidence |
| --- | ---: | ---: | --- |
| Conceptual clarity and progression | 20% | 9.8/10 | The lesson starts with one visible failure, turns it into typed feedback, then introduces state, routing and budgets only when each becomes necessary. |
| Notebook usability and visuals | 20% | 9.7/10 | The 30-minute notebook produces five visuals, an inspectable trace and one exact completion marker. |
| Technical correctness and safety | 20% | 9.8/10 | The real LangGraph graph, typed boundaries, evidence guardrail and separate retry and tool-call budgets are covered by tests. |
| Provider neutrality and recovery | 15% | 9.5/10 | Offline and Ollama passed; OpenAI uses the same structured action schema but remains unexecuted because the key is absent. |
| Deck quality and fidelity | 15% | 9.8/10 | Nine concise sourced slides passed full-size inspection, overflow and template-fidelity checks. |
| Timing and instructor readiness | 10% | 9.7/10 | The chapter provides a precise 10-minute deck, 30-minute notebook and 5-minute debrief route, plus a skip-if-late path. |

Weighted readiness score: **9.72/10**.

## Instructor start path

1. Present [the Lesson 09 deck](../../decks/09-self-correcting-agent.pptx).
2. Open [the Lesson 09 notebook](../../notebooks/09_self_correcting_agent.ipynb).
3. Use [the instructor chapter](../../chapters/09-self-correcting-agent.md) for timing and debrief prompts.
4. Keep the successful trace visible: `PE → unsupported_metric → P/E → evidence → finish`.
5. Transition to Lesson 10 only after students can explain why Python owns validation and stop conditions.
