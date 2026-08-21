# Lesson 08 delivery readiness

Review date: 2026-08-21  
Lesson: Workflows Versus Agents  
Environment: macOS 26.6, Apple Silicon, Python 3.13.9

## Decision

Lesson 08 is ready for an instructor-led test class. The source notebook, typed tool
layer, deterministic workflow, bounded agent, chapter and nine-slide deck form one
coherent 45-minute lesson. The mandatory offline and Ollama paths passed. The OpenAI
path is implemented and documented but was not executed because `OPENAI_API_KEY` was
not configured on this machine.

## Verification evidence

| Gate | Result | Evidence |
| --- | --- | --- |
| Full test suite | PASS | `.venv/bin/pytest -q`: 214 passed in 100.79 seconds. |
| Code quality | PASS | `.venv/bin/ruff check .`: no issues. |
| Notebook source contract | PASS | Lesson 08 passed `scripts/validate_notebooks.py`; the checked-in notebook is output-free. |
| Offline execution | PASS | Exact final marker `LESSON_08_PASS`; six code-generated figures and all assertions passed. |
| Ollama live execution | PASS | `qwen3:8b`; approximately 92 seconds; exact final marker present. |
| Direct workflow | PASS | `workflow_direct_status=completed`. |
| Dependency guardrail | PASS | `workflow_dependency_status=unsupported_dependency`; no conversion amount was fabricated. |
| Bounded agent | PASS | `agent_status=completed`; tool order `get_market_price → convert_currency`. |
| Step budget | PASS | Visible result `Stopped after MAX_STEPS=2.` |
| OpenAI | NOT CONFIGURED | No key was present; no OpenAI success is claimed. |
| Deck render | PASS | Nine slides inspected as a montage and individually at full size. |
| Deck overflow | PASS | `slides_test.py` reported no overflow. |
| Template fidelity | PASS | Template fidelity check reported zero issues. |
| Speaker notes | PASS | All nine slides contain a `[Sources]` block. |

The Ollama run used the checked-in market snapshot and the same two deterministic tools
as the offline fixture. No structured-output retry was visible or required during the
successful run. The local execution artifact is retained at
`/private/tmp/finai-lesson08-ollama/08_workflows_vs_agents.ipynb`.

## Specification score

| Dimension | Weight | Score | Evidence |
| --- | ---: | ---: | --- |
| Conceptual clarity and progression | 20% | 9.7/10 | The lesson begins with the architecture decision, exposes the one-pass dependency failure, then adds only the autonomy needed to resolve it. |
| Notebook usability and visuals | 20% | 9.6/10 | A 30-minute guided notebook produces six readable figures, explicit statuses and one exact completion marker. |
| Technical correctness and safety | 20% | 9.8/10 | Typed Pydantic requests, deterministic Python tools, provenance, aligned traces and explicit step budgets are covered by the passing suite. |
| Provider neutrality and recovery | 15% | 9.4/10 | Offline and Ollama passed; OpenAI uses the same structured interface but remains unexecuted on this machine because the key is absent. |
| Deck quality and fidelity | 15% | 9.7/10 | Nine sourced slides reuse the established visual system and passed full-size inspection, overflow and fidelity checks. |
| Timing and instructor readiness | 10% | 9.6/10 | The chapter provides a minute-by-minute 10-minute deck, 30-minute notebook and 5-minute debrief route. |

Weighted readiness score: **9.65/10**.

## Instructor start path

1. Present [the Lesson 08 deck](../../decks/08-workflows-vs-agents.pptx).
2. Open [the Lesson 08 notebook](../../notebooks/08_workflows_vs_agents.ipynb).
3. Use [the instructor chapter](../../chapters/08-workflows-vs-agents.md) for timing and debrief prompts.
4. Keep the architecture rule visible: use the lowest useful autonomy.
5. Transition to Lesson 09 only after students can explain why the compound workflow stops and why the bounded agent succeeds.
