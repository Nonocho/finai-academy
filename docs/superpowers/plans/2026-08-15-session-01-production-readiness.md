# Session 01 Production Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a 9.5/10 production-ready Session 01 with a dedicated technical deck, real NVIDIA evidence, a grounding rubric and verified Ollama/OpenAI portability.

**Architecture:** Keep provider construction in `finai_academy.providers`, place the deterministic classroom fixture in a focused lesson-support module, and keep the notebook centred on the gateway learning path. Build a separate six-slide technical deck while retaining the introduction deck as the course opening.

**Tech Stack:** Python 3.13, nbformat, LangChain provider adapters, pytest, Ruff, Ollama, OpenAI configuration, PowerPoint via `@oai/artifact-tool`.

## Global Constraints

- Baseline student execution must remain free through Ollama.
- Hosted execution must use an environment-provided `OPENAI_API_KEY`; never write or display the key.
- Audience-facing identity and footers use `First Finance - Arnaud Demes`.
- The NVIDIA evidence must be traceable to the official SEC filing.
- Do not modify unrelated dirty worktree files.

---

### Task 1: Lock the Session 01 notebook contract with failing tests

**Files:**
- Modify: `tests/test_notebook_contracts.py`
- Create: `tests/test_lesson_support.py`

**Interfaces:**
- Produces: an executable contract requiring labelled NVIDIA evidence, an SEC source, a grounding rubric and an imported offline fixture.

- [ ] Add a test that loads `notebooks/01_model_gateway.ipynb` and requires `F1` to `F4`, the official SEC URL, `evaluate_grounding`, a 4/4 threshold and no notebook-local `RecordedChatModel` class.
- [ ] Add a test that imports `RecordedChatModel` from `finai_academy.lesson_support` and verifies distinct vague and grounded responses.
- [ ] Run the two tests and confirm they fail because the new evidence and module do not exist.

### Task 2: Add the focused offline lesson fixture

**Files:**
- Create: `src/finai_academy/lesson_support.py`

**Interfaces:**
- Produces: `RecordedMessage` and `RecordedChatModel.invoke(messages)` for deterministic offline notebook execution.

- [ ] Implement only the two recorded behaviours required by Session 01.
- [ ] Run `tests/test_lesson_support.py` and confirm it passes.
- [ ] Run the provider and settings tests to detect boundary regressions.

### Task 3: Ground the notebook in NVIDIA evidence and add the rubric

**Files:**
- Modify: `notebooks/01_model_gateway.ipynb`
- Modify: `chapters/01-model-gateway.md`

**Interfaces:**
- Consumes: `RecordedChatModel` from `finai_academy.lesson_support`.
- Produces: `evidence_card`, `grounded_messages`, `evaluate_grounding(text)` and `grounding_result`.

- [ ] Replace the notebook-local fixture with the imported support class.
- [ ] Replace the anonymous source with four labelled, paraphrased NVIDIA fiscal 2026 facts and the official SEC source URL.
- [ ] Ask for a bounded analyst answer that cites fact identifiers and states a limitation.
- [ ] Add a transparent four-point grounding rubric and require 4/4 in the guided verification.
- [ ] Update expected results, challenge, checklist, capstone integration and chapter verification copy.
- [ ] Run the contract tests and confirm they pass.
- [ ] Validate the clean notebook and execute it offline.
- [ ] Execute it with live Ollama and require gateway PASS plus 4/4 grounding.
- [ ] Construct the OpenAI adapter from configuration; run live only if a valid environment key is already available.

### Task 4: Align the First Finance identity

**Files:**
- Modify: `assets/brand/finai-academy-style.md`
- Modify: `course.yml`
- Modify: `decks/00-course-introduction.pptx`

**Interfaces:**
- Produces: consistent audience-facing identity without changing the course title.

- [ ] Update the brand signature and footer contract to First Finance.
- [ ] Replace the introduction title marker `FINAI ACADEMY` with `FIRST FINANCE` while preserving all other slide objects.
- [ ] Render and inspect the introduction deck and run the overflow test.

### Task 5: Create the technical Model Gateway micro-deck

**Files:**
- Create: `decks/01-model-gateway.pptx`

**Interfaces:**
- Produces: a six-slide, ten-minute technical teaching sequence with speaker notes and source blocks.

- [ ] Build the six-slide narrative defined in the design using `@oai/artifact-tool`.
- [ ] Add the SEC and official OpenAI documentation URLs to relevant `[Sources]` speaker-note blocks.
- [ ] Export the editable PowerPoint.
- [ ] Render every slide, inspect each at full size and fix visual defects.
- [ ] Run the overflow test and confirm it passes.

### Task 6: Full verification and handoff

**Files:**
- Verify all changed Session 01 artifacts.

**Interfaces:**
- Produces: reproducible evidence that Session 01 is ready for a timed internal delivery.

- [ ] Run all pytest tests and Ruff checks.
- [ ] Run the repository validator without altering unrelated dirty files.
- [ ] Confirm the notebook has no stored outputs, absolute user paths or secret-like values.
- [ ] Confirm both decks have correct branding, readable copy and no overflow.
- [ ] Report any unavailable live OpenAI acceptance run explicitly rather than claiming it passed.

