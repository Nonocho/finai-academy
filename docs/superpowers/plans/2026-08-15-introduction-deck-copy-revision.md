# Introduction Deck Copy Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Revise the existing introduction deck to use factual professional language and the exact requested footer.

**Architecture:** Import the committed PowerPoint with `@oai/artifact-tool`, edit only the named inherited text elements, and export the result back to the same deck path. Preserve all layouts, styling, timings, notes, and non-targeted content.

**Tech Stack:** JavaScript ES modules, `@oai/artifact-tool`, bundled presentation rendering and QA utilities.

## Global Constraints

- Do not add or remove slides.
- Preserve the current visual identity, slide sequence, schedules, and technical content.
- Use `First Finance - Arnaud Demes` as the exact footer on slides with course chrome.
- Keep all visible copy factual, concise, and suitable for a professional technical class.
- Do not modify unrelated repository changes.

---

### Task 1: Revise visible copy in the existing deck

**Files:**
- Modify: `decks/00-course-introduction.pptx`
- Create temporarily: `$TMP_DIR/edit-introduction.mjs`

**Interfaces:**
- Consumes: the committed 12-slide PowerPoint and its named text elements.
- Produces: the same 12-slide PowerPoint with revised text and unchanged visual structure.

- [ ] **Step 1: Inspect all source slides and named text elements**

Run the template inspection utility against `decks/00-course-introduction.pptx` and review all 12 slide renders and the element inventory.

- [ ] **Step 2: Define the exact edit map**

Use these replacements:

```text
subtitle: Build a Financial Analyst Copilot in Two Days
  -> Two-day technical course and capstone project

title-2: In two days, one analyst copilot becomes a complete AI system
  -> Course objective: build a financial analyst copilot across the full stack

promise-bottom: The objective is not to know the vocabulary. It is to leave with an inspectable application that works.
  -> At the end of the course, the application runs end to end and exposes its sources, calculations, tool calls and evaluations.

section-2: PROMISE
  -> OBJECTIVE

title-3: The product comes first
  -> Capstone project: Financial Analyst Copilot

section-3: PRODUCT
  -> CAPSTONE

title-4: Each failure creates the need for the next engineering layer
  -> Technical progression of the application

provider-command: Change configuration — never the lesson code.
  -> Provider selection changes through configuration only.

title-12: Success means a working, inspectable and defensible copilot
  -> Expected deliverables and validation criteria

closing: Now build the first response.
  -> Next: Model Gateway

section-12: SUCCESS
  -> VALIDATION

footer-2 through footer-12: FinAI Academy — Arnaud Demes
  -> First Finance - Arnaud Demes
```

- [ ] **Step 3: Edit and export with Artifact Tool**

Import the existing PPTX, resolve each target by its exact element name, replace its text without changing typography or geometry, assert that every target is found exactly once, and export to the final path.

- [ ] **Step 4: Verify the edit inventory**

Re-inspect the exported deck and confirm that all requested replacements are present, no old footer remains, and the presentation still contains 12 slides.

### Task 2: Render and visually validate the revised deck

**Files:**
- Verify: `decks/00-course-introduction.pptx`
- Create temporarily: `$TMP_DIR/final-render/`

**Interfaces:**
- Consumes: the revised 12-slide deck.
- Produces: rendered slide images, overflow results, and a visual QA ledger.

- [ ] **Step 1: Render every final slide**

Run `render_slides.py` and confirm that slides 1 through 12 render successfully.

- [ ] **Step 2: Run automated slide checks**

Run `slides_test.py decks/00-course-introduction.pptx` and require a passing overflow result.

- [ ] **Step 3: Inspect each slide at full size**

Check title wrapping, body clipping, footer consistency, page numbering, and unintended overlap on each of the 12 rendered PNGs.

- [ ] **Step 4: Commit the isolated deck revision**

```bash
git add decks/00-course-introduction.pptx docs/superpowers/plans/2026-08-15-introduction-deck-copy-revision.md
git diff --cached --check
git commit -m "content: make the introduction deck copy more factual"
```
