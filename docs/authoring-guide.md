# Authoring guide

## Language and voice

- All audience-facing material is written in English.
- Prefer clear technical language over marketing language.
- Explain acronyms when they first appear.
- Use finance examples consistently across chapters.
- Separate facts, calculations, assumptions, and interpretations.

## Chapter contract

Every chapter must contain:

- one chapter brief in `chapters/`;
- one `.pptx` deck in `decks/`;
- one guided `.ipynb` notebook in `notebooks/`;
- checkpoint questions;
- a practical exercise;
- an instructor solution or answer key;
- sources for non-trivial claims and external assets.

## Notebook contract

- Restart-and-run-all must succeed.
- Do not require a paid API for the baseline path.
- Put provider-specific code behind a small adapter.
- Set random seeds when supported.
- Never commit API keys, private client files, or licensed datasets.
- Display expected outputs for non-deterministic steps as ranges, not guarantees.
- Render important system state as executable visuals, not static screenshots.
- From Lesson 03 onward, include at least two meaningful code-generated figures.
- Retrieval and evaluation lessons must visualize document structure, retrieved
  evidence, rankings, or evaluation results rather than relying on printed lists alone.

## Deck contract

- Use 16:9 widescreen.
- Keep one primary claim per slide.
- Use takeaway titles, not topic labels.
- Keep deck titles at 50 pt or larger and slide titles at 35 pt or larger.
- Use 24 pt or larger for subheadings and 16 pt or larger for body copy.
- Put citations and asset sources in speaker notes.
- End with application or synthesis, not a generic thank-you slide.
- Every technical lesson must contain at least one original explanatory diagram.
- Mirror the notebook's central mechanism so students can connect the conceptual
  diagram to the code they execute.
- Prefer simple flows and decision boundaries over decorative diagrams.

## Public-content policy

This repository may be inspired by public patterns and technical documentation,
but it must contain original explanations, exercises, datasets, diagrams, and
implementations. Do not reproduce private course lessons, proprietary screenshots,
or repository code from third parties.
