# Day 1 delivery readiness

Review date: 2026-08-21  
Reviewed commit: `52900390ebe912912220a424310aa75122d597e8`  
Environment: macOS 26.6, Apple Silicon, Python 3.13.9

## Decision

Day 1 is ready for an instructor-led test class. The mandatory repository,
installation, notebook, Ollama and presentation gates passed. The OpenAI path is
documented but was not executed because `OPENAI_API_KEY` was not configured.

## Evidence

| Gate | Result | Evidence |
| --- | --- | --- |
| Code quality | PASS | `uv run ruff check .` returned no issues. |
| Test suite | PASS | `uv run pytest -q`: 192 passed. |
| Repository contract | PASS | `scripts/validate_repo.py` reported a valid structure. |
| Notebook contract | PASS | All seven canonical Day 1 notebooks passed validation. |
| Documentation | PASS | Five onboarding tests passed, including all local Markdown links. |
| Offline setup | READY | Python and course dependencies passed; optional services were skipped explicitly. |
| Ollama setup | READY | Service, `qwen3:8b` and `qwen3-embedding:0.6b` passed. |
| Live notebooks | PASS | Lessons 01–07 all executed successfully with the Ollama provider. |
| OpenAI | NOT CONFIGURED | No key was present; no OpenAI execution is claimed. |
| Fresh clone | PASS | `uv sync --frozen --extra ai --extra rag --extra evaluation --extra dev`, offline readiness, Jupyter import and repository validation passed in `/private/tmp/finai-onboarding.P0LBmP/finai-academy`. |
| Introduction deck | PASS | 13 slides rendered; overflow test passed; template fidelity reported zero issues. |
| Deck integrity | PASS | 13 sourced speaker notes, no placeholders, no font shrink and the required footer on every slide. |

The seven live execution artifacts are retained in
`/private/tmp/finai-notebooks-live.n4tERw` for local inspection. Temporary
validation directories were not deleted.

## Delivery score

| Area | Score | Rationale |
| --- | ---: | --- |
| Clarity | 9.7/10 | The root README gives one path to setup, the Day 1 guide and Lesson 01. |
| Installation | 9.8/10 | Cross-platform instructions, `.env`, Ollama models and a readiness command are explicit; a fresh locked install passed. |
| Technical reliability | 9.8/10 | The full suite, repository contracts and seven live Ollama notebooks passed. |
| Visual quality | 9.7/10 | The introduction deck is consistent, editable, sourced and passed render, overflow and fidelity QA. |
| Overall Day 1 readiness | **9.75/10** | All mandatory acceptance criteria passed; OpenAI remains an optional untested path on this machine. |

## Instructor start path

1. Follow [Getting started](../getting-started.md).
2. Run `uv run python scripts/setup_check.py --provider ollama`.
3. Open [Lesson 01](../../notebooks/01_model_gateway.ipynb).
4. Use the [Day 1 student guide](../day-1-student-guide.md) for the sequence and checkpoints.

Scope note: this review certifies Day 1 only. Lessons 08–12 and the integrated
capstone remain the Day 2 build scope.
