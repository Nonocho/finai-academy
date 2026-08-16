# Day 1 Repository and Student Onboarding Design

**Date:** 2026-08-16  
**Status:** Approved design  
**Audience:** Technical learners who can read Python but may be new to AI engineering tooling

## 1. Outcome

A learner must be able to clone the repository, prepare a supported environment,
open the first notebook, and complete Day 1 without guessing which files or commands
are current.

The onboarding must be factual, short, and usable on macOS or Windows. The course
must remain usable without a paid API.

## 2. Supported execution modes

The repository exposes three explicit modes:

| Mode | Purpose | Required services |
|---|---|---|
| Offline | Installation checks and deterministic course verification | None |
| Ollama | Official free local student path | Ollama and the course models |
| OpenAI | Optional hosted comparison path | OpenAI API key and account credit |

Offline and Ollama are sufficient for Day 1. OpenAI is optional. Docker is not a
Day 1 prerequisite.

## 3. Official toolchain

- Git for cloning the repository.
- `uv` as the only documented Python environment and dependency manager.
- Python 3.11 or later, installed or selected through `uv`.
- Jupyter Lab launched through `uv run`.
- Ollama for the official local live path.
- OpenAI through the existing provider gateway when a key is configured.
- Docker documented as optional professional tooling for later services.

The primary setup contract is:

```bash
uv sync --extra ai --extra rag --extra evaluation --extra dev
uv run python scripts/setup_check.py --offline
uv run python scripts/setup_check.py --provider ollama
uv run jupyter lab
```

The Windows guide presents equivalent PowerShell commands.

## 4. Ollama model policy

One tested pair is the course default:

```bash
ollama pull qwen3:8b
ollama pull qwen3-embedding:0.6b
```

- `qwen3:8b` is the chat model used for prompts, structured outputs, LLM-aware
  chunking, answer generation, and later agent lessons.
- `qwen3-embedding:0.6b` is the embedding model used for semantic chunking and
  retrieval.

The documentation may show hardware adaptations, but only the default pair is the
fully tested course profile:

| Profile | Indicative memory | Chat model | Embedding model | Support level |
|---|---:|---|---|---|
| Light | 8 GB | `qwen3:4b` | `qwen3-embedding:0.6b` | Best effort |
| Course default | 16 GB | `qwen3:8b` | `qwen3-embedding:0.6b` | Fully tested |
| Advanced | 32 GB or more | `qwen3:14b` | `qwen3-embedding:4b` | Optional comparison |

Alternative families such as Gemma, Llama, or BGE-M3 belong in a short comparison
exercise, not the critical setup path. A model change must be recorded because it
can change structured-output reliability, retrieval results, latency, and evaluation
scores.

## 5. Environment and secrets

`.env.example` remains the only committed environment file. It must document the
actual variable names consumed by `finai_academy.settings`, including:

```dotenv
FINAI_MODEL_PROVIDER=ollama
FINAI_CHAT_MODEL=qwen3:8b
FINAI_EMBEDDING_PROVIDER=ollama
FINAI_EMBEDDING_MODEL=qwen3-embedding:0.6b
FINAI_OLLAMA_BASE_URL=http://localhost:11434

# Optional hosted provider
OPENAI_API_KEY=
# FINAI_MODEL_PROVIDER=openai
# FINAI_CHAT_MODEL=gpt-5-mini
# FINAI_EMBEDDING_PROVIDER=openai
# FINAI_EMBEDDING_MODEL=text-embedding-3-small

# Used in later news lessons
TAVILY_API_KEY=
```

The guide instructs learners to copy `.env.example` to `.env`. `.env` remains
ignored by Git. Setup output and notebooks must never print secrets.

## 6. Student-facing documentation

### `README.md`

The root landing page contains only the information needed to orient and start:

1. course name and concrete learning outcome;
2. the Financial Analyst Copilot built across the course;
3. prerequisites;
4. a four-command quick start;
5. Day 1 lesson order and duration;
6. links to setup, student guide, troubleshooting, and instructor material.

### `docs/getting-started.md`

The installation guide contains separate macOS and Windows PowerShell sections:

1. install Git;
2. install `uv` using the current official method;
3. clone the repository and enter its directory;
4. install Python and dependencies;
5. run the offline check;
6. install and start Ollama;
7. pull the two default models;
8. create `.env`;
9. run the Ollama check;
10. optionally configure OpenAI and Docker;
11. start Jupyter Lab.

Every step includes one expected result and one recovery link.

### `docs/day-1-student-guide.md`

The guide lists the seven canonical notebooks in order, the objective and expected
artifact of each lesson, the capstone increment, and the final Day 1 checkpoint.
It links directly to the matching slide deck and notebook.

### `docs/troubleshooting.md`

The troubleshooting guide is a compact symptom/cause/action table covering:

- `uv` not found;
- wrong working directory;
- Jupyter kernel mismatch;
- Ollama service unavailable;
- missing chat or embedding model;
- insufficient memory;
- invalid or absent OpenAI key;
- notebook PASS-gate failure;
- Docker unavailable when an optional later service needs it.

### `notebooks/README.md` and `decks/README.md`

These index only canonical course assets and point back to the student guide. They
must not describe completed notebooks as shells or seed material.

## 7. Day 1 course presentation

`decks/00-course-introduction.pptx` remains the single Day 1 introduction deck. It
must align with the repository onboarding and show:

- professional learning objectives;
- the Financial Analyst Copilot capstone architecture;
- the Day 1 schedule and lesson progression;
- offline, Ollama, and OpenAI execution modes;
- the tested Ollama model pair;
- a simple environment-readiness checklist;
- the exact first command and first notebook;
- the footer `First Finance - Arnaud Demes`.

Slides stay visual and concise. Installation details live in the written guide, not
in the deck.

## 8. Repository cleanup

Only the underscore-named Lessons 01-07 are canonical for Day 1. The older
hyphenated seed notebooks and matching legacy chapter files must no longer appear
in the student-facing course paths.

The implementation plan will list every legacy path before any move or deletion.
Because removal is destructive, it requires explicit approval for the exact paths
and command. No unrelated asset is changed.

## 9. Setup diagnostics

`scripts/setup_check.py` becomes the single readiness entrypoint. Its result must be
short and actionable:

```text
Python               PASS
Dependencies         PASS
Ollama service       PASS
Chat model           qwen3:8b — PASS
Embedding model      qwen3-embedding:0.6b — PASS
OpenAI               OPTIONAL
Docker               OPTIONAL
Course readiness     READY
```

Failure output names one corrective command. Optional components never fail the
Day 1 readiness result.

## 10. Verification and acceptance

Implementation is accepted only when:

1. all documentation links resolve;
2. documented commands match repository scripts and dependency extras;
3. a fresh temporary clone passes the offline setup path;
4. the configured Ollama profile passes the live setup check;
5. macOS and Windows instructions use valid shell syntax;
6. `.env` and API keys cannot be committed accidentally;
7. canonical Day 1 notebooks are clearly ordered 01-07;
8. the introduction deck renders without overflow or placeholders;
9. the footer and factual terminology are consistent;
10. repository validation, notebook validation, tests, and Ruff pass.

The OpenAI live path is tested only when a valid key is available. A missing key is
reported as optional, never as a successful live test.

## 11. Non-goals

- No custom installer or heavyweight setup application.
- No mandatory Docker dependency for Day 1.
- No mandatory paid API.
- No broad model benchmark during setup.
- No redesign of Lessons 01-07 content in this cleanup.
- No removal of legacy files without explicit path-level approval.
