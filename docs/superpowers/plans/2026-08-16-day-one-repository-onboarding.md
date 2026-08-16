# Day 1 Repository and Student Onboarding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a fresh clone of FinAI Academy immediately usable for Day 1 on macOS or Windows, with `uv`, a tested Ollama profile, optional OpenAI, safe `.env` handling, concise student guides, and a matching introduction deck.

**Architecture:** Keep one student-facing path: root README → getting-started guide → readiness check → Day 1 guide → canonical notebook. Configuration remains provider-neutral in `Settings`; `scripts/setup_check.py` is the only diagnostic entrypoint. Documentation and deck contracts are tested from repository files, and legacy seed assets are removed only after explicit path-level approval.

**Tech Stack:** Python 3.11+, `uv`, python-dotenv, pytest, Ruff, Jupyter Lab, Ollama, optional OpenAI, optional Docker, PowerPoint via `@oai/artifact-tool`.

## Global Constraints

- The course language is English.
- `uv` is the only documented Python environment and dependency manager.
- Offline and Ollama are sufficient for Day 1; no paid API is mandatory.
- The fully tested Ollama pair is `qwen3:8b` plus `qwen3-embedding:0.6b`.
- OpenAI and Docker are optional and must not make offline or Ollama readiness fail.
- `.env` is ignored; `.env.example` contains names and safe defaults only.
- Notebooks, diagnostics, tests, and documentation never print secrets.
- The presentation footer is exactly `First Finance - Arnaud Demes`.
- Installation copy is factual, short, and separated into macOS and Windows PowerShell paths.
- No legacy path is removed until the exact `git rm` command receives explicit approval.

---

## File map

### Create

- `docs/getting-started.md` — clone-to-Jupyter instructions for macOS and Windows.
- `docs/day-1-student-guide.md` — schedule, lesson order, outcomes, and capstone increments.
- `docs/troubleshooting.md` — symptom/cause/action reference.
- `tests/test_onboarding_docs.py` — links, command contract, asset order, and secret-safety checks.
- `tests/test_intro_deck.py` — introduction-deck content and footer contract.
- `docs/reviews/day-one-delivery-readiness.md` — final evidence and score after verification.

### Modify

- `README.md` — concise landing page and four-command quick start.
- `.env.example` — safe local defaults and optional hosted/news variables.
- `.gitignore` — verified by tests to retain an exact `.env` ignore rule.
- `src/finai_academy/settings.py` — load the repository `.env` without overriding shell variables.
- `scripts/setup_check.py` — single actionable readiness report.
- `tests/test_settings.py` — `.env` loading and precedence contracts.
- `tests/test_setup_check.py` — offline, Ollama, OpenAI, Docker, and readiness contracts.
- `notebooks/README.md` — canonical Lessons 01-07 only.
- `chapters/README.md` — canonical introduction and Lessons 01-07 only.
- `decks/README.md` — canonical deck index and validation note.
- `tests/test_course_manifest.py` — remove the legacy-material wording.
- `decks/00-course-introduction.pptx` — add the environment-readiness bridge to Lesson 01.

### Remove after explicit approval

- Eleven obsolete hyphenated seed notebooks listed in Task 4.
- Eleven matching obsolete chapter briefs listed in Task 4.

---

### Task 1: Make `.env` configuration real and safe

**Files:**
- Modify: `src/finai_academy/settings.py`
- Modify: `tests/test_settings.py`
- Modify: `.env.example`

**Interfaces:**
- Consumes: environment variables already used by `Settings`.
- Produces: `Settings.from_environment(env_file: str | Path | None = None) -> Settings`; explicit shell variables take precedence over file values.

- [ ] **Step 1: Add failing `.env` tests**

Add tests that use `tmp_path` and `monkeypatch`:

```python
def test_settings_loads_an_explicit_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "FINAI_MODEL_PROVIDER=ollama\n"
        "FINAI_CHAT_MODEL=qwen3:4b\n"
        "FINAI_EMBEDDING_MODEL=qwen3-embedding:0.6b\n",
        encoding="utf-8",
    )
    for variable in (
        "FINAI_MODEL_PROVIDER",
        "FINAI_CHAT_MODEL",
        "FINAI_EMBEDDING_PROVIDER",
        "FINAI_EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(variable, raising=False)

    settings = Settings.from_environment(env_file=env_file)

    assert settings.chat_model == "qwen3:4b"
    assert settings.embedding_model == "qwen3-embedding:0.6b"


def test_shell_environment_overrides_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("FINAI_CHAT_MODEL=qwen3:4b\n", encoding="utf-8")
    monkeypatch.setenv("FINAI_CHAT_MODEL", "qwen3:8b")

    settings = Settings.from_environment(env_file=env_file)

    assert settings.chat_model == "qwen3:8b"
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
uv run pytest tests/test_settings.py -q
```

Expected: the new tests fail because `from_environment` does not accept `env_file`.

- [ ] **Step 3: Load dotenv without overriding the shell**

Implement the exact boundary:

```python
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

@classmethod
def from_environment(cls, env_file: str | Path | None = None) -> "Settings":
    load_dotenv(
        dotenv_path=Path(env_file) if env_file is not None else PROJECT_ROOT / ".env",
        override=False,
    )
    provider = getenv("FINAI_MODEL_PROVIDER", "ollama")
    return cls(
        provider=provider,
        chat_model=getenv("FINAI_CHAT_MODEL", ""),
        embedding_provider=getenv("FINAI_EMBEDDING_PROVIDER", provider),
        embedding_model=getenv("FINAI_EMBEDDING_MODEL", ""),
        ollama_base_url=getenv("FINAI_OLLAMA_BASE_URL", "http://localhost:11434"),
    )
```

Keep `.env.example` aligned with these variables. Leave `OPENAI_API_KEY` and
`TAVILY_API_KEY` empty and commented as optional. Do not add example secrets.

- [ ] **Step 4: Run focused verification**

Run:

```bash
uv run pytest tests/test_settings.py -q
uv run ruff check src/finai_academy/settings.py tests/test_settings.py
git diff --check
```

Expected: all commands pass.

- [ ] **Step 5: Commit the configuration boundary**

```bash
git add .env.example src/finai_academy/settings.py tests/test_settings.py
git commit -m "feat: load course settings from env file"
```

---

### Task 2: Turn `setup_check.py` into one readiness command

**Files:**
- Modify: `scripts/setup_check.py`
- Modify: `tests/test_setup_check.py`

**Interfaces:**
- Consumes: `Settings.from_environment()`, Ollama `/api/tags`, `OPENAI_API_KEY`, and the Docker executable.
- Produces: `CheckResult`, `check_ollama(settings) -> list[CheckResult]`, `check_docker() -> CheckResult`, and one final `Course readiness` row.

- [ ] **Step 1: Write the failing diagnostic contracts**

Retain the existing subprocess helper and add these assertions:

```python
def test_offline_setup_check_is_ready_without_external_services():
    result = run_setup_check("--offline")

    assert result.returncode == 0
    assert "PASS Python" in result.stdout
    assert "PASS Dependencies" in result.stdout
    assert "OPTIONAL Ollama" in result.stdout
    assert "OPTIONAL OpenAI" in result.stdout
    assert "OPTIONAL Docker" in result.stdout or "PASS Docker" in result.stdout
    assert "READY Course readiness" in result.stdout


def test_requested_openai_provider_is_not_ready_without_a_key():
    environment = os.environ.copy()
    environment.pop("OPENAI_API_KEY", None)

    result = run_setup_check("--provider", "openai", environment=environment)

    assert result.returncode == 1
    assert "FAIL OpenAI" in result.stdout
    assert "NOT READY Course readiness" in result.stdout


def test_missing_docker_is_optional(monkeypatch):
    monkeypatch.setattr("scripts.setup_check.shutil.which", lambda _: None)

    result = check_docker()

    assert result.status == "OPTIONAL"
    assert result.name == "Docker"


def test_ollama_reports_the_service_and_each_required_model(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps(
                {
                    "models": [
                        {"name": "qwen3:8b"},
                        {"name": "qwen3-embedding:0.6b"},
                    ]
                }
            ).encode()

    monkeypatch.setattr("scripts.setup_check.urlopen", lambda *_args, **_kwargs: Response())

    results = check_ollama(Settings())

    assert [(result.status, result.name) for result in results] == [
        ("PASS", "Ollama service"),
        ("PASS", "Chat model"),
        ("PASS", "Embedding model"),
    ]
```

Import `json`, `check_docker`, `check_ollama`, and `Settings` for these unit tests.

- [ ] **Step 2: Run the focused tests and confirm RED**

```bash
uv run pytest tests/test_setup_check.py -q
```

Expected: failures mention missing `Dependencies`, `Docker`, model rows, and final
readiness output.

- [ ] **Step 3: Implement concise diagnostics**

Use these statuses and rendering contract:

```python
@dataclass(frozen=True)
class CheckResult:
    status: str
    name: str
    detail: str

    def render(self) -> str:
        return f"{self.status} {self.name} — {self.detail}"
```

Implementation rules:

- Python fails below 3.11.
- Dependencies use the exact recovery command
  `uv sync --extra ai --extra rag --extra evaluation --extra dev`.
- Offline mode performs no network request and marks Ollama/OpenAI optional.
- Ollama mode reports service, chat model, and embedding model separately.
- Each missing Ollama model names one `ollama pull <model>` action.
- OpenAI is `OPTIONAL` unless explicitly selected; if selected without a key it is
  `FAIL`.
- Docker uses `shutil.which("docker")`; absence is always `OPTIONAL` for Day 1.
- The last row is `READY Course readiness` when no result is `FAIL`, otherwise
  `NOT READY Course readiness` and exit code 1.
- No credential value enters a `CheckResult`.

- [ ] **Step 4: Run diagnostic verification**

```bash
uv run pytest tests/test_setup_check.py -q
uv run python scripts/setup_check.py --offline
uv run ruff check scripts/setup_check.py tests/test_setup_check.py
git diff --check
```

Expected: tests pass; the CLI ends with `READY Course readiness`.

- [ ] **Step 5: Commit the readiness command**

```bash
git add scripts/setup_check.py tests/test_setup_check.py
git commit -m "feat: add actionable course readiness check"
```

---

### Task 3: Build the clone-to-Day-1 documentation path

**Files:**
- Create: `docs/getting-started.md`
- Create: `docs/day-1-student-guide.md`
- Create: `docs/troubleshooting.md`
- Create: `tests/test_onboarding_docs.py`
- Modify: `README.md`
- Modify: `notebooks/README.md`
- Modify: `chapters/README.md`
- Modify: `decks/README.md`

**Interfaces:**
- Consumes: `course.yml`, canonical files, `.env.example`, and `scripts/setup_check.py`.
- Produces: one linear student journey and local Markdown links that resolve from each file.

- [ ] **Step 1: Write the failing documentation tests**

Create `tests/test_onboarding_docs.py` with these contracts:

```python
ROOT = Path(__file__).resolve().parents[1]
DAY_ONE_NOTEBOOKS = [
    "notebooks/01_model_gateway.ipynb",
    "notebooks/02_prompts_and_structured_outputs.ipynb",
    "notebooks/03_cag_financial_document.ipynb",
    "notebooks/04_rag_from_scratch.ipynb",
    "notebooks/05_document_and_chunking_lab.ipynb",
    "notebooks/06_hybrid_retrieval.ipynb",
    "notebooks/07_rag_evaluation.ipynb",
]


def test_readme_exposes_one_four_command_quick_start():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for command in (
        "uv sync --extra ai --extra rag --extra evaluation --extra dev",
        "uv run python scripts/setup_check.py --offline",
        "uv run python scripts/setup_check.py --provider ollama",
        "uv run jupyter lab",
    ):
        assert command in text


def test_day_one_guide_lists_canonical_notebooks_in_order():
    text = (ROOT / "docs" / "day-1-student-guide.md").read_text(encoding="utf-8")
    positions = [text.index(path) for path in DAY_ONE_NOTEBOOKS]
    assert positions == sorted(positions)


def test_env_is_ignored_and_example_contains_no_key_value():
    ignore_lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert ".env" in ignore_lines
    assert "OPENAI_API_KEY=" in example
    assert not re.search(r"OPENAI_API_KEY=\S+", example)


def test_local_markdown_links_resolve():
    documents = (
        ROOT / "README.md",
        ROOT / "docs" / "getting-started.md",
        ROOT / "docs" / "day-1-student-guide.md",
        ROOT / "docs" / "troubleshooting.md",
        ROOT / "notebooks" / "README.md",
        ROOT / "chapters" / "README.md",
        ROOT / "decks" / "README.md",
    )
    link_pattern = re.compile(r"\[[^]]+\]\(([^)]+)\)")
    missing: list[str] = []

    for document in documents:
        for raw_target in link_pattern.findall(document.read_text(encoding="utf-8")):
            target = raw_target.split("#", maxsplit=1)[0].strip("<>")
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (document.parent / target).resolve().exists():
                missing.append(f"{document.relative_to(ROOT)} -> {raw_target}")

    assert missing == []
```

Import `Path` and `re`. The link test ignores external links and same-page anchors;
every remaining target resolves relative to its Markdown file.

- [ ] **Step 2: Run the documentation tests and confirm RED**

```bash
uv run pytest tests/test_onboarding_docs.py -q
```

Expected: missing guide files and missing README quick-start commands.

- [ ] **Step 3: Write the root README**

Use this order and keep it under roughly 160 lines:

1. `AI Engineering for Asset Management`;
2. one-sentence outcome: build a Financial Analyst Copilot;
3. what learners build by the end of Day 1;
4. prerequisites;
5. the four-command quick start;
6. execution modes table: Offline, Ollama, OpenAI;
7. Day 1 table sourced from `course.yml`;
8. links to setup, student guide, troubleshooting, and instructor architecture;
9. repository map;
10. copyright.

Remove stale claims that notebooks are seeds or shells. Use the footer/attribution
`First Finance - Arnaud Demes` consistently.

- [ ] **Step 4: Write the platform setup guide**

Use official installation sources and exact commands:

```bash
# macOS: uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows PowerShell: uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Project environment on both platforms
git clone https://github.com/Nonocho/finai-academy.git
cd finai-academy
uv python install 3.11
uv sync --extra ai --extra rag --extra evaluation --extra dev
```

Link to:

- `https://docs.astral.sh/uv/getting-started/installation/`
- `https://ollama.com/download`
- `https://platform.openai.com/api-keys`
- `https://docs.docker.com/get-started/get-docker/`

Show `cp .env.example .env` for macOS and
`Copy-Item .env.example .env` for PowerShell. State that OpenAI API usage may require
account credit and is not needed for the course default. Keep Docker in an optional
section. Include one compact model table:

| Profile | Indicative memory | Chat | Embeddings | Support |
|---|---:|---|---|---|
| Light | 8 GB | `qwen3:4b` | `qwen3-embedding:0.6b` | Best effort |
| Course default | 16 GB | `qwen3:8b` | `qwen3-embedding:0.6b` | Fully tested |
| Advanced | 32 GB+ | `qwen3:14b` | `qwen3-embedding:4b` | Optional |

Explain in two sentences that chat models generate and embedding models encode for
retrieval. Keep Gemma, Llama, and BGE-M3 as optional comparison names outside the
critical setup path.

- [ ] **Step 5: Write the Day 1 and troubleshooting guides**

Build the schedule from `course.yml`, including the 09:00 start, breaks, 12:00-13:30
lunch, 16:45 checkpoint, lesson objectives, deck links, notebook links, and capstone
increments. End with this concrete checkpoint:

```text
Day 1 complete = parsed evidence + configurable chunks + hybrid retrieval + cited answer + evaluation trace.
```

Use one symptom/cause/action table for troubleshooting. Every action is one command
or one link.

- [ ] **Step 6: Rewrite the three asset indexes**

- `notebooks/README.md`: list only canonical Lessons 01-07 as completed Day 1 labs.
- `chapters/README.md`: list introduction plus canonical Lessons 01-07.
- `decks/README.md`: distinguish completed Day 1 decks from planned Day 2 decks and
  link to the student guide.

- [ ] **Step 7: Run documentation verification**

```bash
uv run pytest tests/test_onboarding_docs.py tests/test_course_manifest.py -q
uv run python scripts/validate_repo.py
uv run ruff check tests/test_onboarding_docs.py
git diff --check
```

Expected: all commands pass and every local link resolves.

- [ ] **Step 8: Commit the student journey**

```bash
git add README.md docs/getting-started.md docs/day-1-student-guide.md docs/troubleshooting.md notebooks/README.md chapters/README.md decks/README.md tests/test_onboarding_docs.py
git commit -m "docs: add Day 1 student onboarding"
```

---

### Task 4: Remove obsolete seed curriculum from student paths

**Files:**
- Remove: the exact notebook and chapter files below.
- Modify: `tests/test_onboarding_docs.py`
- Modify: `tests/test_course_manifest.py`

**Interfaces:**
- Consumes: the canonical asset indexes created in Task 3.
- Produces: unambiguous student-facing `notebooks/` and `chapters/` directories.

- [ ] **Step 1: Add the failing legacy-absence test**

```python
LEGACY_PATHS = (
    "notebooks/00-product-demo-and-system-map.ipynb",
    "notebooks/01-ai-and-llm-foundations.ipynb",
    "notebooks/02-prompting-and-structured-outputs.ipynb",
    "notebooks/03-retrieval-from-first-principles.ipynb",
    "notebooks/04-document-ingestion-and-chunking.ipynb",
    "notebooks/05-embeddings-and-advanced-retrieval.ipynb",
    "notebooks/06-rag-with-evidence.ipynb",
    "notebooks/07-tools-and-deterministic-workflows.ipynb",
    "notebooks/08-langgraph-agents-and-self-correction.ipynb",
    "notebooks/09-multi-agent-financial-research.ipynb",
    "notebooks/10-evaluation-observability-and-llmops.ipynb",
    "chapters/00-product-demo-and-system-map.md",
    "chapters/01-ai-and-llm-foundations.md",
    "chapters/02-prompting-and-structured-outputs.md",
    "chapters/03-retrieval-from-first-principles.md",
    "chapters/04-document-ingestion-and-chunking.md",
    "chapters/05-embeddings-and-advanced-retrieval.md",
    "chapters/06-rag-with-evidence.md",
    "chapters/07-tools-and-deterministic-workflows.md",
    "chapters/08-langgraph-agents-and-self-correction.md",
    "chapters/09-multi-agent-financial-research.md",
    "chapters/10-evaluation-observability-and-llmops.md",
)


def test_legacy_seed_assets_are_absent_from_student_paths():
    assert [path for path in LEGACY_PATHS if (ROOT / path).exists()] == []
```

Rename `test_repository_validator_accepts_manifest_paths_alongside_legacy_material`
to `test_repository_validator_accepts_canonical_manifest_paths`.

- [ ] **Step 2: Run the focused test and confirm RED**

```bash
uv run pytest tests/test_onboarding_docs.py::test_legacy_seed_assets_are_absent_from_student_paths -q
```

Expected: the assertion reports all 22 legacy paths.

- [ ] **Step 3: Request explicit approval for the exact removal**

Present the following exact command and wait for approval before executing it:

```bash
git rm notebooks/00-product-demo-and-system-map.ipynb notebooks/01-ai-and-llm-foundations.ipynb notebooks/02-prompting-and-structured-outputs.ipynb notebooks/03-retrieval-from-first-principles.ipynb notebooks/04-document-ingestion-and-chunking.ipynb notebooks/05-embeddings-and-advanced-retrieval.ipynb notebooks/06-rag-with-evidence.ipynb notebooks/07-tools-and-deterministic-workflows.ipynb notebooks/08-langgraph-agents-and-self-correction.ipynb notebooks/09-multi-agent-financial-research.ipynb notebooks/10-evaluation-observability-and-llmops.ipynb chapters/00-product-demo-and-system-map.md chapters/01-ai-and-llm-foundations.md chapters/02-prompting-and-structured-outputs.md chapters/03-retrieval-from-first-principles.md chapters/04-document-ingestion-and-chunking.md chapters/05-embeddings-and-advanced-retrieval.md chapters/06-rag-with-evidence.md chapters/07-tools-and-deterministic-workflows.md chapters/08-langgraph-agents-and-self-correction.md chapters/09-multi-agent-financial-research.md chapters/10-evaluation-observability-and-llmops.md
```

Git history is the recovery path. Do not remove any other file.

- [ ] **Step 4: Execute only the approved `git rm` command**

Confirm `git status --short` shows exactly the 22 deletions plus the two test edits.

- [ ] **Step 5: Run cleanup verification**

```bash
uv run pytest tests/test_onboarding_docs.py tests/test_course_manifest.py -q
uv run python scripts/validate_repo.py
git diff --check
```

Expected: all commands pass.

- [ ] **Step 6: Commit the controlled cleanup**

```bash
git add tests/test_onboarding_docs.py tests/test_course_manifest.py
git commit -m "chore: remove obsolete course seed assets"
```

---

### Task 5: Align the Day 1 introduction deck with onboarding

**Files:**
- Create: `tests/test_intro_deck.py`
- Modify: `decks/00-course-introduction.pptx`

**Interfaces:**
- Consumes: existing 12-slide introduction deck, `course.yml`, default model settings, and the quick-start contract.
- Produces: a rendered, editable introduction deck with one clear environment-readiness bridge.

- [ ] **Step 1: Write the failing deck-content contract**

Create the XML helper and assertions:

```python
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
INTRO_DECK = ROOT / "decks" / "00-course-introduction.pptx"


def intro_slide_texts() -> list[str]:
    with ZipFile(INTRO_DECK) as package:
        slide_parts = sorted(
            name
            for name in package.namelist()
            if name.startswith("ppt/slides/slide") and name.endswith(".xml")
        )
        return [
            " ".join(
                node.text or ""
                for node in ElementTree.fromstring(package.read(part)).iter()
                if node.tag.endswith("}t")
            )
            for part in slide_parts
        ]


def intro_deck_text() -> str:
    return "\n".join(intro_slide_texts())


def test_intro_deck_contains_the_student_start_contract():
    text = intro_deck_text()

    for expected in (
        "AI Engineering for Asset Management",
        "Financial Analyst Copilot",
        "qwen3:8b",
        "qwen3-embedding:0.6b",
        "uv sync",
        "scripts/setup_check.py --provider ollama",
        "01_model_gateway.ipynb",
    ):
        assert expected in text


def test_every_intro_slide_has_the_course_footer():
    slide_texts = intro_slide_texts()
    assert slide_texts
    assert all("First Finance - Arnaud Demes" in text for text in slide_texts)
```

- [ ] **Step 2: Run the deck test and confirm RED**

```bash
uv run pytest tests/test_intro_deck.py -q
```

Expected: the current deck lacks the embedding model, setup command, and first
notebook path.

- [ ] **Step 3: Load the presentation workflow and inspect the source deck**

Use the `presentations:Presentations` skill. Load bundled workspace dependencies,
create a unique temporary build directory, inspect all 12 slides and layouts, and
render the current deck before editing. Do not recreate the visual system.

- [ ] **Step 4: Add one inherited environment-readiness slide**

Use `@oai/artifact-tool` and shapes/layouts inherited from the current deck. Place
the new slide after the provider slide and before the acceptance contract. Its
content is limited to:

```text
ENVIRONMENT READY
1  uv sync
2  ollama pull qwen3:8b
3  ollama pull qwen3-embedding:0.6b
4  scripts/setup_check.py --provider ollama
START  notebooks/01_model_gateway.ipynb
```

Add concise speaker notes that point to `docs/getting-started.md` for macOS,
Windows, OpenAI, and Docker details. Keep the footer unchanged.

- [ ] **Step 5: Render and inspect every slide**

Run the presentation skill's official render and overflow tools. Inspect every
rendered slide at full size, with particular attention to the new command strings.
Require:

- no overflow, collision, or clipped text;
- no placeholder text;
- no missing footer;
- zero template-fidelity issues;
- editable text and shapes;
- one `[Sources]` block in the speaker notes of every slide.

- [ ] **Step 6: Run deck and repository verification**

```bash
uv run pytest tests/test_intro_deck.py tests/test_course_manifest.py -q
uv run python scripts/validate_repo.py
git diff --check
```

Expected: all commands pass and all presentation QA checks report zero issues.

- [ ] **Step 7: Commit the introduction deck**

```bash
git add decks/00-course-introduction.pptx tests/test_intro_deck.py
git commit -m "slides: add Day 1 environment readiness"
```

---

### Task 6: Verify the complete delivery from a fresh clone and grade it

**Files:**
- Create: `docs/reviews/day-one-delivery-readiness.md`
- Modify only if evidence exposes a defect: files owned by Tasks 1-5.

**Interfaces:**
- Consumes: committed onboarding, diagnostics, canonical assets, and intro deck.
- Produces: reproducible readiness evidence and a final score with no unsupported claims.

- [ ] **Step 1: Run the complete repository gates**

```bash
uv run ruff check .
uv run pytest -q
uv run python scripts/validate_repo.py
uv run python scripts/validate_notebooks.py notebooks/01_model_gateway.ipynb notebooks/02_prompts_and_structured_outputs.ipynb notebooks/03_cag_financial_document.ipynb notebooks/04_rag_from_scratch.ipynb notebooks/05_document_and_chunking_lab.ipynb notebooks/06_hybrid_retrieval.ipynb notebooks/07_rag_evaluation.ipynb
git diff --check
```

Expected: all gates pass.

- [ ] **Step 2: Verify the official local execution modes**

```bash
uv run python scripts/setup_check.py --offline
uv run python scripts/setup_check.py --provider ollama
uv run python scripts/execute_notebooks.py --mode live --provider ollama notebooks/01_model_gateway.ipynb notebooks/02_prompts_and_structured_outputs.ipynb notebooks/03_cag_financial_document.ipynb notebooks/04_rag_from_scratch.ipynb notebooks/05_document_and_chunking_lab.ipynb notebooks/06_hybrid_retrieval.ipynb notebooks/07_rag_evaluation.ipynb
```

Expected: readiness is `READY`; all seven notebooks finish with their required PASS
markers. Run the OpenAI path only when `OPENAI_API_KEY` exists; otherwise record
`not configured`, not `passed`.

- [ ] **Step 3: Smoke-test from a fresh temporary clone**

Create a unique directory with:

```bash
mktemp -d /private/tmp/finai-onboarding.XXXXXX
```

Use the returned explicit path for these commands:

```bash
git clone --local /Users/arnauddemes/dev/AIxFinance/finai-academy /private/tmp/finai-onboarding.RETURNED/finai-academy
uv sync --frozen --extra ai --extra rag --extra evaluation --extra dev
uv run python scripts/setup_check.py --offline
uv run python -c "import jupyterlab; print('Jupyter ready')"
```

Expected: dependency sync uses `uv.lock`, offline readiness is `READY`, and Jupyter
imports. Leave the temporary clone intact until the review is written; removal needs
separate approval.

- [ ] **Step 4: Write the evidence-based readiness review**

Record:

- commit hash;
- operating system and Python version;
- documentation-link result;
- fresh-clone result;
- offline and Ollama diagnostic result;
- seven-notebook result;
- OpenAI status stated honestly;
- deck render/overflow/template/source-note results;
- test and Ruff counts;
- a score for clarity, installation, technical reliability, visual quality, and
  overall delivery readiness.

Do not award 9.5/10 or higher unless every mandatory acceptance criterion passes.

- [ ] **Step 5: Run final verification after the review**

```bash
uv run pytest -q
uv run ruff check .
uv run python scripts/validate_repo.py
git diff --check
git status --short
```

Expected: only `docs/reviews/day-one-delivery-readiness.md` is uncommitted.

- [ ] **Step 6: Commit the readiness evidence**

```bash
git add docs/reviews/day-one-delivery-readiness.md
git commit -m "docs: certify Day 1 delivery readiness"
```

- [ ] **Step 7: Confirm the handoff state**

```bash
git status --short
git log -6 --oneline
```

Expected: clean working tree and six focused implementation commits after the design
and plan commits.
