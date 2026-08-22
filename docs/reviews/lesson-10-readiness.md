# Lesson 10 delivery readiness

Review date: 2026-08-22
Lesson: Financial MCP
Environment: macOS, Apple Silicon, Python 3.13.9, MCP Python SDK 2.0.0

## Decision

Lesson 10 is ready for an instructor-led test class with the deterministic
offline path and the local Ollama path. It teaches a real local `stdio` MCP
boundary: one application-controlled resource, two read-only tools, and one
user-controlled prompt. The host retains discovery, allowlisting, validation,
provenance, and the final answer.

The score is deliberately below 10/10: `OPENAI_API_KEY` was not configured, so
the OpenAI live path was not run, and a timed learner rehearsal was not part of
this certification. Neither limitation is represented as a provider pass.

## Verification evidence

| Gate | Result | Evidence |
| --- | --- | --- |
| Targeted Lesson 10 package | PASS | Approved local-port run: `37 passed in 5.17s`. It covers capability registry, in-memory MCP server discovery/error handling, real `stdio` client subprocess/allowlisting, Lesson 10 assets, and course manifest. |
| Source notebook contract | PASS | `.venv/bin/python scripts/validate_notebooks.py notebooks/10_financial_mcp.ipynb`: `1 notebook passed the course notebook contract.` The checked-in notebook remains output-free. |
| Code quality | PASS | `.venv/bin/ruff check .`: `All checks passed!` |
| Offline execution | PASS | Exact offline command completed in 3 seconds. Executed artifact: 26 cells, 12 executed code cells, 5 PNG figures, exact marker `LESSON_10_PASS`. |
| Real `stdio` lifecycle | PASS | Fresh subprocess run discovered `finance://coverage`; `get_company_metric` and `search_financial_documents`; and `compare_companies`. Its 14 ordered events open/close the transport and client, include resource/tool/prompt discovery and calls, and record only the intentional `unsupported_metric` error. |
| In-memory MCP contract | PASS | Targeted package includes server tests for the one resource, two tool schemas, one prompt, structured results, and typed invalid-metric error. |
| Ollama availability | PASS | `ollama` command and local service were available with `qwen3:8b` installed. No model was installed or pulled during review. |
| Ollama live execution | PASS | Exact live command produced an artifact with 5 PNG figures and `LESSON_10_PASS`; runner wall time was 32.5 seconds. Runtime label identifies `ollama` / `qwen3:8b`; it selected the runtime-discovered, allowlisted `search_financial_documents` tool and preserved the expected invalid-metric protocol error. |
| OpenAI live execution | NOT CONFIGURED | `OPENAI_API_KEY` was absent. The OpenAI command was not run and no OpenAI pass is claimed. |
| Full course regression | PASS | Approved interactive local-port run: `.venv/bin/pytest -q` reported `252 passed in 50.26s`. |
| Prior readiness-commit post-gate | PASS | After commit `1b11503`, `git status --porcelain` was empty, the Lesson 10 sequence appeared in `git log -7 --oneline`, and the Task 1-4 targeted gate passed `26` tests in `5.04s`. This evidence applies to that prior commit only. |
| Whitespace | PASS | `git diff --check` exited with no output. |
| Deck overflow | PASS | Bundled-runtime `slides_test.py decks/10-financial-mcp.pptx`: `Test passed. No overflow detected.` |
| Deck structure and notes | PASS | 9 slide XML parts and 9 notes XML parts were found; every notes part contains a complete `[Sources]` and `[/Sources]` block. The Lesson 10 deck contract passed: `1 passed, 3 deselected in 0.70s`. |
| Deck visual and template review | PASS | All nine final slide PNGs were inspected individually. The template-plan validator and template-fidelity check both reported `status: pass` and `issueCount: 0`. The Lesson 09 and Lesson 10 `ppt/theme/theme1.xml` SHA-256 values are identical: `8b500abccb3a86061340d95e2edfe2ca62da665f2741801d8790930dba1507a0`. |
| Instructor timing | PASS | Chapter tables total 10-minute deck + 30-minute notebook + 5-minute verification/debrief = 45 minutes, 11:15-12:00. The explicit five-minutes-late route keeps discovery, coverage, a sourced metric, the prompt, and the trust-boundary debrief while skipping extended walkthrough/live work. |

## Commands and observed results

```bash
.venv/bin/pytest -q tests/test_financial_mcp_capabilities.py tests/test_financial_mcp_server.py tests/test_financial_mcp_client.py tests/test_lesson10_assets.py tests/test_course_manifest.py
# scoped local-port rerun: 37 passed in 5.17s

.venv/bin/ruff check .
# All checks passed!

.venv/bin/python scripts/validate_notebooks.py notebooks/10_financial_mcp.ipynb
# 1 notebook passed the course notebook contract.

.venv/bin/python scripts/execute_notebooks.py notebooks/10_financial_mcp.ipynb --mode offline --output-dir /private/tmp/finai-lesson10-offline-task7
# PASS .../10_financial_mcp.ipynb (3 seconds)

FINAI_LIVE_MODE=1 FINAI_MODEL_PROVIDER=ollama FINAI_CHAT_MODEL=qwen3:8b .venv/bin/python scripts/execute_notebooks.py notebooks/10_financial_mcp.ipynb --mode live --provider ollama --output-dir /private/tmp/finai-lesson10-ollama
# executed artifact verified; runner wall time 32.5 seconds

.venv/bin/pytest -q
# 252 passed in 50.26s
```

The first unapproved targeted run reached `36 passed` before its one notebook
execution test failed with `PermissionError: [Errno 1] Operation not permitted`
while Jupyter attempted to bind a local kernel port. The approved rerun passed
all 37 tests. This is an execution-sandbox restriction, not a product failure.
The full suite was likewise run with local-port permission. Its first
non-interactive capture stopped at 57% without a pytest summary, so the final
certification result uses the subsequently completed interactive run above.

Deck QA used the bundled runtime environment:

```bash
RUNTIME_NODE=/Users/arnauddemes/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node \
RUNTIME_NODE_MODULES=/Users/arnauddemes/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules \
RUNTIME_BIN_DIR=/Users/arnauddemes/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin \
/Users/arnauddemes/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
/Users/arnauddemes/.codex/plugins/cache/openai-primary-runtime/presentations/26.819.11345/skills/presentations/container_tools/slides_test.py decks/10-financial-mcp.pptx
# Test passed. No overflow detected.
```

The same runtime validated `.artifacts/lesson10-deck/template-frame-map.json`
against the Lesson 09 source and checked final fidelity using the starter deck,
starter layouts, final layouts, and edit directory. Both checks reported zero
issues.

## Specification score

| Dimension | Weight | Score | Evidence |
| --- | ---: | ---: | --- |
| Conceptual clarity and progression | 20% | 9.8/10 | The deck and chapter move from coupling to discovery, then to host-owned permission and the Lesson 11 handoff. |
| Notebook usability and visuals | 20% | 9.7/10 | The 30-minute notebook is deterministic offline, exposes a real lifecycle, produces five figures, and finishes with an exact marker. |
| Technical correctness and safety | 20% | 9.9/10 | Fresh tests cover capability validation, in-memory discovery/errors, a real subprocess, runtime allowlisting, provenance, and the typed failure. |
| Provider neutrality and recovery | 15% | 9.2/10 | Offline and the exact Ollama path passed; OpenAI follows the shared gateway but is unverified because it was not configured. |
| Deck quality and fidelity | 15% | 9.8/10 | Nine slides and source-note blocks passed manual review, overflow, contract, plan, fidelity, and theme checks. |
| Timing and instructor readiness | 10% | 9.0/10 | The complete 45-minute route and late route are explicit; a real timed learner rehearsal remains outstanding. |

Weighted readiness score: **9.63/10**.

## Instructor start path

1. Present [the Lesson 10 deck](../../decks/10-financial-mcp.pptx).
2. Open [the Lesson 10 notebook](../../notebooks/10_financial_mcp.ipynb).
3. Use [the instructor chapter](../../chapters/10-financial-mcp.md) for timing, recovery, and the late route.
4. Keep the rule visible: discovery describes an offer; host policy permits the call.
5. Use the offline route if live infrastructure is unavailable; do not substitute direct server-function imports.
