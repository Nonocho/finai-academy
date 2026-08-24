# Task 2 report — certified capstone tools

## Status

Completed.

## Delivered scope

- Added `capstone.tools` with a fixture-backed hybrid certified retriever, the immutable two-tool allowlist, runtime-discovery intersection, and typed tool outcomes.
- Added `capstone.live_news` with an isolated Tavily boundary. It reports missing configuration as unavailable, sanitizes successful items to the permitted public fields, and keeps runtime failures non-blocking for certified analysis.
- Added contract coverage for company isolation, evidence provenance, typed validation failures, fail-closed tool invocation, validated successful calls, and Tavily unavailable/success/error/sanitization paths.

## TDD evidence

The initial `tests/test_capstone_tools.py` run failed at collection because the required adapters did not exist. The added missing-key injection and credential-shaped provider-content tests each failed against the prior implementation, then passed after the minimal corresponding implementation changes.

## Verification

Ran successfully:

```text
.venv/bin/python -m pytest tests/test_capstone_tools.py tests/test_hybrid_retrieval.py tests/test_financial_mcp_capabilities.py -q
58 passed

.venv/bin/ruff check src/finai_academy/capstone/tools.py src/finai_academy/capstone/live_news.py tests/test_capstone_tools.py
All checks passed
```

`git diff --check` also completed without whitespace errors.

## Self-review

- Certified retrieval validates against the existing typed financial capability registry before hybrid ranking and applies canonical exact company metadata filters to both ranking channels.
- Public evidence preserves company, text, evidence ID, document ID, section, period, and source reference.
- Tool errors expose only the typed capability fields; unknown and undiscovered calls never echo arguments.
- News output retains no raw provider response or API key. Credential-shaped external strings are converted to the stable non-blocking error state.

## Concerns

None.
