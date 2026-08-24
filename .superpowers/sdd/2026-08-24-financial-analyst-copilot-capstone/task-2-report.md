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

## Review round 1 — public-boundary hardening

### Root cause and TDD evidence

`AnalystToolRegistry.invoke()` passed arbitrary mappings directly to the financial capability registry and returned `DocumentSearchResult` unchanged. Its public `query` field therefore echoed credential-shaped and personal-path values. The same method interpolated arbitrary tool names into `ValueError` and allowed malformed values to reach downstream string and numeric operations.

Added focused reproductions before changing production code for:

- credential-shaped and `/Users/...` document queries;
- a malicious tool name containing a bearer token;
- document arguments with `query=123`, `company=None`, and `top_k="2"`;
- malformed metric ticker and metric values.

The RED command was:

```text
.venv/bin/python -m pytest tests/test_capstone_tools.py -q
8 failed, 14 passed
```

The failures showed the leaked tool-name exception, successful document outcomes containing unsafe query text, and raw `AttributeError`/`TypeError` stack traces from `financial_mcp_capabilities.py`.

Implemented a complete per-tool schema gate before capability invocation. It permits only the approved field sets and types, applies the Task 1 public-value safety check to all forwarded strings, preserves existing typed capability validation for safe values, and returns this stable public failure for malformed or unsafe input:

```text
status="error"
error_code="invalid_arguments"
message="Tool arguments must match the approved schema."
retryable=True
```

Unknown and undiscovered errors now use fixed messages and never interpolate caller-controlled names or arguments.

### Review-round verification

Focused GREEN verification:

```text
.venv/bin/python -m pytest tests/test_capstone_tools.py -q
22 passed

.venv/bin/ruff check src/finai_academy/capstone/tools.py tests/test_capstone_tools.py
All checks passed
```

Required regression verification:

```text
.venv/bin/python -m pytest tests/test_capstone_tools.py tests/test_hybrid_retrieval.py tests/test_financial_mcp_capabilities.py -q
66 passed

.venv/bin/ruff check src/finai_academy/capstone/tools.py src/finai_academy/capstone/live_news.py tests/test_capstone_tools.py
All checks passed

git diff --check
exit 0
```
