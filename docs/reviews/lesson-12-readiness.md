# Lesson 12 readiness: technical certification

Status: **PASS - READY FOR INSTRUCTOR-LED OFFLINE TEST CLASS**

This report records Task 8 technical certification evidence from the exact base commit
`bde9434ce6d6fd6a5c11a64f6fa49f20b15af379`. The first independent review found two
Important issues despite a numerically passing 9.565/10 score. Fix round 1 addressed
both, and the final independent re-review resolved them, assigned 9.815/10, and recorded
the release decision above.

## Scope

Certification covered the approved Lesson 12 specification, the six-case dataset and
two recorded configurations, pure scorers, local SQLite-backed MLflow runs and traces,
the real Lesson 11 offline reference route, notebook source and fresh execution, all six
notebook figures and visible evidence tables, the nine-slide deck and its notes, optional
provider routes where locally available, the complete targeted package, and the full
repository regression.

The tracked worktree was clean at the start. Initial certification created only this
draft and the ignored SDD implementer report. Fix round 1 then changed the optional
judge attachment guard, focused tests, the chapter summary, and knowledge-check Q5.
The notebook, deck, manifest, ledger, remote, and ignored Task 4 backup were not changed.
No push, deletion, or move occurred during certification. Final authorization permits
this certificate and the four tracked fix files to be committed after the gates below.

## Environment

Certification ran on 23 August 2026 in Europe/Paris.

| Item | Observed value |
| --- | --- |
| Base commit | `bde9434ce6d6fd6a5c11a64f6fa49f20b15af379` |
| Branch | `course-build/two-day-class` |
| Host | macOS 26.6, build `25G72`, Darwin 25.6.0, arm64 |
| Python | 3.13.9 |
| pytest | 9.1.1 |
| Ruff | 0.16.3 |
| MLflow | 3.15.1 |
| nbformat / nbclient / jupyter-client | 5.11.0 / 0.11.0 / 8.9.1 |
| matplotlib / pandas / Pydantic | 3.11.1 / 2.3.3 / 2.13.4 |

The initial sandboxed targeted run could not bind local Jupyter kernel ports and
reported two `PermissionError: [Errno 1] Operation not permitted` failures before the
Lesson 11 and Lesson 12 notebooks executed. The unchanged command was rerun with local
loopback permission and passed all 174 tests. During fix round 1, the affected Task
3/4/12 package passed 101 tests with that same local-only permission. Invoking pytest
through `uv run` also exposed MLflow's absolute executable path as system trace metadata;
the exact repository command `.venv/bin/pytest` avoided that launcher-only difference.
Neither environment symptom required a production change.

## Versioned data and alignment

| Artifact | SHA-256 |
| --- | --- |
| `agent_cases_v1.json` | `c8f81fc59b182df8b2044c70d759fcb1fdac1fa90faead4bb70812b409ba0131` |
| `agent_runs_v1.json` | `a58d3652f84c6b2abdd5d392fd0d2d74c3931835347723c2048a9a0372958469` |
| source notebook | `b63869538a9007a1e5bec4d9a7acd647e96453fa6d3b77ba64ebdafd25185e88` |
| executed notebook | `c21a74eb4ca88e51c378950ebd0981e8259c548b2249c36217bd78d819f369b9` |
| offline SQLite database | `3417e6941691905d8a0d70a8300105eef8bdffaae74cdf5229ddaf27962d85c6` |
| fix-round executed notebook | `522fa28cd488d3e87afd12b5c5c64dbc6bd523f1299aa4bae7d6578569b60284` |
| fix-round offline SQLite database | `e50ae87473e0303236c6d2d0990a4f789cae6c1b88adcc4a46336f5539b8ad8c` |

The fresh notebook resolved `agent-cases-v1`, the exact manifest hash above, six cases,
and both `bounded-agent-v1` and `regressed-agent-v0`. The real Lesson 11 MCP route ran
once and reported `Reference public signature: MATCH`, final status `completed`, five
attempts, one `unsupported_metric` typed error, one replan, execution revisions
`[0, 0, 0, 1, 1]`, the expected evidence IDs, cited facts, provenance kinds, source and
evidence-ID pairs, aggregate sources, and limitations.

The exact six cases were:

1. `reference_completed`
2. `unsupported_metric_not_recovered`
3. `redundant_metric_call`
4. `missing_schneider_document`
5. `document_fact_without_evidence_id`
6. `wrong_source_evidence_pair`

Both configurations used all six cases and the same dataset version and hash.

## Pure scorer results

| Configuration | Metric | Mean | Pass count |
| --- | --- | ---: | ---: |
| `bounded-agent-v1` | `tool_call_correctness` | 1.000000 | 6/6 |
| `bounded-agent-v1` | `tool_call_efficiency` | 1.000000 | 6/6 |
| `bounded-agent-v1` | `answer_relevance` | 1.000000 | 6/6 |
| `bounded-agent-v1` | `answer_completeness` | 0.777778 | 4/6 |
| `bounded-agent-v1` | `citation_integrity` | 1.000000 | 6/6 |
| `regressed-agent-v0` | `tool_call_correctness` | 1.000000 | 6/6 |
| `regressed-agent-v0` | `tool_call_efficiency` | 0.933333 | 5/6 |
| `regressed-agent-v0` | `answer_relevance` | 0.750000 | 4/6 |
| `regressed-agent-v0` | `answer_completeness` | 0.888889 | 2/6 |
| `regressed-agent-v0` | `citation_integrity` | 0.666667 | 4/6 |

The complete six-by-two per-case matrix was:

| Configuration / case | Correct | Efficient | Relevant | Complete | Citation | Release | Earliest public failure owner |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| bounded / `reference_completed` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | pass | none |
| bounded / `unsupported_metric_not_recovered` | 1.00 | 1.00 | 1.00 | 0.333333 | 1.00 | pass | evidence gate |
| bounded / `redundant_metric_call` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | pass | none |
| bounded / `missing_schneider_document` | 1.00 | 1.00 | 1.00 | 0.333333 | 1.00 | pass | evidence gate |
| bounded / `document_fact_without_evidence_id` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | pass | none |
| bounded / `wrong_source_evidence_pair` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | pass | none |
| regressed / `reference_completed` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | pass | none |
| regressed / `unsupported_metric_not_recovered` | 1.00 | 1.00 | 0.00 | 0.583333 | 1.00 | fail | replanner |
| regressed / `redundant_metric_call` | 1.00 | 0.60 | 1.00 | 1.00 | 1.00 | pass | replanner |
| regressed / `missing_schneider_document` | 1.00 | 1.00 | 0.50 | 0.916667 | 1.00 | fail | evidence gate |
| regressed / `document_fact_without_evidence_id` | 1.00 | 1.00 | 1.00 | 0.916667 | 0.00 | fail | report writer |
| regressed / `wrong_source_evidence_pair` | 1.00 | 1.00 | 1.00 | 0.916667 | 0.00 | fail | report writer |

The bounded configuration recorded 27 total calls, zero redundant calls, 53.000000 ms
mean fixture latency, and 60.0 ms maximum fixture latency. The regressed configuration
recorded 28 calls, one redundant call, 55.833333 ms mean fixture latency, and 70.0 ms
maximum fixture latency.

## Local MLflow runs and traces

Fresh offline store:

- database: `/private/tmp/finai-lesson12-certification-mlflow-uZoJ2f/mlflow.db`
- artifacts: `/private/tmp/finai-lesson12-certification-mlflow-uZoJ2f/artifacts`
- expected UI: `http://127.0.0.1:5000`
- exact UI command: `mlflow ui --backend-store-uri sqlite:////private/tmp/finai-lesson12-certification-mlflow-uZoJ2f/mlflow.db`

Both runs finished and persisted the required parameters, metrics, and three evaluation
artifacts. The store contained exactly 12 `OK` root traces and 167 spans.

| Configuration | Run ID | Case | Trace ID |
| --- | --- | --- | --- |
| bounded | `1066a1f620044f6c922c483ba1073723` | `reference_completed` | `tr-218ebf4d37013d603bdfc10449432a3b` |
| bounded | `1066a1f620044f6c922c483ba1073723` | `unsupported_metric_not_recovered` | `tr-05024e94ebb777b91a4582bcc03f020c` |
| bounded | `1066a1f620044f6c922c483ba1073723` | `redundant_metric_call` | `tr-ca9a7a7d5b1891ba7ead71b3ea733378` |
| bounded | `1066a1f620044f6c922c483ba1073723` | `missing_schneider_document` | `tr-d7181c1d1805e9d04d8f358705cbfd84` |
| bounded | `1066a1f620044f6c922c483ba1073723` | `document_fact_without_evidence_id` | `tr-8a4ba40f068000d6ee625182d6764997` |
| bounded | `1066a1f620044f6c922c483ba1073723` | `wrong_source_evidence_pair` | `tr-32dc5cce1c789cef7431b89e0be0ab17` |
| regressed | `d828ea1b232b4fe9987f429517e0ca0d` | `reference_completed` | `tr-1d14fec34801033444f8a8aa335b892f` |
| regressed | `d828ea1b232b4fe9987f429517e0ca0d` | `unsupported_metric_not_recovered` | `tr-6dc3c796c7873ee34a1809ed7ac5ebd7` |
| regressed | `d828ea1b232b4fe9987f429517e0ca0d` | `redundant_metric_call` | `tr-0593dd547666395438a05e17fba94c54` |
| regressed | `d828ea1b232b4fe9987f429517e0ca0d` | `missing_schneider_document` | `tr-4fbe2dd56a602287e7d9339d69fea916` |
| regressed | `d828ea1b232b4fe9987f429517e0ca0d` | `document_fact_without_evidence_id` | `tr-25bbdb1305254ee18c13051938edd76a` |
| regressed | `d828ea1b232b4fe9987f429517e0ca0d` | `wrong_source_evidence_pair` | `tr-f62f271c389edc884ea6de8e3bca97c4` |

The persisted failed-root drill selected
`bounded-agent-v1/unsupported_metric_not_recovered`, associated it with run
`1066a1f620044f6c922c483ba1073723`, trace
`tr-05024e94ebb777b91a4582bcc03f020c`, and root span `cb2189d16f64feea`. Its exact child
order was `planning -> plan_gate -> execution:1 -> replanning -> execution:2 ->
replanning -> execution:3 -> evidence_gate -> report`; attempt 3 retained
`unsupported_metric`, the blocked guardrail, and `evidence_gate` ownership.

Fix round 1 repeated the canonical offline notebook from a fresh kernel into
`/private/tmp/finai-lesson12-fix-round1-offline-mlflow-20260823`. It produced two
additional `FINISHED` runs, `2e97182322f34e21a709743631685608` and
`595ba87ea92240369b7aaaf1c277c27f`, with six associated traces each, 12 `OK` traces in
total, zero notebook error outputs, six PNG outputs, and exactly one
`LESSON_12_PASS`. This rerun did not select or invoke an optional judge.

## Offline notebook execution

Commanded output:

`/private/tmp/finai-lesson12-certification-output-HeVfxl/12_evaluating_agentic_systems.ipynb`

Fix-round commanded output:

`/private/tmp/finai-lesson12-fix-round1-offline-output-20260823/12_evaluating_agentic_systems.ipynb`

The fresh execution passed from the canonical output-free source. It contained exactly
27 stable cells, two distinct run IDs, six cases per configuration, 12 score rows, 12
distinct trace IDs, all five metric columns and ten aggregate metric rows, six PNG
outputs, and exactly one output occurrence of `LESSON_12_PASS`.

Visible evidence inspected included the complete expectation row, public plan, 14-row
trajectory table, typed error row, evidence-gate result, six-row cited briefing with
source/evidence-ID pairs, limitations and aggregate sources, expected-versus-observed
call table, individual scorer rationales, ten-row aggregate scorecard, twelve-row case
score table, tool-call and latency tables, seven failure rows, two teaching diagnosis
rows, the ten-row persisted failed root and child span table, judge-status table, local
paths, UI URL, UI command, and final marker.

## Notebook visual review

All six extracted PNGs were inspected individually at full size from
`/private/tmp/finai-lesson12-certification-render-HeVfxl/12_evaluating_agentic_systems_files/`.
The same six figure hashes were reproduced and re-inspected from
`/private/tmp/finai-lesson12-fix-round1-offline-visuals-20260823/12_evaluating_agentic_systems_files/`.

| Figure | Dimensions | SHA-256 | Finding |
| --- | --- | --- | --- |
| versioned expectations | 1290 x 550 | `61da971bf635ec1d70a0d479de472b74369d90a430b6fa3c2469c8af3be1327b` | clean three-part architecture and release statement |
| expected vs observed calls | 1390 x 610 | `3f43a5c71b0e8db91a9cbe7189c72d19650b06624aa0773d3d4e670078bbe957` | readable five-call sequence, typed error, and dependency arcs |
| public trace timeline | 1381 x 690 | `4846d53b488604583d9c9a7e5c85dcc636109a5ed7ee0aef3547460a29984f2b` | all 14 events, attempts, revisions, statuses, and latency labels visible |
| per-case heatmap | 1390 x 890 | `f50c501f97fcce7f5703a61c205cb4eb287803b158f6581dba4cb5ca8ea9f93f` | all 12 rows and five metrics legible with values and scale |
| aligned mean comparison | 1390 x 650 | `6d1dc339670722d014cd46d39547cdc119183c2e9d17025e18bbbf08ce6eb969` | both configurations and all five direct value labels visible |
| failure ownership | 1390 x 740 | `a97a2a870b43dfcfa2bb0b0c8bab66a923ad4dd71d6ad748b965d75ea4c146c5` | seven owners, symptoms, arrows, and recovery actions readable |

No clipping, overlap, unexpected wrapping, weak contrast, misleading color semantics,
or unreadable annotation was found.

## Deck automated and visual review

The committed deck SHA-256 is
`1c15c02e48b36d9f539a9ee6b054566a01389cf041b98648b242da633ff5c164`.
The Lesson 11 and Lesson 12 `ppt/theme/theme1.xml` bytes have the same SHA-256,
`8b500abccb3a86061340d95e2edfe2ca62da665f2741801d8790930dba1507a0`.

The original certification render was repeated at 1600 x 900 under
`/private/tmp/finai-lesson12-fix-round1-deck-20260823/rendered/`. The fresh montage and
all nine slides were inspected individually at full size. Slide 2 retained warning-orange
failure semantics; slide 4 visibly separated `dataset_version: agent-cases-v1`,
`case_id: reference_completed`, and `max_tool_calls: 5`; slide 6 showed the seven
distinct stages and rightward post-error recovery; slides 3, 7, and 8 kept readable
native tables; every slide retained the exact footer and coherent Lesson 10/11 visual
system. No clipping, collision, overflow, reversed arrow, unexpected wrapping, weak
contrast, dense unreadable copy, footer drift, or template residue was found.

Fresh automated evidence:

- `slides_test.py`: `Test passed. No overflow detected.`
- template-plan validator: `status: pass`, source slides 9, issues 0.
- template-fidelity validator against fresh final layout inspection: `status: pass`,
  issues 0.
- deck-focused tests: 6 passed, 6 deselected.
- PowerPoint ZIP integrity: no compressed-data errors.
- package contract: nine slides and nine notes blocks, three native tables, nine exact
  footers, directly relevant source blocks, no visible em dash, no empty structural
  placeholders.

## Ollama judge coverage

The local Ollama CLI and daemon were available. `ollama list` showed installed model
`qwen3:8b` with ID `500a1f067a9f`. A fresh configured notebook run passed and persisted
two more six-case runs, but its classroom cell intentionally continued to display the
core-route `NOT RUN` table; configuration alone was not counted as provider evidence.

The first certification attached observed results to run
`6a535c440a97473a8e489f4080ee763a`, whose immutable judge parameters were `none` /
`none`. The independent review correctly rejected that run as contradictory. It remains
unchanged and is not used as fix-round provider evidence.

Fix round 1 created a new bounded evaluation run through
`AgentEvaluationConfiguration(judge_provider="ollama", judge_model="qwen3:8b")` before
MLflow logged any parameters or traces. The repository API then ran the four judges over
that same run's six traces with course URI `ollama_chat:/qwen3:8b`, normalized at runtime
to the native same-provider `ollama:/qwen3:8b` adapter.

- tracking directory:
  `/private/tmp/finai-lesson12-ollama-fix-round1-20260823-1950`
- run ID: `d3287b77fc1b40568c221edf6af568b4`; status: `FINISHED`
- experiment ID: `1`; configuration: `bounded-agent-v1`
- dataset: `agent-cases-v1` /
  `c8f81fc59b182df8b2044c70d759fcb1fdac1fa90faead4bb70812b409ba0131`
- immutable judge parameters: `judge_provider=ollama`, `judge_model=qwen3:8b`
- other immutable parameters: `provider=recorded`,
  `agent_model=recorded-public-fixture-v1`, `agent_version=lesson11-certified-v1`,
  `prompt_version=lesson11-recorded-policies-v1`, `max_steps=6`, `max_replans=1`,
  `scorer_contract_version=agent-scorers-v1`

All six traces were `OK` and stored `mlflow.sourceRun` equal to the new run ID:

| Case | Trace ID |
| --- | --- |
| `reference_completed` | `tr-5856cee3a2e63c4d13a000f9545dab08` |
| `unsupported_metric_not_recovered` | `tr-c1cd838f139e7f8e8b16b980faa5fb80` |
| `redundant_metric_call` | `tr-2c6be685d32965ccaf71d4efb5d3a096` |
| `missing_schneider_document` | `tr-eeeb16e14d216fbddee331f1f1839ceb` |
| `document_fact_without_evidence_id` | `tr-9114554b5cec9d58b53d2c99de2547fa` |
| `wrong_source_evidence_pair` | `tr-00aa2b1e6ddbfe749e52cb728786f91b` |

All four scorers returned observed results:

| Scorer | Provider / model | MLflow | Status | Latency ms | Score | Rationale summary |
| --- | --- | --- | --- | ---: | ---: | --- |
| `ToolCallCorrectness` | Ollama / `qwen3:8b` | 3.15.1 | `COMPLETED` | 104637.19599999604 | 0.3333333333333333 | six per-trace rationales disagreed on whether the expected unsupported Revenue attempt and growth queries were reasonable |
| `ToolCallEfficiency` | Ollama / `qwen3:8b` | 3.15.1 | `COMPLETED` | 74678.25745799928 | 0.8333333333333334 | five traces judged the distinct metric/search calls efficient; one called the failed Revenue request redundant |
| `RelevanceToQuery` | Ollama / `qwen3:8b` | 3.15.1 | `COMPLETED` | 55263.30708299065 | 0.6666666666666666 | four completed briefings were relevant; two stopped/incomplete cases lacked the requested briefing |
| `Completeness` | Ollama / `qwen3:8b` | 3.15.1 | `COMPLETED` | 65483.13037501066 | 0.6666666666666666 | four briefings met all explicit requests; two cases with a null briefing were incomplete |

The complete exact sanitized rationales, including all six per-trace observations for
each scorer, are persisted in
`/private/tmp/finai-lesson12-ollama-fix-round1-20260823-1950/artifacts/d3287b77fc1b40568c221edf6af568b4/artifacts/evaluation/judge_results.json`.
Its SHA-256 is
`074fdc20fb733f5b1440a23df09561e454f83345a3f96c22573ca53a538bc119`.

Before and after judge execution, the deterministic metrics were exactly
`tool_call_correctness_mean=1.0`, `tool_call_efficiency_mean=1.0`,
`answer_relevance_mean=1.0`, `answer_completeness_mean=0.7777777777777778`,
`citation_integrity_mean=1.0`, `mean_tool_calls=4.5`, and `mean_latency_ms=53.0`.
All six case-level `release_passed` values remained `true`; the unchanged
`evaluation/case_scores.json` hash was
`06f13a3a3601300c931bf5f4898269591070a9cea998420584da4e7f7d8dc994` before and after.
Only the four separately prefixed `judge_*` metrics were added. The complete temporary
SQLite database hash after judge completion was
`dfcdb5b3b988a24d10e8fb409a665ae3c579ced105ff80c0faed895fc5303aed`.

## OpenAI judge coverage

`OPENAI_API_KEY` was tested only for presence and was not printed. It was not configured.
Status: **NOT CONFIGURED / NOT RUN**. No OpenAI score, latency, rationale, or provider
credit is claimed.

## Timed rehearsal coverage

No timed learner or instructor rehearsal was observed during Task 8.
Status: **NOT PERFORMED / NOT RUN**. No rehearsal credit is claimed.

## Full repository regression

Fresh results:

| Gate | Result |
| --- | --- |
| original complete targeted pytest package | 174 passed in 79.26 s after unchanged loopback-enabled rerun |
| fix-round affected Task 3/4/12 package | 101 passed in 79.93 s |
| focused optional-judge tests | 18 passed, 20 deselected |
| chapter and deck contract tests | 8 passed, 5 deselected |
| targeted and full Ruff | `All checks passed!` |
| Lesson 12 source notebook validator | 1 notebook passed; fresh execution passed |
| full pytest after fix round | 432 passed in 137.14 s |
| full Ruff | `All checks passed!` |
| all source notebooks | 12 notebooks passed the course notebook contract |
| repository validator | `FinAI Academy repository structure is valid.` |
| deck overflow / template plan / fidelity | pass / pass with 0 issues / pass with 0 issues |
| deck-focused tests / ZIP integrity | 6 passed, 7 deselected / no compressed-data errors |
| `git diff --check` before final report update | exit 0, no output |
| status before final report update | four tracked fix files modified; readiness draft untracked |

The plan's literal no-argument command `.venv/bin/python scripts/validate_notebooks.py`
exited 2 because the current CLI requires one or more paths. The corrected complete
invocation `.venv/bin/python scripts/validate_notebooks.py notebooks/*.ipynb` passed all
12 notebooks. This command-contract mismatch required no repository change.

## Independent findings and resolutions

The first independent whole-lesson review found two Important issues. Both are resolved,
and the final fix-round review found no unresolved Critical or Important finding.

1. **Resolved - immutable judge provenance.** The production path now requires the
   MLflow run's immutable `judge_provider` and `judge_model` parameters to match the
   requested `JudgeConfiguration` before any judge scorer, metric, or artifact write.
   Mismatch coverage proves the write is rejected without judge side effects, matching
   coverage proves the intended route, and the provider-free `none` / `none` route
   remains deterministic. Replacement live run
   `d3287b77fc1b40568c221edf6af568b4` is `FINISHED` with
   `judge_provider=ollama`, `judge_model=qwen3:8b`, six associated `OK` traces, four
   `COMPLETED` judge rows and metrics, and judge-artifact SHA-256
   `074fdc20fb733f5b1440a23df09561e454f83345a3f96c22573ca53a538bc119`.
   Its low semantic scores are truthful observational evidence, not a defect. Its
   deterministic metrics and six-case release result are unchanged; its case-score
   artifact is byte-identical to the deterministic bounded run.
2. **Resolved - judge-status taxonomy.** The chapter summary, Q5 answer, implementation,
   and protected chapter contract now consistently teach: unavailable or missing
   configuration/provider/adapter/client/service => `NOT RUN`; an observed semantic
   judgment, including a low or disagreeing score => `COMPLETED`; timeout or runtime
   invocation failure => `ERROR`. Every status is observational and never alters
   deterministic metrics or the deterministic release decision.

Fresh independent regression checks passed: 18 focused judge tests, 8 focused
chapter/deck tests, the complete 432-test suite, Ruff, all 12 notebook validators, and the
repository validator. The fix-round executed notebook and offline MLflow evidence remain
self-consistent and cover all six acceptance cases and five required metrics.

## Lesson-quality rubric and weighted score

| Dimension | Weight | Independent score | Weighted contribution |
| --- | ---: | ---: | ---: |
| Technical correctness and safety | 25% | 9.80 | 2.450 |
| Learner usability and pacing | 20% | 9.80 | 1.960 |
| Conceptual progression | 20% | 9.80 | 1.960 |
| Offline reliability and diagnosability | 15% | 9.90 | 1.485 |
| Notebook and deck visual quality | 10% | 9.80 | 0.980 |
| Repository and test quality | 10% | 9.80 | 0.980 |
| **Weighted lesson-quality score** | **100%** |  | **9.815/10** |

The numerical threshold is met, all six acceptance cases and five required deterministic
metrics are covered, certification evidence is self-consistent, and no Critical or
Important finding remains unresolved. Provider availability and timed rehearsal are
separate zero-weight qualifications.

## Known qualifications

- The final independent fix-round review was completed on base
  `bde9434ce6d6fd6a5c11a64f6fa49f20b15af379` and the current uncommitted fix diff;
  weighted offline lesson quality is **9.815/10**.
- The final release decision is **PASS / READY FOR INSTRUCTOR-LED OFFLINE TEST CLASS**;
  no Critical or Important finding remains unresolved.
- Live Ollama / `qwen3:8b` evidence is provider-certified on replacement run
  `d3287b77fc1b40568c221edf6af568b4`. Its four completed scorer results, including low
  or disagreeing scores, are truthful observational evidence and do not affect the
  deterministic release decision.
- OpenAI was not configured and was not run.
- No timed rehearsal was observed.
- Optional Ollama judge results are observational and disagree with deterministic
  handling of the expected typed error; they do not change the release gate.
- The configured notebook route displays provider configuration but does not itself call
  `run_optional_judges`; the live provider evidence came from the repository API over the
  freshly persisted traces and is labelled separately.
- The app-level workspace dependency loader was not exposed in this agent context. Deck
  QA used the previously certified bundled runtime paths and verified those executables
  and modules by successfully rendering and running every presentation check.
- Jupyter notebook tests require local loopback permission in this execution environment.
- The ignored Task 4 backup was not modified; its SHA-256 remains
  `08c887e426e114d3133114259934965eda8c39c0d58bd11c3443c8fb9aa8c52f`.

## Decision

**PASS - READY FOR INSTRUCTOR-LED OFFLINE TEST CLASS.** The independently recomputed
weighted offline lesson-quality score is **9.815/10**. All six acceptance cases and all
five required deterministic metrics are covered, the certification evidence is truthful
and self-consistent, and zero Critical or Important findings remain unresolved. The
deterministic offline notebook, deck, code, data, and tests are certified for the stated
offline lesson. Provider availability and timed rehearsal remain separate zero-weight
qualifications: replacement Ollama evidence is truthfully reported as observational,
OpenAI was not configured or run, and no timed rehearsal was performed.
