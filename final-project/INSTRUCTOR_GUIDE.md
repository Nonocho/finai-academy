# Financial Analyst Copilot instructor guide

## Prerequisites and preflight

Use the repository root. Before learners arrive, confirm the working tree contains the starter files and run:

```bash
uv sync --extra capstone --extra ai
uv run streamlit run final-project/reference/streamlit_app.py
uv run streamlit run final-project/student/streamlit_app.py
uv run python final-project/student/verify.py
```

Keep the reference sidebar on **Recorded demo** and **Certified snapshots**. This is the certified offline fallback, so it does not need network access, OpenAI, Ollama, or Tavily. Do not claim a live provider or timed rehearsal unless it has been observed. On macOS, copy `.env.example` with `cp .env.example .env`; on Windows PowerShell, use `Copy-Item .env.example .env`. OpenAI requires a learner-owned `OPENAI_API_KEY`; Ollama uses `FINAI_MODEL_PROVIDER=ollama` and `qwen3:4b` when available. `TAVILY_API_KEY` is optional and only supports optional live enrichment.

## Expected reference output

The fixed mission compares NVIDIA and Schneider Electric with certified document evidence and selected financial metrics. **Run reference mission** shows a validated plan, tool activity, one typed recovery and replan where applicable, an evidence gate, five briefing sections, citations, an execution trace, and five deterministic release metrics. The expected release decision is **Release passed**. The optional judge is shown separately and is not a release requirement.

## Facilitation schedule

| Time | Facilitation |
| --- | --- |
| 15:30–15:40 | State the mission, open the recorded reference route, and identify the four seams. |
| 15:40–16:10 | Learners complete seams. Check that they use the verifier after each seam. |
| 16:10–16:25 | Review diagnostics, public pass rule, evidence coverage, and one deliberate regression. |
| 16:25–16:30 | Learners prepare a compact demo and choose one seam to explain. |
| 16:30–17:00 | Demonstration and architecture review. Compare implementation decisions and production boundaries. |

## Pair rotation

For pairs, ask the driver to switch after two seams or after 15 minutes. Both learners must explain one seam and review the other seams together. Individual and pair requirements are otherwise identical.

## Progressive hints

Give one hint only after the learner has read the named verifier diagnostic.

| Seam | Progressive hint |
| --- | --- |
| `wire_retriever` | Find the certified retriever constructor and call its company-scoped search boundary. |
| `register_analyst_capabilities` | Use the existing analyst tool policy rather than naming tools in the student function. |
| `evaluate_student_evidence_gate` | Build coverage for both fixed companies, preserve the incoming hits, and report missing document evidence in mission order. |
| `assemble_public_briefing_view` | Search for the existing transformation that removes domain internals before UI display. |

## Correction

Reveal this only after the challenge or when the skip-if-late route is selected. These are the exact correction bodies from `reference/student_integration_solution.py`.

```python
def wire_retriever(company: str, query: str) -> tuple[CapstoneEvidenceHit, ...]:
    """Return certified document evidence for one company-scoped query."""

    return build_certified_retriever().search(company, query)


def register_analyst_capabilities(discovered: Sequence[str]) -> tuple[str, ...]:
    """Apply runtime discovery through the certified analyst tool policy."""

    return AnalystToolRegistry(discovered=discovered).discover()


def evaluate_student_evidence_gate(
    hits: Sequence[CapstoneEvidenceHit],
) -> EvidenceGateDecision:
    """Require document evidence for both companies in the fixed mission."""

    companies = ("NVIDIA", "Schneider Electric")
    covered = {hit.company for hit in hits}
    missing = tuple(
        f"{company} document evidence" for company in companies if company not in covered
    )
    return EvidenceGateDecision(
        passed=not missing,
        coverage={
            company: (("document",) if company in covered else ())
            for company in companies
        },
        missing_requirements=missing,
        evidence_hits=tuple(hits),
    )


def assemble_public_briefing_view(result: ResearchRunResult) -> CapstoneRunView:
    """Convert a domain result through the display-safe public boundary."""

    return to_run_view(result)
```

## Diagnostic regression and MLflow trace

The verifier includes a deliberately regressed retrieval case that returns no Schneider Electric evidence. It should produce an evidence-gate block, no briefing, and missing requirement `Schneider Electric document evidence`. The expected failure owner in the public trace is `evidence_gate`, not a provider. Use this to discuss why release is withheld even when part of the route completed.

For the recorded completed route, persistence records public evidence in local MLflow when the evaluation dependency is installed. Inspect the returned run and trace IDs from the persistence result, then use the local MLflow UI only if it is already configured for the session. Inspect trace inputs, outputs, release decision, trajectory, evidence identities, and briefing artifact. Do not expose local file paths or credentials during the demonstration. If MLflow is unavailable, the analysis output remains valid and the verifier reports the typed persistence state.

## Common failures, recorded fallback, and recovery

| Symptom | Response |
| --- | --- |
| A seam reports incomplete. | Read its verifier diagnostic, give the matching progressive hint, and keep edits inside `integration.py`. |
| A verifier contract does not complete. | Check return type, company order, and whether the existing certified component was used. |
| `CAPSTONE_PASS` is missing. | Confirm no learner function prints output and rerun the verifier from the repository root. |
| OpenAI, Ollama, or Tavily is unavailable. | Select Recorded demo and Certified snapshots. Do not troubleshoot a live route during the challenge. |
| Windows command differs from macOS. | Use Windows recovery: PowerShell `Copy-Item .env.example .env`; continue with the same `uv` commands. |
| macOS local setup is incomplete. | Use macOS recovery: `cp .env.example .env`, then run `uv sync --extra capstone --extra ai`. |

## Skip-if-late route

At 16:10, if learners have not completed two seams, keep the student app and verifier as the learning surface. Walk through one seam from the correction, let learners implement the remaining seams, then run the public verifier. Use the reference app for the 16:30 demonstration. Do not replace or overwrite learner files.

## Reset procedure

The UI **Reset session** action clears only the current reference application session state. To reset a learner exercise, first ask the learner to preserve their current `integration.py` as a named copy in their own working area. Then provide a fresh starter copy in a separate directory or repository clone. This procedure does not delete or overwrite learner work.

## Demo prompts and architecture discussion

Ask each learner or pair to show the fixed mission, explain one seam, run the public verifier, and identify the evidence gate result. Discuss:

- Why does the public view boundary exclude domain internals and private paths?
- Why does the evidence gate require document evidence for both companies?
- Why is a typed replan preferable to silently substituting a provider or source?
- Which release metric detects an unsupported claim or citation mismatch?

The production non-goals for this exercise are live trading, investment advice, authentication, deployment, persistent learner sessions, arbitrary code execution, and building retrieval, MCP, or MLflow infrastructure from scratch.

First Finance - Arnaud Demes
