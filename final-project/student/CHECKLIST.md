# Capstone integration checklist

## 15:30–15:40: understand mission

- [ ] Keep the certified recorded route as the offline classroom baseline.
- [ ] Read [../STUDENT_BRIEF.md](../STUDENT_BRIEF.md) and identify the four seams.
- [ ] Run `uv sync --extra capstone --extra ai` from the repository root.
- [ ] Launch `uv run streamlit run final-project/student/streamlit_app.py` in Terminal 2.
- [ ] The Streamlit server in Terminal 2 stays running; run `uv run python final-project/student/verify.py` in Terminal 3.
- [ ] If Terminal 2 must be reused, stop its Streamlit server with `Ctrl+C` first.
- [ ] Keep changes inside the four `integration.py` function bodies.

## 15:40–16:10: complete four seams

- [ ] Wire the certified, company-filtered retriever.
- [ ] Register only discovered, approved analyst read capabilities.
- [ ] Require NVIDIA and Schneider Electric document evidence at the gate.
- [ ] Assemble the existing display-safe public briefing view.
- [ ] Re-run `uv run python final-project/student/verify.py` after each seam.

## 16:10–16:25: evaluate and diagnose

- [ ] Run `uv run python final-project/student/diagnose.py run` and record the public run and trace IDs.
- [ ] Run `uv run python final-project/student/diagnose.py inspect` and assign the final failure owner.
- [ ] Correct only `diagnostic_case.json` by setting `drop_company` to JSON `null`.
- [ ] Rerun the diagnostic and confirm `DIAGNOSTIC_STATUS=completed` and `RELEASE=passed`.
- [ ] Rerun the verifier and confirm citation integrity, five metrics, and local persistence pass.
- [ ] Confirm stdout contains one standalone `CAPSTONE_PASS` and stderr is empty.
- [ ] Confirm no API key, network connection, Tavily, or Ollama is required.

## 16:25–16:30: prepare demo

- [ ] Remove debug prints and copied provider details.
- [ ] Check that no personal path, credential, trading action, or investment advice was added.
- [ ] If in a pair, decide who explains each seam.

## 16:30–17:00: demonstration and architecture review

- [ ] Show the public verifier result and explain one seam.
- [ ] Explain why the evidence gate and public view boundary are required.

First Finance - Arnaud Demes
