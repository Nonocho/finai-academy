# Capstone integration checklist

## 15:30–15:40: understand mission

- [ ] Read [../STUDENT_BRIEF.md](../STUDENT_BRIEF.md) and identify the four seams.
- [ ] Run `uv sync --extra capstone --extra ai` from the repository root.
- [ ] Launch `uv run streamlit run final-project/student/streamlit_app.py`.
- [ ] Run `uv run python final-project/student/verify.py` and read each diagnostic.
- [ ] Keep changes inside the four `integration.py` function bodies.

## 15:40–16:10: complete four seams

- [ ] Wire the certified, company-filtered retriever.
- [ ] Register only discovered, approved analyst read capabilities.
- [ ] Require NVIDIA and Schneider Electric document evidence at the gate.
- [ ] Assemble the existing display-safe public briefing view.
- [ ] Re-run `uv run python final-project/student/verify.py` after each seam.

## 16:10–16:25: evaluate and diagnose

- [ ] Confirm the recorded reference mission passes offline.
- [ ] Confirm citation integrity and five deterministic metrics pass.
- [ ] Confirm local persistence passes without exposing a path.
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
