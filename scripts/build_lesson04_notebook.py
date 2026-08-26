"""Build Lesson 04 as a compact real-document naive RAG laboratory."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "04_rag_from_scratch.ipynb"


def markdown(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(dedent(source).strip())


def code(source: str, *, source_hidden: bool = False) -> nbformat.NotebookNode:
    metadata = {"jupyter": {"source_hidden": True}} if source_hidden else {}
    return nbformat.v4.new_code_cell(dedent(source).strip(), metadata=metadata)


cells = [
    markdown(
        """
        # 04 — How RAG Chooses Evidence

        **First Finance - Arnaud Demes**  
        **Question:** Why retrieve instead of sending an entire financial filing?

        ## Learning objectives

        By the end, you can explain why retrieval exists, trace one answer through
        the pipeline, and diagnose its earliest failed stage:

        ```text
        Parsing → chunking → retrieval → generation
        ```

        ## Where this fits

        This transparent baseline connects Lesson 03's source decision to the
        parsing and retrieval upgrades in Lessons 05–06.

        ## Why RAG?

        One answer needs a few passages—not an entire filing. RAG selects that evidence
        **before** the model call, improving focus and traceability only when retrieval works.
        We deliberately build a flawed baseline over NVIDIA's complete FY2026 Form 10-K.
        """
    ),
    code(
        """
        # Instructor setup — run once. The lesson begins with the visual below.
        from __future__ import annotations

        import os
        import time
        from pathlib import Path
        from textwrap import shorten

        from IPython.display import Markdown, display

        from finai_academy import lesson04_visuals as visuals
        from finai_academy.documents import load_source_manifest
        from finai_academy.lesson_support import RecordedRagModel
        from finai_academy.naive_rag import naive_fixed_windows, naive_parse_html
        from finai_academy.providers import create_chat_model
        from finai_academy.retrieval import LexicalRetriever, build_rag_prompt, evaluate_retrieval
        from finai_academy.settings import Settings

        ROOT = Path.cwd().parent if (Path.cwd().parent / "pyproject.toml").exists() else Path.cwd()
        MANIFEST = ROOT / "assets/course-data/manifest.json"
        OFFICIAL = ROOT / "assets/course-data/downloads/nvidia_fy2026_form_10k.html"

        source = next(item for item in load_source_manifest(MANIFEST) if item.company == "NVIDIA")
        assert source.official_path and source.verify_official(ROOT)
        parsed = naive_parse_html(OFFICIAL)

        print("Official filing verified: PASS")
        print(f"{source.company} · {source.period} {source.document_type} · accession {source.accession_number}")
        print(f"{source.official_bytes:,} downloaded bytes → {len(parsed.text):,} flattened characters")
        """,
        source_hidden=True,
    ),
    code(
        """
        # Visual 1 — one question should not inherit the entire filing.
        filing_tokens = max(1, len(parsed.text) // 4)
        answer_budget = 250

        visuals.plot_evidence_scale(filing_tokens, answer_budget)
        """
    ),
    markdown(
        """
        ## 1. Start with deliberately naive HTML parsing

        The baseline removes markup and collapses the entire filing into one whitespace-
        normalized string. This is authentic source text, but it is **not** a trustworthy
        document model: headings, tables, row/column relationships and hierarchy disappear.

        The filing contains tables but no semantic `h1`–`h6` headings. A production parser
        must infer structure from the filing's HTML/XBRL rather than assuming a normal web page.
        """
    ),
    code(
        """
        # Visual 2 — flattening erases document boundaries.
        visuals.plot_flattening(
            parsed.text,
            table_count=parsed.table_count,
            semantic_heading_count=parsed.semantic_heading_count,
        )
        """
    ),
    code(
        """
        # Visual 3 — naive character windows ignore financial boundaries.
        CHUNK_CHARS = 1_600
        OVERLAP_CHARS = 200
        chunks = naive_fixed_windows(
            parsed.text,
            source,
            chunk_chars=CHUNK_CHARS,
            overlap_chars=OVERLAP_CHARS,
        )

        visuals.plot_fixed_windows(
            len(chunks),
            chunk_chars=CHUNK_CHARS,
            overlap_chars=OVERLAP_CHARS,
        )

        print(f"Windows: {len(chunks)} · size: {CHUNK_CHARS} · overlap: {OVERLAP_CHARS}")
        print("Boundary example:")
        print("…" + chunks[151].text[-170:] + " | " + chunks[152].text[:170] + "…")
        """
    ),
    markdown(
        """
        ## 2. Retrieval decides what the model can see

        ```text
        258 real-document windows → TF-IDF → cosine ranking → top-k=2 → prompt → model
        ```

        **Maintained question:** What drove NVIDIA's fiscal 2026 revenue growth?

        TF-IDF rewards lexical overlap. Cosine similarity ranks every window. `top_k=2`
        is an application decision: everything below the line becomes invisible to the model.
        """
    ),
    code(
        """
        QUESTION = "What drove NVIDIA revenue growth in fiscal 2026?"
        TOP_K = 2
        retriever = LexicalRetriever(chunks)
        ranking = retriever.rank(QUESTION)
        hits = retriever.search(QUESTION, top_k=TOP_K)

        visuals.plot_ranking_boundary(ranking, TOP_K)

        print("Selected evidence:", [hit.passage.passage_id for hit in hits])
        for hit in hits:
            print(f"\\n[{hit.passage.passage_id}] {shorten(hit.passage.text, width=310)}")
        """
    ),
    code(
        """
        display(Markdown("## 3. This is everything the model can see"))

        rag_prompt = build_rag_prompt(QUESTION, hits)
        selected_tokens = sum(len(hit.passage.text) // 4 for hit in hits)
        prompt_tokens = len(rag_prompt) // 4

        # Visual 5 — retrieval creates a small, inspectable prompt.
        visuals.plot_prompt_boundary(filing_tokens, selected_tokens, prompt_tokens)

        settings = Settings.from_environment()
        live_mode = os.getenv("FINAI_LIVE_MODE", "1") == "1"
        model = create_chat_model(settings) if live_mode else RecordedRagModel()

        started = time.perf_counter()
        response = model.invoke([("human", rag_prompt)])
        latency_ms = (time.perf_counter() - started) * 1_000
        answer = response.content

        print(f"Provider: {settings.provider} · model: {settings.chat_model}")
        print(f"Generation latency: {latency_ms:,.0f} ms")
        display(Markdown("### Retrieved-evidence answer\\n\\n" + answer))

        expected_ids = {"NVDA-C152", "NVDA-C160"}
        retrieval_check = evaluate_retrieval(hits, expected_ids)
        grounding_checks = {
            "names accelerated computing and AI": "accelerated computing" in answer.casefold() and "ai" in answer.casefold(),
            "identifies Data Center": "data center" in answer.casefold(),
            "cites retrieved evidence": any(f"[{hit.passage.passage_id}]" in answer for hit in hits),
            "states an evidence boundary": "valuation" in answer.casefold() or "evidence" in answer.casefold(),
        }

        print(f"Retrieval check: {'PASS' if retrieval_check.passed else 'FAIL'}")
        print("Grounding check:", "PASS" if all(grounding_checks.values()) else "OBSERVATION")
        for label, passed in grounding_checks.items():
            print(f"  {'✓' if passed else '○'} {label}")

        assert retrieval_check.passed
        if not live_mode:
            assert all(grounding_checks.values())
        else:
            print("Live answer grounding remains an observation; keep any miss visible.")
        """
    ),
    markdown(
        """
        ## Failure lab

        ### The relevant table exists—but top-k never sees it

        Ask a more precise question requiring the actual Data Center revenue table. The
        flattened filing contains `$193,737 million`, yet lexical repetition elsewhere can
        outrank the table window. If that window stays below top-k, generation cannot repair
        the omission.

        Diagnose the earliest broken stage: **parsing, chunking, retrieval or generation?**

        ## Verification

        The final checks keep source integrity, retrieval coverage, failure reproduction
        and provenance separate so a fluent answer cannot hide a broken retrieval step.
        """
    ),
    code(
        """
        FAILURE_QUESTION = "How large was Data Center revenue compared with total revenue in fiscal 2026?"
        failure_ranking = retriever.rank(FAILURE_QUESTION)
        precise_table_rank = next(
            rank
            for rank, hit in enumerate(failure_ranking, start=1)
            if "193,737" in hit.passage.text
        )
        failure_hits = failure_ranking[:TOP_K]

        # Visual 6 — the table is real, but outside the selection boundary.
        visuals.plot_failure_boundary(
            failure_ranking,
            precise_table_rank=precise_table_rank,
            top_k=TOP_K,
        )

        print("Failure top-k:", [hit.passage.passage_id for hit in failure_hits])
        print(f"Required table window rank: #{precise_table_rank}")
        print(f"Failure diagnosis: RETRIEVAL — the required table never enters top-k={TOP_K}.")

        verification = {
            "official source verified": source.verify_official(ROOT),
            "real filing produced many windows": len(chunks) > 200,
            "maintained growth evidence retrieved": retrieval_check.passed,
            "precise table miss reproduced": precise_table_rank > TOP_K,
            "provenance survives prompt construction": source.source_url in rag_prompt,
        }
        for label, passed in verification.items():
            print(f"{'PASS' if passed else 'FAIL'} — {label}")
        assert all(verification.values())
        print("PASS — real-document naive RAG boundary verified")
        """
    ),
    markdown(
        """
        ## Recap

        - **Why RAG:** select a small, traceable evidence set before generation.
        - **Why inspect retrieval:** the model cannot use evidence it never receives.
        - **Why this baseline is naive:** HTML structure was flattened and character windows
          ignored financial boundaries.
        - **Lesson 05:** preserve headings, tables, metadata and meaningful chunk boundaries.
        - **Lesson 06:** add embeddings, filters, hybrid retrieval and reranking.

        ## Challenge

        Change `TOP_K` to 5. Does evidence coverage improve? What extra noise
        enters the prompt—and does the precise table still remain invisible?

        ## Capstone integration

        Reuse the passage IDs, source URL and retrieval checks as the evidence contract
        for the final analyst workflow. Later lessons can replace a stage without changing
        how the answer proves which evidence it received.
        """
    ),
]

notebook = nbformat.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
        "finai": {"expected_runtime_minutes": 4},
    },
)
nbformat.write(notebook, OUTPUT)
print(f"Wrote {OUTPUT.relative_to(ROOT)} with {len(cells)} cells")
