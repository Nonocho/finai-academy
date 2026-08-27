"""Small, rebuildable Financial Document Analyst for the classroom.

The app deliberately keeps the learning path visible:
load documents -> make chunks -> retrieve evidence -> ask the model.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = {
    "NVIDIA": (
        ROOT / "assets/course-data/downloads/nvidia_fy2026_annual_report.pdf",
        "https://s201.q4cdn.com/141608511/files/doc_financials/2026/ar/2026-Annual-Report-Web.pdf",
    ),
    "Schneider Electric": (
        ROOT / "assets/course-data/downloads/schneider_fy2025_full_year_results.pdf",
        "https://www.se.com/ww/en/assets/564/document/528237/release-fy-results-2025.pdf",
    ),
}

_STOPWORDS = {
    "a",
    "and",
    "between",
    "did",
    "do",
    "for",
    "how",
    "the",
    "their",
    "what",
    "which",
}

@dataclass(frozen=True)
class Chunk:
    """The only object students need to understand for retrieval."""

    company: str
    page: int
    text: str
    source_url: str = ""
    kind: str = "text"


def _split_text(text: str, limit: int = 1_200) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for word in words:
        if current and size + len(word) + 1 > limit:
            chunks.append(" ".join(current))
            current, size = [], 0
        current.append(word)
        size += len(word) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def _table_text(rows: list[list[object]]) -> str:
    cleaned = [[str(cell or "").strip() for cell in row] for row in rows]
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return ""
    return "\n".join(" | ".join(row) for row in cleaned)


def load_document_chunks() -> tuple[Chunk, ...]:
    """Parse the two tracked reports into page- and table-aware chunks."""

    chunks: list[Chunk] = []
    for company, (path, source_url) in DOCUMENTS.items():
        with pdfplumber.open(path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                for text in _split_text(page_text):
                    chunks.append(Chunk(company, page_number, text, source_url))
                for table in page.extract_tables() or []:
                    text = _table_text(table)
                    if text:
                        chunks.append(Chunk(company, page_number, text, source_url, "table"))
    return tuple(chunks)


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold())) - _STOPWORDS


def _query_words(question: str, company: str) -> set[str]:
    terms = _words(question)
    if company == "NVIDIA":
        terms.update({"data", "center", "segment", "revenue", "gaming"})
    elif company == "Schneider Electric":
        terms.update({"organic", "growth", "energy", "management", "revenue"})
    return terms


def retrieve_chunks(question: str, chunks: tuple[Chunk, ...], top_k: int = 5) -> tuple[Chunk, ...]:
    """Use transparent word overlap so learners can see why a chunk ranked."""

    mentioned = tuple(name for name in DOCUMENTS if name.casefold() in question.casefold())
    company = mentioned[0] if len(mentioned) == 1 else None
    companies = (company,) if company else tuple(DOCUMENTS)
    per_company = max(1, top_k // len(companies))
    selected: list[Chunk] = []
    for current_company in companies:
        query_words = _query_words(question, current_company)
        ranked = sorted(
            (chunk for chunk in chunks if chunk.company == current_company),
            key=lambda chunk: (
                -len(query_words & _words(chunk.text)),
                0 if chunk.kind == "table" else 1,
                chunk.page,
            ),
        )
        selected.extend(
            chunk for chunk in ranked[:per_company] if query_words & _words(chunk.text)
        )
    return tuple(selected[:top_k])


def build_prompt(question: str, hits: tuple[Chunk, ...]) -> str:
    evidence = "\n\n".join(
        f"[{hit.company}, page {hit.page}]\n{hit.text}" for hit in hits
    )
    return (
        "Answer the financial research question using only the evidence below. "
        "Separate reported facts from interpretation and cite every factual sentence "
        "as [Company, page]. Say when the evidence is insufficient.\n\n"
        f"Question: {question}\n\nEvidence:\n{evidence}"
    )


def _excerpt(text: str, limit: int = 220) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def offline_preview(hits: tuple[Chunk, ...]) -> str:
    """Give the classroom a readable result when no model key is configured."""

    return "Offline evidence preview:\n\n" + "\n\n".join(
        f"{hit.company} · page {hit.page}: {_excerpt(hit.text)}" for hit in hits[:4]
    )


def answer_from_evidence(question: str, hits: tuple[Chunk, ...]) -> str:
    """Call OpenAI when configured; otherwise provide a useful offline preview."""

    if not hits:
        return "No matching evidence was found in the tracked reports."
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        return offline_preview(hits)
    from openai import OpenAI

    response = OpenAI().responses.create(
        model=os.getenv("FINAI_CHAT_MODEL", "gpt-5.6-luna"),
        input=build_prompt(question, hits),
    )
    return response.output_text


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Financial Document Analyst", page_icon="📊")
    st.title("Financial Document Analyst")
    st.caption("A small RAG app built from two official company reports.")
    question = st.text_area(
        "Ask a question",
        "How did operating growth differ between NVIDIA and Schneider Electric?",
    )
    if st.button("Analyze", type="primary"):
        with st.spinner("Parsing reports and finding evidence..."):
            chunks = load_document_chunks()
            hits = retrieve_chunks(question, chunks)
        st.subheader("Answer")
        st.write(answer_from_evidence(question, hits))
        st.subheader("Evidence used")
        for hit in hits:
            with st.container(border=True):
                st.markdown(f"**{hit.company} · page {hit.page} · {hit.kind}**")
                st.write(_excerpt(hit.text, 420))
                with st.expander("Show full chunk"):
                    st.write(hit.text)
                st.markdown(f"[Open official source]({hit.source_url})")


if __name__ == "__main__":
    main()
