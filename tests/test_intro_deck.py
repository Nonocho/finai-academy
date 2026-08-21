from __future__ import annotations

import re
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
INTRO_DECK = ROOT / "decks" / "00-course-introduction.pptx"


def intro_slide_texts() -> list[str]:
    with ZipFile(INTRO_DECK) as package:
        slide_parts = sorted(
            (
                name
                for name in package.namelist()
                if name.startswith("ppt/slides/slide") and name.endswith(".xml")
            ),
            key=lambda name: int(re.search(r"slide(\d+)\.xml$", name).group(1)),
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


def test_intro_deck_contains_the_student_start_contract() -> None:
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


def test_every_intro_slide_has_the_course_footer() -> None:
    slide_texts = intro_slide_texts()

    assert slide_texts
    assert all("First Finance - Arnaud Demes" in text for text in slide_texts)
