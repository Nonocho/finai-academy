from __future__ import annotations

import re
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import pytest


ROOT = Path(__file__).resolve().parents[1]
DECKS = tuple(
    deck
    for deck in sorted((ROOT / "decks").glob("[0-9][0-9]-*.pptx"))
    if 1 <= int(deck.stem[:2]) <= 12
)
FOOTER = "First Finance - Arnaud Demes"


def _part_texts(deck: Path, prefix: str, pattern: str) -> list[str]:
    with ZipFile(deck) as package:
        parts = sorted(
            (
                name
                for name in package.namelist()
                if name.startswith(prefix) and name.endswith(".xml")
            ),
            key=lambda name: int(re.search(pattern, name).group(1)),
        )
        return [
            "\n".join(
                node.text or ""
                for node in ElementTree.fromstring(package.read(part)).iter()
                if node.tag.endswith("}t")
            )
            for part in parts
        ]


def _slide_texts(deck: Path) -> list[str]:
    return _part_texts(deck, "ppt/slides/slide", r"slide(\d+)\.xml$")


def _note_texts(deck: Path) -> list[str]:
    return _part_texts(
        deck,
        "ppt/notesSlides/notesSlide",
        r"notesSlide(\d+)\.xml$",
    )


def _slide_shapes(deck: Path) -> list[dict[str, str]]:
    with ZipFile(deck) as package:
        slide_parts = sorted(
            (
                name
                for name in package.namelist()
                if name.startswith("ppt/slides/slide") and name.endswith(".xml")
            ),
            key=lambda name: int(re.search(r"slide(\d+)\.xml$", name).group(1)),
        )
        result: list[dict[str, str]] = []
        for part in slide_parts:
            root = ElementTree.fromstring(package.read(part))
            shapes: dict[str, str] = {}
            for shape in (node for node in root.iter() if node.tag.endswith("}sp")):
                properties = next(
                    (node for node in shape.iter() if node.tag.endswith("}cNvPr")),
                    None,
                )
                if properties is None:
                    continue
                text = "".join(
                    node.text or "" for node in shape.iter() if node.tag.endswith("}t")
                ).strip()
                shapes[properties.attrib.get("name", "")] = text
            result.append(shapes)
        return result


@pytest.mark.parametrize("deck", DECKS, ids=lambda deck: deck.stem)
def test_each_lesson_deck_ends_with_a_three_question_quiz_and_answers(
    deck: Path,
) -> None:
    slides = _slide_texts(deck)
    shapes = _slide_shapes(deck)
    notes = _note_texts(deck)
    chapter_source = f"chapters/{deck.stem}.md"

    assert len(DECKS) == 12
    assert "Quick quiz" in slides[-2].splitlines()
    assert "Answers" in slides[-1].splitlines()
    assert sum("Quick quiz" in slide.splitlines() for slide in slides) == 1
    assert sum("Answers" in slide.splitlines() for slide in slides) == 1
    assert FOOTER in slides[-2]
    assert FOOTER in slides[-1]
    for number in range(1, 4):
        question = shapes[-2][f"quiz-question-{number}"]
        options = shapes[-2][f"quiz-options-{number}"]
        answer = shapes[-1][f"quiz-answer-{number}"]
        explanation = shapes[-1][f"quiz-explanation-{number}"]

        assert re.fullmatch(rf"{number}\.\s+\S.+", question)
        option_blocks = options.split("|")
        assert len(option_blocks) == 3
        assert all(
            re.fullmatch(rf"\s*{letter}\s+\S.*", block)
            for letter, block in zip("ABC", option_blocks, strict=True)
        )
        assert re.fullmatch(rf"{number}\.\s+[ABC]\.\s+\S.+", answer)
        assert 5 <= len(explanation.split()) <= 20

    assert {
        name for name in shapes[-2] if name.startswith("quiz-question-")
    } == {f"quiz-question-{number}" for number in range(1, 4)}
    assert {
        name for name in shapes[-2] if name.startswith("quiz-options-")
    } == {f"quiz-options-{number}" for number in range(1, 4)}
    assert {
        name for name in shapes[-1] if name.startswith("quiz-answer-")
    } == {f"quiz-answer-{number}" for number in range(1, 4)}
    assert {
        name for name in shapes[-1] if name.startswith("quiz-explanation-")
    } == {f"quiz-explanation-{number}" for number in range(1, 4)}
    assert len(notes) == len(slides)
    assert all("[Sources]" in note and "[/Sources]" in note for note in notes[-2:])
    assert all(chapter_source in note for note in notes[-2:])
