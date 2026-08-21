import subprocess
import sys
import zipfile
from itertools import pairwise
from pathlib import Path
from xml.etree import ElementTree

import yaml

ROOT = Path(__file__).resolve().parents[1]


def time_to_minutes(value: str) -> int:
    hours, minutes = (int(part) for part in value.split(":"))
    return hours * 60 + minutes


def load_course_manifest() -> dict:
    return yaml.safe_load((ROOT / "course.yml").read_text(encoding="utf-8"))


def pptx_text(deck_path: Path, part_name: str) -> str:
    """Return all visible text runs from one presentation package part."""

    with zipfile.ZipFile(deck_path) as package:
        root = ElementTree.fromstring(package.read(part_name))
    return " ".join(node.text or "" for node in root.iter() if node.tag.endswith("}t"))


def test_two_day_manifest_defines_twelve_ordered_notebooks() -> None:
    manifest = load_course_manifest()

    notebook_lessons = [
        lesson for lesson in manifest["lessons"] if lesson.get("notebook") is not None
    ]

    assert len(notebook_lessons) == 12
    assert [lesson["id"] for lesson in notebook_lessons] == [f"{index:02d}" for index in range(1, 13)]
    assert len({lesson["notebook"] for lesson in notebook_lessons}) == 12


def test_two_day_manifest_matches_the_live_delivery_window() -> None:
    manifest = load_course_manifest()
    delivery = manifest["delivery"]["two_day"]

    assert delivery["day_start"] == "09:00"
    assert delivery["day_end"] == "17:00"
    assert delivery["lunch"] == {"start": "12:00", "end": "13:30"}
    assert delivery["breaks"] == [
        {"day": 1, "start": "10:30", "end": "10:45"},
        {"day": 1, "start": "15:00", "end": "15:15"},
        {"day": 2, "start": "10:15", "end": "10:30"},
    ]


def test_lesson_windows_do_not_overlap() -> None:
    manifest = load_course_manifest()

    for day in (1, 2):
        lessons = [lesson for lesson in manifest["lessons"] if lesson["day"] == day]
        windows = [
            (
                time_to_minutes(lesson["start"]),
                time_to_minutes(lesson["end"]),
            )
            for lesson in lessons
        ]

        assert all(start < end for start, end in windows)
        assert all(
            previous_end <= next_start
            for (_, previous_end), (next_start, _) in pairwise(windows)
        )


def test_repository_validator_accepts_canonical_manifest_paths() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_repo.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "repository structure is valid" in result.stdout


def test_implemented_lesson_four_assets_exist() -> None:
    manifest = load_course_manifest()
    lesson = next(item for item in manifest["lessons"] if item["id"] == "04")

    assert (ROOT / lesson["chapter"]).is_file()
    assert (ROOT / lesson["notebook"]).is_file()
    assert (ROOT / lesson["deck"]).is_file()


def test_implemented_lesson_five_assets_exist() -> None:
    manifest = load_course_manifest()
    lesson = next(item for item in manifest["lessons"] if item["id"] == "05")

    assert (ROOT / lesson["chapter"]).is_file()
    assert (ROOT / lesson["notebook"]).is_file()
    assert (ROOT / lesson["deck"]).is_file()


def test_implemented_lesson_six_assets_exist() -> None:
    manifest = load_course_manifest()
    lesson = next(item for item in manifest["lessons"] if item["id"] == "06")

    assert (ROOT / lesson["chapter"]).is_file()
    assert (ROOT / lesson["notebook"]).is_file()
    assert (ROOT / lesson["deck"]).is_file()


def test_implemented_lesson_seven_assets_exist() -> None:
    manifest = load_course_manifest()
    lesson = next(item for item in manifest["lessons"] if item["id"] == "07")

    assert lesson["start"] == "16:00"
    assert lesson["end"] == "16:45"
    assert (ROOT / lesson["chapter"]).is_file()
    assert (ROOT / lesson["notebook"]).is_file()
    assert (ROOT / lesson["deck"]).is_file()


def test_lesson_six_slide_three_uses_one_executed_query_and_traceable_manifest_ids() -> None:
    """The comparison slide must report real ranks from one maintained notebook run."""

    deck_path = ROOT / "decks" / "06-hybrid-retrieval.pptx"
    slide_text = pptx_text(deck_path, "ppt/slides/slide3.xml")
    notes_text = pptx_text(deck_path, "ppt/notesSlides/notesSlide3.xml")

    assert "Which NVIDIA business generated $193.7 billion?" in slide_text
    assert "NVDA…003" in slide_text
    assert "NVDA…001" in slide_text
    assert "NVDA…002" in slide_text
    assert "SU-TABLE" not in slide_text
    assert "NVDA-TABLE" not in slide_text
    assert "NVDA-2026-10K-EXCERPT-CONTEXTUAL-003" in notes_text
    normalized_notes = notes_text.casefold()
    assert "keyword rank 1" in normalized_notes
    assert "dense rank 4" in normalized_notes
