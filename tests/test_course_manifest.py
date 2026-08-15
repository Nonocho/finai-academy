from itertools import pairwise
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def time_to_minutes(value: str) -> int:
    hours, minutes = (int(part) for part in value.split(":"))
    return hours * 60 + minutes


def load_course_manifest() -> dict:
    return yaml.safe_load((ROOT / "course.yml").read_text(encoding="utf-8"))


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
