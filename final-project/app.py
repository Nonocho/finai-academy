"""Command-line entry point for the first Financial Analyst Copilot slice."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from finai_academy import Settings
from finai_academy.capstone import AnalystBriefService
from finai_academy.capstone.model_gateway import create_structured_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company", required=True)
    parser.add_argument("--period", required=True, dest="reporting_period")
    parser.add_argument("--input", required=True, type=Path, dest="input_path")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    source_text = args.input_path.read_text(encoding="utf-8")

    settings = Settings.from_environment()
    model = create_structured_model(settings)
    service = AnalystBriefService(model)
    brief = service.generate(
        company=args.company,
        reporting_period=args.reporting_period,
        source_text=source_text,
    )
    print(brief.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
