"""Build small deterministic binary fixtures used by the document laboratory."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "course-data" / "fixtures" / "schneider_fy2025_excerpt.pdf"


def build_schneider_fixture(output: Path = OUTPUT) -> Path:
    """Create a two-page machine-generated teaching extract with a real table."""

    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        figure = plt.figure(figsize=(8.27, 11.69))
        figure.text(0.10, 0.91, "Schneider Electric FY2025", fontsize=22, weight="bold")
        figure.text(0.10, 0.86, "Full-year results — teaching extract", fontsize=15)
        figure.text(
            0.10,
            0.77,
            (
                "Schneider Electric reported FY2025 revenue of approximately\n"
                "EUR 40.2 billion, with organic growth of 8.9%. Energy\n"
                "Management organic growth was 10%, supported by Data Center\n"
                "demand. This compact classroom fixture is based on the official\n"
                "FY2025 results release; it is not an original report page."
            ),
            fontsize=13,
            linespacing=1.55,
        )
        figure.text(
            0.10,
            0.56,
            "Source: Schneider Electric FY2025 full-year results release",
            fontsize=10,
            color="#475569",
        )
        plt.axis("off")
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(8.27, 11.69))
        axis.axis("off")
        axis.text(
            0.02,
            0.95,
            "Key financial metrics",
            transform=axis.transAxes,
            fontsize=22,
            weight="bold",
        )
        table = axis.table(
            cellText=[
                ["Revenue", "EUR 40.2bn", "+8.9% organic"],
                ["Energy Management", "n/a", "+10% organic"],
                ["Adjusted EBITA", "EUR 7.5bn", "18.7% margin"],
            ],
            colLabels=["Metric", "FY2025", "Change"],
            cellLoc="left",
            colLoc="left",
            bbox=[0.02, 0.53, 0.96, 0.30],
            colWidths=[0.38, 0.25, 0.33],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        for (row, _column), cell in table.get_celld().items():
            cell.set_edgecolor("#334155")
            cell.set_linewidth(1.2)
            if row == 0:
                cell.set_facecolor("#DDEBFF")
                cell.set_text_props(weight="bold")
        axis.text(
            0.02,
            0.45,
            (
                "A structure-aware parser should keep each row with its column headings.\n"
                "A naive character split can separate EUR 40.2bn from Revenue."
            ),
            transform=axis.transAxes,
            fontsize=13,
            linespacing=1.5,
        )
        axis.text(
            0.02,
            0.08,
            "Compact teaching extract based on the official FY2025 results release.",
            transform=axis.transAxes,
            fontsize=10,
            color="#475569",
        )
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

    return output


if __name__ == "__main__":
    print(build_schneider_fixture())
