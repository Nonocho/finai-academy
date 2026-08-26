"""Presentation-only charts for the Lesson 04 naive RAG notebook."""

from __future__ import annotations

from collections.abc import Sequence
from textwrap import fill
from typing import Protocol

import matplotlib.pyplot as plt

NAVY = "#082F49"
ROYAL = "#2563EB"
CYAN = "#06B6D4"
ORANGE = "#F97316"
RED = "#DC2626"
GREY = "#CBD5E1"
LIGHT = "#F1F5F9"


class RankedPassage(Protocol):
    score: float
    passage: object


def _finish(ax: plt.Axes) -> None:
    ax.grid(alpha=0.16)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_evidence_scale(filing_tokens: int, answer_budget: int) -> None:
    """Contrast the filing evidence universe with a concise answer."""

    _, ax = plt.subplots(figsize=(10, 4.4))
    labels = ["Complete filing", "Typical concise answer"]
    values = [filing_tokens, answer_budget]
    bars = ax.barh(labels, values, color=[ROYAL, ORANGE], height=0.52)
    ax.set_xscale("log")
    ax.set_xlabel("Approximate tokens · logarithmic scale")
    ax.set_title(
        "Why RAG? The evidence universe is much larger than one answer",
        loc="left",
        weight="bold",
    )
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            value * 1.08,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,}",
            va="center",
            weight="bold",
        )
    ax.spines["left"].set_visible(False)
    _finish(ax)


def plot_flattening(
    text: str,
    *,
    table_count: int,
    semantic_heading_count: int,
) -> None:
    """Show document structure before and after naive text flattening."""

    anchor_start = text.index("Revenue by End Market:")
    flattened_table = text[anchor_start : anchor_start + 560]
    _, (ax_counts, ax_text) = plt.subplots(
        1,
        2,
        figsize=(13, 4.8),
        gridspec_kw={"width_ratios": [0.8, 1.8]},
    )
    counts = [table_count, semantic_heading_count]
    bars = ax_counts.bar(
        ["HTML tables", "Semantic headings"],
        counts,
        color=[CYAN, RED],
        width=0.58,
    )
    ax_counts.set_title("Structure found before flattening", loc="left", weight="bold")
    ax_counts.set_ylabel("Elements")
    ax_counts.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, counts, strict=True):
        ax_counts.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            str(value),
            ha="center",
            weight="bold",
        )

    ax_text.axis("off")
    ax_text.set_title(
        "After flattening: the table becomes one text stream",
        loc="left",
        weight="bold",
    )
    ax_text.text(
        0.02,
        0.92,
        fill(flattened_table, 82),
        va="top",
        family="monospace",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.8", "facecolor": LIGHT, "edgecolor": GREY},
    )
    plt.tight_layout()
    plt.show()


def plot_fixed_windows(
    chunk_count: int,
    *,
    chunk_chars: int,
    overlap_chars: int,
) -> None:
    """Visualize overlapping character windows over flattened text."""

    _, ax = plt.subplots(figsize=(12, 3.6))
    step = chunk_chars - overlap_chars
    for index in range(6):
        start = index * step
        color = ROYAL if index % 2 == 0 else CYAN
        ax.broken_barh([(start, chunk_chars)], (index * 0.7, 0.45), facecolors=color)
        ax.text(
            start + 35,
            index * 0.7 + 0.225,
            f"C{index + 1:03d}",
            va="center",
            color="white",
            weight="bold",
        )
    ax.axvspan(
        step,
        chunk_chars,
        color=ORANGE,
        alpha=0.28,
        label=f"{overlap_chars}-character overlap",
    )
    ax.set_yticks([])
    ax.set_xlabel("Character position in the flattened filing")
    ax.set_title(
        f"Naive character windows create {chunk_count} retrieval candidates",
        loc="left",
        weight="bold",
    )
    ax.legend(frameon=False, loc="lower right")
    ax.spines["left"].set_visible(False)
    _finish(ax)


def plot_ranking_boundary(ranking: Sequence[RankedPassage], top_k: int) -> None:
    """Show which ranked windows cross the model's evidence boundary."""

    top = list(reversed(ranking[:12]))
    labels = [hit.passage.passage_id for hit in top]
    scores = [hit.score for hit in top]
    selected = [label in {hit.passage.passage_id for hit in ranking[:top_k]} for label in labels]
    _, ax = plt.subplots(figsize=(10.5, 5.1))
    bars = ax.barh(labels, scores, color=[ORANGE if value else GREY for value in selected])
    ax.axvline(ranking[top_k - 1].score, color=ORANGE, linestyle="--", linewidth=1.5)
    ax.set_xlabel("TF-IDF cosine similarity · not probability")
    ax.set_title("The top-k line becomes the model's evidence boundary", loc="left", weight="bold")
    for bar, value in zip(bars, scores, strict=True):
        ax.text(
            value + 0.004,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=9,
        )
    ax.spines["left"].set_visible(False)
    _finish(ax)


def plot_prompt_boundary(
    filing_tokens: int,
    selected_tokens: int,
    prompt_tokens: int,
) -> None:
    """Show how retrieval compresses the filing into the model prompt."""

    _, ax = plt.subplots(figsize=(10.5, 3.9))
    labels = ["Complete filing", "Retrieved evidence", "Final RAG prompt"]
    values = [filing_tokens, selected_tokens, prompt_tokens]
    bars = ax.barh(labels, values, color=[NAVY, ORANGE, ROYAL], height=0.55)
    ax.set_xscale("log")
    ax.set_xlabel("Approximate tokens · logarithmic scale")
    ax.set_title(
        "Retrieval cuts the model input to an auditable evidence set", loc="left", weight="bold"
    )
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            value * 1.08,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,}",
            va="center",
            weight="bold",
        )
    ax.spines["left"].set_visible(False)
    _finish(ax)


def plot_failure_boundary(
    ranking: Sequence[RankedPassage],
    *,
    precise_table_rank: int,
    top_k: int,
) -> None:
    """Show the required table outside the selected top-k evidence."""

    _, ax = plt.subplots(figsize=(11, 4.4))
    ax.axvspan(0.5, top_k + 0.5, color=ORANGE, alpha=0.16, label="Visible to model (top-k)")
    ax.scatter(range(1, 13), [hit.score for hit in ranking[:12]], s=70, color=GREY, edgecolor=NAVY)
    ax.scatter(
        [precise_table_rank],
        [ranking[precise_table_rank - 1].score],
        s=180,
        color=RED,
        marker="X",
        label="Precise revenue table",
    )
    ax.axvline(top_k + 0.5, color=ORANGE, linestyle="--", linewidth=2)
    ax.set_xlim(0.5, max(precise_table_rank + 2, 24))
    ax.set_xlabel("Rank position")
    ax.set_ylabel("TF-IDF cosine similarity")
    ax.set_title(
        "The correct table ranks outside top-k before generation begins", loc="left", weight="bold"
    )
    ax.legend(frameon=False)
    _finish(ax)
