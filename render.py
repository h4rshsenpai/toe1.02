"""Step 3: Render quantized drum hits as traditional 5-line staff notation → PDF."""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from drums import DrumType, QuantizedHit

# ---------------------------------------------------------------------------
# Layout constants (all in "staff units"; 1 unit = line spacing)
# ---------------------------------------------------------------------------

BARS_PER_ROW = 4
ROWS_PER_PAGE = 8
UNITS_PER_BAR = 16
LEAD = 3.5
BAR_W = UNITS_PER_BAR
ROW_W = LEAD + BARS_PER_ROW * BAR_W

LINES = [0.0, 1.0, 2.0, 3.0, 4.0]

DRUM_Y: dict[DrumType, float] = {
    DrumType.CRASH:        5.5,
    DrumType.RIDE:         5.0,
    DrumType.HIHAT_OPEN:   4.5,
    DrumType.HIHAT_CLOSED: 4.5,
    DrumType.TOM_HI:       3.5,
    DrumType.TOM_MID:      2.5,
    DrumType.SNARE:        1.5,
    DrumType.TOM_FLOOR:    0.5,
    DrumType.KICK:        -0.5,
}

# Color per drum type
DRUM_COLOR: dict[DrumType, str] = {
    DrumType.KICK:         "#C0392B",  # red
    DrumType.SNARE:        "#2980B9",  # blue
    DrumType.HIHAT_CLOSED: "#27AE60",  # green
    DrumType.HIHAT_OPEN:   "#1ABC9C",  # teal
    DrumType.CRASH:        "#E67E22",  # orange
    DrumType.RIDE:         "#8E44AD",  # purple
    DrumType.TOM_HI:       "#D4AC0D",  # gold
    DrumType.TOM_MID:      "#E67E22",  # amber
    DrumType.TOM_FLOOR:    "#784212",  # brown
}

# Human-readable label for legend
DRUM_LABEL: dict[DrumType, str] = {
    DrumType.KICK:         "Bass Drum",
    DrumType.SNARE:        "Snare",
    DrumType.HIHAT_CLOSED: "Hi-Hat (closed)",
    DrumType.HIHAT_OPEN:   "Hi-Hat (open)",
    DrumType.CRASH:        "Crash",
    DrumType.RIDE:         "Ride",
    DrumType.TOM_HI:       "Hi Tom",
    DrumType.TOM_MID:      "Mid Tom",
    DrumType.TOM_FLOOR:    "Floor Tom",
}

CYMBAL_TYPES = {DrumType.HIHAT_CLOSED, DrumType.HIHAT_OPEN, DrumType.CRASH, DrumType.RIDE}

FIG_W, FIG_H = 16.5, 11.7
Y_BOT, Y_TOP = -2.0, 7.0


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_staff(ax: plt.Axes) -> None:
    for y in LINES:
        ax.hlines(y, 0, ROW_W, colors="black", linewidths=0.8)


def _draw_perc_clef(ax: plt.Axes) -> None:
    for x in (0.4, 0.75):
        rect = mpatches.FancyBboxPatch(
            (x, LINES[0]), 0.22, LINES[-1] - LINES[0],
            boxstyle="square,pad=0", linewidth=0, facecolor="black",
        )
        ax.add_patch(rect)


def _draw_time_sig(ax: plt.Axes, meter: int = 4) -> None:
    x = 1.35
    mid = (LINES[0] + LINES[-1]) / 2
    kw = dict(ha="center", va="center", fontsize=14, fontweight="bold", fontfamily="serif")
    ax.text(x, mid + 0.9, str(meter), **kw)
    ax.text(x, mid - 0.9, str(meter), **kw)


def _draw_bar_lines(ax: plt.Axes, n_bars: int) -> None:
    for i in range(n_bars + 1):
        x = LEAD + i * BAR_W
        lw = 2.0 if i in (0, n_bars) else 0.8
        ax.vlines(x, LINES[0], LINES[-1], colors="black", linewidths=lw)


def _notehead_x(bar_in_row: int, sixteenth: int) -> float:
    return LEAD + bar_in_row * BAR_W + sixteenth + 0.5


def _draw_note(ax: plt.Axes, x: float, y: float, drum_type: DrumType, velocity: float) -> None:
    is_cymbal = drum_type in CYMBAL_TYPES
    color = DRUM_COLOR[drum_type]
    alpha = 0.55 + 0.45 * velocity

    if is_cymbal:
        d = 0.28
        ax.plot([x - d, x + d], [y - d, y + d], color=color, lw=1.8, solid_capstyle="round")
        ax.plot([x - d, x + d], [y + d, y - d], color=color, lw=1.8, solid_capstyle="round")

        if drum_type == DrumType.HIHAT_OPEN:
            circ = plt.Circle((x, y + 0.55), 0.18, fill=False, edgecolor=color, linewidth=1.2)
            ax.add_patch(circ)

        stem_bot = max(LINES[0], y - 2.5)
        ax.vlines(x, stem_bot, y, colors=color, linewidths=0.9, alpha=0.6)
    else:
        ellipse = mpatches.Ellipse(
            (x, y), width=0.65, height=0.48,
            facecolor=color, edgecolor=color, linewidth=0.5, alpha=alpha,
        )
        ax.add_patch(ellipse)

        if y < LINES[0]:
            ax.hlines(LINES[0], x - 0.55, x + 0.55, colors="black", linewidths=0.8)

        stem_top = max(y + 2.5, LINES[-1] + 0.5)
        ax.vlines(x, y + 0.24, stem_top, colors=color, linewidths=0.9, alpha=0.6)


def _draw_measure_numbers(ax: plt.Axes, first_measure: int, n_bars: int) -> None:
    for i in range(n_bars):
        x = LEAD + i * BAR_W + 0.3
        ax.text(x, Y_TOP - 0.3, str(first_measure + i + 1),
                fontsize=5, color="#555555", va="top")


def _draw_legend(fig: plt.Figure, present_types: set[DrumType]) -> None:
    """Draw a compact legend in the top-right corner of the figure."""
    ordered = [dt for dt in DRUM_LABEL if dt in present_types]

    handles = []
    for dt in ordered:
        color = DRUM_COLOR[dt]
        if dt in CYMBAL_TYPES:
            marker = plt.Line2D([0], [0], marker="x", color=color,
                                markeredgewidth=2, markersize=7,
                                linestyle="None", label=DRUM_LABEL[dt])
        else:
            marker = plt.Line2D([0], [0], marker="o", color=color,
                                markersize=7, linestyle="None",
                                label=DRUM_LABEL[dt])
        handles.append(marker)

    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        fontsize=6.5,
        framealpha=0.85,
        edgecolor="#cccccc",
        ncol=2,
        handletextpad=0.4,
        columnspacing=1.0,
        title="Drum Key",
        title_fontsize=7,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render(
    hits: List[QuantizedHit],
    bpm: float,
    output_path: Path,
    meter: int = 4,
    bars_per_row: int = BARS_PER_ROW,
    rows_per_page: int = ROWS_PER_PAGE,
) -> None:
    if not hits:
        raise ValueError("No hits to render.")

    n_measures = hits[-1].measure + 1
    n_rows  = -(-n_measures // bars_per_row)
    n_pages = -(-n_rows // rows_per_page)

    present_types = {h.drum_type for h in hits}

    by_measure: dict[int, list[QuantizedHit]] = {}
    for h in hits:
        by_measure.setdefault(h.measure, []).append(h)

    with PdfPages(str(output_path)) as pdf:
        for page in range(n_pages):
            fig, axes = plt.subplots(rows_per_page, 1, figsize=(FIG_W, FIG_H))
            if rows_per_page == 1:
                axes = [axes]

            fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01, hspace=0.6)

            for row_on_page, ax in enumerate(axes):
                global_row = page * rows_per_page + row_on_page
                first_measure = global_row * bars_per_row

                ax.set_xlim(0, ROW_W)
                ax.set_ylim(Y_BOT, Y_TOP)
                ax.set_aspect("auto")
                ax.axis("off")

                if first_measure >= n_measures:
                    continue

                actual_bars = min(bars_per_row, n_measures - first_measure)

                _draw_staff(ax)
                _draw_perc_clef(ax)
                if global_row == 0:
                    _draw_time_sig(ax, meter)
                _draw_bar_lines(ax, actual_bars)
                _draw_measure_numbers(ax, first_measure, actual_bars)

                if global_row == 0:
                    ax.text(ROW_W - 0.2, Y_TOP - 0.3, f"♩= {bpm:.0f}",
                            fontsize=7, ha="right", va="top", fontfamily="DejaVu Sans")

                for bar_in_row in range(actual_bars):
                    measure = first_measure + bar_in_row
                    for h in by_measure.get(measure, []):
                        x = _notehead_x(bar_in_row, h.sixteenth)
                        y = DRUM_Y[h.drum_type]
                        _draw_note(ax, x, y, h.drum_type, h.velocity)

            # Legend on every page (top-right, outside the staff rows)
            _draw_legend(fig, present_types)

            pdf.savefig(fig, dpi=150)
            plt.close(fig)

    print(f"Notation PDF written to {output_path}  ({n_pages} page(s), {n_measures} measures)")
