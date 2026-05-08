"""Generate visual + tabular artifacts (PNG charts, CSVs) from persisted eval rows.

Called once per eval run, after the capability report is written. Outputs to
``evaluation_outputs/`` at the repo root. All scoring derives from existing
persisted columns on :class:`EvaluationCaseResult` (no schema change).

The chart generators are tuned for inclusion in a research paper:

* Serif typography, 220 DPI, restrained dashed gridlines, hidden top/right
  spines.
* Color-blind safe Okabe-Ito palette.
* Plain-English query intent labels (``compare`` → "Compare two concepts" …)
  so a reader unfamiliar with the codebase can interpret the figures
  without surrounding prose.
* Value labels on every bar plus per-figure italic captions explaining
  what the metric measures.

Eight PNGs may be produced in ``out_dir`` after a run:

1. ``pipeline_diagram.png`` — static architecture diagram of the answer
   pipeline.
2. ``question_type_breakdown.png`` — case count + pass rate per query
   intent.
3. ``retrieval_accuracy.png`` — share of cases whose top-k retrieval
   contained the expected lecture chunk, broken down by query intent.
4. ``evaluation_summary.png`` — scorecard with explicit small-sample-size
   caveats.
5. ``regression_comparison.png`` — previous vs current run on three
   regression metrics (or a no-movement notice image).
6. ``report_dashboard.png`` — structure compliance, retrieval diagnostics,
   and dataset-health status.
7. ``coverage_by_concept.png`` — concept-level pass/fail coverage counts.
8. ``failure_modes.png`` — failure-type counts with example test IDs.

Two CSVs are also produced for paper-side analysis:

* ``example_answers.csv`` — top and bottom scoring cases: one **physical row
  per example**, metadata first, flat ``course_answer_one_line`` last
  (paragraph breaks shown as `` || ``) so the file is readable in a text
  editor and imports cleanly into Excel / pandas.
* ``error_analysis.csv`` — primary failure types with example queries.
"""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from app.eval.analytics_common import parse_expected_behavior, parse_json_list  # noqa: E402,F401
from app.eval.capability_analytics import (  # noqa: E402
    _case_concept,
    _round_rate,
    _row_intent,
    primary_error_type_for_row,
    retrieval_diagnostics,
    summarize_boost,
    summarize_capability,
    summarize_coverage,
    summarize_errors,
    summarize_retrieval,
    summarize_structure,
)
from app.models.evaluation import EvaluationCaseResult, EvaluationRun  # noqa: E402

# ---------------------------------------------------------------------------
# Publication-grade styling
# ---------------------------------------------------------------------------

# Okabe-Ito color-blind safe palette.
_COLOR_PRIMARY = "#0072B2"        # blue
_COLOR_SECONDARY = "#D55E00"      # vermillion
_COLOR_GOOD = "#117733"           # green (improvement / pass)
_COLOR_BAD = "#CC3311"            # red (regression / fail)
_COLOR_NEUTRAL_BAR = "#999999"    # gray (previous-run / baseline)
_COLOR_NEUTRAL_LINE = "#444444"
_COLOR_GRID = "#CFCFCF"
_COLOR_BOX_FILL = "#EAF2FB"
_COLOR_BOX_EDGE = "#0072B2"
_COLOR_BOX_SHADOW = "#B7C9DD"
_COLOR_TEXT = "#102A43"
_COLOR_TEXT_MUTED = "#555"
_COLOR_TEXT_SOFT = "#334E68"

_FIG_DPI = 220  # crisp for print without bloating file size

_NUMBERED_LIST_RE = re.compile(r"^\s*\d+[\.\)]", re.MULTILINE)

# Plain-English display labels for query intents. Used in tick labels of
# every chart that buckets by intent so readers unfamiliar with the
# codebase can interpret the categories on their own.
_INTENT_DISPLAY = {
    "compare": "Compare two\nconcepts",
    "definition": "Define a\nconcept",
    "retrieval_grounded": "Look up a\nspecific fact",
    "step_by_step": "Walk through /\nsummarize / quiz",
    "synthesis": "Connect multiple\nconcepts",
}

_LOW_N_THRESHOLD = 5
_SMALL_DATASET_THRESHOLD = 50
_LOW_N_CAPTION_SUFFIX = "Percentages are descriptive only when bucket sizes are small."

_NORMAL_RETRIEVAL_INTENTS = {"definition", "retrieval_grounded", "synthesis"}
_MODE_CLARIFICATION_INTENTS = {"compare", "step_by_step"}
_UNDERSPECIFIED_PATTERNS = (
    re.compile(r"^\s*compare\s+(this|these|it|that)\b", re.IGNORECASE),
    re.compile(r"^\s*summarize\s+(this|these|it|that)\b", re.IGNORECASE),
    re.compile(r"^\s*explain\s+(this|these|it|that)\b", re.IGNORECASE),
    re.compile(r"^\s*(compare|summarize|explain)\s*$", re.IGNORECASE),
)


def _format_intent(intent: str) -> str:
    if intent == "underspecified":
        return "Underspecified\nprompt"
    if intent in _INTENT_DISPLAY:
        return _INTENT_DISPLAY[intent]
    return intent.replace("_", " ").strip().title()


def _intent_label_csv(intent: str) -> str:
    """Plain one-line label for CSV (same wording as charts, no embedded newlines)."""
    return _format_intent(intent).replace("\n", " ").strip()


def _low_n_warning(n: int) -> str:
    if n <= 2:
        return "n too small"
    if n < _LOW_N_THRESHOLD:
        return "low n"
    return ""


def _fade_alpha(n: int) -> float:
    return 1.0 if n >= _LOW_N_THRESHOLD else 0.35


def _intent_n_label(intent: str, n: int) -> str:
    warning = _low_n_warning(n)
    suffix = f"  {warning}" if warning else ""
    return f"{_format_intent(intent)}\n(n={n}{suffix})"


def _is_underspecified_query(query: str | None) -> bool:
    text = (query or "").strip()
    if not text:
        return True
    lowered = text.lower()
    for pattern in _UNDERSPECIFIED_PATTERNS:
        if pattern.search(lowered):
            return True
    words = re.findall(r"[a-z0-9']+", lowered)
    if len(words) < 4 and any(token in {"this", "these", "that", "it"} for token in words):
        return True
    return False


def _dataset_size_note(total_cases: int) -> str:
    if total_cases < _SMALL_DATASET_THRESHOLD:
        return (
            f"Small evaluation set (n={total_cases}): use this as a debugging snapshot, "
            "not a final benchmark."
        )
    return (
        f"Evaluation set size (n={total_cases}) supports descriptive comparisons, "
        "but confidence intervals are still recommended."
    )


def _strongest_weakest(cases: list[EvaluationCaseResult]) -> tuple[str, str]:
    capability = summarize_capability(cases)
    rows: list[tuple[str, int, float]] = []
    for intent, stats in capability.get("by_intent", {}).items():
        n = int(stats.get("total_cases", 0))
        if n < _LOW_N_THRESHOLD:
            continue
        rows.append((intent, n, float(stats.get("accuracy", 0.0))))
    if not rows:
        return "insufficient data", "insufficient data"
    strongest = max(rows, key=lambda item: item[2])
    weakest = min(rows, key=lambda item: item[2])
    return (
        f"{_intent_label_csv(strongest[0])} ({strongest[2] * 100:.0f}%, n={strongest[1]})",
        f"{_intent_label_csv(weakest[0])} ({weakest[2] * 100:.0f}%, n={weakest[1]})",
    )


def _dataset_health(
    cases: list[EvaluationCaseResult], *, regression_meaningful: bool | None = None
) -> dict[str, Any]:
    grouped: dict[str, int] = defaultdict(int)
    for row in cases:
        grouped[_row_intent(row)] += 1
    low_n_intents = sorted(intent for intent, n in grouped.items() if n < _LOW_N_THRESHOLD)
    coverage = summarize_coverage(cases)
    return {
        "total_cases": len(cases),
        "under_tested_concepts": list(coverage.get("under_tested_concepts", [])),
        "low_n_intents": low_n_intents,
        "regression_meaningful": bool(regression_meaningful),
        "paired_boost_present": summarize_boost(cases) is not None,
    }


def _flatten_answer_for_csv(text: str, *, max_chars: int = 2500) -> str:
    r"""Turn a multi-line Course Answer into a single spreadsheet-safe line.

    Paragraphs (split on blank lines) are joined with `` || `` so paragraph
    structure is still visible without breaking CSV rows. Internal single
    newlines become spaces.
    """
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    segments = [re.sub(r"\s+", " ", seg.strip()) for seg in re.split(r"\n\s*\n+", t) if seg.strip()]
    if not segments:
        segments = [re.sub(r"\s+", " ", t)]
    out = " || ".join(segments)
    if len(out) > max_chars:
        out = out[: max_chars - 3] + "..."
    return out


@contextmanager
def _paper_style() -> Iterator[None]:
    """Apply restrained, paper-friendly matplotlib defaults for one figure.

    Uses ``plt.rc_context`` so we never mutate the global rcParams (other
    callers in the same process keep their settings).
    """
    rc = {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "regular",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.color": _COLOR_GRID,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "figure.dpi": _FIG_DPI,
        "savefig.dpi": _FIG_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.18,
    }
    with plt.rc_context(rc):
        yield


def _figure_caption(fig, text: str) -> None:
    """Add an italic figure caption pinned to the bottom of the figure."""
    fig.text(
        0.5,
        0.005,
        text,
        ha="center",
        va="bottom",
        fontsize=9,
        color=_COLOR_TEXT_MUTED,
        style="italic",
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def derive_case_scores(row: EvaluationCaseResult) -> dict[str, int | float]:
    """Derive five rubric-style scores for a single persisted case.

    All values are computed from existing columns; no DB writes occur here.
    """
    text = (row.actual_response or "").strip()
    text_lower = text.lower()
    error_categories = parse_json_list(row.error_categories_json)

    diags = retrieval_diagnostics([row])
    retrieval_quality = 1 if diags and diags[0].top_k_contains_correct else 0

    grounding = (
        1
        if "forbidden_leak" not in error_categories
        and (row.primary_error_type or "") != "hallucination"
        else 0
    )

    head = text_lower[:80]
    if len(text) < 20 or "could you" in head or "can you clarify" in head:
        explanation_quality = 0
    elif "key idea" in text_lower and len(text) >= 100:
        explanation_quality = 2
    else:
        explanation_quality = 1

    sentences = [s for s in re.split(r"\.(?:\n|\s|$)", text) if s.strip()]
    if len(sentences) < 2:
        depth = 0
    elif (
        "for example" in text_lower
        or "e.g." in text_lower
        or "because" in text_lower
        or "mechanism" in text_lower
        or _NUMBERED_LIST_RE.search(text) is not None
    ):
        depth = 2
    else:
        depth = 1

    if row.pass_bool:
        question_handling = 1
    else:
        question_handling = (
            1 if not any(cat.startswith("structure_") for cat in error_categories) else 0
        )

    return {
        "retrieval_quality": retrieval_quality,
        "grounding": grounding,
        "explanation_quality": explanation_quality,
        "depth": depth,
        "question_handling": question_handling,
    }


# ---------------------------------------------------------------------------
# Chart generators
# ---------------------------------------------------------------------------


def write_retrieval_accuracy_chart(
    cases: list[EvaluationCaseResult], out_dir: Path
) -> None:
    """Two-panel retrieval view: normal content prompts vs mode/clarification."""
    normal_grouped: dict[str, list[EvaluationCaseResult]] = defaultdict(list)
    special_grouped: dict[str, list[EvaluationCaseResult]] = defaultdict(list)
    for row in cases:
        intent = _row_intent(row)
        underspecified = _is_underspecified_query(row.query_text)
        if intent in _NORMAL_RETRIEVAL_INTENTS and not underspecified:
            normal_grouped[intent].append(row)
            continue
        if underspecified:
            special_grouped["underspecified"].append(row)
            continue
        if intent in _MODE_CLARIFICATION_INTENTS:
            special_grouped[intent].append(row)
            continue
        # Keep any remaining intents out of the "normal retrieval" panel.
        special_grouped[intent].append(row)

    with _paper_style():
        fig, (ax_normal, ax_special) = plt.subplots(1, 2, figsize=(13.0, 5.6), sharey=True)

        def _draw_panel(
            ax: Any,
            grouped_cases: dict[str, list[EvaluationCaseResult]],
            *,
            title: str,
            color: str,
        ) -> None:
            rows: list[tuple[str, float, int]] = []
            for intent, bucket in grouped_cases.items():
                diags = retrieval_diagnostics(bucket)
                total = len(diags)
                correct = sum(1 for d in diags if d.top_k_contains_correct)
                rows.append((intent, _round_rate(correct, total), total))
            rows.sort(key=lambda item: item[2], reverse=True)
            if not rows:
                ax.axis("off")
                ax.text(
                    0.5,
                    0.5,
                    "No cases in this split",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color=_COLOR_TEXT_MUTED,
                    style="italic",
                    transform=ax.transAxes,
                )
                ax.set_title(title)
                return

            xs = list(range(len(rows)))
            bars = ax.bar(
                xs,
                [row[1] for row in rows],
                color=color,
                edgecolor="white",
                linewidth=0.8,
                width=0.62,
                zorder=3,
            )
            for bar, (_, _, n) in zip(bars, rows):
                bar.set_alpha(_fade_alpha(n))

            ax.axhline(0, color=_COLOR_NEUTRAL_LINE, linewidth=0.6, zorder=2)
            ax.set_xticks(xs)
            ax.set_xticklabels([_intent_n_label(intent, n) for intent, _, n in rows])
            ax.set_ylim(0, 1.08)
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
            ax.set_axisbelow(True)
            ax.xaxis.grid(False)
            ax.set_title(title)
            for bar, (_, acc, n) in zip(bars, rows):
                warning = _low_n_warning(n)
                label = f"{acc * 100:.0f}%"
                if warning:
                    label = f"{label}\n{warning}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=8.7,
                    color="#222",
                )

        _draw_panel(
            ax_normal,
            normal_grouped,
            title="A) Normal content retrieval cases",
            color=_COLOR_PRIMARY,
        )
        _draw_panel(
            ax_special,
            special_grouped,
            title="B) Mode / clarification / quiz / summary cases",
            color=_COLOR_SECONDARY,
        )
        ax_normal.set_ylabel(
            "Share of cases where the correct lecture chunk\n"
            "appeared in the top-5 retrieved candidates"
        )

        fig.suptitle("Did the Tutor Retrieve the Right Material?", fontsize=14, fontweight="bold")
        _figure_caption(
            fig,
            "Left panel measures retrieval accuracy on standard content questions. "
            "Right panel isolates mode and clarification prompts so underspecified requests "
            "(for example, 'Compare these') are not treated as normal retrieval failures. "
            f"{_dataset_size_note(len(cases))} {_LOW_N_CAPTION_SUFFIX}",
        )

        fig.tight_layout(rect=(0, 0.07, 1, 0.94))
        fig.savefig(out_dir / "retrieval_accuracy.png")
        plt.close(fig)


def write_question_type_chart(
    cases: list[EvaluationCaseResult], out_dir: Path
) -> None:
    """Grouped bars: case count and pass rate per intent on a shared figure."""
    capability = summarize_capability(cases)
    by_intent = capability.get("by_intent", {})
    rows: list[tuple[str, int, float]] = []
    for intent, stats in by_intent.items():
        rows.append(
            (
                intent,
                int(stats.get("total_cases", 0)),
                float(stats.get("accuracy", 0.0)),
            )
        )
    rows.sort(key=lambda item: item[1], reverse=True)
    intents = [row[0] for row in rows]
    totals = [row[1] for row in rows]
    accuracies = [row[2] for row in rows]
    overall_acc = float(capability.get("overall_accuracy", 0.0))

    with _paper_style():
        fig, ax_count = plt.subplots(figsize=(11.0, 5.6))
        xs = list(range(len(intents)))
        width = 0.38

        max_total = max(totals + [1])
        count_top = max(max_total + 1, 2)

        count_bars = ax_count.bar(
            [x - width / 2 for x in xs],
            totals,
            width=width,
            color=_COLOR_PRIMARY,
            edgecolor="white",
            linewidth=0.8,
            label="Number of evaluation questions",
            zorder=3,
        )
        for bar, n in zip(count_bars, totals):
            bar.set_alpha(_fade_alpha(n))

        ax_count.set_xticks(xs)
        ax_count.set_xticklabels([_intent_n_label(i, n) for i, n in zip(intents, totals)], rotation=0)
        ax_count.set_ylim(0, count_top)
        ax_count.set_yticks(range(0, count_top + 1, max(1, count_top // 5)))
        ax_count.set_ylabel("Number of evaluation questions", color=_COLOR_PRIMARY)
        ax_count.tick_params(axis="y", labelcolor=_COLOR_PRIMARY)
        ax_count.set_axisbelow(True)
        ax_count.xaxis.grid(False)

        ax_acc = ax_count.twinx()
        ax_acc.spines["top"].set_visible(False)
        acc_bars = ax_acc.bar(
            [x + width / 2 for x in xs],
            accuracies,
            width=width,
            color=_COLOR_SECONDARY,
            edgecolor="white",
            linewidth=0.8,
            label="Share that passed scoring",
            zorder=3,
        )
        for bar, n in zip(acc_bars, totals):
            bar.set_alpha(_fade_alpha(n))
        ax_acc.set_ylim(0, 1.08)
        ax_acc.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax_acc.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax_acc.set_ylabel("Share that passed scoring", color=_COLOR_SECONDARY)
        ax_acc.tick_params(axis="y", labelcolor=_COLOR_SECONDARY)
        ax_acc.grid(False)

        for bar, total in zip(count_bars, totals):
            warning = _low_n_warning(total)
            text = str(total) if not warning else f"{total}\n{warning}"
            ax_count.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + count_top * 0.02,
                text,
                ha="center",
                va="bottom",
                fontsize=8.7,
                color=_COLOR_PRIMARY,
            )
        for bar, acc, total in zip(acc_bars, accuracies, totals):
            warning = _low_n_warning(total)
            text = f"{acc * 100:.0f}%" if not warning else f"{acc * 100:.0f}%\n{warning}"
            ax_acc.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.025,
                text,
                ha="center",
                va="bottom",
                fontsize=8.7,
                color=_COLOR_SECONDARY,
            )

        ax_count.set_title("How the Tutor Performs on Different Kinds of Questions")
        _figure_caption(
            fig,
            "Blue bars (left axis) count the evaluation questions in each category. "
            "Orange bars (right axis) show what share of those\nquestions passed the scoring rubric. "
            f"Overall pass rate across all categories: {overall_acc * 100:.1f}%. "
            "Compare and synthesis buckets are unstable when n is small. "
            f"{_dataset_size_note(len(cases))} {_LOW_N_CAPTION_SUFFIX}",
        )

        handles = [count_bars, acc_bars]
        labels = [h.get_label() for h in handles]
        ax_count.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.0, 1.0))

        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(out_dir / "question_type_breakdown.png")
        plt.close(fig)


# Pipeline diagram stages: headline + caption; use explicit "\\n" so long
# labels fit inside the box (matplotlib does not clip text to the patch).
_PIPELINE_STAGES: tuple[tuple[str, str], ...] = (
    ("User Query", "Free-text question"),
    ("Query\nClassification", "Detect intent\nand mode"),
    ("Retrieval", "Find relevant\nlecture chunks"),
    ("Answer Composer", "Build a rule-based\nanswer"),
    ("Optional Boost", "LLM elaboration\n(off by default)"),
    ("Final Output", "Course answer\nto the user"),
)


def _pipeline_text_width_data(ax, fig, text_obj: Any) -> float:
    """Horizontal span of *text_obj* in x data coordinates (axis data space)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bb = text_obj.get_window_extent(renderer=renderer)
    inv = ax.transData.inverted()
    xs = [
        inv.transform((bb.x0, bb.y0))[0],
        inv.transform((bb.x1, bb.y0))[0],
        inv.transform((bb.x0, bb.y1))[0],
        inv.transform((bb.x1, bb.y1))[0],
    ]
    return max(xs) - min(xs)


def _pipeline_fit_text(
    ax: Any,
    fig: Any,
    x: float,
    y: float,
    s: str,
    *,
    max_data_width: float,
    fontsize_start: float,
    fontweight: str | None,
    color: str,
    linespacing: float,
) -> None:
    """Draw centered text and shrink fontsize until it fits *max_data_width*."""
    fs = float(fontsize_start)
    t = None
    while fs >= 6.0:
        if t is not None:
            t.remove()
        t = ax.text(
            x,
            y,
            s,
            ha="center",
            va="center",
            fontsize=fs,
            fontweight=fontweight,
            color=color,
            linespacing=linespacing,
            zorder=3,
        )
        if _pipeline_text_width_data(ax, fig, t) <= max_data_width * 0.86:
            return
        fs -= 0.5


def write_pipeline_diagram(out_dir: Path) -> None:
    """Static, DB-free architecture diagram of the answer pipeline.

    Stages are drawn as rounded boxes with a soft drop shadow, connected by
    arrowheads. Headline and caption strings are measured after layout;
    fontsize steps down until each label fits inside the box width in data
    coordinates (matplotlib does not clip text to the patch outline).
    """
    n = len(_PIPELINE_STAGES)
    box_w = 3.95
    box_h = 1.62
    gap = 0.48
    total_w = n * box_w + (n - 1) * gap
    inner_w = box_w * 0.92

    with _paper_style():
        fig, ax = plt.subplots(figsize=(min(18.5, total_w + 1.6), 4.05))

        ax.set_xlim(-0.4, total_w + 0.4)
        ax.set_ylim(0, 3.5)
        ax.set_aspect("equal")
        ax.axis("off")

        y = 1.12

        for i, (head, caption) in enumerate(_PIPELINE_STAGES):
            x = i * (box_w + gap)

            shadow = FancyBboxPatch(
                (x + 0.06, y - 0.08),
                box_w,
                box_h,
                boxstyle="round,pad=0.02,rounding_size=0.18",
                facecolor=_COLOR_BOX_SHADOW,
                edgecolor="none",
                alpha=0.55,
                zorder=1,
            )
            ax.add_patch(shadow)

            box = FancyBboxPatch(
                (x, y),
                box_w,
                box_h,
                boxstyle="round,pad=0.02,rounding_size=0.18",
                facecolor=_COLOR_BOX_FILL,
                edgecolor=_COLOR_BOX_EDGE,
                linewidth=1.4,
                zorder=2,
            )
            ax.add_patch(box)

            _pipeline_fit_text(
                ax,
                fig,
                x + box_w / 2,
                y + box_h - 0.48,
                head,
                max_data_width=inner_w,
                fontsize_start=11,
                fontweight="bold",
                color=_COLOR_TEXT,
                linespacing=0.92,
            )
            _pipeline_fit_text(
                ax,
                fig,
                x + box_w / 2,
                y + 0.46,
                caption,
                max_data_width=inner_w,
                fontsize_start=9,
                fontweight=None,
                color=_COLOR_TEXT_SOFT,
                linespacing=0.92,
            )

            if i < n - 1:
                start = (x + box_w, y + box_h / 2)
                end = (x + box_w + gap, y + box_h / 2)
                ax.add_patch(
                    FancyArrowPatch(
                        start,
                        end,
                        arrowstyle="-|>",
                        mutation_scale=14,
                        color=_COLOR_NEUTRAL_LINE,
                        linewidth=1.3,
                        zorder=2,
                    )
                )

        ax.text(
            total_w / 2,
            y + box_h + 0.55,
            "Neural Tutor Answer Pipeline",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color=_COLOR_TEXT,
        )
        ax.text(
            total_w / 2,
            y - 0.42,
            "End-to-end flow for a single chat turn: deterministic mode routing, "
            "rule-based answer composition, and an optional LLM boost.\n"
            "Sample size does not apply here; use the metric charts for empirical claims.",
            ha="center",
            va="center",
            fontsize=9,
            color=_COLOR_TEXT_MUTED,
            style="italic",
        )

        fig.savefig(out_dir / "pipeline_diagram.png")
        plt.close(fig)


def write_evaluation_summary_chart(
    run: EvaluationRun,
    cases: list[EvaluationCaseResult],
    out_dir: Path,
    *,
    health: dict[str, Any] | None = None,
) -> None:
    """At-a-glance scorecard with low-sample-size caveats."""
    total = len(cases)
    passed = sum(1 for c in cases if c.pass_bool)
    failed = total - passed
    mean_score = float(run.overall_score) if run.overall_score is not None else 0.0
    pass_rate = passed / total if total else 0.0
    git_short = (run.git_commit or "")[:7]
    strongest, weakest = _strongest_weakest(cases)
    health_data = health or _dataset_health(cases, regression_meaningful=False)
    under_tested_categories = len(health_data.get("low_n_intents", []))

    with _paper_style():
        fig = plt.figure(figsize=(12.2, 5.5), constrained_layout=False)
        gs = fig.add_gridspec(
            1, 2, width_ratios=[0.9, 1.95], wspace=0.05,
            left=0.04, right=0.97, top=0.9, bottom=0.12,
        )
        ax_donut = fig.add_subplot(gs[0])
        ax_text = fig.add_subplot(gs[1])

        sizes = [passed, failed] if total else [1, 0]
        colors = [_COLOR_GOOD, _COLOR_BAD]
        ax_donut.pie(
            sizes,
            colors=colors,
            startangle=90,
            counterclock=False,
            wedgeprops={"width": 0.34, "edgecolor": "white", "linewidth": 2},
        )
        ax_donut.text(
            0,
            0.06,
            f"{pass_rate * 100:.0f}%",
            ha="center",
            va="center",
            fontsize=26,
            fontweight="bold",
            color=_COLOR_TEXT,
        )
        ax_donut.text(
            0,
            -0.22,
            "passed",
            ha="center",
            va="center",
            fontsize=12,
            color=_COLOR_TEXT_SOFT,
        )
        ax_donut.set_aspect("equal")
        ax_donut.grid(False)
        ax_donut.axis("off")

        legend_handles = [
            plt.Line2D(
                [0], [0], marker="s", linestyle="",
                markerfacecolor=_COLOR_GOOD, markeredgecolor=_COLOR_GOOD,
                markersize=11, label=f"Passed ({passed})",
            ),
            plt.Line2D(
                [0], [0], marker="s", linestyle="",
                markerfacecolor=_COLOR_BAD, markeredgecolor=_COLOR_BAD,
                markersize=11, label=f"Failed ({failed})",
            ),
        ]
        ax_donut.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=2,
            frameon=False,
            fontsize=10,
        )

        ax_text.axis("off")
        ax_text.set_xlim(0, 1)
        ax_text.set_ylim(0, 1)

        rows = [
            ("Total questions", str(total)),
            ("Passed", f"{passed}"),
            ("Failed", f"{failed}"),
            ("Mean score (0–1)", f"{mean_score:.3f}"),
            ("Pass rate", f"{pass_rate * 100:.1f}%"),
            ("Under-tested categories (n<5)", str(under_tested_categories)),
            ("Strongest capability", strongest),
            ("Weakest capability", weakest),
        ]
        for i, (label, value) in enumerate(rows):
            yp = 0.94 - i * 0.11
            ax_text.text(
                0.04, yp, label,
                ha="left", va="center",
                fontsize=11, color=_COLOR_TEXT_SOFT,
            )
            ax_text.text(
                0.96, yp, value,
                ha="right", va="center",
                fontsize=13 if i >= 6 else 16,
                fontweight="bold",
                color=_COLOR_TEXT,
            )

        if total < _SMALL_DATASET_THRESHOLD:
            fig.text(
                0.5,
                0.965,
                (
                    f"Small evaluation set (n={total}): use this as a debugging snapshot, "
                    "not a final benchmark."
                ),
                ha="center",
                va="center",
                fontsize=10.5,
                color="#523f00",
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": "#FFF3BF",
                    "edgecolor": "#B08900",
                    "linewidth": 1.0,
                },
            )

        ax_text.text(
            0.04, 0.02,
            f"Run: {run.run_name}   ·   Dataset: {run.dataset_name}   ·   "
            f"Branch: {run.branch_name or '?'}   ·   Commit: {git_short or '?'}",
            ha="left", va="bottom",
            fontsize=9, color=_COLOR_TEXT_MUTED, style="italic",
        )

        fig.suptitle("Evaluation Run Scorecard", fontsize=14, fontweight="bold")
        _figure_caption(
            fig,
            "This scorecard summarizes pass/fail totals, mean score, and which capabilities are strongest or weakest. "
            "It is intended for debugging triage and prioritization. "
            f"{_dataset_size_note(total)} {_LOW_N_CAPTION_SUFFIX}",
        )
        fig.savefig(out_dir / "evaluation_summary.png")
        plt.close(fig)


def _leakage_rate(cases: list[EvaluationCaseResult]) -> float:
    """Fraction of cases tagged with a retrieval-leakage failure."""
    from app.eval.case_result_tags import canonical_failure_tags_for_row

    if not cases:
        return 0.0
    n = sum(
        1 for r in cases
        if "retrieval_leakage" in canonical_failure_tags_for_row(r)
    )
    return n / len(cases)


def _mode_misroute_rate(cases: list[EvaluationCaseResult]) -> float:
    """Fraction of cases tagged with a mode misroute / misclassification."""
    from app.eval.case_result_tags import canonical_failure_tags_for_row

    if not cases:
        return 0.0
    misroute_tags = {"mode_misclassification", "mode_routing_failure"}
    n = sum(
        1 for r in cases
        if set(canonical_failure_tags_for_row(r)) & misroute_tags
    )
    return n / len(cases)


def _regression_metrics(
    prev_run: EvaluationRun,
    prev_cases: list[EvaluationCaseResult],
    curr_run: EvaluationRun,
    curr_cases: list[EvaluationCaseResult],
) -> list[tuple[str, float, float, str]]:
    return [
        (
            "Mean score\n(higher is better)",
            float(prev_run.overall_score or 0.0),
            float(curr_run.overall_score or 0.0),
            "higher",
        ),
        (
            "Retrieval leakage\n(lower is better)",
            _leakage_rate(prev_cases),
            _leakage_rate(curr_cases),
            "lower",
        ),
        (
            "Mode misroute rate\n(lower is better)",
            _mode_misroute_rate(prev_cases),
            _mode_misroute_rate(curr_cases),
            "lower",
        ),
    ]


def _regression_has_movement(
    metrics: list[tuple[str, float, float, str]], *, eps: float = 1e-6
) -> bool:
    return any(abs(curr - prev) > eps for _, prev, curr, _ in metrics)


def write_regression_comparison_chart(
    prev_run: EvaluationRun,
    prev_cases: list[EvaluationCaseResult],
    curr_run: EvaluationRun,
    curr_cases: list[EvaluationCaseResult],
    out_dir: Path,
) -> None:
    """Side-by-side previous-vs-current bars on the regression metrics.

    Mirrors ``regression_report.md`` (mean overall score, retrieval leakage
    share, mode misroute share) but with directional color coding so a
    reader can see at a glance whether each metric improved.
    """
    metrics = _regression_metrics(prev_run, prev_cases, curr_run, curr_cases)

    def _direction_color(prev_v: float, curr_v: float, direction: str) -> str:
        eps = 1e-6
        if abs(curr_v - prev_v) < eps:
            return _COLOR_NEUTRAL_BAR
        improved = (curr_v > prev_v) if direction == "higher" else (curr_v < prev_v)
        return _COLOR_GOOD if improved else _COLOR_BAD

    with _paper_style():
        if not _regression_has_movement(metrics):
            fig, ax = plt.subplots(figsize=(10.0, 4.6))
            ax.axis("off")
            lines = [
                "No regression movement detected between current and previous run.",
                "",
            ]
            for label, prev_v, curr_v, _ in metrics:
                lines.append(f"{label.replace(chr(10), ' ')}: prev={prev_v:.3f}, curr={curr_v:.3f}")
            ax.text(
                0.5,
                0.58,
                "\n".join(lines),
                ha="center",
                va="center",
                fontsize=11,
                color=_COLOR_TEXT,
            )
            ax.set_title("Regression Check", fontsize=14, fontweight="bold")
            _figure_caption(
                fig,
                "Regression chart omitted because all tracked metrics are unchanged. "
                "This prevents visual noise from implying movement where none occurred. "
                f"{_dataset_size_note(len(curr_cases))}",
            )
            fig.tight_layout(rect=(0, 0.06, 1, 1))
            fig.savefig(out_dir / "regression_comparison.png")
            plt.close(fig)
            return

        fig, ax = plt.subplots(figsize=(10.0, 5.4))
        xs = list(range(len(metrics)))
        width = 0.36

        prev_vals = [m[1] for m in metrics]
        curr_vals = [m[2] for m in metrics]
        curr_colors = [_direction_color(m[1], m[2], m[3]) for m in metrics]

        prev_bars = ax.bar(
            [x - width / 2 for x in xs],
            prev_vals,
            width=width,
            color=_COLOR_NEUTRAL_BAR,
            edgecolor="white",
            linewidth=0.8,
            label=f"Previous run (id={prev_run.id})",
            zorder=3,
        )
        curr_bars = ax.bar(
            [x + width / 2 for x in xs],
            curr_vals,
            width=width,
            color=curr_colors,
            edgecolor="white",
            linewidth=0.8,
            label=f"Current run (id={curr_run.id})",
            zorder=3,
        )

        ax.set_xticks(xs)
        ax.set_xticklabels([m[0] for m in metrics])
        ax.set_ylim(0, max(1.05, max(prev_vals + curr_vals + [0.0]) * 1.2 + 0.05))
        ax.set_ylabel("Metric value (0–1)")
        ax.set_axisbelow(True)
        ax.xaxis.grid(False)
        ax.set_title("Regression Check: Previous vs Current Run")

        for bar, val in list(zip(prev_bars, prev_vals)) + list(zip(curr_bars, curr_vals)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#222",
            )

        legend_handles = [
            plt.Line2D([0], [0], marker="s", linestyle="",
                       markerfacecolor=_COLOR_NEUTRAL_BAR, markeredgecolor=_COLOR_NEUTRAL_BAR,
                       markersize=11, label=f"Previous run (id={prev_run.id})"),
            plt.Line2D([0], [0], marker="s", linestyle="",
                       markerfacecolor=_COLOR_GOOD, markeredgecolor=_COLOR_GOOD,
                       markersize=11, label="Current — improved"),
            plt.Line2D([0], [0], marker="s", linestyle="",
                       markerfacecolor=_COLOR_NEUTRAL_BAR, markeredgecolor=_COLOR_NEUTRAL_BAR,
                       markersize=11, label="Current — unchanged"),
            plt.Line2D([0], [0], marker="s", linestyle="",
                       markerfacecolor=_COLOR_BAD, markeredgecolor=_COLOR_BAD,
                       markersize=11, label="Current — regressed"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", ncol=2)

        _figure_caption(
            fig,
            "Same numbers as regression_report.md, color-coded so readers can see direction at a glance. "
            "Mean score should rise; leakage and misroute rates should fall. "
            f"{_dataset_size_note(len(curr_cases))}",
        )
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(out_dir / "regression_comparison.png")
        plt.close(fig)


def write_report_dashboard_chart(
    cases: list[EvaluationCaseResult],
    out_dir: Path,
    *,
    health: dict[str, Any] | None = None,
) -> None:
    """Three-panel summary: structure, retrieval diagnostics, and dataset health."""
    structure = summarize_structure(cases)
    by_intent = structure.get("by_intent", {})
    intent_rows: list[tuple[str, int, float]] = []
    for intent, stats in by_intent.items():
        intent_rows.append(
            (
                intent,
                int(stats.get("total_cases", 0)),
                float(stats.get("compliance_rate", 0.0)),
            )
        )
    intent_rows.sort(key=lambda item: item[1], reverse=True)
    intents = [row[0] for row in intent_rows]
    intent_n = [row[1] for row in intent_rows]
    compliance = [row[2] for row in intent_rows]

    retrieval = summarize_retrieval(cases)
    retrieval_metrics = [
        ("First retrieved\nchunk is correct", float(retrieval.get("top_1_accuracy", 0.0)), "higher"),
        ("Correct chunk in\ntop-k retrieved", float(retrieval.get("top_k_recall", 0.0)), "higher"),
        ("Retrieval noise\n(off-topic chunks)", float(retrieval.get("retrieval_noise_rate", 0.0)), "lower"),
    ]
    health_data = health or _dataset_health(cases, regression_meaningful=False)

    with _paper_style():
        fig, (ax_struct, ax_ret, ax_health) = plt.subplots(
            1, 3, figsize=(17.0, 5.8), gridspec_kw={"width_ratios": [1.35, 1.15, 1.1]}
        )

        # Left: structure compliance per intent
        sx = list(range(len(intents)))
        struct_bars = ax_struct.bar(
            sx,
            compliance,
            color=_COLOR_PRIMARY,
            edgecolor="white",
            linewidth=0.8,
            width=0.62,
            zorder=3,
        )
        for bar, n in zip(struct_bars, intent_n):
            bar.set_alpha(_fade_alpha(n))
        ax_struct.axhline(0, color=_COLOR_NEUTRAL_LINE, linewidth=0.6, zorder=2)
        ax_struct.set_xticks(sx)
        ax_struct.set_xticklabels(
            [_intent_n_label(i, n) for i, n in zip(intents, intent_n)]
        )
        ax_struct.set_ylim(0, 1.08)
        ax_struct.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax_struct.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax_struct.set_ylabel("Share of answers in expected format")
        ax_struct.set_title("Answer-Structure Compliance by Question Type")
        ax_struct.set_axisbelow(True)
        ax_struct.xaxis.grid(False)
        for bar, v, n in zip(struct_bars, compliance, intent_n):
            warning = _low_n_warning(n)
            text = f"{v * 100:.0f}%" if not warning else f"{v * 100:.0f}%\n{warning}"
            ax_struct.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.025,
                text,
                ha="center", va="bottom", fontsize=8.7, color="#222",
            )

        # Right: retrieval diagnostics
        rx = list(range(len(retrieval_metrics)))
        ret_colors = [_COLOR_PRIMARY, _COLOR_PRIMARY, _COLOR_SECONDARY]
        ret_bars = ax_ret.bar(
            rx,
            [m[1] for m in retrieval_metrics],
            color=ret_colors,
            edgecolor="white",
            linewidth=0.8,
            width=0.62,
            zorder=3,
        )
        ax_ret.axhline(0, color=_COLOR_NEUTRAL_LINE, linewidth=0.6, zorder=2)
        ax_ret.set_xticks(rx)
        ax_ret.set_xticklabels([m[0] for m in retrieval_metrics])
        ax_ret.set_ylim(0, 1.08)
        ax_ret.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax_ret.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax_ret.set_ylabel("Share of cases")
        ax_ret.set_title("Overall Retrieval Diagnostics")
        ax_ret.set_axisbelow(True)
        ax_ret.xaxis.grid(False)
        for bar, (_, v, _) in zip(ret_bars, retrieval_metrics):
            ax_ret.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.025,
                f"{v * 100:.0f}%",
                ha="center", va="bottom", fontsize=9, color="#222",
            )

        ax_health.axis("off")
        ax_health.set_title("Dataset Health", fontsize=12, fontweight="bold")
        under_tested_concepts = list(health_data.get("under_tested_concepts", []))
        low_n_intents = list(health_data.get("low_n_intents", []))
        concept_preview = ", ".join(under_tested_concepts[:4])
        if len(under_tested_concepts) > 4:
            concept_preview += ", ..."
        if not concept_preview:
            concept_preview = "(none)"
        intent_preview = ", ".join(_intent_label_csv(i) for i in low_n_intents[:3])
        if len(low_n_intents) > 3:
            intent_preview += ", ..."
        if not intent_preview:
            intent_preview = "(none)"
        health_lines = [
            f"Total cases: {health_data.get('total_cases', len(cases))}",
            f"Under-tested concepts: {len(under_tested_concepts)}",
            f"  {concept_preview}",
            f"Categories with n<{_LOW_N_THRESHOLD}: {len(low_n_intents)}",
            f"  {intent_preview}",
            (
                "Regression chart meaningful: "
                f"{'yes' if health_data.get('regression_meaningful') else 'no'}"
            ),
            (
                "Paired boost data exists: "
                f"{'yes' if health_data.get('paired_boost_present') else 'no'}"
            ),
        ]
        ax_health.text(
            0.02,
            0.98,
            "\n".join(health_lines),
            ha="left",
            va="top",
            fontsize=10,
            color=_COLOR_TEXT,
            linespacing=1.35,
            transform=ax_health.transAxes,
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "#F7FAFC",
                "edgecolor": "#CBD5E0",
                "linewidth": 0.9,
            },
        )

        fig.suptitle("Evaluation Report at a Glance", fontsize=14, fontweight="bold")
        _figure_caption(
            fig,
            "Left: how often the tutor's answer for each question type came back in the expected structure "
            "(higher is better).\nRight: aggregate retrieval diagnostics across all questions. "
            "Blue = higher is better, orange = lower is better. "
            f"{_dataset_size_note(len(cases))} {_LOW_N_CAPTION_SUFFIX}",
        )
        fig.tight_layout(rect=(0, 0.06, 1, 0.94))
        fig.savefig(out_dir / "report_dashboard.png")
        plt.close(fig)


def write_coverage_by_concept_chart(
    cases: list[EvaluationCaseResult], out_dir: Path
) -> None:
    """Stacked pass/fail coverage by concept."""
    concept_rows: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "failed": 0})
    for row in cases:
        concept = _case_concept(row)
        if row.pass_bool:
            concept_rows[concept]["passed"] += 1
        else:
            concept_rows[concept]["failed"] += 1
    ordered = sorted(
        concept_rows.items(),
        key=lambda item: item[1]["passed"] + item[1]["failed"],
        reverse=True,
    )

    with _paper_style():
        fig, ax = plt.subplots(figsize=(10.8, 0.62 * max(4, len(ordered)) + 1.8))
        if not ordered:
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                "No cases available",
                ha="center",
                va="center",
                fontsize=11,
                color=_COLOR_TEXT_MUTED,
                style="italic",
                transform=ax.transAxes,
            )
        else:
            concepts = [name for name, _ in ordered]
            passed = [stats["passed"] for _, stats in ordered]
            failed = [stats["failed"] for _, stats in ordered]
            totals = [p + f for p, f in zip(passed, failed)]
            ys = list(range(len(concepts)))
            pass_bars = ax.barh(
                ys,
                passed,
                color=_COLOR_GOOD,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
                label="Passed",
            )
            fail_bars = ax.barh(
                ys,
                failed,
                left=passed,
                color=_COLOR_BAD,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
                label="Failed",
            )
            for pb, fb, n in zip(pass_bars, fail_bars, totals):
                pb.set_alpha(_fade_alpha(n))
                fb.set_alpha(_fade_alpha(n))

            ax.set_yticks(ys)
            ax.set_yticklabels(concepts)
            ax.invert_yaxis()
            ax.set_xlabel("Number of evaluation cases")
            ax.set_title("Coverage by Concept (Pass/Fail Counts)")
            ax.set_axisbelow(True)
            ax.yaxis.grid(False)
            for y, n in zip(ys, totals):
                warning = _low_n_warning(n)
                label = f"n={n}" if not warning else f"n={n} ({warning})"
                ax.text(
                    n + 0.08,
                    y,
                    label,
                    va="center",
                    ha="left",
                    fontsize=9,
                    color=_COLOR_TEXT,
                )
            ax.legend(loc="lower right")

        _figure_caption(
            fig,
            "This chart measures per-concept evaluation coverage and outcomes (passed vs failed). "
            "It highlights whether quality claims are supported by enough examples per concept. "
            f"{_dataset_size_note(len(cases))} {_LOW_N_CAPTION_SUFFIX}",
        )
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(out_dir / "coverage_by_concept.png")
        plt.close(fig)


def write_failure_modes_chart(
    cases: list[EvaluationCaseResult], out_dir: Path
) -> None:
    """Failure-mode counts with one example test ID per mode."""
    from app.eval.case_result_tags import canonical_failure_tags_for_row

    priority = [
        "missing_required_concept",
        "compare_asymmetry",
        "compare_entity_collapse",
        "validation_missed_error",
    ]

    counts: Counter[str] = Counter()
    first_example: dict[str, str] = {}
    for row in cases:
        if row.pass_bool:
            continue
        tags = set(canonical_failure_tags_for_row(row))
        primary = primary_error_type_for_row(row)
        if primary:
            tags.add(primary)
        if not tags:
            tags.add("unknown_failure")
        for tag in tags:
            counts[tag] += 1
            if tag not in first_example:
                first_example[tag] = row.test_id or "(missing test_id)"

    ordered_keys = [tag for tag in priority if tag in counts]
    ordered_keys.extend(
        sorted((tag for tag in counts if tag not in priority), key=lambda tag: counts[tag], reverse=True)
    )

    with _paper_style():
        fig, ax = plt.subplots(figsize=(11.0, 0.6 * max(4, len(ordered_keys)) + 1.8))
        if not ordered_keys:
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                "No failed cases in this run",
                ha="center",
                va="center",
                fontsize=11,
                color=_COLOR_TEXT_MUTED,
                style="italic",
                transform=ax.transAxes,
            )
        else:
            ys = list(range(len(ordered_keys)))
            vals = [counts[k] for k in ordered_keys]
            bars = ax.barh(
                ys,
                vals,
                color=_COLOR_BAD,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            ax.set_yticks(ys)
            ax.set_yticklabels(ordered_keys)
            ax.invert_yaxis()
            ax.set_xlabel("Count of failed cases")
            ax.set_title("Failure Modes and Example Test IDs")
            ax.set_axisbelow(True)
            ax.yaxis.grid(False)
            for bar, key in zip(bars, ordered_keys):
                ax.text(
                    bar.get_width() + 0.08,
                    bar.get_y() + bar.get_height() / 2,
                    f"{int(bar.get_width())}  (e.g. {first_example.get(key, '?')})",
                    va="center",
                    ha="left",
                    fontsize=9,
                    color=_COLOR_TEXT,
                )

        _figure_caption(
            fig,
            "This chart counts concrete failure modes and links each to an example test case ID. "
            "It is intended to prioritize debugging targets instead of over-reading aggregate pass rates. "
            f"{_dataset_size_note(len(cases))}",
        )
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(out_dir / "failure_modes.png")
        plt.close(fig)


# ---------------------------------------------------------------------------
# CSV generators
# ---------------------------------------------------------------------------


# Short columns first, long answer last; multiline answers flattened so each
# record is a single physical row (readable in cat/Excel).
_EXAMPLE_COLUMNS = [
    "test_id",
    "score_group",
    "rank_in_group",
    "passed_scoring",
    "query_type_label",
    "grounding",
    "notes_or_error_type",
    "user_query",
    "course_answer_one_line",
]


def _example_row(
    row: EvaluationCaseResult,
    *,
    score_group: str,
    rank_in_group: int,
) -> dict[str, Any]:
    scores = derive_case_scores(row)
    grounding = "pass" if scores["grounding"] == 1 else "fail"
    notes = "ok" if row.pass_bool else (row.primary_error_type or "ok")
    return {
        "test_id": row.test_id or "",
        "score_group": score_group,
        "rank_in_group": str(rank_in_group),
        "passed_scoring": "yes" if row.pass_bool else "no",
        "query_type_label": _intent_label_csv(_row_intent(row)),
        "grounding": grounding,
        "notes_or_error_type": notes,
        "user_query": (row.query_text or "").replace("\n", " ").strip(),
        "course_answer_one_line": _flatten_answer_for_csv(row.actual_response or ""),
    }


def write_example_answers_csv(
    cases: list[EvaluationCaseResult], out_dir: Path
) -> None:
    sorted_cases = sorted(cases, key=lambda r: (r.score or 0.0), reverse=True)
    if len(sorted_cases) >= 10:
        top = sorted_cases[:5]
        bottom = sorted_cases[-5:]
        sequenced: list[tuple[EvaluationCaseResult, str, int]] = []
        for i, r in enumerate(top, start=1):
            sequenced.append((r, "top_5_by_score", i))
        for i, r in enumerate(bottom, start=1):
            sequenced.append((r, "bottom_5_by_score", i))
    else:
        sequenced = [
            (r, "full_run_sample", i) for i, r in enumerate(sorted_cases, start=1)
        ]

    path = out_dir / "example_answers.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_EXAMPLE_COLUMNS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for rec, group, rank in sequenced:
            writer.writerow(_example_row(rec, score_group=group, rank_in_group=rank))


_ERROR_COLUMNS = ["error_type", "count", "percentage", "example_query"]


def write_error_analysis_csv(
    cases: list[EvaluationCaseResult], out_dir: Path
) -> None:
    summary = summarize_errors(cases)
    by_error_type: dict[str, dict[str, Any]] = summary.get("by_error_type", {})
    failed_cases = [row for row in cases if not row.pass_bool]

    path = out_dir / "error_analysis.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_ERROR_COLUMNS)
        writer.writeheader()
        for error_type, data in by_error_type.items():
            example_query = ""
            for row in failed_cases:
                if primary_error_type_for_row(row) == error_type:
                    example_query = row.query_text or ""
                    break
            writer.writerow(
                {
                    "error_type": error_type,
                    "count": int(data.get("count", 0)),
                    "percentage": float(data.get("percentage", 0.0)),
                    "example_query": example_query,
                }
            )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _resolve_previous_run(
    current_run: EvaluationRun,
) -> tuple[EvaluationRun, list[EvaluationCaseResult]] | None:
    """Find the most recent prior run on the same dataset, with its cases."""
    from sqlalchemy import asc

    prev = (
        EvaluationRun.query.filter(
            EvaluationRun.dataset_name == current_run.dataset_name,
            EvaluationRun.id != current_run.id,
        )
        .order_by(EvaluationRun.created_at.desc())
        .first()
    )
    if prev is None:
        return None
    prev_cases = (
        EvaluationCaseResult.query.filter_by(evaluation_run_id=prev.id)
        .order_by(asc(EvaluationCaseResult.test_id))
        .all()
    )
    return prev, prev_cases


def generate_evaluation_outputs(
    cases: list[EvaluationCaseResult],
    out_dir: Path,
    current_run: EvaluationRun | None = None,
) -> None:
    """Generate all evaluation artifacts; chart failures never crash the run.

    ``current_run`` is optional purely for backwards compatibility with
    older callers (and the existing test suite). When provided, the
    summary scorecard and (if a prior run exists on the same dataset) the
    regression-comparison chart are written too.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    prev_pair: tuple[EvaluationRun, list[EvaluationCaseResult]] | None = None
    regression_meaningful = False
    if current_run is not None:
        prev_pair = _resolve_previous_run(current_run)
        if prev_pair is not None:
            prev_run, prev_cases = prev_pair
            metrics = _regression_metrics(prev_run, prev_cases, current_run, cases)
            regression_meaningful = _regression_has_movement(metrics)
    health = _dataset_health(cases, regression_meaningful=regression_meaningful)

    generators: list[tuple[str, Any]] = [
        ("retrieval_accuracy.png", lambda: write_retrieval_accuracy_chart(cases, out_dir)),
        ("question_type_breakdown.png", lambda: write_question_type_chart(cases, out_dir)),
        ("pipeline_diagram.png", lambda: write_pipeline_diagram(out_dir)),
        ("report_dashboard.png", lambda: write_report_dashboard_chart(cases, out_dir, health=health)),
        ("coverage_by_concept.png", lambda: write_coverage_by_concept_chart(cases, out_dir)),
        ("failure_modes.png", lambda: write_failure_modes_chart(cases, out_dir)),
        ("example_answers.csv", lambda: write_example_answers_csv(cases, out_dir)),
        ("error_analysis.csv", lambda: write_error_analysis_csv(cases, out_dir)),
    ]
    if current_run is not None:
        generators.append(
            (
                "evaluation_summary.png",
                lambda: write_evaluation_summary_chart(current_run, cases, out_dir, health=health),
            )
        )
        if prev_pair is not None:
            prev_run, prev_cases = prev_pair
            generators.append(
                (
                    "regression_comparison.png",
                    lambda: write_regression_comparison_chart(
                        prev_run, prev_cases, current_run, cases, out_dir
                    ),
                )
            )

    for name, fn in generators:
        try:
            fn()
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[evaluation_outputs] failed to write {name}: {exc!r}")

    print(f"Wrote evaluation outputs to {out_dir}")
