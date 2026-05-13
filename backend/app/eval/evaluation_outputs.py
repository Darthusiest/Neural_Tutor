"""Generate visual + tabular artifacts (PNG charts, CSVs) from persisted eval rows.

Called once per eval run, after the capability report is written. Outputs to
``evaluation_outputs/`` at the repo root. All scoring derives from existing
persisted columns on :class:`EvaluationCaseResult` (no schema change).

    
Chart style is **debug-first** (spot regressions, avoid over-claiming): compact
titles, at most one footer line with dataset size, no per-bar ``low n`` / ``n
too small`` callouts, full-opacity bars, and no deliberately faded bars. Figures
still use serif typography at 220 DPI, dashed grids, hidden top/right spines,
the Okabe-Ito palette, and plain-English intent labels where categories appear.

Eight PNGs may be produced in ``out_dir`` after a run (some are conditional on
suite size or prior runs):

1. ``pipeline_diagram.png`` — static architecture diagram of the answer
   pipeline (live turn + admin Gemini critic on eval batches).
2. ``question_type_breakdown.png`` — pass rate per query intent, sorted by
   sample count; bar labels include ``n``.
3. ``retrieval_accuracy.png`` — share of cases whose top-k retrieval
   contained the expected lecture chunk, for standard content-retrieval
   intents only (definition / fact lookup / multi-concept synthesis).
4. ``evaluation_summary.png`` — dashboard-style pass/fail summary with donut,
   stat tiles, dataset line, and small-``n`` caution when ``n < 50``.
5. ``regression_comparison.png`` — previous vs current run on three
   regression metrics, or a single-line notice when no metric changed.
6. ``report_dashboard.png`` — compact debug dashboard: structure compliance
   (short intent labels on the breakdown) + retrieval bars; title + “for debugging only” subtitle.
7. ``coverage_by_concept.png`` — only when ``total_cases >= 30``: side-by-side
   pass/fail counts per concept (not stacked); omitted otherwise and any stale
   file in ``out_dir`` is removed.
8. ``failure_modes.png`` — failure-type counts (sorted by frequency; details in ``error_analysis.csv``).

Two CSVs are also produced for analysis:

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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

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
_COLOR_CARD_BG = "#F8FAFC"
_COLOR_CARD_EDGE = "#E2E8F0"
_COLOR_STAT_TILE_EDGE = "#CBD5E0"
_COLOR_CAUTION_BG = "#FFFBEB"
_COLOR_CAUTION_EDGE = "#D97706"
_COLOR_CAUTION_TITLE = "#78350F"
_COLOR_CAUTION_NOTE = "#92400E"

_FIG_DPI = 220  # crisp for print without bloating file size

# Evaluation summary donut: ring thickness as a fraction of pie radius. Lower value
# yields a wider inner hole so center labels are not cramped against the ring.
_EVAL_SUMMARY_DONUT_RING_WIDTH = 0.22

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

# Per-concept coverage chart is omitted below this total suite size (misleading buckets).
_COVERAGE_BY_CONCEPT_MIN_TOTAL_CASES = 30

_NORMAL_RETRIEVAL_INTENTS = {"definition", "retrieval_grounded", "synthesis"}
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


def _debug_dataset_footer(fig: Any, n: int, *, y: float = 0.012) -> None:
    """Single bottom line: consistent small-set wording + total ``n``."""
    if n < 0:
        return
    line = (
        f"Small evaluation set (n={n})"
        if n < _SMALL_DATASET_THRESHOLD
        else f"Evaluation set (n={n})"
    )
    fig.text(
        0.5,
        y,
        line,
        ha="center",
        va="bottom",
        fontsize=9,
        color=_COLOR_TEXT_MUTED,
        style="italic",
    )


def _bucket_rate_label(rate: float, n: int) -> str:
    """Bar annotation: a lone ``100%`` with ``n==1`` over-claims; use ``n=1`` instead."""
    if n <= 0:
        return "—"
    if n == 1 and rate >= 1.0 - 1e-9:
        return "n=1"
    if n == 1:
        return f"{rate * 100:.0f}%"
    return f"{rate * 100:.0f}% (n={n})"


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
    """Bar chart: top-k chunk hit rate for standard content-retrieval query types only.

    Includes ``definition``, ``retrieval_grounded``, and ``synthesis`` when the
    query is not underspecified. Omits compare / step-by-step / quiz / summary
    routing and underspecified prompts so the figure does not mix retrieval
    metrics with mode or clarification behavior.
    """
    grouped: dict[str, list[EvaluationCaseResult]] = defaultdict(list)
    for row in cases:
        intent = _row_intent(row)
        if intent not in _NORMAL_RETRIEVAL_INTENTS:
            continue
        if _is_underspecified_query(row.query_text):
            continue
        grouped[intent].append(row)

    total_eval_cases = len(cases)

    with _paper_style():
        fig, ax = plt.subplots(figsize=(8.2, 5.4))

        rows: list[tuple[str, float, int]] = []
        for intent, bucket in grouped.items():
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
                "No qualifying content-retrieval cases",
                ha="center",
                va="center",
                fontsize=11,
                color=_COLOR_TEXT_MUTED,
                style="italic",
                transform=ax.transAxes,
            )
        else:
            xs = list(range(len(rows)))
            bars = ax.bar(
                xs,
                [row[1] for row in rows],
                color=_COLOR_PRIMARY,
                edgecolor="white",
                linewidth=0.8,
                width=0.62,
                zorder=3,
            )
            ax.axhline(0, color=_COLOR_NEUTRAL_LINE, linewidth=0.6, zorder=2)
            ax.set_xticks(xs)
            ax.set_xticklabels([_format_intent(intent) for intent, _, _ in rows])
            ax.set_ylim(0, 1.08)
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
            ax.set_axisbelow(True)
            ax.xaxis.grid(False)
            for bar, (_, acc, n_bucket) in zip(bars, rows):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    _bucket_rate_label(acc, n_bucket),
                    ha="center",
                    va="bottom",
                    fontsize=9.5,
                    color="#222",
                )
            ax.set_ylabel("Share with correct chunk in top-k retrieval")

        fig.suptitle(
            "Retrieval: top-k hit rate (debug)",
            fontsize=13,
            fontweight="bold",
            color=_COLOR_TEXT,
        )
        _debug_dataset_footer(fig, total_eval_cases)
        fig.tight_layout(rect=(0, 0.065, 1, 0.92))
        fig.savefig(out_dir / "retrieval_accuracy.png")
        plt.close(fig)


def write_question_type_chart(
    cases: list[EvaluationCaseResult], out_dir: Path
) -> None:
    """Pass rate (scoring rubric) per query intent, sorted by sample count (desc)."""
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

    n_total = len(cases)

    with _paper_style():
        fig, ax = plt.subplots(figsize=(10.0, 5.5))

        if not rows:
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                "No cases",
                ha="center",
                va="center",
                fontsize=11,
                color=_COLOR_TEXT_MUTED,
                style="italic",
                transform=ax.transAxes,
            )
        else:
            xs = list(range(len(rows)))
            intents = [row[0] for row in rows]
            totals = [row[1] for row in rows]
            accuracies = [row[2] for row in rows]

            bars = ax.bar(
                xs,
                accuracies,
                color=_COLOR_PRIMARY,
                edgecolor="white",
                linewidth=0.8,
                width=0.62,
                zorder=3,
            )
            ax.axhline(0, color=_COLOR_NEUTRAL_LINE, linewidth=0.6, zorder=2)
            ax.set_xticks(xs)
            ax.set_xticklabels([_format_intent(i) for i in intents])
            ax.set_ylim(0, 1.08)
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
            ax.set_ylabel("Pass rate")
            ax.set_axisbelow(True)
            ax.xaxis.grid(False)

            for bar, acc, n in zip(bars, accuracies, totals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    _bucket_rate_label(acc, n),
                    ha="center",
                    va="bottom",
                    fontsize=9.5,
                    color="#222",
                )

        fig.suptitle(
            "Pass rate by question type (debug)",
            fontsize=13,
            fontweight="bold",
            color=_COLOR_TEXT,
        )

        _debug_dataset_footer(fig, n_total)
        fig.tight_layout(rect=(0, 0.065, 1, 0.9))
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
    (
        "Gemini Critic\n(admin / eval)",
        "LLM judge on stored\nbatch runs; not live gating",
    ),
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
        # Seven stages need a slightly wider canvas cap than the original six-box row.
        fig, ax = plt.subplots(figsize=(min(20.5, total_w + 1.6), 4.15))

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
            "Answer pipeline (reference)",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=_COLOR_TEXT,
        )
        ax.text(
            total_w / 2,
            y - 0.42,
            "Schematic only (not eval metrics). Critic is optional; invoked from admin on "
            "persisted eval runs (does not block live chat).",
            ha="center",
            va="center",
            fontsize=8.5,
            color=_COLOR_TEXT_MUTED,
            style="italic",
        )

        fig.savefig(out_dir / "pipeline_diagram.png")
        plt.close(fig)


def _summary_stat_grid(ax: Any, stats: list[tuple[str, str]]) -> None:
    """Four metric tiles in a 2×2 grid (axes coordinates)."""
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    positions = [
        (0.02, 0.52, 0.45, 0.46),
        (0.53, 0.52, 0.45, 0.46),
        (0.02, 0.02, 0.45, 0.46),
        (0.53, 0.02, 0.45, 0.46),
    ]
    for (x, y, w, h), (label, val) in zip(positions, stats):
        tile = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            transform=ax.transAxes,
            facecolor="#FFFFFF",
            edgecolor=_COLOR_STAT_TILE_EDGE,
            linewidth=1.0,
            zorder=1,
        )
        ax.add_patch(tile)
        ax.text(
            x + w / 2,
            y + h * 0.66,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9.5,
            color=_COLOR_TEXT_SOFT,
        )
        ax.text(
            x + w / 2,
            y + h * 0.34,
            val,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color=_COLOR_TEXT,
        )


def write_evaluation_summary_chart(
    run: EvaluationRun,
    cases: list[EvaluationCaseResult],
    out_dir: Path,
    *,
    health: dict[str, Any] | None = None,
    debug: bool = False,
) -> None:
    """Dashboard-style pass/fail summary for one run.

    ``health`` is accepted for backwards compatibility; it is not used.

    Set ``debug=True`` to append ``(debug)`` to the title (e.g. internal QA).
    """
    _ = health
    total = len(cases)
    passed = sum(1 for c in cases if c.pass_bool)
    failed = total - passed
    mean_score = float(run.overall_score) if run.overall_score is not None else 0.0
    pct_passed = int(round(100 * passed / total)) if total else 0

    title = "Evaluation summary (debug)" if debug else "Evaluation summary"
    stat_rows = [
        ("Total questions", str(total)),
        ("Passed", str(passed)),
        ("Failed", str(failed)),
        ("Mean score (0–1)", f"{mean_score:.2f}"),
    ]

    with _paper_style():
        fig = plt.figure(figsize=(10.0, 5.15))
        fig.patch.set_facecolor("white")

        outer = FancyBboxPatch(
            (0.045, 0.035),
            0.91,
            0.885,
            boxstyle="round,pad=0.006,rounding_size=0.02",
            transform=fig.transFigure,
            facecolor=_COLOR_CARD_BG,
            edgecolor=_COLOR_CARD_EDGE,
            linewidth=1.1,
            zorder=0,
        )
        fig.add_artist(outer)

        fig.text(
            0.065,
            0.91,
            title,
            ha="left",
            va="top",
            fontsize=15.5,
            fontweight="bold",
            color=_COLOR_TEXT,
            zorder=2,
        )

        gs = fig.add_gridspec(
            1,
            2,
            left=0.075,
            right=0.94,
            top=0.795,
            bottom=0.245,
            width_ratios=[1.12, 1.0],
            wspace=0.18,
        )
        ax_d = fig.add_subplot(gs[0, 0])
        ax_stats = fig.add_subplot(gs[0, 1])

        if total > 0:
            ax_d.pie(
                [passed, failed],
                colors=[_COLOR_GOOD, _COLOR_BAD],
                startangle=90,
                counterclock=False,
                radius=1.0,
                wedgeprops={
                    "width": _EVAL_SUMMARY_DONUT_RING_WIDTH,
                    "edgecolor": "white",
                    "linewidth": 2.0,
                },
            )
            ax_d.text(
                0,
                0.08,
                f"{pct_passed}% passed",
                ha="center",
                va="center",
                fontsize=20,
                fontweight="bold",
                color=_COLOR_TEXT,
            )
            ax_d.text(
                0,
                -0.26,
                f"{passed}/{total} questions",
                ha="center",
                va="center",
                fontsize=11,
                color=_COLOR_TEXT_SOFT,
            )
            ax_d.legend(
                handles=[
                    Patch(facecolor=_COLOR_GOOD, edgecolor="white", linewidth=0.8, label="Passed"),
                    Patch(facecolor=_COLOR_BAD, edgecolor="white", linewidth=0.8, label="Failed"),
                ],
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
                fontsize=10,
            )
        else:
            ax_d.text(
                0,
                0,
                "No cases",
                ha="center",
                va="center",
                fontsize=12,
                color=_COLOR_TEXT_MUTED,
                style="italic",
            )
        ax_d.set_aspect("equal")
        ax_d.axis("off")

        _summary_stat_grid(ax_stats, stat_rows)

        fig.text(
            0.065,
            0.198,
            f"Dataset: {run.dataset_name}",
            ha="left",
            va="top",
            fontsize=10,
            color=_COLOR_TEXT_MUTED,
            zorder=2,
        )

        if total < _SMALL_DATASET_THRESHOLD:
            bx, by, bw, bh = 0.36, 0.052, 0.575, 0.105
            badge = FancyBboxPatch(
                (bx, by),
                bw,
                bh,
                boxstyle="round,pad=0.012,rounding_size=0.014",
                transform=fig.transFigure,
                facecolor=_COLOR_CAUTION_BG,
                edgecolor=_COLOR_CAUTION_EDGE,
                linewidth=1.0,
                zorder=1,
            )
            fig.add_artist(badge)
            fig.text(
                bx + bw / 2,
                by + bh * 0.62,
                f"Small evaluation set (n={total})",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=_COLOR_CAUTION_TITLE,
            )
            fig.text(
                bx + bw / 2,
                by + bh * 0.30,
                "Interpret trends carefully with small n.",
                ha="center",
                va="center",
                fontsize=8.8,
                color=_COLOR_CAUTION_NOTE,
                style="italic",
            )
        else:
            fig.text(
                0.36,
                0.115,
                f"Evaluation set (n={total}). Estimates are descriptive, not inferential.",
                ha="left",
                va="center",
                fontsize=9,
                color=_COLOR_TEXT_MUTED,
                style="italic",
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
            fig, ax = plt.subplots(figsize=(8.0, 2.35))
            ax.axis("off")
            ax.text(
                0.5,
                0.55,
                "No measurable change between runs.",
                ha="center",
                va="center",
                fontsize=12,
                color=_COLOR_TEXT,
                transform=ax.transAxes,
            )
            _debug_dataset_footer(fig, len(curr_cases), y=0.04)
            fig.subplots_adjust(bottom=0.18)
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
        ax.set_title("Regression (debug)", fontsize=12)

        for bar, val in list(zip(prev_bars, prev_vals)) + list(zip(curr_bars, curr_vals)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#222",
            )

        legend_handles = [
            plt.Line2D([0], [0], marker="s", linestyle="",
                       markerfacecolor=_COLOR_NEUTRAL_BAR, markeredgecolor=_COLOR_NEUTRAL_BAR,
                       markersize=10, label=f"Previous (id={prev_run.id})"),
            plt.Line2D([0], [0], marker="s", linestyle="",
                       markerfacecolor=_COLOR_GOOD, markeredgecolor=_COLOR_GOOD,
                       markersize=10, label="Current — better"),
            plt.Line2D([0], [0], marker="s", linestyle="",
                       markerfacecolor=_COLOR_NEUTRAL_BAR, markeredgecolor=_COLOR_NEUTRAL_BAR,
                       markersize=10, label="Current — same"),
            plt.Line2D([0], [0], marker="s", linestyle="",
                       markerfacecolor=_COLOR_BAD, markeredgecolor=_COLOR_BAD,
                       markersize=10, label="Current — worse"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", ncol=2, fontsize=9)

        _debug_dataset_footer(fig, len(curr_cases))
        fig.tight_layout(rect=(0, 0.07, 1, 0.98))
        fig.savefig(out_dir / "regression_comparison.png")
        plt.close(fig)


# Shorter y-axis labels for report_dashboard structure breakdown (2 lines max).
_REPORT_DASHBOARD_INTENT_SHORT: dict[str, str] = {
    "definition": "Define a concept",
    "step_by_step": "Walkthrough / summary\n/ quiz",
    "compare": "Compare concepts",
    "retrieval_grounded": "Specific fact lookup",
    "synthesis": "Connect multiple concepts",
    "underspecified": "Underspecified\nprompt",
}


def _report_dashboard_intent_label(intent: str) -> str:
    if intent in _REPORT_DASHBOARD_INTENT_SHORT:
        return _REPORT_DASHBOARD_INTENT_SHORT[intent]
    return intent.replace("_", " ").strip().title()


def _report_dashboard_value_label(rate: float, n: int) -> str:
    """Consistent ``pct (n=k)`` for dashboard bars (including n=1)."""
    if n <= 0:
        return "—"
    return f"{rate * 100:.0f}% (n={n})"


def write_report_dashboard_chart(
    cases: list[EvaluationCaseResult],
    out_dir: Path,
) -> None:
    """Two-panel debug snapshot: aggregate structure compliance + retrieval diagnostics."""
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
    intent_ns = [row[1] for row in intent_rows]
    compliance = [row[2] for row in intent_rows]

    total = len(cases)
    compliant_total = sum(int(b.get("compliant_cases", 0)) for b in by_intent.values())
    overall_rate = (compliant_total / total) if total else 0.0

    retrieval = summarize_retrieval(cases)
    diags_all = retrieval_diagnostics(cases)
    evaluable_d = [d for d in diags_all if d.concept and d.concept != "unknown"]
    n_ret_eval = len(evaluable_d)
    n_ret_all = len(diags_all)
    retrieval_metrics = [
        ("First chunk\ncorrect", float(retrieval.get("top_1_accuracy", 0.0))),
        ("Correct chunk\nin top-k", float(retrieval.get("top_k_recall", 0.0))),
        ("Retrieval\nnoise", float(retrieval.get("retrieval_noise_rate", 0.0))),
    ]
    retrieval_bar_ns = (n_ret_eval, n_ret_eval, n_ret_all)

    with _paper_style():
        fig = plt.figure(figsize=(10.35, 5.28), layout="constrained")
        fig.text(
            0.5,
            0.965,
            f"Debug dashboard (n={total})",
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            color=_COLOR_TEXT,
        )
        fig.text(
            0.5,
            0.938,
            "For debugging only",
            ha="center",
            va="top",
            fontsize=8,
            color=_COLOR_TEXT_MUTED,
            style="italic",
        )

        gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.30)
        if len(intent_rows) <= 1:
            gs_left = gs[0].subgridspec(1, 1)
            ax_overall = fig.add_subplot(gs_left[0])
            ax_break = None
        else:
            gs_left = gs[0].subgridspec(2, 1, height_ratios=[1.02, 1.45], hspace=0.50)
            ax_overall = fig.add_subplot(gs_left[0])
            ax_break = fig.add_subplot(gs_left[1])
        ax_ret = fig.add_subplot(gs[1])

        ax_overall.axis("off")
        if total == 0:
            ax_overall.text(
                0.5,
                0.5,
                "—",
                fontsize=22,
                ha="center",
                va="center",
                color=_COLOR_TEXT_MUTED,
            )
        else:
            overall_lbl = (
                "n=1"
                if total == 1 and overall_rate >= 1.0 - 1e-9
                else (f"{overall_rate * 100:.0f}%" if total else "—")
            )
            ax_overall.text(
                0.5,
                0.62,
                overall_lbl,
                fontsize=24,
                ha="center",
                va="center",
                color=_COLOR_TEXT,
                fontweight="bold",
            )
            ax_overall.text(
                0.5,
                0.14,
                "Structure compliance (aggregated)",
                ha="center",
                va="center",
                fontsize=9,
                color=_COLOR_TEXT_SOFT,
            )

        if ax_break is not None:
            y = list(range(len(intents)))
            break_bars = ax_break.barh(
                y,
                compliance,
                color=_COLOR_PRIMARY,
                edgecolor="white",
                linewidth=0.65,
                height=0.58,
                zorder=3,
            )
            ax_break.set_yticks(y)
            ax_break.set_yticklabels(
                [_report_dashboard_intent_label(i) for i in intents],
                fontsize=7.8,
            )
            ax_break.tick_params(axis="y", which="major", pad=8, length=0)
            ax_break.invert_yaxis()
            ax_break.set_xlim(0, 1.14)
            ax_break.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax_break.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
            ax_break.set_xlabel("By question type", fontsize=8.5, color=_COLOR_TEXT_SOFT, labelpad=5)
            ax_break.set_axisbelow(True)
            ax_break.yaxis.grid(False)
            _ann_fs = 7.4
            _x_txt_pad = 0.014
            xmax_txt = 1.10
            for bar, v, n_i in zip(break_bars, compliance, intent_ns):
                tx = min(bar.get_width() + _x_txt_pad, xmax_txt)
                ax_break.text(
                    tx,
                    bar.get_y() + bar.get_height() / 2,
                    _report_dashboard_value_label(v, n_i),
                    va="center",
                    ha="left",
                    fontsize=_ann_fs,
                    color="#333333",
                )

        rx = list(range(len(retrieval_metrics)))
        ret_colors = [_COLOR_PRIMARY, _COLOR_PRIMARY, _COLOR_SECONDARY]
        ret_bars = ax_ret.bar(
            rx,
            [m[1] for m in retrieval_metrics],
            color=ret_colors,
            edgecolor="white",
            linewidth=0.75,
            width=0.58,
            zorder=3,
        )
        ax_ret.axhline(0, color=_COLOR_NEUTRAL_LINE, linewidth=0.6, zorder=2)
        ax_ret.set_xticks(rx)
        ax_ret.set_xticklabels(
            [m[0] for m in retrieval_metrics],
            fontsize=7.8,
            linespacing=0.95,
        )
        ax_ret.tick_params(axis="x", which="major", pad=7)
        ax_ret.set_ylim(0, 1.12)
        ax_ret.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax_ret.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
        ax_ret.set_ylabel("Share of cases", fontsize=9, color=_COLOR_TEXT_SOFT, labelpad=6)
        ax_ret.set_axisbelow(True)
        ax_ret.xaxis.grid(False)
        _ret_ann_fs = 7.2
        _ret_dy = 0.018
        for bar, (_, v), n_basis in zip(ret_bars, retrieval_metrics, retrieval_bar_ns):
            ax_ret.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + _ret_dy,
                _report_dashboard_value_label(v, n_basis),
                ha="center",
                va="bottom",
                fontsize=_ret_ann_fs,
                color="#333333",
            )

        fig.get_layout_engine().set(rect=(0, 0.03, 1, 0.905))
        fig.savefig(out_dir / "report_dashboard.png")
        plt.close(fig)


def write_coverage_by_concept_chart(
    cases: list[EvaluationCaseResult], out_dir: Path
) -> None:
    """Side-by-side pass/fail counts per concept; only if suite is large enough.

    Below :data:`_COVERAGE_BY_CONCEPT_MIN_TOTAL_CASES`, writes nothing and
    removes ``coverage_by_concept.png`` from ``out_dir`` if present (avoids
    stale charts from larger runs).
    """
    out_path = out_dir / "coverage_by_concept.png"
    if len(cases) < _COVERAGE_BY_CONCEPT_MIN_TOTAL_CASES:
        if out_path.exists():
            out_path.unlink()
        return

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
            n_con = len(concepts)
            y_idx = list(range(n_con))
            offset = 0.2
            bar_h = 0.35
            y_pass = [i - offset for i in y_idx]
            y_fail = [i + offset for i in y_idx]

            ax.barh(
                y_pass,
                passed,
                height=bar_h,
                color=_COLOR_GOOD,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
                label="Passed",
            )
            ax.barh(
                y_fail,
                failed,
                height=bar_h,
                color=_COLOR_BAD,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
                label="Failed",
            )

            ax.set_yticks(y_idx)
            ax.set_yticklabels(concepts)
            ax.invert_yaxis()
            ax.set_xlabel("Cases")
            ax.set_title("Coverage by concept (debug)")
            ax.set_axisbelow(True)
            ax.yaxis.grid(False)
            ax.legend(loc="lower right")

        _debug_dataset_footer(fig, len(cases))
        fig.tight_layout(rect=(0, 0.065, 1, 0.96))
        fig.savefig(out_path)
        plt.close(fig)


_FAILURE_MODE_AXIS_LABELS: dict[str, str] = {
    "missing_required_concept": "missing concept",
    "compare_asymmetry": "compare asymmetry",
    "compare_entity_collapse": "entity collapse",
    "validation_missed_error": "missed validation error",
    "retrieval_miss": "retrieval miss",
    "structure_failure": "structure failure",
}


def _failure_mode_yticklabel(tag: str) -> str:
    return _FAILURE_MODE_AXIS_LABELS.get(tag, tag.replace("_", " "))


def write_failure_modes_chart(
    cases: list[EvaluationCaseResult], out_dir: Path
) -> None:
    """Horizontal bar chart of failure-mode counts (descending by count)."""
    from app.eval.case_result_tags import canonical_failure_tags_for_row

    counts: Counter[str] = Counter()
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

    ordered_keys = sorted(counts.keys(), key=lambda tag: (-counts[tag], tag))
    yticklabels = [_failure_mode_yticklabel(k) for k in ordered_keys]

    with _paper_style():
        fig, ax = plt.subplots(figsize=(10.0, 0.55 * max(4, len(ordered_keys)) + 1.5))
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
            ax.set_yticklabels(yticklabels)
            ax.invert_yaxis()
            ax.set_xlabel("Count")
            ax.set_title("Failure modes (debug)")
            ax.set_axisbelow(True)
            ax.yaxis.grid(False)
            ax.margins(y=0.02)

        _debug_dataset_footer(fig, len(cases))
        fig.tight_layout(rect=(0, 0.065, 1, 0.96))
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
    cases: list[Any],
    out_dir: Path,
    current_run: EvaluationRun | None = None,
    *,
    include_regression: bool = True,
    summary_run: Any | None = None,
) -> None:
    """Generate all evaluation artifacts; chart failures never crash the run.

    ``current_run`` is optional purely for backwards compatibility with
    older callers (and the existing test suite). When provided with
    ``include_regression=True`` (default), a regression-comparison chart is
    written when a prior run exists on the same dataset.

    ``summary_run`` overrides the object used for ``evaluation_summary.png``
    (defaults to ``current_run``). Use a :class:`types.SimpleNamespace` with
    ``dataset_name`` and ``overall_score`` for critic-only summaries.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    prev_pair: tuple[EvaluationRun, list[EvaluationCaseResult]] | None = None
    regression_meaningful = False
    if current_run is not None and include_regression:
        prev_pair = _resolve_previous_run(current_run)
        if prev_pair is not None:
            prev_run, prev_cases = prev_pair
            metrics = _regression_metrics(prev_run, prev_cases, current_run, cases)
            regression_meaningful = _regression_has_movement(metrics)
    health = _dataset_health(cases, regression_meaningful=regression_meaningful)
    summary_source = summary_run if summary_run is not None else current_run

    generators: list[tuple[str, Any]] = [
        ("retrieval_accuracy.png", lambda: write_retrieval_accuracy_chart(cases, out_dir)),
        ("question_type_breakdown.png", lambda: write_question_type_chart(cases, out_dir)),
        ("pipeline_diagram.png", lambda: write_pipeline_diagram(out_dir)),
        ("report_dashboard.png", lambda: write_report_dashboard_chart(cases, out_dir)),
        ("coverage_by_concept.png", lambda: write_coverage_by_concept_chart(cases, out_dir)),
        ("failure_modes.png", lambda: write_failure_modes_chart(cases, out_dir)),
        ("example_answers.csv", lambda: write_example_answers_csv(cases, out_dir)),
        ("error_analysis.csv", lambda: write_error_analysis_csv(cases, out_dir)),
    ]
    if summary_source is not None:
        generators.append(
            (
                "evaluation_summary.png",
                lambda: write_evaluation_summary_chart(
                    summary_source, cases, out_dir, health=health
                ),
            )
        )
    if include_regression and prev_pair is not None:
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
