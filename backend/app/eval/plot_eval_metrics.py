"""Plot rule-based eval diagnostics from persisted runs (PNG)."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app import create_app
from app.eval.analytics_common import (
    RunFilter,
    fetch_case_rows_for_runs,
    fetch_ordered_runs,
    parse_json_list,
    suite_category,
)
from app.eval.capability_analytics import (
    summarize_capability,
    summarize_errors,
    summarize_iteration,
)
from app.eval.export_analytics import (
    RETRIEVAL_LEAKAGE_TAG,
    export_latency_by_run,
    export_mode_accuracy_by_run,
)
from app.models import EvaluationCaseResult, EvaluationRun


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


def _default_out_dir() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    d = _repo_root() / "reports" / "eval_plots" / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


def _parse_run_ids(s: str | None) -> frozenset[int] | None:
    if not s or not s.strip():
        return None
    out: set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if part:
            out.add(int(part))
    return frozenset(out)


def _x_labels(runs: list[EvaluationRun]) -> tuple[list[int], list[str]]:
    xs = list(range(len(runs)))
    labels = [f"{r.id}" for r in runs]
    return xs, labels


def _category_pass_rate_by_run(
    runs: list[EvaluationRun], cases: list[EvaluationCaseResult]
) -> tuple[list[str], dict[str, list[float]]]:
    run_ids = [r.id for r in runs]
    cats = sorted({suite_category(c) for c in cases})
    tally: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for c in cases:
        cat = suite_category(c)
        t = tally[c.evaluation_run_id][cat]
        t[1] += 1
        if c.pass_bool:
            t[0] += 1
    series: dict[str, list[float]] = {cat: [] for cat in cats}
    for rid in run_ids:
        for cat in cats:
            p, t = tally[rid].get(cat, [0, 0])
            series[cat].append(float("nan") if t == 0 else p / t)
    return cats, series


def _retrieval_leakage_rate_per_run(
    runs: list[EvaluationRun], cases: list[EvaluationCaseResult]
) -> list[float]:
    rates: list[float] = []
    for rid in [r.id for r in runs]:
        rc = [c for c in cases if c.evaluation_run_id == rid]
        if not rc:
            rates.append(float("nan"))
            continue
        n = sum(1 for c in rc if RETRIEVAL_LEAKAGE_TAG in parse_json_list(c.error_categories_json))
        rates.append(n / len(rc))
    return rates


def _write_iteration_accuracy(
    out_dir: Path,
    runs: list[EvaluationRun],
    cases: list[EvaluationCaseResult],
    xs: list[int],
    xtick_labels: list[str],
) -> None:
    iteration_rows = summarize_iteration(runs, cases)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for key, label in (
        ("overall_accuracy", "Overall accuracy"),
        ("step_by_step_accuracy", "Step-by-step accuracy"),
        ("retrieval_grounded_accuracy", "Retrieval-grounded accuracy"),
    ):
        ax.plot(
            xs,
            [float(row.get(key, float("nan"))) for row in iteration_rows],
            marker="o",
            linestyle="-",
            label=label,
        )
    ax.set_title("Iteration accuracy over runs")
    ax.set_xlabel("Run")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(xs)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "iteration_accuracy_over_runs.png", dpi=120)
    plt.close(fig)


def _write_capability_breakdown(out_dir: Path, cases: list[EvaluationCaseResult]) -> None:
    capability = summarize_capability(cases)["by_intent"]
    labels = list(sorted(capability.keys()))
    values = [float(capability[label]["accuracy"]) for label in labels]
    totals = [int(capability[label]["total_cases"]) for label in labels]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(labels, values, color="#1f77b4")
    ax.set_title("Capability breakdown")
    ax.set_xlabel("Capability")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=30)
    for bar, total in zip(bars, totals, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"n={total}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(out_dir / "capability_breakdown.png", dpi=120)
    plt.close(fig)


def _write_error_distribution(out_dir: Path, cases: list[EvaluationCaseResult]) -> None:
    errors = summarize_errors(cases)["by_error_type"]
    fig, ax = plt.subplots(figsize=(7, 7))
    if errors:
        labels = list(errors.keys())
        sizes = [int(errors[label]["count"]) for label in labels]
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.set_title("Error distribution")
    else:
        ax.text(0.5, 0.5, "No failed cases", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_dir / "error_distribution.png", dpi=120)
    plt.close(fig)


def _write_legacy_plots(
    out_dir: Path,
    runs: list[EvaluationRun],
    cases: list[EvaluationCaseResult],
    xs: list[int],
    xtick_labels: list[str],
) -> None:
    cats, series = _category_pass_rate_by_run(runs, cases)
    fig, ax = plt.subplots(figsize=(10, 5))
    for cat in cats:
        ax.plot(xs, series[cat], marker=".", linestyle="-", label=cat)
    ax.set_title("Suite category pass rate")
    ax.set_xlabel("Run")
    ax.set_ylabel("Pass rate (within category)")
    ax.set_xticks(xs)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
    ax.set_ylim(-0.05, 1.05)
    h, lab = ax.get_legend_handles_labels()
    if lab:
        ax.legend(h, lab, loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pass_rate_by_category_over_time.png", dpi=120)
    plt.close(fig)

    run_ids = [r.id for r in runs]
    mode_rows = {int(r["run_id"]): r for r in export_mode_accuracy_by_run(runs, cases)}
    eff_acc: list[float] = []
    for rid in run_ids:
        row = mode_rows.get(rid, {})
        v = row.get("effective_accuracy", "")
        eff_acc.append(float(v) if v != "" else float("nan"))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(xs, eff_acc, marker="o", linestyle="-", color="#2ca02c")
    ax.set_title("Mode routing accuracy (effective vs expected)")
    ax.set_xlabel("Run")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(xs)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "mode_accuracy_over_time.png", dpi=120)
    plt.close(fig)

    leak_rates = _retrieval_leakage_rate_per_run(runs, cases)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(xs, leak_rates, marker="o", linestyle="-", color="#d62728")
    ax.set_title("Retrieval leakage (share of cases tagged retrieval_leakage)")
    ax.set_xlabel("Run")
    ax.set_ylabel("Fraction of cases")
    ax.set_xticks(xs)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "retrieval_leakage_over_time.png", dpi=120)
    plt.close(fig)

    lat_rows = {int(r["run_id"]): r for r in export_latency_by_run(cases)}
    means: list[float] = []
    medians: list[float] = []
    for rid in run_ids:
        row = lat_rows.get(rid)
        if row:
            means.append(float(row["mean_latency_ms"]))
            medians.append(float(row["median_latency_ms"]))
        else:
            means.append(float("nan"))
            medians.append(float("nan"))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(xs, means, marker="o", linestyle="-", label="Mean")
    ax.plot(xs, medians, marker="s", linestyle="--", label="Median")
    ax.set_title("Latency over runs")
    ax.set_xlabel("Run")
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(xs)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "latency_over_time.png", dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot rule-based eval analytics from the database.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for PNG files (default: reports/eval_plots/<utc-timestamp>/)",
    )
    parser.add_argument("--dataset", default=None, help="Substring filter on dataset_name")
    parser.add_argument("--run-ids", default=None, help="Comma-separated run ids")
    parser.add_argument("--last-n", type=int, default=None, help="Last N runs after ordering")
    parser.add_argument(
        "--legacy-plots",
        action="store_true",
        help="Also write older auxiliary diagnostic PNGs.",
    )
    args = parser.parse_args(argv)

    rf = RunFilter(
        dataset_substring=args.dataset,
        run_ids=_parse_run_ids(args.run_ids),
        last_n_runs=args.last_n,
    )

    app = create_app()
    with app.app_context():
        runs = fetch_ordered_runs(rf)
        if not runs:
            total = EvaluationRun.query.count()
            if total == 0:
                print(
                    "No evaluation runs in the database; no plots written.\n"
                    "Create runs first, for example:\n"
                    "  PYTHONPATH=. python -m app.eval.run_eval "
                    '--dataset data/eval/l487_eval_suite.json --run-name "baseline"'
                )
            else:
                print(
                    "No evaluation runs matched filters; no plots written. "
                    f"(Found {total} run(s) in DB - try different --dataset / --run-ids / --last-n.)"
                )
            return 1
        out_dir = args.out_dir or _default_out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        run_ids = [r.id for r in runs]
        cases = fetch_case_rows_for_runs(run_ids)
        xs, xtick_labels = _x_labels(runs)

        _write_iteration_accuracy(out_dir, runs, cases, xs, xtick_labels)
        _write_capability_breakdown(out_dir, cases)
        _write_error_distribution(out_dir, cases)
        if args.legacy_plots:
            _write_legacy_plots(out_dir, runs, cases, xs, xtick_labels)

    print(f"Wrote PNG plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
