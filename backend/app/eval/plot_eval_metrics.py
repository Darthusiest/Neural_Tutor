"""Plot eval metrics from persisted runs (PNG). Not training loss — validation / regression trends."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
            if t == 0:
                series[cat].append(float("nan"))
            else:
                series[cat].append(p / t)
    return cats, series


def _top_error_tags(cases: list[EvaluationCaseResult], top_n: int) -> list[str]:
    c = Counter()
    for row in cases:
        for e in parse_json_list(row.error_categories_json):
            if e:
                c[e] += 1
    return [k for k, _ in c.most_common(top_n)]


def _error_counts_per_run(
    runs: list[EvaluationRun], cases: list[EvaluationCaseResult], tags: list[str]
) -> dict[str, list[int]]:
    run_ids = [r.id for r in runs]
    out: dict[str, list[int]] = {t: [] for t in tags}
    for rid in run_ids:
        rc = [c for c in cases if c.evaluation_run_id == rid]
        for tag in tags:
            n = sum(1 for c in rc if tag in parse_json_list(c.error_categories_json))
            out[tag].append(n)
    return out


def _retrieval_leakage_rate_per_run(runs: list[EvaluationRun], cases: list[EvaluationCaseResult]) -> list[float]:
    run_ids = [r.id for r in runs]
    rates: list[float] = []
    for rid in run_ids:
        rc = [c for c in cases if c.evaluation_run_id == rid]
        if not rc:
            rates.append(float("nan"))
            continue
        n = sum(1 for c in rc if RETRIEVAL_LEAKAGE_TAG in parse_json_list(c.error_categories_json))
        rates.append(n / len(rc))
    return rates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot eval metrics from the database (PNG).")
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
        "--top-errors",
        type=int,
        default=8,
        help="Top error categories to plot over time",
    )
    args = parser.parse_args(argv)

    out_dir = args.out_dir or _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    rf = RunFilter(
        dataset_substring=args.dataset,
        run_ids=_parse_run_ids(args.run_ids),
        last_n_runs=args.last_n,
    )

    app = create_app()
    with app.app_context():
        runs = fetch_ordered_runs(rf)
        if not runs:
            print("No evaluation runs matched filters; no plots written.")
            return 1
        run_ids = [r.id for r in runs]
        cases = fetch_case_rows_for_runs(run_ids)
        xs, xtick_labels = _x_labels(runs)

        # overall score
        ys_score = [float(r.overall_score) if r.overall_score is not None else float("nan") for r in runs]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(xs, ys_score, marker="o", linestyle="-", color="#1f77b4")
        ax.set_title("Evaluation score over runs")
        ax.set_xlabel("Run (ordered by time)")
        ax.set_ylabel("Mean score")
        fig.text(0.5, 0.02, "Regression trend", ha="center", fontsize=9, style="italic", color="gray")
        ax.set_xticks(xs)
        ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(out_dir / "overall_score_over_time.png", dpi=120)
        plt.close(fig)

        # pass rate by category
        cats, series = _category_pass_rate_by_run(runs, cases)
        fig, ax = plt.subplots(figsize=(10, 5))
        for cat in cats:
            ax.plot(xs, series[cat], marker=".", linestyle="-", label=cat)
        ax.set_title("Validation suite pass rate")
        ax.set_xlabel("Run (ordered by time)")
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

        # error categories over time
        top_tags = _top_error_tags(cases, max(1, args.top_errors))
        if not top_tags:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No error categories in selected runs", ha="center", va="center")
            ax.set_axis_off()
            fig.savefig(out_dir / "error_categories_over_time.png", dpi=120)
            plt.close(fig)
        else:
            err_series = _error_counts_per_run(runs, cases, top_tags)
            fig, ax = plt.subplots(figsize=(10, 5))
            for tag in top_tags:
                ax.plot(xs, err_series[tag], marker=".", linestyle="-", label=tag[:40])
            ax.set_title("Regression trend — error category counts per run")
            ax.set_xlabel("Run (ordered by time)")
            ax.set_ylabel("Case count")
            ax.set_xticks(xs)
            ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
            ax.legend(loc="best", fontsize=7)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / "error_categories_over_time.png", dpi=120)
            plt.close(fig)

        # mode accuracy (effective)
        mode_rows = {int(r["run_id"]): r for r in export_mode_accuracy_by_run(runs, cases)}
        eff_acc: list[float] = []
        for rid in run_ids:
            row = mode_rows.get(rid, {})
            v = row.get("effective_accuracy", "")
            eff_acc.append(float(v) if v != "" else float("nan"))
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(xs, eff_acc, marker="o", linestyle="-", color="#2ca02c")
        ax.set_title("Mode routing accuracy (effective vs expected)")
        ax.set_xlabel("Run (ordered by time)")
        ax.set_ylabel("Accuracy")
        fig.text(0.5, 0.02, "Regression trend", ha="center", fontsize=9, style="italic", color="gray")
        ax.set_xticks(xs)
        ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(out_dir / "mode_accuracy_over_time.png", dpi=120)
        plt.close(fig)

        # retrieval leakage rate
        leak_rates = _retrieval_leakage_rate_per_run(runs, cases)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(xs, leak_rates, marker="o", linestyle="-", color="#d62728")
        ax.set_title("Retrieval leakage (share of cases tagged retrieval_leakage)")
        ax.set_xlabel("Run (ordered by time)")
        ax.set_ylabel("Fraction of cases")
        fig.text(0.5, 0.02, "Regression trend", ha="center", fontsize=9, style="italic", color="gray")
        ax.set_xticks(xs)
        ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(out_dir / "retrieval_leakage_over_time.png", dpi=120)
        plt.close(fig)

        # latency
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
        ax.set_xlabel("Run (ordered by time)")
        ax.set_ylabel("Latency (ms)")
        ax.set_xticks(xs)
        ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "latency_over_time.png", dpi=120)
        plt.close(fig)

    print(f"Wrote PNG plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
