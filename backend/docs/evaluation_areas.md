# Evaluation areas

This suite tracks six evaluation areas per run. PNGs use a **debug-first** style: one italic footer with dataset size (`Small evaluation set (n=…)` below 50 cases), no per-bar “low n” warnings, full-opacity bars, and short titles.

| Evaluation Area | What you show | Primary artifact(s) |
|-----------------|---------------|---------------------|
| Retrieval quality | Did retrieval include the expected lecture chunk? | `retrieval_accuracy.png`, retrieval panel in `report_dashboard.png` |
| Answer grounding | Did the answer stay course-grounded? | `error_analysis.csv`, `failure_modes.png` |
| Explanation quality | Was the answer clear/useful? | case-level scoring (`derive_case_scores`) and `report.md` |
| Depth | Did the answer go beyond a surface statement? | case-level scoring (`derive_case_scores`) and `report.md` |
| Question handling | How does performance vary by intent/mode? | `question_type_breakdown.png` (pass rate by intent), structure panel in `report_dashboard.png` |
| Error analysis | Which failures are blocking quality now? | `error_analysis.csv`, `failure_modes.png` |

## Figure set and intent

All figure writers live in [`backend/app/eval/evaluation_outputs.py`](../app/eval/evaluation_outputs.py). Charts are for spotting regressions, not for benchmark claims:

- `evaluation_summary.png`: dashboard-style card — rounded container, large donut with **«N% passed»** and **«passed/total questions»**, **Passed/Failed** legend, 2×2 stat tiles (total / passed / failed / mean score), dataset line, and an amber **small-n** badge when `n < 50` (plus an italic interpretability line). Optional `debug=True` on the writer appends **(debug)** to the title (default off).
- `question_type_breakdown.png`: pass rate by intent; bar labels ``pct% (n=k)`` (or ``n=1`` when a single case would read as misleading ``100%``); one footer with total `n`.
- `retrieval_accuracy.png`: top-k hit rate for definition / fact / synthesis (non-underspecified); one footer with total `n`.
- `report_dashboard.png`: structure + retrieval metrics; **Debug dashboard (n=…)** title and italic **For debugging only** subtitle; breakdown uses **short** intent labels and consistent ``pct (n=k)`` annotations; retrieval x-axis uses two-line labels.
- `regression_comparison.png`: previous vs current metrics when something moved; otherwise a short no-change PNG; footer with `n`.
- `coverage_by_concept.png`: only when the suite has **at least 30** cases — pass vs fail counts per concept; footer with `n`.
- `failure_modes.png`: failure-type counts; footer with `n`.
- `pipeline_diagram.png`: static architecture schematic (no eval `n`); live turn flow plus **Gemini Critic (admin)** on stored eval batches (does not block live responses).

`example_answers.csv` and `error_analysis.csv` are still emitted for spreadsheet/report workflows.

## Metric sources

- Retrieval diagnostics come from `retrieval_diagnostics` and `summarize_retrieval` in [`backend/app/eval/capability_analytics.py`](../app/eval/capability_analytics.py).
- Capability and intent accuracy come from `summarize_capability` in [`backend/app/eval/capability_analytics.py`](../app/eval/capability_analytics.py).
- Concept under-testing/coverage comes from `summarize_coverage` in [`backend/app/eval/capability_analytics.py`](../app/eval/capability_analytics.py).
- **Phased remediation ordering** comes from `summarize_coverage_phase_buckets` in the same module — exported as **`coverage_phase_plan.csv`** by [`export_analytics.py`](../app/eval/export_analytics.py).
- Failure tags come from `canonical_failure_tags_for_row` in [`backend/app/eval/case_result_tags.py`](../app/eval/case_result_tags.py), plus `primary_error_type_for_row`.

## Commands (from `backend/`)

| Goal | Command |
|------|---------|
| Full chat eval: timestamped **`reports/eval_runs/<ts>/`** plus refresh **`evaluation_outputs/`** at repo root | `PYTHONPATH=. python -m app.eval.run_eval --dataset data/eval/l487_eval_suite.json --run-name "<name>"` |
| Fast pipeline eval: persist **`evaluation_runs`** only (no report folder, no `evaluation_outputs/`) | `flask --app wsgi run-eval` (optional: `--dataset`, `--run-name`, `--compare-last`, …) |
| Admin Gemini critic (LLM judge) on persisted cases | Use **Admin → Gemini critic** (or **`POST /api/admin/eval/runs/<id>/critic`**). **Prep:** run **`PYTHONPATH=. python -m app.eval.run_eval …`** first so rows include **`assistant_message_id`**. Writes **`evaluation_outputs/critic/<batch>/`** (same chart names as rule-based). See **`progress/entries/2026-05-10-gemini-critic.md`**. |
| Rollup CSVs from DB → **`reports/eval_analytics/<ts>/`** | `PYTHONPATH=. python -m app.eval.export_analytics` (optional: `--dataset`, `--run-ids`, `--last-n`, `--out-dir`, `--worst-k`, `--coverage-phase-sort`, `--coverage-min-cases`) |
| Per-concept failure digest | `PYTHONPATH=. python scripts/eval_diagnose_concept.py --concept <bucket> [--run-id N] [--write report.md]` |
| Rebuild **`evaluation_outputs/`** PNGs/CSVs from a persisted run only | `python3 scripts/regenerate_evaluation_outputs.py` or `--run-id <id>` / `--out-dir <path>` |

`flask run-eval` does not write `reports/eval_runs/` or `evaluation_outputs/`; after it, use **`regenerate_evaluation_outputs.py`** and/or **`export_analytics`** if you need those artifacts.
