# Evaluation areas

This suite tracks six evaluation areas per run and now presents them with explicit sample-size caveats (low-`n` labels/fading and small-dataset warnings).

| Evaluation Area | What you show | Primary artifact(s) |
|-----------------|---------------|---------------------|
| Retrieval quality | Did retrieval include the expected lecture chunk? | `retrieval_accuracy.png`, retrieval panel in `report_dashboard.png` |
| Answer grounding | Did the answer stay course-grounded? | `error_analysis.csv`, `failure_modes.png` |
| Explanation quality | Was the answer clear/useful? | case-level scoring (`derive_case_scores`) and `report.md` |
| Depth | Did the answer go beyond a surface statement? | case-level scoring (`derive_case_scores`) and `report.md` |
| Question handling | How does performance vary by intent/mode? | `question_type_breakdown.png`, structure panel in `report_dashboard.png` |
| Error analysis | Which failures are blocking quality now? | `error_analysis.csv`, `failure_modes.png` |

## Figure set and intent

All figure writers live in [`backend/app/eval/evaluation_outputs.py`](../app/eval/evaluation_outputs.py). The chart set is now tuned for honest, debugging-first interpretation on small suites:

- `evaluation_summary.png`: run scorecard with total/passed/failed/mean, strongest/weakest capability, under-tested category count, and an explicit warning when `total_cases < 50`.
- `question_type_breakdown.png`: count + pass-rate by intent, sorted by case volume; intent buckets with `n < 5` are labeled (`low n` / `n too small`) and visually faded.
- `retrieval_accuracy.png`: split view separating normal content-retrieval prompts from mode/clarification/quiz/summary prompts so underspecified queries (for example, "Compare these") are not interpreted as ordinary retrieval failures.
- `report_dashboard.png`: structure compliance + retrieval diagnostics + a dataset-health panel (total cases, low-`n` categories, under-tested concepts, regression meaningfulness, paired-boost availability).
- `regression_comparison.png`: only shows bar comparison when a prior run exists and at least one tracked metric moved; otherwise emits a no-movement notice image.
- `coverage_by_concept.png`: per-concept case coverage with stacked pass/fail counts.
- `failure_modes.png`: failure-type counts with example `test_id` for debugging triage.
- `pipeline_diagram.png`: static architecture view (sample size not applicable).

`example_answers.csv` and `error_analysis.csv` are still emitted for spreadsheet/report workflows.

## Metric sources

- Retrieval diagnostics come from `retrieval_diagnostics` and `summarize_retrieval` in [`backend/app/eval/capability_analytics.py`](../app/eval/capability_analytics.py).
- Capability and intent accuracy come from `summarize_capability` in [`backend/app/eval/capability_analytics.py`](../app/eval/capability_analytics.py).
- Concept under-testing/coverage comes from `summarize_coverage` in [`backend/app/eval/capability_analytics.py`](../app/eval/capability_analytics.py).
- Failure tags come from `canonical_failure_tags_for_row` in [`backend/app/eval/case_result_tags.py`](../app/eval/case_result_tags.py), plus `primary_error_type_for_row`.

## How to regenerate

From `backend/`, run:

```bash
PYTHONPATH=. python -m app.eval.run_eval --dataset data/eval/l487_eval_suite.json --run-name "<name>"
```

Outputs land in `evaluation_outputs/` at the repo root.
