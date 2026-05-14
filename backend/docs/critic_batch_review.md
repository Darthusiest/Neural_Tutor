# Gemini critic batch review (pipeline vs harsh judge)

Use this after **`evaluation_outputs/critic/<batch>/evaluation_summary.png`** shows a critic pass rate below your target (often aligned with **`CRITIC_PASS_THRESHOLD`**; default **0.68**, i.e. about **68%**).

## What gets written per batch

Beside PNGs and CSVs, each completed critic run should emit **`critic_metrics.json`** in the same folder (`critic_pass_rate`, `failure_primary_counts`, manifest alignment flags, …).

## Quick loop

1. Open **`evaluation_summary.png`**, **`failure_modes.png`**, and **`error_analysis.csv`** for orientation.
2. From **`backend/`** with the same DB as the API:

   ```bash
   PYTHONPATH=. python scripts/review_critic_batch.py --latest
   # or
   PYTHONPATH=. python scripts/review_critic_batch.py --batch-id <folder_name>
   ```

3. If **`critic_pass_rate ≥ --pass-threshold`** (CLI default aligns with **`CRITIC_PASS_THRESHOLD`**, typically **0.68**), the script prints a short summary and does **not** emit drill-down files.
4. If **below threshold**, read **`REVIEW_REPORT.md`** and **`critic_failure_review.csv`** in that batch folder.

## Human verification

- **`core_course_query`** rows: confirm the **stored answer** is wrong vs **retrieved lecture chunks** (Admin **Every case**, or DB). If Gemini cites grounding/compare-shape issues that match what you see, prioritize **pipeline** fixes (retrieval, compare entities, composers).
- **`adversarial_noise`** rows: nonsense/off-topic suite tags — **do not** treat as core course quality. If Gemini penalizes completeness here, treat as **rubric** tuning ([`app/services/critic/gemini_critic.py`](../app/services/critic/gemini_critic.py)), not MFCC/compare refactors.
- **`clarification_edge`**: underspecified prompts — decide case-by-case whether the tutor’s clarification shape is acceptable.

## When to call it a rubric issue

The script flags **rubric calibration** when a large share of failures land in **`adversarial_noise`** (default **≥ 45%** of failures; tune via `--rubric-risk-ratio`). Then confirm with stakeholders before spending time on renderer-only changes.

## Reliability notes

- Prefer **`flask run --no-reload`** or **`scripts/run_gemini_critic_batch.py`** for full batches so **debug reload** does not kill the critic thread mid-run ([`README.md`](../../README.md) Admin section).
