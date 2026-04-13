# Admin insights Phase 1 (2026-04-12)

Shipped aggregate **GET `/api/admin/insights?days=`** backed by [`admin_insights.py`](backend/app/services/admin_insights.py), replacing the stub. Validation **severity** histogram uses SQLite **`json_extract(validation_checks_json, '$.severity')`**.

**Follow-ups (later phases):** paged low-confidence drill-down, CSV export, chunk-level JOINs, persist **`model_name`** / **`token_usage_json`** from LLM paths then add cost timeseries.
