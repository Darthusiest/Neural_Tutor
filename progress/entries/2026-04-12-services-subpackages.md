# Services subpackage layout (`app.services`)

**Date:** 2026-04-12

**Summary:** Reorganized `backend/app/services/` into **`answers/`**, **`knowledge/`**, **`generation/`**, and **`lectures/`** so planning, KB, external LLM clients, and corpus import are easy to find. **`retrieval.py`**, **`retrieval_v2.py`**, and **`lecture_data.py`** remain at the services root to avoid a naming clash with a `retrieval` package.

**Rationale:** Clear separation between (1) answer pipeline, (2) course knowledge, (3) API-backed generation, (4) lecture import, vs. shared lexical retrieval.

**Follow-ups:** If the codebase grows, consider a `retrieval/` package with renamed modules (e.g. `lexical.py`) and re-exports from `__init__.py`.
