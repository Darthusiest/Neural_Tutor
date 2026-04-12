# Changelog

All **notable** changes to Neural Tutor are recorded here so the repo stays a **single source of truth** for what shipped and when.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) (Added / Changed / Fixed / Removed / Security). Use **ISO dates** (`YYYY-MM-DD`) in section headings.

## [Unreleased]

<!-- Move items below into a dated section when you cut a release or merge a meaningful batch. -->

### Added

- **Hybrid generation:** primary **Course Answer** via OpenAI when **`PRIMARY_COURSE_ANSWER_OPENAI`** (or legacy **`LLM_ANSWER_GENERATION`**) and **`OPENAI_API_KEY`** are set — [`generation/course_generation.py`](backend/app/services/generation/course_generation.py); validation **`severity`** (`pass` / `weak` / `fail`) on [`ValidationResult`](backend/app/services/answers/answer_validation.py); optional **Boosted Explanation** via **Gemini** when **`GEMINI_API_KEY`** or **`GOOGLE_API_KEY`** is set ([`generation/gemini_boost.py`](backend/app/services/generation/gemini_boost.py)), gated by [`should_use_gemini_boost`](backend/app/services/generation/boost_triggers.py) (validation, confidence, synthesis-style complexity, user phrasing, or legacy heuristics); OpenAI boost remains fallback when Gemini is unavailable. Chat **`payload_json`** adds **`primary_model`**, **`validation_severity`**, **`boost_provider`**, **`boost_reason`**, **`query_complexity`**.
- Documentation policy: [`CHANGELOG.md`](CHANGELOG.md), [`progress/README.md`](progress/README.md) checklist, [`.cursor/rules/documentation.mdc`](.cursor/rules/documentation.mdc) for agents.
- **Structured reasoning pipeline:** [`LING487_STRUCTURED_PIPELINE_KB.json`](backend/data/LING487_STRUCTURED_PIPELINE_KB.json) concept graph; [`knowledge/concept_kb`](backend/app/services/knowledge/concept_kb.py), [`knowledge/structured_query`](backend/app/services/knowledge/structured_query.py), [`answers/answer_planning`](backend/app/services/answers/answer_planning.py), [`answers/answer_generation`](backend/app/services/answers/answer_generation.py), [`answers/answer_validation`](backend/app/services/answers/answer_validation.py), [`reasoning_pipeline`](backend/app/services/reasoning_pipeline.py); chat uses it when **`STRUCTURED_PIPELINE_ENABLED`**; optional **`LLM_ANSWER_GENERATION`** for plan-constrained Course Answer; migration **`004`** adds pipeline columns to **`retrieval_logs`**; tests in [`test_structured_pipeline.py`](backend/tests/test_structured_pipeline.py).

### Changed

- **Theme:** light / dark mode with a **toggle** in the app header ([`Header.jsx`](frontend/src/components/Header.jsx)); auth pages (`/login`, `/register`, etc.) use a fixed top-right control ([`App.jsx`](frontend/src/App.jsx)). Preference is stored in **`localStorage`** (`neural-tutor-theme`); [`index.html`](frontend/index.html) applies it before paint to reduce flash. Styles use CSS variables in [`index.css`](frontend/src/index.css); [`ThemeContext.jsx`](frontend/src/context/ThemeContext.jsx), [`ThemeToggle.jsx`](frontend/src/components/ThemeToggle.jsx).
- **Chat UI:** message list auto-scrolls to the latest message when history updates (including after the assistant reply). [`ChatPanel.jsx`](frontend/src/components/ChatPanel.jsx).
- **Structured rule-based Course Answer:** [`build_answer_plan`](backend/app/services/answers/answer_planning.py) assigns **one distinct retrieved chunk per section** for `direct_definition` and `multi_step_explanation` (ordered by relevance) instead of reusing the same top chunks under every `###` heading; empty section slots are dropped. [`generate_structured_answer`](backend/app/services/answers/answer_generation.py) **deduplicates** chunks across sections, uses **`_sample_questions_as_text`** so JSON `[]` does not render as an example line, and formats sources as `**Lecture N** · topic`. OpenAI plan-constrained prompt ([`generation/llm.py`](backend/app/services/generation/llm.py)) instructs not to repeat the same excerpt under multiple headings. Test: [`test_structured_pipeline.py`](backend/tests/test_structured_pipeline.py) `test_direct_definition_distinct_chunk_per_section`.
- **No-chunk chat replies:** when retrieval returns **no lecture chunks**, the assistant uses **rotating templates** (greeting / short acknowledgement / off-topic) from [`conversational_responses.py`](backend/app/services/conversational_responses.py); copy is **multi-paragraph, ChatGPT-style prose** with examples woven into explanations (bullets only where a short list of prompts helps); **`payload_json`** adds **`no_match_kind`**; **boost** is skipped when there are no chunks. Tests: [`test_conversational_responses.py`](backend/tests/test_conversational_responses.py).
- **Services layout:** grouped modules under [`backend/app/services/`](backend/app/services/) — **`answers/`** (planning, rule generation, validation), **`knowledge/`** (concept KB JSON, domain aliases, structured query), **`generation/`** (OpenAI, Gemini boost, boost triggers), **`lectures/`** (import, chunk keys). Top-level **`retrieval.py`**, **`retrieval_v2.py`**, **`lecture_data.py`**, **`query_understanding.py`**, **`chat_orchestrator.py`**, **`reasoning_pipeline.py`**, **`study.py`**, **`reset_email.py`** unchanged. All imports and tests updated; [`README.md`](README.md) **Backend services layout**; [`backend/docs/schema.md`](backend/docs/schema.md) cross-refs; [`progress/entries/2026-04-12-services-subpackages.md`](progress/entries/2026-04-12-services-subpackages.md).
- **Chat boost:** **Boosted Explanation** defaults to **Gemini only** ([`generation/gemini_boost.py`](backend/app/services/generation/gemini_boost.py)); **`OPENAI_BOOST_FALLBACK`** (default off) enables OpenAI when Gemini is missing or errors. Legacy chat path now builds plan + structured query for Gemini boost when chunks exist. **Structured logging** on each turn (`confidence`, `primary_model`, `validation_severity`, boost reason/provider).
- [`README.md`](README.md): backend **environment variables** table (OpenAI primary vs Gemini boost, pipeline flags); **`POST /api/chat`** documents assistant `payload_json` pipeline fields; **Next steps** / security notes for hybrid LLMs; setup uses `backend/.env` without referencing a missing root `.env.example`.
- [`backend/docs/schema.md`](backend/docs/schema.md): assistant `payload_json` pipeline keys, **`validation_checks_json`** `severity`, **`boost_reason`** codes aligned with [`generation/boost_triggers.py`](backend/app/services/generation/boost_triggers.py).
- [`backend/docs/AUTH_LOCAL.md`](backend/docs/AUTH_LOCAL.md): local `.env` setup without relying on a committed `.env.example`.
- **Retrieval v2 hardening:** deterministic fuzzy domain-term matching (stable tie-break); single-lecture **summary** uses lexical ranking within the lecture with `SUMMARY_MAX_CHUNKS` cap and real `RetrievalDiagnostics` (no fixed 0.85 confidence); **synthesis** runs a primary `retrieve_chunks` pass plus a capped, deduped augmentation pass; **compare** uses side-only subqueries, `min` confidence, merged per-chunk diagnostics, and optional `compare_side_diagnostics` on `EnhancedRetrievalResult`; **diversification** skips reordering when synthesis chunks already span enough distinct lectures; **chat** uses `retrieve_enhanced` with optional `query_type` in `payload_json`.
- **Lexical retrieval:** `retrieve_chunks` / `score_chunks_keyword` accept `lecture_filter` + `summary_rank` for within-lecture ranked summaries.

### Fixed

- **Alembic:** migration **`003`** `down_revision` corrected to **`002_chunk_key`** (was invalid **`002`**). Migration **`001`** `response_variants.retrieval_log_id`: named **`create_foreign_key`** for SQLite batch mode (fixes **“Constraint must have a name”** on `flask db upgrade`).

### Known limitations

- **Gemini boost** uses the Google Generative Language REST API; model availability and IDs vary by account — set **`GEMINI_MODEL`** accordingly. If both Gemini and OpenAI calls fail, no Boosted Explanation is returned even when gating says boost.
- **Structured pipeline** validation is heuristic (lexical / pattern checks); **study** routes still use the legacy retrieval path until opted in.
- Retrieval is still **keyword-only**; embedding / hybrid backends are not implemented.
- **Synthesis** diagnostics refer to the primary lexical pass; after diversification and deduplication, displayed chunk order may not match diagnostic rank order one-to-one.
- **Compare** side queries use the extracted concept strings (plus optional lecture hint) to limit cross-talk; unusually short extractions can yield thinner lexical matches than the full question would.

---

## [2026-04-09]

### Added

- **Retrieval v2:** `domain_knowledge` (LING 487 aliases, concept families, lecture graph), `query_understanding` (query types, expansion, typo hints), `retrieval_v2` orchestrator, `EnhancedRetrievalResult`; chunk metadata `chunk_type`, `concept_family` with migrations `002` / `003`.
- **Pydantic** lecture corpus validation; stable `chunk_key` import identity; configurable `LECTURE_KEYWORD_CAP` and retrieval field weights.
- **Study API** (`/api/study`): quiz (MC + short answer, HMAC tokens), compare (optional OpenAI comparison boost), summary by lecture or topic; chat history via `payload_json` when no `ResponseVariant`.
- **OpenAI (server-only):** `llm.py` chat completions for boosted explanation and comparison boost; `OPENAI_CHAT_MODEL`, `OPENAI_TIMEOUT_SEC` config.
- **Frontend:** Study controls in `ChatPanel`, session helpers on `ChatPage`, `ErrorBoundary` / `ProtectedRoute` wiring.
- **Tests:** domain knowledge, query understanding, retrieval hardening / LING487 goldens, retrieval v2, study.

### Changed

- **`lecture_data.search_lecture_chunks`** delegates to `retrieve_enhanced` (v2); lecture retrieve API may return `query_type`, `supporting_chunks`, `related_topics`, `typo_corrections`.
- **`list_messages`:** falls back to `payload_json` for assistant `course_answer` / `boosted_explanation` when `ResponseVariant` is absent (study turns).
- **README / schema:** synced with analytics layer, migrations, and API surface.

### Fixed

- Quiz MC generation no longer infinite-loops when fewer than four chunks exist (padded distractors).

---

## Earlier history

Work before this changelog was captured in [`progress/entries/`](progress/entries/) (dated files). For archaeology, search that folder and [`progress/scaffold-review-fixes.md`](progress/scaffold-review-fixes.md).
