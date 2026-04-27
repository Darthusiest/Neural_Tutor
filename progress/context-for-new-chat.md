# Context for a new Cursor / AI chat (LING 487 Tutor)

Use this file when you **start a fresh chat** and want the assistant to align quickly with this repo—stack, constraints, and where the important code lives—without re-deriving it from scratch.

---

## How to start a new chat (suggested prompt)

Copy the block below into a new conversation and **fill in the bracketed parts**. Attach or `@`-reference this file so the model can open it.

```text
I'm working on LING 487 Tutor (Neural_Tutor repo). Read progress/context-for-new-chat.md for stack, constraints, and hot paths.

Goal: [what you want done in this session]

Constraints: [e.g. smallest correct change; no full redesign; backend only; match existing patterns in backend/app/]

Optional context: [branch, error message, or feature area]
```

**Recommended attachments**

| Attachment | When |
|------------|------|
| This file (`progress/context-for-new-chat.md`) | Almost always for backend work |
| Root `README.md` | API tables, setup, current behavior |
| `backend/docs/schema.md` | Anything touching DB models or migrations |
| Specific paths (e.g. `backend/app/routes/chat.py`) | Narrow tasks |

**What you should expect from the assistant**

- **Course-grounded product:** This is a **LING 487–scoped tutor**, not a generic chat app. Retrieval and structured pipelines are central; avoid treating it like a blank CRUD app.
- **Small, focused diffs:** Prefer the smallest change that solves the problem; don’t expand scope unless you ask.
- **Follow repo conventions:** Match naming, imports, and patterns already in `backend/app/` (and `frontend/src/` for UI work).
- **Verify backend changes:** Run tests from `backend/` (`pytest`) when behavior changes.
- **Docs policy:** After user-visible or API changes, update what `progress/README.md` requires (`CHANGELOG.md`, `README.md`, `schema.md`, etc.).

---

## Stack (short)

| Layer | Notes |
|-------|--------|
| **Backend** | Flask, SQLAlchemy, SQLite default (PostgreSQL via `DATABASE_URL`). Session cookie auth, CSRF, rate limits. Entry: `backend/app/__init__.py`. |
| **Frontend** | React + Vite; proxy `/api` → Flask (`vite.config.js`). |
| **Config / env** | `backend/app/config.py`; local secrets in `backend/.env` (see `backend/.env.example`). `load_dotenv()` runs at app startup. |

---

## Hot paths (where to look first)

| Concern | Location |
|---------|----------|
| HTTP: chat, sessions, feedback | `backend/app/routes/chat.py` |
| One assistant turn (retrieval → answer → boost → DB) | `backend/app/services/chat_orchestrator.py` |
| Structured pipeline (retrieve → plan → Course Answer → validate) | `backend/app/services/reasoning_pipeline.py` |
| Retrieval + query intent + **mode routing** | `backend/app/services/retrieval_v2.py` (calls `query_mode.py` for API modes: auto / chat / quiz / compare / summary); builds `mode_routing` on `EnhancedRetrievalResult` |
| Deterministic **mode detection** (no LLM) | `backend/app/services/query_mode.py` |
| Compare answers (rule-based): scoped evidence + markdown | `backend/app/services/answers/compare_evidence.py`, `compare_render.py` (`format_two_entity_compare_markdown`, `format_multi_entity_compare_markdown`); wired from `answer_generation.py` |
| Quiz answers (rule-based): `Quiz: …` + 3 questions + `Answer Key:` | `backend/app/services/answers/quiz_render.py` (`format_quiz_markdown`); dispatched by `answer_generation.generate_structured_answer` for `answer_mode == "teaching_plus_check"` |
| Summary answers (rule-based): lecture- and topic-scoped layouts | `backend/app/services/answers/summary_render.py` (`format_summary_markdown`); dispatched by `answer_generation.generate_structured_answer` for `answer_mode == "lecture_summary"` |
| Security event rows (auth/admin) | `backend/app/models/security_log.py` (`SecurityLogEntry` → table `audit_events`); `backend/app/services/security_logging.py` (`log_security_event`) |
| Query classification / aliases | `backend/app/services/query_understanding.py` |
| Structured query + decomposition | `backend/app/services/knowledge/structured_query.py` |
| Primary Course Answer (OpenAI / rules) | `backend/app/services/generation/course_generation.py`, `generation/llm.py` |
| Boosted Explanation (Gemini / OpenAI fallback) | `generation/gemini_boost.py`, `generation/llm.py` |
| Study flows (quiz / compare / summary) | `backend/app/routes/study.py`, `backend/app/services/study.py` |
| ORM models (barrel export) | `backend/app/models/__init__.py` → `chat.py`, `analytics.py`, `user.py`, … |
| Admin analytics | `backend/app/routes/admin.py`, `backend/app/services/admin_insights.py` |

---

## Modes and chat API (quick)

- **`POST /api/chat`** requires `session_id` and `message`. **`mode`** (legacy) and **`mode_override`** are optional; default is **auto** (detect from message text). If both are sent, **`mode_override`** wins. Invalid values fall back to **auto**.
- **Response:** **`course_answer`**, **`answer`** (duplicate for clients that expect one field), **`boosted_explanation`**, **`retrieval_confidence`**, **`boost_applied`**, **`assistant_message_id`**, **`mode`** (stable shape below), **`mode_routing`** (internal field names preserved for older clients).
- **`mode` object:** `detected`, `effective`, `confidence`, `signals`, `overridden`, `ambiguous`, optional `candidate_modes`. Built by `mode_metadata_for_api()` in `chat_orchestrator.py` from `mode_routing`.
- **Frontend:** [`ChatPage.jsx`](../frontend/src/pages/ChatPage.jsx) sends **`mode_override`** only when **Override detected mode** is enabled (otherwise the server auto-detects). New sessions from the chat flow use **`mode: auto`**. [`ChatPanel.jsx`](../frontend/src/components/ChatPanel.jsx) shows **Detected mode** / effective from `data.mode`; dedicated **`/api/study`** flows sit under a collapsible **Study tools** section.

---

## Mode-aware rendering contract (chat-mode dispatch)

A new chat reading this should treat this as the source of truth for *which renderer runs for which mode*. The pipeline is **rule-based by default** for everything except plain `chat`-style answers:

1. `query_mode.detect_query_mode` (deterministic, no LLM) classifies the raw text → `chat | quiz | compare | summary`.
2. `query_mode.resolve_effective_mode` reconciles detection with `mode_override` (or legacy `mode`) → `effective_mode`.
3. `retrieval_v2.apply_effective_api_mode` coerces `QueryIntent.query_type` so retrieval matches the routed mode. Quiz retrieval is dispatched by `_STRATEGY[QueryType.QUIZ] = _handle_quiz`, which delegates to `_handle_summary` for single-lecture quiz queries (e.g. *"Test me on Lecture 11"*) and to `_handle_definition` for topic queries.
4. `structured_query._QUERY_TYPE_TO_ANSWER_INTENT` maps `query_type` → `answer_mode` (`compare`, `compare_multi`, `lecture_summary`, `teaching_plus_check`, `direct_definition`, `multi_step_explanation`, …).
5. `course_generation.generate_course_answer` short-circuits to **rule-based** for `compare`, `compare_multi`, `lecture_summary`, `teaching_plus_check` — these never reach an LLM.
6. `answer_generation.generate_structured_answer` dispatches by `plan.answer_mode`:

| `answer_mode` | Renderer | Output starts with |
|---------------|----------|--------------------|
| `compare` | `compare_render.format_two_entity_compare_markdown` | `Course Answer:` + `### Direct Answer` (compare-shaped) |
| `compare_multi` | `compare_render.format_multi_entity_compare_markdown` | `Course Answer:` + entity table |
| `lecture_summary` | `summary_render.format_summary_markdown` | `Summary: Lecture N` (or `Summary: <topic>`) |
| `teaching_plus_check` | `quiz_render.format_quiz_markdown` | `Quiz: Lecture N` (or `Quiz: <topic>`) + `Answer Key:` |
| anything else | the four-block Course Answer (`### Direct Answer` / `### Explanation` / `### Example / Intuition` / `### Why it matters`) | `Course Answer:` |

**Invariants enforced by tests** ([`test_mode_routing.py`](../backend/tests/test_mode_routing.py), [`test_answers_quiz_render.py`](../backend/tests/test_answers_quiz_render.py), [`test_answers_summary_render.py`](../backend/tests/test_answers_summary_render.py)):

- Quiz output never contains `### Direct Answer` / `### Explanation` / `### Example / Intuition` / `### Why it matters` / `Course Answer:`.
- Summary output never contains `### Direct Answer` / `### Explanation` / `Course Answer:`.
- A single-lecture quiz draws **all** evidence (including MC distractors) from the requested lecture.
- True/false slot is always a **true** statement — fabricating false statements is forbidden by the upstream `allow_incorrect_statements` policy.

If you're touching anything in `answer_generation.py` / `course_generation.py` / `retrieval_v2.py`, run the renderer tests **and** `test_mode_routing.py` to keep these invariants intact.

---

## Recent product / API notes (for handoff)

- **Real summary renderer (lecture- and topic-scoped):** `summary_render.py` now drives summary mode end-to-end. Lecture-scoped path is hard-filtered on `lecture_number`, ordered by `chunk_order` / `position` / `order` metadata when present (forward-compatible only — `LectureChunk` doesn't carry the column today) then by `id`, dedupes section heading prefixes, and emits `Summary: Lecture N` + `### Main idea` / `### Key topics` / `### How the topics connect` / `### Study focus`. Topic-scoped path filters chunks to those mentioning the topic *or* one of its `ConceptKB` aliases (so a "recap of MFCCs" never surfaces formant or softmax chunks even when retrieval pulled them in) and emits `Summary: <topic>` + `### Core idea` / `### Key points` / `### Study focus`. New tests in `test_answers_summary_render.py` cover `Summarize Lecture 10`, `Main takeaways from Lecture 10`, `What are the main ideas of Lecture 16?`, `Give me a recap of MFCCs`, plus topic dedupe and `chunk_order` metadata ordering.
- **Compare evidence bundles V2:** New `ConceptEvidenceBundleV2` (`entity_retrieval.py`) carries `concept` / `aliases` / `evidence_chunks` / `core_lines` / `support_score` / `forbidden_hits` / `shared_lines` / `source_metadata` / `confidence` and exposes legacy `concept_id` / `label` / `chunk_ids` / `gap_flags` as read-only properties (`from_legacy_bundle` / `to_legacy_bundle` adapters keep the four-field surface for older call sites). `AnswerPlan.evidence_bundles` is typed `dict[str, EvidenceBundleLike]` (V1 ∪ V2). New `build_bundles_for_compare_v2` and `build_bundles_multi_v2` per-line classify each unit with `classify_line_for_compare`: a line goes to side A's `core_lines` only when it scores strictly higher for A than B and isn't blocked by an A-side forbidden term; lines that score >= 1 for both entities at a min-ratio threshold land in `shared_lines` on **both** bundles. `compare_render.format_two_entity_compare_markdown` now reads `core_lines` / `shared_lines` directly off V2 bundles and emits a new `### What they share` section when both bundles' `support_score` clears `COMPARE_SHARED_MIN_SUPPORT` (0.25); the section is omitted entirely otherwise. New tests in `test_compare_render.py` cover `Compare CNN and MLP`, `CNN vs transformer` (no self-attention leak into CNN, no convolution leak into transformer), `Difference between MFCCs and formants`, `Bias versus variance` shared section, `Contrast softmax and hardmax` (line-disjoint bundles), and a V2 ↔ legacy adapter round-trip.
- **Per-mode dispatch (chat / quiz / compare / summary):** Mode dispatch now routes quiz and summary to dedicated rule-based renderers (`quiz_render.py`, `summary_render.py`), parallel to the existing compare renderer; `course_generation.py` short-circuit is extended to keep `lecture_summary` and `teaching_plus_check` rule-based as well. Quiz retrieval uses `_handle_quiz` in `retrieval_v2.py` (lecture-aware via `_handle_summary` for single-lecture queries). Tests: `test_mode_routing.py`, `test_answers_quiz_render.py`, `test_answers_summary_render.py`, `TestQuizRouting` in `test_retrieval_v2.py`.
- **Mode-routing analytics columns (migration `006`):** `retrieval_logs` gained `mode_detected`, `mode_effective`, `mode_overridden`, `mode_confidence`, `mode_ambiguous`, `mode_signals_json`, `mode_request_source`. Populated per turn from `mode_metadata_for_api()` in `chat_orchestrator.py`. Wrong-routing can be debugged in plain SQL without parsing `messages.payload_json`. See [`schema.md`](../backend/docs/schema.md), [`006_retrieval_log_mode_routing.py`](../backend/migrations/versions/006_retrieval_log_mode_routing.py).
- **Phase 3 compare renderers:** Entity-scoped lines (`compare_evidence.py`), two-way and multi-entity markdown (`compare_render.py`), tests in `test_compare_render.py` and structured pipeline tests. Compare / compare_multi primary paths stay **rule-based** (`course_generation.py` short-circuit).
- **Phase 4 chat contract:** Optional mode fields; response **`mode`** + **`answer`**; see [`chat.py`](../backend/app/routes/chat.py) `_resolve_user_api_mode`.
- **Naming cleanups:** `answer_generation` uses explicit variable names; compare evidence uses `text_normalized` / `phrase` in `_term_hits`; auth security rows use `security_logging` module (not `audit.py`).

---

## Known structural debt (deferred, not bugs)

The latest pass (per-mode renderer dispatch) was scoped to "smallest correct refactor." A new chat encountering complaints about any of the following should treat them as **planned next passes**, not bugs to fix unprompted — design context lives in the rule-based-pipeline brief and the latest plan file under `~/.cursor/plans/`:

- **Retrieval contamination** — queries for one concept still pull in unrelated topics (e.g. *"What is CNN?"* surfacing transformer / residual content). Needs concept-purity / forbidden-topic scoring on top of `retrieval_v2.py`.
- **Direct-answer drift** — the four-block Course Answer's `### Direct Answer` is not yet mode-aware or concept-aware (e.g. *"What is MFCC?"* sometimes opens with formants).
- **Generic filler** — *"Solid intuition here makes the next topics…"*-style boilerplate still appears in some Course Answer closings; should become concept-specific or removed.
- **Compare evidence isolation** — *(addressed by `ConceptEvidenceBundleV2` + `build_bundles_for_compare_v2`)*. Per-line cross-entity filtering keeps Concept A and Concept B from sharing identical evidence; lines that genuinely score for both surface under `### What they share`. Remaining work is concept-purity in **retrieval** (so the candidate pool isn't already polluted), not in compare evidence assembly.
- **HMAC-graded interactive quiz** — `/api/study/quiz/*` (`build_quiz_next` / `build_quiz_reveal`) is intentionally separate from the chat-mode static quiz renderer. Don't merge them; their contracts differ (single-question + token vs. 3-question static markdown).

Bad / underspecified queries (e.g. `"asf"`, `"compare these"`) currently respond with a clarification request — **preserve that behavior**; don't replace it with hallucinated content.

---

## Tunables (env)

See **`backend/.env.example`** and **`backend/app/config.py`**. Examples: `CHAT_RETRIEVAL_TOP_K`, `PIPELINE_RETRY_TOP_K_EXTRA`, `OPENAI_TEMPERATURE_*`, `GEMINI_TEMPERATURE_BOOST`, `CONFIDENCE_THRESHOLD`, `STRUCTURED_PIPELINE_ENABLED`.

---

## Deeper narrative handoffs

- **`progress/entries/2026-04-12-session-handoff-neural-tutor.md`** — Longer session notes (admin insights, Course Answer prompts, Markdown UI, follow-ups).
- **`progress/README.md`** — When to update `CHANGELOG.md`, `README.md`, `entries/`, and schema docs.

---

## Migrations & DB

- Apply migrations from `backend/` with Flask-Migrate / project conventions (see root `README.md`).
- Logical schema reference: **`backend/docs/schema.md`**.
- **Latest revision:** `006` (`retrieval_log_mode_routing` — adds `mode_*` columns to `retrieval_logs`).

---

## Local dev gotchas

- **Run pending migrations against the live dev DB** after pulling: `cd backend && flask --app wsgi db upgrade`. The test suite uses a scratch SQLite DB (per `TestConfig`), so migrations passing in `pytest` does **not** mean the dev `ling487.db` is up to date. Symptom of skipping: `sqlalchemy.exc.OperationalError: ... no such column: retrieval_logs.mode_detected` on every `POST /api/chat` (chat box appears unresponsive). This bit us when shipping migration `006`.
- **Backend tests:** From `backend/`, run `PYTHONPATH=. python -m pytest -q` (the venv lives at `backend/.venv`; `PYTHONPATH=.` is needed because the package isn't installed editable). Targeted runs: `... pytest tests/test_mode_routing.py -q` etc.
- **Don't import `app.services.answers.summary_render` / `quiz_render` at module top-level inside `answer_generation.py` / `course_generation.py`** — keep them function-scoped (matches the existing compare-render pattern) to avoid circular imports during pipeline bootstrap.
