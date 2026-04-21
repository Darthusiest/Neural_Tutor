# Project progress track

For the **current product/API overview** and setup steps, see the root [`README.md`](../README.md).

For a **concise shipped-features list** (release-style), see [`CHANGELOG.md`](../CHANGELOG.md) at the repo root.

**New chat / AI session:** use [`context-for-new-chat.md`](context-for-new-chat.md) — includes a **copy-paste prompt template**, constraints you want from the assistant, and hot paths. For a longer narrative handoff, see [`entries/2026-04-12-session-handoff-neural-tutor.md`](entries/2026-04-12-session-handoff-neural-tutor.md).

This folder holds a **narrative running record**: design decisions, retrieval behavior notes, bugs fixed, and follow-ups—especially what you would not put in a one-line changelog bullet.

---

## Policy: update documentation when you change behavior

**Every meaningful change** (feature, fix, refactor that affects API, data, retrieval, auth, or UX) should touch **at least one** of the following before you consider the work done:

| What changed | Update |
|--------------|--------|
| User-visible behavior, API routes, env vars, config keys | Root [`README.md`](../README.md) (Current status, API table, setup) |
| DB tables, columns, migrations | [`backend/docs/schema.md`](../backend/docs/schema.md) |
| Admin analytics HTTP API | [`backend/docs/admin_insights.md`](../backend/docs/admin_insights.md) when routes or behavior change |
| Auth, CSRF, local testing quirks | [`backend/docs/AUTH_LOCAL.md`](../backend/docs/AUTH_LOCAL.md) if applicable |
| **Release-style “what shipped”** | [`CHANGELOG.md`](../CHANGELOG.md) — add under `[Unreleased]` or a new dated section |
| **Design / rationale / tuning notes** | New file under [`entries/`](entries/) — `YYYY-MM-DD-short-slug.md` |
| Frontend-only (routes, env) | [`frontend/README.md`](../frontend/README.md) if it affects local dev or build |

**Minimum bar:** if you would tell a teammate in Slack what you did, **CHANGELOG.md** gets a line (or **progress/entries/** gets a short note if the change is internal-only).

Do **not** let the root README drift (e.g. “LLM is a stub” when it is wired): readers trust it first.

---

## How to add a progress entry

1. Create a new file under [`entries/`](entries/).
2. Prefer a **sortable name**: `YYYY-MM-DD-short-slug.md` (second entry same day: `2026-04-09-topic-2.md`).
3. Keep each entry **focused** (one theme per file is fine).

## Suggested sections (pick what applies)

- **Summary** — What changed or what you learned.
- **Changes** — Commits, areas of the codebase, migrations.
- **Decisions** — Why an approach was chosen (or rejected).
- **Inferences / observations** — Model behavior, retrieval quality, cost/latency notes.
- **Follow-ups** — Concrete next tasks or open questions.

## Changelog vs progress entry

| | `CHANGELOG.md` | `progress/entries/*.md` |
|---|----------------|-------------------------|
| **Purpose** | Shippable deltas; easy to scan | Context, tradeoffs, experiments |
| **Length** | Short bullets | As long as needed |
| **Audience** | Future you + release notes | Engineers tuning the system |

## Archive

Older notes stay in `entries/`; clear titles keep search and chronology useful.
