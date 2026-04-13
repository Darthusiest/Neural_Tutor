# Multi-track implementation (session)

Implemented the staged plan: config flags, Render blueprint, email hygiene (no `dev_reset_token` outside debug unless explicitly allowed), auth (verification tokens, lockout, audit events), embedding + hybrid retrieval with `embed-chunks` CLI, study-mode copy + optional structured pipeline overlay, admin cost/content-quality endpoints + UI sparkline, offline `scripts/boost_eval.py`.

**Tradeoffs:** Hybrid fusion uses normalized lexical scores from chunk hits; USD cost is estimate-only via `LLM_COST_USD_PER_MTOKENS`. Email verification when `EMAIL_VERIFICATION_REQUIRED=1` needs Resend; otherwise registration auto-verifies. Existing users get `email_verified_at` backfilled on migration.

**Follow-ups:** Dedicated `/verify-email` SPA route; PostgreSQL on Render for durable DB; tune hybrid weights per course.
