# Hybrid pipeline: rule-based control, OpenAI primary answer, Gemini boost

**Date:** 2026-04-12

**Summary:** Separated **primary Course Answer** generation (`course_generation.generate_course_answer` → OpenAI plan-constrained answer, rule-based fallback) from **secondary Boosted Explanation** (Gemini REST when `GEMINI_API_KEY` / `GOOGLE_API_KEY` is set; OpenAI boost as fallback). Added `ValidationResult.severity` (`pass` / `weak` / `fail`) and `should_use_gemini_boost` so boost is driven by validation, confidence, synthesis-style complexity, and user phrasing (legacy non-structured path keeps toggle / low confidence / mode). Chat `payload_json` records `primary_model`, `validation_severity`, `boost_provider`, `boost_reason`, `query_complexity`.

**Config:** `PRIMARY_COURSE_ANSWER_OPENAI` overrides legacy `LLM_ANSWER_GENERATION`; defaults favor primary LLM when env sets `LLM_ANSWER_GENERATION` to `1`. `GEMINI_MODEL`, `GEMINI_TIMEOUT_SEC` for the boost call.

**Documentation:** [`README.md`](../../README.md) env-var table; [`backend/docs/schema.md`](../../backend/docs/schema.md) for `payload_json` / validation JSON; [`backend/docs/AUTH_LOCAL.md`](../../backend/docs/AUTH_LOCAL.md) local `.env` without `.env.example`; [`CHANGELOG.md`](../../CHANGELOG.md) known limitations for Gemini.

**Follow-ups:** Tune boost gating if synthesis-heavy sessions over-trigger Gemini; add metrics on `boost_provider` win-rate; persist `model_name` / `provider_name` on `response_variants` from orchestrator.
