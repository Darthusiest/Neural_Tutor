# Conversational no-chunk responses (2026-04-12)

When lexical retrieval returns **no chunks**, `handle_chat_turn` does not use the structured pipeline’s generated answer or `format_course_answer`; it routes through **`conversational_responses`**: **`classify_no_match_query`** (greeting vs short acknowledgement vs off-topic) and **`varied_no_chunk_course_answer`** (rotating templates so replies are not identical every time).

**Boost** is forced off when `len(chunks) == 0` so we do not call Gemini/OpenAI for pure steering turns.

Copy was revised toward **multi-paragraph, conversational prose** with example questions embedded in the explanation (ChatGPT-like), using bullet lists only in a few variants for optional “quick prompt” ideas.

**Tests:** `backend/tests/test_conversational_responses.py`.

**Docs:** `README.md` (Current status, `POST /api/chat`), `CHANGELOG.md` [Unreleased], `backend/docs/schema.md` (`messages.payload_json`).
