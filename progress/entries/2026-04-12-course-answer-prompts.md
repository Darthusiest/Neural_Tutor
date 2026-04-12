# Course Answer OpenAI prompts (2026-04-12)

## What changed

- **`format_generation_prompt_user_message`** sends only structured course text: Question, Concepts, teaching-style hint, Primary Content, Supporting Content (newline-joined prose). No duplicate “Remember” bullets in the user message.
- **`_COURSE_ANSWER_SYSTEM_PROMPT`** in **`llm.py`** holds grounding, paraphrase, anti-repetition, section rules, forbidden jargon, and strict `Course Answer:` + four `###` headings in one place.
- **`output_cleanup`** (`clean_output`, `enforce_structure`) remains the post-pass on model output.

## Follow-ups

- If models still leak internal phrasing, extend `clean_output` patterns or add an optional one-retry path with a stricter system message when sections are missing.
