"""OpenAI-backed Boosted Explanation (server-side only). Not implemented in scaffold."""

from flask import current_app


def generate_boosted_explanation(
    user_question: str,
    retrieved_context: str,
) -> tuple[str | None, dict]:
    """
    Returns (text, usage_meta).

    Scaffold returns (None, {}) when OPENAI_API_KEY is unset.
    """
    if not current_app.config.get("OPENAI_API_KEY"):
        return None, {}
    return None, {}
