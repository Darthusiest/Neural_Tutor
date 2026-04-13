"""Post-process Course Answer text: strip leaked system phrasing, normalize whitespace, enforce sections."""

from __future__ import annotations

import re

# Lines or phrases that should not reach students (leaked retrieval / debug)
_BANNED_LINE_PATTERNS = [
    r"(?im)^.*\bkeywords?\s*[:=].*$",
    r"(?im)^.*\blecture\s*scope\b.*$",
    r"(?im)^.*\bconcept\s*graph\b.*$",
    r"(?im)^.*\bretrieved\s+(from|chunks?)\b.*$",
    r"(?im)^.*\bchunk\s*(id|data|shows?)\b.*$",
    r"(?im)^.*\bindexed\b.*$",
    r"(?im)^.*\bsource_material\b.*$",
    r"(?im)^.*\bdebug\b.*$",
    # Planner / outline scaffolding that must not reach students (LLM echo).
    r"(?im)^\s*\*?\*?In one line:.*$",
    r"(?im)^\s*\*?\*?First idea:.*$",
    r"(?im)^\s*\*?\*?Second idea:.*$",
    r"(?im)^\s*\*?\*?Putting them together:.*$",
]


def clean_output(answer: str) -> str:
    """Remove banned patterns and collapse excessive blank lines."""
    if not answer:
        return ""
    out = answer
    for pattern in _BANNED_LINE_PATTERNS:
        out = re.sub(pattern, "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _has_example_section(text: str) -> bool:
    return "### Example / Intuition" in text or re.search(r"###\s*Example\s", text) is not None


def _has_heading(text: str, h: str) -> bool:
    return h in text


def enforce_structure(answer: str) -> str:
    """
    If required ``###`` headings are missing, append minimal placeholders so the UI stays consistent.

    Does not call the LLM again (no ``regenerate_with_strict_prompt`` in production to avoid cost/latency).
    """
    a = answer.strip()
    if not a:
        return _minimal_skeleton()

    missing: list[str] = []
    if not _has_heading(a, "### Direct Answer"):
        missing.append("### Direct Answer")
    if not _has_heading(a, "### Explanation"):
        missing.append("### Explanation")
    if not _has_example_section(a):
        missing.append("### Example / Intuition")
    if not _has_heading(a, "### Why it matters"):
        missing.append("### Why it matters")

    if not missing:
        return a

    append_parts: list[str] = []
    for h in missing:
        if h == "### Direct Answer":
            append_parts.append("### Direct Answer\n\n(Short definition synthesized from the notes above.)")
        elif h == "### Explanation":
            append_parts.append(
                "### Explanation\n\n- Expand on the ideas using only what fits the question."
            )
        elif h == "### Example / Intuition":
            append_parts.append(
                "### Example / Intuition\n\nA concrete mini-example will sharpen this—ask for one if you want numbers."
            )
        elif h == "### Why it matters":
            append_parts.append(
                "### Why it matters\n\nThis idea shows up wherever the course talks about modeling decisions and interpretation."
            )

    return a.rstrip() + "\n\n" + "\n\n".join(append_parts)


def _minimal_skeleton() -> str:
    return (
        "Course Answer:\n\n"
        "### Direct Answer\n\n"
        "(No content generated.)\n\n"
        "### Explanation\n\n"
        "- Please try again with a clearer course term.\n\n"
        "### Example / Intuition\n\n"
        "Ask for a worked example in your next message.\n\n"
        "### Why it matters\n\n"
        "Staying aligned with course vocabulary keeps answers useful for exams and assignments."
    )
