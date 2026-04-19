"""Deterministic parsing of user formatting / pedagogy constraints from raw queries."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ResponseConstraints:
    """Controls section suppression, repetition, and safety."""

    no_examples: bool = False
    no_analogies: bool = False
    intuition_only: bool = False
    brief: bool = False
    exact_explanation_count: int | None = None
    repeat_explanation_times: int | None = None
    allow_incorrect_statements: bool = False
    pipeline_summary_requested: bool = False

    def to_dict(self) -> dict:
        return {
            "no_examples": self.no_examples,
            "no_analogies": self.no_analogies,
            "intuition_only": self.intuition_only,
            "brief": self.brief,
            "exact_explanation_count": self.exact_explanation_count,
            "repeat_explanation_times": self.repeat_explanation_times,
            "allow_incorrect_statements": self.allow_incorrect_statements,
            "pipeline_summary_requested": self.pipeline_summary_requested,
        }


_NO_EXAMPLES_RE = re.compile(
    r"\b(?:no examples?|without examples?|zero examples?|don'?t (?:give|use|include) examples?)\b",
    re.IGNORECASE,
)
_NO_ANALOGY_RE = re.compile(
    r"\b(?:no analog(?:y|ies)|without analog(?:y|ies)|don'?t use analog(?:y|ies))\b",
    re.IGNORECASE,
)
_INTUITION_ONLY_RE = re.compile(
    r"\b(?:intuition only|only intuition|conceptual only|no (?:math|equations?|formulas?))\b",
    re.IGNORECASE,
)
_BRIEF_RE = re.compile(r"\b(?:brief|short answer|in one paragraph|tldr)\b", re.IGNORECASE)
_EXACT_N_RE = re.compile(
    r"\b(?:give|write|provide)\s+(\d+)\s+(?:distinct\s+)?(?:explanation|explanations|ways)\b",
    re.IGNORECASE,
)
_REPEAT_TWICE_RE = re.compile(
    r"\b(?:repeat|say)\s+(?:the\s+)?(?:explanation|answer)\s+(?:twice|two times|2 times)\b",
    re.IGNORECASE,
)
_INCORRECT_STMT_RE = re.compile(
    r"\b(?:both\s+)?correct\s+and\s+incorrect\s+statements?\b",
    re.IGNORECASE,
)
_PIPELINE_RE = re.compile(r"\b(?:mfcc\s+)?pipeline\b|\bpipeline\s+of\b", re.IGNORECASE)


def parse_response_constraints(query: str) -> ResponseConstraints:
    """Extract constraints from free text (best-effort, deterministic)."""
    rc = ResponseConstraints()
    if _NO_EXAMPLES_RE.search(query):
        rc.no_examples = True
    if _NO_ANALOGY_RE.search(query):
        rc.no_analogies = True
    if _INTUITION_ONLY_RE.search(query):
        rc.intuition_only = True
    if _BRIEF_RE.search(query):
        rc.brief = True
    m = _EXACT_N_RE.search(query)
    if m:
        try:
            n = int(m.group(1))
            if 1 <= n <= 12:
                rc.exact_explanation_count = n
        except ValueError:
            pass
    if _REPEAT_TWICE_RE.search(query):
        rc.repeat_explanation_times = 2
    if _INCORRECT_STMT_RE.search(query):
        rc.allow_incorrect_statements = True
    if _PIPELINE_RE.search(query):
        rc.pipeline_summary_requested = True
    return rc
