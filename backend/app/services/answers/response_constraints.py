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
    # User-requested topics to omit (e.g. "do not mention transformers").
    forbidden_topics: list[str] = field(default_factory=list)
    #: "Define … in one sentence" — collapse output to a single grounded sentence.
    one_sentence: bool = False

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
            "forbidden_topics": list(self.forbidden_topics),
            "one_sentence": self.one_sentence,
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
_ONE_SENTENCE_RE = re.compile(r"\bin one sentence\b", re.IGNORECASE)
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

# User exclusions: "do not mention X", "without mentioning X", etc.
_DO_NOT_MENTION_RE = re.compile(
    r"(?:\bdo\s+not\s+mention\b|\bdon'?t\s+mention\b)\s+(.+?)(?:\.|$)",
    re.IGNORECASE | re.DOTALL,
)
_WITHOUT_MENTIONING_RE = re.compile(
    r"\bwithout\s+mentioning\s+(.+?)(?:\.|$)",
    re.IGNORECASE | re.DOTALL,
)
# "Explain X without Y discussion" / "without Y mentioning …"
_WITHOUT_TOPIC_DETAIL_RE = re.compile(
    r"\bwithout\s+(.+?)\s+(?:discussion|mentioning|details?|coverage)\b",
    re.IGNORECASE | re.DOTALL,
)
_EXPLAIN_WITHOUT_TAIL_RE = re.compile(
    r"\bexplain\b.+?\bwithout\s+(.+?)(?:\.|$)",
    re.IGNORECASE | re.DOTALL,
)
_EXCLUDE_TOPICS_RE = re.compile(
    r"\bexclude\s+(.+?)(?:\.|$)",
    re.IGNORECASE | re.DOTALL,
)
_NO_MENTION_NEURAL_NETS_RE = re.compile(
    r"\bdo\s+not\s+mention\s+(?:neural\s+networks?|nns?)\b",
    re.IGNORECASE,
)


def _split_topic_clause(clause: str) -> list[str]:
    """Split a clause like 'transformers or residuals' into topic strings."""
    clause = (clause or "").strip()
    if not clause:
        return []
    parts = re.split(r"\s*(?:,|\bor\b|\band\b)\s*", clause, flags=re.IGNORECASE)
    out: list[str] = []
    for p in parts:
        t = p.strip().rstrip(".,;:")
        if t and len(t) >= 2:
            out.append(t)
    return out


def _collect_forbidden_topics(query: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []

    def add_many(phrases: list[str]) -> None:
        for ph in phrases:
            key = ph.strip().lower()
            if key and key not in seen:
                seen.add(key)
                ordered.append(ph.strip())

    for rx in (_DO_NOT_MENTION_RE, _WITHOUT_MENTIONING_RE, _EXCLUDE_TOPICS_RE):
        for m in rx.finditer(query):
            add_many(_split_topic_clause(m.group(1)))

    for m in _WITHOUT_TOPIC_DETAIL_RE.finditer(query):
        add_many(_split_topic_clause(m.group(1)))
    for m in _EXPLAIN_WITHOUT_TAIL_RE.finditer(query):
        add_many(_split_topic_clause(m.group(1)))

    if _NO_MENTION_NEURAL_NETS_RE.search(query):
        add_many(["neural networks", "neural network"])

    return ordered


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
    if _ONE_SENTENCE_RE.search(query):
        rc.one_sentence = True
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
    rc.forbidden_topics = _collect_forbidden_topics(query)
    return rc
