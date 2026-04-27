"""Load and validate static eval suite JSON for batch runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalCase:
    id: str
    query: str
    expected_mode: str
    must_include: list[str] = field(default_factory=list)
    must_not_include: list[str] = field(default_factory=list)
    expected_sections: list[str] = field(default_factory=list)
    forbidden_sections: list[str] = field(default_factory=list)
    category: str = ""
    error_tags: list[str] = field(default_factory=list)
    note: str | None = None
    mode: str = "auto"
    mode_override: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


def load_eval_dataset(path: Path) -> tuple[dict[str, Any], list[EvalCase]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if "cases" not in data or not isinstance(data["cases"], list):
        raise ValueError("eval suite JSON must contain a 'cases' array")
    cases: list[EvalCase] = []
    for i, item in enumerate(data["cases"]):
        if not isinstance(item, dict):
            raise ValueError(f"cases[{i}] must be an object")
        case_id = (item.get("id") or "").strip()
        if not case_id:
            raise ValueError(f"cases[{i}] is missing 'id'")
        query = (item.get("query") or "").strip()
        if not query:
            raise ValueError(f"case {case_id!r} is missing 'query'")
        mode = (item.get("mode") or "auto").strip().lower()
        mode_override = (item.get("mode_override") or "").strip().lower()
        if mode not in ("auto", "chat", "quiz", "compare", "summary"):
            mode = "auto"
        if mode_override and mode_override not in ("auto", "chat", "quiz", "compare", "summary"):
            mode_override = ""
        cases.append(
            EvalCase(
                id=case_id,
                query=query,
                expected_mode=(item.get("expected_mode") or "chat").strip().lower(),
                must_include=[str(x) for x in (item.get("must_include") or []) if x is not None],
                must_not_include=[str(x) for x in (item.get("must_not_include") or []) if x is not None],
                expected_sections=[str(x) for x in (item.get("expected_sections") or []) if x is not None],
                forbidden_sections=[str(x) for x in (item.get("forbidden_sections") or []) if x is not None],
                category=(item.get("category") or "").strip(),
                error_tags=[str(x) for x in (item.get("error_tags") or []) if x is not None],
                note=(item.get("note") or None),
                mode=mode,
                mode_override=mode_override,
                raw=dict(item),
            )
        )
    return data, cases


def case_expected_behavior_dict(case: EvalCase) -> dict[str, Any]:
    """Fields persisted on EvaluationCaseResult.expected_behavior_json."""
    return {
        "expected_mode": case.expected_mode,
        "must_include": case.must_include,
        "must_not_include": case.must_not_include,
        "expected_sections": case.expected_sections,
        "forbidden_sections": case.forbidden_sections,
        "category": case.category,
        "error_tags": case.error_tags,
        "mode": case.mode,
        "mode_override": case.mode_override,
    }
