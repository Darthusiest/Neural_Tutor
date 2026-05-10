"""Sanity checks for the static 300-case ``l487_eval_suite`` v3 JSON (no app services)."""

from __future__ import annotations

import json
from pathlib import Path

from app.eval.dataset import load_eval_dataset

_BACKEND = Path(__file__).resolve().parent.parent
_KB_PATH = _BACKEND / "data" / "LING487_STRUCTURED_PIPELINE_KB.json"
_V3_PATH = _BACKEND / "data" / "eval" / "l487_eval_suite_v3.json"

_ALLOWED_MODES = frozenset({"chat", "compare", "summary", "quiz"})


def _concept_mentioned(case: dict, concept_id: str, meta: dict) -> bool:
    blob = " ".join(
        [
            str(case.get("id", "")),
            str(case.get("query", "")),
            " ".join(str(x) for x in case.get("must_include", [])),
        ]
    ).lower()
    needles = {concept_id.lower(), concept_id.replace("_", " ").lower()}
    for al in meta.get("aliases") or []:
        t = al.strip().lower()
        if len(t) > 1:
            needles.add(t)
    name = (meta.get("name") or "").strip().lower()
    if name:
        needles.add(name)
        needles.add(name.split(",")[0].strip())
    return any(n and n in blob for n in needles)


def test_v3_suite_metadata_and_count():
    data = json.loads(_V3_PATH.read_text(encoding="utf-8"))
    assert data.get("name") == "l487_eval_suite"
    assert data.get("version") == "3"
    cases = data["cases"]
    assert len(cases) == 300


def test_v3_no_duplicate_ids():
    data = json.loads(_V3_PATH.read_text(encoding="utf-8"))
    ids = [c["id"] for c in data["cases"]]
    assert len(ids) == len(set(ids))


def test_v3_expected_modes():
    data = json.loads(_V3_PATH.read_text(encoding="utf-8"))
    for c in data["cases"]:
        assert c["expected_mode"] in _ALLOWED_MODES, c["id"]


def test_v3_loads_via_dataset_loader():
    meta, cases = load_eval_dataset(_V3_PATH)
    assert meta.get("version") == "3"
    assert len(cases) == 300


def test_v3_every_kb_concept_referenced():
    kb = json.loads(_KB_PATH.read_text(encoding="utf-8"))
    concepts = {c["id"]: c for c in kb["concepts"]}
    data = json.loads(_V3_PATH.read_text(encoding="utf-8"))
    cases = data["cases"]
    for cid, meta in sorted(concepts.items()):
        assert any(_concept_mentioned(c, cid, meta) for c in cases), f"missing coverage for {cid}"


def test_v3_comparison_axes_have_compare_rows():
    kb = json.loads(_KB_PATH.read_text(encoding="utf-8"))
    axes = kb.get("comparison_axes") or {}
    data = json.loads(_V3_PATH.read_text(encoding="utf-8"))
    cmp_cases = [c for c in data["cases"] if c.get("category") == "compare"]
    for key in sorted(axes.keys()):
        prefix = f"cmp_v3_{key}_"
        hits = [c for c in cmp_cases if str(c["id"]).startswith(prefix)]
        assert hits, f"comparison_axes {key!r} has no cmp_v3_{key}_* rows"


def test_v3_every_lecture_has_summary_case():
    kb = json.loads(_KB_PATH.read_text(encoding="utf-8"))
    data = json.loads(_V3_PATH.read_text(encoding="utf-8"))
    ids = {c["id"] for c in data["cases"]}
    for lec in kb["lectures"]:
        n = int(lec["lecture_number"])
        assert f"sum_v3_lec_{n:02d}" in ids, f"lecture {n} missing dedicated summary row"
