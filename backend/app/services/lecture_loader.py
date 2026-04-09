"""Load LING 487 lecture JSON into `lecture_chunks` for retrieval."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.extensions import db
from app.models import LectureChunk

_STOP = frozenset(
    "the a an and or but if in on at to for of as is was are were be been being "
    "it its this that these those with from by not no yes do does did so than then "
    "how what when where which who whom into over out up we our your they them their "
    "can could should would will just like one two all any each some such than".split()
)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def keyword_list(lecture_title: str, heading: str, lines: list[str]) -> list[str]:
    blob = f"{lecture_title} {heading} {' '.join(lines)}"
    seen: set[str] = set()
    out: list[str] = []
    for t in _tokens(blob):
        if len(t) < 3 or t in _STOP:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= 48:
            break
    return out


def _normalize_sample_questions(sec: dict[str, Any]) -> str:
    if "sample_questions" in sec and sec["sample_questions"] is not None:
        raw = sec["sample_questions"]
        if isinstance(raw, str):
            return raw
        if isinstance(raw, list):
            return json.dumps([str(x).strip() for x in raw if str(x).strip()])
    if sec.get("sample_question"):
        return json.dumps([str(sec["sample_question"]).strip()])
    return "[]"


def _sample_answer(sec: dict[str, Any]) -> str | None:
    v = sec.get("sample_answer")
    if v is None or v == "":
        return None
    return str(v).strip() or None


def _clean_explanation(sec: dict[str, Any], source_excerpt: str) -> str:
    for key in ("clean_explanation", "clean"):
        v = sec.get(key)
        if v is not None and str(v).strip():
            return str(v).strip()
    return source_excerpt


def import_lecture_json(path: Path | str, *, upsert: bool = False) -> int:
    """
    Load sections from the JSON file into `lecture_chunks`.

    When upsert is False (default), replace all rows. When True, merge: update rows
    matching (lecture_number, topic), insert new sections otherwise.
    Returns the number of rows written (inserts + updates).
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    if not upsert:
        LectureChunk.query.delete()
        db.session.commit()

    count = 0
    for lec in data.get("lectures", []):
        lecture_number = int(lec["lecture_number"])
        title = str(lec.get("title", "")).strip()
        for sec in lec.get("sections", []):
            heading = str(sec.get("heading", "")).strip()
            lines = [str(s).strip() for s in sec.get("content", []) if str(s).strip()]
            source_excerpt = "\n".join(lines)
            topic = f"{title} — {heading}"
            if len(topic) > 512:
                topic = topic[:509] + "..."
            kw = keyword_list(title, heading, lines)
            kw_json = json.dumps(kw)
            clean = _clean_explanation(sec, source_excerpt)
            sample_q_json = _normalize_sample_questions(sec)
            sample_ans = _sample_answer(sec)

            if upsert:
                existing = LectureChunk.query.filter_by(
                    lecture_number=lecture_number,
                    topic=topic,
                ).first()
                if existing:
                    existing.keywords = kw_json
                    existing.source_excerpt = source_excerpt
                    existing.clean_explanation = clean
                    existing.sample_questions = sample_q_json
                    existing.sample_answer = sample_ans
                else:
                    db.session.add(
                        LectureChunk(
                            topic=topic,
                            lecture_number=lecture_number,
                            keywords=kw_json,
                            source_excerpt=source_excerpt,
                            clean_explanation=clean,
                            sample_questions=sample_q_json,
                            sample_answer=sample_ans,
                        )
                    )
            else:
                db.session.add(
                    LectureChunk(
                        topic=topic,
                        lecture_number=lecture_number,
                        keywords=kw_json,
                        source_excerpt=source_excerpt,
                        clean_explanation=clean,
                        sample_questions=sample_q_json,
                        sample_answer=sample_ans,
                    )
                )
            count += 1

    db.session.commit()
    return count
