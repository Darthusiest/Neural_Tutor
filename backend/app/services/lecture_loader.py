"""Load LING 487 lecture JSON into `lecture_chunks` for retrieval."""

from __future__ import annotations

import json
import re
from pathlib import Path

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
            explanation = "\n".join(lines)
            topic = f"{title} — {heading}"
            if len(topic) > 512:
                topic = topic[:509] + "..."
            kw = keyword_list(title, heading, lines)
            kw_json = json.dumps(kw)

            if upsert:
                existing = LectureChunk.query.filter_by(
                    lecture_number=lecture_number,
                    topic=topic,
                ).first()
                if existing:
                    existing.keywords = kw_json
                    existing.explanation = explanation
                    existing.example_qa = None
                else:
                    db.session.add(
                        LectureChunk(
                            topic=topic,
                            lecture_number=lecture_number,
                            keywords=kw_json,
                            explanation=explanation,
                            example_qa=None,
                        )
                    )
            else:
                db.session.add(
                    LectureChunk(
                        topic=topic,
                        lecture_number=lecture_number,
                        keywords=kw_json,
                        explanation=explanation,
                        example_qa=None,
                    )
                )
            count += 1

    db.session.commit()
    return count
