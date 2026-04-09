"""Load LING 487 lecture JSON into `lecture_chunks` for retrieval."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.extensions import db
from app.models import LectureChunk
from app.schemas.lecture_corpus import LectureCorpus, validate_chunk_keys_unique, validate_lecture_corpus_dict
from app.services.domain_knowledge import (
    expand_term,
    get_concept_family_for_lecture,
    infer_chunk_type,
)
from app.services.lecture_chunk_key import derive_chunk_key

_STOP = frozenset(
    "the a an and or but if in on at to for of as is was are were be been being "
    "it its this that these those with from by not no yes do does did so than then "
    "how what when where which who whom into over out up we our your they them their "
    "can could should would will just like one two all any each some such than".split()
)


def _keyword_cap() -> int:
    try:
        from flask import has_app_context, current_app

        if has_app_context():
            return int(current_app.config.get("LECTURE_KEYWORD_CAP", 48))
    except Exception:
        pass
    from app.config import Config

    return Config.LECTURE_KEYWORD_CAP


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def keyword_list(lecture_title: str, heading: str, lines: list[str], *, cap: int) -> list[str]:
    blob = f"{lecture_title} {heading} {' '.join(lines)}"
    seen: set[str] = set()
    out: list[str] = []
    for t in _tokens(blob):
        if len(t) < 3 or t in _STOP:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= cap:
            break
    return out


def merge_keywords(
    curated: list[str] | None,
    extra: list[str] | None,
    auto: list[str],
    *,
    cap: int,
) -> list[str]:
    """Curated and extra lists first (deduped), then auto-derived tokens until ``cap``."""
    seen: set[str] = set()
    out: list[str] = []

    def _add_many(items: list[str]) -> None:
        for raw in items:
            t = str(raw).strip().lower()
            if len(t) < 2:
                continue
            if t not in seen:
                seen.add(t)
                out.append(str(raw).strip())
            if len(out) >= cap:
                return

    if curated:
        _add_many(curated)
    if len(out) < cap and extra:
        _add_many(extra)
    if len(out) < cap:
        _add_many(auto)
    return out[:cap]


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


def _source_excerpt_from_section(sec: dict[str, Any]) -> str:
    """
    Prefer explicit ``source_excerpt`` or ``source_text``, else join ``content`` lines.
    """
    for key in ("source_excerpt", "source_text"):
        v = sec.get(key)
        if v is not None and str(v).strip():
            return str(v).strip()
    lines = [str(s).strip() for s in sec.get("content", []) if str(s).strip()]
    return "\n".join(lines)


def import_lecture_json(path: Path | str, *, upsert: bool = False) -> int:
    """
    Load sections from the JSON file into `lecture_chunks`.

    JSON is validated (Pydantic) before insert. Each section has a stable ``chunk_key``
    (explicit ``chunk_key`` / ``section_id`` in JSON, or derived from lecture number,
    section index, and heading slug).

    When upsert is False (default), replace all rows. When True, merge: update rows
    matching ``chunk_key``, insert new sections otherwise.
    Returns the number of rows written (inserts + updates).
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw_data = json.load(f)

    corpus: LectureCorpus = validate_lecture_corpus_dict(raw_data)

    keys: list[str] = []
    for lec in corpus.lectures:
        for idx, sec in enumerate(lec.sections):
            explicit = sec.chunk_key or sec.section_id
            keys.append(derive_chunk_key(lec.lecture_number, sec.heading, idx, explicit))
    validate_chunk_keys_unique(keys)

    if not upsert:
        LectureChunk.query.delete()
        db.session.commit()

    cap = _keyword_cap()
    count = 0
    key_iter = iter(keys)

    for lec in corpus.lectures:
        lecture_number = lec.lecture_number
        title = lec.title
        for sec in lec.sections:
            chunk_key = next(key_iter)
            sec_dict = sec.model_dump(mode="python")
            heading = sec.heading
            source_excerpt = _source_excerpt_from_section(sec_dict)
            lines = [str(s).strip() for s in sec_dict.get("content", []) if str(s).strip()]
            if not lines and source_excerpt:
                lines = [ln for ln in source_excerpt.split("\n") if ln.strip()]
            topic = f"{title} — {heading}"
            if len(topic) > 512:
                topic = topic[:509] + "..."
            auto_kw = keyword_list(title, heading, lines, cap=cap)

            alias_kw: list[str] = []
            for kw in auto_kw[:12]:
                for a in expand_term(kw):
                    if a not in alias_kw and a not in auto_kw:
                        alias_kw.append(a)

            merged = merge_keywords(
                sec.keywords,
                sec.keywords_extra,
                auto_kw + alias_kw,
                cap=cap,
            )
            kw_json = json.dumps(merged)
            clean = _clean_explanation(sec_dict, source_excerpt)
            sample_q_json = _normalize_sample_questions(sec_dict)
            sample_ans = _sample_answer(sec_dict)

            chunk_type = infer_chunk_type(heading)
            concept_family = get_concept_family_for_lecture(lecture_number)

            if upsert:
                existing = LectureChunk.query.filter_by(chunk_key=chunk_key).first()
                if existing:
                    existing.lecture_number = lecture_number
                    existing.topic = topic
                    existing.keywords = kw_json
                    existing.source_excerpt = source_excerpt
                    existing.clean_explanation = clean
                    existing.sample_questions = sample_q_json
                    existing.sample_answer = sample_ans
                    existing.chunk_type = chunk_type
                    existing.concept_family = concept_family
                else:
                    db.session.add(
                        LectureChunk(
                            chunk_key=chunk_key,
                            topic=topic,
                            lecture_number=lecture_number,
                            keywords=kw_json,
                            source_excerpt=source_excerpt,
                            clean_explanation=clean,
                            sample_questions=sample_q_json,
                            sample_answer=sample_ans,
                            chunk_type=chunk_type,
                            concept_family=concept_family,
                        )
                    )
            else:
                db.session.add(
                    LectureChunk(
                        chunk_key=chunk_key,
                        topic=topic,
                        lecture_number=lecture_number,
                        keywords=kw_json,
                        source_excerpt=source_excerpt,
                        clean_explanation=clean,
                        sample_questions=sample_q_json,
                        sample_answer=sample_ans,
                        chunk_type=chunk_type,
                        concept_family=concept_family,
                    )
                )
            count += 1

    db.session.commit()
    return count
