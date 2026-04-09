from __future__ import annotations

import json
import tempfile
from pathlib import Path

from app.extensions import db
from app.models import LectureChunk
from app.services.lecture_loader import import_lecture_json


def test_import_uses_source_text_when_no_content(app):
    payload = {
        "lectures": [
            {
                "lecture_number": 1,
                "title": "Test Lecture",
                "sections": [
                    {
                        "heading": "Section A",
                        "source_text": "Line one.\nLine two.",
                        "clean_explanation": "Short explanation.",
                    }
                ],
            }
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        path = Path(f.name)
    try:
        with app.app_context():
            LectureChunk.query.delete()
            db.session.commit()
            n = import_lecture_json(path, upsert=False)
            assert n == 1
            row = LectureChunk.query.one()
            assert row.chunk_key
            assert "Line one" in row.source_excerpt
            assert row.clean_explanation == "Short explanation."
            assert "Test Lecture" in row.topic
    finally:
        path.unlink(missing_ok=True)


def test_import_curated_keywords_prepended(app):
    payload = {
        "lectures": [
            {
                "lecture_number": 99,
                "title": "Curated Test",
                "sections": [
                    {
                        "heading": "Sec",
                        "content": ["alpha beta gamma delta"],
                        "keywords": ["manualterm"],
                    }
                ],
            }
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        path = Path(f.name)
    try:
        with app.app_context():
            LectureChunk.query.delete()
            db.session.commit()
            n = import_lecture_json(path, upsert=False)
            assert n == 1
            row = LectureChunk.query.one()
            kws = json.loads(row.keywords)
            assert kws[0] == "manualterm"
    finally:
        path.unlink(missing_ok=True)
