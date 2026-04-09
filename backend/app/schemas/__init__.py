"""Pydantic schemas for validated ingestion."""

from app.schemas.lecture_corpus import LectureCorpus, validate_lecture_corpus_dict

__all__ = ["LectureCorpus", "validate_lecture_corpus_dict"]
