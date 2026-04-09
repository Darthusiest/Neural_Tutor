"""Tests for LING 487 domain knowledge module."""

from __future__ import annotations

from app.services.domain_knowledge import (
    correct_typos,
    expand_term,
    expand_terms_for_query,
    extract_lecture_range,
    fuzzy_match_domain_term,
    get_aliases,
    get_canonical,
    get_concept_family_for_lecture,
    get_related_lectures,
    infer_chunk_type,
)


class TestAliases:
    def test_backprop_canonical(self):
        assert get_canonical("backprop") == "backpropagation"

    def test_dp_canonical(self):
        assert get_canonical("dp") == "dynamic programming"

    def test_mfcc_aliases(self):
        aliases = get_aliases("mfcc")
        assert "mfccs" in aliases
        assert "mfcc" in aliases

    def test_qkv_aliases(self):
        aliases = get_aliases("qkv")
        assert "query key value" in aliases

    def test_unknown_term(self):
        assert get_canonical("xyznoterm") is None
        assert get_aliases("xyznoterm") == frozenset()

    def test_expand_backprop(self):
        expanded = expand_term("backprop")
        assert "backpropagation" in expanded
        assert "backprop" in expanded

    def test_expand_terms_for_query(self):
        extra = expand_terms_for_query(["dp"])
        assert "dynamic" in extra
        assert "programming" in extra

    def test_bidirectional(self):
        assert get_canonical("cnn") == "convolutional neural network"
        assert "cnn" in get_aliases("convolutional neural network")


class TestConceptFamilies:
    def test_lecture_4_family(self):
        assert get_concept_family_for_lecture(4) == "neural_network_foundations"

    def test_lecture_10_family(self):
        assert get_concept_family_for_lecture(10) == "speech_processing"

    def test_unknown_lecture(self):
        assert get_concept_family_for_lecture(999) is None


class TestRelatedLectures:
    def test_lecture_14_related(self):
        related = get_related_lectures(14)
        assert 13 in related
        assert 15 in related

    def test_lecture_7_isolated(self):
        assert get_related_lectures(7) == []


class TestChunkType:
    def test_core_idea(self):
        assert infer_chunk_type("Core Idea") == "definition"

    def test_analogy(self):
        assert infer_chunk_type("Analogy") == "analogy"

    def test_steps(self):
        assert infer_chunk_type("Steps") == "process"

    def test_unknown_heading(self):
        assert infer_chunk_type("Some Random Section") == "definition"


class TestFuzzyMatching:
    def test_close_typo(self):
        result = fuzzy_match_domain_term("backpropagtion")
        assert result == "backpropagation"

    def test_short_token_skipped(self):
        assert fuzzy_match_domain_term("dp") is None

    def test_no_match_for_gibberish(self):
        assert fuzzy_match_domain_term("xyzwqrst") is None

    def test_correct_typos_map(self):
        corrections = correct_typos(["backpropagtion", "softmax"])
        assert "backpropagtion" in corrections
        assert corrections["backpropagtion"] == "backpropagation"
        assert "softmax" not in corrections


class TestLectureRange:
    def test_through(self):
        assert extract_lecture_range("lectures 13 through 15") == [13, 14, 15]

    def test_dash(self):
        assert extract_lecture_range("lec 10-12") == [10, 11, 12]

    def test_no_range(self):
        assert extract_lecture_range("explain backprop") == []
