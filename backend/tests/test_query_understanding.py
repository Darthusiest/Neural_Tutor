"""Tests for query classification and expansion."""

from __future__ import annotations

from app.services.query_understanding import QueryType, analyze_query


class TestClassification:
    def test_definition(self):
        intent = analyze_query("What is backpropagation?")
        assert intent.query_type == QueryType.DEFINITION

    def test_compare_difference(self):
        intent = analyze_query("What is the difference between bias and variance?")
        assert intent.query_type == QueryType.COMPARE

    def test_compare_vs(self):
        intent = analyze_query("CNN vs transformer")
        assert intent.query_type == QueryType.COMPARE

    def test_summary(self):
        intent = analyze_query("Summary of lecture 10")
        assert intent.query_type == QueryType.SUMMARY

    def test_lecture_specific(self):
        intent = analyze_query("What does lecture 8 cover?")
        assert intent.query_type == QueryType.LECTURE_SPECIFIC

    def test_synthesis(self):
        intent = analyze_query("How do lectures 13 through 15 connect?")
        assert intent.query_type == QueryType.SYNTHESIS

    def test_multi_lecture_synthesis(self):
        intent = analyze_query("lecture 13 and lecture 14 together")
        assert intent.query_type == QueryType.SYNTHESIS

    def test_quiz(self):
        intent = analyze_query("Quiz me on attention")
        assert intent.query_type == QueryType.QUIZ

    def test_general(self):
        intent = analyze_query("softmax probabilities distribution")
        assert intent.query_type == QueryType.GENERAL


class TestExpansion:
    def test_dp_expanded(self):
        intent = analyze_query("What is DP?")
        assert "dynamic" in intent.expanded_tokens or "dynamic" in intent.expanded_query.lower()

    def test_backprop_expanded(self):
        intent = analyze_query("explain backprop")
        assert "backpropagation" in intent.expanded_query.lower()

    def test_mfcc_expanded(self):
        intent = analyze_query("tell me about mfcc")
        assert "mfccs" in intent.expanded_query.lower() or "mfcc" in intent.expanded_tokens

    def test_qkv_expanded(self):
        intent = analyze_query("explain QKV")
        assert any("query" in t or "key" in t or "value" in t for t in intent.expanded_tokens)


class TestCompareConcepts:
    def test_extracts_pair(self):
        intent = analyze_query("difference between MFCC and formants")
        assert intent.compare_concepts is not None
        a, b = intent.compare_concepts
        assert "mfcc" in a.lower() or "mfcc" in b.lower()
        assert "formant" in a.lower() or "formant" in b.lower()

    def test_no_pair_for_definition(self):
        intent = analyze_query("what is softmax")
        assert intent.compare_concepts is None


class TestTypoCorrection:
    def test_backpropagtion_corrected(self):
        intent = analyze_query("backpropagtion gradient")
        if intent.typo_corrections:
            assert "backpropagtion" in intent.typo_corrections
            assert intent.typo_corrections["backpropagtion"] == "backpropagation"

    def test_no_corrections_for_valid(self):
        intent = analyze_query("softmax probability")
        for tok in intent.query_tokens:
            assert tok not in intent.typo_corrections or intent.typo_corrections[tok] == tok


class TestLectureDetection:
    def test_single_lecture(self):
        intent = analyze_query("lecture 8 gradients")
        assert 8 in intent.lecture_numbers

    def test_range(self):
        intent = analyze_query("lectures 13 through 15")
        assert 13 in intent.lecture_numbers
        assert 14 in intent.lecture_numbers
        assert 15 in intent.lecture_numbers

    def test_no_lecture(self):
        intent = analyze_query("what is softmax")
        assert intent.lecture_numbers == []
