"""
Unit tests for ModelRouter core components.
Run: python -m pytest tests/ -v
"""

import pytest
from app.classifier import classify
from app.router import select_model, _quality_estimate


# ── Classifier Tests ────────────────────────────────────────────

class TestClassifier:
    def test_trivial_query(self):
        r = classify("hi")
        assert r.label == "easy"
        assert r.score <= 2.0

    def test_simple_question(self):
        r = classify("What is Python?")
        assert r.label == "easy"
        assert r.score <= 2.0

    def test_medium_query(self):
        r = classify("Explain how gradient descent works step by step and compare SGD vs Adam")
        assert r.label == "medium"
        assert 2.0 < r.score <= 3.5

    def test_hard_query(self):
        r = classify(
            "Design a distributed systems architecture for a real-time recommendation "
            "engine that handles 10M requests per second with less than 50ms latency, "
            "including the data pipeline, feature store, model serving layer and monitoring."
        )
        assert r.label == "hard"
        assert r.score > 3.5

    def test_code_domain(self):
        r = classify("Write a Python function to sort an array")
        assert r.domain == "code"

    def test_math_domain(self):
        r = classify("Solve this integral of x squared")
        assert r.domain == "math"

    def test_turn_count_increases_score(self):
        r1 = classify("Explain transformers", turn_count=1)
        r2 = classify("Explain transformers", turn_count=5)
        assert r2.score > r1.score

    def test_code_block_bonus(self):
        q1 = "Fix this code"
        q2 = "Fix this code ```print('hello')```"
        r1 = classify(q1)
        r2 = classify(q2)
        assert r2.score > r1.score

    def test_score_clamped(self):
        # even an extremely complex query shouldn't exceed 5.0
        r = classify(
            "Design and implement step by step a distributed database with "
            "strong consistency under 10ms latency including the proof of "
            "correctness and optimized for less than $100/month ```code``` "
            "compare three approaches",
            turn_count=10,
        )
        assert r.score <= 5.0
        assert r.score >= 1.0


# ── Router Tests ────────────────────────────────────────────────

class TestRouter:
    def test_easy_routes_to_small_model(self):
        r = classify("What is 2+2?")
        name, config, score = select_model(r)
        assert name == "llama-8b-groq"

    def test_medium_routes_to_medium_model(self):
        r = classify("Explain how OAuth2 works step by step and compare with SAML")
        name, config, score = select_model(r)
        assert config["tier"] in [2, 3]

    def test_latency_budget_filters_slow_models(self):
        r = classify("Design a complex distributed system including the pipeline layer")
        # tight budget should exclude tier 3
        name, config, score = select_model(r, latency_budget=500)
        assert config["avg_latency_ms"] <= 500

    def test_quality_estimate_easy(self):
        # for easy queries, all models should score equally (1.0)
        q_low = _quality_estimate(0.6, 1.5)   # tier 1 model, easy query
        q_high = _quality_estimate(1.0, 1.5)   # tier 3 model, easy query
        assert q_low == q_high == 1.0

    def test_quality_estimate_hard(self):
        # for hard queries, bigger models should score higher
        q_small = _quality_estimate(0.6, 4.5)
        q_big = _quality_estimate(1.0, 4.5)
        assert q_big > q_small

    def test_fallback_on_impossible_budget(self):
        r = classify("hello")
        # budget of 1ms is impossible for all models
        name, config, score = select_model(r, latency_budget=1)
        # should still return something (fallback)
        assert name is not None


# ── Integration-style Tests ─────────────────────────────────────

class TestIntegration:
    def test_classify_returns_all_fields(self):
        r = classify("Test query for validation")
        assert hasattr(r, "score")
        assert hasattr(r, "label")
        assert hasattr(r, "domain")
        assert hasattr(r, "features")
        assert isinstance(r.features, dict)

    def test_routing_explanation(self):
        from app.router import get_routing_explanation
        r = classify("hello")
        explanation = get_routing_explanation(r, "llama-8b-groq", 0.95)
        assert "difficulty_score" in explanation
        assert "routed_to" in explanation
        assert explanation["routed_to"] == "llama-8b-groq"
