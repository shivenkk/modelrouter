"""
Multi-factor routing engine.
Scores each candidate model and picks the best one
within the latency budget.
"""

from app.config import (
    MODELS,
    W_QUALITY,
    W_COST,
    W_LATENCY,
    W_SUCCESS,
    DEFAULT_LATENCY_BUDGET,
)
from app.classifier import ClassificationResult


def _normalize(value: float, max_val: float) -> float:
    """Normalize a value to 0-1 range."""
    if max_val == 0:
        return 0.0
    return min(1.0, value / max_val)


def _quality_estimate(quality_ceiling: float, difficulty_score: float) -> float:
    """
    Estimate how well a model handles a given difficulty.
    Key insight: for easy queries, ALL models perform equally well.
    For hard queries, strongly prefer high-ceiling models.
    """
    norm_diff = (difficulty_score - 1.0) / 4.0  # 0.0 - 1.0

    # easy queries: all models are equally good -> return 1.0 for everyone
    # this lets cost/latency break the tie in favor of smaller models
    if norm_diff <= 0.3:
        return 1.0

    # medium: advantage for bigger models
    if norm_diff <= 0.6:
        return quality_ceiling

    # hard: strongly penalize models below ceiling
    # a model with ceiling 0.6 on a hard query (norm_diff 1.0) gets heavily penalized
    overshoot = norm_diff - quality_ceiling
    if overshoot > 0:
        return max(0.05, quality_ceiling - overshoot * 2.5)
    return quality_ceiling


def select_model(
    classification: ClassificationResult,
    latency_budget: float = DEFAULT_LATENCY_BUDGET,
    cost_cap: float | None = None,
) -> tuple[str, dict, float]:
    """
    Score all models and return (model_name, model_config, score).

    Returns the highest-scoring model within latency budget.
    Falls back to Tier 2 if Tier 3 is stubbed/unavailable.
    """
    max_cost = max(m["cost_per_1k_input"] for m in MODELS.values()) or 1.0
    max_latency = max(m["avg_latency_ms"] for m in MODELS.values()) or 1.0

    candidates = []

    for name, model in MODELS.items():
        # skip if over latency budget
        if model["avg_latency_ms"] > latency_budget:
            continue

        # skip if over cost cap (if set)
        if cost_cap is not None and model["cost_per_1k_input"] > cost_cap:
            continue

        quality = _quality_estimate(model["quality_ceiling"], classification.score)
        cost_score = 1.0 - _normalize(model["cost_per_1k_input"], max_cost)
        latency_score = 1.0 - _normalize(model["avg_latency_ms"], max_latency)
        success_score = 1.0  # placeholder, will use real metrics in Phase 2

        total = (
            W_QUALITY * quality
            + W_COST * cost_score
            + W_LATENCY * latency_score
            + W_SUCCESS * success_score
        )

        candidates.append((name, model, round(total, 4)))

    if not candidates:
        # fallback: pick the cheapest model regardless of budget
        fallback = min(MODELS.items(), key=lambda x: x[1]["cost_per_1k_input"])
        return fallback[0], fallback[1], 0.0

    # sort by score descending
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[0]


def get_routing_explanation(
    classification: ClassificationResult,
    model_name: str,
    score: float,
) -> dict:
    """Build a human-readable routing explanation for the API response."""
    return {
        "difficulty_score": classification.score,
        "difficulty_label": classification.label,
        "domain": classification.domain,
        "routed_to": model_name,
        "routing_score": score,
        "features": classification.features,
    }
