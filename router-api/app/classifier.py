"""
Difficulty classifier for incoming queries.

Phase 1: Rule-based heuristic classifier.
Phase 2: Swap in fine-tuned DistilBERT (drop-in replacement).
"""

import re
from dataclasses import dataclass


@dataclass
class ClassificationResult:
    score: float          # 1.0 - 5.0 difficulty
    label: str            # easy / medium / hard
    domain: str           # general / code / math / reasoning / creative
    features: dict        # raw feature values used


# keyword lists for domain detection
CODE_TOKENS = {
    "function", "class", "def", "import", "return", "variable",
    "loop", "array", "dict", "list", "api", "endpoint", "debug",
    "error", "bug", "code", "program", "script", "compile",
    "runtime", "syntax", "algorithm", "data structure", "sql",
    "query", "database", "docker", "kubernetes", "deploy",
}

MATH_TOKENS = {
    "calculate", "equation", "integral", "derivative", "matrix",
    "probability", "statistics", "proof", "theorem", "solve",
    "graph", "function", "polynomial", "vector", "eigenvalue",
    "regression", "optimization", "gradient", "converge",
}

REASONING_TOKENS = {
    "explain", "why", "how does", "compare", "contrast",
    "analyze", "evaluate", "design", "architect", "trade-off",
    "pros and cons", "strategy", "plan", "recommend", "should i",
    "what if", "implications", "consequences",
}

CREATIVE_TOKENS = {
    "write", "story", "poem", "essay", "blog", "creative",
    "fiction", "narrative", "character", "dialogue", "draft",
    "rewrite", "tone", "style", "persuasive", "compelling",
}


def _detect_domain(query_lower: str) -> tuple[str, float]:
    """Detect query domain and return (domain, domain_difficulty_bonus)."""
    scores = {
        "code": sum(1 for t in CODE_TOKENS if t in query_lower),
        "math": sum(1 for t in MATH_TOKENS if t in query_lower),
        "reasoning": sum(1 for t in REASONING_TOKENS if t in query_lower),
        "creative": sum(1 for t in CREATIVE_TOKENS if t in query_lower),
    }

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "general", 0.0

    bonuses = {"code": 0.8, "math": 1.0, "reasoning": 0.6, "creative": 0.4}
    return best, bonuses[best]


def _has_multi_step(query_lower: str) -> bool:
    """Check if query asks for multi-step work."""
    multi_step_signals = [
        "step by step", "and then", "first.*then",
        "multiple", "compare.*and", "list.*and explain",
        "build", "implement", "create.*with",
        "including.*and", "design.*architecture",
        "pipeline", "end.to.end", "full.stack",
        "system.*that", "layer", "component",
    ]
    return any(re.search(p, query_lower) for p in multi_step_signals)


def _has_constraints(query_lower: str) -> bool:
    """Check for specific constraints that increase difficulty."""
    constraint_signals = [
        "under \\d+", "less than", "within", "budget",
        "optimize", "efficient", "fastest", "minimum",
        "at most", "no more than", "constraint",
    ]
    return any(re.search(p, query_lower) for p in constraint_signals)


def _has_specifics(query_lower: str) -> bool:
    """Check for specific numbers/metrics that signal complex requirements."""
    specifics_signals = [
        r"\d+[kmb]\b", r"\d+\s*(ms|seconds|minutes)",
        r"\d+\s*requests", r"\d+%", r"\$\d+",
        r"per\s+second", r"per\s+hour", r"per\s+day",
        r"\d+\s*million", r"\d+\s*billion",
    ]
    return any(re.search(p, query_lower) for p in specifics_signals)


def classify(query: str, turn_count: int = 1) -> ClassificationResult:
    """
    Classify query difficulty on a 1.0-5.0 scale.

    Scoring breakdown:
    - Base from length:       0.5 - 1.5
    - Domain bonus:           0.0 - 1.0
    - Multi-step bonus:       0.0 - 0.8
    - Constraint bonus:       0.0 - 0.5
    - Conversation depth:     0.0 - 0.7
    - Code block presence:    0.0 - 0.5
    """
    q = query.lower().strip()
    word_count = len(q.split())

    # base score from length
    if word_count <= 5:
        base = 0.5
    elif word_count <= 15:
        base = 0.8
    elif word_count <= 40:
        base = 1.2
    else:
        base = 1.8

    # domain detection
    domain, domain_bonus = _detect_domain(q)

    # multi-step
    multi_step_bonus = 0.8 if _has_multi_step(q) else 0.0

    # constraints
    constraint_bonus = 0.5 if _has_constraints(q) else 0.0

    # conversation depth (multi-turn = harder context)
    turn_bonus = min(0.7, (turn_count - 1) * 0.15)

    # code block presence
    code_block_bonus = 0.5 if "```" in query else 0.0

    # specific numbers/metrics
    specifics_bonus = 0.6 if _has_specifics(q) else 0.0

    raw_score = (
        base
        + domain_bonus
        + multi_step_bonus
        + constraint_bonus
        + turn_bonus
        + code_block_bonus
        + specifics_bonus
    )

    # clamp to 1.0 - 5.0
    score = max(1.0, min(5.0, raw_score))

    # label
    if score <= 2.0:
        label = "easy"
    elif score <= 3.5:
        label = "medium"
    else:
        label = "hard"

    features = {
        "word_count": word_count,
        "base": base,
        "domain_bonus": domain_bonus,
        "multi_step_bonus": multi_step_bonus,
        "constraint_bonus": constraint_bonus,
        "turn_bonus": turn_bonus,
        "code_block_bonus": code_block_bonus,
        "specifics_bonus": specifics_bonus,
    }

    return ClassificationResult(
        score=round(score, 2),
        label=label,
        domain=domain,
        features=features,
    )
