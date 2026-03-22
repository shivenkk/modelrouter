"""
ML-based difficulty classifier using fine-tuned DistilBERT.
Drop-in replacement for classifier.py — same ClassificationResult interface.

4 classes: trivial(0), easy(1), medium(2), hard(3)
Falls back to rule-based classifier if model not found.
"""

import os
import time
from pathlib import Path
from dataclasses import dataclass

MODEL_DIR = os.getenv("CLASSIFIER_MODEL_DIR", "./model_output")

_model = None
_tokenizer = None
_device = None

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_path = Path(MODEL_DIR)
    if model_path.exists() and (model_path / "config.json").exists():
        _device = torch.device("cpu")
        _tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        _model = AutoModelForSequenceClassification.from_pretrained(str(model_path)).to(_device)
        _model.eval()
        print(f"[classifier_ml] Loaded DistilBERT from {model_path}")
    else:
        print(f"[classifier_ml] Model not found at {model_path}, using rule-based fallback")
except Exception as e:
    print(f"[classifier_ml] Failed to load model ({e}), using rule-based fallback")


@dataclass
class ClassificationResult:
    score: float
    label: str
    domain: str
    features: dict


INDEX_TO_LABEL = {0: "easy", 1: "easy", 2: "medium", 3: "hard"}
INDEX_TO_SCORE = {0: 1.0, 1: 2.0, 2: 3.0, 3: 5.0}


def classify(query: str, turn_count: int = 1) -> ClassificationResult:
    """Classify using DistilBERT if available, else rule-based fallback."""

    if _model is None or _tokenizer is None:
        from app.classifier import classify as rule_based
        return rule_based(query, turn_count)

    start = time.perf_counter()

    encoding = _tokenizer(
        query,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        input_ids = encoding["input_ids"].to(_device)
        attention_mask = encoding["attention_mask"].to(_device)
        outputs = _model(input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze()
        pred_class = probs.argmax().item()
        confidence = probs[pred_class].item()

    inference_ms = (time.perf_counter() - start) * 1000

    score = INDEX_TO_SCORE[pred_class]
    turn_bonus = min(0.7, (turn_count - 1) * 0.15)
    score = min(5.0, score + turn_bonus)

    if score <= 2.0:
        label = "easy"
    elif score <= 3.5:
        label = "medium"
    else:
        label = "hard"

    from app.classifier import _detect_domain
    domain, _ = _detect_domain(query.lower())

    return ClassificationResult(
        score=round(score, 2),
        label=label,
        domain=domain,
        features={
            "ml_class": pred_class,
            "ml_confidence": round(confidence, 4),
            "ml_probs": [round(p, 4) for p in probs.tolist()],
            "inference_ms": round(inference_ms, 2),
            "turn_bonus": turn_bonus,
            "model": "distilbert",
        },
    )
