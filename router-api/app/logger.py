"""
Request logger.
Phase 1: Logs to local JSON file.
Phase 2: Swap to S3.
"""

import json
import uuid
import os
from datetime import datetime, timezone
from pathlib import Path


LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def log_request(
    query: str,
    difficulty_score: float,
    difficulty_label: str,
    domain: str,
    routed_to: str,
    latency_ms: float,
    tokens_in: int,
    tokens_out: int,
    cost_per_1k_input: float,
    cost_per_1k_output: float,
    error: str | None = None,
    fallback_used: bool = False,
) -> dict:
    """Log a request to local JSON file. Returns the log entry."""

    cost_input = (tokens_in / 1000) * cost_per_1k_input
    cost_output = (tokens_out / 1000) * cost_per_1k_output
    estimated_cost = round(cost_input + cost_output, 6)

    entry = {
        "query_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query[:500],  # truncate for storage
        "difficulty_score": difficulty_score,
        "difficulty_label": difficulty_label,
        "domain": domain,
        "routed_to": routed_to,
        "latency_ms": latency_ms,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "estimated_cost_usd": estimated_cost,
        "error": error,
        "fallback_used": fallback_used,
    }

    # append to daily log file
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_file = LOG_DIR / f"{today}.jsonl"

    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return entry


def get_logs(date: str | None = None, limit: int = 100) -> list[dict]:
    """Read logs from local file. If no date, use today."""
    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    log_file = LOG_DIR / f"{date}.jsonl"
    if not log_file.exists():
        return []

    entries = []
    with open(log_file) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    return entries[-limit:]


def get_stats(date: str | None = None) -> dict:
    """Compute summary stats from logs."""
    logs = get_logs(date, limit=10000)

    if not logs:
        return {"total_requests": 0}

    total_cost = sum(e["estimated_cost_usd"] for e in logs)
    avg_latency = sum(e["latency_ms"] for e in logs) / len(logs)

    # cost if everything went to tier 3
    tier3_cost_input = 0.003
    tier3_cost_output = 0.015
    baseline_cost = sum(
        (e["tokens_in"] / 1000) * tier3_cost_input
        + (e["tokens_out"] / 1000) * tier3_cost_output
        for e in logs
    )

    model_counts = {}
    label_counts = {}
    for e in logs:
        model_counts[e["routed_to"]] = model_counts.get(e["routed_to"], 0) + 1
        label_counts[e["difficulty_label"]] = label_counts.get(e["difficulty_label"], 0) + 1

    errors = sum(1 for e in logs if e["error"])

    return {
        "total_requests": len(logs),
        "total_cost_usd": round(total_cost, 6),
        "baseline_cost_usd": round(baseline_cost, 6),
        "savings_usd": round(baseline_cost - total_cost, 6),
        "savings_pct": round((1 - total_cost / baseline_cost) * 100, 1) if baseline_cost > 0 else 0,
        "avg_latency_ms": round(avg_latency, 1),
        "model_distribution": model_counts,
        "difficulty_distribution": label_counts,
        "error_count": errors,
        "error_rate_pct": round((errors / len(logs)) * 100, 1),
    }
