"""
S3-backed request logger. Drop-in replacement for logger.py.
Falls back to local file logging if S3 is not configured.

Writes JSONL to S3 with path: logs/{date}/{hour}.jsonl
Uses a local buffer to batch writes (flush every N requests or T seconds).
"""

import json
import uuid
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from collections import deque

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from app.config import S3_BUCKET, AWS_REGION


# ── S3 client setup ─────────────────────────────────────────────

_s3_client = None
_s3_available = False

try:
    _s3_client = boto3.client("s3", region_name=AWS_REGION)
    # test if bucket is accessible
    _s3_client.head_bucket(Bucket=S3_BUCKET)
    _s3_available = True
    print(f"[logger_s3] Connected to S3 bucket: {S3_BUCKET}")
except (ClientError, NoCredentialsError, Exception) as e:
    print(f"[logger_s3] S3 not available ({e}), falling back to local logging")
    _s3_available = False


# ── Buffer for batched writes ───────────────────────────────────

_buffer = deque()
_buffer_lock = threading.Lock()
FLUSH_THRESHOLD = 10      # flush after N entries
FLUSH_INTERVAL = 30.0     # or flush every N seconds
_last_flush = time.time()


def _s3_key() -> str:
    """Generate S3 key: logs/2026-03-22/14.jsonl"""
    now = datetime.now(timezone.utc)
    return f"logs/{now.strftime('%Y-%m-%d')}/{now.strftime('%H')}.jsonl"


def _flush_to_s3():
    """Flush buffer to S3 by appending to hourly JSONL file."""
    global _last_flush

    with _buffer_lock:
        if not _buffer:
            return
        entries = list(_buffer)
        _buffer.clear()
        _last_flush = time.time()

    if not _s3_available or not entries:
        return

    key = _s3_key()
    new_lines = "\n".join(json.dumps(e) for e in entries) + "\n"

    try:
        # try to append to existing object
        existing = ""
        try:
            resp = _s3_client.get_object(Bucket=S3_BUCKET, Key=key)
            existing = resp["Body"].read().decode("utf-8")
        except ClientError as e:
            if e.response["Error"]["Code"] != "NoSuchKey":
                raise

        _s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=(existing + new_lines).encode("utf-8"),
            ContentType="application/jsonl",
        )
    except Exception as e:
        print(f"[logger_s3] Failed to flush to S3: {e}")
        # put entries back in buffer
        with _buffer_lock:
            _buffer.extendleft(reversed(entries))


def _maybe_flush():
    """Flush if threshold or interval exceeded."""
    should_flush = (
        len(_buffer) >= FLUSH_THRESHOLD
        or (time.time() - _last_flush) > FLUSH_INTERVAL
    )
    if should_flush:
        threading.Thread(target=_flush_to_s3, daemon=True).start()


# ── Local fallback (same as logger.py) ──────────────────────────

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def _log_local(entry: dict):
    """Write to local JSONL file."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_file = LOG_DIR / f"{today}.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Public API (same interface as logger.py) ────────────────────

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
    """Log a request. Sends to S3 if available, local file otherwise."""

    cost_input = (tokens_in / 1000) * cost_per_1k_input
    cost_output = (tokens_out / 1000) * cost_per_1k_output
    estimated_cost = round(cost_input + cost_output, 6)

    entry = {
        "query_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query[:500],
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

    if _s3_available:
        with _buffer_lock:
            _buffer.append(entry)
        _maybe_flush()
    else:
        _log_local(entry)

    return entry


def get_logs(date: str | None = None, limit: int = 100) -> list[dict]:
    """Read logs. Tries S3 first, falls back to local."""
    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if _s3_available:
        try:
            entries = []
            # list all hour files for the date
            prefix = f"logs/{date}/"
            resp = _s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)

            for obj in resp.get("Contents", []):
                data = _s3_client.get_object(Bucket=S3_BUCKET, Key=obj["Key"])
                body = data["Body"].read().decode("utf-8")
                for line in body.strip().split("\n"):
                    if line.strip():
                        entries.append(json.loads(line))

            return entries[-limit:]
        except Exception as e:
            print(f"[logger_s3] Failed to read from S3: {e}")

    # local fallback
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
        "storage": "s3" if _s3_available else "local",
    }


def flush():
    """Force flush buffer to S3. Call on shutdown."""
    if _s3_available:
        _flush_to_s3()
