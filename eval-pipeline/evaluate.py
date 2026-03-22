"""
Nightly evaluation pipeline for ModelRouter.
Compares routed responses vs all-Tier-3 baseline on a gold test set.

Designed to run as:
- AWS Lambda (triggered by EventBridge daily)
- Local script for testing

Usage:
    python evaluate.py --api_url http://localhost:8000 --gold_queries gold_queries.json
"""

import json
import time
import argparse
import httpx
from datetime import datetime, timezone
from pathlib import Path


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
JUDGE_MODEL = "llama-3.1-8b-instant"  # free, fast judge


def load_gold_queries(path: str) -> list[dict]:
    """Load gold test set."""
    with open(path) as f:
        return json.load(f)


def call_router(api_url: str, query: str) -> dict:
    """Send query through ModelRouter."""
    try:
        resp = httpx.post(
            f"{api_url}/route",
            json={"query": query},
            timeout=30.0,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        return {"error": str(e)}
    return {"error": f"status {resp.status_code}"}


def call_baseline(query: str, groq_key: str) -> dict:
    """Call Tier 3 equivalent (70B) directly as baseline."""
    headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ],
        "max_tokens": 1024,
        "temperature": 0.7,
    }

    try:
        start = time.perf_counter()
        resp = httpx.post(GROQ_URL, headers=headers, json=payload, timeout=30.0)
        latency = (time.perf_counter() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()
            usage = data.get("usage", {})
            return {
                "text": data["choices"][0]["message"]["content"],
                "tokens_in": usage.get("prompt_tokens", 0),
                "tokens_out": usage.get("completion_tokens", 0),
                "latency_ms": round(latency, 1),
            }
        if resp.status_code == 429:
            time.sleep(5)
            return {"error": "rate_limited"}
    except Exception as e:
        return {"error": str(e)}
    return {"error": "unknown"}


def judge_quality(query: str, response_a: str, response_b: str, groq_key: str) -> dict:
    """Use LLM-as-judge to compare two responses. Returns scores 1-5 for each."""
    headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
    prompt = f"""Compare these two responses to the query. Score each 1-5 for quality (accuracy, completeness, helpfulness).

Query: {query[:500]}

Response A: {response_a[:800]}

Response B: {response_b[:800]}

Return ONLY a JSON object: {{"score_a": <int>, "score_b": <int>, "reason": "<one sentence>"}}"""

    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.1,
    }

    try:
        time.sleep(1)  # rate limit courtesy
        resp = httpx.post(GROQ_URL, headers=headers, json=payload, timeout=15.0)
        if resp.status_code == 200:
            raw = resp.json()["choices"][0]["message"]["content"]
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(raw[start:end])
    except Exception:
        pass

    return {"score_a": 3, "score_b": 3, "reason": "judge_failed"}


def run_eval(api_url: str, gold_path: str, groq_key: str, limit: int = 50) -> dict:
    """Run full evaluation. Returns summary metrics."""
    gold = load_gold_queries(gold_path)[:limit]
    print(f"Evaluating {len(gold)} queries...")

    results = []
    total_router_cost = 0
    total_baseline_cost = 0

    # baseline cost rates (Tier 3 / Claude Sonnet)
    t3_cost_in = 0.003   # per 1K tokens
    t3_cost_out = 0.015

    for i, item in enumerate(gold):
        query = item["query"]
        print(f"  [{i+1}/{len(gold)}] {query[:60]}...")

        # routed response
        router_result = call_router(api_url, query)
        if "error" in router_result:
            print(f"    router error: {router_result['error']}")
            continue

        routed_text = router_result.get("response", "")
        routed_meta = router_result.get("metadata", {})
        routed_cost = routed_meta.get("estimated_cost_usd", 0)

        # baseline response (all queries to 70B)
        baseline_result = call_baseline(query, groq_key)
        if "error" in baseline_result:
            print(f"    baseline error: {baseline_result['error']}")
            time.sleep(3)
            continue

        baseline_text = baseline_result["text"]
        baseline_cost = (
            (baseline_result["tokens_in"] / 1000) * t3_cost_in
            + (baseline_result["tokens_out"] / 1000) * t3_cost_out
        )

        # judge
        judgment = judge_quality(query, routed_text, baseline_text, groq_key)

        total_router_cost += routed_cost
        total_baseline_cost += baseline_cost

        results.append({
            "query": query[:200],
            "difficulty": router_result.get("routing", {}).get("difficulty_label", "?"),
            "routed_to": routed_meta.get("model_used", "?"),
            "router_score": judgment.get("score_a", 0),
            "baseline_score": judgment.get("score_b", 0),
            "router_latency": routed_meta.get("latency_ms", 0),
            "baseline_latency": baseline_result.get("latency_ms", 0),
            "router_cost": routed_cost,
            "baseline_cost": round(baseline_cost, 6),
            "reason": judgment.get("reason", ""),
        })

        time.sleep(0.5)

    # compute summary
    if not results:
        return {"error": "no_results"}

    avg_router = sum(r["router_score"] for r in results) / len(results)
    avg_baseline = sum(r["baseline_score"] for r in results) / len(results)
    quality_retention = (avg_router / avg_baseline * 100) if avg_baseline > 0 else 0
    cost_savings = (
        (1 - total_router_cost / total_baseline_cost) * 100
        if total_baseline_cost > 0 else 100
    )

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "queries_evaluated": len(results),
        "avg_router_quality": round(avg_router, 3),
        "avg_baseline_quality": round(avg_baseline, 3),
        "quality_retention_pct": round(quality_retention, 1),
        "total_router_cost": round(total_router_cost, 6),
        "total_baseline_cost": round(total_baseline_cost, 6),
        "cost_savings_pct": round(cost_savings, 1),
        "results": results,
    }

    # save report
    report_path = Path(f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nReport saved to {report_path}")

    print(f"\n── Evaluation Summary ──")
    print(f"Queries evaluated:   {len(results)}")
    print(f"Router quality:      {avg_router:.2f}/5")
    print(f"Baseline quality:    {avg_baseline:.2f}/5")
    print(f"Quality retention:   {quality_retention:.1f}%")
    print(f"Cost savings:        {cost_savings:.1f}%")

    return summary


# ── Lambda handler (for AWS deployment) ─────────────────────────

def lambda_handler(event, context):
    """AWS Lambda entry point."""
    import os
    api_url = os.environ.get("API_URL", "http://localhost:8000")
    groq_key = os.environ.get("GROQ_API_KEY", "")
    gold_path = os.environ.get("GOLD_QUERIES_PATH", "gold_queries.json")

    summary = run_eval(api_url, gold_path, groq_key, limit=50)
    return {"statusCode": 200, "body": json.dumps(summary)}


if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_url", default="http://localhost:8000")
    parser.add_argument("--gold_queries", default="gold_queries.json")
    parser.add_argument("--groq_key", default=os.environ.get("GROQ_API_KEY", ""))
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    run_eval(args.api_url, args.gold_queries, args.groq_key, args.limit)
