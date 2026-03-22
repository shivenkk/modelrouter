"""
Final test: sends 20 queries across all 3 tiers against the live EC2 endpoint.
Easy/Medium use Groq (free), Hard uses Bedrock (~$0.01-0.02 each).
"""

import httpx
import time

API = "http://3.141.166.205:8000/route"

queries = [
    # EASY (8 queries) → should route to llama-8b
    "What is 2+2?",
    "What color is the sky?",
    "Who wrote Romeo and Juliet?",
    "How many days are in a week?",
    "What is the capital of Japan?",
    "What does CPU stand for?",
    "Is the earth round?",
    "What is the largest ocean?",

    # MEDIUM (7 queries) → should route to llama-70b
    "Explain how a hash table works and analyze its time complexity for each operation.",
    "Compare TCP vs UDP with real-world use cases for each protocol.",
    "Explain how OAuth2 authorization code flow works step by step.",
    "What are the ACID properties in databases? Give an example where each could be violated.",
    "Compare REST and GraphQL APIs with specific trade-offs for different use cases.",
    "Explain how Docker containers differ from virtual machines at the OS level.",
    "Implement a binary search tree with insert and delete in Python and analyze complexity.",

    # HARD (5 queries) → should route to claude-sonnet-bedrock (~$0.08 total)
    "Design a distributed rate limiter for a multi-region API gateway handling 1M requests per second with eventual consistency. Include data flow, failure modes, and trade-offs.",
    "Design a real-time fraud detection pipeline for processing 500K credit card transactions per second with sub-100ms alerting. Include the ML pipeline, feature store, and monitoring.",
    "Explain the Raft consensus algorithm in detail including leader election, log replication, and how it handles network partitions and split-brain scenarios.",
    "Design a CDN from scratch covering DNS-based routing, cache invalidation strategies, origin shielding, and thundering herd mitigation at global scale.",
    "Design an end-to-end ML platform supporting distributed training, model versioning, A/B testing, canary deployments, and real-time monitoring with rollback capabilities.",
]

print(f"Sending {len(queries)} queries to {API}\n")
total_cost = 0

for i, q in enumerate(queries):
    try:
        resp = httpx.post(API, json={"query": q}, timeout=60.0)
        if resp.status_code == 200:
            data = resp.json()
            r = data["routing"]
            m = data["metadata"]
            cost = m["estimated_cost_usd"]
            total_cost += cost
            model = m["model_used"]
            if "8b" in model:
                tier = "T1"
            elif "70b" in model:
                tier = "T2"
            else:
                tier = "T3"
            print(f"  [{i+1:2d}/20] {r['difficulty_label']:6s} → {tier} {model:25s} | {m['latency_ms']:7.0f}ms | ${cost:.4f} | {q[:55]}...")
        else:
            print(f"  [{i+1:2d}/20] ERROR {resp.status_code}")
    except Exception as e:
        print(f"  [{i+1:2d}/20] FAILED: {e}")

    time.sleep(2)  # respect rate limits

print(f"\nDone! Total Bedrock cost: ${total_cost:.4f}")
print(f"Check dashboard at http://localhost:5173")
