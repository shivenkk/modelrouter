"""
Build labeled difficulty dataset for DistilBERT training.
Uses Groq (Llama 70B) as judge to label query difficulty 1-5.

DESIGNED FOR GROQ FREE TIER:
- Hard cap of 100 total API calls
- 2 second delay between calls
- Targets ~800-1000 queries total (enough for DistilBERT)

Usage:
    python build_dataset.py --api_key YOUR_KEY --output training_data.jsonl
"""

import json
import time
import random
import argparse
import httpx
from pathlib import Path
from collections import Counter


# ── seed queries (no API calls needed) ──────────────────────────

SEEDS = {
    "trivial": [
        "What color is the sky?",
        "How many days are in a week?",
        "What is 5 + 3?",
        "Who wrote Romeo and Juliet?",
        "What is the capital of France?",
        "Is water wet?",
        "What language is spoken in Brazil?",
        "How many legs does a dog have?",
        "What comes after Monday?",
        "What is the opposite of hot?",
        "What planet do we live on?",
        "How many months in a year?",
        "What color are bananas?",
        "What animal says moo?",
        "Is the sun a star?",
        "How many sides does a triangle have?",
        "What is the freezing point of water?",
        "Who is Mickey Mouse?",
        "What do bees make?",
        "What is 10 minus 3?",
        "What season comes after summer?",
        "How many continents are there?",
        "What is the largest ocean?",
        "Do fish live in water?",
        "What shape is a soccer ball?",
    ],
    "easy": [
        "Explain what a variable is in programming.",
        "What is the difference between a list and a tuple in Python?",
        "Summarize the water cycle in 3 sentences.",
        "Convert 72 degrees Fahrenheit to Celsius.",
        "What are the three branches of the US government?",
        "Write a Python function that adds two numbers.",
        "What is photosynthesis?",
        "Explain what HTTP status code 404 means.",
        "What is the difference between RAM and ROM?",
        "Name three renewable energy sources.",
        "What is a for loop and when would you use one?",
        "Explain the difference between HTML and CSS.",
        "What does CPU stand for and what does it do?",
        "How does a binary search work?",
        "What is the difference between == and === in JavaScript?",
        "Explain what an API is in simple terms.",
        "What is the Pythagorean theorem?",
        "What is the difference between HTTP and HTTPS?",
        "How do you declare a constant in Python?",
        "What is a primary key in a database?",
        "Explain what version control is.",
        "What is the difference between a compiler and an interpreter?",
        "What is an IP address?",
        "How does a stack data structure work?",
        "What is the time complexity of a linear search?",
    ],
    "medium": [
        "Explain how a hash table works and analyze its time complexity for insert, lookup, and delete.",
        "Write a Python function to find the longest common subsequence of two strings.",
        "Compare and contrast microservices and monolithic architectures. When would you use each?",
        "Explain the CAP theorem and give a real-world example for each trade-off.",
        "What is gradient descent? Walk through the math of one update step.",
        "Design a database schema for a social media app with users, posts, comments, and likes.",
        "Explain the difference between TCP and UDP with examples of when to use each.",
        "Write a recursive solution to the Tower of Hanoi problem and explain the time complexity.",
        "What is the difference between supervised and unsupervised learning? Give three examples of each.",
        "Explain how HTTPS works, including the TLS handshake.",
        "Implement a LRU cache in Python and explain your design choices.",
        "Explain how Docker containers differ from virtual machines at the OS level.",
        "What is database normalization? Walk through 1NF, 2NF, and 3NF with examples.",
        "Explain the observer design pattern and implement it in Python.",
        "How does garbage collection work in Java vs Python?",
        "Write a Python function to detect a cycle in a linked list and explain the algorithm.",
        "Explain how OAuth2 authorization code flow works step by step.",
        "What are the ACID properties in databases? Give an example where each could be violated.",
        "Compare REST and GraphQL APIs with trade-offs for different use cases.",
        "Explain how a neural network learns through backpropagation with a concrete example.",
        "Implement binary search tree insertion and deletion in Python with complexity analysis.",
        "Explain the differences between processes and threads with examples of when to use each.",
        "How does consistent hashing work and why is it useful in distributed systems?",
        "Write a dynamic programming solution to the knapsack problem with explanation.",
        "Explain how DNS resolution works from browser to IP address.",
    ],
    "hard": [
        "Design a distributed rate limiter that works across multiple data centers with eventual consistency. Include the data flow, failure modes, and trade-offs.",
        "Implement a B+ tree in Python with insert, delete, and range query operations. Analyze the complexity.",
        "Explain the Raft consensus algorithm in detail. How does leader election work? How are log entries replicated?",
        "Design a real-time fraud detection system for a payment processor handling 100K transactions per second. Include the ML pipeline, feature store, and alerting.",
        "Write a proof that the halting problem is undecidable. Then explain its practical implications for software verification.",
        "Design a CDN from scratch. Cover DNS routing, cache invalidation strategies, origin shielding, and how you'd handle a thundering herd problem.",
        "Implement a lock-free concurrent queue in C++ using atomic operations. Explain the memory ordering constraints.",
        "Design a search engine's ranking algorithm. Cover indexing, PageRank computation, query parsing, and how you'd handle freshness vs relevance trade-offs.",
        "Explain how transformers work from first principles. Derive the attention mechanism mathematically and explain why it outperforms RNNs for sequence tasks.",
        "Design a multi-region database replication system that supports both strong consistency reads and eventually consistent reads. Handle network partitions and conflict resolution.",
        "Design a real-time bidding system for programmatic advertising that handles 1M bid requests per second with 100ms latency SLA. Include the auction logic, budget pacing, and fraud detection.",
        "Implement a distributed key-value store with Paxos consensus. Handle leader election, log replication, and membership changes.",
        "Design an end-to-end ML platform that supports model training, versioning, A/B testing, and canary deployments. Include the CI/CD pipeline and monitoring.",
        "Explain the mathematical foundations of public key cryptography. Derive RSA from first principles and analyze its security guarantees.",
        "Design a real-time collaborative editing system like Google Docs. Cover conflict resolution using CRDTs or OT, cursor synchronization, and offline support.",
        "Implement a compiler front-end for a simple programming language. Include lexer, parser, and AST generation with error recovery.",
        "Design a distributed task scheduling system that handles dependencies, retries, and exactly-once semantics across a cluster of workers.",
        "Explain how RLHF works for training language models. Cover the reward model, PPO algorithm, and the mathematical objective being optimized.",
        "Design a time-series database optimized for IoT data with 10B data points per day. Cover storage engine, compression, downsampling, and query optimization.",
        "Implement a garbage collector using mark-and-sweep with generational collection. Analyze the trade-offs between throughput and pause times.",
    ],
}

# ── generation prompts ──────────────────────────────────────────

GENERATION_PROMPTS = {
    "trivial": "Generate 20 extremely simple factual questions that a child could answer. Single-sentence questions only, no overlap with common ones about colors, animals, or days. Return ONLY a JSON array of strings.",
    "easy": "Generate 20 simple technical or educational questions needing a short explanation. Cover programming, science, history, and math. Return ONLY a JSON array of strings.",
    "medium": "Generate 20 moderately complex technical questions requiring detailed multi-step explanation or comparison. Cover system design, algorithms, ML, databases, networking. Return ONLY a JSON array of strings.",
    "hard": "Generate 20 very complex questions requiring deep expertise: system design at scale, mathematical proofs, multi-component architectures, or advanced algorithms. Return ONLY a JSON array of strings.",
}

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# ── HARD CAPS FOR FREE TIER ─────────────────────────────────────
MAX_API_CALLS = 80          # absolute max across entire run
DELAY_BETWEEN_CALLS = 2.5   # seconds between each call
MAX_BATCHES_PER_LEVEL = 10  # max generation batches per difficulty
# ────────────────────────────────────────────────────────────────

_api_calls_made = 0


def call_groq(prompt: str, api_key: str, model: str = "llama-3.3-70b-versatile") -> str:
    """Call Groq with conservative rate limiting."""
    global _api_calls_made

    if _api_calls_made >= MAX_API_CALLS:
        print(f"  ⚠ Hit API call cap ({MAX_API_CALLS}), stopping generation")
        return ""

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
        "temperature": 0.8,
    }

    for attempt in range(3):
        try:
            time.sleep(DELAY_BETWEEN_CALLS)
            resp = httpx.post(GROQ_URL, headers=headers, json=payload, timeout=30.0)
            _api_calls_made += 1

            if resp.status_code == 429:
                wait = 5 * (attempt + 1) + random.random()
                print(f"  rate limited, waiting {wait:.0f}s... ({_api_calls_made}/{MAX_API_CALLS} calls used)")
                time.sleep(wait)
                continue

            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]

            print(f"  error {resp.status_code}: {resp.text[:100]}")
        except Exception as e:
            print(f"  exception: {e}")
            time.sleep(3)

    return ""


def generate_queries(api_key: str) -> list[dict]:
    """Generate labeled queries. Seeds + LLM augmentation."""
    all_queries = []
    difficulty_map = {"trivial": 1, "easy": 2, "medium": 3, "hard": 5}

    for level, seeds in SEEDS.items():
        print(f"\n── {level} ──")
        queries = list(seeds)
        target = 200  # 200 per level = ~800 total

        batches = 0
        while len(set(queries)) < target and batches < MAX_BATCHES_PER_LEVEL:
            if _api_calls_made >= MAX_API_CALLS:
                break

            batches += 1
            print(f"  batch {batches}/{MAX_BATCHES_PER_LEVEL} ({len(set(queries))} unique so far)...")

            raw = call_groq(GENERATION_PROMPTS[level], api_key)
            if not raw:
                break

            try:
                start = raw.find("[")
                end = raw.rfind("]") + 1
                if start != -1 and end > start:
                    parsed = json.loads(raw[start:end])
                    new = [q for q in parsed if isinstance(q, str) and len(q) > 5]
                    queries.extend(new)
            except json.JSONDecodeError:
                print(f"  parse failed, skipping")
                continue

        queries = list(set(queries))[:target]
        print(f"  final: {len(queries)} unique queries")

        for q in queries:
            all_queries.append({
                "query": q,
                "difficulty": difficulty_map[level],
                "level": level,
            })

    random.shuffle(all_queries)
    return all_queries


def save_dataset(queries: list[dict], output_path: str):
    """Save as JSONL."""
    path = Path(output_path)
    with open(path, "w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")

    dist = Counter(q["difficulty"] for q in queries)
    print(f"\nSaved {len(queries)} queries to {path}")
    print(f"Distribution: {dict(sorted(dist.items()))}")
    print(f"Total API calls used: {_api_calls_made}/{MAX_API_CALLS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=True, help="Groq API key")
    parser.add_argument("--output", default="training_data.jsonl")
    args = parser.parse_args()

    queries = generate_queries(args.api_key)
    save_dataset(queries, args.output)
