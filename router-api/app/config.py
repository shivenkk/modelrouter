import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# routing weights (tunable)
W_QUALITY = 0.6
W_COST = 0.2
W_LATENCY = 0.1
W_SUCCESS = 0.1

# difficulty thresholds for tier assignment
TIER_THRESHOLDS = {
    "easy": 2.0,      # score <= 2.0 -> Tier 1
    "medium": 3.5,    # score <= 3.5 -> Tier 2
    # anything above  -> Tier 3
}

# model registry
MODELS = {
    "llama-8b-groq": {
        "tier": 1,
        "provider": "groq",
        "model_id": "llama-3.1-8b-instant",
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "avg_latency_ms": 300,
        "quality_ceiling": 0.6,  # max quality score this model can hit (0-1)
    },
    "llama-70b-groq": {
        "tier": 2,
        "provider": "groq",
        "model_id": "llama-3.3-70b-versatile",
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "avg_latency_ms": 800,
        "quality_ceiling": 0.85,
    },
    "claude-sonnet-bedrock": {
        "tier": 3,
        "provider": "bedrock",
        "model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0",
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
        "avg_latency_ms": 1500,
        "quality_ceiling": 1.0,
    },
}

# latency budget default (ms)
DEFAULT_LATENCY_BUDGET = 5000

# s3 config (phase 2)
S3_BUCKET = os.getenv("S3_BUCKET", "modelrouter-logs-089781651268")
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
