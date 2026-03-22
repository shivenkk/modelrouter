"""
AWS Bedrock client for Tier 3 (Claude Sonnet).
"""

import time
import json
import boto3
from dataclasses import dataclass

from app.config import AWS_REGION


@dataclass
class ProviderResponse:
    text: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    error: str | None = None


_client = None

try:
    _client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    print(f"[bedrock] Client initialized (region={AWS_REGION})")
except Exception as e:
    print(f"[bedrock] Failed to init client: {e}")


async def call(
    model_id: str,
    query: str,
    system_prompt: str = "You are a helpful assistant.",
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> ProviderResponse:
    """Call Bedrock Claude."""

    if _client is None:
        return ProviderResponse(
            text="",
            model=model_id,
            tokens_in=0,
            tokens_out=0,
            latency_ms=0,
            error="bedrock_client_not_initialized",
        )

    start = time.perf_counter()
    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": query}],
            "temperature": temperature,
        })

        resp = _client.invoke_model(
            modelId=model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )

        result = json.loads(resp["body"].read())
        latency = (time.perf_counter() - start) * 1000

        return ProviderResponse(
            text=result["content"][0]["text"],
            model=model_id,
            tokens_in=result["usage"]["input_tokens"],
            tokens_out=result["usage"]["output_tokens"],
            latency_ms=round(latency, 1),
        )

    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return ProviderResponse(
            text="",
            model=model_id,
            tokens_in=0,
            tokens_out=0,
            latency_ms=round(latency, 1),
            error=f"bedrock_error: {str(e)}",
        )
