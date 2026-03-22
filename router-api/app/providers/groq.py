"""
Groq API client with retry logic and exponential backoff.
Handles rate limits on the free tier gracefully.
"""

import time
import httpx
import asyncio
from dataclasses import dataclass

from app.config import GROQ_API_KEY


GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
MAX_RETRIES = 3
BASE_DELAY = 1.0  # seconds


@dataclass
class ProviderResponse:
    text: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    error: str | None = None


async def call(
    model_id: str,
    query: str,
    system_prompt: str = "You are a helpful assistant.",
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> ProviderResponse:
    """Send a query to Groq with retry + exponential backoff."""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    last_error = None

    async with httpx.AsyncClient(timeout=30.0) as client:
        for attempt in range(MAX_RETRIES):
            start = time.perf_counter()

            try:
                resp = await client.post(
                    GROQ_BASE_URL,
                    headers=headers,
                    json=payload,
                )

                latency = (time.perf_counter() - start) * 1000

                if resp.status_code == 429:
                    # rate limited, back off
                    delay = BASE_DELAY * (2 ** attempt)
                    print(f"[groq] rate limited, retrying in {delay}s (attempt {attempt + 1})")
                    await asyncio.sleep(delay)
                    last_error = "rate_limited"
                    continue

                if resp.status_code != 200:
                    last_error = f"status_{resp.status_code}: {resp.text[:200]}"
                    continue

                data = resp.json()
                choice = data["choices"][0]
                usage = data.get("usage", {})

                return ProviderResponse(
                    text=choice["message"]["content"],
                    model=data.get("model", model_id),
                    tokens_in=usage.get("prompt_tokens", 0),
                    tokens_out=usage.get("completion_tokens", 0),
                    latency_ms=round(latency, 1),
                )

            except httpx.TimeoutException:
                latency = (time.perf_counter() - start) * 1000
                last_error = "timeout"
                delay = BASE_DELAY * (2 ** attempt)
                print(f"[groq] timeout, retrying in {delay}s (attempt {attempt + 1})")
                await asyncio.sleep(delay)

            except Exception as e:
                last_error = str(e)
                break

    return ProviderResponse(
        text="",
        model=model_id,
        tokens_in=0,
        tokens_out=0,
        latency_ms=0,
        error=f"groq_failed_after_{MAX_RETRIES}_retries: {last_error}",
    )
