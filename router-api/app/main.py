"""
ModelRouter FastAPI Gateway.
Classifies query difficulty, routes to optimal model, logs everything.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import time

from app.classifier_ml import classify
from app.router import select_model, get_routing_explanation
from app.providers import groq, bedrock
from app.logger_s3 import log_request, get_logs, get_stats
from app.config import MODELS


app = FastAPI(
    title="ModelRouter",
    description="Intelligent LLM routing based on query difficulty",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock this down in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response Models ---

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)
    turn_count: int = Field(default=1, ge=1)
    latency_budget_ms: float = Field(default=5000, gt=0)
    cost_cap: float | None = Field(default=None)
    force_model: str | None = Field(default=None)  # override routing


class QueryResponse(BaseModel):
    response: str
    routing: dict
    metadata: dict


# --- Provider dispatch ---

PROVIDER_MAP = {
    "groq": groq.call,
    "bedrock": bedrock.call,
}


async def dispatch(model_name: str, model_config: dict, query: str):
    """Send query to the right provider."""
    provider = model_config["provider"]
    call_fn = PROVIDER_MAP.get(provider)

    if call_fn is None:
        raise HTTPException(status_code=500, detail=f"Unknown provider: {provider}")

    return await call_fn(
        model_id=model_config["model_id"],
        query=query,
    )


# --- Endpoints ---

@app.post("/route", response_model=QueryResponse)
async def route_query(req: QueryRequest):
    """Main endpoint. Classify, route, call model, log, return."""

    # 1. classify difficulty
    classification = classify(req.query, req.turn_count)

    # 2. select model (or use forced override)
    if req.force_model and req.force_model in MODELS:
        model_name = req.force_model
        model_config = MODELS[model_name]
        routing_score = -1.0  # indicates forced
    else:
        model_name, model_config, routing_score = select_model(
            classification,
            latency_budget=req.latency_budget_ms,
            cost_cap=req.cost_cap,
        )

    # 3. call the model
    start = time.perf_counter()
    result = await dispatch(model_name, model_config, req.query)
    total_latency = (time.perf_counter() - start) * 1000

    # 4. check for errors, attempt fallback
    fallback_used = False
    if result.error and model_name != "llama-70b-groq":
        # fallback to tier 2
        fallback_name = "llama-70b-groq"
        fallback_config = MODELS[fallback_name]
        result = await dispatch(fallback_name, fallback_config, req.query)
        total_latency = (time.perf_counter() - start) * 1000
        model_name = fallback_name
        model_config = fallback_config
        fallback_used = True

    # 5. log
    log_entry = log_request(
        query=req.query,
        difficulty_score=classification.score,
        difficulty_label=classification.label,
        domain=classification.domain,
        routed_to=model_name,
        latency_ms=round(total_latency, 1),
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        cost_per_1k_input=model_config["cost_per_1k_input"],
        cost_per_1k_output=model_config["cost_per_1k_output"],
        error=result.error,
        fallback_used=fallback_used,
    )

    # 6. build response
    routing_info = get_routing_explanation(classification, model_name, routing_score)
    routing_info["fallback_used"] = fallback_used

    metadata = {
        "query_id": log_entry["query_id"],
        "model_used": model_name,
        "provider": model_config["provider"],
        "tokens_in": result.tokens_in,
        "tokens_out": result.tokens_out,
        "latency_ms": round(total_latency, 1),
        "estimated_cost_usd": log_entry["estimated_cost_usd"],
    }

    return QueryResponse(
        response=result.text,
        routing=routing_info,
        metadata=metadata,
    )


@app.get("/classify")
async def classify_only(query: str, turn_count: int = 1):
    """Classify a query without routing it. Useful for debugging."""
    result = classify(query, turn_count)
    return {
        "score": result.score,
        "label": result.label,
        "domain": result.domain,
        "features": result.features,
    }


@app.get("/stats")
async def stats(date: str | None = None):
    """Get summary stats for a given day."""
    return get_stats(date)


@app.get("/logs")
async def logs(date: str | None = None, limit: int = 50):
    """Get raw log entries."""
    return get_logs(date, limit)


@app.get("/models")
async def list_models():
    """List available models and their configs."""
    return MODELS


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}
