"""
Kairon FastAPI Server

REST API for the causally-aware semantic cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from kairon.models import ComparisonOperator, Precondition
from kairon.router import CausalRouter


# ---------------------------------------------------------------------------
# Global router state
# ---------------------------------------------------------------------------

_router: Optional[CausalRouter] = None
_data_sources: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _router, _data_sources
    _router = CausalRouter(embedding_dim=768)
    _data_sources = {}
    yield
    _router = None


app = FastAPI(
    title="Kairon",
    description="Causally-Aware Semantic Cache — predictive invalidation via causal fingerprints",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class PutRequest(BaseModel):
    query: str
    response: str
    preconditions: list[dict] = Field(default_factory=list)
    causal_factors: list[str] = Field(default_factory=list)
    confidence: float = 1.0


class PutResponse(BaseModel):
    entry_id: str
    causal_hash: str
    status: str = "cached"


class GetRequest(BaseModel):
    query: str
    validate_preconditions: bool = True


class GetResponse(BaseModel):
    hit: bool
    tier: str
    response: Optional[str] = None
    confidence: float = 0.0
    causal_explanation: str = ""


class InvalidateRequest(BaseModel):
    key: str


class InvalidateResponse(BaseModel):
    invalidated: list[str]


class SourceRequest(BaseModel):
    key: str
    value: Any


class StatsResponse(BaseModel):
    cache: dict
    graph: dict
    thresholds: dict


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/cache", response_model=PutResponse)
async def put_cache(req: PutRequest):
    """Insert a query→response pair with causal preconditions."""
    if _router is None:
        raise HTTPException(500, "Router not initialized")

    preconditions = [
        Precondition(
            key=pc["key"],
            operator=ComparisonOperator(pc.get("operator", "eq")),
            expected_value=pc["expected_value"],
        )
        for pc in req.preconditions
    ]

    entry = _router.insert_with_preconditions(
        query=req.query,
        response=req.response,
        preconditions=preconditions,
        causal_factors=req.causal_factors or None,
        confidence=req.confidence,
    )
    return PutResponse(
        entry_id=entry.id,
        causal_hash=entry.causal_fingerprint.causal_hash,
    )


@app.get("/cache/{query}", response_model=GetResponse)
async def get_cache(query: str, validate_preconditions: bool = True):
    """Query the causal cache for a cached response."""
    if _router is None:
        raise HTTPException(500, "Router not initialized")

    result = _router.route(query, validate_preconditions=validate_preconditions)

    return GetResponse(
        hit=result.hit,
        tier=result.tier.value,
        response=result.entry.response if result.entry else None,
        confidence=result.confidence,
        causal_explanation=result.causal_explanation,
    )


@app.post("/invalidate", response_model=InvalidateResponse)
async def invalidate_by_key(req: InvalidateRequest):
    """Invalidate all entries that depend on a specific causal factor key."""
    if _router is None:
        raise HTTPException(500, "Router not initialized")
    invalidated = _router.invalidate_by_key(req.key)
    return InvalidateResponse(invalidated=invalidated)


@app.post("/source", response_model=dict)
async def register_source(req: SourceRequest):
    """Register or update a real-time data source for precondition validation."""
    global _data_sources
    if _router is None:
        raise HTTPException(500, "Router not initialized")

    _data_sources[req.key] = req.value
    _router.register_source(req.key, lambda k=req.key: _data_sources.get(k))
    return {"status": "registered", "key": req.key}


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get cache and causal graph statistics."""
    if _router is None:
        raise HTTPException(500, "Router not initialized")
    return StatsResponse(**_router.stats())


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}