"""
Kairon: Causally-Aware Semantic Cache

Core data models for causal fingerprints, preconditions, and cached entries.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field, model_validator


class ComparisonOperator(str, Enum):
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    LT = "lt"
    GTE = "gte"
    LTE = "lte"
    BETWEEN = "between"
    IN = "in"


class CacheTier(str, Enum):
    L1 = "L1"  # Exact hash hit (memory dict)
    L2 = "L2"  # Semantic (FAISS vector) hit
    L3 = "L3"  # Causal graph-only hit (no vector match, causal match only)
    MISS = "MISS"


class Precondition(BaseModel):
    """A precondition is a condition that must hold for a cached response to stay valid."""

    key: str
    operator: ComparisonOperator
    expected_value: Any
    last_verified: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "manual"  # "manual", "influx", "api", "inferred"

    def check(self, current_value: Any) -> bool:
        """Validate this precondition against a current value."""
        try:
            if self.operator == ComparisonOperator.EQ:
                return current_value == self.expected_value
            elif self.operator == ComparisonOperator.NE:
                return current_value != self.expected_value
            elif self.operator == ComparisonOperator.GT:
                return float(current_value) > float(self.expected_value)
            elif self.operator == ComparisonOperator.LT:
                return float(current_value) < float(self.expected_value)
            elif self.operator == ComparisonOperator.GTE:
                return float(current_value) >= float(self.expected_value)
            elif self.operator == ComparisonOperator.LTE:
                return float(current_value) <= float(self.expected_value)
            elif self.operator == ComparisonOperator.BETWEEN:
                low, high = self.expected_value
                return float(low) <= float(current_value) <= float(high)
            elif self.operator == ComparisonOperator.IN:
                return current_value in self.expected_value
        except (TypeError, ValueError):
            return False


class CausalFingerprint(BaseModel):
    """Uniquely identifies the causal context of a cached result."""

    causal_hash: str
    graph_node_id: Optional[str] = None
    preconditions: list[Precondition] = Field(default_factory=list)
    dependent_queries: list[str] = Field(default_factory=list)
    causal_factors: list[str] = Field(default_factory=list)

    @classmethod
    def from_preconditions(
        cls, preconditions: list[Precondition], causal_factors: list[str] | None = None
    ) -> "CausalFingerprint":
        """Generate a fingerprint hash from preconditions and factors."""
        content = "|".join(
            f"{p.key}:{p.operator.value}:{p.expected_value}"
            for p in sorted(preconditions, key=lambda x: x.key)
        )
        if causal_factors:
            content += "|factors:" + ",".join(sorted(causal_factors))
        h = hashlib.sha256(content.encode()).hexdigest()[:16]
        return cls(
            causal_hash=h,
            preconditions=preconditions,
            causal_factors=causal_factors or [],
        )


class CachedEntry(BaseModel):
    """A cached query→response pair with causal fingerprint."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    query_text: str
    query_embedding: Optional[list[float]] = None  # Set by vector cache
    response: str
    causal_fingerprint: Optional[CausalFingerprint] = None
    confidence: float = 1.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_access: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    validity_window_seconds: Optional[float] = None

    @property
    def preconditions(self) -> list[Precondition]:
        if self.causal_fingerprint is None:
            return []
        return self.causal_fingerprint.preconditions

    @property
    def is_expired(self) -> bool:
        if self.validity_window_seconds is None:
            return False
        age = (datetime.now(timezone.utc) - self.created_at).total_seconds()
        return age > self.validity_window_seconds

    def record_access(self):
        self.last_access = datetime.now(timezone.utc)
        self.access_count += 1

    @model_validator(mode='after')
    def ensure_causal_fingerprint(self):
        """Auto-generate an empty causal fingerprint if none provided."""
        if self.causal_fingerprint is None:
            self.causal_fingerprint = CausalFingerprint(causal_hash="none")
        return self


class RouteResult(BaseModel):
    """Result of a cache routing decision."""

    hit: bool
    tier: CacheTier
    entry: Optional[CachedEntry] = None
    confidence: float = 0.0
    causal_explanation: str = ""

    @classmethod
    def hit_result(cls, entry: CachedEntry, tier: CacheTier, confidence: float) -> "RouteResult":
        return cls(hit=True, tier=tier, entry=entry, confidence=confidence)

    @classmethod
    def miss_result(cls, reason: str = "") -> "RouteResult":
        return cls(hit=False, tier=CacheTier.MISS, causal_explanation=reason)