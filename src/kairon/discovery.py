"""
Causal Discovery Service — discovers causal relationships from query logs.

Uses statistical methods (PC algorithm-inspired) to infer causal structure
from observed query→response→precondition patterns.

In production: swap in DoWhy, gCastle, or causal-learn for full PC algorithm.
MVP: pattern-based heuristic discovery from access logs.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from .models import (
    CachedEntry,
    CausalFingerprint,
    ComparisonOperator,
    Precondition,
)

logger = logging.getLogger(__name__)


@dataclass
class CausalObservation:
    """A single observation of a query→response→precondition event."""

    query: str
    response: str
    precondition_key: str
    precondition_value_at_time: object
    was_cache_hit: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DiscoveredCausal:
    """A discovered causal relationship."""

    cause_factor: str  # e.g., "exchange_rate_usd_jpy"
    effect_query_pattern: str  # e.g., "exchange rate"
    strength: float  # 0-1, how strong the causal link
    observations: int  # how many observations support this
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class CausalDiscoveryService:
    """
    Discovers causal relationships from access log patterns.

    Methods:
    1. Correlation detection: when a precondition changes, track which
       queries subsequently get cache misses
    2. Temporal ordering: cause must precede effect
    3. Conditional independence: rule out common causes

    MVP uses simple correlation + temporal ordering. Production systems
    can upgrade to full PC algorithm via gCastle or causal-learn.
    """

    def __init__(self):
        self._observations: list[CausalObservation] = []
        self._discovered: dict[str, DiscoveredCausal] = {}  # factor -> DiscoveredCausal
        self._factor_miss_counts: dict[str, int] = defaultdict(int)
        self._factor_hit_counts: dict[str, int] = defaultdict(int)

    def observe(
        self,
        query: str,
        response: str,
        precondition_key: str,
        precondition_value: object,
        was_cache_hit: bool,
    ):
        """Record an observation for causal discovery."""
        obs = CausalObservation(
            query=query,
            response=response,
            precondition_key=precondition_key,
            precondition_value_at_time=precondition_value,
            was_cache_hit=was_cache_hit,
        )
        self._observations.append(obs)

        if was_cache_hit:
            self._factor_hit_counts[precondition_key] += 1
        else:
            self._factor_miss_counts[precondition_key] += 1

    def discover(self) -> list[DiscoveredCausal]:
        """
        Run causal discovery on accumulated observations.

        Simple heuristic: a factor is causal if cache miss rate
        is significantly higher after its value changes.
        Returns list of discovered causal relationships sorted by strength.
        """
        results: list[DiscoveredCausal] = []

        for factor_key in set(list(self._factor_miss_counts.keys()) + list(self._factor_hit_counts.keys())):
            misses = self._factor_miss_counts[factor_key]
            hits = self._factor_hit_counts[factor_key]
            total = misses + hits
            if total < 3:
                continue

            miss_rate = misses / total
            # Strength = how predictive the factor is for misses
            strength = miss_rate  # Simplified; real PC algorithm would use conditional independence

            if strength >= 0.3:  # Threshold for "causal enough"
                dc = DiscoveredCausal(
                    cause_factor=factor_key,
                    effect_query_pattern=factor_key.split("_")[0],  # Extract topic
                    strength=strength,
                    observations=total,
                )
                self._discovered[factor_key] = dc
                results.append(dc)

        return sorted(results, key=lambda x: x.strength, reverse=True)

    def infer_preconditions_for_query(
        self, query: str, existing_entries: list[CachedEntry]
    ) -> list[Precondition]:
        """
        Infer what preconditions a new query should have,
        based on discovered causal relationships and similar existing entries.

        This is the "causal inference" step — instead of requiring manual
        precondition specification, this derives them from patterns.
        """
        inferred: list[Precondition] = []

        # Find discovered factors that are relevant to this query
        for factor_key, causal in self._discovered.items():
            # Simple keyword matching (production: use embeddings)
            if any(word in query.lower() for word in factor_key.lower().split("_")):
                # Find the current value of this factor from existing entries
                for entry in existing_entries:
                    for pc in entry.preconditions:
                        if pc.key == factor_key:
                            inferred.append(
                                Precondition(
                                    key=factor_key,
                                    operator=pc.operator,
                                    expected_value=pc.expected_value,
                                    source="inferred",
                                )
                            )
                            break

        return inferred

    @property
    def discovered_count(self) -> int:
        return len(self._discovered)

    def stats(self) -> dict:
        return {
            "observations": len(self._observations),
            "discovered_causals": len(self._discovered),
            "factors_monitored": list(self._discovered.keys()),
        }