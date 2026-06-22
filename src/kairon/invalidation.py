"""
Predictive Invalidation Engine — predicts WHEN cached entries will become invalid.

Uses a combination of:
1. Causal precondition monitoring (real-time validation)
2. Temporal pattern analysis (when do preconditions typically change?)
3. Drift detection (gradual vs sudden changes in factor values)

MVP: heuristic-based prediction. Production: RL policy (PPO/SAC).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

import numpy as np

from .models import CachedEntry, Precondition

logger = logging.getLogger(__name__)


class InvalidationDecision(str, Enum):
    KEEP = "keep"
    INVALIDATE_NOW = "invalidate_now"
    SCHEDULE_INVALIDATION = "schedule_invalidation"


@dataclass
class InvalidationPrediction:
    """A prediction about when a cached entry will become invalid."""

    entry_id: str
    decision: InvalidationDecision
    confidence: float  # 0-1
    reason: str
    predicted_invalid_at: Optional[datetime] = None
    at_risk_preconditions: list[str] = field(default_factory=list)


@dataclass
class FactorChange:
    """Record of a factor value change."""

    factor_key: str
    old_value: object
    new_value: object
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PredictiveInvalidationEngine:
    """
    Predicts cache invalidation before it happens.

    Monitors:
    - Factor change frequency (how often does a precondition change?)
    - Time since last verification (stale entries)
    - Volatility (how much does a factor typically change?)

    Strategies:
    - High-volatility factor → short validity window
    - Low-volatility factor → long validity window
    - Factor approaching known change point → schedule invalidation
    """

    def __init__(
        self,
        default_validity_seconds: float = 3600.0,  # 1 hour
        volatility_window: int = 50,  # Last N changes for volatility calc
        stale_threshold_seconds: float = 86400.0,  # 24 hours
    ):
        self._default_validity = default_validity_seconds
        self._volatility_window = volatility_window
        self._stale_threshold = stale_threshold_seconds

        # Factor change history
        self._change_history: dict[str, list[FactorChange]] = defaultdict(list)

        # Factor volatility tracking
        self._volatility: dict[str, float] = {}

    def record_factor_change(self, key: str, old_value: object, new_value: object):
        """Record a change in a factor's value."""
        change = FactorChange(factor_key=key, old_value=old_value, new_value=new_value)
        self._change_history[key].append(change)

        # Update volatility
        self._update_volatility(key)

    def should_invalidate(self, entry: CachedEntry) -> InvalidationPrediction:
        """
        Predict whether a cached entry should be invalidated.

        Decision logic:
        1. If any precondition's factor is high-volatility → schedule soon
        2. If entry is stale (old) → mark for invalidation
        3. If recent factor changes suggest invalidation → invalidate now
        4. Otherwise → keep
        """
        at_risk: list[str] = []
        min_predicted_ttl: Optional[float] = None

        for pc in entry.preconditions:
            volatility = self._volatility.get(pc.key, 0.0)

            if volatility > 0.5:
                # High volatility — preconditions likely to change soon
                at_risk.append(pc.key)
                # Estimate TTL inversely proportional to volatility
                estimated_ttl = self._default_validity * (1.0 - volatility)
                if min_predicted_ttl is None or estimated_ttl < min_predicted_ttl:
                    min_predicted_ttl = estimated_ttl

            # Check if there are recent changes to this factor
            recent_changes = self._recent_changes(pc.key, window_hours=1.0)
            if recent_changes:
                at_risk.append(pc.key)
                # Recent changes → invalidate sooner
                if min_predicted_ttl is None or 300.0 < min_predicted_ttl:
                    min_predicted_ttl = 300.0  # 5 minutes

        # Check staleness
        age = (datetime.now(timezone.utc) - entry.created_at).total_seconds()
        if age > self._stale_threshold:
            return InvalidationPrediction(
                entry_id=entry.id,
                decision=InvalidationDecision.INVALIDATE_NOW,
                confidence=0.9,
                reason=f"Entry is {age:.0f}s old (stale threshold: {self._stale_threshold:.0f}s)",
                at_risk_preconditions=at_risk or ["age"],
            )

        # Check expiration
        if entry.is_expired:
            return InvalidationPrediction(
                entry_id=entry.id,
                decision=InvalidationDecision.INVALIDATE_NOW,
                confidence=1.0,
                reason="Validity window expired",
            )

        # Decision based on risk assessment
        if not at_risk:
            return InvalidationPrediction(
                entry_id=entry.id,
                decision=InvalidationDecision.KEEP,
                confidence=0.8,
                reason="No high-volatility factors; preconditions likely stable",
            )

        if min_predicted_ttl and min_predicted_ttl < 60:
            return InvalidationPrediction(
                entry_id=entry.id,
                decision=InvalidationDecision.INVALIDATE_NOW,
                confidence=0.7,
                reason=f"High-volatility factors: {at_risk}; estimated TTL: {min_predicted_ttl:.0f}s",
                at_risk_preconditions=at_risk,
            )

        return InvalidationPrediction(
            entry_id=entry.id,
            decision=InvalidationDecision.SCHEDULE_INVALIDATION,
            confidence=0.6,
            reason=f"Moderate risk factors: {at_risk}",
            predicted_invalid_at=datetime.now(timezone.utc) + timedelta(seconds=min_predicted_ttl or self._default_validity),
            at_risk_preconditions=at_risk,
        )

    def estimate_validity_window(self, entry: CachedEntry) -> float:
        """Estimate how long (seconds) a cached entry will remain valid."""
        prediction = self.should_invalidate(entry)
        if prediction.decision == InvalidationDecision.INVALIDATE_NOW:
            return 0.0
        elif prediction.decision == InvalidationDecision.KEEP:
            return self._default_validity
        elif prediction.predicted_invalid_at:
            remaining = (prediction.predicted_invalid_at - datetime.now(timezone.utc)).total_seconds()
            return max(0.0, remaining)
        return self._default_validity

    # ------------------------------------------------------------------
    # Volatility tracking
    # ------------------------------------------------------------------

    def _update_volatility(self, factor_key: str):
        """Update volatility score for a factor based on change frequency."""
        history = self._change_history[factor_key]
        if len(history) < 2:
            return

        # Calculate change inter-arrival times
        recent = history[-self._volatility_window:]
        intervals = []
        for i in range(1, len(recent)):
            dt = (recent[i].timestamp - recent[i - 1].timestamp).total_seconds()
            intervals.append(dt)

        if not intervals:
            return

        # Volatility = 1 / mean_interval (higher = more volatile)
        mean_interval = sum(intervals) / len(intervals)
        # Normalize: 1 change per hour = volatility ~0.28, 1 per minute = ~16.7
        raw = 1.0 / max(mean_interval, 1.0)
        # Scale to [0, 1] with logistic function
        self._volatility[factor_key] = 1.0 / (1.0 + float(np.exp(-0.3 * (raw - 5.0)))) if float(np.exp(-0.3 * (raw - 5.0))) != 0 else 1.0

    def _recent_changes(self, factor_key: str, window_hours: float = 1.0) -> list[FactorChange]:
        """Get recent factor changes within a time window."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        return [
            c for c in self._change_history.get(factor_key, [])
            if c.timestamp >= cutoff
        ]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "tracked_factors": len(self._change_history),
            "volatility_scores": {k: f"{v:.3f}" for k, v in self._volatility.items()},
            "total_changes": sum(len(v) for v in self._change_history.values()),
        }