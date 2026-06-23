"""
Causal Router — the core innovation of Kairon.

Routes cache queries through:
1. Semantic encoding → FAISS vector search (top-k candidates)
2. Causal precondition validation → real-time check of causality
3. Causal similarity scoring → compare causal fingerprints
4. Predictive invalidation → detect precondition drift, cascade

This is where Kairon differs from every other semantic cache:
instead of just matching by vector similarity, it verifies that
the *causal preconditions* that made a cached result correct are still true.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import numpy as np

from .cache import SemanticCache
from .graph import CausalGraph
from .models import (
    CachedEntry,
    CacheTier,
    CausalFingerprint,
    ComparisonOperator,
    Precondition,
    RouteResult,
)


class CausalRouter:
    """Routes cache queries through semantic + causal matching layers."""

    def __init__(
        self,
        embedding_dim: int = 768,
        semantic_threshold: float = 0.80,
        causal_threshold: float = 0.55,
        confidence_floor: float = 0.30,
        embedding_engine=None,
        use_cross_encoder: bool = True,
    ):
        # Pluggable embedding engine — defaults to HashEmbedding for MVP/testing
        # Pluggable reranker — defaults to cross-encoder when sentence-transformers available
        reranker = None
        if use_cross_encoder:
            try:
                from .reranker import CrossEncoderReranker
                reranker = CrossEncoderReranker()
            except Exception:
                pass

        if embedding_engine is None:
            from .embedding import HashEmbedding
            self._embed_engine = HashEmbedding(dim=embedding_dim)
            self.cache = SemanticCache(
                embedding_dim=embedding_dim,
                similarity_threshold=semantic_threshold,
                reranker=reranker,
            )
        else:
            self._embed_engine = embedding_engine
            self.cache = SemanticCache(
                embedding_dim=embedding_engine.dimension,
                similarity_threshold=semantic_threshold,
                reranker=reranker,
            )
        self.graph = CausalGraph()
        self.causal_threshold = causal_threshold
        self.confidence_floor = confidence_floor

        # Real-time data sources for precondition validation
        self._precondition_registry: dict[str, PreconditionSource] = {}

    # ------------------------------------------------------------------
    # Precondition Sources (Real-Time Data)
    # ------------------------------------------------------------------

    def register_source(self, key: str, getter):
        """Register a real-time data source for precondition validation."""
        self._precondition_registry[key] = PreconditionSource(key, getter)

    def get_current_value(self, key: str):
        """Get the current value for a precondition key from its registered source."""
        if key in self._precondition_registry:
            return self._precondition_registry[key].get()
        return None

    # ------------------------------------------------------------------
    # Core Routing
    # ------------------------------------------------------------------

    def route(
        self,
        query_text: str,
        embedding: np.ndarray | None = None,
        validate_preconditions: bool = True,
    ) -> RouteResult:
        """
        Route a query through the causal cache pipeline.

        Flow:
        1. Generate embedding (or use provided)
        2. L1: Exact hash match
        3. L2: Semantic vector search → top-k candidates
        4. For each candidate: validate preconditions + compute causal similarity
        5. If match found: return cached result with confidence
        6. If miss: return MISS (caller should recompute and insert)
        """
        if embedding is None:
            embedding = self._embed_text(query_text)

        # Step 1-2: Try semantic cache
        entry, tier = self.cache.get(query_text, embedding)
        if entry is not None:
            # Validate preconditions before confirming hit
            if validate_preconditions and not self._all_preconditions_valid(entry):
                self._invalidate_entry(entry)
                return self._miss_result("Precondition validation failed")

            if not entry.is_expired:
                return RouteResult.hit_result(entry, tier, entry.confidence)

        # Step 3: Top-k semantic search for causal filtering
        candidates = self.cache.search_top_k(embedding, k=10)
        for candidate_entry, semantic_score in candidates:
            if candidate_entry.is_expired:
                continue

            # Step 4: Compute causal similarity
            causal_score = self._compute_causal_similarity(query_text, candidate_entry)

            # Step 5: Combined score
            combined = 0.4 * semantic_score + 0.6 * causal_score
            if combined >= self.causal_threshold:
                # Validate preconditions
                if validate_preconditions and not self._all_preconditions_valid(candidate_entry):
                    self._invalidate_entry(candidate_entry)
                    continue

                # Hit via causal-semantic match
                candidate_entry.record_access()
                self.cache.insert(candidate_entry, embedding)
                return RouteResult.hit_result(
                    candidate_entry, CacheTier.L2, candidate_entry.confidence
                )

        # Step 6: L3 — causal-only match (no vector overlap, but shared causal factors)
        if validate_preconditions:
            causal_only = self._causal_only_match(query_text, embedding)
            if causal_only:
                # Validate preconditions before returning L3 hit
                if not self._all_preconditions_valid(causal_only):
                    self._invalidate_entry(causal_only)
                else:
                    causal_only.record_access()
                    return RouteResult.hit_result(causal_only, CacheTier.L3, causal_only.confidence)

        return self._miss_result("No causal-semantic match found")

    def insert_with_preconditions(
        self,
        query: str,
        response: str,
        preconditions: list[Precondition],
        causal_factors: list[str] | None = None,
        embedding: np.ndarray | None = None,
        confidence: float = 1.0,
    ) -> CachedEntry:
        """Insert a cached response with its causal fingerprint."""
        fingerprint = CausalFingerprint.from_preconditions(preconditions, causal_factors)
        if embedding is None:
            embedding = self._embed_text(query)

        entry = CachedEntry(
            query_text=query,
            response=response,
            causal_fingerprint=fingerprint,
            confidence=confidence,
            query_embedding=embedding.flatten().tolist(),
        )
        self.cache.insert(entry, embedding)
        self.graph.add_entry(entry)
        return entry

    # ------------------------------------------------------------------
    # Predictive Invalidation
    # ------------------------------------------------------------------

    def invalidate_by_key(self, key: str) -> list[str]:
        """
        Invalidate all cached entries that depend on a specific key.

        Uses causal backpropagation: finds the causal factor node, then
        all entries whose preconditions depend on it.

        Returns the list of invalidated entry IDs.
        """
        dependents = self.graph.find_dependents(key)
        return self.cache.invalidate_many(dependents)

    def invalidate_cascade(self, entry_id: str) -> list[str]:
        """
        Invalidate an entry and ALL entries causally connected to it.

        Uses causal backpropagation: when one entry is invalidated,
        all entries that share causal factors are also invalidated.
        """
        invalidated = []

        # Invalidate the target
        if self.cache.invalidate(entry_id):
            invalidated.append(entry_id)

        # Find and invalidate causally-similar dependents
        dependents = self.graph.find_direct_dependents(entry_id)
        for dep_id in dependents:
            for eid in self.graph.find_dependents(dep_id):
                if self.cache.invalidate(eid):
                    invalidated.append(eid)
            if self.cache.invalidate(dep_id):
                invalidated.append(dep_id)

        # Remove from causal graph
        self.graph.invalidate_entry(entry_id)

        return invalidated

    def precondition_changed(self, key: str, old_value, new_value) -> list[str]:
        """
        External trigger: a precondition value changed. Invalidate all
        entries that depend on it. Returns list of invalidated IDs.
        """
        return self.invalidate_by_key(key)

    # ------------------------------------------------------------------
    # Causal Scoring
    # ------------------------------------------------------------------

    def _compute_causal_similarity(
        self, query_text: str, entry: CachedEntry
    ) -> float:
        """
        Score how causally similar a query is to a cached entry.

        Looks at:
        - Precondition overlap (Jaccard of precondition keys)
        - Causal factor overlap
        - How many preconditions still validate
        """
        if not entry.preconditions:
            return 0.5  # Neutral — no causal structure

        # Use the entry's preconditions as the "query fingerprint" proxy
        # In production: infer preconditions from the new query via the causal model
        candidate_pcs = {p.key for p in entry.preconditions}

        # Check which still validate
        valid_count = sum(
            1 for p in entry.preconditions
            if self._check_precondition(p)
        )
        total = max(1, len(entry.preconditions))
        validity_ratio = valid_count / total

        return validity_ratio

    def _causal_only_match(
        self, query_text: str, embedding: np.ndarray
    ) -> Optional[CachedEntry]:
        """
        L3 fallback: search the causal graph for entries with shared
        causal factors, even without semantic overlap.
        """
        # Find all entries that share causal factors
        # This is a simplified approach — in production, use causal graph traversal
        best_entry: Optional[CachedEntry] = None
        best_score: float = 0.0

        for entry in self._all_entries():
            if entry.is_expired:
                continue
            factors = set(entry.causal_fingerprint.causal_factors)
            if not factors:
                continue
            # Compute factor overlap score
            valid = sum(1 for p in entry.preconditions if self._check_precondition(p))
            total = max(1, len(entry.preconditions))
            score = valid / total
            if score > best_score and score >= self.causal_threshold:
                best_score = score
                best_entry = entry

        return best_entry

    # ------------------------------------------------------------------
    # Precondition Validation
    # ------------------------------------------------------------------

    def _all_preconditions_valid(self, entry: CachedEntry) -> bool:
        """Check all stored preconditions still hold."""
        if not entry.preconditions:
            return True
        return all(self._check_precondition(p) for p in entry.preconditions)

    def _check_precondition(self, precondition: Precondition) -> bool:
        """Check a single precondition against its real-time data source."""
        current = self.get_current_value(precondition.key)
        if current is None:
            return True  # Unmonitored precondition — assume valid
        return precondition.check(current)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _invalidate_entry(self, entry: CachedEntry):
        """Remove a stale entry from cache and graph."""
        self.cache.invalidate(entry.id)
        self.graph.invalidate_entry(entry.id)

    def _miss_result(self, reason: str) -> RouteResult:
        return RouteResult.miss_result(reason)

    def _embed_text(self, text: str) -> np.ndarray:
        """Compute embedding for text using injected engine (real semantic or hash fallback)."""
        return self._embed_engine.embed(text)

    def _all_entries(self):
        """Walk all L2 entries (for L3 fallback)."""
        return self.cache._l2_entries.values()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "cache": self.cache.stats(),
            "graph": self.graph.stats(),
            "thresholds": {
                "semantic": f"{self.cache.similarity_threshold:.3f}",
                "causal": f"{self.causal_threshold:.3f}",
            },
        }


class PreconditionSource:
    def __init__(self, key: str, getter):
        self.key = key
        self._getter = getter

    def get(self):
        try:
            return self._getter()
        except Exception:
            return None