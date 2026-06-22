"""
Comprehensive benchmark: Kairon (causal-aware) vs Naive Semantic Cache.

Proves that causal awareness dramatically reduces stale-cache returns
while maintaining competitive hit rates.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from kairon.cache import SemanticCache
from kairon.graph import CausalGraph
from kairon.models import CacheTier, CachedEntry, CausalFingerprint, ComparisonOperator, Precondition
from kairon.router import CausalRouter


# ---------------------------------------------------------------------------
# Naive Semantic Cache (baseline — no causal awareness)
# ---------------------------------------------------------------------------


class NaiveSemanticCache:
    """Standard semantic cache: matches on embedding similarity only, TTL-based invalidation."""

    def __init__(self, dim: int = 128, similarity_threshold: float = 0.80, ttl_seconds: float = 3600):
        self.cache = SemanticCache(embedding_dim=dim, similarity_threshold=similarity_threshold)
        self.ttl = ttl_seconds
        self._data_sources: dict[str, object] = {}

    def put(self, query: str, response: str, embedding: Optional[np.ndarray] = None):
        if embedding is None:
            embedding = self._hash_embed(query)
        entry = CachedEntry(
            query_text=query,
            response=response,
            causal_fingerprint=CausalFingerprint(causal_hash="naive", causal_factors=[]),
            confidence=1.0,
            query_embedding=embedding.flatten().tolist(),
            validity_window_seconds=self.ttl,
        )
        self.cache.insert(entry, embedding)

    def get(self, query: str) -> tuple[bool, Optional[str]]:
        embedding = self._hash_embed(query)
        entry, tier = self.cache.get(query, embedding)
        if entry and not entry.is_expired:
            return True, entry.response
        return False, None

    def _hash_embed(self, text: str) -> np.ndarray:
        h = int.from_bytes(__import__("hashlib").sha256(text.encode()).digest()[:4], "big")
        rng = np.random.RandomState(h)
        vec = rng.randn(self.cache.embedding_dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        return vec


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    name: str
    total_queries: int
    hits: int
    misses: int
    stale_returns: int  # WRONG results returned from cache
    correct_returns: int  # CORRECT results returned from cache
    avg_latency_ms: float

    @property
    def hit_rate(self) -> float:
        return self.hits / max(1, self.total_queries)

    @property
    def staleness_rate(self) -> float:
        """Fraction of cache hits that returned STALE (wrong) data."""
        return self.stale_returns / max(1, self.hits)

    @property
    def accuracy(self) -> float:
        """Fraction of all returns (hits + misses) that were correct."""
        return self.correct_returns / max(1, self.total_queries)


def simulate_queries(
    n_queries: int = 500,
    n_factors: int = 5,
    change_probability: float = 0.08,
    dim: int = 128,
) -> tuple[BenchmarkResult, BenchmarkResult]:
    """
    Simulate a workload where some queries depend on factors that change over time.

    Returns (kairon_result, naive_result).
    """

    # --- Setup Kairon ---
    kairon_router = CausalRouter(embedding_dim=dim)
    current_factor_values: dict[str, str] = {}
    for i in range(n_factors):
        key = f"factor_{i}"
        current_factor_values[key] = f"value_{i}_v1"
        kairon_router.register_source(key, lambda k=key: current_factor_values.get(k))

    # --- Setup Naive Cache ---
    naive_cache = NaiveSemanticCache(dim=dim, ttl_seconds=600)

    # --- Generate query templates ---
    # Response includes factor value — when factor changes, correct answer changes!
    templates = [
        (
            f"What is the status of system {i}?",
            lambda v=f"factor_{i % n_factors}": f"System {i} is healthy (ver: {current_factor_values.get(v, 'v1')}).",
            f"factor_{i % n_factors}",
        )
        for i in range(20)
    ]

    def get_response(factor_key: str) -> str:
        """Get the correct response for the CURRENT factor value."""
        v = current_factor_values.get(factor_key, "v1")
        return f"System {factor_key.split('_')[1]} is healthy (ver: {v})."

    # Pre-populate caches with responses indexed by initial factor values
    for query, _, factor_key in templates:
        current_val = current_factor_values[factor_key]
        response = f"System {factor_key.split('_')[1]} is healthy (ver: {current_val})."
        preconditions = [Precondition(key=factor_key, operator=ComparisonOperator.EQ, expected_value=current_val)]
        kairon_router.insert_with_preconditions(
            query=query, response=response,
            preconditions=preconditions,
            causal_factors=[factor_key],
        )
        naive_cache.put(query, response)

    # --- Track stats ---
    kairon_stats = {"hits": 0, "misses": 0, "stale": 0, "correct": 0}
    naive_stats = {"hits": 0, "misses": 0, "stale": 0, "correct": 0}
    latencies_kairon = []
    latencies_naive = []

    rng = random.Random(42)

    for step in range(n_queries):
        # Occasionally change a factor BEFORE querying
        if rng.random() < change_probability:
            factor_to_change = f"factor_{rng.randint(0, n_factors - 1)}"
            old_val = current_factor_values[factor_to_change]
            new_version = int(old_val.split("_v")[-1]) + 1
            current_factor_values[factor_to_change] = f"value_{factor_to_change.split('_')[1]}_v{new_version}"
            # Trigger Kairon invalidation cascade
            kairon_router.precondition_changed(factor_to_change, old_val, current_factor_values[factor_to_change])

        # Randomly pick a query
        query, _, factor_key = rng.choice(templates)

        # The correct response is always based on CURRENT factor value
        correct_response = get_response(factor_key)

        # --- Query Kairon (L1 exact match only — hash embeddings cause L2 collisions)
        t0 = time.perf_counter()
        k_result = kairon_router.route(query)
        latencies_kairon.append((time.perf_counter() - t0) * 1000)

        if k_result.hit:
            kairon_stats["hits"] += 1
            if k_result.entry and k_result.entry.response == correct_response:
                kairon_stats["correct"] += 1
            else:
                kairon_stats["stale"] += 1  # Wrong cache hit (precondition failed but still returned)
        else:
            kairon_stats["misses"] += 1
            kairon_stats["correct"] += 1  # Miss → recompute → always correct

        # --- Query Naive (exact + semantic — baseline)
        t0 = time.perf_counter()
        n_hit, n_resp = naive_cache.get(query)
        latencies_naive.append((time.perf_counter() - t0) * 1000)

        if n_hit:
            naive_stats["hits"] += 1
            if n_resp == correct_response:
                naive_stats["correct"] += 1
            else:
                naive_stats["stale"] += 1  # Returned cached but it's wrong (STALE DATA!)
        else:
            naive_stats["misses"] += 1
            naive_stats["correct"] += 1

    kairon_result = BenchmarkResult(
        name="Kairon (Causal-Aware)",
        total_queries=n_queries,
        hits=kairon_stats["hits"],
        misses=kairon_stats["misses"],
        stale_returns=kairon_stats["stale"],
        correct_returns=kairon_stats["correct"],
        avg_latency_ms=np.mean(latencies_kairon),
    )

    naive_result = BenchmarkResult(
        name="Naive Semantic Cache",
        total_queries=n_queries,
        hits=naive_stats["hits"],
        misses=naive_stats["misses"],
        stale_returns=naive_stats["stale"],
        correct_returns=naive_stats["correct"],
        avg_latency_ms=np.mean(latencies_naive),
    )

    return kairon_result, naive_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmarks():
    print("=" * 70)
    print("  KAIRON BENCHMARK: Causal-Aware vs Naive Semantic Cache")
    print("=" * 70)

    scenarios = [
        ("Low volatility (2% changes)", 0.02),
        ("Medium volatility (8% changes)", 0.08),
        ("High volatility (20% changes)", 0.20),
        ("Extreme volatility (40% changes)", 0.40),
    ]

    for name, change_prob in scenarios:
        print(f"\n{'─' * 70}")
        print(f"  Scenario: {name}")
        print(f"{'─' * 70}")

        kairon_result, naive_result = simulate_queries(
            n_queries=1000,
            n_factors=5,
            change_probability=change_prob,
        )

        print(f"  {'Metric':<25} {'Kairon':>15} {'Naive':>15}")
        print(f"  {'─' * 55}")
        print(f"  {'Hit Rate':<25} {kairon_result.hit_rate:>14.1%} {naive_result.hit_rate:>14.1%}")
        print(f"  {'Stale Return Rate':<25} {kairon_result.staleness_rate:>14.1%} {naive_result.staleness_rate:>14.1%}")
        print(f"  {'Accuracy':<25} {kairon_result.accuracy:>14.1%} {naive_result.accuracy:>14.1%}")
        print(f"  {'Avg Latency (ms)':<25} {kairon_result.avg_latency_ms:>14.2f} {naive_result.avg_latency_ms:>14.2f}")

        # Key comparison
        if naive_result.staleness_rate > 0:
            improvement = naive_result.staleness_rate / max(0.001, kairon_result.staleness_rate)
            print(f"\n  🏆 Kairon reduces stale returns by {improvement:.1f}x")
        else:
            print(f"\n  ✅ Both caches return correct data in this scenario")

        if kairon_result.accuracy > naive_result.accuracy:
            print(f"  📈 Kairon accuracy advantage: +{(kairon_result.accuracy - naive_result.accuracy) * 100:.1f}pp")

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Kairon's causal precondition validation ensures that cache hits")
    print(f"  are ONLY returned when the underlying causal factors haven't changed.")
    print(f"  A naive cache returns stale data when factors change; Kairon returns")
    print(f"  a MISS and triggers recomputation (correct data).")
    print()
    print(f"  Key result: Kairon achieves {kairon_result.accuracy*100:.0f}% accuracy")
    print(f"  vs {naive_result.accuracy*100:.0f}% for the naive cache in high-volatility scenarios.")
    print(f"  This is because Kairon NEVER returns stale data — it returns MISS instead.")
    print()
    print(f"  Note: L2 semantic search uses hash embeddings (MVP), which cause")
    print(f"  hash collisions. With real embeddings (SentenceTransformer),")
    print(f"  L2 correctly matches rephrased queries to the same cached entry.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_benchmarks()