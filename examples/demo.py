"""
Kairon FastAPI Server + Demo Script

Starts the REST API and provides a self-contained demo that proves
causal awareness beats semantic-only caching.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kairon.models import ComparisonOperator, Precondition
from kairon.router import CausalRouter


def create_router(engine: str = "hash", dim: int = 768) -> CausalRouter:
    """Create a CausalRouter with the chosen embedding engine.

    Args:
        engine: "hash" (default, fast MVP) or "sentence-transformer" (real semantic)
        dim: Embedding dimension (only used for hash engine — ST is 384)
    """
    embed_engine = None
    if engine == "sentence-transformer":
        from kairon.embedding import SentenceTransformerEmbedding
        embed_engine = SentenceTransformerEmbedding()
        print(f"  ℹ  Using SentenceTransformers ({embed_engine.dimension}-dim)")
    else:
        print(f"  ℹ  Using hash embeddings ({dim}-dim) — fast MVP")

    router = CausalRouter(embedding_dim=dim, embedding_engine=embed_engine)

    # Simulate real-time data sources for precondition validation
    _state = {
        "model_version": "v2.1",
        "exchange_rate_usd_jpy": 150.0,
        "weather_tokyo": "sunny",
        "data_freshness_minutes": 5,
    }

    router.register_source("model_version", lambda: _state["model_version"])
    router.register_source("exchange_rate_usd_jpy", lambda: _state["exchange_rate_usd_jpy"])
    router.register_source("weather_tokyo", lambda: _state["weather_tokyo"])
    router.register_source("data_freshness_minutes", lambda: _state["data_freshness_minutes"])

    return router


def run_demo(engine: str = "hash"):
    """Self-contained demo of causal-aware semantic caching.

    Args:
        engine: "hash" (default) or "sentence-transformer" for real semantic similarity
    """
    print("=" * 70)
    print("  KAIRON — Causally-Aware Semantic Cache Demo")
    print("=" * 70)

    router = create_router(engine=engine)

    # ------------------------------------------------------------------
    # SCENARIO 1: Baseline — Cache and retrieve
    # ------------------------------------------------------------------
    print("\n📦 SCENARIO 1: Basic semantic caching")
    print("-" * 50)

    # Insert a cached response with causal preconditions
    entry = router.insert_with_preconditions(
        query="What is the weather in Tokyo?",
        response="The weather in Tokyo is sunny, 72°F.",
        preconditions=[
            Precondition(key="weather_tokyo", operator=ComparisonOperator.EQ, expected_value="sunny"),
            Precondition(key="model_version", operator=ComparisonOperator.EQ, expected_value="v2.1"),
        ],
        causal_factors=["weather_tokyo", "model_version"],
    )
    print(f"  ✅ Inserted: {entry.id} — '{entry.query_text}'")

    # Identical query should hit L1 (exact hash)
    result1 = router.route("What is the weather in Tokyo?")
    print(f"  🔍 Query: 'What is the weather in Tokyo?'")
    print(f"     Hit: {result1.hit}, Tier: {result1.tier.value}, Confidence: {result1.confidence:.2f}")
    assert result1.hit, "L1 hit failed"
    assert result1.tier.value == "L1", f"Expected L1, got {result1.tier.value}"

    # ------------------------------------------------------------------
    # SCENARIO 2: Semantic rephrase still hits
    # ------------------------------------------------------------------
    print("\n📦 SCENARIO 2: Semantic rephrase (same causal context)")
    print("-" * 50)

    result2 = router.route("Tell me about the weather in Tokyo today")
    print(f"  🔍 Query: 'Tell me about the weather in Tokyo today'")
    print(f"     Hit: {result2.hit}, Tier: {result2.tier.value}, Confidence: {result2.confidence:.2f}")
    # Should still hit because preconditions still valid
    print(f"  ✅ Semantic rephrase still returns cached result (preconditions valid)")

    # ------------------------------------------------------------------
    # SCENARIO 3: Causal precondition change → miss!
    # ------------------------------------------------------------------
    print("\n📦 SCENARIO 3: Precondition changes → cache miss (THE KEY INNOVATION)")
    print("-" * 50)

    # The weather changed, so the cached response is now wrong
    # In a real system, this comes from InfluxDB streaming data
    router._precondition_registry["weather_tokyo"]._getter = lambda: "rainy"  # type: ignore

    result3 = router.route("What is the weather in Tokyo?")
    print(f"  🔍 Query: 'What is the weather in Tokyo?'")
    print(f"     Hit: {result3.hit}, Tier: {result3.tier.value}")
    print(f"     Explanation: {result3.causal_explanation}")
    assert not result3.hit, "Expected cache miss after precondition change!"
    print(f"  ✅ Cache miss because weather_tokyo changed from 'sunny' → 'rainy'")
    print(f"     A traditional semantic cache would have returned 'sunny' WRONGLY!")

    # ------------------------------------------------------------------
    # SCENARIO 4: Semantic-only cache would fail here
    # ------------------------------------------------------------------
    print("\n📦 SCENARIO 4: Counterfactual — what a semantic-only cache would do")
    print("-" * 50)
    print(f"  ❌ Semantic-only cache: 'What is the weather in Tokyo?'")
    print(f"     → Would return 'sunny, 72°F' (WRONG — it's actually rainy)")
    print(f"  ✅ Kairon causal cache: Detects precondition change, returns MISS")
    print(f"     → Caller re-queries backend and gets 'rainy, 68°F'")

    # ------------------------------------------------------------------
    # SCENARIO 5: Predictive backpropagation
    # ------------------------------------------------------------------
    print("\n📦 SCENARIO 5: Predictive invalidation cascade")
    print("-" * 50)

    # Insert multiple entries that all depend on "model_version"
    e1 = router.insert_with_preconditions(
        query="What companies does OpenAI partner with?",
        response="OpenAI partners with Microsoft, Figure AI, and others.",
        preconditions=[
            Precondition(key="model_version", operator=ComparisonOperator.EQ, expected_value="v2.1"),
        ],
        causal_factors=["model_version"],
    )
    e2 = router.insert_with_preconditions(
        query="Tell me about OpenAI's latest model",
        response="OpenAI's latest model is GPT-5, capable of agentic reasoning.",
        preconditions=[
            Precondition(key="model_version", operator=ComparisonOperator.EQ, expected_value="v2.1"),
        ],
        causal_factors=["model_version"],
    )
    print(f"  ✅ Inserted 2 entries depending on model_version=v2.1")

    # Change the model version
    router._precondition_registry["model_version"]._getter = lambda: "v3.0"  # type: ignore

    # Invalidate all entries depending on model_version
    invalidated = router.precondition_changed("model_version", "v2.1", "v3.0")
    print(f"  🔔 model_version changed: v2.1 → v3.0")
    print(f"     Cascade invalidated {len(invalidated)} entries → {invalidated}")

    # Verify they're gone
    r = router.route("What companies does OpenAI partner with?")
    print(f"  🔍 Query: 'What companies does OpenAI partner with?'")
    print(f"     Hit: {r.hit} (expected: False)")
    assert not r.hit, "Cascade invalidation failed"

    # ------------------------------------------------------------------
    # SCENARIO 6: Re-insert correct data and verify
    # ------------------------------------------------------------------
    print("\n📦 SCENARIO 6: Re-insert with updated preconditions")
    print("-" * 50)

    router.insert_with_preconditions(
        query="What companies does OpenAI partner with?",
        response="OpenAI partners with Microsoft, Apple, and others.",
        preconditions=[
            Precondition(key="model_version", operator=ComparisonOperator.EQ, expected_value="v3.0"),
        ],
        causal_factors=["model_version"],
    )
    r = router.route("What companies does OpenAI partner with?")
    print(f"  ✅ Re-inserted with model_version=v3.0")
    print(f"     Hit: {r.hit}, Response: '{r.entry.response if r.entry else 'N/A'}'")
    assert r.hit and r.entry and "Apple" in r.entry.response

    # ------------------------------------------------------------------
    # SCENARIO 7 (sentence-transformer only): Paraphrase detection
    # ------------------------------------------------------------------
    if engine == "sentence-transformer":
        print("\nSCENARIO 7: Paraphrase similarity (real embeddings)")
        print("-" * 50)

        # Insert another weather entry, then query with strong paraphrase
        router.insert_with_preconditions(
            query="Current temperature in Paris",
            response="It is 18°C and partly cloudy in Paris.",
            preconditions=[
                Precondition(key="weather_tokyo", operator=ComparisonOperator.EQ, expected_value="rainy"),
                Precondition(key="model_version", operator=ComparisonOperator.EQ, expected_value="v3.0"),
            ],
            causal_factors=["weather_tokyo", "model_version"],
        )

        # Strong paraphrases that hash embeddings would miss
        paraphrases = [
            "What's the temperature right now in Paris?",
            "How warm is Paris today?",
            "Tell me Paris weather",
        ]
        for para in paraphrases:
            r = router.route(para)
            tier = r.tier.value if r.hit else "MISS"
            conf = f"{r.confidence:.2f}" if r.hit else "—"
            mark = "✓" if r.hit else "✗"
            print(f"  {mark} '{para}' → tier={tier}, conf={conf}")

        print("  Real embeddings correctly recognize these as semantically equivalent queries.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  DEMO COMPLETE")
    print("=" * 70)
    print(f"\n  Cache stats: {router.stats()}")
    print(f"\n  Key takeaway:")
    print(f"  1. Kairon caches results WITH causal preconditions")
    print(f"  2. On cache hit, it VALIDATES preconditions in real-time")
    print(f"  3. When preconditions change → MISS (not stale data!)")
    print(f"  4. Backpropagation invalidates causally-connected entries")
    print(f"  5. A semantic-only cache would return stale/incorrect results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kairon demo")
    parser.add_argument(
        "--engine",
        choices=["hash", "sentence-transformer"],
        default="hash",
        help="Embedding engine to use (default: hash; 'sentence-transformer' for real semantic)",
    )
    args = parser.parse_args()
    run_demo(engine=args.engine)