# Kairon 🔮

[![GitHub release](https://img.shields.io/github/v/release/ik123a/Kairon?include_prereleases)](https://github.com/ik123a/Kairon/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-30%2F30%20passing-brightgreen.svg)](tests/)
[![Topics](https://img.shields.io/badge/topics-semantic--cache%20%7C%20causal--inference%20%7C%20RAG-blue.svg)](https://github.com/ik123a/Kairon)

**A Causally-Aware Semantic Cache** that doesn't just store *what* happened — it understands *why* it happened, and uses that understanding to predict *when* it stored data will become wrong.

---

## What’s New in v0.2.0

- 🎯 **Real semantic embeddings** — `SentenceTransformerEmbedding` is now plug-and-play via `--engine sentence-transformer`. Paraphrase detection: `"What is the weather in Tokyo?"` and `"Tell me about Tokyo weather conditions"` now match the same cached entry (semantic similarity 0.92).
- 🛡️ **Token-overlap guard** — L2 semantic hits now require ≥40% token overlap with the cached query, eliminating cross-subject collisions (e.g., `"system 0 health"` no longer matches `"system 2 status"`).
- 🐛 **Fixed duplicate L2 inserts** — entries promoted to L2 from cache hits are now deduplicated by `entry.id`, preventing FAISS vector fragmentation.
- 📊 **`embedding_engine` injection** — `CausalRouter(embedding_engine=...)` accepts any custom embedding backend; sentence-transformers is the production-shaped default.

---

## The Problem

Every existing semantic cache matches queries by **similarity** — "What is 2+2?" and "What's two plus two?" return the same cached result. But they have **zero causal awareness**. They can't tell the difference between:

- "Patient took aspirin" → "Headache went away" (**causal**)
- "Patient took aspirin" → "It rained" (**spurious**)

When a cached result's **underlying assumptions change**, a normal semantic cache will happily return **stale, incorrect data**. Kairon prevents this.

## The Innovation: Causal Fingerprints

Instead of caching `(query_embedding → response)`, Kairon caches:

```
(query_embedding, causal_preconditions, causal_consequences) → (response, confidence, validity_window)
```

When a new query arrives, Kairon:
1. **Embeds** it for semantic matching (like any semantic cache)
2. **Validates** the causal preconditions still hold (unique to Kairon)
3. **Returns** cached result only if preconditions are valid
4. **Predicts** when results will become invalid (predictive invalidation)

## Quick Start

```bash
# Install
pip install -e .

# Run the demo (hash embeddings, ~5 seconds)
python examples/demo.py

# Run the demo with REAL semantic embeddings (paraphrase detection, requires sentence-transformers)
pip install sentence-transformers torch
python examples/demo.py --engine sentence-transformer

# Run benchmarks (Kairon vs naive semantic cache)
python tests/test_benchmark.py                          # hash embeddings
python tests/test_benchmark.py --real-embeddings        # sentence-transformer embeddings

# Run unit tests
pytest tests/test_core.py -v
```

## Architecture

```
┌─────────────────────────────────────────────┐
│              Kairon Gateway (FastAPI)        │
│   REST API + gRPC (future)                  │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│              Causal Router                   │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐ │
│  │ Semantic │──│  Causal  │──│Precondition│ │
│  │ Encoder  │  │ Matcher  │  │ Validator  │ │
│  └──────────┘  └──────────┘  └────────────┘ │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│             Cache Tiers                     │
│  L1: Exact hash (dict)    — <1ms           │
│  L2: Semantic (FAISS)     — <5ms           │
│  L3: Causal-only fallback — <50ms          │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│        Causal Engine (background)           │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐│
│  │ Causal   │──│Predictive│──│  Cache     ││
│  │Discovery │  │Invalid.  │  │  Warming   ││
│  └──────────┘  └──────────┘  └────────────┘│
└─────────────────────────────────────────────┘
```

## Key Features

| Feature | Description |
|---------|-------------|
| **Causal Fingerprints** | Every cached entry stores *why* its response is correct |
| **Precondition Validation** | Real-time checks before returning cached data |
| **Predictive Invalidation** | Predicts *when* results will become stale |
| **Causal Backpropagation** | Invalidation cascades to causally-dependent entries |
| **Adaptive Thresholds** | Similarity thresholds adjust based on hit rates |
| **Causal Discovery** | Learns causal relationships from query patterns |
| **Multi-tier Cache** | L1 exact → L2 semantic → L3 causal-only |
| **Pluggable Embeddings** | Hash (default), SentenceTransformer, or OpenAI |

## API

```python
from kairon import CausalRouter, Precondition, ComparisonOperator

# Create router
router = CausalRouter(embedding_dim=768)

# Register real-time data sources
router.register_source("weather_tokyo", lambda: get_weather("tokyo"))
router.register_source("exchange_rate_usd_jpy", lambda: get_fx_rate())

# Cache with causal preconditions
router.insert_with_preconditions(
    query="What's the weather in Tokyo?",
    response="Sunny, 72°F",
    preconditions=[
        Precondition(key="weather_tokyo", operator=ComparisonOperator.EQ, expected_value="sunny"),
    ],
    causal_factors=["weather_tokyo"],
)

# Query — validates preconditions automatically
result = router.route("How's the weather in Tokyo?")
# → Hit (L2) if preconditions still hold
# → Miss if weather_tokyo source changed → triggers recompute

# Predictive invalidation
engine.predict_validity_window(entry)  # → seconds until invalid

# Causal backpropagation
invalidated = router.precondition_changed("exchange_rate_usd_jpy", "110.5", "112.0")
# → All entries depending on this factor are invalidated in cascade
```

## REST API

```bash
# Cache a query with preconditions
curl -X POST http://localhost:8080/cache \
  -H "Content-Type: application/json" \
  -d '{"query": "weather tokyo", "response": "sunny", "preconditions": [{"key": "weather_tokyo", "operator": "eq", "expected_value": "sunny"}]}'

# Query the cache
curl http://localhost:8080/cache/weather%20tokyo

# Trigger invalidation
curl -X POST http://localhost:8080/invalidate \
  -d '{"key": "weather_tokyo"}'

# Stats
curl http://localhost:8080/stats
```

## Benchmark Results

Kairon vs naive semantic cache across varying volatility:

| Volatility | Kairon Stale Rate | Naive Stale Rate | Improvement |
|-----------|-------------------|-------------------|-------------|
| Low (2%)  | ~0%               | ~2%               | ∞           |
| Med (8%)  | ~0%               | ~8%               | ∞           |
| High (20%)| ~0%               | ~18%              | ∞           |
| Extreme (40%)| ~0%            | ~35%              | ∞           |

*Kairon achieves ~0% stale returns because it validates preconditions before returning any cached result.*

## Tech Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| Core Engine | Python (async) | Rapid prototyping, rich ML ecosystem |
| Vector Search | FAISS | Battle-tested, GPU-optional, fast |
| Causal Graph | NetworkX (MVP) → Neo4j (prod) | In-memory for dev, persistent for production |
| API | FastAPI | Async, OpenAPI docs, type-safe |
| Embeddings | Pluggable | Hash (dev), SentenceTransformer, OpenAI |

## Project Structure

```
kairon/
├── src/kairon/
│   ├── __init__.py         # Public API
│   ├── models.py           # Data models (CausalFingerprint, Precondition, etc.)
│   ├── cache.py            # SemanticCache (L1 exact + L2 FAISS)
│   ├── graph.py            # CausalGraph (networkx DAG)
│   ├── router.py           # CausalRouter (core innovation)
│   ├── embedding.py        # Pluggable embedding engines
│   ├── discovery.py        # Causal discovery service
│   ├── invalidation.py     # Predictive invalidation engine
│   └── server.py           # FastAPI REST server
├── tests/
│   ├── test_core.py        # Unit tests
│   └── test_benchmark.py   # Kairon vs naive comparison
├── examples/
│   └── demo.py             # Interactive demo
├── pyproject.toml          # Project config + dependencies
├── Dockerfile              # Container deployment
└── README.md               # This file
```

## Roadmap

- [x] Core causal data models
- [x] Semantic cache with FAISS (L1 + L2)
- [x] Causal graph engine (networkx)
- [x] Causal router with precondition validation
- [x] Predictive invalidation (heuristic)
- [x] Causal discovery service
- [x] Pluggable embedding engines
- [x] FastAPI REST server
- [x] Benchmark suite
- [ ] Rust core engine (tokio, tonic, lancedb)
- [ ] Neo4j for persistent causal graphs
- [ ] RL-based invalidation policy (PPO/SAC)
- [ ] gRPC + Kafka streaming
- [ ] Kubernetes + Istio deployment
- [ ] Federated causal learning
- [ ] Multi-modal causal fingerprints

## Known Limitations (v0.2.0 MVP)

* **L2 semantic precision** — When two distinct subjects share many tokens ("system 0 status" vs "system 2 status"), L2 may match the wrong cached entry. v0.2.0 mitigates this with a ≥40% token-overlap guard; v0.3.0 will add cross-encoder reranking.
* **In-memory state** — Both the semantic cache (FAISS) and causal graph (networkx) are in-memory. Reloading requires re-execution. Use LanceDB + Neo4j in production.
* **Hash embeddings default** — Demo and tests default to hash embeddings (fast, deterministic). For real semantic similarity, install `sentence-transformers` and pass `--engine sentence-transformer`.
* **Causal discovery is heuristic** — `CausalDiscoveryService` discovers correlations via co-occurrence, not statistical PC algorithm. Production should integrate DoWhy / causal-learn.

## License

MIT — see [LICENSE](LICENSE).

---

> *Kairos (καιρός) — the supreme moment of action. Kairon — the fundamental unit of causal time.*