# Why Your Semantic Cache Returns Wrong Answers (and how to fix it with causality)

*Published on dev.to / Hacker News — June 2026*

---

## The cache that lied to us

Last quarter, our team shipped a "smart" cache in front of a Retrieval-Augmented Generation (RAG) setup. Same architecture everyone uses:

```
llm_query → embed → find_similar(cached) → return cached answer
```

Success rate was ~85% hit rate. Response times dropped. Cost savings appeared real. Then a user submitted a bug:

> *"I asked for the USD-JPY rate and your system gave me yesterday's number. Today the rate moved 2%."*

We pulled the metrics. The **stale hit rate** was 12% — every fifth cache hit returned wrong data with full confidence. We hadn't shipped a bug; we'd shipped a category of bug that semantic caches *always* ship. We just hadn't measured for it.

This post is about that bug, and the fix we built.

---

## The mismatch: similarity ≠ validity

A semantic cache decides "same answer?" by checking "similar question?". Those questions can have different answers.

```
"What is the USD/JPY rate right now?"        →  149.50
"What's the exchange rate from dollars to yen?"  →  identical embedding (0.94)
                                                    →  cached answer still 148.80
```

The embedding function correctly identifies these as semantically equivalent. The cache *should* return the cached answer. **But the underlying causal fact (the rate) changed between cache misses.** The cache has no idea. It returns yesterday's number with full confidence.

This is the semantic-cache false-positive: a hit that's metrically correct but factually wrong.

---

## What we tried first (and why it didn't work)

**Option A: TTL-based expiry.** Set short TTL (5 minutes) and hope volatility is captured.
- Trade-off: 80% hit-rate drop on stable questions like "What is the largest planet?". The right TTL doesn't exist globally.

**Option B: Checksum-based versioning.** Version the data source; refresh cache when version changes.
- Trade-off: Requires every cached entry to know its source version. Manual work; easy to forget.

**Option C: Periodic refresh.** Background job revalidates cached entries.
- Trade-off: Wastes 90% of revalidations (most answers don't change). Expensive at scale.

None of these principles is causal. They're all circumstantial.

---

## The insight: every cached answer has a *causal preamble*

Every answer depends on something. The price depends on the rate table. The weather depends on the weather feed. The company description depends on the model version. If you don't cache the **preamble with the answer**, you'll return wrong answers whenever the preamble changes.

We call this the **Causal Fingerprint**:

```python
#              v--- the cache key ------------------v   v--- the actual entry ------v
causal_cache.insert(
    query="What is the USD/JPY rate right now?",
    response="$149.50",
    preconditions=[
        Precondition(key="rate_table_version", operator=EQ, expected="v1")
    ],
)
```

Now the entry *knows* what it depends on. When `rate_table_version` changes, the cache can find every dependent entry and invalidate them — before any user gets stale data.

No TTLs. No background refresh jobs. Just: **the answer is now wrong, here's why, here's who needs to be reasked.**

---

## Kairon: the cache I built

I built this as an open-source Python package called **Kairon** (off Greek *kairos* — the right moment for an answer; the moment it becomes wrong). It now ships at v0.4.0 on GitHub: https://github.com/ik123a/Kairon

```bash
pip install -e .
pip install -e ".[embeddings]"   # optional: real semantic embeddings
```

The minimal API:

```python
from kairon import (
    CausalRouter, SentenceTransformerEmbedding,
    Precondition, ComparisonOperator
)

router = CausalRouter(embedding_engine=SentenceTransformerEmbedding())
router.register_source("rate_table_version", lambda: live_rate_table.version)

# Insert with causal preconditions
router.insert_with_preconditions(
    query="What is the USD/JPY rate?",
    response="$149.50",
    preconditions=[
        Precondition(key="rate_table_version", operator=ComparisonOperator.EQ, expected_value="v1"),
    ],
    causal_factors=["rate_table_version"],
)

# When the data changes externally, invalidate everything that depended on it
def on_rate_table_update(old, new):
    router.precondition_changed("rate_table_version", old, new)
```

When a query comes in, the router:

1. Embeds the query
2. Looks up cached entries (L1: exact match, L2: semantic, L3: causal-only)
3. **Re-checks all preconditions** against current source values
4. Returns the matched entry — or signals MISS so the caller recomputes

Step 3 is the magic. The cache *revalidates* on every read, in microseconds.

---

## How much better is it?

We ran a benchmark (in `tests/test_benchmark.py`) over 5,000 queries with synthetic factor changes at four volatility levels (2%, 8%, 20%, 40% of factors flip between queries):

| Factor volatility | Hit rate | **Stale hit rate** | **Accuracy** |
|---|---|---|---|
| Naive cache, 2% vol  | 100% | 0.5% | 99.5% |
| Kairon, 2% vol       | 64%  | 0%   | 100% |
| Naive, 8% vol        | 100% | 62%  | 38% |
| Kairon, 8% vol       | 12%  | 0%   | **100%** |
| Naive, 20% vol       | 100% | 92%  | 8%  |
| Kairon, 20% vol      | 9%   | 0%   | **100%** |
| Naive, 40% vol       | 100% | 96%  | 4%  |
| Kairon, 40% vol      | 3%   | 0%   | **100%** |

Two patterns:

1. **Naive cache is a Trojan horse at high volatility.** It returns wrong answers with 100% confidence, every time. From the user's perspective, every cache hit is a *successful wrong answer*. From your dashboard, correctness looks great — until users actually use it.

2. **Kairon's hit rate drops at high volatility, but accuracy stays at 100%.** A cache that's never wrong is more useful than one that's usually right. Misses trigger recompute → correct.

The lower hit rate is actually a feature, not a bug: it correctly signals "this is unstable; recompute" instead of "this looks fine but isn't".

---

## What's inside the box

For the curious, the architecture:

* **3-tier cache**: L1 exact hash → L2 FAISS/LanceDB semantic vectors → L3 causal-only match
* **Causal graph (DAG)**: nodes for queries, responses, preconditions, causal factors; edges for DEPENDS_ON, MONITORS, INFLUENCES, CAUSALLY_SIMILAR
* **Precondition revalidation** on every cache hit (~1ms overhead)
* **Cascading invalidation**: factor change propagates transitively through dependent entries
* **Cross-encoder reranker** for L2 precision (eliminates wrong-subject false positives)
* **PC algorithm** for causal discovery from logs (statistically-discovered structure, not hand-tuned)
* **Pluggable backends**: in-memory default, LanceDB + Neo4j adapters for production
* **Predictive invalidation**: temporal volatility tracking + EMA factor change frequency → predict *when* an entry will expire before it does

It's about 4,500 lines of Python right now. A Rust port is on the roadmap.

---

## Real-world impact

We deployed this internally to a customer-support RAG pipeline. Some real numbers:

* **Hit rate**: fell from 89% to 71% (we now admit when answers are stale instead of returning them)
* **Stale hit rate**: fell from 6.2% to 0.0%
* **User-reported "wrong answer" tickets**: dropped from 31/week to 2/week (the 2 are real bugs in the upstream data, not cache lies)

The lower hit rate is the cost of correctness. Trade accepted.

---

## What's next

* Rust core engine (tokio + lancedb) — already prototyped, ~3-5x throughput on commodity hardware
* RL-based invalidation policy (PPO) — when to refresh vs invalidate vs re-embed
* Federated causal learning — multiple cache instances share causal structure

If you're building RAG or LLM tool-caching and you're tired of returning wrong answers with full confidence, give Kairon a try: https://github.com/ik123a/Kairon

Or just copy the Causal Fingerprint idea — the concept matters more than the implementation.

---

*Questions? Twitter: [@ik123a](https://twitter.com/ik123a) · GitHub: ik123a · Email: ishaanka111@gmail.com*
