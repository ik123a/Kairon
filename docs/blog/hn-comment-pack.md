# HN title (under 80 chars):
Your semantic cache has a 12% stale-hit rate. Here's why.

# HN link text (URL: https://github.com/ik123a/Kairon)

---

# Comment 1 (post as the author, immediate):

We were getting 89% hit rate on a semantic cache in front of a RAG pipeline.
Looked great on the dashboard.

Then we measured stale-hit rate separately. It was 12%.

That means ~1 in 9 cached answers was *wrong with full confidence*. The semantic cache had no idea the underlying data changed; it just returned yesterday's answer for today's query.

I built Kairon (https://github.com/ik123a/Kairon) as an open-source fix. The trick is the "Causal Fingerprint": cache (query, preconditions, response) instead of (query → response), and revalidate every precond on every read.

Benchmarks (vs naive cache):

| Volatility | Naive accuracy | Kairon accuracy |
|------------|---------------|-----------------|
| 2% factor flips  | 99.5% | 100%   |
| 8% factor flips  | 38%   | 100%   |
| 20% factor flips | 8%    | 100%   |
| 40% factor flips | 4%    | 100%   |

Kairon trades hit rate for accuracy. At 40% volatility, naive is 100% hit rate but 4% correct; Kairon is 3% hit rate and 100% correct.

Yes the hit rate drops. Yes that's a feature.

5,000 LOC Python, ~4k lines, MIT. A Rust core is on the roadmap.

Happy to answer questions on the design choices.

---

# Comment 2 (in response to "Why not just TTL?"):

We tried TTL first. Two problems:
1. Stable facts ("largest planet") get purged anyway → wasted backend calls
2. Volatile facts ("USD/JPY rate") exceed TTL → still serve stale in the gap

TTL is a probabilistic bet. Causal fingerprinting is deterministic — we know *exactly* what each entry depends on, and invalidate the moment that thing changes.

---

# Comment 3 (in response to "Why not use cron + versioned data?"):

That's actually the second approach we tried. Pros: simple. Cons:
- Background job has to enumerate every cached entry on every change → O(N) per data update
- Doesn't catch entries the job missed (operator forgot to register the source)
- No "predictive" mode — you can only invalidate AFTER the change, never BEFORE

The causal graph approach is incremental: when a factor changes, the graph already knows who depends on it. O(dependents) instead of O(N).

Plus we discovered we could predict WHEN entries would become invalid (temporal volatility tracking) and proactively refresh them.

---

# Comment 4 (in response to "How does it compare to GPTCache / Redis caching?"):

GPTCache is similarity matching only — same problem.
Redis with key-versioning gets you halfway but doesn't propagate invalidation.
LangChain's caching layer is opaque — hard to debug stale returns.

The reason Kairon exists is because none of the existing systems exposed the *causal structure* of cached answers as a first-class object. Once you can see "this entry depends on X, Y, Z", invalidation becomes a graph traversal instead of a heuristic.

---

# Comment 5 (closing, if asked about RL/PPO future):

Rodney, yeah — the v0.5.0 roadmap includes an RL agent that learns WHEN to invalidate vs WHEN to refresh vs WHEN to wait. State = factor volatility patterns. Action = invalidation timing. Reward = +1 for each non-stale hit, -10 for each stale hit, +0.01 for each saved backend call.

Currently it's a heuristic; the RL version is in a Rust crate (private repo for now).

The reason we shipped heuristic-first is so the *infrastructure* for invalidation is right before we let an agent make policy decisions. Garbage in → garbage out, and the garbage here is "did the precondition correctly flag stale returns?" — which the heuristic gets right 100% in benchmarks.
