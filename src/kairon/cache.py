"""
Semantic vector cache using FAISS for similarity-based cache hits.

Three-tier architecture:
    L1: Exact query-hash → dict lookup (sub-ms)
    L2: Semantic vector → FAISS index cosine search (<5ms)
    L3: Causal-only fallback (handled by router, not this module)
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Optional

import faiss
import numpy as np

from .models import CachedEntry, CacheTier
from .reranker import CrossEncoderRerankerBackend, create_reranker


def _token_overlap_ok(query: str, cached: str, threshold: float = 0.40) -> bool:
    """Lightweight token-set Jaccard guard — used to reject L2 semantic hits
    that match the wrong cached entry (e.g., paraphrased 'system 0 health'
    matching the cached 'system 2 status' entry).

    Threshold of 0.40 means: at least 40% of query tokens must appear in the cached query.
    This filters out cross-system collisions while accepting rephrasings of the same query.
    """
    q_tokens = {t for t in query.lower().split() if len(t) > 1}
    c_tokens = {t for t in cached.lower().split() if len(t) > 1}
    if not q_tokens or not c_tokens:
        return True  # Empty token set — can't judge, allow through
    # Jaccard with recall bias: focus on what fraction of QUERY tokens are in cached
    overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens))
    return overlap >= threshold


class SemanticCache:
    """
    Two-tier semantic cache: L1 exact hash lookup + L2 FAISS vector similarity.

    In production: L1 = Redis, L2 = LanceDB. MVP uses in-memory dict + FAISS.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        l1_max_size: int = 1000,
        similarity_threshold: float = 0.80,
        adaptive_threshold: bool = True,
        reranker: CrossEncoderRerankerBackend | None = None,
        reranking_threshold: float = 0.25,
    ):
        self.embedding_dim = embedding_dim
        self.l1_max_size = l1_max_size
        self.similarity_threshold = similarity_threshold
        self.adaptive_threshold = adaptive_threshold
        # Optional cross-encoder reranker for L2 precision (v0.3.0)
        self.reranker = reranker if reranker is not None else create_reranker("stub")
        self.reranking_threshold = reranking_threshold

        # L1: exact hash → entry (OrderedDict = LRU eviction)
        self._l1: OrderedDict[str, CachedEntry] = OrderedDict()

        # L2: FAISS index with ID mapping
        self._l2_index = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_dim))
        self._l2_entries: dict[int, CachedEntry] = {}
        self._next_id: int = 0

        # Metrics (EMA-smoothed)
        self._hit_count: int = 0
        self._miss_count: int = 0

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert(self, entry: CachedEntry, embedding: np.ndarray | None = None):
        """Insert a cached entry into both L1 and L2 (skipping duplicates)."""
        # If no embedding provided, use the entry's stored embedding
        if embedding is None and entry.query_embedding:
            embedding = np.array(entry.query_embedding, dtype=np.float32)
        if embedding is None:
            embedding = np.random.randn(self.embedding_dim).astype(np.float32)  # Fallback dummy

        # Ensure float32, 2D
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        embedding = embedding.astype(np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(embedding)

        # Store embedding on entry
        entry.query_embedding = embedding.flatten().tolist()

        # L1: exact hash key — always promote/update
        key = self._exact_key(entry.query_text)
        self._l1[key] = entry
        if len(self._l1) > self.l1_max_size:
            self._l1.popitem(last=False)  # LRU eviction

        # L2: FAISS vector index — check for existing entry to avoid duplicates
        existing_idx = None
        for existing_idx_candidate, existing_entry in self._l2_entries.items():
            if existing_entry.id == entry.id:
                existing_idx = existing_idx_candidate
                break
        if existing_idx is not None:
            # Already in L2 — only refresh the in-memory entry object (FAISS vector stays)
            self._l2_entries[existing_idx] = entry
            return

        idx = self._next_id
        self._next_id += 1
        faiss_id = np.array([idx], dtype=np.int64)
        self._l2_index.add_with_ids(embedding, faiss_id)
        self._l2_entries[idx] = entry

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, query_text: str, embedding: np.ndarray) -> tuple[Optional[CachedEntry], CacheTier]:
        """
        Two-tier lookup: L1 exact → L2 semantic.
        Returns (entry, tier) or (None, MISS).
        """
        # L1: exact hash match
        key = self._exact_key(query_text)
        if key in self._l1:
            self._hit_count += 1
            entry = self._l1[key]
            entry.record_access()
            # Move to end (LRU)
            self._l1.move_to_end(key)
            return entry, CacheTier.L1

        # L2: semantic similarity search
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        embedding = embedding.astype(np.float32)
        faiss.normalize_L2(embedding)

        if self._l2_index.ntotal > 0:
            scores, ids = self._l2_index.search(embedding, k=5)  # top-5 from FAISS
            candidates: list[tuple[CachedEntry, float]] = []
            for i in range(len(ids[0])):
                faiss_id = int(ids[0][i])
                score = float(scores[0][i])
                if faiss_id == -1:
                    continue
                if score >= self.similarity_threshold:
                    entry = self._l2_entries[faiss_id]
                    candidates.append((entry, score))

            if candidates:
                # Rerank with cross-encoder (or stub) — eliminates cross-subject false positives
                cand_texts = [c.query_text for c, _ in candidates]
                rerank_results = self.reranker.score_pairs(query_text, cand_texts)
                # Sort by reranker score descending
                rerank_results.sort(key=lambda r: r.score, reverse=True)
                # Pick the first one that passes the reranking threshold
                for r in rerank_results:
                    if r.score >= self.reranking_threshold:
                        entry, score = candidates[r.index]
                        entry.record_access()
                        # Promote to L1
                        self._l1[self._exact_key(query_text)] = entry
                        self._hit_count += 1
                        return entry, CacheTier.L2

        self._miss_count += 1
        return None, CacheTier.MISS

    def search_top_k(self, embedding: np.ndarray, k: int = 5) -> list[tuple[CachedEntry, float]]:
        """Return top-k semantically similar entries with scores (for causal scoring)."""
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        embedding = embedding.astype(np.float32)
        faiss.normalize_L2(embedding)

        if self._l2_index.ntotal == 0:
            return []

        scores, ids = self._l2_index.search(embedding, k=k)
        results: list[tuple[CachedEntry, float]] = []
        for i in range(len(ids[0])):
            faiss_id = int(ids[0][i])
            score = float(scores[0][i])
            if faiss_id in self._l2_entries:
                results.append((self._l2_entries[faiss_id], score))
        return results

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def invalidate(self, entry_id: str) -> bool:
        """Remove a cached entry by its CachedEntry.id."""
        # Remove from L1
        keys_to_remove = [k for k, v in self._l1.items() if v.id == entry_id]
        for k in keys_to_remove:
            del self._l1[k]

        # Remove from L2
        idx_to_remove = [i for i, e in self._l2_entries.items() if e.id == entry_id]
        if idx_to_remove:
            for idx in idx_to_remove:
                self._l2_index.remove_ids(np.array([idx], dtype=np.int64))
                del self._l2_entries[idx]
            return True
        return False

    def invalidate_many(self, entry_ids: list[str]) -> list[str]:
        """Batch invalidation — returns list of entry IDs that were actually invalidated."""
        invalidated = []
        for eid in entry_ids:
            if self.invalidate(eid):
                invalidated.append(eid)
        return invalidated

    # ------------------------------------------------------------------
    # Adaptive threshold
    # ------------------------------------------------------------------

    def adapt_threshold(self):
        """Adjust similarity threshold based on hit rate (EMA)."""
        if not self.adaptive_threshold:
            return
        total = self._hit_count + self._miss_count
        if total < 100:
            return
        hit_rate = self._hit_count / total
        alpha = 0.05
        # If hit rate is low, relax threshold. If high, tighten.
        target = 0.5
        adjustment = (target - hit_rate) * alpha
        self.similarity_threshold = max(0.6, min(0.95, self.similarity_threshold + adjustment))

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def hit_rate(self) -> float:
        total = self._hit_count + self._miss_count
        return self._hit_count / total if total > 0 else 0.0

    @property
    def size(self) -> tuple[int, int]:
        return len(self._l1), self._l2_index.ntotal

    def stats(self) -> dict:
        l1_size, l2_size = self.size
        return {
            "l1_size": l1_size,
            "l2_size": l2_size,
            "hit_rate": f"{self.hit_rate:.2%}",
            "threshold": f"{self.similarity_threshold:.3f}",
            "hits": self._hit_count,
            "misses": self._miss_count,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _exact_key(query: str) -> str:
        return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]