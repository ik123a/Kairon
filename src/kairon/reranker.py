"""
Cross-encoder reranker for L2 semantic cache hits.

A cross-encoder takes (query, candidate) AS A PAIR and outputs a single
relevance score. Way more accurate than bi-encoder cosine similarity
(which only sees each text in isolation), at the cost of being slower
(~50ms per pair, must be computed per query).

In Kairon's L2 cache:
- Bi-encoder (FAISS) finds top-k candidates fast (~1ms)
- Cross-encoder reranks those candidates precisely (~50ms total)

This eliminates the cross-subject false-positive problem:
bi-encoders think 'system 0 status' and 'system 2 status' are similar;
a cross-encoder correctly identifies in-topic vs out-of-topic based on
the actual sentence-pair semantics.

Backends (whicheve is installed):
  - sentence_transformers.CrossEncoder (default, pulls ~80MB model)
  - "stub" backend for testing (returns string-overlap heuristic)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

# Lazy imports
_CROSSENCODER_CLASS = None


@dataclass
class RerankResult:
    """Score for a single (query, candidate) pair."""
    index: int          # Original index in the candidate list
    score: float        # 0.0 – 1.0 relevance
    text: str           # Candidate text (for debugging)


class CrossEncoderRerankerBackend(ABC):
    """Abstract base — implement score_pairs()."""

    @abstractmethod
    def score_pairs(self, query: str, candidates: list[str]) -> list[RerankResult]:
        """Score each candidate against the query. Higher = more relevant."""
        ...

    @abstractmethod
    def warmup(self) -> None:
        """Pre-load model weights. Called once at init."""
        ...


class _StubReranker(CrossEncoderRerankerBackend):
    """
    Deterministic stub — used when sentence_transformers isn't installed.
    Score includes:
      - token_overlap_jaccard (40% weight)
      - bonus for matching entity tokens (numbers, capital letters): 30%
      - exact-prefix match bonus: 30%
    """

    def warmup(self) -> None:
        pass

    def score_pairs(self, query: str, candidates: list[str]) -> list[RerankResult]:
        import re
        q_tokens = set(t for t in re.findall(r"\w+", query.lower()) if len(t) > 1)
        # Entity tokens: numbers, capitalized words (likely subjects)
        q_entities = set(re.findall(r"\d+", query.lower())) | set(
            w.lower() for w in re.findall(r"\b[A-Z][a-z]+\b|[A-Z]+|\d+", query) if len(w) > 1
        )

        results = []
        for i, c in enumerate(candidates):
            c_tokens = set(t for t in re.findall(r"\w+", c.lower()) if len(t) > 1)
            c_entities = set(re.findall(r"\d+", c.lower())) | set(
                w.lower() for w in re.findall(r"\b[A-Z][a-z]+\b|[A-Z]+|\d+", c) if len(w) > 1
            )
            # Jaccard
            if not q_tokens or not c_tokens:
                jaccard = 0.0
            else:
                jaccard = len(q_tokens & c_tokens) / len(q_tokens | c_tokens)
            # Entity overlap
            if q_entities:
                entity_overlap = len(q_entities & c_entities) / max(1, len(q_entities))
            else:
                entity_overlap = 0.0
            # Prefix match (40+ char prefix identical?)
            prefix_match = 1.0 if query[:40].lower().strip() == c[:40].lower().strip() else 0.0
            # Combined score
            score = 0.4 * jaccard + 0.3 * entity_overlap + 0.3 * prefix_match
            results.append(RerankResult(index=i, score=score, text=c))
        return results


class CrossEncoderReranker(CrossEncoderRerankerBackend):
    """
    Cross-encoder reranker using sentence-transformers.CrossEncoder.
    Model: 'cross-encoder/ms-marco-MiniLM-L-6-v2' (~80MB).
    Trained on MS MARCO passage relevance — exactly the use case here
    (is this passage/document relevant to this query).

    Lazy loading: the model is NOT loaded at construction. It is loaded
    on first call to `score_pairs()` (or explicit `ensure_loaded()`).
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None

    def warmup(self) -> None:
        """Deprecated — use `ensure_loaded()` for explicit, or let first `score_pairs()` call load it."""
        self.ensure_loaded()

    def ensure_loaded(self) -> None:
        """Eager-load the cross-encoder model. Safe to call multiple times."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            self._model = CrossEncoder(self.model_name)
        except ImportError:
            self._model = None

    def score_pairs(self, query: str, candidates: list[str]) -> list[RerankResult]:
        # Lazy load on first use
        self.ensure_loaded()
        if self._model is None:
            return _StubReranker().score_pairs(query, candidates)
        try:
            pairs = [(query, c) for c in candidates]
            scores = self._model.predict(pairs)
            # Sigmoid-normalize to [0, 1] (cross-encoder scores are logit-like)
            import math
            norm = [1.0 / (1.0 + math.exp(-float(s))) for s in scores]
            return [RerankResult(index=i, score=norm[i], text=candidates[i]) for i in range(len(candidates))]
        except Exception:
            return _StubReranker().score_pairs(query, candidates)


def create_reranker(prefer: str = "cross-encoder") -> CrossEncoderRerankerBackend:
    """
    Factory: returns a CrossEncoderReranker if sentence-transformers is
    available, otherwise returns the deterministic stub reranker.

    Args:
        prefer: "cross-encoder" (default — production) or "stub" (fast testing)
    """
    if prefer == "stub":
        return _StubReranker()
    try:
        import sentence_transformers  # noqa: F401
        return CrossEncoderReranker()
    except ImportError:
        return _StubReranker()
