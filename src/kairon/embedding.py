"""
Embedding engine — deterministic hash-based embeddings for MVP,
with a pluggable interface for real embedding models.

In production: swap in SentenceTransformer, OpenAI, or Cohere embeddings.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class EmbeddingEngine(ABC):
    """Abstract base for embedding engines."""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Generate an embedding vector for the given text."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimensionality."""
        ...


class HashEmbedding(EmbeddingEngine):
    """
    Deterministic hash-based embedding for MVP / testing.

    NOT semantically meaningful, but:
    - Same text → same vector (deterministic)
    - Different text → different vector (with high probability)
    - Fast, no model download needed
    """

    def __init__(self, dim: int = 768):
        self._dim = dim

    def embed(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode()).digest()
        seed = int.from_bytes(h[:4], "big")
        rng = np.random.RandomState(seed)
        vec = rng.randn(self._dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        return vec

    @property
    def dimension(self) -> int:
        return self._dim


class SentenceTransformerEmbedding(EmbeddingEngine):
    """Real embedding using sentence-transformers (optional dependency)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for real embeddings. "
                "Install with: pip install sentence-transformers"
            )

    def embed(self, text: str) -> np.ndarray:
        vec = self._model.encode(text, convert_to_numpy=True)
        vec = vec.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    @property
    def dimension(self) -> int:
        return self._dim


class OpenAIEmbedding(EmbeddingEngine):
    """OpenAI API-based embedding (optional dependency)."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        dim: int = 1536,
    ):
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key)
            self._model = model
            self._dim = dim
        except ImportError:
            raise ImportError("openai is required: pip install openai")

    def embed(self, text: str) -> np.ndarray:
        response = self._client.embeddings.create(
            input=text,
            model=self._model,
        )
        vec = np.array(response.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    @property
    def dimension(self) -> int:
        return self._dim


# Factory function
def create_embedding_engine(
    engine: str = "hash",
    model_name: Optional[str] = None,
    dim: int = 768,
    api_key: Optional[str] = None,
) -> EmbeddingEngine:
    """
    Create an embedding engine by name.

    Args:
        engine: One of "hash", "sentence-transformer", "openai"
        model_name: Model name (for sentence-transformer or openai)
        dim: Embedding dimension (for hash engine)
        api_key: OpenAI API key (for openai engine)
    """
    if engine == "hash":
        return HashEmbedding(dim=dim)
    elif engine == "sentence-transformer":
        return SentenceTransformerEmbedding(model_name=model_name or "all-MiniLM-L6-v2")
    elif engine == "openai":
        return OpenAIEmbedding(model=model_name or "text-embedding-3-small", api_key=api_key, dim=dim)
    else:
        raise ValueError(f"Unknown embedding engine: {engine}")