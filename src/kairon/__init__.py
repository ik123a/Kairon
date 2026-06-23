"""
Kairon: Causally-Aware Semantic Cache

A semantic cache that understands WHY a cached result is correct—and
uses that understanding to predict WHEN it will become wrong.
"""

from .models import (
    CachedEntry,
    CacheTier,
    CausalFingerprint,
    ComparisonOperator,
    Precondition,
    RouteResult,
)
from .cache import SemanticCache
from .graph import CausalGraph
from .router import CausalRouter
from .embedding import (
    EmbeddingEngine,
    HashEmbedding,
    SentenceTransformerEmbedding,
    OpenAIEmbedding,
    create_embedding_engine,
)
from .invalidation import (
    PredictiveInvalidationEngine,
    InvalidationDecision,
    InvalidationPrediction,
    FactorChange,
)
from .discovery import (
    CausalDiscoveryService,
    CausalObservation,
    DiscoveredCausal,
)
from .reranker import (
    CrossEncoderReranker,
    CrossEncoderRerankerBackend,
    create_reranker,
)
from .storage import (
    VectorBackend,
    GraphBackend,
    InMemoryVectorBackend,
    InMemoryGraphBackend,
    LanceVectorBackend,
    Neo4jGraphBackend,
    create_vector_backend,
    create_graph_backend,
)

__version__ = "0.3.0"
__all__ = [
    "CachedEntry",
    "CacheTier",
    "CausalFingerprint",
    "ComparisonOperator",
    "Precondition",
    "RouteResult",
    "SemanticCache",
    "CausalGraph",
    "CausalRouter",
    "HashEmbedding",
    "SentenceTransformerEmbedding",
    "OpenAIEmbedding",
    "EmbeddingEngine",
    "create_embedding_engine",
    "PredictiveInvalidationEngine",
    "InvalidationDecision",
    "InvalidationPrediction",
    "FactorChange",
    "CausalDiscoveryService",
    "CausalObservation",
    "DiscoveredCausal",
    "CrossEncoderReranker",
    "CrossEncoderRerankerBackend",
    "create_reranker",
    "VectorBackend",
    "GraphBackend",
    "InMemoryVectorBackend",
    "InMemoryGraphBackend",
    "LanceVectorBackend",
    "Neo4jGraphBackend",
    "create_vector_backend",
    "create_graph_backend",
]
