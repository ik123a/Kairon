"""
Pluggable backend adapters for persistence.

Kairon's MVP uses in-memory storage (FAISS for vectors, networkx for graphs).
For production, swap these in:

  - LanceVectorBackend  →  LanceDB (columnar, embedded, disk-persistent)
  - Neo4jGraphBackend    →  Neo4j (production graph DB)
  - InMemoryVectorBackend  →  FAISS-equivalent (default; used in tests)
  - InMemoryGraphBackend   →  networkx-equivalent (default; used in tests)

Adapter pattern makes it drop-in:

    from kairon.storage import create_vector_backend, create_graph_backend
    cache_vector = create_vector_backend("lancedb", path="/var/kairon/lance")
    cache_graph  = create_graph_backend("neo4j", uri="bolt://localhost:7687")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .models import CachedEntry


# =====================================================================
# Vector backend (L2 semantic cache)
# =====================================================================

class VectorBackend(ABC):
    """Pluggable vector index for L2 semantic cache."""

    @abstractmethod
    def add(self, embedding: np.ndarray, entry: CachedEntry) -> int:
        """Add a vector + metadata pair. Returns stable backend ID."""
        ...

    @abstractmethod
    def search(self, embedding: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
        """Return list of (backend_id, similarity_score) sorted by score desc."""
        ...

    @abstractmethod
    def remove(self, backend_id: int) -> bool:
        """Remove a vector by backend ID."""
        ...

    @abstractmethod
    def ntotal(self) -> int:
        """Total vectors stored."""
        ...

    @abstractmethod
    def warmup(self) -> None:
        """Eager-load model files / open connections."""
        ...


class InMemoryVectorBackend(VectorBackend):
    """FAISS-equivalent in-memory backend. Default for tests/MVP."""

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        import faiss
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_dim))
        self._entries: dict[int, CachedEntry] = {}
        self._next_id = 0
        self.warmup()

    def warmup(self) -> None:
        pass

    def add(self, embedding: np.ndarray, entry: CachedEntry) -> int:
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        embedding = embedding.astype(np.float32)
        import faiss
        faiss.normalize_L2(embedding)
        idx = self._next_id
        self._next_id += 1
        self._index.add_with_ids(embedding, np.array([idx], dtype=np.int64))
        self._entries[idx] = entry
        return idx

    def search(self, embedding: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        embedding = embedding.astype(np.float32)
        import faiss
        faiss.normalize_L2(embedding)
        if self._index.ntotal == 0:
            return []
        scores, ids = self._index.search(embedding, k=k)
        result = []
        for s, i in zip(scores[0], ids[0]):
            if int(i) == -1:
                continue
            result.append((int(i), float(s)))
        return result

    def remove(self, backend_id: int) -> bool:
        if backend_id not in self._entries:
            return False
        self._index.remove_ids(np.array([backend_id], dtype=np.int64))
        del self._entries[backend_id]
        return True

    def ntotal(self) -> int:
        return self._index.ntotal


class LanceVectorBackend(VectorBackend):
    """
    LanceDB-backed vector store (columnar, embedded, disk-persistent).

    Requires `pip install lancedb`. Provides persistence across process restarts.
    Vector index is built on Apache Arrow + IVF-PQ.

    Usage:
        backend = LanceVectorBackend(uri="/var/kairon/lance_cache",
                                     embedding_dim=768)
        backend.warmup()
        backend.add(embedding, entry)
        backend.search(query_embedding, k=5)
    """

    def __init__(self, uri: str = "./kairon_lance", embedding_dim: int = 768):
        self.uri = uri
        self.embedding_dim = embedding_dim
        self._db = None
        self._table = None
        self._init_failed: bool = False

    def warmup(self) -> None:
        import os
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        try:
            import lancedb
            self._db = lancedb.connect(self.uri)
            # Create or open the table
            try:
                schema = self._infer_schema()
                # Empty table on first init
                self._table = self._db.create_table("kairon", schema=schema, mode="overwrite")
            except Exception:
                self._table = self._db.open_table("kairon")
        except ImportError:
            self._init_failed = True

    def _infer_schema(self):
        try:
            import pyarrow as pa
            return pa.schema(
                [
                    pa.field("vector", pa.list_(pa.float32(), self.embedding_dim)),
                    pa.field("entry_id", pa.string()),
                    pa.field("query_text", pa.string()),
                    pa.field("response", pa.string()),
                ]
            )
        except ImportError:
            return None

    def add(self, embedding: np.ndarray, entry: CachedEntry) -> int:
        if self._init_failed or self._table is None:
            raise RuntimeError("LanceDB not available — install with `pip install lancedb`")
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        embedding = embedding.astype(np.float32)
        # Normalize
        norm = np.linalg.norm(embedding[0])
        if norm > 0:
            embedding[0] = embedding[0] / norm
        row = {
            "vector": embedding[0].tolist(),
            "entry_id": entry.id,
            "query_text": entry.query_text,
            "response": entry.response,
        }
        self._table.add([row])
        # LanceDB auto-increments IDs; estimate from length
        return self.ntotal() - 1

    def search(self, embedding: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
        if self._init_failed or self._table is None:
            raise RuntimeError("LanceDB not available")
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        embedding = embedding.astype(np.float32)
        norm = np.linalg.norm(embedding[0])
        if norm > 0:
            embedding[0] = embedding[0] / norm
        results = self._table.search(embedding[0].tolist()).limit(k).to_list()
        return [(i, float(r.get("_distance", 0.0))) for i, r in enumerate(results)]

    def remove(self, backend_id: int) -> bool:
        if self._init_failed or self._table is None:
            raise RuntimeError("LanceDB not available")
        # LanceDB deletion requires predicate
        try:
            self._table.delete(f"row_index = {backend_id}")
            return True
        except Exception:
            return False

    def ntotal(self) -> int:
        if self._init_failed or self._table is None:
            return 0
        try:
            return self._table.count_rows()
        except Exception:
            return 0


# =====================================================================
# Causal graph backend
# =====================================================================

class GraphBackend(ABC):
    """Pluggable graph store for the causal dependency DAG."""

    @abstractmethod
    def add_edge(self, src: str, dst: str, relation: str = "DEPENDS_ON", **attrs) -> None:
        ...

    @abstractmethod
    def add_node(self, node_id: str, **attrs) -> None:
        ...

    @abstractmethod
    def remove_node(self, node_id: str) -> None:
        ...

    @abstractmethod
    def dependents_of(self, key: str) -> list[str]:
        """All entry IDs that depend on (key)."""
        ...

    @abstractmethod
    def nodes(self) -> list[str]:
        ...

    @abstractmethod
    def edges(self) -> list[tuple[str, str, str]]:
        """Returns (src, dst, relation) tuples."""
        ...

    @abstractmethod
    def warmup(self) -> None:
        ...


class InMemoryGraphBackend(GraphBackend):
    """networkx-equivalent in-memory backend."""

    def __init__(self):
        import networkx as nx
        self._g = nx.DiGraph()
        self.warmup()

    def warmup(self) -> None:
        pass

    def add_edge(self, src: str, dst: str, relation: str = "DEPENDS_ON", **attrs) -> None:
        self._g.add_edge(src, dst, relation=relation, **attrs)

    def add_node(self, node_id: str, **attrs) -> None:
        self._g.add_node(node_id, **attrs)

    def remove_node(self, node_id: str) -> None:
        if self._g.has_node(node_id):
            self._g.remove_node(node_id)

    def dependents_of(self, key: str) -> list[str]:
        # Find all entry:N nodes with a path from key
        results = []
        for node, data in self._g.nodes(data=True):
            if data.get("type") == "query" and self._g.has_node(f"factor:{key}"):
                if self._g.has_edge(node, f"factor:{key}") or self._g.has_edge(
                    node, f"precondition:{key}"
                ):
                    entry_id = data.get("entry_id", node.replace("query:", ""))
                    results.append(entry_id)
        return results

    def nodes(self) -> list[str]:
        return list(self._g.nodes())

    def edges(self) -> list[tuple[str, str, str]]:
        result = []
        for u, v, data in self._g.edges(data=True):
            rel = data.get("relation", "?")
            result.append((u, v, rel))
        return result


class Neo4jGraphBackend(GraphBackend):
    """
    Neo4j-backed graph store for production.

    Requires: `pip install neo4j` and a running Neo4j instance.
    Usage:
        backend = Neo4jGraphBackend(uri="bolt://localhost:7687", user="neo4j", password="...")
        backend.warmup()
    """

    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "neo4j"):
        self.uri = uri
        self.user = user
        self.password = password
        self._driver = None
        self._init_failed: bool = False

    def warmup(self) -> None:
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Verify connectivity
            with self._driver.session() as session:
                session.run("RETURN 1").single()
        except ImportError:
            self._init_failed = True
        except Exception:
            self._init_failed = True

    def _check(self):
        if self._init_failed:
            raise RuntimeError("Neo4j not available — pip install neo4j + check URI/credentials")

    def add_edge(self, src: str, dst: str, relation: str = "DEPENDS_ON", **attrs) -> None:
        self._check()
        attrs_str = "{".join(f"{k}: ${k}" for k in attrs) + "}" if attrs else "{}"
        with self._driver.session() as s:
            s.run(
                f"MERGE (a:Node {{id: $src}}) "
                f"MERGE (b:Node {{id: $dst}}) "
                f"MERGE (a)-[r:{relation} {attrs_str}]->(b)",
                src=src, dst=dst, **attrs,
            )

    def add_node(self, node_id: str, **attrs) -> None:
        self._check()
        with self._driver.session() as s:
            s.run("MERGE (n:Node {id: $id}) SET n += $attrs", id=node_id, attrs=attrs)

    def remove_node(self, node_id: str) -> None:
        self._check()
        with self._driver.session() as s:
            s.run("MATCH (n:Node {id: $id}) DETACH DELETE n", id=node_id)

    def dependents_of(self, key: str) -> list[str]:
        self._check()
        with self._driver.session() as s:
            result = s.run(
                "MATCH (n:Node)-[*1..3]-(f:Node {id: $key}) "
                "WHERE n.type = 'query' "
                "RETURN n.entry_id AS eid",
                key=f"factor:{key}",
            )
            return [r["eid"] for r in result if r["eid"]]

    def nodes(self) -> list[str]:
        self._check()
        with self._driver.session() as s:
            return [r["id"] for r in s.run("MATCH (n:Node) RETURN n.id AS id")]

    def edges(self) -> list[tuple[str, str, str]]:
        self._check()
        with self._driver.session() as s:
            result = s.run("MATCH (a:Node)-[r]->(b:Node) RETURN a.id, b.id, type(r) AS rel")
            return [(r[0], r[1], r["rel"]) for r in result]


# =====================================================================
# Factories
# =====================================================================

def create_vector_backend(kind: str, **kwargs) -> VectorBackend:
    if kind == "memory":
        return InMemoryVectorBackend(**kwargs)
    if kind == "lancedb":
        return LanceVectorBackend(**kwargs)
    raise ValueError(
        f"Unknown vector backend: {kind!r}. "
        "Options: 'memory' (default), 'lancedb'."
    )


def create_graph_backend(kind: str, **kwargs) -> GraphBackend:
    if kind == "memory":
        return InMemoryGraphBackend(**kwargs)
    if kind == "neo4j":
        return Neo4jGraphBackend(**kwargs)
    raise ValueError(
        f"Unknown graph backend: {kind!r}. "
        "Options: 'memory' (default), 'neo4j'."
    )
