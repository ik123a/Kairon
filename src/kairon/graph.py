"""
Causal graph engine — directed causal dependency graph using networkx.

Stores:
- Causal nodes (queries, responses, causal factors, preconditions)
- Directed edges (DEPENDS_ON, MONITORS, INFLUENCES, CAUSALLY_SIMILAR)

Supports: precondition traversal, dependency lookup (backpropagation), causal similarity.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Optional

import networkx as nx

from .models import CachedEntry, CausalFingerprint, Precondition


class CausalGraph:
    """In-memory causal dependency graph (production: Neo4j-backed)."""

    def __init__(self):
        self.graph = nx.DiGraph()
        # Fast index: factor_name -> set of entry IDs that depend on it
        self._factor_index: dict[str, set[str]] = defaultdict(set)

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def add_entry(self, entry: CachedEntry) -> str:
        """Register a cached entry and its causal dependencies in the graph."""
        query_id = f"query:{entry.id}"
        response_id = f"response:{entry.id}"

        self.graph.add_node(query_id, type="query", text=entry.query_text, entry_id=entry.id)
        self.graph.add_node(response_id, type="response", content=entry.response)
        self.graph.add_edge(query_id, response_id, relation="HAS_RESPONSE")

        # Set causal fingerprint info
        fp = entry.causal_fingerprint
        self.graph.nodes[query_id]["causal_hash"] = fp.causal_hash

        for pc in fp.preconditions:
            pc_id = f"precondition:{pc.key}"
            self.graph.add_node(pc_id, type="precondition", key=pc.key, value=pc.expected_value)
            self.graph.add_edge(query_id, pc_id, relation="DEPENDS_ON")

            # Link to causal factor
            cf_id = f"factor:{pc.source}:{pc.key}"
            self.graph.add_node(cf_id, type="causal_factor", name=pc.key, source=pc.source)
            self.graph.add_edge(pc_id, cf_id, relation="MONITORS")

        # Build the factor index for fast backpropagation lookups
        for factor_name in fp.causal_factors:
            self._factor_index[factor_name].add(entry.id)

        return query_id

    def add_causal_similarity_edge(
        self, entry_a: CachedEntry, entry_b: CachedEntry, score: float, threshold: float = 0.7
    ):
        """Add a CAUSALLY_SIMILAR edge if above threshold."""
        if score >= threshold:
            self.graph.add_edge(
                f"query:{entry_a.id}",
                f"query:{entry_b.id}",
                relation="CAUSALLY_SIMILAR",
                score=score,
            )

    def invalidate_entry(self, entry_id: str) -> None:
        """Remove a cached entry from the causal graph and factor index."""
        query_id = f"query:{entry_id}"
        response_id = f"response:{entry_id}"

        # Remove from factor index — remove this entry from ALL factor lists
        # since it's being fully invalidated and will no longer be queryable
        for factor_entry_ids in self._factor_index.values():
            factor_entry_ids.discard(entry_id)

        # Remove graph nodes
        if query_id in self.graph:
            self.graph.remove_node(query_id)
        if self.graph.has_node(response_id):
            self.graph.remove_node(response_id)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_preconditions(self, entry_id: str) -> list[Precondition]:
        """Return all preconditions for a cached entry by traversing DEPENDS_ON."""
        query_id = f"query:{entry_id}"
        if query_id not in self.graph:
            return []
        preconditions = []
        for _, neighbor, data in self.graph.out_edges(query_id, data=True):
            if data.get("relation") == "DEPENDS_ON":
                node = self.graph.nodes[neighbor]
                preconditions.append(
                    Precondition(
                        key=node["key"],
                        operator=None,  # type: ignore
                        expected_value=node["value"],
                    )
                )
        return preconditions

    def find_dependents(self, factor_name: str) -> list[str]:
        """
        Find all cached entry IDs whose causal fingerprint includes a given factor.
        Uses the fast factor index for O(1) lookup.

        Used for causal backpropagation invalidation.
        """
        return list(self._factor_index.get(factor_name, set()))

    def find_graph_dependents(self, factor_name: str) -> list[str]:
        """
        Graph-walk version: find all entries dependent on a factor
        by traversing the graph (MONITORS ← DEPENDS_ON).
        Used as fallback or for graph-level queries.
        """
        cf_id = None
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "causal_factor" and data.get("name") == factor_name:
                cf_id = node
                break
        if cf_id is None:
            return []

        entry_ids: set[str] = set()
        for predecessor in self.graph.predecessors(cf_id):
            for query_node in self.graph.predecessors(predecessor):
                node = self.graph.nodes[query_node]
                if node.get("type") == "query" and "entry_id" in node:
                    entry_ids.add(node["entry_id"])
        return list(entry_ids)

    def find_direct_dependents(self, entry_id: str) -> list[str]:
        """Find entries that are CAUSALLY_SIMILAR to this entry (for cascade invalidation)."""
        query_id = f"query:{entry_id}"
        if query_id not in self.graph:
            return []
        dependents: list[str] = []
        for _, neighbor, data in self.graph.out_edges(query_id, data=True):
            if data.get("relation") == "CAUSALLY_SIMILAR":
                node = self.graph.nodes[neighbor]
                if "entry_id" in node:
                    dependents.append(node["entry_id"])
        # Also check reverse direction
        for predecessor, _, data in self.graph.in_edges(query_id, data=True):
            if data.get("relation") == "CAUSALLY_SIMILAR":
                node = self.graph.nodes[predecessor]
                if "entry_id" in node:
                    dependents.append(node["entry_id"])
        return dependents

    def query_causal_similarity(
        self, new_entry: CachedEntry, candidate_entry: CachedEntry
    ) -> float:
        """
        Compute a causal similarity score between two entries by comparing their
        causal fingerprints: precondition overlap + graph structural similarity.
        Returns a score in [0, 1].
        """
        new_pcs = {p.key for p in new_entry.preconditions}
        cand_pcs = {p.key for p in candidate_entry.preconditions}

        if not new_pcs and not cand_pcs:
            return 0.5

        intersect = new_pcs & cand_pcs
        union = new_pcs | cand_pcs
        if not union:
            return 0.0

        jaccard = len(intersect) / len(union)

        new_factors = set(new_entry.causal_fingerprint.causal_factors)
        cand_factors = set(candidate_entry.causal_fingerprint.causal_factors)
        factor_overlap = len(new_factors & cand_factors) / max(1, len(new_factors | cand_factors))

        return 0.5 * jaccard + 0.3 * factor_overlap + 0.2 * (1.0 if intersect else 0.0)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "type_counts": dict(self._count_by_type()),
            "factor_index_size": {k: len(v) for k, v in self._factor_index.items()},
        }

    def _count_by_type(self) -> defaultdict[str, int]:
        counts: defaultdict[str, int] = defaultdict(int)
        for _, data in self.graph.nodes(data=True):
            counts[data.get("type", "unknown")] += 1
        return counts

    def to_dict(self) -> dict:
        """Export the graph for serialization."""
        return {
            "nodes": [(n, d) for n, d in self.graph.nodes(data=True)],
            "edges": [(u, v, d) for u, v, d in self.graph.edges(data=True)],
        }