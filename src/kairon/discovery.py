"""
Causal Discovery Service — discovers causal relationships from query logs.

Implements the **PC algorithm** (constraint-based causal discovery):

    1. **Start** with a complete graph over all observed variables
       (factors + queries + responses).
    2. **Test conditional independence**: for each pair (X, Y),
       check whether they're independent given various subsets of Z.
       If X ⊥ Y | Z, remove the edge X — Y from the graph.
    3. **Orient edges**: use v-structure patterns (X — Z — Y with no
       edge X — Y, and X,Y not in Z) to identify colliders and orient
       remaining edges accordingly.

MVP simplification: we implement the *independence test* using
**partial correlation** (Fisher's z-test), then derive a causal
adjacency list. The full orientation step requires DoWhy/gCastle
when edge counts get large (>20 variables).

Reference:
    Spirtes, Glymour, Scheines. "Causation, Prediction, and Search" (2000).
    DoWhy (Microsoft). https://github.com/py-why/dowhy
    gCastle (Huawei). https://github.com/huawei-noah/trustworthyAI
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from typing import Optional

import numpy as np
from scipy import stats

from .models import (
    CachedEntry,
    CausalFingerprint,
    ComparisonOperator,
    Precondition,
)

logger = logging.getLogger(__name__)


@dataclass
class CausalObservation:
    """A single observation of a query→response→precondition event."""

    query: str
    response: str
    precondition_key: str
    precondition_value_at_time: object
    was_cache_hit: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DiscoveredCausal:
    """A discovered causal relationship."""

    cause_factor: str  # e.g., "exchange_rate_usd_jpy"
    effect_query_pattern: str  # e.g., "exchange rate"
    strength: float  # 0-1, how strong the causal link
    observations: int  # how many observations support this
    # PC algorithm output -----------------------------------------------------
    partial_correlation: Optional[float] = None  # Pearson partial correlation
    p_value: Optional[float] = None  # Fisher-z p-value for conditional indep test
    conditioning_set: tuple = ()  # variables controlled for in the partial corr
    is_conditional_independent: bool = False
    # -------------------------------------------------------------------------
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class CausalDiscoveryService:
    """
    Discovers causal relationships from access log patterns.

    Methods:
    1. Correlation detection: when a precondition changes, track which
       queries subsequently get cache misses
    2. Temporal ordering: cause must precede effect
    3. Conditional independence: rule out common causes

    MVP uses simple correlation + temporal ordering. Production systems
    can upgrade to full PC algorithm via gCastle or causal-learn.
    """

    def __init__(self):
        self._observations: list[CausalObservation] = []
        self._discovered: dict[str, DiscoveredCausal] = {}  # factor -> DiscoveredCausal
        self._factor_miss_counts: dict[str, int] = defaultdict(int)
        self._factor_hit_counts: dict[str, int] = defaultdict(int)
        # PC algorithm graph (undirected skeleton)
        self._pc_graph = SimpleUndirectedGraph()
        self._pc_result: dict = {}

    def observe(
        self,
        query: str,
        response: str,
        precondition_key: str,
        precondition_value: object,
        was_cache_hit: bool,
    ):
        """Record an observation for causal discovery."""
        obs = CausalObservation(
            query=query,
            response=response,
            precondition_key=precondition_key,
            precondition_value_at_time=precondition_value,
            was_cache_hit=was_cache_hit,
        )
        self._observations.append(obs)

        if was_cache_hit:
            self._factor_hit_counts[precondition_key] += 1
        else:
            self._factor_miss_counts[precondition_key] += 1

    def discover(self) -> list[DiscoveredCausal]:
        """
        Run causal discovery on accumulated observations.

        Simple heuristic: a factor is causal if cache miss rate
        is significantly higher after its value changes.
        Returns list of discovered causal relationships sorted by strength.
        """
        results: list[DiscoveredCausal] = []

        for factor_key in set(list(self._factor_miss_counts.keys()) + list(self._factor_hit_counts.keys())):
            misses = self._factor_miss_counts[factor_key]
            hits = self._factor_hit_counts[factor_key]
            total = misses + hits
            if total < 3:
                continue

            miss_rate = misses / total
            # Strength = how predictive the factor is for misses
            strength = miss_rate  # Simplified; real PC algorithm would use conditional independence

            if strength >= 0.3:  # Threshold for "causal enough"
                dc = DiscoveredCausal(
                    cause_factor=factor_key,
                    effect_query_pattern=factor_key.split("_")[0],  # Extract topic
                    strength=strength,
                    observations=total,
                )
                self._discovered[factor_key] = dc
                results.append(dc)

        return sorted(results, key=lambda x: x.strength, reverse=True)

    def infer_preconditions_for_query(
        self, query: str, existing_entries: list[CachedEntry]
    ) -> list[Precondition]:
        """
        Infer what preconditions a new query should have,
        based on discovered causal relationships and similar existing entries.

        This is the "causal inference" step — instead of requiring manual
        precondition specification, this derives them from patterns.
        """
        inferred: list[Precondition] = []

        # Find discovered factors that are relevant to this query
        for factor_key, causal in self._discovered.items():
            # Simple keyword matching (production: use embeddings)
            if any(word in query.lower() for word in factor_key.lower().split("_")):
                # Find the current value of this factor from existing entries
                for entry in existing_entries:
                    for pc in entry.preconditions:
                        if pc.key == factor_key:
                            inferred.append(
                                Precondition(
                                    key=factor_key,
                                    operator=pc.operator,
                                    expected_value=pc.expected_value,
                                    source="inferred",
                                )
                            )
                            break

        return inferred

    @property
    def discovered_count(self) -> int:
        return len(self._discovered)

    def stats(self) -> dict:
        return {
            "observations": len(self._observations),
            "discovered_causals": len(self._discovered),
            "factors_monitored": list(self._discovered.keys()),
            "pc_edges": [(e[0], e[1]) for e in self._pc_graph.edges],
        }

    # ==================================================================
    # PC Algorithm — Constraint-Based Causal Discovery (v0.3.0)
    # ==================================================================

    @staticmethod
    def _partial_correlation(
        data: np.ndarray, x_idx: int, y_idx: int, z_idxs: tuple = ()
    ) -> tuple[float, float]:
        """
        Compute partial Pearson correlation between columns x and y,
        controlling for columns in z_idxs, with Fisher-z p-value.

        data:  shape (n_samples, n_variables) — float array
        x_idx, y_idx:  column indices
        z_idxs: tuple of column indices to condition on

        Returns (rho, p_value). rho in [-1, 1], p_value in [0, 1].
        """
        n = data.shape[0]
        if len(z_idxs) == 0:
            # Plain Pearson correlation
            r, p = stats.pearsonr(data[:, x_idx], data[:, y_idx])
            return float(r), float(p) if not np.isnan(p) else 1.0

        # Compute partial correlation via precision matrix approach
        # (regression residuals of x ~ Z, y ~ Z; then correlate residuals)
        Z = data[:, list(z_idxs)]
        # Regress x on Z
        X = data[:, x_idx]
        Y = data[:, y_idx]
        # Use numpy least squares with intercept
        Z_aug = np.column_stack([np.ones(n), Z])
        try:
            beta_x, *_ = np.linalg.lstsq(Z_aug, X, rcond=None)
            beta_y, *_ = np.linalg.lstsq(Z_aug, Y, rcond=None)
        except np.linalg.LinAlgError:
            return 0.0, 1.0
        resid_x = X - Z_aug @ beta_x
        resid_y = Y - Z_aug @ beta_y
        r, p = stats.pearsonr(resid_x, resid_y)
        if np.isnan(r):
            return 0.0, 1.0
        return float(r), float(p) if not np.isnan(p) else 1.0

    def run_pc_algorithm(
        self,
        data_matrix: np.ndarray,
        variable_names: list[str],
        alpha: float = 0.05,
        max_conditioning_size: int = 3,
    ) -> dict:
        """
        Run the PC algorithm: build a skeleton (undirected edges) by
        iteratively testing conditional independence with partial correlation.

        Args:
            data_matrix: shape (n_samples, n_variables). Columns correspond
                to `variable_names`. Must be numeric (factor values may need
                encoding as ordinal integers or one-hot z-scores).
            variable_names: human-readable names for each column.
            alpha: significance threshold for the independence test
                (Fisher-z p-value). Lower = more edges preserved.
            max_conditioning_size: largest conditioning set to try.
                Higher = more thorough but combinatorial cost.

        Returns:
            dict with:
              - "edges": list of (i, j, rho, p_value, z_set) tuples
                representing undirected edges that survived the tests
              - "removed": list of (i, j, z_set, p_value) pairs
                representing edges removed due to conditional independence
              - "stats": summary count
        """
        n_vars = data_matrix.shape[1]
        if len(variable_names) != n_vars:
            raise ValueError(
                f"Mismatch: data has {n_vars} columns, "
                f"got {len(variable_names)} names"
            )
        if data_matrix.shape[0] < 10:
            raise ValueError(
                f"Need at least 10 observations for PC algorithm, "
                f"got {data_matrix.shape[0]}"
            )

        # Skeleton: undirected adjacency (symmetric)
        adjacent = [[True] * n_vars for _ in range(n_vars)]
        for i in range(n_vars):
            adjacent[i][i] = False
        # Z = not yet adjacent (candidate separating sets)
        sep_sets: dict[tuple[int, int], tuple[int, ...]] = {}

        # Increasing size of conditioning set
        removed_edges = []
        for z_size in range(max_conditioning_size + 1):
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    if not adjacent[i][j]:
                        continue  # Already removed by previous test
                    # Adjacent nodes to i (excluding j) form the candidate pool
                    adj_to_i = [
                        k for k in range(n_vars) if adjacent[i][k] and k != j
                    ]
                    if len(adj_to_i) < z_size:
                        continue
                    for z_set in combinations(adj_to_i, z_size):
                        rho, p = self._partial_correlation(
                            data_matrix, i, j, z_set
                        )
                        # Fisher-z significance test:
                        # H0: partial correlation = 0 (independent given Z)
                        # Reject H0 if p < alpha (keep edge)
                        if p > alpha:
                            # X ⊥ Y | Z → remove edge i — j
                            adjacent[i][j] = False
                            adjacent[j][i] = False
                            sep_sets[(i, j)] = z_set
                            sep_sets[(j, i)] = z_set
                            removed_edges.append((i, j, z_set, p))
                            break

        # Build surviving edge list
        edges = []
        seen = set()
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if adjacent[i][j] and (i, j) not in seen:
                    # Compute marginal correlation for the surviving edge
                    rho, p = self._partial_correlation(data_matrix, i, j, ())
                    edges.append((i, j, float(rho), float(p), ()))
                    seen.add((i, j))

        result = {
            "edges": [
                {
                    "cause_idx": e[0],
                    "effect_idx": e[1],
                    "cause": variable_names[e[0]],
                    "effect": variable_names[e[1]],
                    "partial_correlation": e[2],
                    "p_value": e[3],
                    "conditioning_set": [
                        variable_names[k] for k in e[4]
                    ],
                }
                for e in edges
            ],
            "removed": [
                {
                    "i_idx": r[0],
                    "j_idx": r[1],
                    "i": variable_names[r[0]],
                    "j": variable_names[r[1]],
                    "separating_set": [variable_names[k] for k in r[2]],
                    "p_value": r[3],
                }
                for r in removed_edges
            ],
            "stats": {
                "n_variables": n_vars,
                "n_edges": len(edges),
                "n_removed": len(removed_edges),
                "alpha": alpha,
                "max_conditioning_size": max_conditioning_size,
            },
        }
        # Cache the undirected adjacency graph
        self._pc_graph.clear()
        n_edges_added = 0
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if adjacent[i][j]:
                    self._pc_graph.add_edge(variable_names[i], variable_names[j])
                    n_edges_added += 1
        self._pc_result = result
        return result

    def pc_discover(
        self,
        encoding: str = "ordinal",
        alpha: float = 0.05,
    ) -> dict:
        """
        Build a data matrix from cached observations and run the PC algorithm.

        Encoding:
          - "ordinal": each factor value → integer rank (0, 1, 2, ...) in
            order of first appearance. Fast, simple, works for monotonic
            value sequences (good for many real-world factors like versions).
          - "onehot": each (factor, value) pair → one binary column.
            Slower but more accurate. Use when factor values are categorical
            without natural order.

        Returns the same dict format as `run_pc_algorithm`.
        """
        if not self._observations:
            return {"edges": [], "removed": [], "stats": {"n_variables": 0}}

        # Pivot observations: rows = event/timebucket, cols = variables
        # Simple approach: one row per observation, variables = factor values
        # + query/response hash. We focus on factor pairs.
        rows = []
        for obs in self._observations:
            row = {
                "factor:{}".format(obs.precondition_key): _safe_hash(
                    obs.precondition_value_at_time
                ),
            }
            rows.append(row)

        if not rows:
            return {"edges": [], "removed": [], "stats": {"n_variables": 0}}

        # Build variable list
        all_vars = sorted({k for r in rows for k in r})
        if len(all_vars) < 2:
            return {
                "edges": [],
                "removed": [],
                "stats": {"n_variables": len(all_vars)},
            }

        # Build data matrix
        import numpy as np_local
        data = np_local.array(
            [[float(r.get(v, 0.0)) for v in all_vars] for r in rows],
            dtype=float,
        )

        return self.run_pc_algorithm(
            data_matrix=data,
            variable_names=all_vars,
            alpha=alpha,
            max_conditioning_size=min(3, len(all_vars) - 2),
        )


def _safe_hash(value) -> int:
    """Stable hash for any picklable value (used for encoding in PC algorithm)."""
    try:
        if isinstance(value, (int, float, str, bool)):
            return int(hash((type(value).__name__, str(value))) % (2**31))
        return int(hash(repr(value)) % (2**31))
    except Exception:
        return 0


class SimpleUndirectedGraph:
    """
    Minimal undirected graph (no networkx dependency so PC algorithm
    stays lightweight).
    """

    def __init__(self):
        self._adj: dict[str, set[str]] = defaultdict(set)

    @property
    def edges(self):
        seen = set()
        for u, neighbors in self._adj.items():
            for v in neighbors:
                key = tuple(sorted([u, v]))
                if key not in seen:
                    seen.add(key)
                    yield key[0], key[1]

    def add_edge(self, u: str, v: str):
        self._adj[u].add(v)
        self._adj[v].add(u)

    def clear(self):
        self._adj.clear()

    def __len__(self):
        return sum(len(v) for v in self._adj.values()) // 2