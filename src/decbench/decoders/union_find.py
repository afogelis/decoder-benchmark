"""Union-Find decoder (from-scratch implementation).

This implements the Delfosse-Nickerson union-find decoder with unweighted
cluster growth followed by spanning-forest peeling. It operates on the matching
graph extracted from a circuit's detector error model (via PyMatching's graph),
so it shares the exact same graph and ``fault_ids`` bookkeeping as the MWPM
reference decoder.

Why implement it ourselves: union-find is the leading *near-linear-time*
alternative to MWPM, and a benchmark that only calls a black-box library cannot
expose the accuracy/runtime trade-off that motivates the algorithm. The growth
phase merges odd-parity clusters until each is either even or touches the
boundary; the peeling phase then reads a correction off each cluster's spanning
forest.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pymatching
import stim


class _DisjointSet:
    """Union-find over integer node ids with union-by-rank and path compression."""

    def __init__(self, num_nodes: int) -> None:
        self.parent = list(range(num_nodes))
        self.rank = [0] * num_nodes

    def find(self, node: int) -> int:
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != root:
            self.parent[node], node = root, self.parent[node]
        return root

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra


class UnionFindDecoder:
    """A graph union-find decoder built on the PyMatching matching graph."""

    name = "union_find"

    def __init__(self) -> None:
        self._num_nodes = 0
        self._num_observables = 0
        self._is_boundary: np.ndarray | None = None
        # Edge arrays, indexed by edge id.
        self._edge_u: np.ndarray | None = None
        self._edge_v: np.ndarray | None = None
        self._edge_faults: list[frozenset[int]] = []
        self._incident: list[list[int]] = []  # node -> list of edge ids
        self._detector_nodes: list[int] = []

    def fit(self, circuit: stim.Circuit) -> None:
        dem = circuit.detector_error_model(decompose_errors=True)
        matching = pymatching.Matching.from_detector_error_model(dem)
        graph = matching.to_networkx()

        self._num_nodes = graph.number_of_nodes()
        self._num_observables = int(matching.num_fault_ids)
        self._is_boundary = np.zeros(self._num_nodes, dtype=bool)
        for node, data in graph.nodes(data=True):
            if data.get("is_boundary", False):
                self._is_boundary[node] = True
        self._detector_nodes = [n for n in range(self._num_nodes) if not self._is_boundary[n]]

        edge_u: list[int] = []
        edge_v: list[int] = []
        self._edge_faults = []
        self._incident = [[] for _ in range(self._num_nodes)]
        for u, v, data in graph.edges(data=True):
            edge_id = len(edge_u)
            edge_u.append(u)
            edge_v.append(v)
            fault_ids = data.get("fault_ids", set()) or set()
            self._edge_faults.append(frozenset(int(f) for f in fault_ids))
            self._incident[u].append(edge_id)
            self._incident[v].append(edge_id)
        self._edge_u = np.asarray(edge_u, dtype=np.int64)
        self._edge_v = np.asarray(edge_v, dtype=np.int64)

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        if self._edge_u is None or self._is_boundary is None:
            raise RuntimeError("decoder must be fit() before decode_batch()")
        events = np.asarray(detection_events, dtype=bool)
        predictions = np.zeros((events.shape[0], self._num_observables), dtype=bool)
        for shot in range(events.shape[0]):
            predictions[shot] = self._decode_single(events[shot])
        return predictions

    def _decode_single(self, syndrome_row: np.ndarray) -> np.ndarray:
        syndrome = {node for node in self._detector_nodes if syndrome_row[node]}
        prediction = np.zeros(self._num_observables, dtype=bool)
        if not syndrome:
            return prediction

        grown_edges = self._grow_clusters(syndrome)
        correction_edges = self._peel(syndrome, grown_edges)
        for edge_id in correction_edges:
            for fault in self._edge_faults[edge_id]:
                prediction[fault] ^= True
        return prediction

    def _cluster_parity(self, dsets: _DisjointSet, syndrome: set[int]) -> dict[int, int]:
        parity: dict[int, int] = {}
        for node in syndrome:
            root = dsets.find(node)
            parity[root] = parity.get(root, 0) ^ 1
        return parity

    def _cluster_has_boundary(self, dsets: _DisjointSet) -> set[int]:
        assert self._is_boundary is not None
        roots: set[int] = set()
        for node in range(self._num_nodes):
            if self._is_boundary[node]:
                roots.add(dsets.find(node))
        return roots

    def _grow_clusters(self, syndrome: set[int]) -> set[int]:
        """Grow odd clusters by half-edges until all are even or hit the boundary.

        Returns the set of fully grown edge ids.
        """
        assert self._edge_u is not None and self._edge_v is not None
        dsets = _DisjointSet(self._num_nodes)
        num_edges = self._edge_u.shape[0]
        growth = np.zeros(num_edges, dtype=np.int8)  # 0, 1 or 2 half-steps
        fully_grown: set[int] = set()

        def odd_roots() -> set[int]:
            parity = self._cluster_parity(dsets, syndrome)
            boundary_roots = self._cluster_has_boundary(dsets)
            return {root for root, par in parity.items() if par == 1 and root not in boundary_roots}

        active = odd_roots()
        # Bound the number of rounds defensively; each round adds >= one half-step.
        max_rounds = 2 * num_edges + 4
        rounds = 0
        while active and rounds < max_rounds:
            rounds += 1
            newly_grown: list[int] = []
            growable = [e for e in range(num_edges) if e not in fully_grown]
            for edge_id in growable:
                u = int(self._edge_u[edge_id])
                v = int(self._edge_v[edge_id])
                touch = (dsets.find(u) in active) + (dsets.find(v) in active)
                if touch == 0:
                    continue
                growth[edge_id] = min(2, growth[edge_id] + touch)
                if growth[edge_id] >= 2:
                    newly_grown.append(edge_id)
            for edge_id in newly_grown:
                fully_grown.add(edge_id)
                dsets.union(int(self._edge_u[edge_id]), int(self._edge_v[edge_id]))
            active = odd_roots()
        return fully_grown

    def _peel(self, syndrome: set[int], grown_edges: set[int]) -> list[int]:
        """Peel spanning forests of the grown subgraph into a correction set."""
        assert self._is_boundary is not None
        adjacency: dict[int, list[tuple[int, int]]] = {n: [] for n in range(self._num_nodes)}
        for edge_id in grown_edges:
            u = int(self._edge_u[edge_id])  # type: ignore[index]
            v = int(self._edge_v[edge_id])  # type: ignore[index]
            adjacency[u].append((v, edge_id))
            adjacency[v].append((u, edge_id))

        residual = {node: (node in syndrome) for node in range(self._num_nodes)}
        visited = [False] * self._num_nodes
        correction: list[int] = []

        for component_seed in range(self._num_nodes):
            if visited[component_seed] or not adjacency[component_seed]:
                continue
            root = self._select_root(component_seed, adjacency)
            order, parent_edge, parent_node = self._spanning_tree(root, adjacency, visited)
            # Peel leaves first: reverse breadth-first discovery order.
            for node in reversed(order):
                if node == root:
                    continue
                if residual[node]:
                    edge_id = parent_edge[node]
                    correction.append(edge_id)
                    residual[node] = False
                    parent = parent_node[node]
                    if not self._is_boundary[parent]:
                        residual[parent] = not residual[parent]
        return correction

    def _select_root(self, seed: int, adjacency: dict[int, list[tuple[int, int]]]) -> int:
        """Prefer a boundary node as the spanning-tree root so it can absorb parity."""
        assert self._is_boundary is not None
        queue = deque([seed])
        seen = {seed}
        while queue:
            node = queue.popleft()
            if self._is_boundary[node]:
                return node
            for neighbour, _ in adjacency[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
        return seed

    def _spanning_tree(
        self,
        root: int,
        adjacency: dict[int, list[tuple[int, int]]],
        visited: list[bool],
    ) -> tuple[list[int], dict[int, int], dict[int, int]]:
        order: list[int] = []
        parent_edge: dict[int, int] = {}
        parent_node: dict[int, int] = {}
        queue = deque([root])
        visited[root] = True
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbour, edge_id in adjacency[node]:
                if not visited[neighbour]:
                    visited[neighbour] = True
                    parent_edge[neighbour] = edge_id
                    parent_node[neighbour] = node
                    queue.append(neighbour)
        return order, parent_edge, parent_node
