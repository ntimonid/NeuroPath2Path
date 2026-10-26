"""Root-to-dendritic-tip features corresponding to computenP2P.m."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree

from .io import Morphology, morphology_from_array


FEATURE_NAMES = ("tortuosity", "bifurcation_angle", "partition_asymmetry",
                 "concurrence", "segment_length", "competition")


@dataclass(frozen=True)
class PathFeatures:
    tortuosity: np.ndarray
    bifurcation_angle: np.ndarray
    partition_asymmetry: np.ndarray
    concurrence: np.ndarray
    hierarchy: np.ndarray
    segment_length: np.ndarray
    competition: np.ndarray
    node_ids: np.ndarray

    @property
    def values(self) -> np.ndarray:
        """Six distance features, excluding hierarchy and original node IDs."""
        return np.column_stack([getattr(self, name) for name in FEATURE_NAMES])


@dataclass(frozen=True)
class NeuronFeatures:
    paths: tuple[PathFeatures, ...]
    average_path_length: float
    max_order: int

    @property
    def path_count(self) -> int:
        return len(self.paths)


def path_concurrence(morphology: Morphology, counts: np.ndarray, node: int) -> np.ndarray:
    """Increment the ancestors of a zero-based node index, in place."""
    while morphology.parents[node] != -1:
        node = int(morphology.parents[node])
        counts[node] += 1
    return counts


def extract_features(data: Morphology | np.ndarray) -> NeuronFeatures:
    """Extract seven per-segment descriptors and root-to-tip node IDs.

    Terminal types 3 and 4 define dendritic paths, as in MATLAB. Axon-only
    arbors return zero paths. Axonal branches still participate in node degree
    and branch-angle calculations, matching the original algorithm.
    """
    tree = data if isinstance(data, Morphology) else morphology_from_array(data)
    root = tree.root
    nodes = tree.nodes
    # Centering and axis permutation preserve distances and branch angles.
    coords = nodes[:, 2:5] - nodes[root, 2:5]
    coords = coords[:, np.argsort(np.ptp(coords, axis=0), kind="stable")]
    degree = np.array([len(c) + int(i != root) for i, c in enumerate(tree.children)])
    tips = [i for i in range(len(nodes))
            if i != root and degree[i] == 1 and nodes[i, 1] in (3, 4)]
    if not tips:
        return NeuronFeatures((), 0.0, 0)

    concurrence = np.zeros(len(nodes), dtype=np.int64)
    concurrence[tips] = 1
    # Reverse traversal avoids Python recursion limits on long SWC branches.
    for node in tree.order[::-1]:
        parent = tree.parents[node]
        if parent != -1:
            concurrence[parent] += concurrence[node]

    path_rows = []
    memberships: list[set[int]] = [set() for _ in nodes]
    for path_id, tip in enumerate(tips):
        path = [tip]
        while path[-1] != root:
            path.append(int(tree.parents[path[-1]]))
        path.reverse()
        path_rows.append(np.array(path, dtype=np.int64))
        for node in path:
            memberships[node].add(path_id)

    bifurcations = set(np.flatnonzero(degree >= 3)) - {root}
    angle = np.zeros(len(nodes))
    asymmetry = np.zeros(len(nodes))
    for node in bifurcations | {root}:
        for a, b in combinations(tree.children[node], 2):
            u, v = coords[a] - coords[node], coords[b] - coords[node]
            denominator = np.linalg.norm(u) * np.linalg.norm(v)
            if denominator > 0:
                theta = np.arccos(np.clip(np.dot(u, v) / denominator, -1, 1))
                angle[node] = max(angle[node], float(theta))
            total = concurrence[a] + concurrence[b]
            if total:
                asymmetry[node] = max(asymmetry[node],
                                      abs(concurrence[a] - concurrence[b]) / total)

    # A spatial query replaces the original branch x path x node scan.
    std = coords.std(axis=0, ddof=0)
    standardized = (coords - coords.mean(axis=0)) / np.where(std > 0, std, 1)
    spatial = cKDTree(standardized)
    competition = np.zeros(len(nodes))
    for node in bifurcations:
        nearby_paths: set[int] = set()
        for neighbor in spatial.query_ball_point(standardized[node], r=0.5):
            nearby_paths.update(memberships[neighbor])
        competition[node] = len(nearby_paths - memberships[node]) / len(tips)

    paths = []
    for rows in path_rows:
        # Path order also supports SWCs whose IDs are not topologically sorted.
        junction_positions = [0] + [i for i, node in enumerate(rows[1:-1], 1)
                                    if node in bifurcations] + [len(rows) - 1]
        junctions = rows[junction_positions[:-1]]
        segment_lengths, tortuosity = [], []
        for start, stop in zip(junction_positions[:-1], junction_positions[1:]):
            points = coords[rows[start:stop + 1]]
            arc = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            chord = float(np.linalg.norm(points[-1] - points[0]))
            segment_lengths.append(arc)
            tortuosity.append(1.0 if stop == start + 1 or arc == 0
                               else arc / max(chord, np.finfo(float).eps))
        conc = concurrence[junctions].copy()
        # MATLAB stores the 1-based sort permutation rather than tied ranks.
        hierarchy = np.argsort(-conc, kind="stable") + 1
        paths.append(PathFeatures(
            np.array(tortuosity), angle[junctions].copy(), asymmetry[junctions].copy(),
            conc, hierarchy, np.array(segment_lengths), competition[junctions].copy(),
            nodes[rows, 0].astype(np.int64),
        ))
    return NeuronFeatures(tuple(paths),
                          float(np.mean([p.segment_length.sum() for p in paths])),
                          max(len(p.hierarchy) for p in paths))


def compute_np2p(data: Morphology | np.ndarray) -> tuple[tuple[PathFeatures, ...], int, float, int]:
    """Four-return counterpart of computenP2P.m."""
    result = extract_features(data)
    return result.paths, result.path_count, result.average_path_length, result.max_order
