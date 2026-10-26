"""Path alignment, repeated Hungarian assignment, and neuron distances."""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import expit

from .features import NeuronFeatures, PathFeatures


INTER_WEIGHTS = np.array([0.0304, 0.0372, 0.0417, 0.1786, 0.6489, 0.0633])
INTRA_WEIGHTS = np.array([0.0785, 0.0807, 0.1736, 0.1956, 0.2600, 0.2076])


@dataclass(frozen=True)
class DistanceResult:
    distance: float
    feature_vectors: np.ndarray  # six features x number of matched path pairs
    pairs: np.ndarray            # zero-based indices into the two input path lists
    cost_matrix: np.ndarray

    def __iter__(self):
        """Allow the original three-return unpacking convention."""
        yield self.distance
        yield self.feature_vectors
        yield self.pairs


def _paths(data: NeuronFeatures | Sequence[PathFeatures]) -> tuple[PathFeatures, ...]:
    paths = data.paths if isinstance(data, NeuronFeatures) else tuple(data)
    if not paths:
        raise ValueError("Neuron distance requires at least one dendritic path per neuron")
    for path in paths:
        if (not len(path.hierarchy) or path.values.shape != (len(path.hierarchy), 6)
                or not np.isfinite(path.values).all()
                or not np.isfinite(path.hierarchy).all() or np.any(path.hierarchy <= 0)):
            raise ValueError("Path features must be finite and have a common positive segment count")
    return paths


def _difference_block(a: Sequence[PathFeatures], b: Sequence[PathFeatures],
                      mode: str, matlab_compatible: bool) -> np.ndarray:
    """Vectorized distances between groups with a common segment count."""
    av, bv = np.stack([p.values for p in a]), np.stack([p.values for p in b])
    la, lb = av.shape[1], bv.shape[1]
    length = max(la, lb)
    minimum = np.minimum(av.min(axis=1)[:, None, :], bv.min(axis=1)[None, :, :])
    maximum = np.maximum(av.max(axis=1)[:, None, :], bv.max(axis=1)[None, :, :]) + 0.0001
    # The original length normalization uses mean and sample standard deviation.
    am, bm = av[:, :, 4].mean(axis=1), bv[:, :, 4].mean(axis=1)
    mean = (am[:, None] * la + bm[None, :] * lb) / (la + lb)
    variance = ((av[:, :, 4].var(axis=1) * la)[:, None]
                + (bv[:, :, 4].var(axis=1) * lb)[None, :]
                + (am[:, None] - bm[None, :]) ** 2 * la * lb / (la + lb)) / (la + lb - 1)
    minimum[:, :, 4] = np.sqrt(variance)
    maximum[:, :, 4] = mean + 0.0001
    denominator = maximum - minimum
    # Only singular or effectively singular normalizations need a safeguard.
    denominator = np.where(np.abs(denominator) < 1e-12,
                           np.where(denominator < 0, -1e-12, 1e-12), denominator)
    aa = (av[:, None, :, :] - minimum[:, :, None, :]) / denominator[:, :, None, :]
    bb = (bv[None, :, :, :] - minimum[:, :, None, :]) / denominator[:, :, None, :]
    leading = mode == "intra"
    pad_a = (length - la, 0) if leading else (0, length - la)
    pad_b = (length - lb, 0) if leading else (0, length - lb)
    aa = np.pad(aa, ((0, 0), (0, 0), pad_a, (0, 0)))
    bb = np.pad(bb, ((0, 0), (0, 0), pad_b, (0, 0)))
    if matlab_compatible and mode != "raw" and la >= lb:
        # computeDSTInter/Intra use tort1 instead of normalized tort11 here.
        aa[:, :, :, 0] = av[:, None, :, 0]
    hierarchy = np.stack([p.hierarchy for p in (a if la >= lb else b)])
    order = 1 / (hierarchy + 0.1)
    order /= order.sum(axis=1, keepdims=True)
    order = order[:, None, :, None] if la >= lb else order[None, :, :, None]
    vectors = np.sqrt((((aa - bb) * order) ** 2).sum(axis=2))
    vectors[:, :, 4] /= np.sqrt(length)
    if mode == "raw":
        vectors[:, :, 0] /= np.sqrt(length)
    return vectors


def path_distances(data_a: NeuronFeatures | Sequence[PathFeatures],
                   data_b: NeuronFeatures | Sequence[PathFeatures], *,
                   mode: str = "inter", weights: np.ndarray | None = None,
                   matlab_compatible: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return path costs and a (paths_a, paths_b, 6) feature-distance tensor.

    Interclass alignment pads at the tips and applies the original sigmoid;
    intraclass alignment pads at the root and uses a weighted sum. ``raw``
    implements computeDST.m, used when learning feature weights.
    """
    if mode not in {"inter", "intra", "raw"}:
        raise ValueError("mode must be 'inter', 'intra', or 'raw'")
    a, b = _paths(data_a), _paths(data_b)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if mode == "raw":
            raise ValueError("raw distances do not use feature weights")
        if weights.shape != (6,) or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("weights must contain six finite nonnegative values")
        if weights.sum() <= 0:
            raise ValueError("at least one feature weight must be positive")
    groups_a: dict[int, list[int]] = defaultdict(list)
    groups_b: dict[int, list[int]] = defaultdict(list)
    for i, path in enumerate(a):
        groups_a[len(path.hierarchy)].append(i)
    for j, path in enumerate(b):
        groups_b[len(path.hierarchy)].append(j)
    vectors = np.empty((len(a), len(b), 6))
    for la, rows in groups_a.items():
        for lb, cols in groups_b.items():
            # Bound temporary broadcast arrays even for highly branched arbors.
            chunk = max(1, min(64, int(np.sqrt(131072 / max(la, lb)))))
            for i in range(0, len(rows), chunk):
                rr = rows[i:i + chunk]
                for j in range(0, len(cols), chunk):
                    cc = cols[j:j + chunk]
                    vectors[np.ix_(rr, cc)] = _difference_block(
                        [a[r] for r in rr], [b[c] for c in cc], mode, matlab_compatible)
    if not np.isfinite(vectors).all():
        raise ValueError("Path distance normalization produced nonfinite values")
    if mode == "raw":
        costs = np.linalg.norm(vectors, axis=2)
    else:
        if weights is None:
            weights = INTER_WEIGHTS if mode == "inter" else INTRA_WEIGHTS
        costs = vectors @ weights
        if mode == "inter":
            costs = expit(costs)
    return costs, vectors


def munkres(cost_matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Optimal partial assignment; zero-based columns, -1 for unassigned rows.

    NaN and infinity mark forbidden edges. Feasible edges are matched with
    maximum cardinality and then minimum total cost, including rectangular cases.
    """
    costs = np.asarray(cost_matrix, dtype=float)
    if costs.ndim != 2:
        raise ValueError("Assignment costs must be a matrix")
    assigned = np.full(costs.shape[0], -1, dtype=np.int64)
    valid = np.isfinite(costs)
    if not valid.any():
        return assigned, 0.0
    scale = max(1.0, float(np.abs(costs[valid]).max()))
    normalized = np.where(valid, costs / scale, 0)
    # A forbidden edge costs more than all possible finite-cost differences.
    normalized[~valid] = 2 * min(costs.shape) + 1
    rows, cols = linear_sum_assignment(normalized)
    keep = valid[rows, cols]
    assigned[rows[keep]] = cols[keep]
    return assigned, float(costs[rows[keep], cols[keep]].sum())


def match_paths(cost_matrix: np.ndarray, lengths_a: Sequence[int],
                lengths_b: Sequence[int]) -> tuple[float, np.ndarray]:
    """Repeated one-to-one rounds followed by the original outlier reassignment.

    Each path of the larger arbor gets one match. For equal optima, SciPy may
    choose a different correspondence than the MATLAB Hungarian implementation.
    """
    costs = np.asarray(cost_matrix, dtype=float)
    if costs.shape != (len(lengths_a), len(lengths_b)) or not costs.size:
        raise ValueError("Cost matrix must match two nonempty path lists")
    if not np.isfinite(costs).all() or np.any(costs < 0):
        raise ValueError("Path costs must be finite and nonnegative")
    swapped = costs.shape[0] > costs.shape[1]
    if swapped:
        costs = costs.T
        lengths_a, lengths_b = lengths_b, lengths_a
    m, n = costs.shape
    remaining = np.arange(n)
    pairs = []
    for _ in range(n // m):
        rows, columns = linear_sum_assignment(costs[:, remaining])
        selected = remaining[columns]
        pairs.extend(zip(rows.tolist(), selected.tolist()))
        remaining = np.setdiff1d(remaining, selected, assume_unique=True)
    if len(remaining):
        rows, columns = linear_sum_assignment(costs[:, remaining].T)
        pairs.extend(zip(columns.tolist(), remaining[rows].tolist()))
    pairs_array = np.asarray(pairs, dtype=np.int64)
    values = costs[pairs_array[:, 0], pairs_array[:, 1]]
    degrees = np.bincount(pairs_array[:, 0], minlength=m)
    centered = values - values.mean()
    variance = np.mean(centered ** 2)
    skew = np.mean(centered ** 3) / variance ** 1.5 if variance > 0 else 0
    if skew > 0 and n > 1:
        outliers = np.flatnonzero(values >= np.median(values) + values.std(ddof=1))
        removed, reassigned = [], []
        for index in outliers:
            a, b = pairs_array[index]
            la, lb = lengths_a[a], lengths_b[b]
            if degrees[a] > 1 and abs(la - lb) < max(la, lb) / 2:
                removed.append(index)
                reassigned.append((int(np.argmin(costs[:, b])), int(b)))
        if removed:
            pairs_array = np.vstack((np.delete(pairs_array, removed, axis=0), reassigned))
    distance = float(costs[pairs_array[:, 0], pairs_array[:, 1]].sum() / n)
    return distance, pairs_array[:, ::-1].copy() if swapped else pairs_array


def compute_distance(data_a: NeuronFeatures | Sequence[PathFeatures],
                     data_b: NeuronFeatures | Sequence[PathFeatures], *,
                     mode: str = "inter", weights: np.ndarray | None = None,
                     matlab_compatible: bool = False) -> DistanceResult:
    a, b = _paths(data_a), _paths(data_b)
    costs, vectors = path_distances(a, b, mode=mode, weights=weights,
                                    matlab_compatible=matlab_compatible)
    distance, pairs = match_paths(costs, [len(p.hierarchy) for p in a],
                                 [len(p.hierarchy) for p in b])
    return DistanceResult(distance, vectors[pairs[:, 0], pairs[:, 1]].T.copy(), pairs, costs)


def compute_dst(data_a, data_b) -> DistanceResult:
    """Counterpart of computeDST.m, for feature-weight estimation."""
    return compute_distance(data_a, data_b, mode="raw")


def compute_dst_inter(data_a, data_b, **kwargs) -> DistanceResult:
    return compute_distance(data_a, data_b, mode="inter", **kwargs)


def compute_dst_intra(data_a, data_b, **kwargs) -> DistanceResult:
    return compute_distance(data_a, data_b, mode="intra", **kwargs)


def distance_neuron_new(cost_matrix, data_a, data_b) -> tuple[float, np.ndarray]:
    a, b = _paths(data_a), _paths(data_b)
    return match_paths(cost_matrix, [len(p.hierarchy) for p in a], [len(p.hierarchy) for p in b])
