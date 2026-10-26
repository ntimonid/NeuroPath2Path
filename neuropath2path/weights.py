"""Feature-distance scatter assembly and constrained importance-weight fitting."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from .classification import Dataset
from .distance import compute_dst
from .features import NeuronFeatures, PathFeatures


@dataclass(frozen=True)
class WeightFit:
    weights: np.ndarray
    iterations: int
    objective: float
    method: str


def get_distance_vec_assembly(
    classes: Sequence[Sequence[NeuronFeatures | Sequence[PathFeatures]]],
    indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum vv.T for between-class pairs and separately for within-class pairs.

    ``indices`` are zero-based and actually select the requested classes; the
    MATLAB implementation mistakenly always selects the first len(indices).
    The within-class tensor has shape (6, 6, number_of_selected_classes).
    """
    indices = tuple(range(len(classes))) if indices is None else tuple(indices)
    if len(indices) < 2 or len(set(indices)) != len(indices):
        raise ValueError("Choose at least two distinct class indices")
    if any(i < 0 or i >= len(classes) for i in indices):
        raise ValueError("Class index out of range")
    selected = [classes[i] for i in indices]
    if any(not len(group) for group in selected):
        raise ValueError("Selected classes must contain neurons")
    between = np.zeros((6, 6))
    within = np.zeros((6, 6, len(indices)))
    for a, b in combinations(selected, 2):
        for first, second in product(a, b):
            vectors = compute_dst(first, second).feature_vectors
            between += vectors @ vectors.T
    for i, group in enumerate(selected):
        for first, second in combinations(group, 2):
            vectors = compute_dst(first, second).feature_vectors
            within[:, :, i] += vectors @ vectors.T
    return between, within


def fit_feature_weights(between: np.ndarray, within: np.ndarray, *,
                        method: str = "constrained", seed: int = 100,
                        initial: np.ndarray | None = None, max_iterations: int = 10000,
                        tolerance: float | None = None) -> WeightFit:
    """Fit the original quadratic/log-barrier objective on the weight simplex.

    ``constrained`` rescales the objective and uses bounded SLSQP to keep
    weights positive and summing to one. ``legacy`` translates the original
    normalized gradient iteration, adding a finite iteration limit and checks.
    Identical initialization can be passed for cross-language comparisons.
    """
    between, within = np.asarray(between, dtype=float), np.asarray(within, dtype=float)
    if between.shape != (6, 6) or within.ndim != 3 or within.shape[:2] != (6, 6) or within.shape[2] < 1:
        raise ValueError("Expected between shape (6, 6) and within shape (6, 6, classes)")
    if not np.isfinite(between).all() or not np.isfinite(within).all():
        raise ValueError("Scatter matrices must be finite")
    if (not np.allclose(between, between.T)
            or not np.allclose(within, within.swapaxes(0, 1))):
        raise ValueError("Scatter matrices must be symmetric")
    if method not in {"constrained", "legacy"}:
        raise ValueError("method must be 'constrained' or 'legacy'")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    tolerance = (0.01 if method == "legacy" else 1e-12) if tolerance is None else tolerance
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if initial is None:
        initial = np.random.default_rng(seed).integers(1, 11, size=6).astype(float)
    current = np.asarray(initial, dtype=float).copy()
    if current.shape != (6,) or not np.isfinite(current).all() or np.any(current <= 0):
        raise ValueError("initial must contain six finite positive values")
    current /= current.sum()
    quadratic = within.sum(axis=2) - between
    if method == "legacy":
        previous = np.full(6, 1 / 6)
        for iteration in range(max_iterations + 1):
            if np.linalg.norm(current - previous) <= tolerance:
                objective = float(0.5 * current @ quadratic @ current
                                  - np.log(current + 0.0001).sum())
                return WeightFit(current, iteration, objective, method)
            if iteration == max_iterations:
                break
            gradient = quadratic @ current - 1 / (current + 0.0001)
            previous = current.copy()
            current -= 0.0001 * gradient
            total = current.sum()
            if not np.isfinite(current).all() or abs(total) < 1e-15:
                raise RuntimeError("Legacy weight iteration diverged; use method='constrained'")
            current /= total
            if np.any(current <= 0):
                raise RuntimeError("Legacy iteration produced nonpositive weights; "
                                   "use method='constrained'")
        raise RuntimeError(f"Legacy weights did not converge in {max_iterations} iterations")

    # Raw scatter magnitude grows with dataset size; scale for numerical stability.
    scale = max(1.0, float(np.linalg.norm(between, ord=2)),
                float(np.linalg.norm(within.sum(axis=2), ord=2)))
    quadratic = quadratic / scale

    def objective(weights):
        return float(0.5 * weights @ quadratic @ weights
                     - np.log(weights + 0.0001).sum() / scale)

    def gradient(weights):
        return quadratic @ weights - 1 / (weights + 0.0001) / scale

    result = minimize(objective, current, jac=gradient, method="SLSQP",
                      bounds=[(1e-10, 1)] * 6,
                      constraints={"type": "eq", "fun": lambda w: w.sum() - 1,
                                   "jac": lambda w: np.ones(6)},
                      options={"maxiter": max_iterations, "ftol": tolerance})
    if not result.success:
        raise RuntimeError(f"Feature-weight optimization failed: {result.message}")
    return WeightFit(result.x, int(result.nit), float(result.fun), method)


def get_feature_weights(between: np.ndarray, within: np.ndarray, **kwargs) -> np.ndarray:
    return fit_feature_weights(between, within, **kwargs).weights


def learn_dataset_weights(dataset: Dataset, *, method: str = "constrained", seed: int = 100,
                          progress: Callable[[str], None] | None = None) -> Mapping[str, WeightFit]:
    """Counterpart of jobIW.m: fit each unordered class pair separately."""
    labels = dataset.classes
    if len(labels) < 2:
        raise ValueError("Weight learning needs at least two classes")
    groups = [[record.features for record in dataset.records if record.label == label]
              for label in labels]
    fits = {}
    for a, b in combinations(range(len(labels)), 2):
        name = f"{labels[a]} / {labels[b]}"
        if progress:
            progress(f"Assembling feature scatter: {name}")
        between, within = get_distance_vec_assembly(groups, (a, b))
        fits[name] = fit_feature_weights(between, within, method=method, seed=seed)
    return fits
