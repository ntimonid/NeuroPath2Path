from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from neuropath2path import (
    NeuronFeatures, PathFeatures, compute_distance, extract_features, match_paths,
    munkres, path_distances,
)


def make_path(length):
    values = np.arange(1, length + 1, dtype=float)
    return PathFeatures(values + 0.2, values / 3, values / 4, values[::-1],
                        values, values * 2 + 1, values / 8, np.arange(length + 1))


@pytest.mark.parametrize("mode, expected", [("raw", 0), ("intra", 0), ("inter", 0.5)])
def test_identical_neuron(branched_tree, mode, expected):
    features = extract_features(branched_tree)
    result = compute_distance(features, features, mode=mode)
    assert result.distance == expected
    assert_allclose(result.feature_vectors, 0)
    assert result.pairs.shape == (2, 2)
    distance, vectors, pairs = result
    assert distance == expected and vectors.shape == (6, 2) and pairs.shape == (2, 2)


def test_original_tortuosity_bug_is_explicit(branched_tree):
    features = extract_features(branched_tree)
    corrected = compute_distance(features, features, mode="intra")
    legacy = compute_distance(features, features, mode="intra", matlab_compatible=True)
    assert corrected.distance == 0 and legacy.distance > 0
    assert np.all(legacy.feature_vectors[0] > 0)


def scalar_reference(a, b, mode, legacy=False):
    output = []
    na, nb = len(a.hierarchy), len(b.hierarchy)
    weights = 1 / ((a if na >= nb else b).hierarchy + 0.1)
    weights /= weights.sum()
    for i in range(6):
        left, right = a.values[:, i], b.values[:, i]
        joined = np.r_[left, right]
        low, high = ((joined.std(ddof=1), joined.mean() + 0.0001) if i == 4
                     else (joined.min(), joined.max() + 0.0001))
        l, r = (left - low) / (high - low), (right - low) / (high - low)
        if legacy and mode != "raw" and i == 0 and na >= nb:
            l = left.copy()
        front = mode == "intra"
        if na < nb:
            l = np.pad(l, (nb - na, 0) if front else (0, nb - na))
        elif nb < na:
            r = np.pad(r, (na - nb, 0) if front else (0, na - nb))
        value = np.linalg.norm((l - r) * weights)
        if i == 4 or i == 0 and mode == "raw":
            value /= np.sqrt(max(na, nb))
        output.append(value)
    return output


@pytest.mark.parametrize("mode", ["raw", "inter", "intra"])
@pytest.mark.parametrize("legacy", [False, True])
def test_vectorized_distances_against_scalar_formula(mode, legacy):
    first = [make_path(n) for n in [1, 4, 2, 3, 2]]
    second = [replace(make_path(n), segment_length=make_path(n).segment_length * 1.25)
              for n in [3, 1, 4]]
    _, vectors = path_distances(first, second, mode=mode, matlab_compatible=legacy)
    for i, a in enumerate(first):
        for j, b in enumerate(second):
            assert_allclose(vectors[i, j], scalar_reference(a, b, mode, legacy), atol=1e-12)


@pytest.mark.parametrize("mode", ["raw", "inter", "intra"])
def test_cost_matrix_transposes_for_unequal_orders(mode):
    a, b = [make_path(n) for n in [1, 2]], [make_path(n) for n in [3, 4, 5]]
    ab, _ = path_distances(a, b, mode=mode)
    ba, _ = path_distances(b, a, mode=mode)
    assert_allclose(ab, ba.T)


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (3, 3), (2, 5), (5, 2), (3, 9)])
def test_assignment_covers_every_larger_path(shape):
    costs = np.random.default_rng(9).random(shape)
    result, pairs = match_paths(costs, [3] * shape[0], [3] * shape[1])
    assert len(pairs) == max(shape)
    larger_axis = 1 if shape[1] >= shape[0] else 0
    assert sorted(pairs[:, larger_axis]) == list(range(max(shape)))
    assert_allclose(result, costs[pairs[:, 0], pairs[:, 1]].mean())


def test_outlier_reassignment():
    costs = np.array([[0, 0.1, 0.2, 0.25, 10], [0.05, 0.06, 0.3, 0.35, 11]])
    _, pairs = match_paths(costs, [3, 3], [3] * 5)
    assert pairs[pairs[:, 1] == 4, 0].item() == 0


def test_partial_munkres():
    assignment, cost = munkres(np.array([[1, 3, np.inf], [np.inf, np.inf, 5],
                                         [np.inf, np.inf, 0.5]]))
    assert_array_equal(assignment, [0, -1, 2])
    assert cost == 1.5
    assignment, cost = munkres(np.full((2, 3), np.inf))
    assert_array_equal(assignment, [-1, -1])
    assert cost == 0


@pytest.mark.parametrize("weights", [[1], [-1, 1, 1, 1, 1, 1], [0] * 6, [np.nan] * 6])
def test_invalid_weights(branched_tree, weights):
    f = extract_features(branched_tree)
    with pytest.raises(ValueError, match="weight"):
        compute_distance(f, f, weights=weights)


def test_empty_neuron_distance(branched_tree):
    with pytest.raises(ValueError, match="dendritic path"):
        compute_distance(NeuronFeatures((), 0, 0), extract_features(branched_tree))
