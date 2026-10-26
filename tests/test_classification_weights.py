from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from neuropath2path import (
    Dataset, Split, classify, compute_dst, fit_feature_weights,
    get_distance_vec_assembly, load_dataset, stratified_split,
)


def test_split_reproducible_disjoint_and_complete(dataset_dir):
    dataset = load_dataset(dataset_dir)
    a = stratified_split(dataset, test_proportion=0.4, bin_width=10, seed=3)
    assert a == stratified_split(dataset, test_proportion=0.4, bin_width=10, seed=3)
    assert len(a.train) == 6 and len(a.test) == 4
    assert not set(a.train) & set(a.test)
    assert set(a.train) | set(a.test) == set(range(10))
    assert a != stratified_split(dataset, test_proportion=0.4, bin_width=10, seed=5)


def test_high_test_proportion_keeps_training_member(dataset_dir):
    dataset = load_dataset(dataset_dir)
    split = stratified_split(dataset, test_proportion=0.99, bin_width=1)
    assert len(split.train) == 2 and len(split.test) == 8


@pytest.mark.parametrize("mode", ["inter", "intra"])
def test_classification_and_concurrency(dataset_dir, mode):
    dataset = load_dataset(dataset_dir)
    kwargs = dict(mode=mode, test_proportion=0.4, bin_width=10, neighbors=1, seed=3)
    serial, parallel = classify(dataset, **kwargs), classify(dataset, jobs=2, **kwargs)
    assert serial.accuracy == parallel.accuracy == 1
    assert serial.predictions == parallel.predictions
    assert_array_equal(serial.confusion_matrix, [[2, 0], [0, 2]])
    training_sources = {dataset.records[i].source for i in serial.split.train}
    for prediction in serial.predictions:
        assert set(prediction.neighbor_sources) <= training_sources
        assert prediction.source not in training_sources


def test_candidate_fallback_and_too_few_neighbors(dataset_dir):
    dataset = load_dataset(dataset_dir)
    result = classify(dataset, test_proportion=0.4, bin_width=10, search_step=0.001,
                      neighbors=99, search_span=1)
    assert all(p.used_fallback for p in result.predictions)
    assert all(len(p.neighbor_sources) == len(result.split.train) for p in result.predictions)


def test_singletons_are_training_only(dataset_dir):
    dataset = load_dataset(dataset_dir, limit_per_class=1)
    split = stratified_split(dataset)
    assert len(split.train) == 2 and not split.test
    with pytest.raises(ValueError, match="singleton"):
        classify(dataset)


def test_invalid_explicit_split(dataset_dir):
    with pytest.raises(ValueError, match="disjoint"):
        classify(load_dataset(dataset_dir), split=Split((0, 1), (1,)))


def test_correct_class_indices_in_scatter_assembly(dataset_dir):
    data = load_dataset(dataset_dir)
    first = data.records[0].features
    second = data.records[5].features
    third = replace(second, paths=tuple(replace(p, segment_length=p.segment_length * 3)
                                         for p in second.paths))
    classes = [[first], [second], [third]]
    between, within = get_distance_vec_assembly(classes, (1, 2))
    vectors = compute_dst(second, third).feature_vectors
    assert_allclose(between, vectors @ vectors.T)
    assert_allclose(within, 0)
    wrong, _ = get_distance_vec_assembly(classes, (0, 1))
    assert not np.allclose(between, wrong)


def test_scatter_is_positive_semidefinite(dataset_dir):
    data = load_dataset(dataset_dir, limit_per_class=2)
    classes = [[r.features for r in data.records if r.label == label] for label in data.classes]
    between, within = get_distance_vec_assembly(classes)
    assert_allclose(between, between.T)
    assert np.linalg.eigvalsh(between).min() > -1e-10
    for i in range(2):
        assert np.linalg.eigvalsh(within[:, :, i]).min() > -1e-10


def test_constrained_weights_are_positive_and_normalized():
    between, within = np.diag([100, 5, 4, 3, 2, 1]), np.zeros((6, 6, 2))
    fit = fit_feature_weights(between, within)
    assert np.all(fit.weights > 0)
    assert_allclose(fit.weights.sum(), 1)
    assert fit.weights[0] > fit.weights[1:].max()
    assert fit.iterations > 0
    assert_allclose(fit.weights, fit_feature_weights(between, within).weights)


def test_zero_scatter_gives_uniform_weights():
    fit = fit_feature_weights(np.zeros((6, 6)), np.zeros((6, 6, 2)))
    assert_allclose(fit.weights, np.ones(6) / 6, atol=1e-6)


def test_legacy_gradient_step():
    initial = np.arange(1, 7, dtype=float) / 21
    expected = initial + 0.0001 / (initial + 0.0001)
    expected /= expected.sum()
    fit = fit_feature_weights(np.zeros((6, 6)), np.zeros((6, 6, 2)),
                              method="legacy", initial=initial)
    assert fit.iterations == 1
    assert_allclose(fit.weights, expected)


def test_weight_iteration_limit():
    with pytest.raises(RuntimeError, match="did not converge"):
        fit_feature_weights(np.zeros((6, 6)), np.zeros((6, 6, 2)), method="legacy",
                            initial=np.arange(1, 7), max_iterations=1, tolerance=1e-15)
