from io import StringIO
from zipfile import ZipFile

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from neuropath2path import (
    compute_np2p, extract_features, iter_dataset, load_dataset,
    morphology_from_array, path_concurrence, read_input, read_swc,
)


def test_hand_calculated_features(branched_tree):
    result = extract_features(branched_tree)
    assert result.path_count == 2
    assert result.max_order == 2
    assert_allclose(result.average_path_length, 1 + np.sqrt(2))
    for i, path in enumerate(result.paths):
        assert_array_equal(path.tortuosity, [1, 1])
        assert_allclose(path.bifurcation_angle, [0, np.pi / 2])
        assert_array_equal(path.partition_asymmetry, [0, 0])
        assert_array_equal(path.concurrence, [2, 2])
        assert_array_equal(path.hierarchy, [1, 2])
        assert_allclose(path.segment_length, [1, np.sqrt(2)])
        assert_array_equal(path.competition, [0, 0])
        assert_array_equal(path.node_ids, [1, 2, i + 3])
        assert np.isfinite(path.values).all()
    paths, count, mean, order = compute_np2p(branched_tree)
    assert len(paths) == count == 2
    assert mean == result.average_path_length and order == 2


def test_arbitrary_ids_and_row_order(branched_tree):
    original = extract_features(branched_tree)
    data = branched_tree.nodes.copy()
    mapping = {1: 42, 2: 8, 3: 99, 4: 11, -1: -1}
    data[:, 0] = [mapping[int(i)] for i in data[:, 0]]
    data[:, 6] = [mapping[int(i)] for i in data[:, 6]]
    data = data[[2, 0, 3, 1]]
    result = extract_features(data)
    assert result.path_count == original.path_count
    for actual, expected in zip(result.paths, original.paths):
        assert_allclose(actual.values, expected.values)
    assert_array_equal(result.paths[0].node_ids, [42, 8, 99])


def test_translation_invariance(branched_tree):
    shifted = branched_tree.nodes.copy()
    shifted[:, 2:5] += [100, -40, 31]
    a, b = extract_features(branched_tree), extract_features(shifted)
    for left, right in zip(a.paths, b.paths):
        assert_allclose(left.values, right.values)


def test_tortuosity_on_bent_unbranched_path():
    data = np.array([[1, 1, 0, 0, 0, 1, -1], [2, 3, 1, 0, 0, 1, 1],
                     [3, 3, 1, 1, 0, 1, 2]])
    features = extract_features(data)
    assert_allclose(features.paths[0].tortuosity, [np.sqrt(2)])
    assert_allclose(features.paths[0].segment_length, [2])


def test_asymmetry_and_multifurcation():
    data = np.array([[1, 1, 0, 0, 0, 1, -1], [2, 3, 1, 0, 0, 1, 1],
                     [3, 3, 2, 1, 0, 1, 2], [4, 3, 2, -1, 0, 1, 2],
                     [5, 3, 3, 0, 1, 1, 3], [6, 3, 3, 2, -1, 1, 3],
                     [7, 3, 3, 1, 0, 1, 3]])
    features = extract_features(data)
    assert features.path_count == 4
    assert_allclose(features.paths[0].partition_asymmetry[1], 0.5)
    assert_array_equal(features.paths[0].concurrence, [4, 4])


def test_axon_only_and_root_only(branched_tree):
    axon = branched_tree.nodes.copy()
    axon[1:, 1] = 2
    assert extract_features(axon).path_count == 0
    assert extract_features(branched_tree.nodes[:1]).path_count == 0


def test_deep_tree_avoids_recursion():
    n = 3000
    data = np.zeros((n, 7))
    data[:, 0] = np.arange(1, n + 1)
    data[:, 1] = 3
    data[:, 2] = np.arange(n)
    data[:, 5] = 1
    data[:, 6] = np.arange(n)
    data[0, 6] = -1
    features = extract_features(data)
    assert features.path_count == 1
    assert features.average_path_length == n - 1


def test_ancestor_concurrence(branched_tree):
    counts = np.zeros(4, dtype=int)
    path_concurrence(branched_tree, counts, 2)
    assert_array_equal(counts, [1, 1, 0, 0])


@pytest.mark.parametrize("mutation, message", [
    (lambda x: x.__setitem__((1, 0), 1), "unique"),
    (lambda x: x.__setitem__((1, 6), 99), "missing parent"),
    (lambda x: x.__setitem__((1, 6), 2), "own parent"),
    (lambda x: x.__setitem__((1, 6), -1), "exactly one root"),
    (lambda x: x.__setitem__((0, 6), 2), "exactly one root"),
    (lambda x: x.__setitem__((1, 0), 2.5), "integers"),
    (lambda x: x.__setitem__((1, 5), -1), "radii"),
    (lambda x: x.__setitem__((1, 2), np.nan), "nonfinite"),
    (lambda x: x.__setitem__((1, 6), 3), "cycle"),
])
def test_invalid_trees(branched_tree, mutation, message):
    data = branched_tree.nodes.copy()
    mutation(data)
    with pytest.raises(ValueError, match=message):
        morphology_from_array(data)


def test_invalid_columns():
    with pytest.raises(ValueError, match="N x 7"):
        read_swc(StringIO("1 2 3\n"))


def test_dataset_directory_and_zip(dataset_dir, tmp_path):
    files = list(iter_dataset(dataset_dir))
    assert len(files) == 10
    assert {label for label, _, _ in files} == {"alpha", "beta"}
    assert len(read_input(dataset_dir / "alpha")) == 5
    archive_path = tmp_path / "data.zip"
    with ZipFile(archive_path, "w") as archive:
        for file in dataset_dir.rglob("*.swc"):
            archive.write(file, "wrapper/" + str(file.relative_to(dataset_dir)))
        archive.writestr("__MACOSX/junk.swc", "ignore")
        archive.writestr("wrapper/.hidden/junk.swc", "ignore")
    zipped = list(iter_dataset(archive_path))
    assert [label for label, _, _ in zipped] == [label for label, _, _ in files]
    for (_, _, a), (_, _, b) in zip(files, zipped):
        assert_array_equal(a.nodes, b.nodes)
    dataset = load_dataset(archive_path, limit_per_class=2)
    assert len(dataset.records) == 4


def test_dataset_axon_skip(dataset_dir):
    file = dataset_dir / "alpha" / "axon.swc"
    file.write_text("1 1 0 0 0 1 -1\n2 2 1 0 0 1 1\n")
    result = load_dataset(dataset_dir)
    assert len(result.records) == 10
    assert result.skipped == (str(file),)


def test_empty_dataset(tmp_path):
    with pytest.raises(ValueError, match="No SWC"):
        list(iter_dataset(tmp_path))
