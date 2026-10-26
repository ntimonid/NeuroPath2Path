"""Golden results produced by the original MATLAB code under GNU Octave."""

from io import StringIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from neuropath2path import compute_distance, extract_features, fit_feature_weights, read_swc


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def reference():
    with np.load(DATA / "octave_reference.npz", allow_pickle=False) as archive:
        yield archive


@pytest.fixture(scope="module")
def neurons():
    metadata = json.loads((DATA / "reference_metadata.json").read_text())
    result = []
    for entry in metadata["neurons"]:
        with ZipFile(ROOT / entry["archive"]) as archive:
            tree = read_swc(StringIO(archive.read(entry["member"]).decode("utf-8-sig")))
            result.append(extract_features(tree))
    return result


@pytest.mark.parametrize("index", range(4))
def test_all_real_neuron_features(neurons, reference, index):
    neuron = neurons[index]
    assert_allclose([neuron.path_count, neuron.average_path_length, neuron.max_order],
                    reference[f"summary_{index}"], atol=1e-10, rtol=1e-12)
    offsets, node_offsets = reference[f"offsets_{index}"], reference[f"node_offsets_{index}"]
    for i, path in enumerate(neuron.paths):
        actual = np.column_stack([path.tortuosity, path.bifurcation_angle,
                                  path.partition_asymmetry, path.concurrence, path.hierarchy,
                                  path.segment_length, path.competition])
        expected = reference[f"features_{index}"][offsets[i]:offsets[i + 1]]
        assert_allclose(actual, expected, atol=1e-10, rtol=1e-12)
        assert_array_equal(path.node_ids,
                           reference[f"nodes_{index}"][node_offsets[i]:node_offsets[i + 1]])


@pytest.mark.parametrize("pair", [0, 1])
@pytest.mark.parametrize("mode", ["raw", "inter", "intra"])
def test_distances_and_correspondence(neurons, reference, pair, mode):
    result = compute_distance(neurons[2 * pair], neurons[2 * pair + 1], mode=mode,
                              matlab_compatible=True)
    assert_allclose(result.distance, reference[f"distance_{pair}_{mode}"].item(),
                    atol=1e-12, rtol=1e-12)
    assert_allclose(result.feature_vectors, reference[f"vectors_{pair}_{mode}"],
                    atol=1e-10, rtol=1e-12)
    assert_array_equal(result.pairs, reference[f"pairs_{pair}_{mode}"])


def test_legacy_weights_with_identical_initialization(reference):
    fit = fit_feature_weights(reference["weight_between"], reference["weight_within"],
                              method="legacy", initial=reference["weight_initial"].ravel())
    assert_allclose(fit.weights, reference["weight_result"].ravel(), atol=1e-14)
