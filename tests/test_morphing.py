from io import StringIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from neuropath2path import (
    curve_to_q, curve_to_srvf, interpolate_paths, morph_neurons, q_to_curve,
    read_swc, resample_path, srvf_to_curve, write_morph_html,
)
from neuropath2path.cli import main


@pytest.mark.parametrize("curve", [
    [[0, 0, 0], [1, 0, 0]],
    [[4, 2, 3], [5, 3, 1], [5, 3, 1], [9, 8, 7]],
    [[0, 0], [0, 0], [0, 0]],
])
def test_interval_srvf_roundtrip(curve):
    curve = np.array(curve, dtype=float)
    parameter = np.linspace(0, 1, len(curve)) ** 2
    q = curve_to_srvf(curve, parameter)
    assert q.shape == (len(curve) - 1, curve.shape[1])
    assert_allclose(srvf_to_curve(q, parameter, curve[0]), curve, atol=1e-13)


def test_paper_interpolation_is_linear_in_srvf_not_coordinates():
    first, second = np.array([[0, 0], [1, 0]]), np.array([[0, 0], [4, 0]])
    morph = interpolate_paths(first, second, times=[0, 0.5, 1], method="paper")
    assert_allclose(morph.frames[:, -1, 0], [1, 2.25, 4])
    assert_allclose(morph.frames[:, :, 1], 0)
    halfway_q = curve_to_srvf(morph.frames[1], morph.parameter)
    assert_allclose(halfway_q, (curve_to_srvf(first) + curve_to_srvf(second)) / 2)


@pytest.mark.parametrize("method", ["paper", "elasticpath2path"])
def test_morph_endpoints_and_identical_curves(method):
    a = np.array([[3, 4, 1], [4, 5, 3], [8, 2, 4]], dtype=float)
    b = np.array([[6, 1, 2], [7, 3, 5], [4, 5, 6]], dtype=float)
    result = interpolate_paths(a, b, times=[0, 1], method=method)
    if method == "elasticpath2path":
        assert_allclose(result.frames[0], a, atol=1e-12)
        assert_allclose(result.frames[-1], b, atol=1e-12)
    else:
        for curve, endpoint in zip((a, b), result.frames):
            knots = np.r_[0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
            knots /= knots[-1]
            expected = np.column_stack([np.interp(result.parameter, knots, curve[:, d])
                                         for d in range(3)])
            assert_allclose(endpoint, expected, atol=1e-12)
    identical = interpolate_paths(a, a, times=[0, 0.3, 0.7, 1], method=method)
    for frame in identical.frames:
        assert_allclose(frame, identical.frames[0], atol=1e-12)


def test_stationary_to_nonstationary_path():
    a, b = np.array([[1, 2], [1, 2]]), np.array([[5, 6], [7, 6]])
    result = interpolate_paths(a, b, times=[0, 0.5, 1], center=True, method="paper")
    assert_allclose(result.frames[:, 0], 0)
    assert_allclose(result.frames[:, -1, 0], [0, 0.5, 2])


def test_opposed_directions_may_collapse_without_rotation_optimization():
    a, b = np.array([[0, 0], [1, 0]]), np.array([[0, 0], [-1, 0]])
    result = interpolate_paths(a, b, times=[0.5], method="paper")
    assert_allclose(result.frames, 0)


def test_midpoint_resampling_preserves_order_and_original_vertices():
    curve = np.array([[0, 0], [2, 2], [4, 0]], dtype=float)
    assert_allclose(resample_path(curve, 6),
                    [[0, 0], [0.5, 0.5], [1, 1], [2, 2], [3, 1], [4, 0]])
    assert_allclose(resample_path(curve, 2), curve[[0, -1]])
    assert_allclose(resample_path(curve, 3), curve)


def test_original_derivative_and_cumsum_convention():
    curve = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    q = curve_to_q(curve, 0.5)
    assert_allclose(q, [[np.sqrt(2), 0, 0]] * 3)
    assert_allclose(q_to_curve(q, 0.5), [[1, 0, 0], [2, 0, 0], [3, 0, 0]])


@pytest.mark.parametrize("method", ["paper", "elasticpath2path"])
def test_neuron_morph_covers_both_arbors_and_keeps_input_order(branched_tree, method):
    extra = np.vstack((branched_tree.nodes, [5, 3, 3, 2, 1, 1, 2]))
    result = morph_neurons(branched_tree, extra, samples=16, times=[0, 0.5, 1], method=method)
    assert sorted(set(result.pairs[:, 0])) == [0, 1]
    assert sorted(set(result.pairs[:, 1])) == [0, 1, 2]
    assert all(p.frames.shape == (3, 16, 3) for p in result.paths)
    assert len(result.source_node_ids) == 2 and len(result.target_node_ids) == 3
    for p in result.paths:
        assert_allclose(p.frames[[0, -1], 0], 0, atol=1e-12)
        assert np.isfinite(p.frames).all()


def test_paper_roots_stay_centered_and_axis_permutation_is_explicit(branched_tree):
    shifted = branched_tree.nodes.copy()
    shifted[:, 2:5] += [10, 20, 30]
    result = morph_neurons(branched_tree, shifted, axis_order="range", method="paper", times=[0, .5, 1])
    assert_array_equal(result.source_axis_order, [0, 1, 2])
    assert_allclose(result.target_root, [10, 20, 30])
    for path in result.paths:
        assert_allclose(path.frames[:, 0], 0, atol=1e-12)


@pytest.mark.parametrize("kwargs", [
    {"times": []}, {"times": [-0.1]}, {"times": [1.1]}, {"times": [np.nan]},
    {"samples": 1}, {"samples": 2.5}, {"method": "invalid"},
])
def test_invalid_interpolation_parameters(kwargs):
    with pytest.raises(ValueError):
        interpolate_paths([[0, 0], [1, 1]], [[0, 0], [1, 0]], **kwargs)


def test_neuron_with_no_dendrites_fails(branched_tree):
    axon = branched_tree.nodes.copy()
    axon[1:, 1] = 2
    with pytest.raises(ValueError, match="dendritic path"):
        morph_neurons(axon, branched_tree)


def test_standalone_html_and_cli(dataset_dir, tmp_path):
    source, target = sorted(dataset_dir.rglob("*.swc"))[:2]
    output, html = tmp_path / "morph.json", tmp_path / "morph.html"
    assert main(["morph", str(source), str(target), "--frames", "3", "--samples", "8",
                 "--output", str(output), "--html", str(html)]) == 0
    result = json.loads(output.read_text())["result"]
    assert result["method"] == "elasticpath2path"
    assert len(result["paths"][0]["frames"]) == 3
    content = html.read_text()
    assert "<canvas" in content and "const data=" in content and "__MORPH_DATA__" not in content
    assert "src=\"http" not in content


@pytest.mark.parametrize("method", ["paper", "elasticpath2path"])
def test_real_neuron_morph(method):
    root = Path(__file__).resolve().parents[1]
    with ZipFile(root / "Intercellular_data.zip") as archive:
        files = sorted(n for n in archive.namelist() if "/Granule/" in n and n.endswith(".swc"))[:2]
        neurons = [read_swc(StringIO(archive.read(n).decode("utf-8-sig"))) for n in files]
    result = morph_neurons(*neurons, samples=32, times=[0, 0.5, 1], method=method)
    assert all(np.isfinite(path.frames).all() for path in result.paths)
    assert len(np.unique(result.pairs[:, 0])) == len(result.source_node_ids)
    assert len(np.unique(result.pairs[:, 1])) == len(result.target_node_ids)


@pytest.mark.parametrize("case", range(3))
def test_original_elasticpath2path_intermediate_frames(case):
    source = Path(__file__).parent / "data" / "elasticpath2path_reference.npz"
    with np.load(source, allow_pickle=False) as reference:
        samples = int(reference["sample_counts"][case])
        times = reference["times"]
        first, second = reference[f"first_{case}"], reference[f"second_{case}"]
        sampled_first, sampled_second = resample_path(first, samples), resample_path(second, samples)
        assert_allclose(sampled_first, reference[f"sampled_first_{case}"], atol=1e-12)
        assert_allclose(sampled_second, reference[f"sampled_second_{case}"], atol=1e-12)
        assert_allclose(curve_to_q(sampled_first, 1 / (samples - 1)),
                        reference[f"q_first_{case}"], atol=1e-12)
        assert_allclose(curve_to_q(sampled_second, 1 / (samples - 1)),
                        reference[f"q_second_{case}"], atol=1e-12)
        result = interpolate_paths(first, second, times=times, samples=samples,
                                   method="elasticpath2path")
        assert_allclose(result.frames, reference[f"frames_{case}"], atol=1e-11, rtol=1e-12)


@pytest.mark.parametrize("method", ["paper", "elasticpath2path"])
@pytest.mark.parametrize("times", [[0.4], [0.8, 0.2], [1, 0]])
def test_viewer_keeps_true_endpoints_without_endpoint_frames(branched_tree, tmp_path, method, times):
    target = branched_tree.nodes.copy()
    target[:, 2:5] *= [3, 2, 1]
    # Three target paths matched to two source paths also tests reference deduplication.
    target = np.vstack((target, [5, 3, 4, 3, 1, 1, 2]))
    morph = morph_neurons(branched_tree, target, times=times, samples=16, method=method)
    endpoints = morph_neurons(branched_tree, target, times=[0, 1], samples=16, method=method)
    html = write_morph_html(morph, tmp_path / "viewer.html").read_text()
    payload = json.loads(html.split("const data=", 1)[1].split(";\n", 1)[0])
    assert payload["times"] == times
    assert len(payload["source"]) == len(morph.source_node_ids) == 2
    assert len(payload["target"]) == len(morph.target_node_ids) == 3
    for column, name, endpoint_frame in [(0, "source", 0), (1, "target", 1)]:
        indices = list(dict.fromkeys(morph.pairs[:, column]))
        for curve, path_index in zip(payload[name], indices):
            pair_index = np.flatnonzero(morph.pairs[:, column] == path_index)[0]
            assert_allclose(curve, endpoints.paths[pair_index].frames[endpoint_frame], atol=1e-12)
    assert all(len(frames) == len(times) for frames in payload["paths"])
