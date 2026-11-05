from io import StringIO
import json

import numpy as np
import pytest
from numpy.testing import assert_allclose

from neuropath2path import extract_features, morph_neurons, read_swc, write_morph_html
from neuropath2path.cli import main


@pytest.fixture
def mixed_tree():
    return read_swc(StringIO("""1 1 0 0 0 1 -1
2 2 -1 0 0 1 1
3 2 -2 1 0 1 2
4 2 -2 -1 0 1 2
5 3 1 0 0 1 1
6 3 2 1 0 1 5
7 4 2 -1 0 1 5
"""))


@pytest.mark.parametrize("arbor, tips", [("dendrite", {6, 7}), ("axon", {3, 4}),
                                         ("both", {3, 4, 6, 7})])
def test_selected_terminal_types(mixed_tree, arbor, tips):
    features = extract_features(mixed_tree, arbor=arbor)
    assert {int(path.node_ids[-1]) for path in features.paths} == tips
    assert all(path.node_ids[0] == 1 for path in features.paths)
    assert all(np.isfinite(path.values).all() for path in features.paths)


@pytest.mark.parametrize("method", ["paper", "elasticpath2path"])
@pytest.mark.parametrize("arbor", ["axon", "both"])
def test_axon_pairs_keep_compartment_and_cover_unequal_arbors(mixed_tree, method, arbor, tmp_path):
    target = np.vstack((mixed_tree.nodes, [8, 2, -3, 0, 1, 1, 2]))
    # Swap the spatial sides, so geometry alone would favor cross-compartment matches.
    target[:, 2] *= -1
    morph = morph_neurons(mixed_tree, target, arbor=arbor, method=method,
                          times=[0, .5, 1], samples=16)
    lookup_a = {int(row[0]): row for row in mixed_tree.nodes}
    lookup_b = {int(row[0]): row for row in target}
    assert set(morph.path_kinds) == ({"axon", "dendrite"} if arbor == "both" else {"axon"})
    for (i, j), path, kind in zip(morph.pairs, morph.paths, morph.path_kinds):
        a = lookup_a[int(morph.source_node_ids[i][-1])]
        b = lookup_b[int(morph.target_node_ids[j][-1])]
        assert (a[1] == 2) == (b[1] == 2) == (kind == "axon")
        assert_allclose(path.frames[0, -1], a[2:5], atol=1e-12)
        assert_allclose(path.frames[-1, -1], b[2:5], atol=1e-12)
    assert len(np.unique(morph.pairs[:, 0])) == len(morph.source_node_ids)
    assert len(np.unique(morph.pairs[:, 1])) == len(morph.target_node_ids)
    assert len(morph.paths) == (5 if arbor == "both" else 3)
    html = write_morph_html(morph, tmp_path / "mixed.html").read_text()
    payload = json.loads(html.split("const data=", 1)[1].split(";\n", 1)[0])
    assert payload["arbor"] == arbor
    assert payload["path_kinds"] == list(morph.path_kinds)


@pytest.mark.parametrize("arbor", ["axon", "both"])
def test_axon_only_morphologies_work(mixed_tree, arbor):
    axon = mixed_tree.nodes.copy()
    axon[axon[:, 1] != 1, 1] = 2
    morph = morph_neurons(axon, axon, arbor=arbor, samples=8, times=[0.4])
    assert len(morph.paths) == 4
    assert set(morph.path_kinds) == {"axon"}


@pytest.mark.parametrize("arbor", ["axon", "both"])
def test_missing_axon_in_one_input_has_clear_error(mixed_tree, arbor):
    dendrite_only = mixed_tree.nodes.copy()
    dendrite_only[dendrite_only[:, 1] == 2, 1] = 3
    with pytest.raises(ValueError, match=r"axonal paths.*source=0, target=2"):
        morph_neurons(dendrite_only, mixed_tree, arbor=arbor)


def test_dendrite_only_both_mode_skips_absent_axon_group(branched_tree):
    morph = morph_neurons(branched_tree, branched_tree, arbor="both", times=[0.4])
    assert len(morph.paths) == 2
    assert set(morph.path_kinds) == {"dendrite"}


def test_cli_axon_selection(mixed_tree, tmp_path):
    source, target, output = [tmp_path / name for name in ("source.swc", "target.swc", "axon.json")]
    np.savetxt(source, mixed_tree.nodes)
    np.savetxt(target, mixed_tree.nodes)
    assert main(["morph", str(source), str(target), "--arbor", "axon", "--frames", "3",
                 "--samples", "8", "--output", str(output)]) == 0
    result = json.loads(output.read_text())["result"]
    assert result["arbor"] == "axon"
    assert result["path_kinds"] == ["axon", "axon"]


def test_unknown_arbor_is_rejected(mixed_tree):
    with pytest.raises(ValueError, match="arbor must be"):
        morph_neurons(mixed_tree, mixed_tree, arbor="unknown")
