from io import StringIO

import numpy as np
import pytest

from neuropath2path import read_swc


@pytest.fixture
def branched_tree():
    return read_swc(StringIO("""# planar dendrite with two tips
1 1 0 0 0 1 -1
2 3 1 0 0 1 1
3 3 2 1 0 1 2
4 4 2 -1 0 1 2
"""))


@pytest.fixture
def dataset_dir(tmp_path):
    for label, branches in [("alpha", 2), ("beta", 3)]:
        folder = tmp_path / label / "archive" / "CNG version"
        folder.mkdir(parents=True)
        for sample in range(5):
            rows = [[1, 1, 0, 0, 0, 1, -1], [2, 3, 1, 0, 0, 1, 1]]
            for i in range(branches):
                rows.append([i + 3, 3, 2 + sample * 0.02, i - 1, i % 2, 1, 2])
            np.savetxt(folder / f"neuron_{sample}.swc", rows, fmt="%.8g")
    return tmp_path
