# NeuroPath2Path

Python port of the workflow accompanying **“NeuroPath2Path: Classification and
elastic morphing between neuronal arbors using path-wise similarity”**, by
Tamal Batabyal, Barry Condron, and Scott T. Acton.

The port covers every computation supplied in this repository: SWC loading,
root-to-tip features, path alignment, repeated Hungarian matching, interclass
and intraclass classification, and feature importance estimation. The original
MATLAB files and data archives remain available.

The original NeuroPath2Path checkout lacks morphing code, but the authors'
earlier [ElasticPath2Path repository](https://github.com/50-Cent/ElasticPath2Path)
includes it. The Python workflow now also implements that SRVF morphing stage,
using NeuroPath2Path's path matching and a standalone interactive viewer.

## Installation

Python 3.10+ is required. Runtime dependencies are NumPy and SciPy; MATLAB is
not required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m neuropath2path --help
```

After installation, `neuropath2path ...` also works. From the repository root,
`python -m neuropath2path ...` works without installation if dependencies are
already present.

Both included ZIP archives work directly, without extraction or hard-coded
path edits. Dataset directories have one subdirectory per class and SWC files
anywhere below each class, including `class/author/CNG version/*.swc`. A single
enclosing dataset folder is detected automatically. Pass the directory that
contains all classes, rather than one author's directory.

## Elastic path morphing

```bash
python -m neuropath2path morph first.swc second.swc \
  --frames 21 --samples 128 --output results/morph.json --html results/morph.html
```

Open `results/morph.html` in a browser to play/scrub the animation, drag to
rotate, and scroll to zoom. Source and target cells remain visible beside the
current morph, with synchronized rotation/zoom and a common coordinate scale.
An optional overlay shows faint blue source and orange target outlines on the
current morph. Endpoint references are retained even for a single requested
fraction such as `times=[0.4]`. Regenerate an existing HTML file with
`write_morph_html` to use the updated viewer. The HTML is standalone and needs
no web service.
JSON contains the intermediate coordinates, morph fractions, original node
IDs, and zero-based source/target path pairs. See
[notebooks/path_morphing.ipynb](notebooks/path_morphing.ipynb) for a runnable
example using two neurons from the supplied archive.

The default `--method elasticpath2path` follows the numerical stage in the
authors' [visualMorphingNeuron.m](https://github.com/50-Cent/ElasticPath2Path/blob/7e2c6926d8c6723593df4b041e31517767a63938/visualMorphingNeuron.m):
sequential midpoint resampling (or local-length-based point deletion), vertex
gradients, SRVF interpolation, cumulative integration, and linearly
interpolated reconstruction-error corrections. The corrections recover the
resampled input curves at the first and last frames. They can allow the roots
of individual intermediate paths to drift. Although the original function
computes a Kabsch rotation, it never applies it to the morphing coordinates;
the Python implementation likewise retains orientation.

Alternatively, `--method paper` implements the equations in
[NeuroPath2Path §3.4](https://arxiv.org/pdf/1903.02147) with interval SRVFs:

```text
q(t) = f'(t) / sqrt(||f'(t)||)
q_alpha(t) = (1 - alpha) q_source(t) + alpha q_target(t)
f_alpha(t) = origin + integral_0^t q_alpha(s) ||q_alpha(s)|| ds
```

This second method uses normalized arc-length sampling and keeps roots fixed
at the origin. It integrates the interval representation exactly. It is a
separate discretization of the paper equations, not a numerical reproduction
of the earlier MATLAB tool. Neither method normalizes path lengths to unity
or estimates an optimal elastic time-warp. SRVF interpolation generally
differs from linear interpolation of Cartesian coordinates; opposite tangent
directions can cancel during morphing.

Input somata are translated to the origin. `--axis-order original` (default)
preserves coordinate axes; `--axis-order range` applies the descending-range
coordinate permutation described in Fig. 6 of NeuroPath2Path. With a finite
`--samples`, endpoints reproduce the **resampled** curves. In the Python API,
`samples=None` avoids downsampling: the earlier method upsamples to the larger
input size, while the paper method uses the union of both arc-length knot sets
to preserve every endpoint bend.

**The result is a collection of independently evolving root-to-tip paths.**
Repeated matches account for paths merging or splitting when path counts
differ. Shared internal branches need not remain coincident during the morph;
the output does not define a single branching SWC tree or interpolate radii.
It should not be treated as a connected intermediate skeleton for simulation.
If outlier reassignment omits an original path, the morph adds its cheapest
partner so both input arbors are represented; these repairs appear in
`added_pairs`. The CLI's `--mode` and `--matlab-compatible` affect path matching,
not the interpolation backend.

```python
import numpy as np
from neuropath2path import read_swc, morph_neurons, write_morph_html

morph = morph_neurons(
    read_swc("first.swc"), read_swc("second.swc"),
    times=np.linspace(0, 1, 21), samples=128, method="elasticpath2path",
)
write_morph_html(morph, "results/morph.html")
# morph.paths[k].frames has shape (time points, curve samples, 3).
# morph.pairs[k] identifies its source and target path indices.
```

For two curves already paired, use `interpolate_paths(first_xyz, second_xyz)`.
`curve_to_q`, `q_to_curve`, and `resample_path` expose the earlier tool's
numerical conventions; `curve_to_srvf` and `srvf_to_curve` expose the interval
representation.

## Classification

Original interclass experiment settings:

```bash
python -m neuropath2path classify Intercellular_data.zip \
  --mode inter --test-proportion 0.1 --bin-width 15 \
  --search-step 5 --neighbors 5 --search-span 9 \
  --seed 100 --jobs 6 --output results/inter.json
```

Original intraclass experiment settings:

```bash
python -m neuropath2path classify Intracellular_data.zip \
  --mode intra --test-proportion 0.1 --bin-width 7 \
  --search-step 5 --neighbors 5 --search-span 10 \
  --seed 100 --jobs 6 --output results/intra.json
```

Quick check on a small, deterministic subset:

```bash
python -m neuropath2path classify Intercellular_data.zip \
  --limit-per-class 4 --test-proportion 0.5 --bin-width 100000 \
  --neighbors 1 --seed 100 --output results/smoke.json
```

The subset takes the first N dendritic neurons per class in sorted file order.
Its accuracy is a smoke check, not an estimate of full-dataset performance.
The large bin width keeps each small class in a single split bin.

Each class is split into path-count bins. Training uses
`max(1, floor((1 - test_proportion) * bin_count))` neurons from each bin, so
singleton bins stay in training. Actual test proportions can differ from the
requested fraction. Candidate search expands a strict path-count window
separately for each class. If fewer than k candidates are found in total, all
training neurons are considered and k is capped at the training set size.
Test neurons never enter the candidate pool.

JSON output includes class order, the confusion matrix (true labels in rows),
accuracy as a fraction, predictions, neighbor distances and sources, fallback
flags, and train/test indices into `record_sources`. Parameters and seed are
saved. Progress goes to stderr; omit `--output` to emit JSON to stdout, or
add `--quiet` to suppress progress.

Full experiments, especially feature-weight learning, can be expensive:
each candidate comparison evaluates all pairs of dendritic paths. `--jobs`
controls concurrent test-neuron evaluations. On Slurm, activate the installed
environment and submit from the repository root:

```bash
sbatch scripts/classify_inter.slurm
sbatch scripts/classify_intra.slurm
sbatch scripts/learn_weights.slurm
```

Add your cluster's account/partition directives. An optional first script
argument overrides the dataset path.

## Features and comparison

```bash
python -m neuropath2path features neuron.swc --output results/features.json
python -m neuropath2path compare first.swc second.swc \
  --mode inter --output results/comparison.json
python -m neuropath2path extract Intercellular_data.zip \
  --output results/dataset-summary.json
```

`features` exports all per-path arrays. `extract` reports path counts, average
lengths, and maximum orders; add `--include-paths` for all arrays. Tip types
3 and 4 define dendritic paths. Axon-only neurons are recorded and skipped
when loading a dataset.

Each path stores tortuosity, bifurcation angle (radians), partition asymmetry,
concurrence, hierarchy, segment length, competition, and original SWC node IDs.
The six distance features, in coefficient order, are tortuosity, bifurcation
angle, partition asymmetry, concurrence, segment length, and competition.
Hierarchy supplies segment weights.

`compare` returns a scalar distance, a 6-by-number-of-matches feature-distance
matrix, zero-based path pairs, and the path cost matrix. Original SWC node IDs
remain unchanged; they are distinct from zero-based path indices.

| Mode | Alignment | Path cost |
| --- | --- | --- |
| `inter` | Pad at the tip | Sigmoid of a weighted sum |
| `intra` | Pad at the root | Weighted sum |
| `raw` | Pad at the tip | Euclidean norm, for learning weights |

Interclass costs have a baseline of **0.5**, including identical neurons with
the corrected calculation. Intraclass and raw identical-neuron costs are zero.
Matching repeatedly assigns the smaller arbor to unused paths in the larger
arbor, assigns the remainder, then applies the original skewness-based outlier
reassignment. These scores are not guaranteed to be mathematical metrics.

```python
from neuropath2path import read_swc, extract_features, compute_distance

first = extract_features(read_swc("first.swc"))
second = extract_features(read_swc("second.swc"))
result = compute_distance(first, second, mode="inter")
print(result.distance, result.pairs)

# Three-return unpacking is also supported:
distance, feature_vectors, path_pairs = result
```

## Feature importance

```bash
python -m neuropath2path weights Intercellular_data.zip \
  --limit-per-class 3 --seed 100 --output results/weights.json
```

Remove the limit for the full pairwise experiment. Like `jobIW.m`, this fits
a separate weight vector for every unordered class pair. Scatter matrices
sum outer products of matched raw feature-distance vectors between classes
and within each class.

The default solver minimizes the source objective,
`0.5 * w.T @ (sum(within) - between) @ w - sum(log(w + 0.0001))`, subject to
positive weights and `sum(w) == 1`. The entire objective is rescaled for
numerical stability. SLSQP finds a local solution; global optimality is not
guaranteed. `--method legacy` uses the original normalized gradient iteration
with convergence/positivity checks and an iteration limit.

`compare` and `classify` accept `--weights W1 W2 W3 W4 W5 W6`. Choose the
appropriate pair-specific fit for your experiment. Fit custom weights on
training data only when evaluating generalization; `weights` uses all supplied
neurons and does not create a train/test split.

## Compatibility and corrections

`--matlab-compatible` on `compare` or `classify` reproduces the original
inter/intraclass tortuosity formula and candidate search-span bound. The API
uses `matlab_compatible=True`. The following choices are explicit:

- The source subtracts normalized `tort2` from unnormalized `tort1` when the
  first path is at least as long as the second. Python normalizes both by
  default; compatibility mode reproduces the source asymmetry.
- The source stops candidate expansion at `search_span - 1`. Python searches
  through `search_span`; compatibility mode retains the original bound.
- Scatter assembly honors the selected class indices. The source ignores
  their values and repeatedly uses the first classes.
- Default weight fitting enforces positive, normalized weights. The legacy
  method reproduces the source iteration given identical initialization and
  reports divergence rather than looping indefinitely.
- SWC validation supports arbitrary IDs and row order and rejects missing
  parents, cycles, duplicate IDs, and multiple roots. Segments follow tree
  topology rather than sorted numeric IDs.
- Coordinates are centered at the root. Constant dimensions standardize to
  zero; angle inputs are clipped before `acos`; coincident child edges
  contribute zero angle. Zero-length segments have tortuosity 1; closed
  nonzero-length segments use a machine-epsilon chord floor. Near-zero length
  normalization denominators use a signed `1e-12` floor. The source can
  generate NaN/Inf in these cases.
- Splitting retains all neurons, including final bin edges and bins whose
  training count would round to zero. Candidate shortages use the fallback
  above instead of indexing past available neighbors.

Compatibility mode does **not** promise identical experiment accuracy:
NumPy and MATLAB use different random streams, histogram-edge handling is
explicit in Python, and SciPy may choose different correspondences when
Hungarian assignments have exactly equal costs. Unsafe-input bugs remain
corrected in compatibility mode.

## Validation

```bash
python -m pytest -q
```

Tests cover hand-calculated trees, malformed SWCs, deep trees, planar arbors,
directory/ZIP loading, scalar versus vectorized alignment, partial/repeated
assignment, reproducible classification, split isolation, and both weight
solvers.

Frozen fixtures were generated by running the unmodified repository MATLAB
functions in GNU Octave 11.1.0 on four real neurons from both supplied archives.
Every path feature agrees within about `3e-13`; all six checked distances agree
within `6e-16`, with identical path pairs. The legacy weight iteration is also
checked with identical initialization. Octave requires a small integer-degree
`histcounts` shim, included only in the fixture generator. Octave is not needed
to run the tests. To regenerate fixtures:

```bash
python scripts/generate_reference.py
python -m pytest -q
```

Morphing has an additional independent fixture generated from the earlier
ElasticPath2Path code at commit `7e2c6926d8c6723593df4b041e31517767a63938`.
Resampling, SRVFs, and intermediate frames are checked on synthetic and real
paths, including upsampling and downsampling. The reference executes the
unmodified `curve_to_q.m`, `q_to_curve.m`, and `pathresample.m` in Octave and
assembles the numerical frame equations from `visualMorphingNeuron.m`, without
its MATLAB video/plotting calls. To regenerate it:

```bash
python scripts/generate_morph_reference.py
```

This optional generator downloads the pinned source files; an existing local
checkout can be supplied with `--source-directory`. Normal use and tests are
offline. The interval-based `paper` backend has analytic SRVF/inverse and
endpoint tests, rather than a claim of parity with the earlier implementation.

Subset smoke runs also exercise both classification modes and all ten
intercellular class-pair weight fits. Feature extraction completed on every
SWC in both archives:

| Archive | SWC files | Dendritic neurons retained | Axon-only files skipped |
| --- | ---: | ---: | ---: |
| `Intercellular_data.zip` | 1,887 | 1,786 | 101 |
| `Intracellular_data.zip` | 625 | 625 | 0 |

Package wheel installation and the installed command-line entry point were
also checked. Full-dataset classification and full importance-weight
experiments have not been run as validation.

## MATLAB to Python map

| Original | Python equivalent |
| --- | --- |
| `readInput.m` | `read_input`, `read_swc`, `iter_dataset` |
| `computenP2P.m` | `extract_features`, four-return `compute_np2p` |
| `pathConcurrence.m` | `path_concurrence` (zero-based row index) |
| `computeDST.m` | `compute_dst` / `compute_distance(mode="raw")` |
| `computeDSTInter.m` | `compute_dst_inter` / `compute_distance(mode="inter")` |
| `computeDSTIntra.m` | `compute_dst_intra` / `compute_distance(mode="intra")` |
| `distanceNeuronNew.m` | `distance_neuron_new`, `match_paths` |
| `munkres.m` | `munkres` (zero-based columns; -1 means unassigned) |
| `neuroClassifyInterclass.m` | `neuro_classify_interclass`, CLI `classify --mode inter` |
| `neuroClassifyIntraclass.m` | `neuro_classify_intraclass`, CLI `classify --mode intra` |
| `getDistanceVecAssembly.m` | `get_distance_vec_assembly` (zero-based class indices) |
| `getFeatureWeights.m` | `get_feature_weights`, `fit_feature_weights` |
| `jobIW.m` | `learn_dataset_weights`, CLI `weights` |
| Classification `.slurm` files | Python scripts in `scripts/` |

The original MATLAB instructions remain in [ReadMe.txt](ReadMe.txt).
The project retains its existing [GPL license](LICENSE).
