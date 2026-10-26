"""Python implementation of the NeuroPath2Path MATLAB workflow."""

from .classification import (
    ClassificationResult, Dataset, NeuronRecord, Prediction, Split, classify,
    load_dataset, neuro_classify_interclass, neuro_classify_intraclass, stratified_split,
)
from .distance import (
    INTER_WEIGHTS, INTRA_WEIGHTS, DistanceResult, compute_distance, compute_dst,
    compute_dst_inter, compute_dst_intra, distance_neuron_new, match_paths,
    munkres, path_distances,
)
from .features import (
    FEATURE_NAMES, NeuronFeatures, PathFeatures, compute_np2p, extract_features,
    path_concurrence,
)
from .io import Morphology, iter_dataset, morphology_from_array, read_input, read_swc
from .morphing import (
    NeuronMorph, PathMorph, curve_to_q, curve_to_srvf, interpolate_paths,
    morph_neurons, q_to_curve, resample_path, srvf_to_curve, write_morph_html,
)
from .weights import (
    WeightFit, fit_feature_weights, get_distance_vec_assembly, get_feature_weights,
    learn_dataset_weights,
)

__version__ = "0.1.0"

__all__ = [
    "__version__", "FEATURE_NAMES", "INTER_WEIGHTS", "INTRA_WEIGHTS", "Morphology",
    "NeuronFeatures", "PathFeatures", "DistanceResult", "ClassificationResult", "Dataset",
    "NeuronRecord", "Prediction", "Split", "WeightFit", "classify", "compute_distance",
    "compute_dst", "compute_dst_inter", "compute_dst_intra", "compute_np2p",
    "distance_neuron_new", "extract_features", "fit_feature_weights",
    "get_distance_vec_assembly", "get_feature_weights", "iter_dataset",
    "learn_dataset_weights", "load_dataset", "match_paths", "morphology_from_array",
    "munkres", "neuro_classify_interclass", "neuro_classify_intraclass", "path_concurrence",
    "path_distances", "read_input", "read_swc", "stratified_split",
    "NeuronMorph", "PathMorph", "curve_to_q", "curve_to_srvf", "interpolate_paths",
    "morph_neurons", "q_to_curve", "resample_path", "srvf_to_curve", "write_morph_html",
]
