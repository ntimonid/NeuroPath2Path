"""Path-count-stratified evaluation with per-class candidate search and k-NN."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Callable

import numpy as np

from .distance import compute_distance
from .features import NeuronFeatures, extract_features
from .io import iter_dataset


@dataclass(frozen=True)
class NeuronRecord:
    label: str
    source: str
    features: NeuronFeatures


@dataclass(frozen=True)
class Dataset:
    records: tuple[NeuronRecord, ...]
    skipped: tuple[str, ...] = ()

    @property
    def classes(self) -> tuple[str, ...]:
        return tuple(sorted({record.label for record in self.records}))


@dataclass(frozen=True)
class Split:
    train: tuple[int, ...]
    test: tuple[int, ...]


@dataclass(frozen=True)
class Prediction:
    source: str
    true_label: str
    predicted_label: str
    neighbor_sources: tuple[str, ...]
    neighbor_labels: tuple[str, ...]
    neighbor_distances: tuple[float, ...]
    used_fallback: bool


@dataclass(frozen=True)
class ClassificationResult:
    classes: tuple[str, ...]
    confusion_matrix: np.ndarray
    accuracy: float  # fraction in [0, 1]
    predictions: tuple[Prediction, ...]
    split: Split
    seed: int


def load_dataset(path: str | Path, *, limit_per_class: int | None = None,
                 progress: Callable[[str], None] | None = None) -> Dataset:
    """Extract dataset features; explicitly record axon-only files that are skipped."""
    if limit_per_class is not None and limit_per_class < 1:
        raise ValueError("limit_per_class must be positive")
    records, skipped = [], []
    seen: set[str] = set()
    for label, group in groupby(iter_dataset(path), key=lambda item: item[0]):
        seen.add(label)
        count = 0
        for _, source, tree in group:
            if limit_per_class is not None and count >= limit_per_class:
                continue
            features = extract_features(tree)
            if not features.path_count:
                skipped.append(source)
                continue
            records.append(NeuronRecord(label, source, features))
            count += 1
        if progress:
            progress(f"{label}: extracted {count} dendritic neurons")
    missing = seen - {record.label for record in records}
    if missing:
        raise ValueError(f"Classes contain no dendritic neurons: {', '.join(sorted(missing))}")
    if not records:
        raise ValueError("Dataset contains no dendritic neurons")
    return Dataset(tuple(records), tuple(skipped))


def stratified_split(dataset: Dataset, *, test_proportion: float = 0.1,
                     bin_width: float = 15, seed: int = 100) -> Split:
    """Split each class by path-count bins; singleton bins stay in training.

    Bin edges are integer multiples of bin_width. floor((1-test_prop)*count)
    selects training members, with at least one member per nonempty bin.
    No sample is dropped at the last edge or when the training fraction is small.
    """
    if not 0 < test_proportion < 1:
        raise ValueError("test_proportion must be between 0 and 1")
    if not np.isfinite(bin_width) or bin_width <= 0:
        raise ValueError("bin_width must be finite and positive")
    rng = np.random.default_rng(seed)
    train, test = [], []
    for label in dataset.classes:
        bins: dict[int, list[int]] = {}
        for index, record in enumerate(dataset.records):
            if record.label == label:
                key = int(np.floor(record.features.path_count / bin_width))
                bins.setdefault(key, []).append(index)
        for key in sorted(bins):
            members = rng.permutation(bins[key]).tolist()
            n_train = max(1, int(np.floor((1 - test_proportion) * len(members))))
            train.extend(members[:n_train])
            test.extend(members[n_train:])
    return Split(tuple(train), tuple(test))


def _predict(target: NeuronRecord, training: tuple[NeuronRecord, ...], *,
             mode: str, search_step: float, neighbors: int, search_span: int,
             weights: np.ndarray | None, matlab_compatible: bool) -> Prediction:
    candidates = []
    count = target.features.path_count
    for _, group in groupby(training, key=lambda record: record.label):
        records = list(group)
        selected = []
        max_span = max(1, search_span - 1) if matlab_compatible else search_span
        for factor in range(1, max_span + 1):
            selected = [r for r in records
                        if abs(r.features.path_count - count) < factor * search_step]
            if len(selected) >= neighbors:
                break
        candidates.extend(selected)
    fallback = len(candidates) < neighbors
    if fallback:
        # The original code indexes past the available neighbors in this case.
        candidates = list(training)
    ranked = []
    for record in candidates:
        distance = compute_distance(record.features, target.features, mode=mode,
                                    weights=weights,
                                    matlab_compatible=matlab_compatible).distance
        ranked.append((distance, record.label, record.source))
    ranked.sort(key=lambda item: item[0])  # stable: class/source order breaks equal costs
    nearest = ranked[:min(neighbors, len(ranked))]
    labels, counts = np.unique([item[1] for item in nearest], return_counts=True)
    predicted = str(labels[np.argmax(counts)])  # MATLAB mode: smallest class wins a tie
    return Prediction(target.source, target.label, predicted,
                      tuple(item[2] for item in nearest), tuple(item[1] for item in nearest),
                      tuple(item[0] for item in nearest), fallback)


def classify(dataset: Dataset, *, mode: str = "inter", test_proportion: float = 0.1,
             bin_width: float = 15, search_step: float = 5, neighbors: int = 5,
             search_span: int = 9, seed: int = 100, jobs: int = 1,
             weights: np.ndarray | None = None, matlab_compatible: bool = False,
             split: Split | None = None,
             progress: Callable[[str], None] | None = None) -> ClassificationResult:
    if mode not in {"inter", "intra"}:
        raise ValueError("Classification mode must be 'inter' or 'intra'")
    if not np.isfinite(search_step) or search_step <= 0:
        raise ValueError("search_step must be finite and positive")
    if any(not isinstance(v, (int, np.integer)) or v < 1
           for v in (neighbors, search_span, jobs)):
        raise ValueError("neighbors, search_span, and jobs must be positive integers")
    split = split if split is not None else stratified_split(
        dataset, test_proportion=test_proportion, bin_width=bin_width, seed=seed)
    train, test = split.train, split.test
    if not train or not test:
        raise ValueError("Split needs training and test neurons; singleton path-count bins "
                         "stay in training. Use more data or a larger bin_width.")
    if (any(i < 0 or i >= len(dataset.records) for i in (*train, *test))
            or len(set(train)) != len(train) or len(set(test)) != len(test)
            or set(train) & set(test)):
        raise ValueError("Split indices must be valid, unique, and disjoint")
    training = tuple(sorted((dataset.records[i] for i in train),
                            key=lambda r: (r.label, r.source)))
    if {record.label for record in training} != set(dataset.classes):
        raise ValueError("Training split must represent every class")

    def predict(index: int) -> Prediction:
        return _predict(dataset.records[index], training, mode=mode, search_step=search_step,
                        neighbors=neighbors, search_span=search_span, weights=weights,
                        matlab_compatible=matlab_compatible)

    predictions = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        for i, prediction in enumerate(executor.map(predict, test), 1):
            predictions.append(prediction)
            if progress:
                progress(f"Classified {i}/{len(test)}: {prediction.true_label} -> "
                         f"{prediction.predicted_label}")
    classes = dataset.classes
    lookup = {label: i for i, label in enumerate(classes)}
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for prediction in predictions:
        confusion[lookup[prediction.true_label], lookup[prediction.predicted_label]] += 1
    return ClassificationResult(classes, confusion, float(np.trace(confusion) / confusion.sum()),
                                tuple(predictions), split, seed)


def neuro_classify_interclass(dataset: Dataset, **kwargs) -> ClassificationResult:
    return classify(dataset, mode="inter", **kwargs)


def neuro_classify_intraclass(dataset: Dataset, **kwargs) -> ClassificationResult:
    return classify(dataset, mode="intra", **kwargs)
