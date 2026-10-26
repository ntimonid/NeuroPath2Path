"""Command-line entry points for all workflows present in the MATLAB sources."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import sys
from zipfile import BadZipFile

import numpy as np

from . import __version__
from .classification import classify, load_dataset
from .distance import compute_distance
from .features import FEATURE_NAMES, extract_features
from .io import read_swc
from .morphing import morph_neurons, write_morph_html
from .weights import learn_dataset_weights


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write(result, destination: Path | None):
    content = json.dumps(result, default=_json_default, indent=2, allow_nan=False) + "\n"
    if destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        print(f"Saved {destination}", file=sys.stderr)
    else:
        print(content, end="")


def _progress(message: str):
    print(message, file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    features = commands.add_parser("features", help="Extract all path features from an SWC")
    features.add_argument("swc", type=Path)

    compare = commands.add_parser("compare", help="Compare two SWC neuronal arbors")
    compare.add_argument("first", type=Path)
    compare.add_argument("second", type=Path)
    compare.add_argument("--mode", choices=["inter", "intra", "raw"], default="inter")
    compare.add_argument("--weights", nargs=6, type=float, metavar="W")
    compare.add_argument("--matlab-compatible", action="store_true",
                         help="Reproduce the original asymmetric tortuosity calculation")

    morph = commands.add_parser("morph", help="Interpolate two SWCs using matched-path SRVFs")
    morph.add_argument("first", type=Path)
    morph.add_argument("second", type=Path)
    morph.add_argument("--method", choices=["elasticpath2path", "paper"], default="elasticpath2path",
                       help="Earlier tool's implementation (default) or interval SRVF equations")
    morph.add_argument("--mode", choices=["inter", "intra", "raw"], default="inter",
                       help="NeuroPath2Path distance mode for matching paths")
    morph.add_argument("--frames", type=int, default=21)
    morph.add_argument("--samples", type=int, default=128)
    morph.add_argument("--axis-order", choices=["original", "range"], default="original")
    morph.add_argument("--matlab-compatible", action="store_true",
                       help="Use the original NeuroPath2Path tortuosity formula for matching")
    morph.add_argument("--html", type=Path, help="Also save a standalone 3D animation with a slider")

    extract = commands.add_parser("extract", help="Summarize/extract a directory or ZIP dataset")
    extract.add_argument("--include-paths", action="store_true", help="Include all per-path arrays")
    classification = commands.add_parser("classify", help="Run interclass/intraclass evaluation")
    classification.add_argument("--mode", choices=["inter", "intra"], default="inter")
    classification.add_argument("--test-proportion", type=float, default=0.1)
    classification.add_argument("--bin-width", type=float, help="Default: 15 inter, 7 intra")
    classification.add_argument("--search-step", type=float, default=5)
    classification.add_argument("--neighbors", type=int, default=5)
    classification.add_argument("--search-span", type=int, help="Default: 9 inter, 10 intra")
    classification.add_argument("--seed", type=int, default=100)
    classification.add_argument("--jobs", type=int, default=1, help="Concurrent test-neuron evaluations")
    classification.add_argument("--weights", nargs=6, type=float, metavar="W")
    classification.add_argument("--matlab-compatible", action="store_true",
                                help="Original tortuosity formula and search-span bound")
    weights = commands.add_parser("weights", help="Learn importance weights for every class pair")
    weights.add_argument("--method", choices=["constrained", "legacy"], default="constrained")
    weights.add_argument("--seed", type=int, default=100)
    for command in (extract, classification, weights):
        command.add_argument("dataset", type=Path, help="Class directories or a dataset ZIP archive")
        command.add_argument("--limit-per-class", type=int, help="Take the first N dendritic neurons per class")
        command.add_argument("--quiet", action="store_true")
    for command in (features, compare, morph, extract, classification, weights):
        command.add_argument("--output", "-o", type=Path, help="Write JSON here instead of stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "features":
            result = extract_features(read_swc(args.swc))
        elif args.command == "compare":
            result = compute_distance(extract_features(read_swc(args.first)),
                                      extract_features(read_swc(args.second)), mode=args.mode,
                                      weights=args.weights, matlab_compatible=args.matlab_compatible)
        elif args.command == "morph":
            if args.frames < 2:
                raise ValueError("frames must be at least 2 to include both endpoints")
            result = morph_neurons(read_swc(args.first), read_swc(args.second),
                                   times=np.linspace(0, 1, args.frames), samples=args.samples,
                                   mode=args.mode, axis_order=args.axis_order, method=args.method,
                                   matlab_compatible=args.matlab_compatible)
            if args.html:
                write_morph_html(result, args.html)
                _progress(f"Saved interactive animation: {args.html}")
        else:
            progress = None if args.quiet else _progress
            dataset = load_dataset(args.dataset, limit_per_class=args.limit_per_class,
                                   progress=progress)
            if args.command == "extract":
                result = {
                    "classes": dataset.classes, "skipped_axon_only": dataset.skipped,
                    "records": [
                        {"label": record.label, "source": record.source,
                         "path_count": record.features.path_count,
                         "average_path_length": record.features.average_path_length,
                         "max_order": record.features.max_order,
                         **({"paths": record.features.paths} if args.include_paths else {})}
                        for record in dataset.records],
                }
            elif args.command == "classify":
                bin_width = args.bin_width if args.bin_width is not None else (15 if args.mode == "inter" else 7)
                search_span = args.search_span if args.search_span is not None else (9 if args.mode == "inter" else 10)
                result = classify(
                    dataset, mode=args.mode, test_proportion=args.test_proportion,
                    bin_width=bin_width, search_step=args.search_step, neighbors=args.neighbors,
                    search_span=search_span, seed=args.seed, jobs=args.jobs, weights=args.weights,
                    matlab_compatible=args.matlab_compatible, progress=progress)
                if progress:
                    progress(f"Accuracy: {result.accuracy * 100:.2f}% ({len(result.predictions)} test neurons)")
                result = {**asdict(result), "skipped_axon_only": dataset.skipped,
                          "record_sources": [r.source for r in dataset.records],
                          "effective_bin_width": bin_width, "effective_search_span": search_span}
            else:
                result = learn_dataset_weights(dataset, method=args.method, seed=args.seed,
                                               progress=progress)
        _write({"version": __version__, "feature_names": FEATURE_NAMES,
                "parameters": {k: v for k, v in vars(args).items() if k != "output"},
                "result": result}, args.output)
    except (OSError, ValueError, RuntimeError, BadZipFile) as exc:
        parser.exit(2, f"neuropath2path: {exc}\n")
    return 0
