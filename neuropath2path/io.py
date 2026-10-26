"""SWC validation and deterministic directory/ZIP dataset discovery."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from io import StringIO
from pathlib import Path, PurePosixPath
from typing import Iterator, TextIO
from zipfile import ZipFile

import numpy as np


@dataclass(frozen=True)
class Morphology:
    """A single rooted SWC tree; parents and order contain zero-based row indices."""

    nodes: np.ndarray
    parents: np.ndarray
    children: tuple[tuple[int, ...], ...]
    order: np.ndarray
    source: str = ""

    @property
    def root(self) -> int:
        return int(self.order[0])


def morphology_from_array(data: np.ndarray, *, source: str = "") -> Morphology:
    """Validate seven SWC columns without assuming contiguous or sorted node IDs."""
    nodes = np.array(data, dtype=float, copy=True)
    if nodes.ndim != 2 or nodes.shape[1] != 7 or not len(nodes):
        raise ValueError("SWC data must be a nonempty N x 7 array")
    if not np.isfinite(nodes).all():
        raise ValueError("SWC data contains nonfinite values")
    if not np.equal(nodes[:, [0, 1, 6]], np.floor(nodes[:, [0, 1, 6]])).all():
        raise ValueError("SWC node IDs, types, and parent IDs must be integers")
    ids = nodes[:, 0].astype(np.int64)
    if np.any(ids <= 0) or len(np.unique(ids)) != len(ids):
        raise ValueError("SWC node IDs must be positive and unique")
    if np.any(nodes[:, 5] < 0):
        raise ValueError("SWC radii must be nonnegative")
    roots = np.flatnonzero(nodes[:, 6] == -1)
    if len(roots) != 1:
        raise ValueError("SWC must contain exactly one root with parent -1")
    lookup = {int(node_id): i for i, node_id in enumerate(ids)}
    parents = np.full(len(nodes), -1, dtype=np.int64)
    children: list[list[int]] = [[] for _ in nodes]
    for i, parent_id in enumerate(nodes[:, 6].astype(np.int64)):
        if parent_id == -1:
            continue
        if int(parent_id) not in lookup:
            raise ValueError(f"SWC node {ids[i]} has missing parent {parent_id}")
        parent = lookup[int(parent_id)]
        if parent == i:
            raise ValueError(f"SWC node {ids[i]} is its own parent")
        parents[i] = parent
        children[parent].append(i)
    pending = [int(roots[0])]
    order = []
    while pending:
        i = heapq.heappop(pending)
        order.append(i)
        for child in children[i]:
            heapq.heappush(pending, child)
    if len(order) != len(nodes):
        raise ValueError("SWC contains a cycle or nodes disconnected from its root")
    return Morphology(nodes, parents, tuple(map(tuple, children)), np.array(order), source)


def read_swc(source: str | Path | TextIO) -> Morphology:
    """Read a seven-column SWC file, allowing comments and arbitrary whitespace."""
    name = str(source) if isinstance(source, (str, Path)) else getattr(source, "name", "<stream>")
    try:
        data = np.loadtxt(source, comments="#", ndmin=2, encoding="utf-8-sig")
        return morphology_from_array(data, source=name)
    except ValueError as exc:
        raise ValueError(f"{name}: {exc}") from exc


def _visible_parts(parts: tuple[str, ...]) -> bool:
    return not any(p.startswith(".") or p == "__MACOSX" for p in parts)


def iter_dataset(path: str | Path) -> Iterator[tuple[str, str, Morphology]]:
    """Yield (class, source, morphology) from class directories or a bundled ZIP.

    One enclosing directory (e.g. ``Cell_type``) is detected automatically.
    Every visible .swc below each class is included, including CNG version trees.
    ZIP members are read directly, without extraction.
    """
    path = Path(path)
    if path.is_file() and path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            members = sorted(
                (info for info in archive.infolist()
                 if not info.is_dir() and info.filename.lower().endswith(".swc")
                 and _visible_parts(PurePosixPath(info.filename).parts)),
                key=lambda info: info.filename,
            )
            parts = [PurePosixPath(info.filename).parts for info in members]
            if not parts:
                raise ValueError(f"No SWC files found in {path}")
            # The supplied archives both have a single dataset wrapper directory.
            wrapped = (len({p[0] for p in parts}) == 1
                       and all(len(p) >= 3 for p in parts))
            offset = int(wrapped)
            for info, member_parts in zip(members, parts):
                if len(member_parts) < offset + 2:
                    raise ValueError("Dataset must have one directory per class")
                source = f"{path}!{info.filename}"
                try:
                    text = archive.read(info).decode("utf-8-sig")
                    morphology = read_swc(StringIO(text))
                except (ValueError, UnicodeError) as exc:
                    raise ValueError(f"{source}: {exc}") from exc
                yield member_parts[offset], source, morphology
        return
    if not path.is_dir():
        raise ValueError(f"Dataset does not exist or is not a directory/ZIP: {path}")
    files = sorted(p for p in path.rglob("*")
                   if p.is_file() and p.suffix.lower() == ".swc"
                   and _visible_parts(p.relative_to(path).parts))
    if not files:
        raise ValueError(f"No SWC files found in {path}")
    parts = [p.relative_to(path).parts for p in files]
    offset = int(len({p[0] for p in parts}) == 1 and all(len(p) >= 3 for p in parts))
    for file, relative in zip(files, parts):
        if len(relative) < offset + 2:
            raise ValueError("Dataset must have one directory per class")
        yield relative[offset], str(file), read_swc(file)


def read_input(path: str | Path) -> list[Morphology]:
    """Python counterpart of readInput.m for a single class directory."""
    path = Path(path)
    return [read_swc(p) for p in sorted(path.rglob("*"))
            if p.is_file() and p.suffix.lower() == ".swc"
            and _visible_parts(p.relative_to(path).parts)]
