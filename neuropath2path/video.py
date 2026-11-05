"""Optional H.264 MP4 export for presentations, rendered without a GUI."""

from __future__ import annotations

from colorsys import hls_to_rgb
from pathlib import Path
import subprocess

import numpy as np

from .morphing import NeuronMorph, _endpoint_curves


def write_morph_video(morph: NeuronMorph, destination: str | Path, *,
                      fps: float = 12, hold_seconds: float = 1,
                      width: int = 1920, height: int = 1080,
                      overlay: bool = True) -> Path:
    """Export source/current/target panels to silent H.264/yuv420p MP4.

    Every requested morph time gets one frame, in the supplied order. The
    first and last requested frames each receive hold_seconds of extra pause.
    All panels use the same fixed camera and scale as the HTML viewer's default
    view. By default, faint source/target outlines are overlaid on the current
    morph; set overlay=False to disable them. Requires optional Matplotlib and
    an FFmpeg installation with libx264.
    """
    destination = Path(destination)
    if destination.suffix.lower() != ".mp4":
        raise ValueError("Video destination must have an .mp4 extension")
    if not morph.paths or not len(morph.times):
        raise ValueError("Morph must contain paths and frames")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if not np.isfinite(hold_seconds) or hold_seconds < 0:
        raise ValueError("hold_seconds must be finite and nonnegative")
    if any(not isinstance(v, (int, np.integer)) or v < 2 or v % 2 for v in (width, height)):
        raise ValueError("Video width and height must be positive even integers")
    try:
        from matplotlib.animation import FFMpegWriter
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.collections import LineCollection
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise RuntimeError("Video export requires Matplotlib: pip install -e '.[video]'") from exc
    if not FFMpegWriter.isAvailable():
        raise RuntimeError("Video export requires FFmpeg with the libx264 encoder on PATH")

    source = _endpoint_curves(morph, "source_curve", 0, 0)
    target = _endpoint_curves(morph, "target_curve", 1, 1)
    points = np.concatenate([*(path.frames.reshape(-1, 3) for path in morph.paths),
                             *source, *target])
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2
    radius = max(1e-12, float(np.linalg.norm(high - low) / 2))
    yaw, pitch = 0.4, -0.3

    def project(curve):
        x, y, z = ((curve - center) / radius).T
        a = np.cos(yaw) * x + np.sin(yaw) * z
        b = -np.sin(yaw) * x + np.cos(yaw) * z
        c = np.cos(pitch) * y - np.sin(pitch) * b
        depth = np.sin(pitch) * y + np.cos(pitch) * b
        return np.column_stack((a, c)) * (4 / (4 - depth))[:, None]

    figure = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor="white")
    FigureCanvasAgg(figure)
    arbor_label = {"dendrite": "Dendrites", "axon": "Axons", "both": "Dendrites and axons"}[morph.arbor]
    figure.suptitle(f"NeuroPath2Path — path morphing · {arbor_label}", fontsize=24, y=0.94)
    caption = ("Blue: source · Orange: target · Colored paths: current morph" if overlay
               else "Source and target remain fixed · Shared scale and viewing angle")
    figure.text(0.5, 0.075, caption,
                ha="center", fontsize=14, color="#475867")
    axes = figure.subplots(1, 3)
    figure.subplots_adjust(left=0.025, right=0.975, bottom=0.15, top=0.82, wspace=0.08)
    colors = [hls_to_rgb((i * 137.508 % 360) / 360, 0.4, 0.65)
              for i in range(len(morph.paths))]
    for ax in axes:
        ax.set(xlim=(-1.25, 1.25), ylim=(-1.25, 1.25), aspect="equal")
        ax.set_axis_off()
        origin = project(np.zeros((1, 3)))[0]
        ax.plot(*origin, marker="o", color="#172b3a", markersize=3)
    axes[0].set_title("Source · α = 0", color="#2563eb", fontsize=19)
    axes[2].set_title("Target · α = 1", color="#c2410c", fontsize=19)
    axes[0].add_collection(LineCollection([project(c) for c in source], colors="#2563eb", linewidths=1.4))
    axes[2].add_collection(LineCollection([project(c) for c in target], colors="#c2410c", linewidths=1.4))
    if overlay:
        axes[1].add_collection(LineCollection([project(c) for c in source], colors="#2563eb",
                                              linewidths=1.2, alpha=0.2, zorder=1))
        axes[1].add_collection(LineCollection([project(c) for c in target], colors="#c2410c",
                                              linewidths=1.2, alpha=0.2, zorder=1))
    current = LineCollection([], colors=colors, linewidths=1.4)
    axes[1].add_collection(current)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=-1,
                          extra_args=["-pix_fmt", "yuv420p", "-crf", "18", "-threads", "2",
                                      "-movflags", "+faststart"],
                          metadata={"title": "NeuroPath2Path path morphing"})
    hold_frames = int(round(hold_seconds * fps))
    try:
        with writer.saving(figure, str(destination), dpi=100):
            for i, time in enumerate(morph.times):
                current.set_segments([project(path.frames[i]) for path in morph.paths])
                axes[1].set_title(f"Current morph · α = {time:.3f}", fontsize=19)
                repeats = 1 + hold_frames * (int(i == 0) + int(i == len(morph.times) - 1))
                for _ in range(repeats):
                    writer.grab_frame(facecolor="white")
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"MP4 export failed; check FFmpeg/libx264 availability: {exc}") from exc
    finally:
        figure.clear()
    return destination
