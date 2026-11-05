import importlib.util
import json
import shutil
import subprocess

import pytest

from neuropath2path import morph_neurons, write_morph_video


def test_exported_mp4_codec_size_and_endpoint_pauses(branched_tree, tmp_path, monkeypatch):
    if not importlib.util.find_spec("matplotlib") or not all(shutil.which(n) for n in ("ffmpeg", "ffprobe")):
        pytest.skip("Optional Matplotlib/FFmpeg/ffprobe video dependencies unavailable")
    encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                              capture_output=True, text=True, check=True).stdout
    if "libx264" not in encoders:
        pytest.skip("FFmpeg libx264 encoder unavailable")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl-cache"))
    target = branched_tree.nodes.copy()
    target[:, 2:5] *= 2
    morph = morph_neurons(branched_tree, target, times=[0, 0.5, 1], samples=8)
    destination = tmp_path / "morph.mp4"
    assert write_morph_video(morph, destination, fps=4, hold_seconds=0.5,
                            width=640, height=360) == destination
    probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                            "-show_entries", "stream=codec_name,pix_fmt,width,height,nb_read_frames",
                            "-of", "json", str(destination)], capture_output=True, text=True, check=True)
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["codec_name"] == "h264"
    assert stream["pix_fmt"] == "yuv420p"
    assert (stream["width"], stream["height"]) == (640, 360)
    assert int(stream["nb_read_frames"]) == 7  # 3 morph frames + 2 pauses of 2 frames
