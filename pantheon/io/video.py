"""Video probing via PyAV — reads metadata + detects truncation/corruption.

We read headers and (cheaply) verify decodability; we never transcode (Part 1 §2).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import av


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    duration_ns: int
    n_frames: int
    codec: str
    truncated: bool


def probe_video(path: Path, verify_frames: int = 3) -> VideoInfo:
    """Probe an mp4/h26x file. Raises on a file that cannot be opened at all;
    sets `truncated=True` (rather than raising) when the stream opens but decoding
    fails partway — so a partially-usable clip degrades instead of vanishing."""
    try:
        container = av.open(str(path))
    except Exception as e:  # unopenable -> hard failure for the caller to quarantine
        raise ValueError(f"cannot open video {path}: {e}") from e

    try:
        vstreams = [s for s in container.streams if s.type == "video"]
        if not vstreams:
            raise ValueError(f"no video stream in {path}")
        vs = vstreams[0]
        fps = float(vs.average_rate) if vs.average_rate else 0.0
        width = vs.codec_context.width
        height = vs.codec_context.height
        codec = vs.codec_context.name
        duration_ns = 0
        if vs.duration is not None and vs.time_base is not None:
            duration_ns = int(vs.duration * vs.time_base * 1e9)
        elif container.duration is not None:
            duration_ns = int(container.duration / av.time_base * 1e9)

        # cheap decodability check: try to pull the first few frames
        truncated = False
        decoded = 0
        try:
            for frame in container.decode(video=0):
                decoded += 1
                if decoded >= verify_frames:
                    break
        except Exception:
            truncated = True
        if decoded == 0:
            truncated = True

        n_frames = vs.frames or 0
        if n_frames == 0 and fps > 0 and duration_ns > 0:
            n_frames = int(round(fps * duration_ns / 1e9))
        return VideoInfo(width, height, fps, duration_ns, n_frames, codec, truncated)
    finally:
        container.close()


def decode_window(path: Path, base_offset_s: float, t0_s: float, t1_s: float,
                  sample_fps: float = 2.0):
    """Decode RGB frames in the window [base_offset+t0, base_offset+t1), sampled at
    `sample_fps`. `base_offset_s` is the segment's offset inside a packed file (from the
    stream's payload_locator, e.g. LeRobot's from_timestamp); 0 for standalone clips.

    Returns a list of HxWx3 uint8 RGB ndarrays. Tolerant of truncation: stops at the
    window end or whatever decodes."""
    import numpy as np  # local import keeps probe path import-light

    start = base_offset_s + t0_s
    end = base_offset_s + t1_s
    container = av.open(str(path))
    frames: list = []
    try:
        vs = container.streams.video[0]
        tb = vs.time_base
        if tb:
            try:
                container.seek(int(start / tb), stream=vs, any_frame=False, backward=True)
            except Exception:
                pass  # un-seekable -> decode from head, filter by timestamp below
        interval = 1.0 / sample_fps if sample_fps > 0 else 0.0
        next_t = start
        try:
            for frame in container.decode(video=0):
                if frame.pts is None or tb is None:
                    continue
                t = float(frame.pts * tb)
                if t < start:
                    continue
                if t >= end:
                    break
                if t + 1e-6 >= next_t:
                    frames.append(frame.to_ndarray(format="rgb24"))
                    next_t += interval
        except Exception:
            pass  # truncated mid-window -> return what we got
        return frames
    finally:
        container.close()
