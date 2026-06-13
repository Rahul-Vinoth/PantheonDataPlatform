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
