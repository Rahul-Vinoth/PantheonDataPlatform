"""Encoder contract — pixels -> embedding (the encode-once derived-tier producer).

Mirrors the SourceAdapter pattern: a uniform interface behind a registry, with the
model `version` first-class (it becomes Embedding.encoder_version). An encoder defines
its own windowing (window_s / stride_s / sample_fps) and pooling.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Encoder(ABC):
    name: str = "base"
    version: str = "base-0"        # -> Embedding.encoder_version (first-class)
    dim: int = 0
    window_s: float = 2.0          # window length in seconds
    stride_s: float = 2.0          # hop between windows
    sample_fps: float = 2.0        # frames decoded per second within a window
    pooling: str = "mean"          # how per-frame features collapse to one vector

    @abstractmethod
    def embed(self, frames: list[np.ndarray]) -> np.ndarray:
        """frames: list of HxWx3 uint8 RGB arrays -> float32[dim] (L2-normalized)."""
