"""Delta IDM — a dependency-free baseline.

Inverse dynamics in latent space: the action that took you from observation t to t+1 is
represented by the (normalized) change in embedding, delta = norm(emb_{t+1} - emb_t).
Confidence scales with how much the representation moved (a larger change = stronger
evidence that *something* happened).

This is a real, runnable baseline that exercises the whole ActionLatent path. A learned
IDM (an MLP/transformer over embedding pairs or windows) drops in behind the same
`infer()` interface with a new `version` — no driver changes.
"""
from __future__ import annotations

import numpy as np

from .base import IDM
from .registry import register_idm


@register_idm("delta")
class DeltaIDM(IDM):
    version = "idm-delta-0.1"
    latent_dim = 0  # latent matches the input embedding dim

    def infer(self, prev: np.ndarray, nxt: np.ndarray) -> tuple[np.ndarray, float]:
        delta = nxt - prev
        mag = float(np.linalg.norm(delta))
        latent = (delta / mag) if mag > 1e-8 else delta
        # embeddings are unit-normalized, so ||delta|| in [0,2]; map to a [0,1] confidence
        confidence = float(min(1.0, mag / 2.0))
        return latent.astype(np.float32), confidence
