"""CLIP (open_clip ViT-B/32) encoder — real semantic embeddings.

Per window: decode sampled frames -> CLIP image features per frame -> L2-normalize ->
mean-pool -> L2-normalize. Output is a 512-d float32 vector suitable for cosine ANN.

The torch/open_clip import is lazy so the rest of the package stays importable without
the heavy ML stack installed.
"""
from __future__ import annotations

import numpy as np

from .base import Encoder
from .registry import register_encoder


@register_encoder("clip-vit-b32")
class ClipEncoder(Encoder):
    version = "clip-vit-b32-openai"
    dim = 512
    window_s = 2.0
    stride_s = 2.0
    sample_fps = 2.0
    pooling = "mean"

    def __init__(self):
        self._model = None
        self._preprocess = None
        self._device = None
        self._torch = None

    def _lazy(self):
        if self._model is not None:
            return
        import torch
        import open_clip

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai")
        model.eval().to(self._device)
        self._model = model
        self._preprocess = preprocess

    def embed(self, frames: list[np.ndarray]) -> np.ndarray:
        if not frames:
            return np.zeros(self.dim, dtype=np.float32)
        self._lazy()
        torch = self._torch
        from PIL import Image

        batch = torch.stack(
            [self._preprocess(Image.fromarray(f)) for f in frames]
        ).to(self._device)
        with torch.no_grad():
            feats = self._model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)   # per-frame L2
            pooled = feats.mean(dim=0)                          # temporal mean-pool
            pooled = pooled / pooled.norm()                    # re-normalize
        return pooled.detach().cpu().numpy().astype(np.float32)
