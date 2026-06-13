"""IDM contract — infer an action latent from consecutive observation embeddings.

Mirrors the Encoder/SourceAdapter pattern: a uniform interface behind a registry, with
the model `version` first-class (it becomes ActionLatent.idm_version). The IDM consumes
EMBEDDINGS, never pixels — which is what makes re-labeling the corpus cheap.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class IDM(ABC):
    name: str = "base"
    version: str = "base-0"        # -> ActionLatent.idm_version (first-class)
    latent_dim: int = 0            # informational; 0 = same as the input embedding dim

    @abstractmethod
    def infer(self, prev: np.ndarray, nxt: np.ndarray) -> tuple[np.ndarray, float]:
        """Two consecutive observation embeddings -> (action latent float32[K],
        confidence in [0,1])."""
