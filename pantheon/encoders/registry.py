"""Self-registering encoder registry (same pattern as the adapter registry).

New encoder = a new module with `@register_encoder("name")`; no central file changes.
The encoder's `version` is what gets stamped onto every Embedding row, so multiple
encoder generations coexist in the derived tier.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Encoder

_ENCODER_REGISTRY: dict[str, type["Encoder"]] = {}


def register_encoder(name: str):
    def deco(cls):
        if name in _ENCODER_REGISTRY:
            raise ValueError(f"duplicate encoder: {name}")
        cls.name = name
        _ENCODER_REGISTRY[name] = cls
        return cls
    return deco


def get_encoder(name: str) -> type["Encoder"]:
    if name not in _ENCODER_REGISTRY:
        raise KeyError(f"no encoder '{name}'; known: {sorted(_ENCODER_REGISTRY)}")
    return _ENCODER_REGISTRY[name]


def list_encoders() -> list[str]:
    return sorted(_ENCODER_REGISTRY)
