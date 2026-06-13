"""Self-registering IDM registry (same pattern as the encoder/adapter registries).

New IDM = a new module with `@register_idm("name")`; no central file changes. The IDM's
`version` is stamped onto every ActionLatent row, so multiple IDM generations coexist in
the derived tier and a re-label is just an append with a new idm_version.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import IDM

_IDM_REGISTRY: dict[str, type["IDM"]] = {}


def register_idm(name: str):
    def deco(cls):
        if name in _IDM_REGISTRY:
            raise ValueError(f"duplicate idm: {name}")
        cls.name = name
        _IDM_REGISTRY[name] = cls
        return cls
    return deco


def get_idm(name: str) -> type["IDM"]:
    if name not in _IDM_REGISTRY:
        raise KeyError(f"no idm '{name}'; known: {sorted(_IDM_REGISTRY)}")
    return _IDM_REGISTRY[name]


def list_idms() -> list[str]:
    return sorted(_IDM_REGISTRY)
