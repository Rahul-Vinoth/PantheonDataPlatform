"""Self-registering SourceAdapter registry (Trossen takeaway #1, arrows reversed).

One registry per extensible axis. New dataset format = a new adapter module with a
`@register_adapter("name")` decorator; no central file changes. Adapters extend the
*ingester*, never the schema.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .adapters.base import SourceAdapter

_ADAPTER_REGISTRY: dict[str, type["SourceAdapter"]] = {}


def register_adapter(name: str):
    def deco(cls):
        if name in _ADAPTER_REGISTRY:
            raise ValueError(f"duplicate adapter: {name}")
        cls.name = name
        _ADAPTER_REGISTRY[name] = cls
        return cls
    return deco


def get_adapter(name: str) -> type["SourceAdapter"]:
    if name not in _ADAPTER_REGISTRY:
        raise KeyError(f"no adapter '{name}'; known: {sorted(_ADAPTER_REGISTRY)}")
    return _ADAPTER_REGISTRY[name]


def list_adapters() -> list[str]:
    return sorted(_ADAPTER_REGISTRY)


def select_adapter(root) -> "SourceAdapter":
    """Pick the adapter whose .probe(root) accepts the layout."""
    matches = []
    for name, cls in _ADAPTER_REGISTRY.items():
        inst = cls()
        try:
            if inst.probe(root):
                matches.append(inst)
        except Exception:
            continue
    if not matches:
        raise KeyError(f"no adapter matched layout at {root}")
    if len(matches) > 1:
        names = [m.name for m in matches]
        raise ValueError(f"ambiguous: multiple adapters matched {root}: {names}")
    return matches[0]
