"""SourceAdapter contract (uniform interface, Trossen #5) + lifecycle split (#2).

Each adapter separates *asset access* (locate/open/validate native bytes) from *schema
mapping* (native metadata -> canonical records). `emit()` references native payloads
and reads only headers/metadata — it never re-encodes (Part 1 §2: keep payloads native).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from ..schema.records import CanonicalRecords, Embodiment, Source, Taxonomy


class IngestError(Exception):
    """Raised by adapters to quarantine a unit; `stage` localizes the failure."""

    def __init__(self, message: str, stage: str = "map"):
        super().__init__(message)
        self.stage = stage


@dataclass
class EpisodeUnit:
    """One discovered source episode: a stable ref + the native paths it comprises."""
    unit_ref: str                       # stable id within the source (e.g. relpath)
    root: Path
    paths: dict[str, Path] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


class SourceAdapter(ABC):
    name: str = "base"

    # ----- source-level metadata (registries + Source row) -----
    @abstractmethod
    def source(self) -> Source: ...

    def embodiments(self) -> list[Embodiment]:
        """Embodiment registry rows this source introduces (default: none)."""
        return []

    def taxonomies(self) -> list[Taxonomy]:
        return []

    # ----- discovery / probing -----
    @abstractmethod
    def probe(self, root: Path) -> bool:
        """Cheap check: does this adapter handle the layout at `root`?"""

    @abstractmethod
    def iter_episodes(self, root: Path) -> Iterator[EpisodeUnit]:
        """Discover episode units; tolerate dupes/truncation/missing side-cars."""

    # ----- mapping -----
    @abstractmethod
    def emit(self, unit: EpisodeUnit, *, episode_id: str, ingest_run_id: str
             ) -> CanonicalRecords:
        """Map ONE unit -> canonical records. Raise IngestError to quarantine."""
