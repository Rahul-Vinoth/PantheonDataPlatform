"""Source-agnostic ingest loop + CLI.

    select adapter -> iter episode units -> emit canonical records -> write to Lance
    quarantine on failure (surface, never drop/crash)  [Part 1 §5.4]
"""
from __future__ import annotations

import argparse
import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path

from .io.hashing import stable_id
from .registry import select_adapter, get_adapter, list_adapters
from .schema.records import IngestRun, Quarantine, Source
from .writer import CanonicalWriter
from . import adapters as _adapters  # noqa: F401  (triggers adapter registration)

CODE_VERSION = "pantheon-ingest-0.1.0"


@dataclass
class IngestStats:
    ok: int = 0
    partial: int = 0
    quarantined: int = 0
    by_reason: dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        return (f"ok={self.ok} partial={self.partial} quarantined={self.quarantined}"
                + (f" reasons={self.by_reason}" if self.by_reason else ""))


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def ingest_source(root: str | Path, lakehouse: str | Path,
                  adapter_name: str | None = None) -> IngestStats:
    root = Path(root)
    adapter = get_adapter(adapter_name)() if adapter_name else select_adapter(root)
    writer = CanonicalWriter(lakehouse)
    stats = IngestStats()

    # registries + source row (idempotent; appended once per run)
    src: Source = adapter.source()
    for emb in adapter.embodiments():
        writer.add_row("embodiment", emb.to_row())
    for tax in adapter.taxonomies():
        writer.add_row("taxonomy", tax.to_row())
    writer.add_row("source", src.to_row())

    run_id = stable_id("run", adapter.name, str(root), _now())
    writer.add_row("ingest_run", IngestRun(
        ingest_run_id=run_id, code_version=CODE_VERSION, started_at=_now(),
        source_path=str(root)).to_row())

    for unit in adapter.iter_episodes(root):
        episode_id = stable_id(src.source_id, unit.unit_ref)
        try:
            rec = adapter.emit(unit, episode_id=episode_id, ingest_run_id=run_id)
        except Exception as e:
            stage = getattr(e, "stage", "map")
            reason = f"{stage}: {e}"
            writer.add_quarantine(Quarantine(
                quarantine_id=stable_id("q", episode_id, reason),
                source_name=src.name, unit_ref=unit.unit_ref, stage=stage,
                reason=str(e), ingest_run_id=run_id, ts=_now()))
            stats.quarantined += 1
            stats.by_reason[stage] = stats.by_reason.get(stage, 0) + 1
            continue

        writer.add_records(rec)
        status = rec.episode.quality_status.value if rec.episode else "ok"
        if status == "partial":
            stats.partial += 1
        else:
            stats.ok += 1

    counts = writer.flush()
    print(f"[ingest] adapter={adapter.name} root={root}")
    print(f"[ingest] {stats}")
    print(f"[ingest] rows written: {counts}")
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description="Pantheon canonical ingester")
    p.add_argument("root", help="path to a source dataset root")
    p.add_argument("--lakehouse", default="./lakehouse", help="output Lance dir")
    p.add_argument("--adapter", default=None,
                   help=f"force an adapter; known: {list_adapters()}")
    args = p.parse_args()
    ingest_source(args.root, args.lakehouse, args.adapter)


if __name__ == "__main__":
    main()
