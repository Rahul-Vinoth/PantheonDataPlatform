"""CanonicalWriter — materializes canonical records into Lance datasets.

One Lance dataset per table (the lakehouse). Rows are buffered per table and flushed in
batches (Trossen #4: batch the consumer side). `oneof`/struct fields arrive already
flattened by the record `to_row()` methods.
"""
from __future__ import annotations

from pathlib import Path

import lance
import pyarrow as pa

from .schema.records import CanonicalRecords, Quarantine
from .schema.tables import SCHEMAS, ALL_TABLES


class CanonicalWriter:
    def __init__(self, lakehouse_dir: str | Path):
        self.dir = Path(lakehouse_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._buf: dict[str, list[dict]] = {t: [] for t in ALL_TABLES}

    # --------------------------------------------------------------- buffering
    def add_row(self, table: str, row: dict) -> None:
        self._buf[table].append(row)

    def add_records(self, rec: CanonicalRecords) -> None:
        if rec.episode:
            self.add_row("episode", rec.episode.to_row())
        for c in rec.clocks:
            self.add_row("clock", c.to_row())
        for cs in rec.clock_syncs:
            self.add_row("clock_sync", cs.to_row())
        for s in rec.streams:
            self.add_row("stream", s.to_row())
        for cal in rec.calibrations:
            self.add_row("calibration", cal.to_row())
        for a in rec.annotations:
            self.add_row("annotation", a.to_row())
        for q in rec.quality_signals:
            self.add_row("quality_signal", q.to_row())

    def add_quarantine(self, q: Quarantine) -> None:
        self.add_row("quarantine", q.to_row())

    def add_embedding(self, e) -> None:
        """Derived-tier write path (the encoder fills this; ingest does not)."""
        self.add_row("embedding", e.to_row())

    # ------------------------------------------------------------------- flush
    def _path(self, table: str) -> str:
        return str(self.dir / f"{table}.lance")

    def flush(self) -> dict[str, int]:
        """Write all buffered rows to their Lance datasets (append if exists)."""
        counts: dict[str, int] = {}
        for table, rows in self._buf.items():
            if not rows:
                continue
            schema = SCHEMAS[table]
            tbl = _rows_to_table(rows, schema)
            path = self._path(table)
            mode = "append" if Path(path).exists() else "create"
            lance.write_dataset(tbl, path, mode=mode)
            counts[table] = len(rows)
            rows.clear()
        return counts

    # ------------------------------------------------------------------- read
    def open(self, table: str) -> "lance.LanceDataset":
        return lance.dataset(self._path(table))


def _rows_to_table(rows: list[dict], schema: pa.Schema) -> pa.Table:
    """Build a pyarrow table with the exact declared schema, filling missing
    columns with nulls and coercing types (keeps appends schema-stable)."""
    cols: dict[str, list] = {f.name: [] for f in schema}
    for r in rows:
        for f in schema:
            cols[f.name].append(r.get(f.name))
    arrays = [pa.array(cols[f.name], type=f.type) for f in schema]
    return pa.table(arrays, schema=schema)
