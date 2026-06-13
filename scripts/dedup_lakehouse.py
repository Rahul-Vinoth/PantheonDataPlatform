"""Deduplicate lakehouse tables by primary key (keep first occurrence).

Re-ingesting the same source currently appends rows (the documented ingest idempotency
gap). This maintenance pass collapses each table to one row per PK — the PK is the first
column in every schema (embedding_id, episode_id, source_id, …). `ingest_run` is left
alone: its rows are legitimately distinct (one per invocation), and `clock_sync` has a
composite key so it is skipped.

Usage:  python scripts/dedup_lakehouse.py ./lakehouse
"""
from __future__ import annotations

import sys
from pathlib import Path

import lance

SKIP = {"clock_sync"}          # composite PK
KEEP_ALL = {"ingest_run"}      # rows are distinct by design (run history)


def dedup(lakehouse: str | Path) -> None:
    lakehouse = Path(lakehouse)
    for path in sorted(lakehouse.glob("*.lance")):
        name = path.stem
        ds = lance.dataset(str(path))
        t = ds.to_table()
        if name in SKIP or name in KEEP_ALL or t.num_rows == 0:
            print(f"  {name:16s} {t.num_rows:5d} rows  (left as-is)")
            continue
        pk = t.column_names[0]
        col = t.column(pk).to_pylist()
        seen, keep = set(), []
        for i, v in enumerate(col):
            if v in seen:
                continue
            seen.add(v)
            keep.append(i)
        if len(keep) == t.num_rows:
            print(f"  {name:16s} {t.num_rows:5d} rows  (no dups)")
            continue
        deduped = t.take(keep)
        lance.write_dataset(deduped, str(path), mode="overwrite")
        print(f"  {name:16s} {t.num_rows:5d} -> {deduped.num_rows:5d} rows  "
              f"(removed {t.num_rows - deduped.num_rows} dup, key={pk})")


if __name__ == "__main__":
    dedup(sys.argv[1] if len(sys.argv) > 1 else "./lakehouse")
