"""Content hashing for dedup + provenance (Trossen #9: source_id/seq + checksum)."""
from __future__ import annotations

import hashlib
from pathlib import Path


def file_checksum(path: Path, chunk: int = 1 << 20, max_bytes: int | None = None) -> str:
    """Streaming sha256 of a file. `max_bytes` caps work for large media (a prefix
    hash is enough for near-dup detection at ingest; full hash optional)."""
    h = hashlib.sha256()
    read = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            read += len(b)
            if max_bytes is not None and read >= max_bytes:
                break
    return h.hexdigest()


def stable_id(*parts: str) -> str:
    """Deterministic id from parts -> idempotent re-ingest (same unit -> same id)."""
    return hashlib.sha1("\x1f".join(parts).encode()).hexdigest()[:16]
