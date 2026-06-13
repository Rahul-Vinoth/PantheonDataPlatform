"""Encode-once driver: video/image_seq Streams -> Embedding rows (+ vector index).

This is the first derived-tier producer and the ONE place we read cold payloads. It
scans the catalog for video streams, decodes each window from its native payload (using
the stream's payload_locator to seek the right segment in packed files), runs the chosen
encoder, and appends Embedding rows. Re-runnable: existing (stream_id, window) pairs for
this encoder_version are skipped, so it's idempotent and append-only.

Usage:
    python -m pantheon.encode ./lakehouse --encoder clip-vit-b32
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lance

from .encoders.registry import get_encoder, list_encoders
from .io.hashing import stable_id
from .io.video import decode_window
from .schema.records import Embedding
from .writer import CanonicalWriter
import pantheon.encoders  # noqa: F401  (registers built-in encoders)


@dataclass
class EncodeStats:
    streams: int = 0
    embeddings: int = 0
    skipped: int = 0
    empty_windows: int = 0


def _read_table(lakehouse: Path, name: str) -> list[dict]:
    path = lakehouse / f"{name}.lance"
    if not path.exists():
        return []
    return lance.dataset(str(path)).to_table().to_pylist()


def _windows(t_start_ns: int, t_end_ns: int, window_s: float, stride_s: float):
    dur_s = (t_end_ns - t_start_ns) / 1e9
    t = 0.0
    while t < dur_s - 1e-6:
        t1 = min(t + window_s, dur_s)
        yield t, t1, t_start_ns + int(t * 1e9), t_start_ns + int(t1 * 1e9)
        t += stride_s


def encode_lakehouse(lakehouse: str | Path, encoder_name: str) -> EncodeStats:
    lakehouse = Path(lakehouse)
    enc = get_encoder(encoder_name)()
    print(f"[encode] encoder={enc.name} version={enc.version} dim={enc.dim} "
          f"window={enc.window_s}s stride={enc.stride_s}s")

    # idempotency: which (stream_id, t_start_ns) already exist for this version
    done = {(e["stream_id"], e["t_start_ns"])
            for e in _read_table(lakehouse, "embedding")
            if e.get("encoder_version") == enc.version}

    streams = [s for s in _read_table(lakehouse, "stream")
               if s["modality"] in ("video", "image_seq")]

    writer = CanonicalWriter(lakehouse)
    st = EncodeStats()
    for s in streams:
        uri = s.get("payload_uri")
        if not uri or str(uri).startswith("tar://"):
            continue  # in-tar payloads not supported by the baseline decoder
        if not Path(uri).exists():
            continue
        st.streams += 1
        loc = json.loads(s["payload_locator_json"]) if s.get("payload_locator_json") else {}
        base = float(loc.get("from_timestamp", 0.0))

        for t0, t1, t0_ns, t1_ns in _windows(
                s["t_start_ns"], s["t_end_ns"], enc.window_s, enc.stride_s):
            if (s["stream_id"], t0_ns) in done:
                st.skipped += 1
                continue
            frames = decode_window(Path(uri), base, t0, t1, enc.sample_fps)
            if not frames:
                st.empty_windows += 1
                continue
            vec = enc.embed(frames)
            writer.add_embedding(Embedding(
                embedding_id=stable_id(s["stream_id"], enc.version, str(t0_ns)),
                episode_id=s["episode_id"], stream_id=s["stream_id"],
                t_start_ns=t0_ns, t_end_ns=t1_ns, encoder_version=enc.version,
                vector=vec.tolist(),
                meta={"dim": enc.dim, "pooling": enc.pooling,
                      "sample_fps": enc.sample_fps, "n_frames": len(frames)}))
            st.embeddings += 1

    counts = writer.flush()
    print(f"[encode] streams={st.streams} embeddings={st.embeddings} "
          f"skipped={st.skipped} empty_windows={st.empty_windows}")
    print(f"[encode] rows written: {counts}")
    _build_vector_index(lakehouse)
    return st


def _build_vector_index(lakehouse: Path) -> None:
    """Build a cosine ANN index on the embedding vectors (best-effort: small corpora
    fall back to exact/brute-force search, which is correct, just unindexed)."""
    try:
        import lancedb
        db = lancedb.connect(str(lakehouse))
        tbl = db.open_table("embedding")
        n = tbl.count_rows()
        if n < 256:
            print(f"[encode] {n} embeddings (<256): skipping ANN index "
                  f"(exact search is used)")
            return
        tbl.create_index(metric="cosine", vector_column_name="vector")
        print(f"[encode] built cosine ANN index over {n} embeddings")
    except Exception as e:
        print(f"[encode] vector index skipped: {e}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Encode video streams into embeddings.")
    ap.add_argument("lakehouse")
    ap.add_argument("--encoder", default="clip-vit-b32",
                    help=f"one of: {list_encoders()}")
    args = ap.parse_args()
    encode_lakehouse(args.lakehouse, args.encoder)
