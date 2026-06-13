"""IDM driver: Embedding rows -> ActionLatent rows (re-label often, cheap).

Runs after the encoder. For each stream, it orders that stream's embeddings by time and
runs the IDM over consecutive (obs_t, obs_t+1) pairs, writing one ActionLatent per
transition. Latents key off the *source embedding* (source_embedding_id), never pixels —
so swapping in a new IDM is a scan-and-write over the hot Embedding table, stamped with a
new idm_version. Idempotent: (source_embedding_id, idm_version) pairs already present are
skipped.

Embeddings are paired only within the same (stream_id, encoder_version), so vectors being
differenced are always comparable.

Usage:
    python -m pantheon.idm ./lakehouse --idm delta
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import lance
import numpy as np

from .idms.registry import get_idm, list_idms
from .io.hashing import stable_id
from .schema.records import ActionLatent
from .writer import CanonicalWriter
import pantheon.idms  # noqa: F401  (registers built-in IDMs)


@dataclass
class IdmStats:
    streams: int = 0
    latents: int = 0
    skipped: int = 0


def _read(lakehouse: Path, name: str) -> list[dict]:
    path = lakehouse / f"{name}.lance"
    return lance.dataset(str(path)).to_table().to_pylist() if path.exists() else []


def label_lakehouse(lakehouse: str | Path, idm_name: str) -> IdmStats:
    lakehouse = Path(lakehouse)
    idm = get_idm(idm_name)()
    print(f"[idm] idm={idm.name} version={idm.version}")

    embeddings = _read(lakehouse, "embedding")
    if not embeddings:
        print("[idm] no embeddings found — run the encoder first")
        return IdmStats()

    # idempotency: which source embeddings already labeled by this idm_version
    done = {(a["source_embedding_id"], a["idm_version"])
            for a in _read(lakehouse, "action_latent")
            if a.get("idm_version") == idm.version}

    # group embeddings by (stream, encoder_version), ordered in time
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for e in embeddings:
        groups[(e["stream_id"], e["encoder_version"])].append(e)
    for g in groups.values():
        g.sort(key=lambda e: e["t_start_ns"])

    writer = CanonicalWriter(lakehouse)
    st = IdmStats()
    for (stream_id, enc_ver), seq in groups.items():
        if len(seq) < 2:
            continue  # need a consecutive pair to infer a transition
        st.streams += 1
        for a, b in zip(seq, seq[1:]):
            if (a["embedding_id"], idm.version) in done:
                st.skipped += 1
                continue
            latent, conf = idm.infer(np.asarray(a["vector"], dtype=np.float32),
                                     np.asarray(b["vector"], dtype=np.float32))
            writer.add_action_latent(ActionLatent(
                latent_id=stable_id(a["embedding_id"], idm.version),
                episode_id=a["episode_id"], stream_id=stream_id,
                source_embedding_id=a["embedding_id"],
                encoder_version=enc_ver, idm_version=idm.version,
                t_start_ns=a["t_start_ns"], t_end_ns=b["t_start_ns"],
                latent=latent.tolist(), confidence=conf))
            st.latents += 1

    counts = writer.flush()
    print(f"[idm] streams={st.streams} latents={st.latents} skipped={st.skipped}")
    print(f"[idm] rows written: {counts}")
    return st


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Label embeddings with an IDM.")
    ap.add_argument("lakehouse")
    ap.add_argument("--idm", default="delta", help=f"one of: {list_idms()}")
    args = ap.parse_args()
    label_lakehouse(args.lakehouse, args.idm)
