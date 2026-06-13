"""Delivery export (Part 4 §4.3) — package a curated lakehouse subset for training.

This is NOT part of the ingest->QC->encode->IDM pipeline. It is an explicit, user-driven
step: the user selects which slice of the lakehouse to package, and this materializes a
portable, self-contained "delivery object" that a trainer consumes directly — decoupled
from our internal Lance format.

The export is written as **Parquet** (universal, not Lance) plus an optional copy of the
referenced media, and a `manifest.json` stamping the selection predicate and the
encoder/idm/ingest versions so a training run is reproducible.

Layout:
    exports/<name>/
      manifest.json
      episode.parquet  stream.parquet  annotation.parquet  calibration.parquet
      embedding.parquet            (if included)
      action_latent.parquet        (if included)
      source.parquet  embodiment.parquet
      media/...                    (if copy_media; referenced payloads, deduped)

Usage:
    python -m pantheon.export ./lakehouse --name my_bundle \
        --sources egodex,lerobot_unitreeh1 --embeddings --action-latents --copy-media
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import lance
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def _ds(lakehouse: Path, name: str) -> Optional[pa.Table]:
    path = lakehouse / f"{name}.lance"
    return lance.dataset(str(path)).to_table() if path.exists() else None


def _by_episode(t: Optional[pa.Table], ids: set[str]) -> Optional[pa.Table]:
    if t is None or t.num_rows == 0:
        return t
    return t.filter(pc.is_in(t["episode_id"], value_set=pa.array(sorted(ids))))


def build_export(
    lakehouse: str | Path,
    name: str,
    *,
    out_root: str | Path = "exports",
    sources: Optional[list[str]] = None,
    embodiment_kinds: Optional[list[str]] = None,
    include_partial: bool = True,
    include_embeddings: bool = True,
    include_action_latents: bool = True,
    copy_media: bool = True,
    encoder_version: Optional[str] = None,
    idm_version: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> dict:
    lakehouse = Path(lakehouse)
    project_root = lakehouse.parent
    out_dir = Path(out_root) / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"[export] building delivery '{name}' from {lakehouse}")

    episodes = _ds(lakehouse, "episode")
    if episodes is None or episodes.num_rows == 0:
        raise ValueError("no episodes in lakehouse")

    embodiments = _ds(lakehouse, "embodiment")
    kind_of = {r["embodiment_id"]: r["kind"]
               for r in (embodiments.to_pylist() if embodiments else [])}

    # ----- selection predicate -----
    selected = []
    for e in episodes.to_pylist():
        if sources and e["source_id"] not in sources:
            continue
        if embodiment_kinds and kind_of.get(e["embodiment_id"]) not in embodiment_kinds:
            continue
        if not include_partial and e["quality_status"] != "ok":
            continue
        selected.append(e)
    sel_ids = {e["episode_id"] for e in selected}
    if not sel_ids:
        raise ValueError("selection matched zero episodes")
    log(f"[export] selected {len(sel_ids)} episodes")

    # ----- subset catalog tables -----
    streams = _by_episode(_ds(lakehouse, "stream"), sel_ids)
    annotations = _by_episode(_ds(lakehouse, "annotation"), sel_ids)
    clocks = _by_episode(_ds(lakehouse, "clock"), sel_ids)
    quality = _by_episode(_ds(lakehouse, "quality_signal"), sel_ids)

    # calibration has no episode_id — it links via calib_id prefix "<episode_id>/..."
    calibrations = _ds(lakehouse, "calibration")
    if calibrations is not None and calibrations.num_rows:
        keep = [r for r in calibrations.to_pylist()
                if r["calib_id"].split("/")[0] in sel_ids]
        calibrations = pa.Table.from_pylist(keep, schema=calibrations.schema) \
            if keep else calibrations.slice(0, 0)

    ep_tbl = episodes.filter(pc.is_in(episodes["episode_id"], value_set=pa.array(sorted(sel_ids))))

    # source/embodiment subset to what the selection references
    src_ids = {e["source_id"] for e in selected}
    emb_ids = {e["embodiment_id"] for e in selected}
    src_tbl = _ds(lakehouse, "source")
    if src_tbl is not None:
        src_tbl = src_tbl.filter(pc.is_in(src_tbl["source_id"], value_set=pa.array(sorted(src_ids))))
    emb_tbl = embodiments
    if emb_tbl is not None:
        emb_tbl = emb_tbl.filter(pc.is_in(emb_tbl["embodiment_id"], value_set=pa.array(sorted(emb_ids))))

    # ----- derived tier (optional) -----
    embeddings = _by_episode(_ds(lakehouse, "embedding"), sel_ids) if include_embeddings else None
    if embeddings is not None and encoder_version:
        embeddings = embeddings.filter(pc.equal(embeddings["encoder_version"], encoder_version))
    latents = _by_episode(_ds(lakehouse, "action_latent"), sel_ids) if include_action_latents else None
    if latents is not None and idm_version:
        latents = latents.filter(pc.equal(latents["idm_version"], idm_version))

    # ----- media copy + payload_uri rewrite -----
    media_files = 0
    if copy_media and streams is not None and streams.num_rows:
        uris = streams.column("payload_uri").to_pylist()
        new_uris, copied = [], {}
        for u in uris:
            if not u or str(u).startswith("tar://"):
                new_uris.append(u)
                continue
            p = Path(u)
            if not p.exists():
                new_uris.append(u)
                continue
            if u in copied:
                new_uris.append(copied[u])
                continue
            # mirror under media/, preserving a relative path to avoid collisions
            if p.is_absolute():
                try:
                    rel = p.relative_to(project_root)
                except ValueError:
                    rel = Path(p.name)
            else:
                rel = p
            dest = out_dir / "media" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
            relpath = str(Path("media") / rel)
            copied[u] = relpath
            new_uris.append(relpath)
            media_files += 1
        idx = streams.schema.get_field_index("payload_uri")
        streams = streams.set_column(idx, "payload_uri", pa.array(new_uris))
        log(f"[export] copied {media_files} media files into media/")

    # ----- write parquet -----
    def write(t: Optional[pa.Table], fname: str) -> int:
        if t is None:
            return 0
        pq.write_table(t, str(out_dir / fname))
        return t.num_rows

    counts = {
        "episode": write(ep_tbl, "episode.parquet"),
        "stream": write(streams, "stream.parquet"),
        "clock": write(clocks, "clock.parquet"),
        "annotation": write(annotations, "annotation.parquet"),
        "calibration": write(calibrations, "calibration.parquet"),
        "quality_signal": write(quality, "quality_signal.parquet"),
        "source": write(src_tbl, "source.parquet"),
        "embodiment": write(emb_tbl, "embodiment.parquet"),
        "embedding": write(embeddings, "embedding.parquet"),
        "action_latent": write(latents, "action_latent.parquet"),
        "media_files": media_files,
    }

    # ----- manifest (reproducibility stamp) -----
    enc_versions = sorted(set(embeddings.column("encoder_version").to_pylist())) if embeddings is not None and embeddings.num_rows else []
    idm_versions = sorted(set(latents.column("idm_version").to_pylist())) if latents is not None and latents.num_rows else []
    run_ids = sorted({e["ingest_run_id"] for e in selected})

    manifest = {
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lakehouse": str(lakehouse),
        "format": "parquet",
        "selection": {
            "sources": sources, "embodiment_kinds": embodiment_kinds,
            "include_partial": include_partial,
            "include_embeddings": include_embeddings,
            "include_action_latents": include_action_latents,
            "copy_media": copy_media,
            "encoder_version": encoder_version, "idm_version": idm_version,
        },
        "counts": counts,
        "encoder_versions": enc_versions,
        "idm_versions": idm_versions,
        "ingest_run_ids": run_ids,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"[export] wrote {out_dir}/  counts={counts}")
    return manifest


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Package a lakehouse subset for training.")
    ap.add_argument("lakehouse")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-root", default="exports")
    ap.add_argument("--sources", default=None, help="comma-separated source_ids")
    ap.add_argument("--embodiment-kinds", default=None, help="comma-separated: human,robot,unknown")
    ap.add_argument("--ok-only", action="store_true", help="exclude partial episodes")
    ap.add_argument("--embeddings", action="store_true", help="include embeddings")
    ap.add_argument("--action-latents", action="store_true", help="include action latents")
    ap.add_argument("--copy-media", action="store_true", help="copy referenced media into the bundle")
    ap.add_argument("--encoder-version", default=None)
    ap.add_argument("--idm-version", default=None)
    args = ap.parse_args()

    build_export(
        args.lakehouse, args.name, out_root=args.out_root,
        sources=args.sources.split(",") if args.sources else None,
        embodiment_kinds=args.embodiment_kinds.split(",") if args.embodiment_kinds else None,
        include_partial=not args.ok_only,
        include_embeddings=args.embeddings,
        include_action_latents=args.action_latents,
        copy_media=args.copy_media,
        encoder_version=args.encoder_version, idm_version=args.idm_version,
    )
