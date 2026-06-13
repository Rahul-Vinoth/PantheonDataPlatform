"""Example queries over the lakehouse (LanceDB).

Demonstrates that the materialized schema is queryable and that the entity model makes
key questions structural — e.g. the label-less corpus, multi-rate streams, moving-camera
calibration, and quarantined units.

Each query operates on one or more LanceDB tables. Cross-table joins are done in Python
by matching on shared keys (episode_id, source_id) rather than SQL JOIN syntax.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import lancedb


def _db(lakehouse: str | Path) -> lancedb.DBConnection:
    return lancedb.connect(str(lakehouse))


def _open(db: lancedb.DBConnection, name: str) -> list[dict]:
    """Return all rows from a table as a list of dicts, or [] if table absent."""
    try:
        return db.open_table(name).to_arrow().to_pylist()
    except Exception:
        return []


# --------------------------------------------------------------------------- queries

def episodes_by_source_and_status(db: lancedb.DBConnection) -> list[dict]:
    sources = {r["source_id"]: r["name"] for r in _open(db, "source")}
    episodes = _open(db, "episode")
    counts: dict[tuple, int] = defaultdict(int)
    for e in episodes:
        key = (sources.get(e["source_id"], e["source_id"]), e["quality_status"])
        counts[key] += 1
    return [{"source": k[0], "quality_status": k[1], "n": v}
            for k, v in sorted(counts.items())]


def streams_by_modality(db: lancedb.DBConnection) -> list[dict]:
    counts: dict[str, int] = defaultdict(int)
    for s in _open(db, "stream"):
        counts[s["modality"]] += 1
    return [{"modality": m, "n": n}
            for m, n in sorted(counts.items(), key=lambda x: -x[1])]


def label_less_episodes(db: lancedb.DBConnection) -> list[dict]:
    """Episodes with no annotations and no action streams (§5.5 label-less corpus)."""
    annotated = {a["episode_id"] for a in _open(db, "annotation")}
    has_action = {s["episode_id"] for s in _open(db, "stream")
                  if s.get("role_class") == "action"}
    return [{"episode_id": e["episode_id"]}
            for e in _open(db, "episode")
            if e["episode_id"] not in annotated
            and e["episode_id"] not in has_action]


def moving_camera_calibrations(db: lancedb.DBConnection) -> list[dict]:
    return [{"calib_id": c["calib_id"],
             "target_ref": c["target_ref"],
             "extr_dynamic_stream_id": c["extr_dynamic_stream_id"]}
            for c in _open(db, "calibration")
            if c.get("extrinsics_kind") == "dynamic"]


def quality_signal_counts(db: lancedb.DBConnection) -> list[dict]:
    counts: dict[str, int] = defaultdict(int)
    for q in _open(db, "quality_signal"):
        counts[q["signal_type"]] += 1
    return [{"signal_type": t, "n": n}
            for t, n in sorted(counts.items(), key=lambda x: -x[1])]


def quarantined(db: lancedb.DBConnection) -> list[dict]:
    counts: dict[str, int] = defaultdict(int)
    for q in _open(db, "quarantine"):
        counts[q["stage"]] += 1
    return [{"stage": s, "n": n} for s, n in sorted(counts.items())]


# --------------------------------------------------------------------------- report

QUERIES = [
    ("episodes_by_source_and_status", episodes_by_source_and_status),
    ("streams_by_modality",           streams_by_modality),
    ("label_less_episodes",           label_less_episodes),
    ("moving_camera_calibrations",    moving_camera_calibrations),
    ("quality_signal_counts",         quality_signal_counts),
    ("quarantined",                   quarantined),
]


def _fmt(rows: list[dict]) -> str:
    if not rows:
        return "  (no rows)"
    keys = list(rows[0].keys())
    col_w = {k: max(len(k), max(len(str(r[k])) for r in rows)) for k in keys}
    header = "  " + "  ".join(k.ljust(col_w[k]) for k in keys)
    sep    = "  " + "  ".join("-" * col_w[k] for k in keys)
    lines  = [header, sep]
    for r in rows:
        lines.append("  " + "  ".join(str(r[k]).ljust(col_w[k]) for k in keys))
    return "\n".join(lines)


def run_report(lakehouse: str | Path) -> None:
    db = _db(lakehouse)
    for title, fn in QUERIES:
        print(f"\n=== {title} ===")
        try:
            print(_fmt(fn(db)))
        except Exception as e:
            print(f"  (skipped: {e})")


if __name__ == "__main__":
    import sys
    run_report(sys.argv[1] if len(sys.argv) > 1 else "./lakehouse")
