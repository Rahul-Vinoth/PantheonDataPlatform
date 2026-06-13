"""Quality-control pass (Part 3) — PLACEHOLDER.

Real QC will scan video/image streams for blur, exposure, dropped/duplicate frames,
near-duplicate clustering, and a manipulation-present classifier, writing the results as
`QualitySignal` rows. For now this is a no-op stage that reports what it *would* inspect,
so the ingest -> QC -> encode pipeline is wired end-to-end and has a real seam to drop
the implementation into later.

Usage:
    python -m pantheon.qc ./lakehouse
"""
from __future__ import annotations

from pathlib import Path

import lance


def _count(lakehouse: Path, name: str) -> int:
    path = lakehouse / f"{name}.lance"
    return lance.dataset(str(path)).count_rows() if path.exists() else 0


def run_qc(lakehouse: str | Path) -> int:
    lakehouse = Path(lakehouse)
    episodes = _count(lakehouse, "episode")

    n_video = 0
    stream_path = lakehouse / "stream.lance"
    if stream_path.exists():
        rows = lance.dataset(str(stream_path)).to_table().to_pylist()
        n_video = sum(1 for s in rows if s["modality"] in ("video", "image_seq"))

    print("[qc] PLACEHOLDER — no checks implemented yet (Part 3)")
    print(f"[qc] would inspect {n_video} video/image streams across {episodes} episodes")
    print("[qc] planned checks: blur · exposure · dropped/duplicate frames · "
          "near-duplicate clusters · manipulation-present · task/env tags")
    print("[qc] wrote 0 quality_signal rows (placeholder)")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(run_qc(sys.argv[1] if len(sys.argv) > 1 else "./lakehouse"))
