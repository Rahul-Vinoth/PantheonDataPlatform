"""Pantheon Data Platform — FastAPI backend.

Runs the curation pipeline (ingest -> QC -> encode) for a source, one at a time, and
streams its stage-by-stage log to the frontend.

Endpoints:
  GET  /api/sources          list subdirs of realdata/
  POST /api/ingest           start the pipeline for a source
  GET  /api/ingest/status    current job status + stage + log length
  GET  /api/ingest/stream    SSE stream of live log lines
  GET  /api/lakehouse        table row-counts + query results
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).parent
REALDATA = ROOT / "realdata"
LAKEHOUSE = ROOT / "lakehouse"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

app = FastAPI(title="Pantheon Data Platform")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------------- job state
ENCODER = "clip-vit-b32"
IDM = "delta"
PIPELINE_STAGES = ["ingest", "qc", "encode", "idm"]
_job: dict = {"status": "idle", "source": None, "stage": None, "stages": PIPELINE_STAGES,
              "log": [], "returncode": None}


# -------------------------------------------------------------------------- sources

@app.get("/api/sources")
def list_sources():
    if not REALDATA.exists():
        return {"sources": []}
    sources = []
    for p in sorted(REALDATA.iterdir()):
        if p.is_dir():
            files = list(p.rglob("*"))
            sources.append({
                "name": p.name,
                "path": str(p.relative_to(ROOT)),
                "file_count": sum(1 for f in files if f.is_file()),
            })
    return {"sources": sources}


# -------------------------------------------------------------------------- ingest

class IngestRequest(BaseModel):
    source: str  # subfolder name under realdata/


@app.post("/api/ingest")
async def start_ingest(req: IngestRequest):
    if _job["status"] == "running":
        raise HTTPException(409, "An ingest job is already running.")

    source_path = REALDATA / req.source
    if not source_path.exists():
        raise HTTPException(404, f"Source not found: {req.source}")

    _job["status"] = "running"
    _job["source"] = req.source
    _job["stage"] = None
    _job["log"] = []
    _job["returncode"] = None

    asyncio.create_task(_run_pipeline(source_path))
    return {"status": "started", "source": req.source}


def _pipeline_cmds(source_path: Path) -> list[tuple[str, list[str]]]:
    """The three pipeline stages, in order. QC is a placeholder for now (Part 3)."""
    return [
        ("ingest", [PYTHON, "-m", "pantheon.ingest", str(source_path),
                    "--lakehouse", str(LAKEHOUSE)]),
        ("qc",     [PYTHON, "-m", "pantheon.qc", str(LAKEHOUSE)]),
        ("encode", [PYTHON, "-m", "pantheon.encode", str(LAKEHOUSE),
                    "--encoder", ENCODER]),
        ("idm",    [PYTHON, "-m", "pantheon.idm", str(LAKEHOUSE),
                    "--idm", IDM]),
    ]


async def _run_stage(stage: str, cmd: list[str]) -> int:
    """Run one stage, streaming its stdout into the shared log. Returns its exit code."""
    _job["stage"] = stage
    _job["log"].append(f"━━━ stage: {stage} ━━━")
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT, cwd=str(ROOT),
    )
    assert proc.stdout
    async for raw in proc.stdout:
        _job["log"].append(raw.decode(errors="replace").rstrip())
    await proc.wait()
    return proc.returncode or 0


async def _run_pipeline(source_path: Path) -> None:
    """ingest -> QC -> encode, sequentially. Stop at the first failing stage."""
    try:
        for stage, cmd in _pipeline_cmds(source_path):
            rc = await _run_stage(stage, cmd)
            if rc != 0:
                _job["log"].append(f"[pipeline] stage '{stage}' failed (rc={rc}); stopping")
                _job["returncode"] = rc
                _job["status"] = "error"
                return
        _job["returncode"] = 0
        _job["status"] = "done"
        _job["stage"] = None
    except Exception as exc:
        _job["log"].append(f"[server error] {exc}")
        _job["status"] = "error"


@app.get("/api/ingest/status")
def ingest_status():
    return {
        "status": _job["status"],
        "source": _job["source"],
        "stage": _job["stage"],
        "stages": _job["stages"],
        "lines": len(_job["log"]),
        "returncode": _job["returncode"],
    }


@app.get("/api/ingest/stream")
async def ingest_stream():
    """SSE endpoint: replays buffered lines then follows live output."""
    async def generate() -> AsyncIterator[str]:
        idx = 0
        while True:
            while idx < len(_job["log"]):
                yield f"data: {json.dumps(_job['log'][idx])}\n\n"
                idx += 1
            if _job["status"] != "running" and idx >= len(_job["log"]):
                yield f"data: {json.dumps('__done__')}\n\n"
                break
            await asyncio.sleep(0.1)

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# -------------------------------------------------------------------------- lakehouse

@app.get("/api/lakehouse")
def lakehouse_state():
    if not LAKEHOUSE.exists():
        return {"tables": {}, "queries": {}}

    import lancedb
    from pantheon.query import (
        episodes_by_source_and_status,
        streams_by_modality,
        label_less_episodes,
        moving_camera_calibrations,
        quality_signal_counts,
        quarantined as quarantined_q,
    )
    from pantheon.schema.tables import CATALOG_TABLES, DERIVED_TABLES

    db = lancedb.connect(str(LAKEHOUSE))

    tables: dict[str, int] = {}
    for name in CATALOG_TABLES + DERIVED_TABLES:
        path = LAKEHOUSE / f"{name}.lance"
        if path.exists():
            try:
                tables[name] = db.open_table(name).count_rows()
            except Exception:
                tables[name] = -1

    def safe(fn):
        try:
            return fn(db)
        except Exception as e:
            return {"error": str(e)}

    return {
        "tables": tables,
        "queries": {
            "episodes_by_source_and_status": safe(episodes_by_source_and_status),
            "streams_by_modality": safe(streams_by_modality),
            "label_less_episodes": safe(label_less_episodes),
            "moving_camera_calibrations": safe(moving_camera_calibrations),
            "quality_signal_counts": safe(quality_signal_counts),
            "quarantined": safe(quarantined_q),
        },
    }


# -------------------------------------------------------------------------- export
EXPORTS_DIR = ROOT / "exports"
_export_job: dict = {"status": "idle", "name": None, "log": [], "manifest": None}


@app.get("/api/export/options")
def export_options():
    """What the user can package: sources present in the catalog (with episode counts),
    embodiment kinds, and which derived versions exist."""
    import lance

    def tbl(name):
        p = LAKEHOUSE / f"{name}.lance"
        return lance.dataset(str(p)).to_table().to_pylist() if p.exists() else []

    eps = tbl("episode")
    src_names = {s["source_id"]: s["name"] for s in tbl("source")}
    kind_of = {e["embodiment_id"]: e["kind"] for e in tbl("embodiment")}
    from collections import Counter
    by_src = Counter(e["source_id"] for e in eps)
    sources = [{"source_id": sid, "name": src_names.get(sid, sid), "episodes": n}
               for sid, n in sorted(by_src.items())]
    kinds = sorted({kind_of.get(e["embodiment_id"], "unknown") for e in eps})
    enc = sorted({e["encoder_version"] for e in tbl("embedding")})
    idm = sorted({a["idm_version"] for a in tbl("action_latent")})
    return {"sources": sources, "embodiment_kinds": kinds,
            "encoder_versions": enc, "idm_versions": idm,
            "has_embeddings": bool(enc), "has_action_latents": bool(idm)}


class ExportRequest(BaseModel):
    name: str
    sources: list[str] | None = None
    embodiment_kinds: list[str] | None = None
    include_partial: bool = True
    include_embeddings: bool = True
    include_action_latents: bool = True
    copy_media: bool = True


@app.post("/api/export")
async def start_export(req: ExportRequest):
    if _export_job["status"] == "running":
        raise HTTPException(409, "An export is already running.")
    name = "".join(c for c in req.name if c.isalnum() or c in "-_") or "delivery"
    _export_job.update(status="running", name=name, log=[], manifest=None)

    def _log(line: str):
        _export_job["log"].append(line)

    async def _run():
        from pantheon.export import build_export
        try:
            manifest = await asyncio.to_thread(
                build_export, str(LAKEHOUSE), name, out_root=str(EXPORTS_DIR),
                sources=req.sources or None,
                embodiment_kinds=req.embodiment_kinds or None,
                include_partial=req.include_partial,
                include_embeddings=req.include_embeddings,
                include_action_latents=req.include_action_latents,
                copy_media=req.copy_media, log=_log)
            _export_job["manifest"] = manifest
            _export_job["status"] = "done"
        except Exception as exc:
            _export_job["log"].append(f"[export error] {exc}")
            _export_job["status"] = "error"

    asyncio.create_task(_run())
    return {"status": "started", "name": name}


@app.get("/api/export/status")
def export_status():
    return {"status": _export_job["status"], "name": _export_job["name"],
            "lines": len(_export_job["log"]), "manifest": _export_job["manifest"]}


@app.get("/api/export/stream")
async def export_stream():
    async def generate() -> AsyncIterator[str]:
        idx = 0
        while True:
            while idx < len(_export_job["log"]):
                yield f"data: {json.dumps(_export_job['log'][idx])}\n\n"
                idx += 1
            if _export_job["status"] != "running" and idx >= len(_export_job["log"]):
                yield f"data: {json.dumps('__done__')}\n\n"
                break
            await asyncio.sleep(0.1)
    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/exports")
def list_exports():
    out = []
    if EXPORTS_DIR.exists():
        for d in sorted(EXPORTS_DIR.iterdir()):
            mf = d / "manifest.json"
            if mf.exists():
                try:
                    out.append(json.loads(mf.read_text()))
                except Exception:
                    pass
    return {"exports": out}


# -------------------------------------------------------------------------- static
# Serve built React frontend. Must be mounted last so /api routes take priority.
_dist = ROOT / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
