"""Pantheon Data Platform — FastAPI backend.

Endpoints:
  GET  /api/sources          list subdirs of realdata/
  POST /api/ingest           start an ingest run (one at a time)
  GET  /api/ingest/status    current job status + log
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
_job: dict = {"status": "idle", "source": None, "log": [], "returncode": None}


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
    _job["log"] = []
    _job["returncode"] = None

    asyncio.create_task(_run_ingest(source_path))
    return {"status": "started", "source": req.source}


async def _run_ingest(source_path: Path) -> None:
    cmd = [PYTHON, "-m", "pantheon.ingest", str(source_path),
           "--lakehouse", str(LAKEHOUSE)]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(ROOT),
        )
        assert proc.stdout
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            _job["log"].append(line)
        await proc.wait()
        _job["returncode"] = proc.returncode
        _job["status"] = "done" if proc.returncode == 0 else "error"
    except Exception as exc:
        _job["log"].append(f"[server error] {exc}")
        _job["status"] = "error"


@app.get("/api/ingest/status")
def ingest_status():
    return {
        "status": _job["status"],
        "source": _job["source"],
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


# -------------------------------------------------------------------------- static
# Serve built React frontend. Must be mounted last so /api routes take priority.
_dist = ROOT / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
