# Pantheon Data Platform

A canonical schema + curation lakehouse that normalizes heterogeneous robot / egocentric
datasets into one queryable, training-ready shape — then encodes, labels, and packages
curated slices for training.

- **Part 1 — schema design:** [docs/canonical_schema.md](docs/canonical_schema.md)
  (spec + rationale); [docs/part1_schema.md](docs/part1_schema.md) (full reasoning).
- **Part 2 — ingestion & normalization:** [docs/part2_ingestion.md](docs/part2_ingestion.md).
- **Part 4 — scale, storage & training hand-off:** [docs/part4_scale.md](docs/part4_scale.md).

## What's built

An end-to-end pipeline that maps source datasets into the Part 1 canonical schema,
materializes them as **Lance** datasets (one per table — the lakehouse), then runs the
derived tier (embeddings + action latents) and packages curated subsets for training.

```
                ┌──────── curation pipeline (per source) ────────┐
  realdata/  ─▶ ingest ─▶  QC  ─▶ encode ─▶  IDM        ──▶  lakehouse (Lance)
                 │         │       │          │                catalog + derived tiers
            adapters   (Part 3   CLIP     delta(IDM)               │
            (registry) placeholder) (registry) (registry)          ▼
                                                          Delivery export (manual)
                                                          portable Parquet + media bundle
```

Every stage is **registry-pluggable** (adapters, encoders, IDMs each self-register), so
new datasets/models drop in without touching the core. The **delivery export** is
deliberately *not* part of the pipeline — it's a user-driven packaging step.

### Adapters (ingest coverage)

| adapter | source class | stresses |
|---|---|---|
| `ego_raw` | raw egocentric video (mp4 / `.tar` shards) | the **label-less** minimal episode; messy data (corrupt files, dupes, archives) |
| `egodex`  | richly annotated ego (mp4 + paired hdf5) | **pose streams**, **moving-camera calibration**, language captions, **partial degradation** |
| `lerobot` | cross-embodiment robot teleop (LeRobot v3: parquet + packed mp4) | simultaneous **proprio + dense action** streams referencing parquet columns; **multi-camera rig**; episode-as-time-segment in a packed mp4 |

New datasets are added as **self-registering adapters** — adapters extend the *ingester*,
never the schema (Trossen-style registry pattern).

### Derived-tier producers

| stage | registry | built-in | fills | notes |
|---|---|---|---|---|
| `encode` | `@register_encoder` | `clip-vit-b32` (open_clip ViT-B/32) | `embedding` | encode-once over pixels; the one place we read cold payloads |
| `idm` | `@register_idm` | `delta` (normalized Δembedding) | `action_latent` | keys off embeddings (not pixels); `delta` is a **placeholder** baseline |
| `qc` | — | placeholder | `quality_signal` | Part 3 seam; reports what it *would* inspect |

## Layout

```
pantheon/
  schema/         enums, record dataclasses (records.py), Arrow table schemas (tables.py)
  registry.py     adapter registry + select_adapter(root) via probe()
  adapters/       base.py (access vs mapping split) + ego_raw, egodex, lerobot
  io/             video.py (PyAV probe/decode + truncation detection), hashing.py (dedup/ids)
  writer.py       buffers rows, batch-writes Lance datasets
  ingest.py       source-agnostic ingest loop + CLI; quarantines failures
  qc.py           QC pass (Part 3 placeholder)
  encoders/       base + registry + clip_encoder
  encode.py       encode-once driver + CLI -> embedding (builds ANN index at scale)
  idms/           base + registry + delta_idm
  idm.py          IDM driver + CLI -> action_latent
  export.py       delivery export (curated slice -> portable Parquet + media bundle)
  query.py        example queries over the lakehouse (LanceDB)
server.py         FastAPI: pipeline orchestration + /api/export* + lakehouse views
frontend/         React (Vite) UI — pipeline runner, lakehouse viewer, delivery packager
scripts/          make_fixtures.py, dedup_lakehouse.py
```

Storage is standardized on **Lance** (catalog *and* derived tiers); queries run via
**LanceDB / DuckDB-style** scans over the Lance datasets. `oneof`/structs flatten to a
discriminator + JSON branch columns.

## Run it

### UI (recommended)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# terminal 1 — API backend
python server.py

# terminal 2 — React frontend
cd frontend && npm install && npm run dev
```

Open **http://localhost:5173**:
1. drop a dataset folder into `realdata/`,
2. click **Run** on it → the **ingest → QC → encode → IDM** pipeline streams a live,
   per-stage log,
3. inspect the lakehouse tables/queries,
4. in **Delivery**, select a slice and **Package** a portable training bundle.

### CLI

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. generate synthetic, messy fixtures (or point at real data in realdata/)
python scripts/make_fixtures.py

# 2. pipeline stages (each auto-selects its plugin)
python -m pantheon.ingest fixtures/ego_raw --lakehouse ./lakehouse   # adapter via probe()
python -m pantheon.qc     ./lakehouse                                # placeholder
python -m pantheon.encode ./lakehouse --encoder clip-vit-b32        # -> embedding
python -m pantheon.idm    ./lakehouse --idm delta                   # -> action_latent

# 3. query the lakehouse
python -m pantheon.query ./lakehouse

# 4. package a curated delivery bundle for training
python -m pantheon.export ./lakehouse --name my_bundle \
    --sources egodex,lerobot_unitreeh1 --embeddings --action-latents --copy-media

# maintenance: dedup catalog tables (re-ingest is not yet idempotent)
python scripts/dedup_lakehouse.py ./lakehouse

# tests
pytest -q
```

## Delivery bundle structure

`python -m pantheon.export` (or the UI Delivery panel) writes a self-contained,
**Lance-decoupled** bundle to `exports/<name>/`:

```
exports/<name>/
├── manifest.json          selection predicate + encoder/idm/ingest versions (reproducibility)
├── episode|stream|clock|annotation|calibration|quality_signal.parquet   catalog
├── source|embodiment.parquet                                            registry
├── embedding.parquet | action_latent.parquet                           derived (if selected)
└── media/...              copied native payloads, path-mirrored (if "copy media")
```

Catalog + derived Parquet are tiny (KBs); `media/` carries the heavy pixels and is copied
only when you ask for a self-contained bundle. `stream.parquet` payload URIs are rewritten
into `media/` so the bundle is relocatable.

## Messy-data handling (no silent drops / crashes)

| problem | behavior | result |
|---|---|---|
| corrupt/unopenable mp4 or tar | caught at asset-access | `quarantine` row (stage=access) |
| missing hdf5 side-car | emit video-only episode | `partial` + `hdf5_missing` signal |
| missing camera intrinsics | emit streams, no calib | `partial` + `intrinsics_missing` signal |
| content-duplicate clip | checksum dedup | `duplicate_clip` signal (kept + flagged) |
| truncated/incomplete packed video | degrade per-stream | `partial` + `video_unreadable`/`video_segment_incomplete` |

## Notes / status / future work

- **Derived tier is populated**: `encode` (real CLIP ViT-B/32) fills `embedding`; `idm`
  fills `action_latent`. The `delta` IDM is a **placeholder** — a genuine IDM is a small
  model *trained on the embeddings* (CLAM/DynaMo-style); it drops in behind `@register_idm`.
- **QC is a placeholder** (`pantheon/qc.py`) — the real Part 3 checks (blur, exposure,
  dropped/duplicate frames, near-dup clustering, manipulation-present) are the next build.
- **Re-ingest appends** (not idempotent). `scripts/dedup_lakehouse.py` cleans duplicates
  by PK; the real fix is `merge_insert` on the deterministic primary keys.
- **ANN index** needs the `embedding.vector` column declared as `FixedSizeList(float32, d)`
  (currently variable `List`); below 256 rows the encoder uses exact search.
- `droid` (RLDS: multi-camera + dual-space actions) is the natural next adapter.
- `fixtures/`, `lakehouse/`, `realdata/`, `exports/` are generated artifacts (gitignored).

## Example query output

Reflects the synthetic-fixture run (`scripts/make_fixtures.py`); real datasets in
`realdata/` produce analogous, larger tables.

```
episodes_by_source_and_status   EgoDex: 1 ok / 2 partial ; ego_raw: 4 ok
streams_by_modality             pose_se3: 8 ; video: 7
label_less_episodes             5   (raw clips + the egodex clip whose hdf5 was missing)
moving_camera_calibrations      1   (video extrinsics → the pose/camera stream)
quality_signal_counts           uncalibrated:6 duplicate_clip:1 intrinsics_missing:1 hdf5_missing:1
quarantined                     access: 2   (corrupt mp4 + corrupt tar — surfaced, not dropped)
```
