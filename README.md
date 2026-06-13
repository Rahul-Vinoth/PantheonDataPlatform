# Pantheon Data Platform

A canonical schema + curation lakehouse that normalizes heterogeneous robot / egocentric
datasets into one queryable, training-ready shape.

- **Part 1 — schema design:** [docs/canonical_schema.md](docs/canonical_schema.md)
  (spec + rationale); [docs/part1_schema.md](docs/part1_schema.md) (full reasoning).
- **Part 2 — ingestion & normalization (this code):**
  [docs/part2_ingestion.md](docs/part2_ingestion.md).
- **Part 4 — scale, storage & training hand-off:**
  [docs/part4_scale.md](docs/part4_scale.md).

## What's built

A Python ingester that maps source datasets into the Part 1 canonical schema and
materializes them as **Lance** datasets (one per table — the lakehouse). Two adapters:

| adapter | source class | stresses |
|---|---|---|
| `ego_raw` | raw egocentric video (mp4 / `.tar` shards) | the **label-less** minimal episode; messy data (corrupt files, dupes, archives) |
| `egodex`  | richly annotated ego (mp4 + paired hdf5) | **pose streams**, **moving-camera calibration**, language captions, **partial degradation** |
| `lerobot` | cross-embodiment robot teleop (LeRobot v3: parquet + packed mp4) | simultaneous **proprio + dense action** streams referencing parquet columns; **multi-camera rig**; episode-as-time-segment in a packed mp4 |

New datasets are added as **self-registering adapters** — no changes to the core
(Trossen-style registry pattern). Adapters extend the *ingester*, never the schema.

## Architecture

```
                  @register_adapter("…")
  source files ──▶ SourceAdapter            uniform emit          one writer
   (mp4/hdf5/   ──▶  .probe()         ─────────────────────▶  CanonicalWriter ─▶ Lance
    tar/…)            .iter_episodes()   Episode/Stream/         (catalog +
                      .emit()            Annotation/…  records    derived datasets)
                                         + Quarantine on failure)
```

- `pantheon/schema/` — `enums`, record dataclasses (`records.py`), Arrow table schemas
  (`tables.py`). `oneof`/structs flatten to discriminator + JSON branch columns.
- `pantheon/registry.py` — the adapter registry + `select_adapter(root)` via `probe()`.
- `pantheon/adapters/` — `base.py` (contract: access vs mapping split) + `ego_raw`, `egodex`.
- `pantheon/io/` — `video.py` (PyAV probe + truncation detection), `hashing.py` (dedup/ids).
- `pantheon/writer.py` — buffers rows, batch-writes to Lance datasets.
- `pantheon/ingest.py` — source-agnostic loop + CLI; quarantines failures.
- `pantheon/query.py` — example DuckDB-over-Lance reports.

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

Open **http://localhost:5173** — drop data into `realdata/`, click Ingest, watch the
live log, then inspect the lakehouse tables and queries.

### CLI

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. generate synthetic, messy fixtures
python scripts/make_fixtures.py

# 2. ingest (auto-selects the adapter via probe())
python -m pantheon.ingest fixtures/ego_raw --lakehouse ./lakehouse
python -m pantheon.ingest fixtures/egodex  --lakehouse ./lakehouse

# 3. query the lakehouse
python -m pantheon.query ./lakehouse

# tests
pytest -q
```

## What the output shows

```
episodes_by_source_and_status   EgoDex: 1 ok / 2 partial ; ego_raw: 4 ok
streams_by_modality             pose_se3: 8 ; video: 7
label_less_episodes             5   (raw clips + the egodex clip whose hdf5 was missing)
moving_camera_calibrations      1   (video extrinsics → the pose/camera stream)
quality_signal_counts           uncalibrated:6 duplicate_clip:1 intrinsics_missing:1 hdf5_missing:1
quarantined                     access: 2   (corrupt mp4 + corrupt tar — surfaced, not dropped)
```

## Messy-data handling (no silent drops / crashes)

| problem | behavior | result |
|---|---|---|
| corrupt/unopenable mp4 or tar | caught at asset-access | `quarantine` row (stage=access) |
| missing hdf5 side-car | emit video-only episode | `partial` + `hdf5_missing` signal |
| missing camera intrinsics | emit pose streams, no calib | `partial` + `intrinsics_missing` signal |
| content-duplicate clip | checksum dedup | `duplicate_clip` signal (kept + flagged) |

## Notes / future work

- Derived tables (`embedding`, `action_latent`) are declared but not populated — that's
  Part 3+ (encode-once / re-label-often).
- Re-ingest currently appends; idempotent upsert via Lance `merge_insert` on primary
  keys is straightforward future work (ids are already deterministic).
- `droid` (RLDS: multi-camera + dual-space actions) is the natural next adapter.
- `fixtures/` and `lakehouse/` are generated artifacts.
