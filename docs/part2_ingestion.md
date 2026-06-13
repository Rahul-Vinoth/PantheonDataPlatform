# Part 2 — Ingestion & Normalization (design)

Maps heterogeneous source datasets into the Part 1 canonical schema
([canonical_schema.md](canonical_schema.md)) and materializes them as queryable Lance
datasets. Messy real data is handled by surfacing and quarantining problems, never
silently dropping or crashing.

> **Status: built.** Runnable implementation + instructions in
> [../README.md](../README.md); code under `pantheon/`, tests under `tests/`. Two
> adapters (`ego_raw`, `egodex`) ingest synthetic messy fixtures end-to-end into a Lance
> lakehouse, all queryable via DuckDB. This document is the design behind that code.

---

## 1. Two axes of extensibility (keep them separate)

| axis | extends | mechanism | touches schema? |
|---|---|---|---|
| **ingest coverage** | what sources we can read | a new `SourceAdapter` | **no** — maps into the existing shape |
| **schema coverage** | what the canonical shape covers | new `Modality` enum / `Embodiment` / `Taxonomy` row | yes, but additive |

Most adapters map onto existing modalities and never touch the schema. A genuinely new
modality is the rare case where you do both. This separation means we onboard many
datasets without editing the schema, and evolve the schema without rewriting adapters.

---

## 2. The `SourceAdapter` registry pattern

Adapted from the Trossen SDK (`trossen_sdk/design_takeaways.md`): registry-per-axis
(#1) + uniform interface behind pluggable components (#5) + device-lifecycle vs
data-production split (#2) — **with the arrows reversed.** Trossen registers *producers*
that emit into *pluggable backends*; we register *adapters* that read heterogeneous
sources and emit into **one canonical backend (Lance)**.

```
DROID files  ─┐
EgoDex files ─┤   @register_adapter        uniform emit          one writer
EPIC files   ─┼─▶  SourceAdapter(.probe/   ─────────────────▶   CanonicalWriter ─▶ Lance
ego_raw      ─┘    .iter_episodes/.emit)    Episode/Stream/        (catalog + derived
                                            Annotation records)     datasets)
```

### Registry (Trossen #1 — self-registering, open for extension)

```python
_ADAPTER_REGISTRY: dict[str, type["SourceAdapter"]] = {}

def register_adapter(name: str):
    def deco(cls):
        if name in _ADAPTER_REGISTRY:
            raise ValueError(f"duplicate adapter: {name}")
        _ADAPTER_REGISTRY[name] = cls
        return cls
    return deco

@register_adapter("egodex")
class EgoDexAdapter(SourceAdapter): ...
```

New dataset format = a new adapter file with a decorator. No central file changes.

### Uniform interface (Trossen #5 — one contract, many sources)

```python
class SourceAdapter(ABC):
    @abstractmethod
    def probe(self, root: Path) -> bool:
        """Cheap check: does this adapter handle the layout at `root`?"""

    @abstractmethod
    def iter_episodes(self, root: Path) -> Iterator["EpisodeUnit"]:
        """Discover episode units (handles dupes/truncation/missing side-cars)."""

    @abstractmethod
    def emit(self, unit: "EpisodeUnit") -> "CanonicalRecords":
        """Map ONE source episode → canonical Episode/Stream/Annotation rows.
        Reads metadata/headers, references native payloads — does NOT re-encode."""
```

The core ingest loop is source-agnostic:

```python
adapter = select_adapter(root)                 # via .probe()
for unit in adapter.iter_episodes(root):
    try:
        records = adapter.emit(unit)           # native → canonical
        writer.write(records)                  # → Lance (catalog + derived)
    except IngestError as e:
        writer.quarantine(unit, reason=e)      # surface, don't drop/crash
```

### Lifecycle split (Trossen #2 — access vs mapping)

Each adapter separates two concerns, so corruption is isolated and components are
testable:
- **asset access** — locate/open native bytes (mp4 header, hdf5 keys, jpg dir, pkl,
  rlds shard); validate integrity.
- **schema mapping** — translate the accessed metadata into canonical records.

A failure in *access* (truncated mp4) quarantines that component; *mapping* stays pure
and unit-testable against fixtures.

---

## 3. Why this mirrors the schema's self-describing design

The adapter's `emit()` is exactly where Trossen #9 (typed records: tag + `timestamp_ns`
+ `source_id` + `seq`) and #10 (self-describing metadata before first write) land in
our system: it produces `Stream` rows whose `descriptor` fully specifies the payload,
tagged by `modality`, on a canonical `Clock`, with `IngestRun` provenance — so the
canonical store is fully specified without re-parsing native bytes downstream.
`source_id` + per-sample `seq` carried on emitted records power dedup, ordering, and
provenance.

---

## 4. Messy-data handling (per Part 1 §5.4 partial degradation)

Adapters degrade per-component and quarantine at component granularity:

| problem | adapter behavior | result |
|---|---|---|
| truncated / corrupt mp4 | catch at asset-access | `quarantined` + `IngestRun.note` |
| missing hdf5 / intrinsics side-car | emit what survived | video-only episode, `partial` |
| duplicated clip | dedup via checksum (`seq`/`source_id`) | one episode, dup noted |
| partially-extracted `.tar` shard | iter over what extracted | `partial`, rest quarantined |

Never silently drop, never crash the run — every anomaly becomes a queryable
`quality_status` + provenance reason.

---

## 5. Execution model (forward-looking, Part 2/3 scale)

From Trossen's capture-side execution patterns, re-pointed at batch ingest at scale:
- **#4 decouple decode from I/O** — decode/probe workers feed a queue; a writer batches
  Lance commits. Producers (decoders) never block on write I/O.
- **#6 config-driven + CLI overrides + early validation** — adapter selection, paths,
  sampling, and encoder/IDM versions from config; validate at startup.
- **#8 best-effort side channels** — ingest progress / QC metrics fan out to a
  dashboard without backpressuring the durable Lance writes.
- **locality** — compute goes to the data; adapters run where the bytes are mounted.

---

## 6. Plan: which adapters to build

Chosen to stress the schema across the extremes (Part 1 §6 mapping table):
- **`ego_raw`** — the label-less / scale / messy case (truncation, dupes, `.tar`).
- **`egodex`** — richly annotated: hdf5 SE(3) pose streams + moving-camera calibration
  + language captions.
- **`droid`** (stretch) — multi-camera + dual-space actions + per-scene calibration.

Two done well (one rich + the raw case) beats five shallow.
