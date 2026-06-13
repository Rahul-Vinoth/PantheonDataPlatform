# Part 4 — Scale, Storage & Training Hand-off

How the canonical schema ([canonical_schema.md](canonical_schema.md)) and the
encode-once / re-label-often design behave at petabyte scale: why embeddings are the
durable asset, how a PB on-prem store tiers behind a single cache/compute layer, and how
a curated subset is handed off to training as a portable export decoupled from our
internal storage format.

---

## 4.1 The embedding ↔ IDM economics

**Why embeddings are the durable asset.** Two facts about the two models (see the
encoder vs IDM split in Part 3):

| | Encoder (pixels → embeddings) | IDM (embeddings → actions) |
|---|---|---|
| reads | the **petabytes** of video | the **embeddings** (small) |
| cost per run | huge (decode + big forward pass) | tiny (small model over vectors) |
| change cadence | **rarely** (a new encoder is a campaign) | **often** (the thing you keep improving) |

The encoder is the only step that ever pays the pixel cost, so the embedding it produces
is the **expensive, stable, write-once artifact** — and everything downstream is
engineered to consume *it*, not the pixels, ever again.

**The size asymmetry that makes this work.** Embeddings are 1–3 orders of magnitude
smaller than the video they summarize. Rough intuition: a corpus at **~1 PB of video**,
embedded at segment/clip granularity (a ~1–4 KB pooled vector roughly every second),
collapses to **tens of TB of embeddings**. That difference is the whole game: *the
petabytes can't be hot, but the embeddings can.*

**How the schema makes re-labeling cheap.** Recall the one starred FK:

```
ActionLatent.source_embedding_id  FK Embedding   -- keys off EMBEDDINGS, not pixels
```

So shipping a new IDM is:

```
for each row in Embedding (scan the hot ~10s-of-TB table):
    latent = new_idm(row.vector)          # tiny forward pass, no decode
    append ActionLatent(source_embedding_id=row.id, idm_version="idm-v7", latent=...)
```

A **scan-and-write over hot vectors**, append-only, stamped with a new `idm_version`.
Old labels stay (different `idm_version`); nothing is mutated. Contrast the alternative —
a model that went pixels→labels — where a new IDM means **re-reading and re-decoding the
full PB corpus**. The schema converts a multi-PB cold re-decode into a TB-scale hot pass.

**The amortization rule:** many `idm_version`s ride on top of one `encoder_version`. You
pay the expensive PB encode pass once per encoder generation, then re-label the corpus
dozens of times for cheap. `encoder_version` and `idm_version` being *separate
first-class columns* is exactly what lets those two cadences run independently.

---

## 4.2 Storage & tiering: a PB cold store with one cache/compute tier in front

The deployment is a large on-prem cold store with a **single cache/compute tier** in
front of it. The design question is just: *what has to be hot?* The rule:

> **Hot = anything you scan or random-access repeatedly. Cold = bytes you stream
> sequentially and rarely.**

Applied to our tables:

| data | tier | why |
|---|---|---|
| **Catalog** (Episode/Stream/Clock/Annotation/Calibration) — Lance metadata | **HOT** | tiny (GBs), queried constantly during curation; scalar/BTree/FTS indices |
| **Embedding** vectors + ANN index | **HOT** | the training-relevant representation; vector search + IDM scans hit it repeatedly (tens of TB) |
| **ActionLatent**, **QualitySignal** | **HOT** | small, append-often, filtered constantly |
| **Native payloads** (mp4/parquet/hdf5) — what `payload_ref` / `native_root_uri` point at | **COLD** | the petabytes; touched only twice (below) |

The cold petabytes are touched in exactly **two situations**, both sequential and rare:

1. **The encode-once pass** — stream video through the encoder to fill `Embedding`.
2. **The final export materialization** — pull the *selected* clips for a curated
   training subset (§4.3).

Everything else — browsing, filtering, vector search, QC queries, re-labeling — lives
entirely in the hot tier and **never drags a pixel off cold storage.** That is the payoff
of the catalog being pointers-not-payloads: interactive work happens over small metadata
+ embeddings.

**Locality (the constraint we started with).** Compute goes *to* the data: the encoder,
IDM, and QC jobs run co-located with the cold store so a PB is never shipped across the
network. The cache tier holds the catalog + embeddings so curation is interactive without
round-tripping to cold. Lance is the single substrate spanning the hot tiers — columnar
scans, scalar indices for catalog filters, the vector index for embeddings, zero-copy
versioning for snapshots.

---

## 4.3 Training hand-off: a portable, decoupled export

**The principle:** the trainer must **not** couple to our internal Lance lakehouse or
storage layout. We hand off a **frozen, self-contained snapshot** the GPU dataloader
reads directly.

**The flow — curate in the hot tier, materialize once from cold:**

```
1. SELECT (runs over hot catalog + derived):
     episodes where embodiment.kind='robot'
       AND has an action stream
       AND quality_signal: no blur/dup, manipulation_present
       AND an ActionLatent exists for idm_version='idm-v7'
   -> a manifest of (episode_id, stream_ids, time windows)

2. MATERIALIZE (the one cold-store read):
     for each selected clip, pull the native payload slice,
     align/resample onto the training grid (the alignment view, done ONCE here),
     attach its Embedding + ActionLatent + labels

3. WRITE a portable export, decoupled from Lance.
```

**Export format — speak a standard, not our internals.** The natural target is a
conventional training format: **LeRobot v3 / RLDS / WebDataset** (sharded `parquet` for
tabular state/action/latents + `mp4` or pre-decoded frames + an `info.json` catalog). We
already speak LeRobot (we built the adapter), so exporting *to* it closes the loop — and
the trainer needs zero knowledge of Lance. Sequential, shard-based layout is exactly what
GPU dataloaders want.

**What goes in the export:**

- the chosen streams' payloads (sliced to the selected windows), **aligned to the
  training grid** — the multi-rate alignment view from §4.2 is materialized here, once,
  instead of at every training step;
- the **embeddings and action-latents** for those windows (so policies can train on the
  representation directly);
- labels (`label_raw` + normalized `label_ids`);
- a **manifest stamping reproducibility**: the selection predicate, `encoder_version`,
  `idm_version`, taxonomy versions, and the source `ingest_run` ids.

**Why this decoupling matters:**

- **Reproducibility** — a training run is pinned to an immutable export keyed by
  (selection query + encoder/idm versions). Re-label the corpus tomorrow with `idm-v8`
  and yesterday's run is still exactly reproducible.
- **Portability** — the export moves to wherever the GPUs are (on-prem cluster, cloud)
  without exposing the lakehouse.
- **Format independence** — we can evolve Lance internally without breaking a single
  trainer, because the contract is the standard export, not our tables.

---

## How the three tie together

```
COLD (PB):   native video --encode once--┐
                                          v
HOT (10s TB): Embedding  <-- IDM re-labels often --> ActionLatent
                  |  (durable asset, vector-indexed)        |
                  +---------- curate over hot tier ---------+
                                          | materialize selected slices once
                                          v
EXPORT:      portable LeRobot/WebDataset snapshot --> trainer (decoupled, reproducible)
```

The embedding is the pivot: **expensive to make, so it lives hot and durable; cheap to
label off, so the IDM churns over it; and it is the representation the export ships** —
which is why the whole storage hierarchy and hand-off are organized around it rather than
around the pixels.
