# Canonical Dataset Schema — Specification

One schema normalizing every workspace source (teleop, UMI, cross-embodiment,
egocentric-human, academic video, tactile/force, internet video, hand-pose, sim) and
the raw label-less video into a single shape.

**Shape in one line:** a thin `Episode` spine + a polymorphic `Stream` for everything
time-varying + sparse `Annotation`s for labels + time/geometry/embodiment as referenced
entities + a versioned derived tier hubbed on `Embedding`. Absence = zero rows.

Notation: `PK` primary key, `FK` foreign key, `?` nullable/optional, `[T]` list of `T`,
`oneof{…}` tagged union, `T[N]` fixed-length array.

**Storage: standardized on Lance.** Every table below — catalog *and* derived — is a
Lance dataset (one dataset per table). Lance is a columnar Arrow format, so scalar
catalog tables, nested structs/lists, scalar indices (BTree/bitmap/FTS), schema
evolution, transactions, and zero-copy versioning all apply; derived tables additionally
use Lance's vector index. The catalog/derived split below is *logical* (different access
patterns and lifecycles), not two technologies. Implications: `FK`s are logical
contracts and cross-table joins run in the query engine (DuckDB/DataFusion/Polars) over
the Lance datasets; `oneof` unions are materialized as a discriminator column + nullable
branch sub-structs.

---

## 1. Registries (controlled vocabularies)

```
enum Modality       = video | image_seq | pose_se3 | joint_angles | action
                    | force_torque | tactile | depth | audio | keypoints
                    | boxes | masks | scalar
enum RoleClass      = camera | proprio | action | pose | tactile | audio | misc
enum ClockKind      = device_monotonic | wallclock | frame_counter
enum ClockUnit      = ns | s | frame_index
enum Timing         = uniform | explicit
enum CalibScope     = source | episode | stream
enum AnnotationKind = segment | event | caption
enum AnnotationScope= episode | stream | frame_range
enum TaxonomyKind   = closed_vocab | open_vocab
enum QualityStatus  = ok | partial | quarantined
enum ProvKind       = dataset | human | model

registry Embodiment {           -- referenced; `unknown` sentinel, never null
  embodiment_id  PK
  name                          -- unknown | human_hands | franka_panda | …
  kind                          -- human | robot | unknown
  dof            ?
  kinematic_model_ref ?         -- URDF / hand model for derived FK projection
}

registry Taxonomy {
  taxonomy_id    PK
  name                          -- "epic-kitchens-100-verbs"
  kind           TaxonomyKind
  version
  classes        [{id, name, parent?}]
}
```

---

## 2. Catalog tables (normalized at rest; Lance datasets)

```
table Source {                  -- one collection (DROID, EgoDex, ego_raw/factory…)
  source_id          PK
  name
  data_type                     -- catalog class: teleop | egocentric | sim | …
  license
  native_format                 -- rlds | hdf5+mp4 | jpeg_dir+pkl | mp4+csv | …
  default_embodiment FK Embodiment
  modality_profile   [Modality] -- advisory: what this source typically carries
}

table Episode {                 -- thin spine; the unit of curation & handoff
  episode_id         PK
  source_id          FK Source
  embodiment_id      FK Embodiment   -- `unknown` for raw video (sentinel, not null)
  primary_clock_id   FK Clock
  t_start_ns         int64           -- episode wall-clock span
  duration_ns        int64
  quality_status     QualityStatus
  ingest_run_id      FK IngestRun
  native_root_uri                    -- untouched source bytes
}
-- NOTE: no action_*, pose_*, intrinsics_*, language_* columns by design.

table IngestRun {               -- provenance of ingestion itself
  ingest_run_id      PK
  code_version
  started_at
  source_path
  checksum
  note               ?               -- quarantine/partial reason
}
```

### Time

```
table Clock {
  clock_id           PK
  episode_id         FK Episode
  kind               ClockKind
  unit               ClockUnit
  is_primary         bool
}

table ClockSync {               -- map a non-primary clock onto a reference clock
  clock_id           FK Clock
  ref_clock_id       FK Clock
  model              enum{offset, linear, anchors}
  offset_ns          int64?
  skew_ppm           float?            -- linear drift
  anchors            [(t_clock_ns, t_ref_ns)]?
}
```

### Signal — the polymorphic core

```
table Stream {                  -- ONE time-series channel of ONE modality on ONE clock
  stream_id          PK
  episode_id         FK Episode
  modality           Modality
  role               { slug, role_class RoleClass, attrs{} }  -- e.g. exterior_1, action.joint
  group_id           ?               -- rig / stereo / action co-representation grouping
  clock_id           FK Clock
  timing             Timing
  rate_hz            float
  t_start_ns         int64
  t_end_ns           int64
  n_samples          int64
  timestamps_ref     ?               -- native per-sample timestamps (timing=explicit)
  payload_ref        { uri, locator, format }   -- native blob + how to index in
  descriptor         Descriptor      -- self-describing contract (below)
}

struct Descriptor {             -- normalized metadata that enables lazy canonicalization
  shape          [int]
  dtype
  units          ?              -- e.g. m, rad, N
  frame_id       ?              -- reference frame name (robot_base, arkit_origin…)
  rotation_rep   ?              -- matrix | quat | euler | 6d   (pose/action)
  channel_layout ?              -- incl. companion channels (e.g. per-joint confidence)
  dof_labels     [string]?      -- per-channel names
  -- action-only fields:
  space          ?             -- joint | cartesian | ee_pose
  control_mode   ?             -- position | velocity | torque
  is_delta       ?bool
  gripper        ?{ encoding: continuous|binary, units }
}
```

### Geometry

```
table CalibrationProfile {      -- absent when uncalibrated (no nulls)
  calib_id           PK
  scope              CalibScope          -- inherits stream → episode → source
  target_ref                             -- source|episode|stream id per scope
  intrinsics         { K float[3][3], distortion_model, dist_coeffs[], width, height,
                       rolling_shutter? }
  extrinsics         oneof {
                       static  { T_cam_to_frame float[4][4], frame_id },
                       dynamic { extrinsics_stream_id FK Stream(pose_se3), frame_id }
                     }                    -- dynamic = MOVING CAMERA
  rig                ?{ group_id, T_cam_to_rig float[4][4] }
}
```

### Sparse labels

```
table Annotation {
  annotation_id      PK
  episode_id         FK Episode
  kind               AnnotationKind
  scope              AnnotationScope
  target_ref         ?  FK Stream         -- when scope=stream
  clock_id           FK Clock
  t_start_ns         int64?               -- segment / frame_range
  t_stop_ns          int64?
  t_at_ns            int64?               -- event
  payload            oneof {
                       verb_noun { verb_id?, noun_id?, verb_raw?, noun_raw? },
                       text      { text },
                       event     { event_type, attrs{} }
                     }
  taxonomy_id        ?  FK Taxonomy
  label_raw          ?                    -- original string, always kept
  label_ids          [int]?               -- normalized class ids (null if withheld)
  confidence         float?
  provenance         { source ProvKind, model_version? }
}
```

---

## 3. Derived tables (Lance datasets; vector-indexed, versioned, append-only)

```
table Embedding {               -- encode ONCE; expensive; durable; hub of derived tier
  embedding_id       PK
  episode_id         FK Episode
  stream_id          FK Stream          -- usually video/image_seq
  t_start_ns         int64              -- window on canonical clock (alignable)
  t_end_ns           int64
  encoder_version                       -- defines windowing; first-class version
  vector             float[D]           -- Lance ANN-indexed column
  meta               { dim, pooling, frame_stride }
}

table ActionLatent {            -- re-label OFTEN; cheap; append-only
  latent_id            PK
  episode_id           FK Episode
  stream_id            FK Stream
  source_embedding_id  FK Embedding      -- ★ keys off EMBEDDINGS, not pixels
  encoder_version
  idm_version                            -- first-class version
  t_start_ns           int64
  t_end_ns             int64
  latent               float[K]
  action_space         ?                 -- if decoded to a concrete space
  confidence           float
}

table QualitySignal {           -- curation/QC outputs (Part 3)
  signal_id          PK
  episode_id         FK Episode
  stream_id          ? FK Stream
  t_start_ns         int64?             -- episode | stream | frame_range scope
  t_end_ns           int64?
  signal_type                           -- blur | exposure | dropped_frame | dup_frame
                                        -- | near_dup_cluster | manipulation_present
                                        -- | task_tag | env_tag
  value              json               -- score / flag / cluster_id / tag
  detector_version
}
```

---

## 4. Entity relationships

```
Source 1──< Episode 1──< Stream >──1 Clock ──< ClockSync
                │            │  │
                │            │  └─1 CalibrationProfile (cameras; optional)
                │            ├─< Embedding ──< ActionLatent
                │            └─< QualitySignal
                ├──< Annotation >──1 Taxonomy
                ├──1 Embodiment (ref; `unknown` sentinel)
                └──1 IngestRun
```

---

## 5. Load-bearing rationale

1. **Thin spine + polymorphic `Stream`.** Everything time-varying is a `Stream` row,
   not an episode column. Variable camera counts are rows; new modalities are new enum
   values, not migrations. Avoids the RLDS/OXE fixed step-dict.

2. **Absence = zero rows, never null columns.** The episode has no modality-specific
   fields, so the label-less raw-video case is structurally complete with one `video`
   stream and zero optional rows — no null pile. Even `embodiment` uses an `unknown`
   sentinel, not null.

3. **Three tiers: normalize structure, keep signal native, canonicalize as views.**
   At rest: ids, `Descriptor` metadata, canonical time index. Native: pixels, pose
   arrays, joint *and* cartesian actions, high-rate force — never re-encoded. Derived:
   units/frame/action-space/rate conversions, computed on demand. The `Descriptor` is
   the contract that makes lazy canonicalization correct. Eager payload normalization
   (resample-to-30Hz, collapse-to-7DoF, transcode video) is irreversible loss.

4. **Multi-rate is explicit; alignment is a view.** Each `Stream` names its own `Clock`;
   `ClockSync` relates clocks; `timing=explicit` preserves native timestamps so dropped/
   duplicate frames stay visible. Resampling onto a common clock is a training-time
   choice, never baked in.

5. **A moving camera's extrinsics *are* a pose stream.** `CalibrationProfile.extrinsics`
   is `static SE(3)` or `dynamic → pose_se3 Stream`, unifying fixed and moving cameras
   and reusing the clock/alignment machinery. Missing calibration = absent row.

6. **Labels split by cadence × provenance.** Dense → `Stream`; sparse → `Annotation`.
   Shipped-with-data → native tier; inferred-by-us → versioned derived tier (same shape,
   different lifecycle). Raw and normalized labels both kept (`label_raw` + `label_ids`).

7. **Multiple action spaces kept side by side.** `action.joint` and `action.cartesian`
   are separate grouped streams; the OXE 7-DoF normalization is a derived view using the
   `Embodiment` kinematic model — never a lossy collapse at rest.

8. **Derived tier hubbed on `Embedding`; latents key off embeddings, not pixels.**
   `ActionLatent.source_embedding_id` is the decision that makes re-labeling the whole
   corpus with a new IDM a cheap scan-and-write over small hot vectors instead of a
   multi-PB re-decode. Versions (`encoder_version`, `idm_version`) are first-class
   columns; everything is append-only and regenerable.

---

## 6. Source → schema mapping

| source | embodiment | streams | calibration | annotations | derived |
|---|---|---|---|---|---|
| **ego_raw** (raw ego) | unknown | 1× video | — | — | Embedding → ActionLatent (IDM) |
| **EgoDex** (annot. ego) | human_hands | video + N× pose_se3 (head/wrist/fingers, +confidence channels) | intrinsics @source; **dynamic** extrinsics → camera pose stream; frame=arkit_origin | caption (llm_description) | Embedding, QC |
| **EPIC** (academic) | unknown/human | 1× video | — | segment (verb_noun + raw narration) | Embedding, QC |
| **DROID** (teleop X-emb) | robot | N× video (stereo groups) + proprio.{joint,cartesian} + action.{joint,cartesian} | episode-scope static extrinsics | caption (instruction) | Embedding |
| **DexCap** (dex multicam) | human_hands | N× image_seq + joint_angles (glove) + pose_se3 (wrist SLAM) | **dynamic** extrinsics → wrist pose; rig groups; per-modality clocks | — | Embedding |

---

## 7. Minimal (label-less) instance

```
Source     ego_raw/factory  license=… default_embodiment=unknown
Episode    embodiment=unknown primary_clock=C quality=ok native_root_uri=…/clip.mp4
Clock C    uniform 30 Hz
Stream     video role=head_cam clock=C timing=uniform payload_ref=…/clip.mp4
IngestRun  checksum … code_version …
           # 0 annotations · 0 calibration · 0 other streams · 0 embeddings · 0 latents
```

Valid and complete. It becomes training-ready by *accreting derived rows*
(`+Embedding → +ActionLatent → +QualitySignal`), never by adding native labels.
