# Part 1 — Canonical Dataset Schema

The goal: **one shape** that normalizes every source in the workspace — the richly
annotated robot/ego sets *and* the raw label-less video — without a pile of nulls
and without a schema migration every time a new modality appears.

Source data types we must hold (from the catalog): **teleop, UMI, cross-embodiment,
egocentric-human, academic video, tactile/force, internet video, hand-pose, sim.**

This document is built incrementally, one design question at a time.

- [x] **1. Entity model** — what an episode is, what hangs off it, entity vs inline
- [x] **2. Normalized-at-rest vs kept-native**
- [x] **3. Streams, multi-rate clocks, calibration**
- [x] **4. Annotations, multi-space actions**
- [x] **5. Label-less graceful degradation**
- [x] **6. Embeddings & versioned IDM latents**

---

## 1. Entity model

### 1.1 The governing principle

The recurring failure mode in prior art (RLDS/OXE) is the **fixed step-dict**: every
episode is forced into one observation/action shape, so a new modality means a
migration, and absent modalities become nulls. We avoid this with one rule:

> **The episode is a thin spine. Everything whose cardinality, presence, rate, or
> lifecycle varies across sources is a *separate entity referenced by `episode_id`*,
> not a field on the episode.**

Concretely, something is a **separate entity** when *any* of these hold:

- its count per episode varies independently (`0..N` cameras, `0..N` annotations),
- it is **absent** for some sources (calibration, actions, pose) — making it an
  entity means absence = *zero rows*, never a null column,
- it is **shared / referenced** (a dataset's license, an embodiment definition, a
  label taxonomy, a clock),
- it is a **derived artifact with its own version and lifecycle** (embeddings, IDM
  latents, QC signals),
- it is **large / native** payload bytes we refuse to re-encode on ingest.

Something is an **inline field** only when it is **exactly one per episode, small,
and always meaningful** (duration, source ref, quality status, ingest timestamp).

This is what makes the label-less case degrade gracefully: the raw-video episode is
*structurally complete* with one video stream and **zero** rows in every optional
entity. There is nothing to null out, because the modality-specific data never lived
on the episode in the first place.

### 1.2 Entity catalog

```
Source 1───< Episode 1───< Stream >───1 Clock
  │              │            │  │
  │              │            │  └──1 CalibrationProfile      (cameras only, optional)
  │              │            │
  │              │            └──< Embedding        (derived, versioned by encoder)
  │              │            └──< ActionLatent     (derived, versioned by IDM)
  │              │            └──< QualitySignal    (derived, curation/QC)
  │              │
  │              ├──< Annotation >───1 Taxonomy     (segments / events, optional)
  │              ├──1 Embodiment (ref)              (nullable → human/unknown)
  │              └──1 IngestRun  (ref, provenance)
  │
  └──1 License / provenance defaults
```

#### Spine & context

**`Source`** — the dataset/collection an episode came from (DROID, EgoDex,
EPIC-KITCHENS, the `factory_*` ego-raw set). Holds what is true for *every* episode
in it: `source_id`, name, data-type class, **license**, native format family,
default embodiment, default calibration, modality profile. *Entity, not inline:*
license and provenance are per-source — repeating them on millions of episodes is
both wasteful and a consistency hazard.

**`Episode`** — the central spine and the unit of curation/handoff. Deliberately
**thin**; carries only what is one-per-episode and always meaningful:

| field | notes |
|---|---|
| `episode_id` | global stable id |
| `source_id` → Source | provenance / license inheritance |
| `embodiment_id` → Embodiment | **nullable** (raw human video → null/“unknown”) |
| `t_start`, `duration_s` | episode-level wall-clock span; per-stream timing lives on streams |
| `primary_clock_id` → Clock | the reference clock for alignment (§3) |
| `quality_status` | enum: `ok / quarantined / partial` (Part 2 ingest) |
| `ingest_run_id` → IngestRun | when/how it was ingested |
| `native_root_uri` | where the untouched source bytes live |

Note what is **absent**: no camera fields, no action fields, no pose fields, no
calibration. Those are entities below.

#### The polymorphic core

**`Stream`** — *the* key generalization, replacing the fixed observation/action dict.
A stream is **one time-series channel of one modality, on one clock, at one rate.**
Everything time-varying is a stream.

| field | notes |
|---|---|
| `stream_id`, `episode_id` | |
| `modality` | enum: `video / image_seq / pose_se3 / joint_angles / action / force_torque / tactile / depth / audio / scalar` |
| `role` | semantic slot: `head_cam`, `exterior_1`, `left_wrist`, `right_hand`, `action.joint`, `action.cartesian`, … |
| `clock_id` → Clock | which time base its timestamps are in (§3) |
| `rate_hz` | nominal rate (streams are **native-rate**, not resampled) |
| `payload_ref` | URI + locator into the native asset (mp4 file, hdf5 dataset path, jpg dir, pkl key) |
| `descriptor` | shape, dtype, units, frame convention, DoF labels — the self-describing schema of the payload |

This single entity absorbs the whole heterogeneity matrix:

- **variable # of cameras** → N rows with `modality=video`, different `role`.
- **multi-rate clocks** → each stream names its own `clock_id` + `rate_hz`.
- **actions in two spaces simultaneously** → two streams, `action.joint` *and*
  `action.cartesian`, both kept (no OXE-style lossy collapse).
- **glove vs SE(3) vs none** → `joint_angles` stream, or `pose_se3` stream, or simply
  no such stream.
- **label-less** → exactly one `video` stream and nothing else.

**Load-bearing distinction — stream vs annotation:** a *dense, regularly-sampled
value-per-frame* time-series is a **stream** (EgoDex per-frame SE(3) hand transforms =
a `pose_se3` stream; per-frame boxes/masks = a stream). A *sparse labeled interval or
instant* is an **annotation** (EPIC verb+noun segment; a “contact” event). The test:
*is there a value at (almost) every tick, or only at a few marked times?*

**`Clock`** — a monotonic time base. Streams reference a `clock_id`; multiple streams
can share one (all DROID proprio on the robot clock). Cross-clock alignment is
described by an offset/skew or a shared wall-clock anchor. *Entity because* the
multi-rate problem is real (DexCap’s per-modality `timesteps.txt`) and alignment must
be explicit and well-defined, not implied by a shared frame index.

**`CalibrationProfile`** — camera intrinsics/extrinsics, attached to a video/image
stream. Must cover both the static and **moving-camera** cases: intrinsics (3×3 +
distortion); extrinsics that are *either* a static SE(3) *or* a **reference to a
`pose_se3` stream** that supplies per-frame extrinsics (EgoDex SLAM, DexCap). *Entity,
not inline:* often absent (quarantine-tolerant), shared across streams, and may be
time-varying — it cannot be a scalar column.

#### Labels & meaning

**`Annotation`** — sparse, polymorphic labels referencing an episode + a time
reference on a clock. Subtypes via a `kind` discriminator:
- `segment` (interval): `t_start`, `t_stop`, label payload (verb+noun, free text).
- `event` (instant): single timestamp + payload.
Each links to a **`Taxonomy`** (e.g. EPIC’s 97 verbs / 300 nouns) so open-vocab vs
fixed-vocab labels are both representable and queryable. *Entity:* `0..N`, optional,
varies wildly by source.

**`Embodiment`** — a registry defining what produced the actions: robot type, DoF,
kinematic class, or `human_hands`, or `unknown`. Referenced (not inlined) because it
gives *meaning* to joint/action streams (a 7-DoF Franka vs a Manus glove) and is
shared across many episodes. **Nullable on the episode** is the crux of graceful
degradation — raw video simply has no embodiment.

#### Derived layer (encode-once / re-label-often)

These are **separate entities with their own versioned lifecycle**, keyed back to a
stream (and frame-range). They live in the Lance vector tables (per earlier decision).

**`Embedding`** — durable, expensive, computed **once**: vectors for a `(stream,
frame_range)` tagged by `encoder_version`. The corpus’s permanent asset.

**`ActionLatent`** — IDM output, **cheap and re-run often**: action latents for a
`(stream/episode, frame_range)` tagged by `idm_version`, **computed over Embeddings,
never over pixels.** Append-only: a new IDM = new rows, not a re-decode. This is the
schema-level expression of the encode-once/re-label-often economics.

**`QualitySignal`** — curation/QC outputs (Part 3): blur/exposure scores, dropped/
duplicate-frame flags, near-duplicate cluster id, manipulation-present flag,
task/environment diversity tags. Attached at episode / stream / frame-range. *Entity:*
a queryable derived layer that drives corpus balancing and filtering.

**`IngestRun`** — provenance of the ingestion itself: run id, code version,
source path, checksum, transformations applied, quarantine reason. *Entity* so every
episode is traceable and re-ingestion is auditable.

### 1.3 Why this holds all nine data types

| data type (workspace dir) | episode = | streams | annotations | embodiment |
|---|---|---|---|---|
| egocentric-human raw (`ego_raw`) | one clip | 1 × `video` | — | null |
| egocentric-human annotated (`ego_annot`/EgoDex) | one clip | `video` + many `pose_se3` (head/wrist/fingers) | — (language as event/attr) | `human_hands` |
| academic video (`segmented_video`/EPIC) | one clip | 1 × `video` | `segment` (verb+noun) | null/human |
| teleop / cross-embodiment (`robot_rlds`/DROID) | one episode | N × `video` + proprio + `action.joint` + `action.cartesian` | language `event` | robot |
| dexterous multicam (DexCap) | one ep | N × `image_seq` + `joint_angles` (glove) + `pose_se3` (wrist) | — | `human_hands`+glove |
| tactile/force | one ep | `force_torque` / `tactile` streams alongside video | — | varies |
| hand-pose | one ep | `pose_se3` / keypoint streams | — | `human_hands` |
| sim | one ep | rendered `video` + full GT `action`/`pose` streams | optional | robot |

Every column above is the *same* episode→stream(→derived) shape. New modalities
(e.g. event-camera, EMG) are **new `modality` enum values + a `descriptor`**, not a
new table. That is the test the eval cares about: *one shape, no per-modality
migration, no null pile.*

### 1.4 Open questions deferred to later sections

- physical layout of derived tables in Lance + versioning keys (§6)

---

## 2. Normalized at rest vs kept native

### 2.1 The core stance: three tiers, not two

The instinct to "normalize everything on ingest" is the expensive mistake. We split
the data into **three tiers** by *who pays the cost and when*:

1. **Normalized at rest** — the **catalog**: identities, relationships, and
   *metadata about* the payloads. Small, cheap, lossless, queryable without ever
   decoding a byte. Written once on ingest.
2. **Kept native** — the **payloads**: the raw signal bytes, in their original
   container, untouched. We reference *into* them; we never re-encode them.
3. **Canonicalized later** — **derived views**: opinionated transforms (resample to a
   common clock, project to one action space / frame / unit system). Either computed
   *lazily at read* or *materialized as versioned, regenerable artifacts* — but never
   overwriting the native tier.

The dividing question for any field is: **"Is this transform faithful, cheap, and
stable — or is it opinionated, model-dependent, or downstream-specific?"** The first
goes to rest; the second stays native and is canonicalized in a view.

### 2.2 What gets normalized at rest (the catalog)

Everything you need to **find, align, govern, and interpret** a payload *without
touching it*:

| normalized at rest | why it is safe to normalize |
|---|---|
| entity graph & ids (episode/stream/annotation/…) | structure, not signal — lossless |
| **stream `descriptor`**: modality, role, shape, dtype, **native units**, **frame convention**, **rotation rep**, DoF labels, channel layout | this is *metadata about* the payload; recording it costs nothing and is what makes lazy canonicalization possible later |
| **canonical time index**: every event/sample tagged with a normalized timestamp (int64 ns) *alongside* the preserved native timestamp; per-clock offset/skew anchors | timestamps are tiny and are the precondition for alignment (§3). We add a canonical index, we do **not** discard native time |
| provenance, license, `quality_status`, checksums, ingest run | governance must be queryable corpus-wide |
| taxonomy **pointer** + the **original raw label string** | keep both: the normalized class id for queries, the raw string so nothing is lost |

Key move: **the `descriptor` is the contract.** Because every stream self-describes
its native units, frame, and rotation representation at rest, a derived view can
canonicalize correctly *on demand* without anyone re-decoding or re-guessing. We
normalize the *description*, not the *data*.

### 2.3 What stays native (payloads — referenced, never rewritten)

| kept native | what would be lost by normalizing at rest |
|---|---|
| video pixels (MP4 / H.265), JPEG-on-disk | re-encoding is lossy *and* a multi-PB compute that breaks data locality; embeddings must come from original pixels |
| HDF5 SE(3) transform arrays + confidences | the raw numeric signal; reference by `(file, dataset_path, frame_range)` |
| robot **joint *and* cartesian** action arrays | both are first-class — collapsing to one space is irreversible (the OXE lesson) |
| glove joint-angles (`manus.pkl`) | converting to fingertip poses needs a hand model (see §2.5) |
| force/torque & tactile raw samples (e.g. 1 kHz) | resampling at rest destroys the high-rate signal forever |

We store these as immutable blobs and point at them with `payload_ref`; ingestion
reads metadata and headers, not full payloads.

### 2.4 What is canonicalized later (derived)

These are all **opinionated or downstream-specific** transforms — exactly what must
*not* be baked in:

- **common-clock alignment** — resampling N streams onto one rate presupposes a target
  rate and an interp/nearest policy; it is a *training-time* choice (§3).
- **canonical action space** — an OXE-style normalized end-effector view is useful for
  cross-embodiment training but lossy; we expose it as a *view* over the native
  joint/cartesian streams, not as storage.
- **canonical units / frame** — SI units and a unified world/camera frame are derived
  from the descriptor’s native units + calibration; if calibration improves, the view
  is recomputed, native data is untouched.
- **embeddings & IDM action-latents** — the durable-but-derived artifacts; materialized
  and versioned (§6), never replacing pixels.

Derived splits into **(a) cheap lazy views** computed at read (unit/frame/space
conversions) and **(b) materialized artifacts** that are expensive to recompute and so
are stored — but always *versioned and regenerable* from native + descriptor, so they
are caches, not sources of truth.

### 2.5 Why eager payload normalization is lossy — concrete cases

Each of these is a transform that *looks* like a tidy normalization and is actually a
one-way loss:

1. **Resample-to-30 Hz at rest** → the 1 kHz force/tactile signal is gone; any model
   that wanted contact dynamics is now starved. *Native rate is irrecoverable.*
2. **Collapse joint+cartesian → 7-DoF ee** (OXE) → joint-space control, redundancy
   resolution, and multi-arm structure are discarded permanently.
3. **Glove angles → fingertip SE(3)** → requires a hand-kinematics/retargeting model;
   bake it in and you can never re-retarget when that model improves. Keep angles
   native, retarget in a versioned view.
4. **Overwrite native pose frame → one world frame** → depends on calibration that may
   be missing or wrong; if you discard the native ARKit/SLAM frame you cannot
   re-derive when calibration is fixed.
5. **Transcode video to a “standard” codec/res** → lossy pixel destruction, breaks the
   encode-once embedding contract, and is economically impossible at petabyte scale.

The unifying principle mirrors the encode-once/re-label-often economics: **keep the
faithful raw thing once; make every opinionated interpretation a cheap, versioned,
regenerable derivation on top of it.** Normalize the *map* (descriptor + ids +
timestamps), never the *territory* (the signal).

### 2.6 Summary rule

> **Normalize structure, metadata, and time at rest. Keep every signal payload native.
> Canonicalize units, frames, action spaces, and rates as versioned derived views.**
> If a transform is faithful + cheap + stable → rest. If it is the raw signal → native.
> If it is opinionated, model-dependent, or downstream-specific → derived.

---

## 3. Streams, multi-rate clocks, calibration

This section turns the `Stream`, `Clock`, and `CalibrationProfile` entities from §1
into concrete field-level schemas. Three sub-problems: a *variable number* of streams,
*multi-rate* streams with *their own clocks*, and *calibration* including the
*moving-camera* case.

### 3.1 A variable number of camera/sensor streams

The count problem is solved by §1’s decision that **streams are rows, not columns.**
Ten cameras = ten `Stream` rows; one camera = one row; zero = zero rows. No schema
ever encodes "camera_1 … camera_N" as fixed fields. Querying "the cameras of episode
E" is `SELECT * FROM stream WHERE episode_id=E AND modality='video'`.

```
Stream {
  stream_id          : id
  episode_id         : id
  modality           : enum {video, image_seq, pose_se3, joint_angles,
                              action, force_torque, tactile, depth, audio, scalar}
  role               : { slug, role_class, attrs{} }   # see below
  group_id           : id?            # rig / stereo grouping (§3.1)
  clock_id           : id  -> Clock   # (§3.2)
  timing             : enum {uniform, explicit}
  rate_hz            : float          # nominal; exact for uniform timing
  t_start_ns         : int64          # on the canonical time index
  t_end_ns           : int64
  n_samples          : int64
  timestamps_ref     : ref?           # native per-sample timestamps (explicit timing)
  payload_ref        : { uri, locator, format }   # native blob + how to index into it
  descriptor         : { shape, dtype, units, frame_id, rotation_rep?,
                         channel_layout, dof_labels }   # the §2 self-describing contract
}
```

**`role` is a structured, controlled slug — not a free string.** It carries a
`role_class` (`camera / proprio / action / pose / tactile / …`) plus a `slug`
(`head_cam`, `exterior_1`, `left_wrist`, `action.joint`, `action.cartesian`) and free
`attrs`. Two reasons it is structured: (a) roles must be **stable within a source** so
a dataloader can batch "the wrist cam" across thousands of episodes; (b) `role_class`
lets corpus-wide queries find "all exterior cameras" or "all action streams" without
hard-coding source-specific names. Meaning across *different* sources comes from the
`Embodiment` + `Source` context, not from forcing one global naming scheme.

**`group_id`** captures rig structure that a flat list loses: a **stereo pair** (DROID
Zed left/right) or all cameras on one **rig**. Grouped streams can share a rig frame
and relative extrinsics (§3.3) even when the rig’s world pose is unknown.

This is also where graceful degradation lives mechanically: the label-less episode is
exactly one row with `modality=video`, `role.slug=head_cam`, and no other streams.

### 3.2 Multi-rate streams with their own clocks

Cameras at 30 Hz, proprio at 100–500 Hz, force at 1 kHz, language once — **no shared
frame index means the same instant across streams.** We refuse to resample at rest
(§2). Instead the schema stores *enough to align correctly later*: per-stream native
timestamps + explicit clock relationships.

```
Clock {
  clock_id     : id
  episode_id   : id
  kind         : enum {device_monotonic, wallclock, frame_counter}
  unit         : enum {ns, s, frame_index}
  is_primary   : bool          # episode.primary_clock_id points at one of these
}

ClockSync {                     # how a non-primary clock maps onto the reference
  clock_id     : id -> Clock
  ref_clock_id : id -> Clock
  model        : enum {offset, linear, anchors}
  offset_ns    : int64?         # model=offset|linear
  skew_ppm     : float?         # model=linear  (clock drift)
  anchors      : [(t_clock_ns, t_ref_ns)]?   # model=anchors (piecewise)
}
```

Design points:

- **Each stream names its own `clock_id`.** All DROID proprio/action share the robot
  clock; the camera may be on another. DexCap’s `timesteps.txt` gives a *separate*
  timestamp series per modality → each modality is a stream on its own clock,
  reconciled by `ClockSync`.
- **Timing is `uniform` or `explicit`.** A true-30 fps video is `uniform`: timestamps
  are implicit (`t_start + i/rate`), so we store none. An event-driven, dropped-frame,
  or jittery stream is `explicit`: native per-sample timestamps are referenced
  (`timestamps_ref`) and we materialize the canonical int64-ns index alongside them.
- **Dropped/duplicate frames are visible, not silently absorbed.** Because explicit
  streams keep real timestamps, a gap or a duplicate timestamp is detectable — it
  becomes a §-Part-3 `QualitySignal`, not a hidden misalignment. (Assuming uniform
  timing where it isn’t true is precisely how corpora corrupt themselves.)
- **Alignment is a derived view, never stored.** Given `(target_clock, target_rate,
  policy ∈ {nearest, linear, hold})`, a view produces aligned multi-stream frames by
  mapping each stream’s timestamps through `ClockSync` onto the target. The policy is a
  training-time choice; baking one in at rest would destroy the high-rate signal (§2.5).

### 3.3 Calibration — intrinsics, extrinsics, and the moving camera

Calibration is a **separate entity** (often absent; shared; time-varying — cannot be a
scalar column) with **scope-based inheritance**: it can attach at `Source` (a constant
rig, e.g. EgoDex’s fixed 3×3 intrinsics), at `Episode` (per-scene, e.g. DROID), or at a
single `Stream` (override). A stream resolves its calibration by walking
stream → episode → source.

```
CalibrationProfile {
  calib_id     : id
  scope        : enum {source, episode, stream}
  target_ref   : id      # source_id | episode_id | stream_id per scope

  intrinsics   : { K : 3x3, distortion_model, dist_coeffs[],
                   width, height, rolling_shutter? }

  extrinsics   : oneof {
     static  { T_cam_to_frame : 4x4, frame_id },               # fixed pose
     dynamic { extrinsics_stream_id -> Stream(pose_se3), frame_id }  # MOVING CAMERA
  }

  rig          : { group_id, T_cam_to_rig : 4x4 }?   # relative pose within a rig
}
```

The **load-bearing idea: a moving camera’s extrinsics *are* a pose stream.** Rather
than inventing a separate per-frame-extrinsics structure, the `dynamic` case points
`extrinsics_stream_id` at a `pose_se3` stream that already supplies the camera’s SE(3)
pose at every frame (EgoDex’s SLAM camera transform; DexCap’s hand-mounted SLAM). This
unifies the static and moving cases under one schema — static is a single SE(3),
moving is a reference to the same kind of pose data we already store — and it reuses
the multi-rate machinery (the extrinsics stream has its own clock and can be aligned to
the image stream).

Supporting decisions:

- **`frame_id` is explicit.** Every extrinsic is *relative to a named reference frame*
  (`robot_base`, `arkit_origin`, `rig_0`, `world`). Frames are per-episode/source and
  may themselves relate via a pose, so cross-stream geometry is always well-defined.
  EgoDex’s per-session ARKit origin is just a per-episode `frame_id`.
- **Rigs.** `rig.T_cam_to_rig` gives each camera’s pose *within its `group_id` rig*, so
  a stereo pair or multi-cam rig has known relative geometry even when the rig’s world
  pose is unknown or moving.
- **Intrinsics vs extrinsics split** matches reality: intrinsics are usually constant
  (store once at source scope, reference everywhere) while extrinsics vary per scene or
  per frame.
- **Missing calibration → the entity is simply absent.** No nulls; geometry-dependent
  derived views just can’t be computed for that stream, and ingest flags it
  `uncalibrated` as a `QualitySignal`. This is the graceful-degradation principle
  applied to calibration.

### 3.4 Worked micro-examples

| source | streams | clocks | calibration |
|---|---|---|---|
| **DROID** | 3× `video` (2 exterior + 1 wrist, stereo `group_id`s) + `joint_angles` proprio + `action.joint` + `action.cartesian` | robot clock (primary); camera clock + `ClockSync` if separate | `episode` scope; **static** extrinsics per scene; wrist cam may be `dynamic` ref to ee-pose |
| **EgoDex** | 1× `video` + many `pose_se3` (head/wrists/25 finger joints) | single 30 Hz clock, `uniform` timing | intrinsics at `source` scope (constant 3×3); **dynamic** extrinsics → the camera `pose_se3` stream; `frame_id = arkit_origin` (per-episode) |
| **DexCap** | N× `image_seq` (palm+stereo) + `joint_angles` (glove) + `pose_se3` (wrist SLAM) | **per-modality clocks** from `timesteps.txt`, reconciled by `ClockSync` | hand-mounted cams → **dynamic** extrinsics via wrist `pose_se3`; rig grouping for stereo |
| **ego_raw** | 1× `video` | one `uniform` clock | none (absent) — flagged `uncalibrated` |

Every row uses the *same* `Stream` / `Clock` / `CalibrationProfile` shapes; the only
thing that varies is which rows exist and which `oneof` branch the extrinsics take.

---

## 4. Annotations and multi-space actions

The spec lumps several things together — temporal segments, per-frame pose/boxes/masks,
point events, and robot actions in more than one space. They do **not** all belong in
one table. §4 sorts them with two orthogonal axes and then specifies the action model.

### 4.1 The two axes that decide where a label lives

| axis | question | values |
|---|---|---|
| **cadence** | is there a value at (almost) every tick, or only at a few marked times? | **dense** → a `Stream`; **sparse** → an `Annotation` |
| **provenance** | did it ship *with* the source, or did *we* infer it? | **shipped** → native tier; **inferred** → derived/versioned tier (§6) |

Cadence decides the *shape* (stream vs annotation); provenance decides the *tier*
(native vs versioned derived). This cleanly places every label type:

| label | cadence | → shape | example |
|---|---|---|---|
| temporal action segment | sparse interval | **Annotation** `segment` | EPIC verb+noun |
| point event | sparse instant | **Annotation** `event` | "contact onset" |
| task / instruction text | sparse (episode or interval) | **Annotation** `caption` | EgoDex `llm_description`, DROID instruction |
| per-frame SE(3) pose | dense | **Stream** `pose_se3` | EgoDex hand/finger transforms |
| per-frame boxes / masks / keypoints | dense | **Stream** `boxes`/`masks`/`keypoints` | hand-pose datasets |
| IDM action latents | dense | **derived** stream-shaped (§6) | inferred, versioned |

The important consequence: **a dense label shipped with the data is a native stream;
the *same shape* inferred by us is a versioned derived artifact.** Per-frame masks that
come with a segmentation dataset live in the native tier; masks we predict during
curation live in the derived tier and carry a `model_version`. Same physical shape,
different lifecycle — which is exactly the encode-once/re-label-often split.

### 4.2 The `Annotation` entity (sparse labels)

```
Annotation {
  annotation_id : id
  episode_id    : id
  kind          : enum {segment, event, caption}
  scope         : enum {episode, stream, frame_range}
  target_ref    : id?          # stream_id when scope=stream
  clock_id      : id -> Clock  # time reference (segments/events live on a clock)
  t_start_ns    : int64?       # segment / frame_range
  t_stop_ns     : int64?
  t_at_ns       : int64?       # event
  payload       : { type, ... }      # typed union, see below
  taxonomy_id   : id -> Taxonomy?
  label_raw     : string?      # ORIGINAL string, always kept (narration, llm_description)
  label_ids     : [int]?       # normalized class ids into the taxonomy
  confidence    : float?
  provenance    : { source: dataset|human|model, model_version? }
}
```

**`payload` is a typed, extensible union** keyed by `payload.type` — a new annotation
kind is a new payload type, not a new table:
- `verb_noun` → `{verb_id, noun_id, verb_raw, noun_raw}`
- `text` → `{text}` (instructions, captions, narration)
- `event` → `{event_type, attrs{}}`

**Keep raw *and* normalized** (the §2 rule): `label_raw` preserves the source string
(open-vocab narration, GPT-generated description) and `label_ids` holds the mapped
class ids. So EPIC’s fixed 97-verb/300-noun taxonomy and free open-vocab verbs coexist,
and "test labels withheld" is just `label_ids = null` with `label_raw` possibly present.

```
Taxonomy {
  taxonomy_id : id
  name        : string         # "epic-kitchens-100-verbs"
  kind        : enum {closed_vocab, open_vocab}
  version     : string
  classes     : [{id, name, parent?}]
}
```

A closed vocab maps `label_ids` into `classes`; open vocab leaves `label_raw`
authoritative. Language is *not* a special case — a per-episode instruction is a
`caption` with `scope=episode`; EPIC narration is a `segment` carrying both `verb_noun`
and the raw narration string.

### 4.3 Robot actions in more than one space

The §1/§2 decision stands: **each action space is its own `Stream`, both kept native,
never collapsed.** §4 specifies the descriptor and the relationships.

A robot episode like DROID yields up to four control-related streams, separated by
`role.role_class` and `role.slug`:

```
proprio.joint       joint_angles   # observed state, joint space
proprio.cartesian   pose_se3 / scalar  # observed ee pose
action.joint        action         # commanded, joint space
action.cartesian    action         # commanded, ee space  (side by side)
```

The **action `descriptor`** carries the semantics needed to interpret and later
canonicalize it:

```
descriptor (action) {
  space        : enum {joint, cartesian, ee_pose}
  control_mode : enum {position, velocity, torque}
  is_delta     : bool            # absolute vs relative command
  frame_id     : id?             # for cartesian/ee_pose (e.g. robot_base)
  rotation_rep : enum?           # for ee_pose (matrix/quat/euler/6d)
  gripper      : { encoding: continuous|binary, units }?
  dof_labels   : [string]        # per-channel names
  units        : string
}
```

Relationships:
- **`action.joint` and `action.cartesian` are co-representations of one command** →
  grouped by an action `group_id`, and typically share a clock and rate. We store both
  because each is lossy relative to the other (cartesian hides redundancy resolution;
  joint hides task-space intent). This is the explicit refusal of the OXE collapse.
- **proprio (observed) vs action (commanded)** is the `role_class` distinction, not a
  schema difference — the model picks whichever it needs.

**The canonical cross-embodiment action (the OXE 7-DoF) is a derived view, not at
rest.** When cross-embodiment training wants a normalized end-effector action, a view
projects the native action streams to a common `(Δtranslation, Δrotation, gripper)`
space. If only `action.joint` exists, the view uses the **`Embodiment`’s kinematic
model** (forward kinematics) to produce ee actions — which is exactly why `Embodiment`
is a referenced entity (§1): it supplies the model that makes the projection possible,
and that model can improve without rewriting any stored action. Lossy normalization
stays a recomputable view; the native joint/cartesian truth is never overwritten.

### 4.4 Worked micro-examples

| source | annotations | action streams |
|---|---|---|
| **EPIC-KITCHENS** | `segment` per action: `payload=verb_noun`, `label_raw=narration`, `taxonomy=epic-verbs/nouns`, `label_ids` (or null if withheld) | none |
| **EgoDex** | `caption` `scope=episode` `payload=text` (`llm_description`); 2nd description as another caption | none (human ego); hand motion is `pose_se3` *streams*, not annotations |
| **DROID** | `caption` `scope=episode` (instruction) | `proprio.joint` + `proprio.cartesian` + `action.joint` + `action.cartesian`, grouped; canonical 7-DoF is a derived view |
| **ego_raw** | none | none |

Same `Annotation` / `Stream` shapes throughout — sparse labels are annotation rows,
dense labels are streams, multiple action spaces are multiple streams, and every
opinionated normalization (class mapping, 7-DoF projection) is kept recomputable rather
than baked in.

---

## 5. The label-less case degrades gracefully

The raw-video set (`ego_raw`: one monocular stream, no action, no pose, no embodiment,
no annotation) is not an edge case — at petabyte scale it is the **majority of the
corpus**. So "one video stream and nothing else" must be the schema’s *natural
minimum*, not a degenerate row full of nulls. This is the single criterion the eval
calls out explicitly.

### 5.1 The mechanism: absence is zero rows, never a null column

Everything modality-specific is a **referenced entity with `0..N` cardinality**
(§1.1), so a missing modality is the *absence of rows*, not a nullable field on the
episode. The episode spine carries no `action_*`, `pose_*`, `intrinsics_*`, or
`language_*` columns to be null in the first place. Degradation is therefore graceful
on **every axis independently**:

| missing thing | how it disappears | nulls? |
|---|---|---|
| actions | zero `action.*` streams | none |
| pose / dense labels | zero `pose_se3`/`keypoints` streams | none |
| annotations / language | zero `Annotation` rows | none |
| calibration | zero `CalibrationProfile` rows (flagged `uncalibrated`, §3.3) | none |
| extra cameras | one `video` stream instead of N | none |
| embodiment | `unknown` sentinel, not SQL null (see 5.2) | none |

No axis forces a null because no axis was ever a column.

### 5.2 The minimal viable episode (the required core)

An episode is **valid** iff it has the small required core below; everything else is
optional entities. This is the contract the ingester must satisfy.

```
REQUIRED:
  Source        ×1   # provenance + license (inherited)
  Episode       ×1   # id, source_id, embodiment_id, time span, quality_status, native_root_uri
  Clock         ×1   # at least the primary clock
  Stream        ≥1   # at least one signal channel (for ego_raw: a single video stream)
  IngestRun     ×1   # how/when ingested, checksum

OPTIONAL (0..N, absent for label-less):
  Stream (more) · CalibrationProfile · Annotation · Taxonomy
  Embedding · ActionLatent · QualitySignal
```

**Embodiment uses an `unknown` sentinel, not null** — this *refines* §1’s "nullable"
note. A registry entry `embodiment = unknown` (and `human_hands`, specific robots, …)
keeps "no known embodiment" an explicit, queryable, joinable value rather than a null
with ambiguous meaning. Consistent with the no-nulls principle, even the one reference
that could be null isn’t.

Worked minimal example — a `factory_001/worker_*/clip.mp4` from `ego_raw`:

```
Source     : "ego_raw/factory", license=…, default_embodiment=unknown
Episode    : embodiment_id=unknown, primary_clock_id=C, quality_status=ok,
             native_root_uri=…/clip.mp4
Clock   C  : uniform, 30 Hz
Stream     : modality=video, role=head_cam, clock=C, timing=uniform,
             payload_ref=…/clip.mp4
IngestRun  : checksum, code_version
# 0 annotations · 0 calibration · 0 other streams · 0 embeddings · 0 latents
```

That row set is *structurally complete and valid*. Nothing is missing — there is
simply less of the optional graph.

### 5.3 Value accretes through the derived tier, not by re-labeling in place

The label-less episode becomes training-useful **without ever gaining native
annotations** — it accretes *derived* rows, which is exactly the encode-once/
re-label-often story (§2, §6) seen from the schema side:

```
video-only episode
      │  (encode once, expensive, durable)
      ▼
  + Embedding            ──▶ now searchable / dedupable / curatable
      │  (run IDM over embeddings, cheap, versioned)
      ▼
  + ActionLatent v1, v2… ──▶ now has pseudo-actions for pre-training
      │  (curation pass)
      ▼
  + QualitySignal        ──▶ now filterable & diversity-balanced (Part 3)
```

A raw clip turns into a first-class training citizen purely by adding derived entities
keyed to its one stream — never by mutating the episode or back-filling native labels.
The minimum-shaped majority of the corpus and the richly-annotated minority differ only
in *which optional rows exist*, so a single read path serves both.

### 5.4 Partial degradation for messy data (not all-or-nothing)

Degradation is also **per-component within a rich episode**, which is what makes the
messy real data (Part 2) tractable. An EgoDex clip whose `.hdf5` is missing or whose
intrinsics are absent does **not** get dropped — it degrades to whatever survived:

```
EgoDex clip, hdf5 missing
  → keep the video Stream  (valid episode)
  → omit the pose_se3 streams + calibration  (absent rows)
  → quality_status = partial ; IngestRun.note = "hdf5 missing"
```

So a corrupt or truncated component costs only that component, not the episode. The
ingester **surfaces and quarantines** at component granularity (`quality_status ∈
{ok, partial, quarantined}` + an `IngestRun` reason), never silently drops or crashes.
A truncated MP4 → `quarantined`; a missing side-car → `partial`; a clean clip → `ok`.

### 5.5 Querying the label-less corpus is a first-class query

Because absence is structural, the unlabeled set is a clean query, not a null scan:

```
-- everything the IDM still needs to label:
SELECT e.* FROM episode e
WHERE NOT EXISTS (SELECT 1 FROM annotation a WHERE a.episode_id=e.id)
  AND NOT EXISTS (SELECT 1 FROM stream s
                  WHERE s.episode_id=e.id AND s.role.role_class='action')
  AND EXISTS    (SELECT 1 FROM embedding em WHERE em.episode_id=e.id);
```

"Label-less but embedded" — i.e. ready for an IDM pass — falls straight out of the
entity model. The contrast with a flat schema is the whole point: there, the same
question is a fragile `WHERE action_col IS NULL AND pose_col IS NULL AND …` scan that
breaks the moment a new modality column is added. Here it is a structural property.

---

## 6. Embeddings and versioned IDM action-latents

This is where the schema meets the cost structure. The whole platform exists because
**embeddings are expensive but stable** (encode once, store forever) and the **IDM is
cheap but changes often** (re-run frequently). The derived tier must make re-labeling
the entire corpus with a *new* IDM a cheap pass over stored embeddings — never a
multi-petabyte re-decode. Two entities carry this, and one key decision makes it work.

### 6.1 The load-bearing decision: latents key off embeddings, not pixels

> **`ActionLatent` references the `Embedding` it was computed from, not the video.**

That single edge is what turns re-labeling into a scan-and-write instead of a decode.
A new IDM version reads the (small, hot) embeddings table, runs a (cheap) forward pass,
and appends new latent rows — the petabytes of video are never touched.

```
       encode ONCE (expensive, GPU, durable)        run IDM OFTEN (cheap, versioned)
video ──────────────────────────────────────▶ Embedding ──────────────────────────▶ ActionLatent
(cold, PB)                                     (hot, small,     reads embeddings,    (v1, v2, v3…)
                                                vector-indexed)  never pixels
```

### 6.2 The two derived entities

```
Embedding {                         # durable, expensive, computed once per (window, encoder)
  embedding_id     : id
  episode_id       : id
  stream_id        : id -> Stream   # which signal it encodes (usually video/image_seq)
  t_start_ns       : int64          # window on the canonical clock (§3) — alignable
  t_end_ns         : int64
  encoder_version  : string         # e.g. "vjepa2-2026-03" — defines the windowing too
  vector           : float[D]       # the Lance vector column (ANN-indexed)
  meta             : { dim, pooling, frame_stride }
}

ActionLatent {                      # cheap, re-run often, append-only, versioned by IDM
  latent_id           : id
  episode_id          : id
  stream_id           : id -> Stream      # the (video) stream being labeled
  source_embedding_id : id -> Embedding   # ← THE KEY EDGE: computed from embeddings
  encoder_version     : string            # which embedding generation it ran on
  idm_version         : string            # e.g. "idm-2026-05-v3"
  t_start_ns          : int64             # inherits the embedding window
  t_end_ns            : int64
  latent              : float[K]          # inferred action latent / pseudo-action
  action_space        : enum?             # if decoded to a concrete space (else raw latent)
  confidence          : float
}
```

Both also serve as the model for **`QualitySignal`**-style derivations and any future
inferred dense label (per §4.1, inferred dense labels are derived, stream-shaped, and
versioned) — same pattern: reference upstream, carry a `*_version`, append don’t mutate.

### 6.3 Why these decisions

- **Window granularity is explicit (`t_start_ns,t_end_ns`) and encoder-defined.**
  Different encoders have different temporal receptive fields (per-frame vs 16-frame
  clip); making the window explicit lets multiple `encoder_version`s coexist and keeps
  embeddings *alignable* to every other stream through the same `Clock`/`ClockSync`
  machinery (§3). Coverage may be **sparse** — we embed at a sampled rate chosen by
  curation (Part 3), not necessarily every frame; the schema imposes no full tiling.
- **Versioning is a first-class column, not just a storage snapshot.** `encoder_version`
  and `idm_version` are queryable values, so "give me IDM v3 latents over v-jepa2
  embeddings for the unlabeled corpus" is a filter. Lance’s dataset-level versioning
  (zero-copy snapshots, time-travel) sits *underneath* this for reproducibility — the
  two are complementary: **semantic versions in columns, storage versions in Lance.**
- **Append-only, never overwrite.** A new IDM does not replace v2 latents; it writes v3
  rows. Old training runs remain reproducible; A/B of IDM versions is a `WHERE
  idm_version=…`. This is the §2 "derived artifacts are caches, regenerable, never
  truth" rule applied to the most-rewritten data in the system.
- **`source_embedding_id` makes the dependency explicit and auditable.** You can always
  answer "which embedding generation produced this latent" and re-run exactly.

### 6.4 The new-IDM re-label pass (the cheap-pass sketch)

```
# Re-label the entire corpus with IDM v4 — no video decode, no re-embedding.
for batch in lance.scan("embedding",
                         filter="encoder_version = 'vjepa2-2026-03'",
                         columns=["embedding_id","episode_id","stream_id",
                                  "t_start_ns","t_end_ns","vector"]):
    latents = idm_v4(batch.vector)                 # cheap GPU forward over vectors
    write("action_latent", rows(batch, latents, idm_version="idm-2026-07-v4",
                                 source_embedding_id=batch.embedding_id))
```

Cost intuition (why this is the whole point):

| | raw frame | embedding | IDM forward |
|---|---|---|---|
| size / unit | ~MB (decoded) | ~KB vector | — |
| who reads it | encode pass, **once** | IDM/search/QC, **often** | — |
| compute | decode + huge encoder | — | tiny model |
| storage tier | **cold** (PB, on-prem) | **hot** (small, cached) | — |

Re-labeling reads the **hot KB-scale** vectors and runs a **tiny** model; it never pays
the **cold MB-scale** decode + heavy-encoder cost. A new IDM over the corpus is hours of
cheap GPU over a small hot table, versus weeks of decode over petabytes. The embedding
table being small + hot is also what lets the petabyte video stay cold (Part 4 tiering).

### 6.5 Embeddings are the hub of the derived tier

Every downstream consumer reads embeddings, not pixels: the **IDM** (action latents),
**near-duplicate detection / diversity balancing** and **semantic search** (vector
index over the same column), and most **curation QC** (Part 3). That is why the
embedding is the *durable join key* of the whole derived layer — and why §5’s
label-less episode becomes a full training citizen the moment it has an `Embedding` row,
with no native labels ever added.

---

## 7. Part 1 complete — consolidated entity reference

```
CONTEXT / SPINE        Source · Episode · Embodiment(reg) · IngestRun
TIME                   Clock · ClockSync
SIGNAL (polymorphic)   Stream  [modality × role, native-rate, on a Clock]
GEOMETRY               CalibrationProfile  [static SE(3) | dynamic→pose_se3 stream]
SPARSE LABELS          Annotation · Taxonomy   [segment | event | caption]
DERIVED (versioned)    Embedding(encode-once) · ActionLatent(IDM, re-label-often)
                       · QualitySignal(curation)
```

The whole schema reduces to one sentence: **a thin `Episode` spine, a polymorphic
`Stream` for everything time-varying, sparse `Annotation`s for labels, geometry/time as
referenced entities, and a versioned derived tier hubbed on `Embedding` — with absence
modeled as zero rows and every opinionated transform kept as a recomputable view.** One
shape holds all nine data types and the label-less extreme, no per-modality migration,
no pile of nulls.

Deferred to the build (Parts 2–4): physical Lance/Iceberg-style table DDL, the ingester
that maps DROID + EgoDex + `ego_raw` into this shape, the curation/QC taxonomy, and the
storage-tiering + training-handoff writeup.
