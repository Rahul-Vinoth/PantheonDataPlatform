## Data Platform & Schema Technical

### Background

A robot foundation model is only as good as the data feeding it, and that data
arrives in a dozen incompatible shapes. Teleop and UMI rigs, production robots,
simulation, and large public/internet video sets each record differently: one
camera or ten, glove joint-angles or SLAM poses or no pose at all, robot actions
in joint *and* cartesian space, dense per-frame annotations or none, their own
clocks, their own licenses.

To train one model across all of it, you need **one canonical layer**: every
source converted to a single episode shape, labels and embeddings hanging off it
by reference, so that pre-training, labeling, eval, and dashboards all read the
same thing. The hard, greenfield part is the **internet-scale / public-dataset**
side — the curation lakehouse that normalizes heterogeneous datasets and filters
them down to clean, diverse, training-ready data.

Two facts shape every design decision here:

* **Scale is petabytes, on-prem.** A single public set is already ~16 TB / 10,000
  hours; the corpus target is petabytes on owned hardware. Cloud is economically
  infeasible at this scale, so **data has locality** and **compute goes to the
  data** — you cannot assume a clip is cheap to move or re-decode.
* **Most of the corpus is unlabeled.** Annotations come from an **inverse-dynamics
  model (IDM)** trained on the labeled data and run over the unlabeled video to
  infer action latents. Crucially, the IDM runs on **embeddings, not raw pixels**:
  embeddings are **expensive but stable** (compute once, store forever); the IDM
  is **cheap and changes often** (re-label by re-running over stored embeddings,
  never re-decoding video). The storage and schema design must make that
  re-labeling a cheap pass, not a corpus-wide re-encode.

Your job for the day sits at the seam of those two facts: design the schema that
holds everything, and stand up enough of the platform to prove it ingests and
curates real, messy, multi-format data at scale.

### Task

Design and partially build the **canonical dataset schema** and a **slice of the
curation platform** that turns the heterogeneous datasets on the provided VM into
one normalized, queryable, training-ready layer.

The task has four parts. They build on each other, but **you are not expected to
finish all four** — pick where to go deep. We would rather see Part 1 and Part 2
done with real judgment than all four done shallowly.

---

**Part 1 — Canonical schema (design).**
Design one schema that normalizes *every* dataset in the workspace into a single
shape — the richly annotated ones and the raw-video one alike. Decide explicitly:

* the entity model (what is an episode; what hangs off it; what is a separate
  entity vs an inline field)
* what you normalize **at rest** vs what you keep **native** and canonicalize
  later (be deliberate — normalizing payloads too eagerly is lossy)
* how you represent: a variable number of camera/sensor **streams**; **multi-rate
  streams with their own clocks**; **calibration** (intrinsics/extrinsics, and
  the moving-camera case); **annotations** (temporal segments, per-frame
  pose/boxes/masks, point events); **robot actions** in more than one space;
  **provenance, license, and quality**
* how the **label-less case** (one video stream, no action/pose/embodiment)
  degrades gracefully instead of becoming a pile of nulls
* where **embeddings** and **versioned IDM-inferred action latents** live, given
  the "encode-once, re-label-often" cost structure above

Deliver this as a concrete schema definition (types/tables/registries — your
choice of notation) plus a short rationale for the load-bearing decisions.

---

**Part 2 — Ingestion & normalization (build).**
Stand up a lakehouse table (or tables) and write an ingester that maps **at least
two** of the provided datasets — chosen to stress the schema, e.g. one richly
annotated set **and** the raw-video set — into your Part 1 schema. Materialize it
so it is queryable (vector/metadata search, or just columnar scans — your call).

The data is real and **messy**: inconsistent layouts, missing calibration or
metadata, duplicated or partially-extracted clips, the occasional corrupt or
truncated file. Handle it the way you would in production — surface and quarantine
problems, don't silently drop or crash.

---

**Part 3 — Curation & QC at scale (build + reason).**
Run a curation/QC pass over the egocentric video and attach the results to your
schema as queryable signals. At minimum, define and detect a few **data-quality
issues** (e.g. blur, exposure, dropped/duplicate frames, near-duplicate clips),
and reason about **corpus-level curation**: keep a clip only if it shows useful
manipulation (human performing a task, hands visible, an arm/gripper present),
and track **task / environment diversity** so the corpus can be balanced rather
than dominated by repetitive footage.

Then reason explicitly about doing this over **16 TB / 10,000 hours** and beyond:
what runs per-frame vs sampled, what is CPU vs GPU, what you compute once vs
re-compute, and where the bottleneck actually is (decode? embedding? I/O? the
link to storage?).

---

**Part 4 — Scale, storage & training hand-off (writeup).**
A short design writeup tying it together:

* the **embedding ↔ IDM** economics: why embeddings are the durable asset, and
  how your schema makes re-labeling the whole corpus with a *new* IDM a cheap
  pass over stored embeddings rather than a multi-PB re-decode
* **storage/tiering** intuition for a petabyte on-prem store with a single
  cache/compute tier in front of it (what must be hot, what stays cold)
* how a curated subset gets **handed off to training** (a portable export the
  trainer can consume directly, decoupled from your internal storage format)

---

You are free to define the entity model, what's normalized vs derived, the
storage/indexing format, the execution model, the curation taxonomy, and the
prioritization. You may use heuristics, models, metadata, sampling, and any
local/distributed processing you like. **The problem is intentionally
open-ended.**

### Time Limit

**Full day (~6–7 working hours).**

You are **not** expected to fully solve this, or to touch all four parts. A
deep, well-reasoned Part 1 + Part 2 is a strong outcome.

We care significantly more about:

* schema design judgment (what generalizes vs what hard-codes today's data)
* framing and prioritization
* scalability and operational reasoning
* engineering judgment under an open, messy problem

than about completeness or coverage.

### Workspace

You will work on a provided VM:

```
ssh pantheon@<VM_IP>
pwd: <given on the day>
```

A subset of several datasets is mounted in their **native, unmodified formats** —
the heterogeneity *is* the point. You may consult public documentation for any of
them; the task is normalizing and engineering over them, not recalling their
schemas from memory.

```
/data/
├── ego_raw/                         # raw egocentric video — the scale / label-less case
│   ├── factory_001/ … factory_0NN/      # MP4 / H.265, 1080p 30fps, ~clip-length each
│   │   └── worker_*/ *.mp4               # head-mounted monocular; NO per-frame annotation
│   └── (some shards only as .tar; some clips duplicated or truncated)
│
├── ego_annot/                       # richly annotated egocentric (pose + language)
│   └── partN/taskM/{idx}.mp4 + {idx}.hdf5    # paired by index
│       # hdf5: camera intrinsics (3×3) · per-frame SE(3) transforms (head/wrists/
│       #       ~per-finger joints) · per-joint confidence · LLM task description
│       # (intrinsics or hdf5 missing for a handful of clips)
│
├── dexterous_multicam/              # multi-camera + glove rig
│   └── ep_XXXX/
│       ├── <cam>/ *.jpg                  # several camera streams (palm + stereo), images-on-disk
│       ├── *_manus.pkl / *_tracker.pkl   # hand joint-angles + 6DoF wrist
│       └── timesteps.txt                 # per-modality timestamps (for sync)
│
├── robot_rlds/                      # teleop robot episodes (RLDS/LeRobot-style)
│   └── ...                               # multi-camera + proprio + action in JOINT and
│       #                                 # CARTESIAN space side by side · per-scene calibration ·
│       #                                 # a few post-hoc language instructions per episode
│
└── segmented_video/                 # egocentric video with a temporal-segment annotation file
    └── *.mp4 + segments.csv              # action segments: start/stop, open-vocab verb+noun
        #                                 # (a fixed verb/noun taxonomy; test labels withheld)
```

Across these you will encounter: **1 camera vs many**; **annotated vs raw**;
**SE(3) poses vs glove joint-angles vs none**; **MP4 vs JPEG-on-disk vs HDF5 vs
RLDS**; **calibration present vs absent**; **multi-rate streams with separate
clocks**; and **different licenses**. Some files or layouts are inconsistent,
incomplete, duplicated, corrupted, or partially missing — by design.

You are not expected to process the full mounted subset; operate on whatever is
tractable within the day.

## Deliverables

### Required

* **Schema definition** — the canonical episode/dataset schema (Part 1), as a
  concrete spec, with a short rationale for the key choices.
* **README with runnable code** — the ingester + curation pass (Parts 2–3),
  enough to reproduce what you built on a slice of the data.
* **Writeup** covering:
  * architecture / design decisions
  * what's normalized at rest vs derived later, and why
  * tradeoffs, assumptions, bottlenecks
  * how it scales to petabytes and feeds training (Part 4)
  * future work

### Optional

* benchmarks / profiling (decode, embedding, ingest throughput)
* a mapping table: each source class → your schema
* an architecture diagram
* a worked example of the schema's minimal (label-less) case
* curation metrics / a ranking or diversity-balancing scheme
* a sketch of how a *new* IDM version re-labels the corpus over stored embeddings

Keep documentation concise and functional.

## Evaluation

We are primarily evaluating:

* **schema design** — does one shape genuinely hold all the sources, including
  the label-less extreme, without a pile of nulls or a migration per new modality?
* **normalization judgment** — what you canonicalize vs keep native; how you
  handle multi-rate clocks, calibration, multiple action spaces, provenance,
  license, and versioned derivations
* **systems & large-scale data-engineering intuition** — locality, hot/cold
  tiering, encode-once vs re-label-often, where the real bottleneck is
* **operational reasoning** — handling messy/partial data, concurrency, cost,
  latency, coverage tradeoffs
* code quality and organization
* prioritization under an open, time-boxed problem

We are **not** evaluating:

* detector accuracy in isolation
* polished infrastructure or full dataset coverage
* exhaustive modeling work
* getting every dataset ingested — two done well beats five done shallowly
