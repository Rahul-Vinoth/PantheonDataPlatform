"""Canonical record dataclasses — the clean Python API that adapters emit.

These mirror docs/canonical_schema.md. Each record knows how to flatten itself into a
row dict matching the Arrow table schema in `tables.py`. Nested structs and `oneof`
unions are flattened to a discriminator column + JSON-encoded branches (the
materialization rule from the spec's storage note).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from .enums import (
    Modality, RoleClass, ClockKind, ClockUnit, Timing, CalibScope, ExtrinsicsKind,
    AnnotationKind, AnnotationScope, QualityStatus, ProvKind, EmbodimentKind,
)


def _json(v: Any) -> Optional[str]:
    return None if v is None else json.dumps(v, separators=(",", ":"), sort_keys=True)


# --------------------------------------------------------------------------- registries
@dataclass
class Embodiment:
    embodiment_id: str
    name: str
    kind: EmbodimentKind
    dof: Optional[int] = None
    kinematic_model_ref: Optional[str] = None

    def to_row(self) -> dict:
        return {**asdict(self), "kind": str(self.kind)}


@dataclass
class Taxonomy:
    taxonomy_id: str
    name: str
    kind: str  # closed_vocab | open_vocab
    version: str
    classes: list[dict] = field(default_factory=list)  # [{id,name,parent?}]

    def to_row(self) -> dict:
        return {
            "taxonomy_id": self.taxonomy_id, "name": self.name, "kind": self.kind,
            "version": self.version, "classes_json": _json(self.classes),
        }


# ------------------------------------------------------------------------------- catalog
@dataclass
class Source:
    source_id: str
    name: str
    data_type: str
    license: str
    native_format: str
    default_embodiment_id: str
    modality_profile: list[str] = field(default_factory=list)

    def to_row(self) -> dict:
        return {
            "source_id": self.source_id, "name": self.name, "data_type": self.data_type,
            "license": self.license, "native_format": self.native_format,
            "default_embodiment_id": self.default_embodiment_id,
            "modality_profile_json": _json(self.modality_profile),
        }


@dataclass
class Episode:
    episode_id: str
    source_id: str
    embodiment_id: str
    primary_clock_id: str
    t_start_ns: int
    duration_ns: int
    quality_status: QualityStatus
    ingest_run_id: str
    native_root_uri: str

    def to_row(self) -> dict:
        return {**asdict(self), "quality_status": str(self.quality_status)}


@dataclass
class IngestRun:
    ingest_run_id: str
    code_version: str
    started_at: str
    source_path: str
    checksum: Optional[str] = None
    note: Optional[str] = None

    def to_row(self) -> dict:
        return asdict(self)


@dataclass
class Clock:
    clock_id: str
    episode_id: str
    kind: ClockKind
    unit: ClockUnit
    is_primary: bool

    def to_row(self) -> dict:
        return {**asdict(self), "kind": str(self.kind), "unit": str(self.unit)}


@dataclass
class ClockSync:
    clock_id: str
    ref_clock_id: str
    model: str  # offset | linear | anchors
    offset_ns: Optional[int] = None
    skew_ppm: Optional[float] = None
    anchors: Optional[list[list[int]]] = None

    def to_row(self) -> dict:
        return {
            "clock_id": self.clock_id, "ref_clock_id": self.ref_clock_id,
            "model": self.model, "offset_ns": self.offset_ns, "skew_ppm": self.skew_ppm,
            "anchors_json": _json(self.anchors),
        }


@dataclass
class Role:
    slug: str
    role_class: RoleClass
    attrs: dict = field(default_factory=dict)


@dataclass
class Descriptor:
    shape: list[int] = field(default_factory=list)
    dtype: str = ""
    units: Optional[str] = None
    frame_id: Optional[str] = None
    rotation_rep: Optional[str] = None
    channel_layout: Optional[list[str]] = None
    dof_labels: Optional[list[str]] = None
    # action-only
    space: Optional[str] = None
    control_mode: Optional[str] = None
    is_delta: Optional[bool] = None
    gripper: Optional[dict] = None


@dataclass
class Stream:
    stream_id: str
    episode_id: str
    modality: Modality
    role: Role
    clock_id: str
    timing: Timing
    rate_hz: float
    t_start_ns: int
    t_end_ns: int
    n_samples: int
    payload_uri: str
    payload_format: str
    descriptor: Descriptor
    group_id: Optional[str] = None
    timestamps_ref: Optional[str] = None
    payload_locator: Optional[dict] = None

    def to_row(self) -> dict:
        d = self.descriptor
        return {
            "stream_id": self.stream_id, "episode_id": self.episode_id,
            "modality": str(self.modality),
            "role_slug": self.role.slug, "role_class": str(self.role.role_class),
            "role_attrs_json": _json(self.role.attrs),
            "group_id": self.group_id, "clock_id": self.clock_id,
            "timing": str(self.timing), "rate_hz": float(self.rate_hz),
            "t_start_ns": self.t_start_ns, "t_end_ns": self.t_end_ns,
            "n_samples": self.n_samples, "timestamps_ref": self.timestamps_ref,
            "payload_uri": self.payload_uri,
            "payload_locator_json": _json(self.payload_locator),
            "payload_format": self.payload_format,
            # promoted descriptor scalars for queryability + full json blob
            "units": d.units, "frame_id": d.frame_id, "rotation_rep": d.rotation_rep,
            "space": d.space, "dof_labels_json": _json(d.dof_labels),
            "descriptor_json": _json(asdict(d)),
        }


@dataclass
class CalibrationProfile:
    calib_id: str
    scope: CalibScope
    target_ref: str
    intrinsics: dict  # {K, distortion_model, dist_coeffs, width, height, rolling_shutter?}
    extrinsics_kind: ExtrinsicsKind
    frame_id: str
    extr_static_T: Optional[list[list[float]]] = None       # static branch (4x4)
    extr_dynamic_stream_id: Optional[str] = None            # dynamic branch (moving cam)
    rig: Optional[dict] = None

    def to_row(self) -> dict:
        return {
            "calib_id": self.calib_id, "scope": str(self.scope),
            "target_ref": self.target_ref, "intrinsics_json": _json(self.intrinsics),
            "extrinsics_kind": str(self.extrinsics_kind), "frame_id": self.frame_id,
            "extr_static_T_json": _json(self.extr_static_T),
            "extr_dynamic_stream_id": self.extr_dynamic_stream_id,
            "rig_json": _json(self.rig),
        }


@dataclass
class Annotation:
    annotation_id: str
    episode_id: str
    kind: AnnotationKind
    scope: AnnotationScope
    clock_id: str
    payload_kind: str           # verb_noun | text | event
    payload: dict
    provenance_source: ProvKind
    target_ref: Optional[str] = None
    t_start_ns: Optional[int] = None
    t_stop_ns: Optional[int] = None
    t_at_ns: Optional[int] = None
    taxonomy_id: Optional[str] = None
    label_raw: Optional[str] = None
    label_ids: Optional[list[int]] = None
    confidence: Optional[float] = None
    provenance_model_version: Optional[str] = None

    def to_row(self) -> dict:
        return {
            "annotation_id": self.annotation_id, "episode_id": self.episode_id,
            "kind": str(self.kind), "scope": str(self.scope),
            "target_ref": self.target_ref, "clock_id": self.clock_id,
            "t_start_ns": self.t_start_ns, "t_stop_ns": self.t_stop_ns,
            "t_at_ns": self.t_at_ns, "payload_kind": self.payload_kind,
            "payload_json": _json(self.payload), "taxonomy_id": self.taxonomy_id,
            "label_raw": self.label_raw, "label_ids_json": _json(self.label_ids),
            "confidence": self.confidence,
            "provenance_source": str(self.provenance_source),
            "provenance_model_version": self.provenance_model_version,
        }


@dataclass
class QualitySignal:
    signal_id: str
    episode_id: str
    signal_type: str
    value: Any
    detector_version: str
    stream_id: Optional[str] = None
    t_start_ns: Optional[int] = None
    t_end_ns: Optional[int] = None

    def to_row(self) -> dict:
        return {
            "signal_id": self.signal_id, "episode_id": self.episode_id,
            "stream_id": self.stream_id, "t_start_ns": self.t_start_ns,
            "t_end_ns": self.t_end_ns, "signal_type": self.signal_type,
            "value_json": _json(self.value), "detector_version": self.detector_version,
        }


@dataclass
class Embedding:
    """Derived-tier: encode-once representation of a video/image_seq window."""
    embedding_id: str
    episode_id: str
    stream_id: str
    t_start_ns: int
    t_end_ns: int
    encoder_version: str
    vector: list[float]
    meta: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        return {
            "embedding_id": self.embedding_id, "episode_id": self.episode_id,
            "stream_id": self.stream_id, "t_start_ns": self.t_start_ns,
            "t_end_ns": self.t_end_ns, "encoder_version": self.encoder_version,
            "vector": [float(x) for x in self.vector], "meta_json": _json(self.meta),
        }


@dataclass
class ActionLatent:
    """Derived-tier: IDM output. Keys off a source Embedding (not pixels), so a new IDM
    re-labels the corpus as a cheap pass over stored embeddings."""
    latent_id: str
    episode_id: str
    stream_id: str
    source_embedding_id: str
    encoder_version: str
    idm_version: str
    t_start_ns: int
    t_end_ns: int
    latent: list[float]
    confidence: float
    action_space: Optional[str] = None

    def to_row(self) -> dict:
        return {
            "latent_id": self.latent_id, "episode_id": self.episode_id,
            "stream_id": self.stream_id, "source_embedding_id": self.source_embedding_id,
            "encoder_version": self.encoder_version, "idm_version": self.idm_version,
            "t_start_ns": self.t_start_ns, "t_end_ns": self.t_end_ns,
            "latent": [float(x) for x in self.latent],
            "action_space": self.action_space, "confidence": float(self.confidence),
        }


@dataclass
class Quarantine:
    quarantine_id: str
    source_name: str
    unit_ref: str
    stage: str          # discover | access | map | write
    reason: str
    ingest_run_id: str
    ts: str

    def to_row(self) -> dict:
        return asdict(self)


@dataclass
class CanonicalRecords:
    """Everything an adapter emits for ONE source episode unit."""
    episode: Optional[Episode] = None
    clocks: list[Clock] = field(default_factory=list)
    clock_syncs: list[ClockSync] = field(default_factory=list)
    streams: list[Stream] = field(default_factory=list)
    calibrations: list[CalibrationProfile] = field(default_factory=list)
    annotations: list[Annotation] = field(default_factory=list)
    quality_signals: list[QualitySignal] = field(default_factory=list)
