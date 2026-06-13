"""Arrow schemas for each Lance dataset (one dataset per table).

`oneof` unions and nested structs are materialized as discriminator columns + JSON
branch columns, per the storage note in docs/canonical_schema.md. Derived tables
(embedding / action_latent) are declared so the lakehouse is complete, even though the
Part 2 ingester does not populate them.
"""
from __future__ import annotations

import pyarrow as pa

_s = pa.string()
_i64 = pa.int64()
_f64 = pa.float64()
_b = pa.bool_()

SCHEMAS: dict[str, pa.Schema] = {
    # ----------------------------------------------------------------- registries
    "embodiment": pa.schema([
        ("embodiment_id", _s), ("name", _s), ("kind", _s),
        ("dof", _i64), ("kinematic_model_ref", _s),
    ]),
    "taxonomy": pa.schema([
        ("taxonomy_id", _s), ("name", _s), ("kind", _s), ("version", _s),
        ("classes_json", _s),
    ]),
    # -------------------------------------------------------------------- catalog
    "source": pa.schema([
        ("source_id", _s), ("name", _s), ("data_type", _s), ("license", _s),
        ("native_format", _s), ("default_embodiment_id", _s),
        ("modality_profile_json", _s),
    ]),
    "episode": pa.schema([
        ("episode_id", _s), ("source_id", _s), ("embodiment_id", _s),
        ("primary_clock_id", _s), ("t_start_ns", _i64), ("duration_ns", _i64),
        ("quality_status", _s), ("ingest_run_id", _s), ("native_root_uri", _s),
    ]),
    "ingest_run": pa.schema([
        ("ingest_run_id", _s), ("code_version", _s), ("started_at", _s),
        ("source_path", _s), ("checksum", _s), ("note", _s),
    ]),
    "clock": pa.schema([
        ("clock_id", _s), ("episode_id", _s), ("kind", _s), ("unit", _s),
        ("is_primary", _b),
    ]),
    "clock_sync": pa.schema([
        ("clock_id", _s), ("ref_clock_id", _s), ("model", _s), ("offset_ns", _i64),
        ("skew_ppm", _f64), ("anchors_json", _s),
    ]),
    "stream": pa.schema([
        ("stream_id", _s), ("episode_id", _s), ("modality", _s),
        ("role_slug", _s), ("role_class", _s), ("role_attrs_json", _s),
        ("group_id", _s), ("clock_id", _s), ("timing", _s), ("rate_hz", _f64),
        ("t_start_ns", _i64), ("t_end_ns", _i64), ("n_samples", _i64),
        ("timestamps_ref", _s), ("payload_uri", _s), ("payload_locator_json", _s),
        ("payload_format", _s), ("units", _s), ("frame_id", _s),
        ("rotation_rep", _s), ("space", _s), ("dof_labels_json", _s),
        ("descriptor_json", _s),
    ]),
    "calibration": pa.schema([
        ("calib_id", _s), ("scope", _s), ("target_ref", _s), ("intrinsics_json", _s),
        ("extrinsics_kind", _s), ("frame_id", _s), ("extr_static_T_json", _s),
        ("extr_dynamic_stream_id", _s), ("rig_json", _s),
    ]),
    "annotation": pa.schema([
        ("annotation_id", _s), ("episode_id", _s), ("kind", _s), ("scope", _s),
        ("target_ref", _s), ("clock_id", _s), ("t_start_ns", _i64),
        ("t_stop_ns", _i64), ("t_at_ns", _i64), ("payload_kind", _s),
        ("payload_json", _s), ("taxonomy_id", _s), ("label_raw", _s),
        ("label_ids_json", _s), ("confidence", _f64), ("provenance_source", _s),
        ("provenance_model_version", _s),
    ]),
    "quality_signal": pa.schema([
        ("signal_id", _s), ("episode_id", _s), ("stream_id", _s),
        ("t_start_ns", _i64), ("t_end_ns", _i64), ("signal_type", _s),
        ("value_json", _s), ("detector_version", _s),
    ]),
    "quarantine": pa.schema([
        ("quarantine_id", _s), ("source_name", _s), ("unit_ref", _s), ("stage", _s),
        ("reason", _s), ("ingest_run_id", _s), ("ts", _s),
    ]),
    # --------------------------------------------------------- derived (declared)
    "embedding": pa.schema([
        ("embedding_id", _s), ("episode_id", _s), ("stream_id", _s),
        ("t_start_ns", _i64), ("t_end_ns", _i64), ("encoder_version", _s),
        ("vector", pa.list_(pa.float32())), ("meta_json", _s),
    ]),
    "action_latent": pa.schema([
        ("latent_id", _s), ("episode_id", _s), ("stream_id", _s),
        ("source_embedding_id", _s), ("encoder_version", _s), ("idm_version", _s),
        ("t_start_ns", _i64), ("t_end_ns", _i64),
        ("latent", pa.list_(pa.float32())), ("action_space", _s), ("confidence", _f64),
    ]),
}

# Tables the Part 2 ingester writes to (in dependency order is not required for Lance).
CATALOG_TABLES = [
    "embodiment", "taxonomy", "source", "episode", "ingest_run", "clock",
    "clock_sync", "stream", "calibration", "annotation", "quality_signal",
    "quarantine",
]
DERIVED_TABLES = ["embedding", "action_latent"]
ALL_TABLES = CATALOG_TABLES + DERIVED_TABLES
