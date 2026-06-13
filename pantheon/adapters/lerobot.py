"""LeRobot v3 adapter — Cross-Embodiment robot teleop (parquet + packed MP4).

Layout (LeRobot dataset format v3.0):
  <root>/meta/info.json                      catalog: fps, features (shapes/dtypes/dof)
  <root>/meta/tasks.parquet                  task_index -> task string
  <root>/meta/episodes/chunk-*/file-*.parquet   per-episode index + video segments
  <root>/data/chunk-*/file-*.parquet         per-frame observation.state / action
  <root>/videos/<video_key>/chunk-*/file-*.mp4   MANY episodes packed in one mp4

Key trait: one mp4 holds every episode; an episode is a *time segment*
[from_timestamp, to_timestamp] within it. State/action live as list-columns in a
shared parquet, sliced by [dataset_from_index, dataset_to_index).

Exercises: dual-camera RIG (grouped video streams), simultaneous proprio + dense
ACTION streams referencing parquet columns, task captions, and — because the payload is
a packed mp4 — real partial degradation when a clip's segment isn't fully present
(e.g. a capped/truncated download): state/action survive, video degrades.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

import pyarrow.parquet as pq

from ..io.hashing import stable_id
from ..io.video import probe_video
from ..registry import register_adapter
from ..schema.enums import (
    Modality, RoleClass, ClockKind, ClockUnit, Timing, QualityStatus, AnnotationKind,
    AnnotationScope, ProvKind, EmbodimentKind, CalibScope, ExtrinsicsKind,
)
from ..schema.records import (
    Annotation, CalibrationProfile, CanonicalRecords, Clock, Descriptor, Embodiment,
    Episode, QualitySignal, Role, Source, Stream,
)
from .base import EpisodeUnit, IngestError, SourceAdapter

SOURCE_ID = "lerobot_unitreeh1"
EMB_ROBOT = "emb_unitree_h1"
DETECTOR = "lerobot-0.1"


@register_adapter("lerobot")
class LeRobotAdapter(SourceAdapter):

    def __init__(self):
        self._info: Optional[dict] = None

    # ----------------------------------------------------------------- registries
    def source(self) -> Source:
        return Source(
            source_id=SOURCE_ID, name="LeRobot/UnitreeH1", data_type="cross_embodiment",
            license="apache-2.0", native_format="parquet+mp4",
            default_embodiment_id=EMB_ROBOT,
            modality_profile=["video", "joint_angles", "action"])

    def embodiments(self) -> list[Embodiment]:
        # proprioceptive state is 19-DoF (action commands 40 motors incl. targets)
        return [Embodiment(EMB_ROBOT, "unitree_h1", EmbodimentKind.ROBOT,
                           dof=19, kinematic_model_ref="unitree_h1")]

    # ----------------------------------------------------------------- probing
    def probe(self, root: Path) -> bool:
        """LeRobot v3 signature: meta/info.json carrying a features map. Unambiguous,
        so ego_raw (which also sees the packed mp4s) defers to this."""
        info_path = Path(root) / "meta" / "info.json"
        if not info_path.exists():
            return False
        try:
            info = json.loads(info_path.read_text())
        except Exception:
            return False
        return "features" in info and "data_path" in info

    # ----------------------------------------------------------------- discovery
    def iter_episodes(self, root: Path) -> Iterator[EpisodeUnit]:
        root = Path(root)
        self._info = json.loads((root / "meta" / "info.json").read_text())

        ep_meta_files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
        for ep_file in ep_meta_files:
            rows = pq.read_table(ep_file).to_pylist()
            for r in rows:
                yield EpisodeUnit(
                    unit_ref=f"episode_{r['episode_index']:06d}", root=root,
                    paths={}, meta=r)

    # ----------------------------------------------------------------- mapping
    def emit(self, unit: EpisodeUnit, *, episode_id: str,
             ingest_run_id: str) -> CanonicalRecords:
        root = unit.root
        info = self._info or json.loads((root / "meta" / "info.json").read_text())
        m = unit.meta
        fps = float(info.get("fps", 0.0)) or 1.0
        features = info["features"]

        ep_idx = m["episode_index"]
        length = int(m.get("length", 0))
        from_idx = int(m.get("dataset_from_index", 0))
        to_idx = int(m.get("dataset_to_index", from_idx + length))
        duration_ns = int(round(length / fps * 1e9))

        clock_id = f"{episode_id}/clk0"
        signals: list[QualitySignal] = []
        status = QualityStatus.OK

        # --- data parquet must be resolvable (asset access) ---
        data_parquet = self._data_path(root, m, info)
        if data_parquet is None or not data_parquet.exists():
            raise IngestError(
                f"data parquet missing for episode {ep_idx}", stage="access")

        streams: list[Stream] = []

        # --- proprioceptive state -> JOINT_ANGLES stream ---
        if "observation.state" in features:
            streams.append(self._tabular_stream(
                episode_id, clock_id, "observation.state", features["observation.state"],
                Modality.JOINT_ANGLES, RoleClass.PROPRIO, "proprio_state",
                data_parquet, from_idx, to_idx, length, fps, duration_ns))

        # --- action -> ACTION stream (dual-space joint command) ---
        if "action" in features:
            streams.append(self._tabular_stream(
                episode_id, clock_id, "action", features["action"],
                Modality.ACTION, RoleClass.ACTION, "action",
                data_parquet, from_idx, to_idx, length, fps, duration_ns,
                space="joint"))

        # --- video streams (one per camera), grouped as a rig ---
        rig_group = f"{episode_id}/camera_rig"
        cam_keys = [k for k, f in features.items() if f.get("dtype") == "video"]
        rig_members: list[str] = []
        for key in sorted(cam_keys):
            vstream, vsig, degraded = self._video_stream(
                episode_id, clock_id, key, features[key], m, root, info, fps)
            if vstream is not None:
                streams.append(vstream)
                rig_members.append(vstream.stream_id)
            signals += vsig
            if degraded:
                status = QualityStatus.PARTIAL

        # multi-camera rig with no intrinsics provided -> grouped + flagged uncalibrated
        calibrations: list[CalibrationProfile] = []
        if rig_members:
            calibrations.append(CalibrationProfile(
                calib_id=f"{episode_id}/rig0", scope=CalibScope.EPISODE,
                target_ref=rig_group, intrinsics={},
                extrinsics_kind=ExtrinsicsKind.STATIC, frame_id="robot_base",
                rig={"members": rig_members, "layout": "stereo_pair"}))
            signals.append(self._sig(episode_id, "intrinsics_missing", True))
            signals.append(self._sig(episode_id, "uncalibrated", True))

        # --- task captions ---
        annotations: list[Annotation] = []
        for i, task in enumerate(m.get("tasks") or []):
            if not task:
                continue
            annotations.append(Annotation(
                annotation_id=stable_id("ann", episode_id, str(i)),
                episode_id=episode_id, kind=AnnotationKind.CAPTION,
                scope=AnnotationScope.EPISODE, clock_id=clock_id,
                payload_kind="text", payload={"text": task},
                provenance_source=ProvKind.DATASET, label_raw=task))

        ep = Episode(
            episode_id=episode_id, source_id=SOURCE_ID, embodiment_id=EMB_ROBOT,
            primary_clock_id=clock_id, t_start_ns=0, duration_ns=duration_ns,
            quality_status=status, ingest_run_id=ingest_run_id,
            native_root_uri=str(root))
        clock = Clock(clock_id, episode_id, ClockKind.DEVICE_MONOTONIC, ClockUnit.NS,
                      is_primary=True)
        return CanonicalRecords(
            episode=ep, clocks=[clock], streams=streams, calibrations=calibrations,
            annotations=annotations, quality_signals=signals)

    # --------------------------------------------------------------- helpers
    def _tabular_stream(self, episode_id, clock_id, column, feat, modality, role_class,
                        slug, parquet, from_idx, to_idx, length, fps, duration_ns,
                        space=None) -> Stream:
        shape = list(feat.get("shape", []))
        dof_labels = None
        names = feat.get("names")
        if isinstance(names, dict) and "motors" in names:
            dof_labels = list(names["motors"])
        return Stream(
            stream_id=f"{episode_id}/{slug}", episode_id=episode_id,
            modality=modality, role=Role(slug, role_class),
            clock_id=clock_id, timing=Timing.UNIFORM, rate_hz=fps,
            t_start_ns=0, t_end_ns=duration_ns, n_samples=length,
            payload_uri=str(parquet), payload_format="parquet",
            payload_locator={"column": column, "from_index": from_idx,
                             "to_index": to_idx},
            descriptor=Descriptor(
                shape=shape, dtype=feat.get("dtype", "float32"),
                dof_labels=dof_labels, space=space,
                is_delta=(False if space else None)))

    def _video_stream(self, episode_id, clock_id, key, feat, m, root, info, fps):
        """Returns (stream|None, signals, degraded). The mp4 packs every episode; this
        episode is the segment [from_ts, to_ts). We reference that segment without
        decoding. If the packed file can't be opened or doesn't reach to_ts (e.g. a
        capped download), the video degrades while state/action survive."""
        signals: list[QualitySignal] = []
        slug = key.split(".")[-1]                       # observation.images.cam_left -> cam_left
        mp4 = self._video_path(root, key, m, info)
        from_ts = float(m.get(f"videos/{key}/from_timestamp", 0.0) or 0.0)
        to_ts = float(m.get(f"videos/{key}/to_timestamp", 0.0) or 0.0)
        shape = list(feat.get("shape", []))
        codec = (feat.get("video_info", {}) or {}).get("video.codec", "")

        if mp4 is None or not mp4.exists():
            signals.append(self._sig(episode_id, "video_missing", {"key": key}))
            return None, signals, True

        # asset access: probe the packed file (cheap header read + few-frame decode)
        try:
            vinfo = probe_video(mp4)
            probed_codec = vinfo.codec
            seg_present = (vinfo.duration_ns / 1e9) >= to_ts - 1e-3
        except Exception:
            # packed mp4 unreadable (e.g. truncated download) -> degrade, keep parquet
            signals.append(self._sig(episode_id, "video_unreadable", {"key": key}))
            return None, signals, True

        if not seg_present:
            signals.append(self._sig(
                episode_id, "video_segment_incomplete",
                {"key": key, "to_ts": to_ts, "file_dur_s": round(vinfo.duration_ns / 1e9, 2)}))
            degraded = True
        else:
            degraded = False

        n_samples = int(round((to_ts - from_ts) * fps)) if to_ts > from_ts else 0
        stream = Stream(
            stream_id=f"{episode_id}/{slug}", episode_id=episode_id,
            modality=Modality.VIDEO, role=Role(slug, RoleClass.CAMERA),
            group_id=f"{episode_id}/camera_rig",
            clock_id=clock_id, timing=Timing.UNIFORM, rate_hz=fps,
            t_start_ns=0, t_end_ns=int(round((to_ts - from_ts) * 1e9)),
            n_samples=n_samples,
            payload_uri=str(mp4), payload_format=probed_codec or codec,
            payload_locator={"key": key, "from_timestamp": from_ts,
                             "to_timestamp": to_ts},
            descriptor=Descriptor(
                shape=(shape[:2] + [3]) if len(shape) >= 2 else shape, dtype="uint8"))
        return stream, signals, degraded

    def _data_path(self, root: Path, m: dict, info: dict) -> Optional[Path]:
        tmpl = info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
        try:
            rel = tmpl.format(chunk_index=int(m.get("data/chunk_index", 0)),
                              file_index=int(m.get("data/file_index", 0)))
        except Exception:
            return None
        return root / rel

    def _video_path(self, root: Path, key: str, m: dict, info: dict) -> Optional[Path]:
        tmpl = info.get("video_path",
                        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
        try:
            rel = tmpl.format(
                video_key=key,
                chunk_index=int(m.get(f"videos/{key}/chunk_index", 0)),
                file_index=int(m.get(f"videos/{key}/file_index", 0)))
        except Exception:
            return None
        return root / rel

    def _sig(self, episode_id: str, kind: str, value) -> QualitySignal:
        return QualitySignal(
            signal_id=stable_id("qs", episode_id, kind), episode_id=episode_id,
            signal_type=kind, value=value, detector_version=DETECTOR)
